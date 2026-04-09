from __future__ import annotations

"""
FastAPI application for ProductionIncidentEnv.

The server maintains a single active session per instance (stateful).
For parallel evaluation, run multiple instances or extend with session IDs.

Endpoints:
  POST /reset             — start a new episode
  POST /step              — take an action
  GET  /state             — full internal state (for debugging)
  GET  /tasks             — list tasks + action schema
  POST /grader            — grade a completed episode
  POST /baseline          — run the baseline agent and return results
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure the parent directory is on the path when running from /api/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import ProductionIncidentEnv
from env.graders import EpisodeTrajectory, StepRecord, grade_episode
from env.models import (
    Action,
    ActionType,
    EnvConfig,
    FaultType,
    Observation,
    StepResult,
)
from env.tasks import ALL_TASKS, build_env_config, get_task

app = FastAPI(
    title="ProductionIncidentEnv",
    description=(
        "OpenEnv-compliant reinforcement learning environment for "
        "production SRE incident response. Agents must diagnose and "
        "remediate microservice faults before SLO violations cascade."
    ),
    version="1.0.0",
)

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session state — single active episode
# ---------------------------------------------------------------------------

_env: Optional[ProductionIncidentEnv] = None
_trajectory: EpisodeTrajectory = []
_active_task_id: Optional[str] = None
_episode_done: bool = False


def _require_env() -> ProductionIncidentEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _env


def _require_active_episode() -> None:
    if _episode_done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new one.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    fault_type: Optional[str] = None
    fault_origin: Optional[str] = None
    seed: Optional[int] = None
    max_steps: Optional[int] = None


class ResetResponse(BaseModel):
    task_id: Optional[str]
    observation: Dict[str, Any]
    message: str


class StepRequest(BaseModel):
    action_type: str
    params: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    reward_components: Dict[str, float]
    done: bool
    info: Dict[str, Any]
    step: int


class GraderRequest(BaseModel):
    task_id: str


class GraderResponse(BaseModel):
    task_id: str
    score: float
    components: Dict[str, float]
    notes: List[str]


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    objective: str
    relevant_actions: List[str]
    max_steps: int


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]
    action_schema: Dict[str, Any]


class BaselineRunRequest(BaseModel):
    task_ids: Optional[List[str]] = None  # None = run all 3
    max_steps_override: Optional[int] = None


class BaselineTaskResult(BaseModel):
    task_id: str
    score: float
    grade_components: Dict[str, float]
    notes: List[str]
    steps_taken: int
    terminal_reason: Optional[str]
    duration_seconds: float


class BaselineRunResponse(BaseModel):
    results: List[BaselineTaskResult]
    average_score: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/reset", response_model=ResetResponse)
async def reset(req: Optional[ResetRequest] = Body(default=None)) -> ResetResponse:
    """
    Start a new episode. Optionally pin to a specific task (which sets the
    fault type, origin, and seed deterministically) or configure freely.
    """
    if req is None:
        req = ResetRequest()

    global _env, _trajectory, _active_task_id, _episode_done

    fault_type: Optional[FaultType] = None

    if req.task_id:
        task = get_task(req.task_id)
        config = build_env_config(task)
        _active_task_id = req.task_id
    else:
        fault_type = FaultType(req.fault_type) if req.fault_type else None
        config = EnvConfig(
            fault_type=fault_type,
            fault_origin=req.fault_origin,
            max_steps=req.max_steps or 300,
            noise_seed=req.seed,
        )
        _active_task_id = None

    _env = ProductionIncidentEnv(config=config)
    _trajectory = []
    _episode_done = False

    fault_origin = req.fault_origin if not req.task_id else None
    obs = _env.reset(
        fault_type=fault_type,
        fault_origin=fault_origin,
        seed=req.seed,
    )

    msg = f"Episode started."
    if req.task_id:
        task = get_task(req.task_id)
        msg = f"Task '{task.name}' ({task.difficulty}) started. Objective: {task.objective}"

    return ResetResponse(
        task_id=_active_task_id,
        observation=obs.model_dump(),
        message=msg,
    )


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest) -> StepResponse:
    """
    Execute one action in the current episode.
    """
    global _trajectory, _episode_done

    env = _require_env()
    _require_active_episode()

    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type '{req.action_type}'. Valid: {[a.value for a in ActionType]}",
        )

    action = Action(type=action_type, params=req.params)
    state_snapshot = env.state()
    result = env.step(action)

    _trajectory.append(
        StepRecord(
            step=result.observation.step,
            action_type=action_type.value,
            action_params=req.params,
            result=result,
            state_snapshot=state_snapshot,
        )
    )

    if result.done:
        _episode_done = True

    return StepResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        reward_components=result.reward_components.model_dump(),
        done=result.done,
        info=result.info,
        step=result.observation.step,
    )


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return the full internal (ground-truth) state. Intended for debugging and graders."""
    env = _require_env()
    return env.state()


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks() -> TaskListResponse:
    """List all available tasks and the full action schema."""
    tasks = [
        TaskInfo(
            id=t.id,
            name=t.name,
            difficulty=t.difficulty,
            description=t.description,
            objective=t.objective,
            relevant_actions=t.relevant_actions,
            max_steps=t.max_steps,
        )
        for t in ALL_TASKS
    ]

    action_schema = {
        "actions": {
            ActionType.SCALE_SERVICE.value: {
                "params": {"service_id": "str", "delta": "int in {-3,-2,-1,1,2,3}"},
            },
            ActionType.ROLLBACK_SERVICE.value: {
                "params": {"service_id": "str", "steps_back": "int in {1,2,3}"},
            },
            ActionType.RESTART_SERVICE.value: {
                "params": {"service_id": "str"},
            },
            ActionType.OPEN_CIRCUIT_BREAKER.value: {
                "params": {"service_id": "str"},
            },
            ActionType.CLOSE_CIRCUIT_BREAKER.value: {
                "params": {"service_id": "str"},
            },
            ActionType.ADJUST_RATE_LIMIT.value: {
                "params": {"service_id": "str", "multiplier": "float in {0.25,0.5,0.75,1.0,1.5,2.0}"},
            },
            ActionType.REROUTE_TRAFFIC.value: {
                "params": {
                    "from_service": "str",
                    "to_service": "str",
                    "percentage": "int in {25,50,75,100}",
                },
            },
            ActionType.QUERY_LOGS.value: {
                "params": {"service_id": "str", "window_minutes": "int in {5,15,30}"},
            },
            ActionType.QUERY_METRICS.value: {
                "params": {
                    "service_id": "str",
                    "metric": "str (error_rate|latency_p99|cpu_utilization|memory_utilization|request_rate)",
                },
            },
            ActionType.ACKNOWLEDGE_ALERT.value: {
                "params": {"alert_id": "str"},
            },
            ActionType.PAGE_HUMAN.value: {
                "params": {"severity": "str in {P1,P2}"},
            },
            ActionType.NO_OP.value: {
                "params": {},
            },
        },
        "services": [
            "api-gateway", "auth-service", "product-service", "cart-service",
            "order-service", "payment-service", "inventory-service", "db-proxy",
        ],
    }

    return TaskListResponse(tasks=tasks, action_schema=action_schema)


@app.post("/grader", response_model=GraderResponse)
async def grader(req: GraderRequest) -> GraderResponse:
    """Grade the current (completed) episode against the specified task."""
    env = _require_env()

    if not _episode_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is still running. Complete the episode before grading.",
        )

    final_state = env.state()
    result = grade_episode(req.task_id, _trajectory, final_state)

    return GraderResponse(
        task_id=req.task_id,
        score=result.score,
        components=result.components,
        notes=result.notes,
    )


@app.post("/baseline", response_model=BaselineRunResponse)
async def run_baseline(req: BaselineRunRequest) -> BaselineRunResponse:
    """
    Run the built-in heuristic baseline agent (no LLM required) on the
    specified tasks and return graded results. This is the deterministic
    reference implementation used for scoring calibration.
    """
    from scripts.baseline import HeuristicBaselineAgent, run_heuristic_baseline

    task_ids = req.task_ids or [t.id for t in ALL_TASKS]
    results = []

    for task_id in task_ids:
        start = time.time()
        task = get_task(task_id)
        config = build_env_config(task)
        env = ProductionIncidentEnv(config=config)
        obs = env.reset(seed=task.seed)

        agent = HeuristicBaselineAgent(task)
        traj: EpisodeTrajectory = []
        done = False

        while not done:
            action = agent.act(obs, env.state())
            state_snap = env.state()
            result = env.step(action)
            traj.append(
                StepRecord(
                    step=result.observation.step,
                    action_type=action.type.value,
                    action_params=action.params,
                    result=result,
                    state_snapshot=state_snap,
                )
            )
            obs = result.observation
            done = result.done

        grade = grade_episode(task_id, traj, env.state())
        terminal = env.state().get("terminal_reason")
        elapsed = time.time() - start

        results.append(
            BaselineTaskResult(
                task_id=task_id,
                score=grade.score,
                grade_components=grade.components,
                notes=grade.notes,
                steps_taken=len(traj),
                terminal_reason=terminal,
                duration_seconds=round(elapsed, 3),
            )
        )

    avg = sum(r.score for r in results) / len(results) if results else 0.0
    return BaselineRunResponse(results=results, average_score=round(avg, 4))


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "env": "ProductionIncidentEnv", "version": "1.0.0"}
