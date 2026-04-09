#!/usr/bin/env python3
"""
Baseline agents for ProductionIncidentEnv.

Two agents are implemented:

  HeuristicBaselineAgent — a deterministic rule-based agent that codifies
  basic SRE reasoning. It serves as the scoring floor and demonstrates
  correct action sequencing for each task type.

  LLMBaselineAgent — an OpenAI-powered agent that receives the full
  observation as a structured JSON prompt and responds with an action.
  Reads OPENAI_API_KEY from the environment.

Usage:
  python scripts/baseline.py                         # heuristic on all tasks
  python scripts/baseline.py --agent llm             # LLM on all tasks
  python scripts/baseline.py --task task_easy_bad_deploy --agent heuristic
  python scripts/baseline.py --seed 999              # override seed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import ProductionIncidentEnv
from env.graders import EpisodeTrajectory, StepRecord, grade_episode
from env.models import (
    Action,
    ActionType,
    FaultType,
    Observation,
    StepResult,
)
from env.tasks import ALL_TASKS, TaskSpec, build_env_config, get_task


# ---------------------------------------------------------------------------
# Heuristic baseline
# ---------------------------------------------------------------------------


class HeuristicBaselineAgent:
    """
    Rule-based SRE agent. Implements a structured diagnosis → act → verify loop.

    Priority order:
      1. Acknowledge P1 alerts (bookkeeping, no-cost)
      2. Query diagnostics for the most degraded service (if not yet queried)
      3. Apply targeted remediation based on observed symptoms
      4. Issue NO_OP to allow recovery, then verify
    """

    def __init__(self, task: TaskSpec):
        self.task = task
        self._queried: set = set()
        self._acted_on: set = set()
        self._step = 0
        self._wait_counter = 0
        self._db_stabilized = False

    def act(self, obs: Observation, internal_state: Dict[str, Any]) -> Action:
        self._step += 1

        # --- Step 1: Acknowledge unacknowledged P1 alerts ---
        for alert in obs.alerts:
            if alert.severity == "P1" and not alert.acknowledged:
                return Action(type=ActionType.ACKNOWLEDGE_ALERT, params={"alert_id": alert.id})

        # --- Identify most problematic service ---
        violated = [
            (sid, svc)
            for sid, svc in obs.services.items()
            if svc.is_slo_violated
        ]
        if not violated:
            # All services healthy — wait for confirmation
            return Action(type=ActionType.NO_OP, params={})

        # Sort by error_rate descending (most broken first)
        violated.sort(key=lambda x: x[1].error_rate, reverse=True)
        worst_sid, worst_svc = violated[0]

        # --- Step 2: Diagnostic queries before acting ---
        diag_key_logs = f"logs:{worst_sid}"
        diag_key_metrics = f"metrics:{worst_sid}:error_rate"
        if diag_key_logs not in self._queried:
            self._queried.add(diag_key_logs)
            return Action(
                type=ActionType.QUERY_LOGS,
                params={"service_id": worst_sid, "window_minutes": 15},
            )
        if diag_key_metrics not in self._queried:
            self._queried.add(diag_key_metrics)
            return Action(
                type=ActionType.QUERY_METRICS,
                params={"service_id": worst_sid, "metric": "error_rate"},
            )

        # --- Step 3: Task-specific remediation ---
        return self._remediate(obs, internal_state, worst_sid, worst_svc)

    def _remediate(
        self,
        obs: Observation,
        internal_state: Dict[str, Any],
        worst_sid: str,
        worst_svc: Any,
    ) -> Action:
        # Version bump → rollback
        # Heuristic: if active_version > 3 (nominal) and error rate is high, assume bad deploy
        if worst_svc.active_version > 3 and worst_svc.error_rate > 0.05:
            if worst_sid not in self._acted_on:
                self._acted_on.add(worst_sid)
                return Action(
                    type=ActionType.ROLLBACK_SERVICE,
                    params={"service_id": worst_sid, "steps_back": 1},
                )

        # High CPU + high RPS → rate limit or scale out
        if worst_svc.cpu_utilization > 0.80 and worst_svc.request_rate > 200:
            if f"ratelimit:{worst_sid}" not in self._acted_on:
                self._acted_on.add(f"ratelimit:{worst_sid}")
                return Action(
                    type=ActionType.ADJUST_RATE_LIMIT,
                    params={"service_id": worst_sid, "multiplier": 0.5},
                )
            if f"scale:{worst_sid}" not in self._acted_on:
                self._acted_on.add(f"scale:{worst_sid}")
                node_max = 30  # generous upper bound for heuristic
                if worst_svc.pod_count < node_max - 2:
                    return Action(
                        type=ActionType.SCALE_SERVICE,
                        params={"service_id": worst_sid, "delta": 3},
                    )

        # DB pool pressure → query memory on callers, then rate limit callers
        if obs.infra.db_conn_pool_used_pct > 0.75:
            db_mem_key = "metrics:db-proxy:memory_utilization"
            if db_mem_key not in self._queried:
                self._queried.add(db_mem_key)
                return Action(
                    type=ActionType.QUERY_METRICS,
                    params={"service_id": "db-proxy", "metric": "memory_utilization"},
                )
            if "ratelimit:order-service" not in self._acted_on:
                self._acted_on.add("ratelimit:order-service")
                return Action(
                    type=ActionType.ADJUST_RATE_LIMIT,
                    params={"service_id": "order-service", "multiplier": 0.5},
                )
            # After pool pressure drops, restart order-service
            if obs.infra.db_conn_pool_used_pct < 0.6 and "restart:order-service" not in self._acted_on:
                self._acted_on.add("restart:order-service")
                return Action(
                    type=ActionType.RESTART_SERVICE,
                    params={"service_id": "order-service"},
                )

        # High memory → restart (last resort, costs blast budget)
        if worst_svc.memory_utilization > 0.88 and f"restart:{worst_sid}" not in self._acted_on:
            self._acted_on.add(f"restart:{worst_sid}")
            return Action(
                type=ActionType.RESTART_SERVICE,
                params={"service_id": worst_sid},
            )

        # Default: wait and observe
        return Action(type=ActionType.NO_OP, params={})


def run_heuristic_baseline(task: TaskSpec, verbose: bool = False) -> Dict[str, Any]:
    """Run the heuristic agent on a task and return graded results."""
    config = build_env_config(task)
    env = ProductionIncidentEnv(config=config)
    obs = env.reset(seed=task.seed)

    agent = HeuristicBaselineAgent(task)
    trajectory: EpisodeTrajectory = []
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        action = agent.act(obs, env.state())
        state_snap = env.state()
        result = env.step(action)

        trajectory.append(
            StepRecord(
                step=result.observation.step,
                action_type=action.type.value,
                action_params=action.params,
                result=result,
                state_snapshot=state_snap,
            )
        )
        total_reward += result.reward
        obs = result.observation
        done = result.done
        step += 1

        if verbose:
            _log_step(step, action, result)

    grade = grade_episode(task.id, trajectory, env.state())
    return {
        "task_id": task.id,
        "agent": "heuristic",
        "score": grade.score,
        "components": grade.components,
        "notes": grade.notes,
        "steps_taken": step,
        "total_reward": round(total_reward, 3),
        "terminal_reason": env.state().get("terminal_reason"),
    }


# ---------------------------------------------------------------------------
# LLM baseline (OpenAI)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.
You will receive a JSON observation describing the current state of a microservices system.
Your task is to diagnose the root cause of the incident and take appropriate remediation actions.

Available action types and their parameters:
- SCALE_SERVICE: {"service_id": str, "delta": int in [-3,-2,-1,1,2,3]}
- ROLLBACK_SERVICE: {"service_id": str, "steps_back": int in [1,2,3]}
- RESTART_SERVICE: {"service_id": str}
- OPEN_CIRCUIT_BREAKER: {"service_id": str}
- CLOSE_CIRCUIT_BREAKER: {"service_id": str}
- ADJUST_RATE_LIMIT: {"service_id": str, "multiplier": float in [0.25,0.5,0.75,1.0,1.5,2.0]}
- REROUTE_TRAFFIC: {"from_service": str, "to_service": str, "percentage": int in [25,50,75,100]}
- QUERY_LOGS: {"service_id": str, "window_minutes": int in [5,15,30]}
- QUERY_METRICS: {"service_id": str, "metric": str}
- ACKNOWLEDGE_ALERT: {"alert_id": str}
- PAGE_HUMAN: {"severity": str in ["P1","P2"]}
- NO_OP: {}

Services: api-gateway, auth-service, product-service, cart-service, order-service, payment-service, inventory-service, db-proxy

Respond ONLY with a JSON object in this format:
{"action_type": "<ACTION_TYPE>", "params": {<params>}, "reasoning": "<brief explanation>"}

Guidelines:
1. Start by querying logs/metrics for the most degraded service before acting
2. Identify the root cause before applying remediations
3. Prefer targeted, minimal-blast-radius actions
4. Do NOT restart or rollback services unless you have evidence they are the root cause
5. Monitor blast_radius_budget — it depletes with high-risk actions
6. A version > 3 on a degraded service strongly suggests a bad deploy
7. High cpu_utilization + request_rate spike suggests traffic surge
8. High db_conn_pool_used_pct suggests connection exhaustion
9. IMPORTANT - WAIT FOR RECOVERY: Services do not heal instantly. After executing a remediation action (like ROLLBACK_SERVICE or SCALE_SERVICE), you MUST issue NO_OP for several consecutive steps to allow the system health to recover and alerts to clear. 
10. SUCCESS CONDITION: To successfully resolve the incident, the system must remain healthy for several steps. Once you believe the fix is applied, keep issuing NO_OP until the task finishes."""


class LLMBaselineAgent:
    """OpenAI-powered agent. Sends the full observation as context each step."""

    def __init__(self, model: str = "gpt-4o", max_history: int = 6):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Install openai: pip install openai")

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.max_history = max_history
        self._history: List[Dict[str, str]] = []

    def act(self, obs: Observation, internal_state: Dict[str, Any]) -> Action:
        obs_payload = json.dumps(obs.model_dump(), indent=2, default=str)
        user_msg = f"Current observation (step {obs.step}):\n{obs_payload}"

        self._history.append({"role": "user", "content": user_msg})
        # Trim history to avoid context overflow
        trimmed = self._history[-self.max_history * 2:]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + trimmed,
            temperature=0.0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        self._history.append({"role": "assistant", "content": raw})

        return self._parse_action(raw)

    def _parse_action(self, raw: str) -> Action:
        # Strip markdown fences if model wraps in ```json
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            parsed = json.loads(clean)
            action_type = ActionType(parsed["action_type"])
            params = parsed.get("params", {})
            return Action(type=action_type, params=params)
        except Exception as exc:
            print(f"[LLM] Parse error ({exc}), defaulting to NO_OP. Raw: {raw[:200]}")
            return Action(type=ActionType.NO_OP, params={})


def run_llm_baseline(task: TaskSpec, model: str = "gpt-4o", verbose: bool = True) -> Dict[str, Any]:
    config = build_env_config(task)
    env = ProductionIncidentEnv(config=config)
    obs = env.reset(seed=task.seed)

    agent = LLMBaselineAgent(model=model)
    trajectory: EpisodeTrajectory = []
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        action = agent.act(obs, env.state())
        state_snap = env.state()
        result = env.step(action)

        trajectory.append(
            StepRecord(
                step=result.observation.step,
                action_type=action.type.value,
                action_params=action.params,
                result=result,
                state_snapshot=state_snap,
            )
        )
        total_reward += result.reward
        obs = result.observation
        done = result.done
        step += 1

        if verbose:
            _log_step(step, action, result)

    grade = grade_episode(task.id, trajectory, env.state())
    return {
        "task_id": task.id,
        "agent": f"llm/{model}",
        "score": grade.score,
        "components": grade.components,
        "notes": grade.notes,
        "steps_taken": step,
        "total_reward": round(total_reward, 3),
        "terminal_reason": env.state().get("terminal_reason"),
    }


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_step(step: int, action: Action, result: StepResult) -> None:
    valid = result.info.get("action_valid", True)
    outcome = result.info.get("action_outcome", "?")
    terminal = result.info.get("terminal", "")
    reward_str = f"{result.reward:+.3f}"

    params_str = json.dumps(action.params) if action.params else "{}"
    status = "✓" if valid else "✗"
    term_str = f"  [TERMINAL: {terminal}]" if terminal else ""

    print(
        f"  Step {step:3d} | {status} {action.type.value:<28} {params_str:<45} "
        f"r={reward_str}  outcome={outcome}{term_str}"
    )


def _print_result(result: Dict[str, Any]) -> None:
    task_id = result["task_id"]
    score = result["score"]
    steps = result["steps_taken"]
    terminal = result.get("terminal_reason", "unknown")
    agent = result.get("agent", "?")

    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"\n{'─'*70}")
    print(f"  Task:     {task_id}")
    print(f"  Agent:    {agent}")
    print(f"  Score:    {score:.4f}  [{bar}]")
    print(f"  Steps:    {steps}")
    print(f"  Terminal: {terminal}")
    print(f"  Reward:   {result['total_reward']:.3f}")
    print(f"  Components:")
    for k, v in result["components"].items():
        print(f"    {k:<30} {v:.4f}")
    print(f"  Notes:")
    for note in result["notes"]:
        print(f"    • {note}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline agents on ProductionIncidentEnv")
    parser.add_argument(
        "--agent",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Which agent to run (default: heuristic)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task ID to run (default: all tasks)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model for LLM agent (default: gpt-4o)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override task seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each step",
    )
    args = parser.parse_args()

    tasks = [get_task(args.task)] if args.task else ALL_TASKS

    print(f"\n{'='*70}")
    print(f"  ProductionIncidentEnv — Baseline Evaluation")
    print(f"  Agent: {args.agent.upper()}")
    print(f"  Tasks: {[t.id for t in tasks]}")
    print(f"{'='*70}\n")

    results = []
    for task in tasks:
        if args.seed is not None:
            # Patch seed for reproducibility override
            object.__setattr__(task, "seed", args.seed)  # frozen dataclass workaround

        print(f"\n[Task: {task.name} ({task.difficulty.upper()})]")
        print(f"  Objective: {task.objective}\n")

        t0 = time.time()
        if args.agent == "heuristic":
            result = run_heuristic_baseline(task, verbose=args.verbose)
        else:
            if "OPENAI_API_KEY" not in os.environ:
                print("ERROR: OPENAI_API_KEY not set in environment.")
                sys.exit(1)
            result = run_llm_baseline(task, model=args.model, verbose=args.verbose)

        result["duration_seconds"] = round(time.time() - t0, 2)
        _print_result(result)
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n{'='*70}")
    print(f"  AVERAGE SCORE: {avg_score:.4f}")
    print(f"  TASK SCORES:   " + "  |  ".join(f"{r['task_id'].split('_')[1]}: {r['score']:.4f}" for r in results))
    print(f"{'='*70}\n")

    # Machine-readable summary
    summary = {
        "agent": args.agent,
        "tasks": results,
        "average_score": round(avg_score, 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
