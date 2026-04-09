from __future__ import annotations

"""
Task definitions for ProductionIncidentEnv.

Each task is a fully specified scenario with a deterministic seed, a fixed
fault configuration, and a grader-compatible success criterion. Tasks are
ordered by the skill they stress:

  Task 1 (easy)   — single-service rollback, clear signal, short trajectory
  Task 2 (medium) — traffic surge with cascading degradation, multi-service
  Task 3 (hard)   — compound fault (DB exhaustion + memory leak), requires
                    diagnostic reasoning and sequenced remediations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from env.models import ActionType, EnvConfig, FaultType


@dataclass(frozen=True)
class TaskSpec:
    id: str
    name: str
    difficulty: str  # "easy" | "medium" | "hard"
    description: str
    objective: str
    fault_type: FaultType
    fault_origin: str
    seed: int
    max_steps: int
    env_config_overrides: Dict[str, Any] = field(default_factory=dict)
    # Grader metadata — interpreted by the corresponding grader
    grader_params: Dict[str, Any] = field(default_factory=dict)
    # Hints about the action schema a baseline should consider
    relevant_actions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 1 — Easy: Isolated Bad Deploy
# ---------------------------------------------------------------------------
#
# A single service (cart-service) received a bad deployment 3 steps before
# the agent starts. The bad version has a 15% error rate and 2x latency.
# The service has a clean 3-version history, so a single ROLLBACK_SERVICE
# with steps_back=1 completely resolves the incident.
#
# A competent agent should identify the version bump in observations and
# issue the rollback within 10 steps. The grader rewards for:
#   - Resolution achieved at all (+0.4)
#   - Rollback correctly targeting cart-service (+0.3)
#   - Resolved within 20 steps (+0.2)
#   - Blast radius budget > 0.7 at resolution (+0.1)

TASK_EASY = TaskSpec(
    id="task_easy_bad_deploy",
    name="Isolated Bad Deployment",
    difficulty="easy",
    description=(
        "cart-service was just deployed with a bad artifact. Its error rate has "
        "spiked to ~15% and p99 latency doubled. No other services are significantly "
        "affected yet. Identify the bad deploy and roll it back."
    ),
    objective=(
        "Roll back cart-service to a healthy version and restore all services to SLO "
        "compliance within 20 steps."
    ),
    fault_type=FaultType.BAD_DEPLOY,
    fault_origin="cart-service",
    seed=42,
    max_steps=60,
    env_config_overrides={"max_steps": 60},
    grader_params={
        "resolution_step_threshold": 20,
        "target_service": "cart-service",
        "expected_action": ActionType.ROLLBACK_SERVICE.value,
        "blast_budget_threshold": 0.7,
    },
    relevant_actions=[
        ActionType.QUERY_LOGS.value,
        ActionType.QUERY_METRICS.value,
        ActionType.ROLLBACK_SERVICE.value,
        ActionType.ACKNOWLEDGE_ALERT.value,
    ],
)


# ---------------------------------------------------------------------------
# Task 2 — Medium: Traffic Surge Cascade
# ---------------------------------------------------------------------------
#
# A Black Friday-style traffic surge has hit api-gateway at 4x nominal RPS.
# The gate is saturating, which is cascading latency and errors downstream
# into auth-service and cart-service. The agent must:
#   1. Recognize the RPS spike on api-gateway (not a code bug)
#   2. Apply rate limiting or scale out api-gateway pods
#   3. Allow downstream services to recover naturally
#
# The fault naturally evolves: if the agent does nothing, CPU saturation
# crosses 95% by step 12 and error rates cascade to all dependent services.
#
# Grader rewards:
#   - Resolution achieved (+0.35)
#   - Correct root-cause action on api-gateway (SCALE or RATE_LIMIT) (+0.25)
#   - No unnecessary actions on non-origin services (+0.15)
#   - Resolved within 40 steps (+0.15)
#   - Blast radius budget > 0.6 (+0.10)

TASK_MEDIUM = TaskSpec(
    id="task_medium_traffic_surge",
    name="Traffic Surge Cascade",
    difficulty="medium",
    description=(
        "api-gateway is receiving 4x its normal request rate. CPU utilization is "
        "climbing and p99 latency is spiking. auth-service and cart-service are "
        "beginning to show elevated error rates due to the backpressure. Diagnose "
        "the root cause and implement a remediation that stops the cascade."
    ),
    objective=(
        "Diagnose the traffic surge at api-gateway and apply appropriate mitigation "
        "(rate limiting or horizontal scaling) to restore SLO compliance across all "
        "services within 40 steps."
    ),
    fault_type=FaultType.TRAFFIC_SURGE,
    fault_origin="api-gateway",
    seed=137,
    max_steps=100,
    env_config_overrides={"max_steps": 100},
    grader_params={
        "resolution_step_threshold": 40,
        "target_service": "api-gateway",
        "correct_action_types": [
            ActionType.SCALE_SERVICE.value,
            ActionType.ADJUST_RATE_LIMIT.value,
        ],
        "non_origin_services": [
            "auth-service", "product-service", "cart-service",
            "order-service", "payment-service", "inventory-service", "db-proxy",
        ],
        "blast_budget_threshold": 0.6,
        "max_non_origin_actions": 3,
    },
    relevant_actions=[
        ActionType.QUERY_METRICS.value,
        ActionType.SCALE_SERVICE.value,
        ActionType.ADJUST_RATE_LIMIT.value,
        ActionType.QUERY_LOGS.value,
        ActionType.NO_OP.value,
    ],
)


# ---------------------------------------------------------------------------
# Task 3 — Hard: DB Connection Exhaustion + Induced Memory Leak
# ---------------------------------------------------------------------------
#
# db-proxy is leaking database connections — its pool is filling up at ~5/step.
# Simultaneously, order-service has a slow memory leak caused by unfreed
# connection handles it accumulated while waiting for db-proxy to respond.
# The agent faces:
#   - Two faults interacting (compounding, not independent)
#   - db-proxy symptoms visible first; order-service memory bloat emerges ~15 steps in
#   - Fixing db-proxy alone doesn't resolve order-service memory state
#   - Must restart order-service (draining blast budget) AFTER stabilizing db-proxy
#   - Restarting order-service before fixing db-proxy causes immediate re-leak
#
# Correct sequence:
#   1. Diagnose db-proxy (QUERY_METRICS: db_conn_pool_used_pct)
#   2. Scale down db-proxy callers OR rate limit to reduce new connections
#   3. Once pool pressure < 60%, restart order-service to clear leaked handles
#   4. Monitor and confirm recovery
#
# Grader rewards:
#   - Full resolution (+0.30)
#   - db-proxy addressed before order-service restart (+0.20)
#   - order-service restarted after pool pressure drops (+0.15)
#   - Correct use of diagnostic queries (+0.15)
#   - Resolved within 70 steps (+0.10)
#   - Blast radius budget > 0.4 at resolution (+0.10)

TASK_HARD = TaskSpec(
    id="task_hard_db_cascade",
    name="DB Connection Exhaustion with Memory Cascade",
    difficulty="hard",
    description=(
        "db-proxy is experiencing connection pool exhaustion — the pool is filling "
        "up at roughly 5 connections per step. order-service, which makes heavy use "
        "of db-proxy, has started accumulating leaked connection handles in memory. "
        "If order-service is restarted before the db-proxy issue is addressed, it "
        "will immediately re-accumulate the leak. You must sequence your remediations "
        "correctly: stabilize the database layer first, then address the application "
        "memory state."
    ),
    objective=(
        "Stabilize db-proxy connection pool utilization below 60%, then restart "
        "order-service to clear its memory leak. Restore all services to SLO "
        "compliance within 70 steps without triggering a catastrophic failure."
    ),
    fault_type=FaultType.DB_CONN_EXHAUSTION,
    fault_origin="db-proxy",
    seed=314,
    max_steps=150,
    env_config_overrides={"max_steps": 150},
    grader_params={
        "resolution_step_threshold": 70,
        "primary_service": "db-proxy",
        "secondary_service": "order-service",
        "pool_pressure_threshold": 0.6,
        "blast_budget_threshold": 0.4,
        "correct_db_actions": [
            ActionType.ADJUST_RATE_LIMIT.value,
            ActionType.SCALE_SERVICE.value,
            ActionType.OPEN_CIRCUIT_BREAKER.value,
        ],
        "required_secondary_action": ActionType.RESTART_SERVICE.value,
        "diagnostic_bonus_queries": ["db_conn_pool_used_pct", "memory_utilization"],
    },
    relevant_actions=[
        ActionType.QUERY_METRICS.value,
        ActionType.QUERY_LOGS.value,
        ActionType.ADJUST_RATE_LIMIT.value,
        ActionType.SCALE_SERVICE.value,
        ActionType.OPEN_CIRCUIT_BREAKER.value,
        ActionType.RESTART_SERVICE.value,
        ActionType.CLOSE_CIRCUIT_BREAKER.value,
        ActionType.NO_OP.value,
    ],
)


ALL_TASKS: List[TaskSpec] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]

TASK_REGISTRY: Dict[str, TaskSpec] = {t.id: t for t in ALL_TASKS}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_id]


def build_env_config(task: TaskSpec) -> EnvConfig:
    base = EnvConfig(
        fault_type=task.fault_type,
        fault_origin=task.fault_origin,
        max_steps=task.max_steps,
        noise_seed=task.seed,
    )
    overrides = task.env_config_overrides
    if overrides:
        return base.model_copy(update=overrides)
    return base
