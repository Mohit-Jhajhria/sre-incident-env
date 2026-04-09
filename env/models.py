from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FaultType(str, Enum):
    MEMORY_LEAK = "MEMORY_LEAK"
    BAD_DEPLOY = "BAD_DEPLOY"
    TRAFFIC_SURGE = "TRAFFIC_SURGE"
    DB_CONN_EXHAUSTION = "DB_CONN_EXHAUSTION"
    DEPENDENCY_LATENCY = "DEPENDENCY_LATENCY"
    NETWORK_PARTITION = "NETWORK_PARTITION"


class Severity(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class ActionType(str, Enum):
    SCALE_SERVICE = "SCALE_SERVICE"
    ROLLBACK_SERVICE = "ROLLBACK_SERVICE"
    RESTART_SERVICE = "RESTART_SERVICE"
    OPEN_CIRCUIT_BREAKER = "OPEN_CIRCUIT_BREAKER"
    CLOSE_CIRCUIT_BREAKER = "CLOSE_CIRCUIT_BREAKER"
    ADJUST_RATE_LIMIT = "ADJUST_RATE_LIMIT"
    REROUTE_TRAFFIC = "REROUTE_TRAFFIC"
    QUERY_LOGS = "QUERY_LOGS"
    QUERY_METRICS = "QUERY_METRICS"
    ACKNOWLEDGE_ALERT = "ACKNOWLEDGE_ALERT"
    PAGE_HUMAN = "PAGE_HUMAN"
    NO_OP = "NO_OP"


class TerminalReason(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    CATASTROPHIC = "CATASTROPHIC"
    ESCALATION = "ESCALATION"


class ActionOutcome(str, Enum):
    APPLIED = "applied"
    NO_EFFECT = "no_effect"
    WORSENED = "worsened"
    INVALID = "invalid"


# ---------------------------------------------------------------------------
# Topology / Configuration Models
# ---------------------------------------------------------------------------


class ServiceNode(BaseModel):
    """Static topology definition for a single service. Never mutated after init."""

    id: str
    tier: int  # 0=edge, 1=core, 2=data
    dependencies: List[str] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)
    nominal_rps: float = 100.0
    slo_error_rate: float = 0.01
    slo_latency_p99: float = 200.0
    pod_min: int = 1
    pod_max: int = 20
    nominal_pod_count: int = 4


# ---------------------------------------------------------------------------
# Runtime State Models (internal, not directly observable)
# ---------------------------------------------------------------------------


class ServiceState(BaseModel):
    pod_count: int = 4
    error_rate: float = 0.001
    latency_p50: float = 50.0
    latency_p99: float = 80.0
    cpu_utilization: float = 0.35
    memory_utilization: float = 0.40
    request_rate: float = 100.0
    circuit_breaker_open: bool = False
    rate_limit_multiplier: float = 1.0
    active_version: int = 3
    version_history: List[int] = Field(default_factory=lambda: [1, 2, 3])
    is_crashing: bool = False
    # Internal fault propagation signal; hidden from agent
    degradation_factor: float = 0.0

    def is_slo_violated(self, node: ServiceNode) -> bool:
        return (
            self.error_rate > node.slo_error_rate
            or self.latency_p99 > node.slo_latency_p99
        )

    def health_score(self, node: ServiceNode) -> float:
        err_ratio = min(self.error_rate / max(node.slo_error_rate, 1e-6), 1.0)
        lat_ratio = min(self.latency_p99 / max(node.slo_latency_p99, 1e-6), 1.0)
        return 1.0 - err_ratio * 0.5 - lat_ratio * 0.5


class InfraState(BaseModel):
    total_node_capacity: int = 120
    db_conn_pool_used: int = 20
    db_conn_pool_max: int = 100
    # Serialized as list of 2-tuples for JSON compatibility
    network_partitions: List[Tuple[str, str]] = Field(default_factory=list)


class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    service_id: str
    metric: str
    severity: Severity
    threshold: float
    current_val: float
    fired_at: int
    resolved: bool = False


class ChangeLogEntry(BaseModel):
    step: int
    action_type: str
    action_params: Dict[str, Any]
    outcome: ActionOutcome
    health_delta: float  # aggregate health change after action


class IncidentState(BaseModel):
    fault_type: FaultType
    fault_origin: str
    fault_params: Dict[str, Any] = Field(default_factory=dict)
    injected_at_step: int = 0
    active_alerts: List[Alert] = Field(default_factory=list)
    acknowledged_alerts: Set[str] = Field(default_factory=set)
    step_count: int = 0
    blast_radius_budget: float = 1.0
    consecutive_healthy_steps: int = 0
    # Track which (service, metric) pairs have already been queried for R8
    queried_diagnostics: Set[str] = Field(default_factory=set)
    # Track recent (action_type, params_hash) to detect redundancy
    recent_actions: List[Tuple[str, str]] = Field(default_factory=list)
    consecutive_nop_count: int = 0
    terminal_reason: Optional[TerminalReason] = None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Observation Models (agent-visible)
# ---------------------------------------------------------------------------


class ServiceObservation(BaseModel):
    error_rate: float
    latency_p50: float
    latency_p99: float
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    pod_count: int
    circuit_breaker_open: bool
    rate_limit_multiplier: float
    active_version: int
    is_slo_violated: bool
    slo_error_rate: float
    slo_latency_p99: float


class AlertObservation(BaseModel):
    id: str
    service_id: str
    metric: str
    severity: str
    current_val: float
    age_steps: int
    acknowledged: bool


class InfraObservation(BaseModel):
    cluster_capacity_used_pct: float
    db_conn_pool_used_pct: float


class Observation(BaseModel):
    services: Dict[str, ServiceObservation]
    alerts: List[AlertObservation]
    infra: InfraObservation
    change_log: List[ChangeLogEntry]
    step: int
    time_remaining: int
    blast_radius_budget: float
    query_result: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------


class Action(BaseModel):
    type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def params_must_be_serializable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Shallow check; deeper validation happens in env
        for key in v:
            if not isinstance(key, str):
                raise ValueError("Action param keys must be strings")
        return v


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------


class RewardComponents(BaseModel):
    slo_health: float = 0.0
    resolution_bonus: float = 0.0
    invalid_action: float = 0.0
    redundant_action: float = 0.0
    worsening: float = 0.0
    blast_radius: float = 0.0
    escalation: float = 0.0
    diagnostic_novelty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.slo_health
            + self.resolution_bonus
            + self.invalid_action
            + self.redundant_action
            + self.worsening
            + self.blast_radius
            + self.escalation
            + self.diagnostic_novelty
        )


class StepResult(BaseModel):
    observation: Observation
    reward: float
    reward_components: RewardComponents
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment Config
# ---------------------------------------------------------------------------


class EnvConfig(BaseModel):
    max_steps: int = 300
    resolution_threshold: int = 3  # consecutive healthy steps required
    noise_seed: Optional[int] = None
    fault_type: Optional[FaultType] = None  # None = random
    fault_origin: Optional[str] = None  # None = random
    scale_slo: float = 2.0
    resolution_bonus: float = 50.0
    catastrophic_penalty: float = 50.0
    invalid_action_penalty: float = 2.0
    redundant_action_penalty: float = 1.5
    diagnostic_novelty_reward: float = 0.3
