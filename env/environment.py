from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action,
    ActionOutcome,
    ActionType,
    Alert,
    AlertObservation,
    ChangeLogEntry,
    EnvConfig,
    FaultType,
    IncidentState,
    InfraObservation,
    InfraState,
    Observation,
    RewardComponents,
    ServiceNode,
    ServiceObservation,
    ServiceState,
    Severity,
    StepResult,
    TerminalReason,
)

# ---------------------------------------------------------------------------
# Default topology: simplified e-commerce microservices stack
# ---------------------------------------------------------------------------
#
# Traffic flows:  internet → api-gateway → {auth, product, cart}
#                 cart → order-service → {payment, inventory}
#                 {auth, cart, order, inventory} → db-proxy
#                 product → inventory
#
# This gives a realistic 3-tier DAG with multiple dependency paths.

DEFAULT_TOPOLOGY: List[ServiceNode] = [
    ServiceNode(
        id="api-gateway",
        tier=0,
        dependencies=[],
        dependents=["auth-service", "product-service", "cart-service"],
        nominal_rps=500.0,
        slo_error_rate=0.005,
        slo_latency_p99=150.0,
        pod_min=2,
        pod_max=30,
        nominal_pod_count=6,
    ),
    ServiceNode(
        id="auth-service",
        tier=1,
        dependencies=["api-gateway"],
        dependents=["db-proxy"],
        nominal_rps=450.0,
        slo_error_rate=0.01,
        slo_latency_p99=200.0,
        pod_min=2,
        pod_max=20,
        nominal_pod_count=4,
    ),
    ServiceNode(
        id="product-service",
        tier=1,
        dependencies=["api-gateway"],
        dependents=["inventory-service"],
        nominal_rps=300.0,
        slo_error_rate=0.01,
        slo_latency_p99=300.0,
        pod_min=1,
        pod_max=20,
        nominal_pod_count=4,
    ),
    ServiceNode(
        id="cart-service",
        tier=1,
        dependencies=["api-gateway"],
        dependents=["order-service", "db-proxy"],
        nominal_rps=200.0,
        slo_error_rate=0.01,
        slo_latency_p99=250.0,
        pod_min=1,
        pod_max=16,
        nominal_pod_count=3,
    ),
    ServiceNode(
        id="order-service",
        tier=1,
        dependencies=["cart-service"],
        dependents=["payment-service", "inventory-service", "db-proxy"],
        nominal_rps=80.0,
        slo_error_rate=0.005,
        slo_latency_p99=500.0,
        pod_min=1,
        pod_max=16,
        nominal_pod_count=3,
    ),
    ServiceNode(
        id="payment-service",
        tier=2,
        dependencies=["order-service"],
        dependents=["db-proxy"],
        nominal_rps=80.0,
        slo_error_rate=0.001,
        slo_latency_p99=800.0,
        pod_min=2,
        pod_max=12,
        nominal_pod_count=4,
    ),
    ServiceNode(
        id="inventory-service",
        tier=2,
        dependencies=["order-service", "product-service"],
        dependents=["db-proxy"],
        nominal_rps=150.0,
        slo_error_rate=0.01,
        slo_latency_p99=400.0,
        pod_min=1,
        pod_max=16,
        nominal_pod_count=3,
    ),
    ServiceNode(
        id="db-proxy",
        tier=2,
        dependencies=["auth-service", "cart-service", "order-service", "payment-service", "inventory-service"],
        dependents=[],
        nominal_rps=600.0,
        slo_error_rate=0.001,
        slo_latency_p99=100.0,
        pod_min=2,
        pod_max=8,
        nominal_pod_count=4,
    ),
]

# How strongly a dependency's degradation propagates to its dependents.
# Keyed as (dependency_id, dependent_id): propagation_weight
PROPAGATION_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("api-gateway", "auth-service"): 0.7,
    ("api-gateway", "product-service"): 0.7,
    ("api-gateway", "cart-service"): 0.7,
    ("auth-service", "db-proxy"): 0.5,
    ("cart-service", "order-service"): 0.8,
    ("cart-service", "db-proxy"): 0.5,
    ("order-service", "payment-service"): 0.9,
    ("order-service", "inventory-service"): 0.6,
    ("order-service", "db-proxy"): 0.6,
    ("payment-service", "db-proxy"): 0.5,
    ("inventory-service", "db-proxy"): 0.4,
    ("product-service", "inventory-service"): 0.5,
}


def _propagation_weight(dependency_id: str, dependent_id: str) -> float:
    return PROPAGATION_WEIGHTS.get((dependency_id, dependent_id), 0.3)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ProductionIncidentEnv:
    """
    OpenEnv-compliant environment simulating SRE incident response.

    The agent must diagnose and remediate a production fault across a
    microservices topology before SLO violations cascade into a full outage.
    """

    def __init__(self, config: Optional[EnvConfig] = None, topology: Optional[List[ServiceNode]] = None):
        self.config = config or EnvConfig()
        self.topology = topology or DEFAULT_TOPOLOGY
        self._node_map: Dict[str, ServiceNode] = {n.id: n for n in self.topology}

        self._rng = random.Random(self.config.noise_seed)
        self._obs_rng = random.Random((self.config.noise_seed or 0) + 1)

        # These are set by reset()
        self._service_states: Dict[str, ServiceState] = {}
        self._infra: InfraState = InfraState()
        self._incident: IncidentState = None  # type: ignore[assignment]
        self._change_log: List[ChangeLogEntry] = []
        self._prev_aggregate_health: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        fault_type: Optional[FaultType] = None,
        fault_origin: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
            self._obs_rng = random.Random(seed + 1)

        self._initialize_healthy_state()

        chosen_fault = fault_type or self.config.fault_type or self._rng.choice(list(FaultType))
        chosen_origin = fault_origin or self.config.fault_origin or self._pick_fault_origin(chosen_fault)

        fault_params = self._build_fault_params(chosen_fault, chosen_origin)
        self._incident = IncidentState(
            fault_type=chosen_fault,
            fault_origin=chosen_origin,
            fault_params=fault_params,
            injected_at_step=0,
        )

        # Apply initial perturbation so the agent sees something non-trivial at step 0
        self._apply_fault_tick(initial=True)
        self._fire_alerts()

        self._prev_aggregate_health = self._aggregate_health()
        self._change_log = []

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        if self._incident is None:
            raise RuntimeError("Call reset() before step()")

        self._incident.step_count += 1
        prev_health = self._aggregate_health()

        reward_components = RewardComponents()
        query_result: Optional[Dict[str, Any]] = None
        action_outcome = ActionOutcome.APPLIED

        # --- Execute action ---
        valid, reason = self._validate_action(action)
        if not valid:
            reward_components.invalid_action = -self.config.invalid_action_penalty
            action_outcome = ActionOutcome.INVALID
        else:
            query_result = self._execute_action(action)
            action_outcome = self._classify_outcome(action, prev_health)

        # --- Redundancy check (only for non-diagnostic actions) ---
        if valid and action.type not in (ActionType.QUERY_LOGS, ActionType.QUERY_METRICS, ActionType.NO_OP):
            if self._is_redundant(action, action_outcome):
                reward_components.redundant_action = -self.config.redundant_action_penalty

        # --- NOP on active P1 check ---
        if action.type == ActionType.NO_OP:
            self._incident.consecutive_nop_count += 1
        else:
            self._incident.consecutive_nop_count = 0

        has_p1 = any(a.severity == Severity.P1 for a in self._incident.active_alerts if not a.resolved)
        if self._incident.consecutive_nop_count > 3 and has_p1:
            reward_components.redundant_action = min(
                reward_components.redundant_action - self.config.redundant_action_penalty,
                -self.config.redundant_action_penalty,
            )

        # --- Record in change log ---
        post_health = self._aggregate_health()
        self._append_change_log(action, action_outcome, post_health - prev_health)
        self._update_recent_actions(action, action_outcome)

        # --- Advance world state ---
        self._apply_fault_tick()
        self._fire_alerts()
        self._update_consecutive_healthy()

        # --- Compute rewards ---
        current_health = self._aggregate_health()
        reward_components.slo_health = current_health * self.config.scale_slo

        health_drop = prev_health - current_health
        # Only penalize agent-caused degradation, not natural fault progression.
        # We attribute degradation to the agent if the action was consequential and health dropped.
        if valid and action_outcome == ActionOutcome.WORSENED and health_drop > 0.05:
            reward_components.worsening = -3.0 * health_drop

        if self._incident.blast_radius_budget < 0.3:
            reward_components.blast_radius = -5.0 * (1.0 - self._incident.blast_radius_budget)

        if action.type in (ActionType.QUERY_LOGS, ActionType.QUERY_METRICS) and valid:
            key = f"{action.params.get('service_id')}:{action.params.get('metric', 'logs')}"
            if key not in self._incident.queried_diagnostics:
                self._incident.queried_diagnostics.add(key)
                reward_components.diagnostic_novelty = self.config.diagnostic_novelty_reward

        # --- Terminal conditions ---
        done = False
        info: Dict[str, Any] = {"action_valid": valid, "action_outcome": action_outcome.value}

        if action.type == ActionType.PAGE_HUMAN and valid:
            done = True
            self._incident.terminal_reason = TerminalReason.ESCALATION
            paging_health = current_health
            # Credit escalation if things are genuinely broken, penalize if premature
            if paging_health > 0.8:
                reward_components.escalation = -10.0
            else:
                reward_components.escalation = self.config.resolution_bonus * 0.4
            info["terminal"] = TerminalReason.ESCALATION.value

        elif self._incident.blast_radius_budget <= 0.0:
            done = True
            self._incident.terminal_reason = TerminalReason.CATASTROPHIC
            reward_components.resolution_bonus = -self.config.catastrophic_penalty
            info["terminal"] = TerminalReason.CATASTROPHIC.value

        elif self._incident.consecutive_healthy_steps >= self.config.resolution_threshold:
            done = True
            self._incident.terminal_reason = TerminalReason.SUCCESS
            speed_factor = max(0.0, (self.config.max_steps - self._incident.step_count) / self.config.max_steps) * 0.5
            reward_components.resolution_bonus = self.config.resolution_bonus * (1.0 + speed_factor)
            info["terminal"] = TerminalReason.SUCCESS.value

        elif self._incident.step_count >= self.config.max_steps:
            done = True
            self._incident.terminal_reason = TerminalReason.TIMEOUT
            info["terminal"] = TerminalReason.TIMEOUT.value

        self._prev_aggregate_health = current_health
        obs = self._build_observation(query_result=query_result)

        return StepResult(
            observation=obs,
            reward=reward_components.total,
            reward_components=reward_components,
            done=done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """Full internal state snapshot (for debugging and graders)."""
        if self._incident is None:
            return {}
        return {
            "step": self._incident.step_count,
            "fault_type": self._incident.fault_type.value,
            "fault_origin": self._incident.fault_origin,
            "fault_params": self._incident.fault_params,
            "blast_radius_budget": self._incident.blast_radius_budget,
            "consecutive_healthy_steps": self._incident.consecutive_healthy_steps,
            "terminal_reason": self._incident.terminal_reason.value if self._incident.terminal_reason else None,
            "aggregate_health": self._aggregate_health(),
            "services": {
                sid: {
                    "error_rate": s.error_rate,
                    "latency_p99": s.latency_p99,
                    "cpu_utilization": s.cpu_utilization,
                    "memory_utilization": s.memory_utilization,
                    "pod_count": s.pod_count,
                    "circuit_breaker_open": s.circuit_breaker_open,
                    "active_version": s.active_version,
                    "degradation_factor": s.degradation_factor,
                    "is_crashing": s.is_crashing,
                    "is_slo_violated": s.is_slo_violated(self._node_map[sid]),
                }
                for sid, s in self._service_states.items()
            },
            "infra": self._infra.model_dump(),
            "active_alerts": [
                {"id": a.id, "service_id": a.service_id, "metric": a.metric, "severity": a.severity.value}
                for a in self._incident.active_alerts
                if not a.resolved
            ],
        }

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_healthy_state(self) -> None:
        self._service_states = {}
        for node in self.topology:
            self._service_states[node.id] = ServiceState(
                pod_count=node.nominal_pod_count,
                error_rate=self._rng.uniform(0.0001, 0.002),
                latency_p50=self._rng.uniform(40.0, 70.0),
                latency_p99=self._rng.uniform(60.0, 100.0),
                cpu_utilization=self._rng.uniform(0.25, 0.45),
                memory_utilization=self._rng.uniform(0.30, 0.50),
                request_rate=node.nominal_rps * self._rng.uniform(0.85, 1.15),
                active_version=3,
                version_history=[1, 2, 3],
                degradation_factor=0.0,
            )

        total_pods = sum(s.pod_count for s in self._service_states.values())
        self._infra = InfraState(
            total_node_capacity=120,
            db_conn_pool_used=20,
            db_conn_pool_max=100,
            network_partitions=[],
        )
        self._change_log = []
        self._prev_aggregate_health = 1.0

    def _pick_fault_origin(self, fault_type: FaultType) -> str:
        # Traffic surges always start at the edge; DB issues at data tier; others random
        if fault_type == FaultType.TRAFFIC_SURGE:
            candidates = [n.id for n in self.topology if n.tier == 0]
        elif fault_type in (FaultType.DB_CONN_EXHAUSTION,):
            candidates = [n.id for n in self.topology if n.tier == 2]
        elif fault_type == FaultType.NETWORK_PARTITION:
            # Need two services; we return the "source" and store the pair in params
            candidates = [n.id for n in self.topology if n.tier != 2]
        else:
            candidates = [n.id for n in self.topology]
        return self._rng.choice(candidates)

    def _build_fault_params(self, fault_type: FaultType, origin: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fault_type == FaultType.MEMORY_LEAK:
            params["leak_rate"] = self._rng.uniform(0.03, 0.07)  # degradation_factor increase per step
        elif fault_type == FaultType.BAD_DEPLOY:
            # Bad version has a high intrinsic error rate
            params["bad_version"] = 4
            params["bad_error_rate"] = self._rng.uniform(0.08, 0.25)
            params["bad_latency_multiplier"] = self._rng.uniform(1.5, 3.0)
            # Immediately deploy the bad version
            self._service_states[origin].active_version = 4
            self._service_states[origin].version_history = [2, 3, 4]
        elif fault_type == FaultType.TRAFFIC_SURGE:
            params["surge_multiplier"] = self._rng.uniform(3.0, 5.0)
            params["ramp_steps"] = self._rng.randint(3, 8)
        elif fault_type == FaultType.DB_CONN_EXHAUSTION:
            params["leak_rate_conns"] = self._rng.randint(3, 7)  # conns consumed per step
        elif fault_type == FaultType.DEPENDENCY_LATENCY:
            params["latency_multiplier"] = self._rng.uniform(4.0, 10.0)
        elif fault_type == FaultType.NETWORK_PARTITION:
            # Pick a partner service in a different tier to partition with
            others = [n.id for n in self.topology if n.id != origin]
            partner = self._rng.choice(others)
            params["partner"] = partner
            self._infra.network_partitions = [(origin, partner)]
        return params

    # ------------------------------------------------------------------
    # World simulation
    # ------------------------------------------------------------------

    def _apply_fault_tick(self, initial: bool = False) -> None:
        """Advance the fault one step forward, then propagate through the DAG."""
        origin = self._incident.fault_origin
        fault = self._incident.fault_type
        params = self._incident.fault_params
        step = self._incident.step_count

        origin_state = self._service_states[origin]
        origin_node = self._node_map[origin]

        if fault == FaultType.MEMORY_LEAK:
            rate = params["leak_rate"]
            origin_state.degradation_factor = min(origin_state.degradation_factor + rate, 1.0)
            origin_state.memory_utilization = min(0.3 + origin_state.degradation_factor * 0.7, 1.0)
            if origin_state.degradation_factor > 0.85:
                origin_state.is_crashing = True
                origin_state.error_rate = min(origin_state.error_rate + 0.15, 1.0)
            else:
                origin_state.latency_p99 *= 1.0 + origin_state.degradation_factor * 0.3

        elif fault == FaultType.BAD_DEPLOY:
            if origin_state.active_version == params["bad_version"]:
                origin_state.error_rate = params["bad_error_rate"]
                origin_state.latency_p99 = (
                    origin_node.slo_latency_p99 * params["bad_latency_multiplier"]
                )
                origin_state.degradation_factor = min(
                    params["bad_error_rate"] / max(origin_node.slo_error_rate, 1e-6), 1.0
                )

        elif fault == FaultType.TRAFFIC_SURGE:
            ramp = params["ramp_steps"]
            mult = params["surge_multiplier"]
            progress = min(step / max(ramp, 1), 1.0) if not initial else 0.3
            effective_mult = 1.0 + (mult - 1.0) * progress
            origin_state.request_rate = origin_node.nominal_rps * effective_mult
            origin_state.cpu_utilization = min(0.35 + (effective_mult - 1.0) * 0.25, 1.0)
            if origin_state.cpu_utilization > 0.85:
                origin_state.latency_p99 = origin_node.slo_latency_p99 * (
                    1.0 + (origin_state.cpu_utilization - 0.85) * 8.0
                )
                origin_state.error_rate = min((origin_state.cpu_utilization - 0.85) * 2.0, 1.0)
            origin_state.degradation_factor = max(origin_state.cpu_utilization - 0.5, 0.0) * 2.0

        elif fault == FaultType.DB_CONN_EXHAUSTION:
            leak = params["leak_rate_conns"]
            self._infra.db_conn_pool_used = min(
                self._infra.db_conn_pool_used + leak, self._infra.db_conn_pool_max
            )
            pool_pressure = self._infra.db_conn_pool_used / self._infra.db_conn_pool_max
            origin_state.degradation_factor = pool_pressure
            if pool_pressure > 0.9:
                origin_state.error_rate = min((pool_pressure - 0.9) * 5.0, 1.0)
                origin_state.latency_p99 = origin_node.slo_latency_p99 * (1.0 + pool_pressure * 3.0)
            origin_state.memory_utilization = min(0.3 + pool_pressure * 0.5, 1.0)

        elif fault == FaultType.DEPENDENCY_LATENCY:
            # Slow external call makes origin service's p99 climb steadily
            target_lat = origin_node.slo_latency_p99 * params["latency_multiplier"]
            origin_state.latency_p99 = min(
                origin_state.latency_p99 + (target_lat - origin_state.latency_p99) * 0.2,
                target_lat,
            )
            origin_state.degradation_factor = min(
                origin_state.latency_p99 / (origin_node.slo_latency_p99 * params["latency_multiplier"]),
                1.0,
            )

        elif fault == FaultType.NETWORK_PARTITION:
            # Handled by propagation: partitioned pairs see 100% errors on cross-calls
            origin_state.degradation_factor = 1.0
            origin_state.error_rate = 1.0

        self._propagate_degradation()
        self._apply_recovery_dynamics()

    def _propagate_degradation(self) -> None:
        """BFS from fault origin outward through the dependency graph."""
        # Process tiers in order: data → core → edge (reverse dependency direction)
        visited: set = set()
        # We propagate upstream: dependents of a degraded service inherit partial degradation
        for tier in range(3):  # 0=edge, 1=core, 2=data
            for node in self.topology:
                if node.tier != tier:
                    continue
                state = self._service_states[node.id]
                if state.circuit_breaker_open:
                    # CB isolates this service from its callers
                    continue
                for dependent_id in node.dependents:
                    dep_state = self._service_states[dependent_id]
                    if dep_state.circuit_breaker_open:
                        continue
                    # Check for network partition
                    partitioned = (node.id, dependent_id) in self._infra.network_partitions or \
                                  (dependent_id, node.id) in self._infra.network_partitions
                    if partitioned:
                        dep_state.error_rate = min(dep_state.error_rate + 0.3, 1.0)
                        continue

                    weight = _propagation_weight(node.id, dependent_id)
                    inherited_deg = state.degradation_factor * weight

                    if inherited_deg > dep_state.degradation_factor:
                        dep_state.degradation_factor = min(
                            dep_state.degradation_factor * 0.7 + inherited_deg * 0.3,
                            1.0,
                        )

                    dep_node = self._node_map[dependent_id]
                    if dep_state.degradation_factor > 0.2:
                        target_err = dep_node.slo_error_rate * (1.0 + dep_state.degradation_factor * 8.0)
                        dep_state.error_rate = min(
                            dep_state.error_rate * 0.8 + target_err * 0.2,
                            1.0,
                        )
                        target_lat = dep_node.slo_latency_p99 * (1.0 + dep_state.degradation_factor * 4.0)
                        dep_state.latency_p99 = min(
                            dep_state.latency_p99 * 0.8 + target_lat * 0.2,
                            dep_node.slo_latency_p99 * 15.0,
                        )

    def _apply_recovery_dynamics(self) -> None:
        """Services with no inherited degradation drift back toward nominal."""
        for node in self.topology:
            state = self._service_states[node.id]
            if node.id == self._incident.fault_origin:
                continue  # fault origin doesn't self-heal; agent must act

            state.degradation_factor = max(state.degradation_factor - 0.05, 0.0)

            # Slow natural recovery for non-origin services when their deps improve
            if state.degradation_factor < 0.05:
                state.error_rate = max(state.error_rate * 0.85, 0.0005)
                state.latency_p99 = max(state.latency_p99 * 0.95, node.slo_latency_p99 * 0.4)

    def _fire_alerts(self) -> None:
        """Create alerts for services breaching SLO thresholds; resolve cleared ones."""
        step = self._incident.step_count
        existing_keys: set = {
            (a.service_id, a.metric)
            for a in self._incident.active_alerts
            if not a.resolved
        }

        for node in self.topology:
            state = self._service_states[node.id]

            checks = [
                ("error_rate", state.error_rate, node.slo_error_rate),
                ("latency_p99", state.latency_p99, node.slo_latency_p99),
            ]
            for metric, val, slo_thresh in checks:
                key = (node.id, metric)
                ratio = val / max(slo_thresh, 1e-9)

                if ratio >= 2.0:
                    if key not in existing_keys:
                        severity = Severity.P1 if ratio >= 10.0 else (Severity.P2 if ratio >= 5.0 else Severity.P3)
                        self._incident.active_alerts.append(
                            Alert(
                                service_id=node.id,
                                metric=metric,
                                severity=severity,
                                threshold=slo_thresh * 2.0,
                                current_val=val,
                                fired_at=step,
                            )
                        )
                else:
                    # Resolve matching alert
                    for alert in self._incident.active_alerts:
                        if alert.service_id == node.id and alert.metric == metric and not alert.resolved:
                            alert.resolved = True

    def _update_consecutive_healthy(self) -> None:
        all_healthy = all(
            not self._service_states[node.id].is_slo_violated(node)
            for node in self.topology
        )
        if all_healthy:
            self._incident.consecutive_healthy_steps += 1
        else:
            self._incident.consecutive_healthy_steps = 0

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _validate_action(self, action: Action) -> Tuple[bool, str]:
        p = action.params
        states = self._service_states
        infra = self._infra

        def svc(key: str = "service_id") -> Optional[str]:
            return p.get(key)

        if action.type in (
            ActionType.SCALE_SERVICE,
            ActionType.ROLLBACK_SERVICE,
            ActionType.RESTART_SERVICE,
            ActionType.OPEN_CIRCUIT_BREAKER,
            ActionType.CLOSE_CIRCUIT_BREAKER,
            ActionType.ADJUST_RATE_LIMIT,
            ActionType.QUERY_LOGS,
            ActionType.QUERY_METRICS,
        ):
            sid = svc()
            if sid not in states:
                return False, f"Unknown service_id: {sid}"

        if action.type == ActionType.SCALE_SERVICE:
            delta = p.get("delta", 0)
            if delta not in (-3, -2, -1, 1, 2, 3):
                return False, "delta must be in {-3,-2,-1,1,2,3}"
            sid = svc()
            node = self._node_map[sid]
            new_count = states[sid].pod_count + delta
            if not (node.pod_min <= new_count <= node.pod_max):
                return False, f"pod_count {new_count} out of bounds [{node.pod_min}, {node.pod_max}]"
            total = sum(s.pod_count for s in states.values()) + delta
            if total > infra.total_node_capacity:
                return False, "Would exceed cluster capacity"

        elif action.type == ActionType.ROLLBACK_SERVICE:
            sid = svc()
            steps_back = p.get("steps_back", 1)
            if steps_back not in (1, 2, 3):
                return False, "steps_back must be 1, 2, or 3"
            history = states[sid].version_history
            idx = -steps_back
            if abs(idx) > len(history):
                return False, "Version history not deep enough"

        elif action.type == ActionType.OPEN_CIRCUIT_BREAKER:
            if states[svc()].circuit_breaker_open:
                return False, "Circuit breaker already open"

        elif action.type == ActionType.CLOSE_CIRCUIT_BREAKER:
            if not states[svc()].circuit_breaker_open:
                return False, "Circuit breaker is not open"

        elif action.type == ActionType.ADJUST_RATE_LIMIT:
            if p.get("multiplier") not in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                return False, "multiplier must be one of {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}"

        elif action.type == ActionType.REROUTE_TRAFFIC:
            from_svc = p.get("from_service")
            to_svc = p.get("to_service")
            pct = p.get("percentage")
            if from_svc not in states or to_svc not in states:
                return False, "from_service or to_service unknown"
            if pct not in (25, 50, 75, 100):
                return False, "percentage must be 25, 50, 75, or 100"
            from_node = self._node_map[from_svc]
            to_node = self._node_map[to_svc]
            if from_node.tier != to_node.tier:
                return False, "Can only reroute within the same tier"
            to_state = states[to_svc]
            if to_state.pod_count <= self._node_map[to_svc].pod_min:
                return False, "Destination has no spare capacity"

        elif action.type == ActionType.ACKNOWLEDGE_ALERT:
            alert_id = p.get("alert_id")
            if not any(a.id == alert_id for a in self._incident.active_alerts):
                return False, f"Unknown alert_id: {alert_id}"

        elif action.type == ActionType.PAGE_HUMAN:
            sev = p.get("severity")
            if sev not in (Severity.P1.value, Severity.P2.value, "P1", "P2"):
                return False, "severity must be P1 or P2"
            has_p1 = any(
                a.severity == Severity.P1 and not a.resolved
                for a in self._incident.active_alerts
            )
            if sev in ("P2", Severity.P2.value) and has_p1:
                return False, "Must page P1 severity when P1 alerts are active"

        return True, "ok"

    def _execute_action(self, action: Action) -> Optional[Dict[str, Any]]:
        """Apply the action to world state. Returns query result if applicable."""
        p = action.params
        states = self._service_states
        fault = self._incident.fault_type
        origin = self._incident.fault_origin

        if action.type == ActionType.SCALE_SERVICE:
            sid = p["service_id"]
            states[sid].pod_count += p["delta"]
            # Scaling up reduces CPU pressure and can help throughput
            if p["delta"] > 0:
                states[sid].cpu_utilization = max(
                    states[sid].cpu_utilization - p["delta"] * 0.06, 0.1
                )
                # If origin is being scaled and fault is TRAFFIC_SURGE, it helps
                if sid == origin and fault == FaultType.TRAFFIC_SURGE:
                    states[sid].degradation_factor = max(
                        states[sid].degradation_factor - p["delta"] * 0.08, 0.0
                    )
            else:
                states[sid].cpu_utilization = min(
                    states[sid].cpu_utilization - p["delta"] * 0.06, 1.0
                )

        elif action.type == ActionType.ROLLBACK_SERVICE:
            sid = p["service_id"]
            steps_back = p["steps_back"]
            history = states[sid].version_history
            target_version = history[-steps_back]
            states[sid].active_version = target_version
            # If this is a BAD_DEPLOY and we're rolling back the origin, fix it
            if sid == origin and fault == FaultType.BAD_DEPLOY:
                bad_ver = self._incident.fault_params.get("bad_version")
                if target_version != bad_ver:
                    states[sid].degradation_factor = 0.0
                    states[sid].error_rate = 0.001
                    node = self._node_map[sid]
                    states[sid].latency_p99 = node.slo_latency_p99 * 0.4

        elif action.type == ActionType.RESTART_SERVICE:
            sid = p["service_id"]
            # Restart temporarily spikes errors (2-step window tracked via degradation bump)
            # then clears crash state
            states[sid].is_crashing = False
            states[sid].degradation_factor = min(states[sid].degradation_factor + 0.3, 1.0)
            if fault == FaultType.MEMORY_LEAK and sid == origin:
                # Restart clears the leak but degradation hasn't fully resolved yet
                self._incident.fault_params["leak_rate"] *= 0.3
            # drain blast radius
            self._consume_blast_budget(0.20)

        elif action.type == ActionType.OPEN_CIRCUIT_BREAKER:
            sid = p["service_id"]
            states[sid].circuit_breaker_open = True
            self._consume_blast_budget(0.05)

        elif action.type == ActionType.CLOSE_CIRCUIT_BREAKER:
            states[p["service_id"]].circuit_breaker_open = False

        elif action.type == ActionType.ADJUST_RATE_LIMIT:
            sid = p["service_id"]
            mult = p["multiplier"]
            states[sid].rate_limit_multiplier = mult
            if mult < 1.0:
                states[sid].request_rate = self._node_map[sid].nominal_rps * mult
                states[sid].cpu_utilization = max(states[sid].cpu_utilization * mult, 0.1)
                if fault == FaultType.TRAFFIC_SURGE and sid == origin:
                    states[sid].degradation_factor = max(
                        states[sid].degradation_factor - (1.0 - mult) * 0.4, 0.0
                    )
                if mult <= 0.5:
                    self._consume_blast_budget(0.10)
            else:
                states[sid].request_rate = min(
                    self._node_map[sid].nominal_rps * mult,
                    self._node_map[sid].nominal_rps * 2.0,
                )

        elif action.type == ActionType.REROUTE_TRAFFIC:
            from_svc = p["from_service"]
            to_svc = p["to_service"]
            pct = p["percentage"] / 100.0
            rps_shift = states[from_svc].request_rate * pct
            states[from_svc].request_rate = max(states[from_svc].request_rate - rps_shift, 0.0)
            states[to_svc].request_rate += rps_shift
            cost = 0.25 if pct == 1.0 else 0.10
            self._consume_blast_budget(cost)

        elif action.type == ActionType.QUERY_LOGS:
            sid = p["service_id"]
            return self._generate_log_result(sid, p.get("window_minutes", 15))

        elif action.type == ActionType.QUERY_METRICS:
            sid = p["service_id"]
            metric = p.get("metric", "error_rate")
            return self._generate_metric_history(sid, metric)

        elif action.type == ActionType.ACKNOWLEDGE_ALERT:
            alert_id = p["alert_id"]
            self._incident.acknowledged_alerts.add(alert_id)
            for alert in self._incident.active_alerts:
                if alert.id == alert_id:
                    alert.resolved = False  # ack doesn't resolve, just notes it

        return None

    def _classify_outcome(self, action: Action, pre_health: float) -> ActionOutcome:
        post_health = self._aggregate_health()
        delta = post_health - pre_health
        if delta > 0.02:
            return ActionOutcome.APPLIED
        if delta < -0.05:
            return ActionOutcome.WORSENED
        if action.type in (ActionType.NO_OP, ActionType.ACKNOWLEDGE_ALERT):
            return ActionOutcome.NO_EFFECT
        return ActionOutcome.APPLIED

    def _consume_blast_budget(self, cost: float) -> None:
        self._incident.blast_radius_budget = max(self._incident.blast_radius_budget - cost, 0.0)

    # ------------------------------------------------------------------
    # Redundancy detection
    # ------------------------------------------------------------------

    def _is_redundant(self, action: Action, outcome: ActionOutcome) -> bool:
        key = self._action_fingerprint(action)
        recent = self._incident.recent_actions[-4:]
        bad_outcomes = {ActionOutcome.NO_EFFECT, ActionOutcome.WORSENED}
        for prev_key, prev_outcome in recent:
            if prev_key == key and prev_outcome in bad_outcomes:
                return True
        return False

    def _action_fingerprint(self, action: Action) -> str:
        blob = json.dumps({"type": action.type.value, "params": action.params}, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()[:8]

    def _update_recent_actions(self, action: Action, outcome: ActionOutcome) -> None:
        key = self._action_fingerprint(action)
        self._incident.recent_actions.append((key, outcome))
        if len(self._incident.recent_actions) > 8:
            self._incident.recent_actions.pop(0)

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_observation(self, query_result: Optional[Dict[str, Any]] = None) -> Observation:
        services = {}
        for node in self.topology:
            s = self._service_states[node.id]
            services[node.id] = ServiceObservation(
                error_rate=self._noisy(s.error_rate, 0.005),
                latency_p50=self._noisy(s.latency_p50, 2.0),
                latency_p99=self._noisy(s.latency_p99, 10.0),
                cpu_utilization=self._noisy(s.cpu_utilization, 0.02),
                memory_utilization=self._noisy(s.memory_utilization, 0.02),
                request_rate=self._noisy(s.request_rate, 5.0),
                pod_count=s.pod_count,
                circuit_breaker_open=s.circuit_breaker_open,
                rate_limit_multiplier=s.rate_limit_multiplier,
                active_version=s.active_version,
                is_slo_violated=s.is_slo_violated(node),
                slo_error_rate=node.slo_error_rate,
                slo_latency_p99=node.slo_latency_p99,
            )

        active_alerts = [
            AlertObservation(
                id=a.id,
                service_id=a.service_id,
                metric=a.metric,
                severity=a.severity.value,
                current_val=self._noisy(a.current_val, 0.002),
                age_steps=self._incident.step_count - a.fired_at,
                acknowledged=a.id in self._incident.acknowledged_alerts,
            )
            for a in self._incident.active_alerts
            if not a.resolved
        ]

        total_pods = sum(s.pod_count for s in self._service_states.values())
        infra_obs = InfraObservation(
            cluster_capacity_used_pct=total_pods / self._infra.total_node_capacity,
            db_conn_pool_used_pct=self._noisy(
                self._infra.db_conn_pool_used / self._infra.db_conn_pool_max, 0.02
            ),
        )

        return Observation(
            services=services,
            alerts=active_alerts,
            infra=infra_obs,
            change_log=self._change_log[-5:],
            step=self._incident.step_count,
            time_remaining=self.config.max_steps - self._incident.step_count,
            blast_radius_budget=self._incident.blast_radius_budget,
            query_result=query_result,
        )

    def _noisy(self, value: float, sigma: float) -> float:
        noisy = value + self._obs_rng.gauss(0, sigma)
        return max(noisy, 0.0)

    # ------------------------------------------------------------------
    # Diagnostic query generators
    # ------------------------------------------------------------------

    def _generate_log_result(self, service_id: str, window_minutes: int) -> Dict[str, Any]:
        state = self._service_states[service_id]
        fault = self._incident.fault_type
        origin = self._incident.fault_origin

        logs = []
        base_count = int(window_minutes * 2.5)

        # Logs hint at the fault if querying the origin (with some noise)
        if service_id == origin:
            if fault == FaultType.MEMORY_LEAK:
                logs.append(f"[WARN] heap usage at {state.memory_utilization*100:.1f}% — GC pressure increasing")
                if state.memory_utilization > 0.8:
                    logs.append("[ERROR] OOMKilled — container restarted")
            elif fault == FaultType.BAD_DEPLOY:
                bad_ver = self._incident.fault_params.get("bad_version", "?")
                logs.append(f"[ERROR] v{bad_ver}: NullPointerException in request handler (x{self._obs_rng.randint(10,80)})")
                logs.append(f"[INFO] Deployment v{bad_ver} rolled out {self._incident.step_count} steps ago")
            elif fault == FaultType.TRAFFIC_SURGE:
                logs.append(f"[WARN] request queue depth: {int(state.request_rate * 0.4)} — throughput limited")
                logs.append(f"[WARN] CPU throttling active — {state.cpu_utilization*100:.0f}% utilization")
            elif fault == FaultType.DB_CONN_EXHAUSTION:
                pool_pct = self._infra.db_conn_pool_used / self._infra.db_conn_pool_max * 100
                logs.append(f"[ERROR] connection pool exhausted ({pool_pct:.0f}%) — requests timing out")
                logs.append("[WARN] connection wait time > 5000ms")
            elif fault == FaultType.DEPENDENCY_LATENCY:
                logs.append("[WARN] downstream call timeout threshold exceeded (x multiple)")
                logs.append("[ERROR] circuit half-open: downstream latency spike detected")
            elif fault == FaultType.NETWORK_PARTITION:
                partner = self._incident.fault_params.get("partner", "unknown")
                logs.append(f"[ERROR] connection refused: {partner} unreachable — network timeout")
                logs.append("[ERROR] all retries exhausted for downstream call")
        else:
            # Non-origin services show symptom logs
            if state.error_rate > 0.05:
                logs.append(f"[ERROR] upstream dependency returning 5xx ({state.error_rate*100:.1f}% of requests)")
            if state.latency_p99 > self._node_map[service_id].slo_latency_p99:
                logs.append(f"[WARN] p99 latency {state.latency_p99:.0f}ms exceeds SLO")

        # Pad with benign noise
        while len(logs) < min(base_count, 6):
            logs.append(f"[INFO] health check ok — {self._obs_rng.randint(100, 999)} req/s")

        return {
            "service_id": service_id,
            "window_minutes": window_minutes,
            "log_count": len(logs) + self._obs_rng.randint(0, 20),
            "samples": logs,
        }

    def _generate_metric_history(self, service_id: str, metric: str) -> Dict[str, Any]:
        state = self._service_states[service_id]
        current = getattr(state, metric, None)
        if current is None:
            return {"error": f"Unknown metric: {metric}"}

        # Reconstruct plausible history: stable → fault onset → current
        history = []
        steps = min(self._incident.step_count + 1, 10)
        node = self._node_map[service_id]
        origin = self._incident.fault_origin

        for i in range(steps):
            t = i / max(steps - 1, 1)
            if service_id == origin:
                # Interpolate from nominal toward current
                if metric == "error_rate":
                    nominal = 0.001
                elif metric == "latency_p99":
                    nominal = node.slo_latency_p99 * 0.4
                else:
                    nominal = current * 0.5
                val = nominal + (current - nominal) * t
            else:
                val = current * (0.3 + 0.7 * t)
            history.append(round(self._noisy(val, abs(current) * 0.05 + 0.001), 4))

        return {
            "service_id": service_id,
            "metric": metric,
            "history": history,
            "current": current,
            "unit": "ms" if "latency" in metric else ("fraction" if "rate" in metric or "utilization" in metric else ""),
        }

    # ------------------------------------------------------------------
    # Aggregate health
    # ------------------------------------------------------------------

    def _aggregate_health(self) -> float:
        scores = [
            self._service_states[node.id].health_score(node)
            for node in self.topology
        ]
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Change log
    # ------------------------------------------------------------------

    def _append_change_log(self, action: Action, outcome: ActionOutcome, health_delta: float) -> None:
        self._change_log.append(
            ChangeLogEntry(
                step=self._incident.step_count,
                action_type=action.type.value,
                action_params=action.params,
                outcome=outcome,
                health_delta=round(health_delta, 4),
            )
        )
        if len(self._change_log) > 50:
            self._change_log.pop(0)