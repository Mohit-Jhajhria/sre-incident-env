from __future__ import annotations

"""
Deterministic graders for each task.

Graders operate on the episode trajectory — the sequence of (action, step_result)
pairs collected during a run — plus the final internal state snapshot returned
by env.state(). They produce a score in [0.0, 1.0] with a breakdown dict.

Each grader is a pure function: same inputs → same score, no randomness.
"""

from typing import Any, Dict, List, Optional, Tuple

from env.models import ActionType, StepResult
from env.tasks import TASK_EASY, TASK_HARD, TASK_MEDIUM, TaskSpec


# ---------------------------------------------------------------------------
# Trajectory record
# ---------------------------------------------------------------------------


class StepRecord:
    """A single step in an episode trajectory."""

    __slots__ = ("step", "action_type", "action_params", "result", "state_snapshot")

    def __init__(
        self,
        step: int,
        action_type: str,
        action_params: Dict[str, Any],
        result: StepResult,
        state_snapshot: Optional[Dict[str, Any]] = None,
    ):
        self.step = step
        self.action_type = action_type
        self.action_params = action_params
        self.result = result
        self.state_snapshot = state_snapshot


EpisodeTrajectory = List[StepRecord]


# ---------------------------------------------------------------------------
# Grade result
# ---------------------------------------------------------------------------


class GradeResult:
    def __init__(self, score: float, components: Dict[str, float], notes: List[str]):
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"
        self.score = round(score, 4)
        self.components = {k: round(v, 4) for k, v in components.items()}
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "components": self.components,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Shared grading utilities
# ---------------------------------------------------------------------------


def _resolved(trajectory: EpisodeTrajectory) -> Tuple[bool, Optional[int]]:
    """Returns (resolved, resolution_step) based on the terminal step."""
    for record in trajectory:
        info = record.result.info
        if info.get("terminal") == "SUCCESS":
            return True, record.step
    return False, None


def _terminal_reason(trajectory: EpisodeTrajectory) -> Optional[str]:
    for record in reversed(trajectory):
        t = record.result.info.get("terminal")
        if t:
            return t
    return None


def _actions_of_type(trajectory: EpisodeTrajectory, action_type: str) -> List[StepRecord]:
    return [r for r in trajectory if r.action_type == action_type]


def _actions_on_service(trajectory: EpisodeTrajectory, service_id: str) -> List[StepRecord]:
    return [
        r for r in trajectory
        if r.action_params.get("service_id") == service_id
        or r.action_params.get("from_service") == service_id
    ]


def _invalid_action_count(trajectory: EpisodeTrajectory) -> int:
    return sum(1 for r in trajectory if not r.result.info.get("action_valid", True))


def _final_blast_budget(trajectory: EpisodeTrajectory) -> float:
    if not trajectory:
        return 1.0
    return trajectory[-1].result.observation.blast_radius_budget


def _diagnostic_queries(trajectory: EpisodeTrajectory) -> List[StepRecord]:
    return [
        r for r in trajectory
        if r.action_type in (ActionType.QUERY_LOGS.value, ActionType.QUERY_METRICS.value)
    ]


# ---------------------------------------------------------------------------
# Task 1 Grader — Easy: Isolated Bad Deploy
# ---------------------------------------------------------------------------


def grade_easy(trajectory: EpisodeTrajectory, final_state: Dict[str, Any]) -> GradeResult:
    """
    Score breakdown (max 1.0):
      resolution_achieved  : 0.40 — did the episode end in SUCCESS?
      correct_target       : 0.30 — was cart-service the target of a ROLLBACK?
      speed_bonus          : 0.20 — resolved within 20 steps
      blast_budget_bonus   : 0.10 — blast_radius_budget > 0.7 at resolution
    """
    gp = TASK_EASY.grader_params
    components: Dict[str, float] = {}
    notes: List[str] = []

    resolved, resolution_step = _resolved(trajectory)

    # --- Resolution ---
    if resolved:
        components["resolution_achieved"] = 0.40
        notes.append(f"Episode resolved successfully at step {resolution_step}.")
    else:
        components["resolution_achieved"] = 0.0
        reason = _terminal_reason(trajectory)
        notes.append(f"Episode did not resolve (terminal: {reason}).")

    # --- Correct target ---
    rollbacks = _actions_of_type(trajectory, ActionType.ROLLBACK_SERVICE.value)
    target_rollbacks = [
        r for r in rollbacks
        if r.action_params.get("service_id") == gp["target_service"]
    ]
    if target_rollbacks:
        components["correct_target"] = 0.30
        notes.append(f"Agent correctly rolled back {gp['target_service']}.")
    else:
        components["correct_target"] = 0.0
        if rollbacks:
            wrong_targets = list({r.action_params.get("service_id") for r in rollbacks})
            notes.append(f"Agent issued rollbacks, but not on {gp['target_service']} (targets: {wrong_targets}).")
        else:
            notes.append("Agent never issued a ROLLBACK_SERVICE action.")

    # --- Speed bonus ---
    speed_threshold = gp["resolution_step_threshold"]
    if resolved and resolution_step is not None and resolution_step <= speed_threshold:
        components["speed_bonus"] = 0.20
        notes.append(f"Resolved in {resolution_step} steps (threshold: {speed_threshold}).")
    else:
        components["speed_bonus"] = 0.0
        if resolved:
            notes.append(f"Resolved, but too slowly (step {resolution_step} > {speed_threshold}).")

    # --- Blast radius ---
    blast_budget = _final_blast_budget(trajectory)
    threshold = gp["blast_budget_threshold"]
    if blast_budget >= threshold:
        components["blast_budget_bonus"] = 0.10
        notes.append(f"Blast radius budget preserved: {blast_budget:.2f} >= {threshold}.")
    else:
        components["blast_budget_bonus"] = 0.0
        notes.append(f"Blast radius budget depleted: {blast_budget:.2f} < {threshold}.")

    invalid = _invalid_action_count(trajectory)
    if invalid > 0:
        notes.append(f"Agent issued {invalid} invalid action(s).")

    score = sum(components.values())
    return GradeResult(min(score, 1.0), components, notes)


# ---------------------------------------------------------------------------
# Task 2 Grader — Medium: Traffic Surge Cascade
# ---------------------------------------------------------------------------


def grade_medium(trajectory: EpisodeTrajectory, final_state: Dict[str, Any]) -> GradeResult:
    """
    Score breakdown (max 1.0):
      resolution_achieved  : 0.35 — episode ended in SUCCESS
      correct_origin_action: 0.25 — SCALE or RATE_LIMIT applied to api-gateway
      surgical_focus       : 0.15 — few actions on non-origin services
      speed_bonus          : 0.15 — resolved within 40 steps
      blast_budget_bonus   : 0.10 — budget > 0.6 at end
    """
    gp = TASK_MEDIUM.grader_params
    components: Dict[str, float] = {}
    notes: List[str] = []

    resolved, resolution_step = _resolved(trajectory)

    # --- Resolution ---
    if resolved:
        components["resolution_achieved"] = 0.35
        notes.append(f"Episode resolved at step {resolution_step}.")
    else:
        components["resolution_achieved"] = 0.0
        notes.append(f"Not resolved (terminal: {_terminal_reason(trajectory)}).")

    # --- Correct origin action ---
    target = gp["target_service"]
    correct_types = set(gp["correct_action_types"])
    origin_actions = [
        r for r in trajectory
        if r.action_type in correct_types
        and r.action_params.get("service_id") == target
    ]
    if origin_actions:
        components["correct_origin_action"] = 0.25
        notes.append(
            f"Agent applied {origin_actions[0].action_type} to {target} at step {origin_actions[0].step}."
        )
    else:
        components["correct_origin_action"] = 0.0
        notes.append(f"Agent never applied a scaling or rate-limit action to {target}.")

    # --- Surgical focus: penalize excess non-origin actions ---
    non_origin_services = set(gp["non_origin_services"])
    max_allowed = gp["max_non_origin_actions"]
    consequential_non_origin = [
        r for r in trajectory
        if r.action_params.get("service_id") in non_origin_services
        and r.action_type not in (
            ActionType.QUERY_LOGS.value,
            ActionType.QUERY_METRICS.value,
            ActionType.ACKNOWLEDGE_ALERT.value,
            ActionType.NO_OP.value,
        )
    ]
    excess = max(0, len(consequential_non_origin) - max_allowed)
    if excess == 0:
        components["surgical_focus"] = 0.15
        notes.append(f"Agent stayed focused: {len(consequential_non_origin)} non-origin consequential actions.")
    else:
        deduction = min(excess * 0.05, 0.15)
        components["surgical_focus"] = round(max(0.15 - deduction, 0.0), 4)
        notes.append(
            f"Agent issued {len(consequential_non_origin)} non-origin consequential actions "
            f"({excess} over limit of {max_allowed})."
        )

    # --- Speed ---
    speed_threshold = gp["resolution_step_threshold"]
    if resolved and resolution_step is not None and resolution_step <= speed_threshold:
        components["speed_bonus"] = 0.15
        notes.append(f"Resolved within {speed_threshold} steps (actual: {resolution_step}).")
    else:
        components["speed_bonus"] = 0.0
        if resolved:
            notes.append(f"Resolved, but at step {resolution_step} (threshold {speed_threshold}).")

    # --- Blast budget ---
    blast_budget = _final_blast_budget(trajectory)
    threshold = gp["blast_budget_threshold"]
    if blast_budget >= threshold:
        components["blast_budget_bonus"] = 0.10
        notes.append(f"Blast radius budget: {blast_budget:.2f}.")
    else:
        components["blast_budget_bonus"] = 0.0
        notes.append(f"Blast radius budget depleted: {blast_budget:.2f} < {threshold}.")

    score = sum(components.values())
    return GradeResult(min(score, 1.0), components, notes)


# ---------------------------------------------------------------------------
# Task 3 Grader — Hard: DB Connection Exhaustion + Memory Cascade
# ---------------------------------------------------------------------------


def grade_hard(trajectory: EpisodeTrajectory, final_state: Dict[str, Any]) -> GradeResult:
    """
    Score breakdown (max 1.0):
      resolution_achieved       : 0.30 — SUCCESS terminal
      db_addressed_first        : 0.20 — a db-proxy action precedes order-service restart
      correct_restart_timing    : 0.15 — order-service restarted after pool pressure < 0.6
      diagnostic_quality        : 0.15 — queried relevant metrics before acting
      speed_bonus               : 0.10 — resolved within 70 steps
      blast_budget_bonus        : 0.10 — budget > 0.4 at resolution
    """
    gp = TASK_HARD.grader_params
    components: Dict[str, float] = {}
    notes: List[str] = []

    resolved, resolution_step = _resolved(trajectory)

    # --- Resolution ---
    if resolved:
        components["resolution_achieved"] = 0.30
        notes.append(f"Episode resolved at step {resolution_step}.")
    else:
        components["resolution_achieved"] = 0.0
        notes.append(f"Not resolved (terminal: {_terminal_reason(trajectory)}).")

    # --- DB addressed before order-service restart ---
    primary = gp["primary_service"]      # db-proxy
    secondary = gp["secondary_service"]  # order-service
    db_correct_types = set(gp["correct_db_actions"])

    db_actions = [
        r for r in trajectory
        if r.action_params.get("service_id") == primary
        and r.action_type in db_correct_types
    ]
    restart_actions = [
        r for r in trajectory
        if r.action_type == gp["required_secondary_action"]
        and r.action_params.get("service_id") == secondary
    ]

    db_first = False
    if db_actions and restart_actions:
        first_db_step = min(r.step for r in db_actions)
        first_restart_step = min(r.step for r in restart_actions)
        if first_db_step < first_restart_step:
            db_first = True
            components["db_addressed_first"] = 0.20
            notes.append(
                f"Agent correctly addressed db-proxy at step {first_db_step} "
                f"before restarting order-service at step {first_restart_step}."
            )
        else:
            components["db_addressed_first"] = 0.0
            notes.append(
                f"Agent restarted order-service (step {first_restart_step}) "
                f"before addressing db-proxy (step {first_db_step}) — incorrect sequence."
            )
    elif not db_actions:
        components["db_addressed_first"] = 0.0
        notes.append("Agent never applied a remediation action to db-proxy.")
    elif not restart_actions:
        components["db_addressed_first"] = 0.0
        notes.append("Agent never restarted order-service.")

    # --- Restart timing: pool pressure must be low at restart ---
    pool_threshold = gp["pool_pressure_threshold"]
    restart_timing_ok = False
    if restart_actions:
        # Find the state snapshot at the first restart step to check pool pressure
        first_restart_step = min(r.step for r in restart_actions)
        for record in trajectory:
            if record.step == first_restart_step and record.state_snapshot:
                infra = record.state_snapshot.get("infra", {})
                pool_used = infra.get("db_conn_pool_used", 100)
                pool_max = infra.get("db_conn_pool_max", 100)
                actual_pressure = pool_used / max(pool_max, 1)
                if actual_pressure < pool_threshold:
                    restart_timing_ok = True
                    components["correct_restart_timing"] = 0.15
                    notes.append(
                        f"order-service restarted when pool pressure was {actual_pressure:.2%} "
                        f"(threshold: {pool_threshold:.0%}) — good timing."
                    )
                else:
                    components["correct_restart_timing"] = 0.0
                    notes.append(
                        f"order-service restarted too early: pool pressure was {actual_pressure:.2%} "
                        f"(threshold: {pool_threshold:.0%})."
                    )
                break
        else:
            # No state snapshot available; award partial credit if sequence was correct
            if db_first:
                components["correct_restart_timing"] = 0.08
                notes.append("No state snapshot for timing validation; awarded partial credit for correct sequence.")
            else:
                components["correct_restart_timing"] = 0.0
    else:
        components["correct_restart_timing"] = 0.0
        notes.append("No order-service restart found; timing not evaluable.")

    # --- Diagnostic quality ---
    queries = _diagnostic_queries(trajectory)
    bonus_metrics = set(gp["diagnostic_bonus_queries"])
    queried_metrics = {r.action_params.get("metric", "logs") for r in queries}
    relevant_hits = len(queried_metrics & bonus_metrics)
    # Also reward querying db-proxy or order-service logs
    log_targets = {r.action_params.get("service_id") for r in queries}
    relevant_services_queried = bool(
        {primary, secondary} & log_targets
    )

    diag_score = 0.0
    if relevant_hits >= 2:
        diag_score = 0.15
        notes.append(f"Strong diagnostic use: queried {relevant_hits} relevant metrics.")
    elif relevant_hits == 1 or relevant_services_queried:
        diag_score = 0.08
        notes.append(f"Partial diagnostic: queried {relevant_hits} relevant metric(s), services: {log_targets}.")
    else:
        notes.append("No targeted diagnostic queries made.")
    components["diagnostic_quality"] = diag_score

    # --- Speed ---
    speed_threshold = gp["resolution_step_threshold"]
    if resolved and resolution_step is not None and resolution_step <= speed_threshold:
        components["speed_bonus"] = 0.10
        notes.append(f"Resolved within {speed_threshold} steps (actual: {resolution_step}).")
    else:
        components["speed_bonus"] = 0.0
        if resolved:
            notes.append(f"Resolved but slowly: step {resolution_step} > {speed_threshold}.")

    # --- Blast budget ---
    blast_budget = _final_blast_budget(trajectory)
    threshold = gp["blast_budget_threshold"]
    if blast_budget >= threshold:
        components["blast_budget_bonus"] = 0.10
        notes.append(f"Blast radius budget: {blast_budget:.2f} >= {threshold}.")
    else:
        components["blast_budget_bonus"] = 0.0
        notes.append(f"Blast radius budget depleted to {blast_budget:.2f} (threshold {threshold}).")

    score = sum(components.values())
    return GradeResult(min(score, 1.0), components, notes)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


GRADER_REGISTRY = {
    TASK_EASY.id: grade_easy,
    TASK_MEDIUM.id: grade_medium,
    TASK_HARD.id: grade_hard,
}


def grade_episode(
    task_id: str,
    trajectory: EpisodeTrajectory,
    final_state: Dict[str, Any],
) -> GradeResult:
    if task_id not in GRADER_REGISTRY:
        raise KeyError(f"No grader registered for task '{task_id}'")
    return GRADER_REGISTRY[task_id](trajectory, final_state)
