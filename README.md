---
title: ProductionIncidentEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - sre
  - agent-evaluation
short_description: OpenEnv environment for SRE incident response agent training
---

# ProductionIncidentEnv

**An OpenEnv-compliant reinforcement learning environment for production SRE incident response.**

Agents act as on-call Site Reliability Engineers managing a simulated 8-service microservices stack. A hidden fault has been injected into the system; the agent must diagnose the root cause, apply targeted remediations, and restore SLO compliance before cascading failures trigger a full outage.

---

## Why this domain?

Modern LLM agents are increasingly deployed in agentic SRE workflows — reading dashboards, proposing remediations, and executing runbooks. Yet no standardized evaluation environment exists for this capability.

`ProductionIncidentEnv` fills that gap. It tests exactly what matters for production-grade agents:

- **Multi-hop causal diagnosis** — the fault origin isn't labeled; the agent must infer it from noisy metrics and partial log signals
- **Consequential actions** — wrong remediations actively worsen the system and drain a blast-radius budget
- **Temporal reasoning** — faults evolve step-by-step; a restart before stabilization re-triggers the leak
- **Structured restraint** — the highest-scoring agents are *surgical*, not aggressive

The environment is designed to be unsolvable by random or reactive agents, and meaningfully differentiate between LLMs at different capability levels.

---

## Environment Overview

### Service Topology

A 3-tier directed acyclic graph modeling a simplified e-commerce backend:

```
Tier 0 (edge):    api-gateway
                  ↓         ↓           ↓
Tier 1 (core):  auth-svc  product-svc  cart-svc
                                          ↓
                                      order-svc
                                      ↓        ↓
Tier 2 (data): payment-svc   inventory-svc   db-proxy
```

All services have nominal SLO thresholds for error rate and p99 latency. Faults propagate upstream through the dependency graph with configurable weights.

### Fault Classes

| Fault | Origin | Cascade Pattern |
|---|---|---|
| `MEMORY_LEAK` | Any service | Degradation climbs over ~20 steps until OOM crash |
| `BAD_DEPLOY` | Any service | Immediate error spike; version pinpointed in observations |
| `TRAFFIC_SURGE` | Edge tier | CPU saturation → latency → downstream error cascade |
| `DB_CONN_EXHAUSTION` | Data tier | Connection pool fills; all callers time out |
| `DEPENDENCY_LATENCY` | Any service | p99 climbs; callers start timing out |
| `NETWORK_PARTITION` | Any pair | 100% errors on calls crossing the partition |

---

## Action Space

All actions are parameterized discrete. Invalid actions are penalized but not blocked.

| Action | Parameters | Blast Cost |
|---|---|---|
| `SCALE_SERVICE` | `service_id`, `delta ∈ {-3,-2,-1,1,2,3}` | — |
| `ROLLBACK_SERVICE` | `service_id`, `steps_back ∈ {1,2,3}` | 0.15 |
| `RESTART_SERVICE` | `service_id` | 0.20 |
| `OPEN_CIRCUIT_BREAKER` | `service_id` | 0.05 |
| `CLOSE_CIRCUIT_BREAKER` | `service_id` | — |
| `ADJUST_RATE_LIMIT` | `service_id`, `multiplier ∈ {0.25,0.5,0.75,1.0,1.5,2.0}` | 0.10 (if ≤ 0.5) |
| `REROUTE_TRAFFIC` | `from_service`, `to_service`, `percentage ∈ {25,50,75,100}` | 0.10–0.25 |
| `QUERY_LOGS` | `service_id`, `window_minutes ∈ {5,15,30}` | — |
| `QUERY_METRICS` | `service_id`, `metric` | — |
| `ACKNOWLEDGE_ALERT` | `alert_id` | — |
| `PAGE_HUMAN` | `severity ∈ {P1,P2}` | — (terminates) |
| `NO_OP` | — | — |

**Blast radius budget** starts at 1.0 and is consumed by high-risk actions. Reaching 0.0 triggers an immediate `CATASTROPHIC` terminal with a −50 penalty.

---

## Observation Space

At each step, the agent receives:

```json
{
  "services": {
    "api-gateway": {
      "error_rate": 0.0023,
      "latency_p50": 52.1,
      "latency_p99": 88.4,
      "cpu_utilization": 0.38,
      "memory_utilization": 0.44,
      "request_rate": 498.3,
      "pod_count": 6,
      "circuit_breaker_open": false,
      "rate_limit_multiplier": 1.0,
      "active_version": 3,
      "is_slo_violated": false,
      "slo_error_rate": 0.005,
      "slo_latency_p99": 150.0
    },
    "...": "..."
  },
  "alerts": [
    {
      "id": "a3f2b1",
      "service_id": "cart-service",
      "metric": "error_rate",
      "severity": "P2",
      "current_val": 0.082,
      "age_steps": 3,
      "acknowledged": false
    }
  ],
  "infra": {
    "cluster_capacity_used_pct": 0.27,
    "db_conn_pool_used_pct": 0.21
  },
  "change_log": [...],
  "step": 4,
  "time_remaining": 56,
  "blast_radius_budget": 1.0,
  "query_result": null
}
```

Metric values are noisy (Gaussian σ proportional to measurement type). The agent never directly observes `degradation_factor`, `is_crashing`, or the fault type/origin.

---

## Reward Design

The reward is dense and shaped across the full trajectory.

| Component | Description | Value |
|---|---|---|
| `slo_health` | Average health score across all services, every step | `[0, 2.0]` |
| `resolution_bonus` | Success bonus, scales with speed | `[50, 75]` |
| `invalid_action` | Penalty for structurally invalid actions | `−2.0` |
| `redundant_action` | Penalty for repeating ineffective actions | `−1.5` |
| `worsening` | Proportional penalty when action degrades health > 5% | `−3.0 × drop` |
| `blast_radius` | Progressive per-step penalty when budget < 0.3 | `−5.0 × (1 − budget)` |
| `escalation` | Credit/penalty for human escalation decision | `−10 to +20` |
| `diagnostic_novelty` | One-time reward for first query on each service/metric | `+0.3` |

**Typical score range:** `−200` to `+100`. A well-calibrated agent resolves incidents in 15–60 steps and scores `+60 to +85`.

---

## Tasks

### Task 1 — Easy: Isolated Bad Deployment

```
Fault:   BAD_DEPLOY on cart-service (version 4, ~15% error rate)
Seed:    42
Steps:   max 60
```

cart-service received a bad artifact. Its error rate has spiked and version 4 is visible in the observation. No cascade has started yet. A single `ROLLBACK_SERVICE` with `steps_back=1` resolves the incident.

**Grader:** resolution achieved (0.40) + correct rollback target (0.30) + within 20 steps (0.20) + budget preserved (0.10)

---

### Task 2 — Medium: Traffic Surge Cascade

```
Fault:   TRAFFIC_SURGE on api-gateway (4× nominal RPS)
Seed:    137
Steps:   max 100
```

Black Friday-style traffic surge hitting the edge. CPU is saturating, and the latency cascade is spreading downstream. The agent must identify that the issue is throughput, not a code bug, and apply scaling or rate limiting to the *origin* rather than chasing symptoms.

**Grader:** resolution (0.35) + correct origin action (0.25) + surgical focus (0.15) + speed (0.15) + budget (0.10)

---

### Task 3 — Hard: DB Connection Exhaustion + Memory Cascade

```
Fault:   DB_CONN_EXHAUSTION on db-proxy (~5 conns/step leak)
         + induced memory leak in order-service (compound)
Seed:    314
Steps:   max 150
```

The database connection pool is filling up. order-service, which calls db-proxy heavily, has accumulated leaked connection handles in memory. Restarting order-service before stabilizing db-proxy will cause it to immediately re-leak. The agent must sequence:

1. Stabilize db-proxy (rate limit or circuit-break callers)
2. Wait for pool pressure < 60%
3. Restart order-service to clear its memory state

**Grader:** resolution (0.30) + db addressed first (0.20) + restart timing (0.15) + diagnostic quality (0.15) + speed (0.10) + budget (0.10)

---

## Baseline Results

Reference scores from the heuristic baseline agent (deterministic, no LLM):

| Task | Score | Terminal | Steps |
|---|---|---|---|
| task_easy_bad_deploy | ~0.80 | SUCCESS | ~12 |
| task_medium_traffic_surge | ~0.65 | SUCCESS | ~35 |
| task_hard_db_cascade | ~0.45 | SUCCESS/TIMEOUT | ~80 |
| **Average** | **~0.63** | | |

An LLM agent using GPT-4o typically scores 0.70–0.85 on easy, 0.55–0.75 on medium, and 0.35–0.60 on hard.

---

## Setup

### Local (Python 3.11+)

```bash
git clone <repo>
cd production-incident-env
pip install -r requirements.txt

# Run the API server
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload

# Run the heuristic baseline
PYTHONPATH=. python scripts/baseline.py

# Run the LLM baseline (requires OpenAI key)
OPENAI_API_KEY=sk-... PYTHONPATH=. python scripts/baseline.py --agent llm

# Run a specific task with verbose logging
PYTHONPATH=. python scripts/baseline.py --task task_easy_bad_deploy --verbose
```

### Docker

```bash
# Build
docker build -t production-incident-env .

# Run (API server)
docker run -p 7860:7860 production-incident-env

# Run with LLM baseline support
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... production-incident-env

# Run baseline script in container
docker run -e OPENAI_API_KEY=sk-... production-incident-env \
  python scripts/baseline.py --agent heuristic
```

Server will be available at `http://localhost:7860`.

---

## API Examples

### Start an episode on a specific task

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy_bad_deploy"}'
```

### Take an action

```bash
# Query logs first (good diagnostic practice)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "QUERY_LOGS", "params": {"service_id": "cart-service", "window_minutes": 15}}'

# Roll back cart-service
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "ROLLBACK_SERVICE", "params": {"service_id": "cart-service", "steps_back": 1}}'
```

### Get internal state (ground truth)

```bash
curl http://localhost:7860/state
```

### List tasks and action schema

```bash
curl http://localhost:7860/tasks
```

### Grade the episode

```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy_bad_deploy"}'
```

### Run the built-in baseline

```bash
curl -X POST http://localhost:7860/baseline \
  -H "Content-Type: application/json" \
  -d '{"task_ids": ["task_easy_bad_deploy", "task_medium_traffic_surge"]}'
```

---

## Repository Structure

```
production-incident-env/
├── env/
│   ├── __init__.py
│   ├── environment.py     # Core environment: step(), reset(), state()
│   ├── models.py          # Pydantic models: Observation, Action, Reward, ...
│   ├── tasks.py           # Task definitions (easy / medium / hard)
│   └── graders.py         # Deterministic graders, 0.0–1.0 scores
├── api/
│   ├── __init__.py
│   └── main.py            # FastAPI app
├── scripts/
│   └── baseline.py        # Heuristic + LLM baseline agents, CLI runner
├── openenv.yaml           # OpenEnv spec compliance manifest
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Design Notes

**Why noisy observations?** Real monitoring systems have measurement lag and noise. A perfect-information oracle environment would be too easy and unrealistic — agents should learn to be robust to sensor noise.

**Why blast radius budget?** In production, an agent that restarts 6 services "just in case" is dangerous. The budget forces surgical thinking and makes the cost of uncertainty concrete.

**Why compound faults on Hard?** Most real incidents involve interaction effects, not a single root cause. Task 3 specifically tests whether an agent can discover that fixing symptom A before root cause B will cause immediate regression — a form of temporal causal reasoning that simple reactive policies can't handle.

**Fault propagation model.** Degradation flows through the DAG with per-edge weights. Circuit breakers interrupt propagation. This means the correct response to a cascading fault depends on topology knowledge that the agent must infer, not observe directly.
