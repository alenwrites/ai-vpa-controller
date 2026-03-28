# ai-vpa-controller

![Status: WIP](https://img.shields.io/badge/status-work%20in%20progress-orange)
![Kubernetes](https://img.shields.io/badge/kubernetes-%3E%3D%201.35-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![KEP-1287](https://img.shields.io/badge/KEP--1287-GA%20%28v1.35%29-green)
![GSoC 2026](https://img.shields.io/badge/GSoC-2026%20Applicant-red)

> **⚠️ Work in Progress — Prototype Stage**
> This project is under active development. It is not yet tested against a live cluster. Contributions, feedback, and issue reports are welcome.

---

## Intelligent In-Place Vertical Pod Autoscaler with GRU-Based Predictive Scaling

A Kubernetes controller that combines the **KEP-1287 In-Place Pod Resize API** (GA in Kubernetes v1.35) with a **Gated Recurrent Unit (GRU) neural network** to proactively scale stateful workloads — without restarting Pods.

Standard Kubernetes VPA requires a Pod eviction for every resource change. For stateful workloads like Redis, PostgreSQL, and Kafka, this means downtime, connection drops, and cache invalidation. This controller eliminates that trade-off.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Installation & Quick Start](#installation--quick-start)
- [Configuration Reference](#configuration-reference)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Author](#author)

---

## Why This Exists

The Kubernetes ecosystem has three unsolved problems with vertical scaling:

**1. Restarts are destructive for stateful workloads.**
Every standard VPA recommendation evicts the Pod. A 10-second Redis eviction causes client timeouts, cache cold-starts, and potential data loss for unflushed AOF/RDB writes.

**2. KEP-1287 provides the mechanism, not the intelligence.**
Kubernetes 1.35 made in-place Pod resizing stable. The native VPA gained `updateMode: InPlaceOrRecreate`. But it still uses threshold-based recommendations with no temporal awareness. The kubelet's built-in feasibility check is a basic RSS comparison — it has no knowledge of Redis fork behaviour, application load cycles, or downscale safety margins.

**3. Autoscalers are reactive, not predictive.**
Existing autoscalers respond to spikes that have already degraded performance. Workloads with predictable patterns — nightly batch jobs, morning traffic surges — can be scaled *before* the spike arrives.

This controller addresses all three simultaneously.

---

## How It Works

The controller runs a continuous reconciliation loop with five stages:

```
Prometheus (20 features)
        │
        ▼
   GRU Model (ml_engine.py)
   60-second sliding window
   → predicted CPU + MEM
        │
        ▼
   Policy Engine (policy_engine.py)
   cooldown · significance · maintenance windows
        │
        ▼
   Safety Firewall (safety_layer.py)
   Redis fork-safety · downscale dampening · bounds
        │
        ▼
   Kubernetes Adapter (k8s_adapter.py)
   ┌─────────────────────────────────────┐
   │ Action A: Patch live Pod directly   │  ← KEP-1287, zero restart
   │ Action B: Patch workload spec       │  ← persistence on restart
   └─────────────────────────────────────┘
```

### The Dual-Patch Strategy

Most controllers patch only the workload spec (Deployment/StatefulSet), which triggers a rollout. This controller does two things simultaneously:

- **Action A — Live Pod Patch:** Directly patches `spec.containers[].resources` on running Pods. Under KEP-1287, the kubelet resizes the container's cgroup without sending SIGTERM. The application process keeps running.
- **Action B — Workload Spec Patch:** Updates the Deployment/StatefulSet template so that if the Pod ever restarts for any other reason, it starts with the correct values.

### The Safety Pipeline

Before any patch reaches the cluster, the prediction passes through three layers:

| Layer | Component | What It Enforces |
|---|---|---|
| 1 | `policy_engine.py` | Cooldown (600s default), minimum change threshold, maintenance window freeze |
| 2 | `safety_layer.py` | Redis fork-safety floor (`RSS × 1.5`), max 50% downscale per cycle |
| 3 | `config.py` | Hard infrastructure bounds (CPU: 0.01–1.0 cores, Mem: 32–1024 MiB) |

All three layers are **fail-closed** — invalid input results in a denied resize, never a fallback approximation.

---

## Architecture

### Component Map

| Component | File | Role |
|---|---|---|
| Prometheus Ingestion | `prometheus_client.py` | 20-feature time-series matrix; circuit breaker, vectorised alignment |
| ML Inference Engine | `ml_engine.py` | TorchScript GRU; shape validation, safety buffer, bounds clamping |
| Policy Engine | `policy_engine.py` | Cooldown, significance threshold, maintenance windows; fail-closed |
| Safety Firewall | `safety_layer.py` | Redis fork-safety floor, downscale dampening, bounds enforcement |
| Kubernetes Adapter | `k8s_adapter.py` | Dual-patch: live Pod in-place resize + workload spec persistence |
| Orchestrator | `main.py` | Reconciliation loop; wires all components together |
| State Manager | `state_manager.py` | Crash-safe JSON state; atomic writes, UTC timestamps |
| Leader Election | `leader_election.py` | Kubernetes Lease-based HA; fencing tokens prevent split-brain |
| Discovery Engine | `discovery.py` | Annotation-driven workload discovery; paginated, fail-closed |
| Model Lifecycle Mgr | `lifecycle_manager.py` | Ed25519-signed model hot-reload; SHA-256 verification |
| Admission Controller | `admission_controller.py` | Mutating webhook; injects AI-optimised resources at Pod creation |
| Event Logger | `event_logger.py` | Kubernetes Events for audit trail; never blocks reconciliation |
| Health Sidecar | `health.py` | Separate OS process for `/healthz` and `/readyz` |
| Metrics Exporter | `metrics_exporter.py` | Prometheus metrics: AI suggestions, safety clamps, decisions |
| Helm Chart | `Chart.yaml`, `values.yaml` | RBAC, NetworkPolicy, PDB, PodAntiAffinity, ServiceMonitor |

### The 20 Prometheus Features

The GRU model operates on a 60-second sliding window of 20 features across four dimensions:

| Dimension | Features |
|---|---|
| Resource consumption | `cpu_usage`, `mem_usage`, `cpu_throttling`, `throttled_periods` |
| Resource allocation | `cpu_request`, `cpu_limit`, `mem_request`, `mem_limit` |
| Application health | `latency` (p95), `http_requests_total`, `errors_5xx`, `ops_per_sec` |
| Infrastructure context | `node_cpu_usage`, `node_cpu_steal`, `node_mem_available`, `pod_restarts`, `oom_kills`, `net_receive`, `net_transmit`, `replicas` |

### Workload Lifecycle

- **Day 0 — Pod Creation:** The Mutating Admission Webhook intercepts new Pod requests and injects the last AI-optimised resource values from the StateManager. Pods start right-sized from their first second.
- **Day 1 — Observation:** The Discovery Engine finds workloads annotated with `ai-vpa.io/managed=true` and begins feeding their metrics into the GRU sliding window.
- **Day N — Live Optimisation:** The controller continuously predicts, validates through all safety layers, and applies in-place resource patches. The workload never restarts.

---

## Installation & Quick Start

> **⚠️ Work in Progress — The scripts, Helm chart, and training pipeline referenced in this section are not yet functional. This section documents the intended workflow. See the [Roadmap](#roadmap) for current status.**

> **Prerequisites:** Kubernetes >= 1.35, Helm >= 3.10, Prometheus stack deployed in your cluster, a GRU model artifact built and available (see [Model Training](#model-training) below).

### 1. Clone the repository

```bash
git clone https://github.com/alenwrites/ai-vpa-controller.git
cd ai-vpa-controller
```

### 2. Annotate your target workload

The controller only manages workloads that explicitly opt in:

```bash
kubectl annotate statefulset <your-workload> ai-vpa.io/managed=true
kubectl label statefulset <your-workload> ai-vpa.io/enabled=true
```

### 3. Install via Helm

```bash
helm install ai-vpa ./helm \
  --namespace ai-vpa-system \
  --create-namespace \
  --set metricsBackend.service.name=prometheus-server \
  --set metricsBackend.service.namespace=monitoring
```

### 4. Verify the controller is running

```bash
kubectl get pods -n ai-vpa-system
kubectl logs -n ai-vpa-system -l app=ai-vpa-controller -f
```

### 5. Observe a scaling event

```bash
# Watch for controller events on your workload
kubectl get events --field-selector involvedObject.name=<your-workload> -w

# Check that Pod resources changed without a restart
kubectl describe pod <your-pod-name> | grep -A5 "Limits:"
```

### Model Training

> ⚠️ The training pipeline scripts below are **not yet implemented**. This section documents the intended workflow for when they are available.

```bash
# Collect metrics from your cluster (requires Prometheus access)
python data/collect_metrics.py --prometheus-url http://localhost:9090 --output data/raw/

# Train the GRU model
python training/train_gru.py --data data/raw/ --output data/models/

# Export to TorchScript (required by ml_engine.py)
python training/export_torchscript.py --model data/models/checkpoint.pt \
  --output data/models/k8s_gru_independent.pt
```

---

## Configuration Reference

All configuration is managed through `values.yaml` for Helm deployments, or environment variables for local development.

### Core Settings

| Environment Variable | Default | Description |
|---|---|---|
| `POLL_INTERVAL` | `3` | Reconciliation loop interval in seconds |
| `COOLDOWN_SECONDS` | `600` | Minimum seconds between successive patches |
| `K8S_NAMESPACE` | `default` | Namespace to watch |
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus base URL |
| `CONTAINER_NAME` | `redis` | Target container name within the Pod |
| `STATEFULSET_NAME` | `vpa-test-app` | Target workload name |

### Resource Bounds

| Environment Variable | Default | Description |
|---|---|---|
| `MIN_CPU_CHANGE_THRESHOLD` | `0.1` | Minimum relative CPU change to trigger a resize (10%) |
| `MIN_CPU_CHANGE_ABSOLUTE` | `0.05` | Minimum absolute CPU change in cores |
| `MIN_MEM_CHANGE_MiB` | `64` | Minimum memory change in MiB to trigger a resize |

### Resource Governance (set in `values.yaml`)

```yaml
governance:
  cpu:
    absoluteMin: 0.1    # cores
    absoluteMax: 2.0    # cores
  memory:
    absoluteMin: 256    # MiB
    absoluteMax: 4096   # MiB

mlEngine:
  polling:
    intervalSeconds: 15
    cooldownSeconds: 600
```

### Opting a Workload Out

To exclude a specific Pod from mutation by the admission webhook:

```yaml
metadata:
  annotations:
    ai.vpa.io/disable: "true"
```

### Maintenance Windows

By default, resizes are frozen between 02:00 and 03:00 UTC. To customise, set `MAINTENANCE_WINDOWS_UTC` in `config.py`:

```python
MAINTENANCE_WINDOWS_UTC: list[tuple[int, int]] = [(2, 3), (14, 15)]
```

---

## Roadmap

The project is currently at the prototype/design stage. The following milestones are planned:

- [ ] **Phase 1 — Integration:** End-to-end pipeline validation on a Kind cluster with a live Redis StatefulSet; confirm KEP-1287 in-place resize works without Pod restarts
- [ ] **Phase 2 — Model Training:** Collect real cluster metrics, train GRU on 20-feature sequences, export to TorchScript, validate inference latency < 50ms
- [ ] **Phase 3 — Safety Validation:** Full unit test coverage (90%+) for `safety_layer.py` and `policy_engine.py`; fix known test contract mismatches in `test_mock_safety.py`
- [ ] **Phase 4 — Benchmarking:** Quantitative comparison of Standard VPA vs. Predictive In-Place: restart count, p95 latency, cache cold-start recovery time
- [ ] **Phase 5 — Upstream Contribution:** Open PR to `kubernetes/autoscaler` or a new CNCF sandbox repo; publish Helm chart to ArtifactHub
- [ ] **Future — HPA + VPA Coordination:** Integrate horizontal and vertical scaling decisions
- [ ] **Future — Multi-workload Correlation:** Scale a cache proactively based on load patterns of the database that drives it

---

## Contributing

Contributions are welcome, especially at this early stage. Here is how to help:

### Good first issues to work on

- Writing integration tests for `k8s_adapter.py` against a Kind cluster
- Fixing the contract mismatches in `tests/test_mock_safety.py`
- Improving error messages in `prometheus_client.py` for missing metrics
- Writing the model training pipeline (`training/train_gru.py`)

### How to contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and add tests where applicable
4. Open a pull request with a clear description of what you changed and why

### Development setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run existing tests
python -m pytest tests/

# Run with a local kubeconfig (outside cluster)
export PROMETHEUS_URL=http://localhost:9090
export K8S_NAMESPACE=default
python main.py
```

### Reporting issues

Please open a GitHub issue with:
- A clear description of the problem
- Steps to reproduce
- Your Kubernetes version and cluster setup (Kind, Minikube, managed cluster)
- Relevant logs from the controller

---

## Author

**Alen P Praveen**
GitHub: [@alenwrites](https://github.com/alenwrites)

---

> This project is submitted as part of a Google Summer of Code 2026 application to the Cloud Native Computing Foundation (CNCF), targeting the Kubernetes SIG-Autoscaling working group.
