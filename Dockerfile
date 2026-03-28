# syntax=docker/dockerfile:1.6

#####################################################################
# BUILD MODEL DECISION
#
# We intentionally constrain builds to linux/amd64.
# Scientific Python wheels (numpy/torch) are manylinux glibc wheels.
# Cross-building without native arch risks subtle ABI/runtime breakage.
#
# If multi-arch is required, separate CI pipelines must build per-arch.
#####################################################################

ARG TARGETPLATFORM
FROM --platform=linux/amd64 python:3.11.8-slim-bookworm@sha256:e0b57e4e6677f5973211516f1c4e756184518335359a35e4e2a86847c20c0251 AS builder

# Fail fast if attempting cross-platform build
RUN test "${TARGETPLATFORM:-linux/amd64}" = "linux/amd64"

ARG UID=10001
ARG GID=10001

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

#####################################################################
# Builder Dependencies
# Only packages required to compile scientific wheels if needed.
#####################################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

#####################################################################
# Deterministic Python Toolchain
#####################################################################

RUN python -m pip install --upgrade \
        pip==23.3.2 \
        setuptools==69.0.3 \
        wheel==0.42.0

WORKDIR /build

# requirements.txt MUST:
# - Pin every version
# - Include sha256 hashes
COPY requirements.txt .

#####################################################################
# Wheel Build Phase (Hash-Verified)
#####################################################################

RUN pip wheel \
        --require-hashes \
        --no-deps \
        --wheel-dir /wheels \
        -r requirements.txt

#####################################################################
# Virtualenv Install Phase (Offline, Deterministic)
#####################################################################

RUN python -m venv ${VIRTUAL_ENV}

RUN ${VIRTUAL_ENV}/bin/pip install \
        --require-hashes \
        --no-index \
        --find-links=/wheels \
        -r requirements.txt

#####################################################################
# Explicit Application Boundary
#####################################################################

# Instead of app/, copy all your synchronized logic files
COPY main.py /app/
COPY config.py /app/
COPY ml_engine.py /app/
COPY prometheus_client.py /app/
COPY k8s_adapter.py /app/
COPY lifecycle_manager.py /app/
COPY safety_layer.py /app/
COPY leader_election.py /app/
COPY admission_controller.py /app/
COPY discovery.py /app/
COPY policy_engine.py /app/
COPY state_manager.py /app/
COPY event_logger.py /app/
COPY errors.py /app/
COPY health.py /app/
COPY metrics_exporter.py /app/

#####################################################################
# Non-root user creation (numeric IDs for distroless compatibility)
#####################################################################

RUN groupadd -g ${GID} appgroup \
    && useradd -u ${UID} -g ${GID} -r -M -d /nonexistent appuser

# Pre-create runtime directories here (immutability guarantee)
RUN mkdir -p /app/data/models /app/data/state \
    && chown -R ${UID}:${GID} /app ${VIRTUAL_ENV}

#####################################################################
# RUNTIME STAGE (DISTROLESS, IMMUTABLE)
#
# DESIGN DECISION:
# We use DIRECT PATH COPYING (venv + /app only).
#
# Tradeoff:
# - More maintainable than full root filesystem staging
# - Clear boundaries (only Python + app copied)
# - Avoids accidental OS file propagation
# - Maintains immutability because runtime contains zero RUN steps
#####################################################################

FROM gcr.io/distroless/python3-debian12@sha256:69076d3330687f87f4f6696d72493a74a8a5f8e5352c80145c11030e46123440 AS runtime

ARG UID=10001
ARG GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/app/data/models \
    STATE_DIR=/app/data/state

WORKDIR /app

# Add metadata to the image to match Chart.yaml
LABEL org.opencontainers.image.version="0.4.0" \
      org.opencontainers.image.description="GRU-based Vertical Pod Autoscaler"

#####################################################################
# CA TRUST STORE STRATEGY
#
# We explicitly rely on distroless Debian 12 trust store.
# Image is pinned by digest to prevent trust-store drift.
# No additional CA manipulation is performed.
#####################################################################

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

USER ${UID}:${GID}

EXPOSE 8000
EXPOSE 8080

#####################################################################
# PROCESS MODEL & SIGNAL ASSUMPTIONS
#
# Assumption:
# - Application does NOT spawn subprocesses or multiprocessing pools.
# - No orphan/zombie risk.
#
# If multiprocessing is introduced in future,
# tini must be added explicitly in builder and copied in.
#
# Exec-form ENTRYPOINT ensures:
# - Python is PID 1
# - SIGTERM delivered directly
# - Kubernetes graceful shutdown works
#####################################################################

ENTRYPOINT ["/opt/venv/bin/python", "-m", "main"]

#####################################################################
# HEALTH STRATEGY
#
# No Docker HEALTHCHECK defined.
# Kubernetes readiness/liveness probes are authoritative.
#####################################################################
