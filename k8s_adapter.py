# k8s_adapter.py
import logging
import math
import time
from typing import Tuple, Optional

from kubernetes import client
from kubernetes import config as k8s_config
from kubernetes.client.exceptions import ApiException
from errors import TransientError

import config

log = logging.getLogger("vpa.k8s_adapter")

class KubernetesPatchError(RuntimeError):
    """Raised when a Kubernetes patch operation fails."""

class KubernetesAdapter:
    def __init__(self) -> None:
        try:
            try:
                k8s_config.load_incluster_config()
                log.info("Loaded in-cluster Kubernetes configuration")
            except Exception:
                k8s_config.load_kube_config()
                log.info("Loaded local kubeconfig")

            self.apps_v1 = client.AppsV1Api()
        except Exception as e:
            log.error("Failed to initialize Kubernetes client", exc_info=True)
            raise RuntimeError("Kubernetes client initialization failed") from e

    def _read_workload(self, name: str, namespace: str, kind: str):
        """Read either a Deployment or StatefulSet dynamically."""
        try:
            if kind == "Deployment":
                return self.apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
            elif kind == "StatefulSet":
                return self.apps_v1.read_namespaced_stateful_set(name=name, namespace=namespace)
            raise ValueError(f"Unsupported workload kind: {kind}")
        except ApiException as e:
            raise KubernetesPatchError(f"Failed to read {kind} {namespace}/{name}: {e.status} {e.reason}") from e


    def _validate_and_get_container(self, deployment) -> client.V1Container:
        pod_spec = (
            deployment.spec
            and deployment.spec.template
            and deployment.spec.template.spec
        )

        if not pod_spec or not pod_spec.containers:
            raise ValueError(
                f"Deployment {config.STATEFULSET_NAME} has no containers defined"
            )

        matches = [
            c for c in pod_spec.containers if c.name == config.CONTAINER_NAME
        ]

        if not matches:
            raise ValueError(
                f"Container '{config.CONTAINER_NAME}' not found in Deployment"
            )
            
        return matches[0]

    @staticmethod
    def _validate_cpu(cpu_cores: float) -> int:
        millicores = math.ceil(cpu_cores * 1000)
        return max(1, millicores)

    @staticmethod
    def _validate_memory(mem_mib: int) -> int:
        return max(1, mem_mib)

    def _log_update_strategy(self, deployment) -> str:
        # Deployments use 'strategy', StatefulSets use 'update_strategy'
        strategy = deployment.spec.strategy
        strategy_type = strategy.type if strategy and strategy.type else "RollingUpdate"
        log.info(f"Deployment strategy={strategy_type}")
        return strategy_type

    def _wait_for_rollout_start(self, generation: int) -> None:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                deploy = self._read_deployment()
                status = deploy.status
                if not status:
                    time.sleep(1)
                    continue

                # Check if the controller has acknowledged the new generation
                if status.observed_generation and status.observed_generation >= generation:
                    # Also check if at least one replica is being updated
                    if status.updated_replicas and status.updated_replicas >= 1:
                        log.info(f"Rollout acknowledged: generation {generation}")
                        return
            except Exception:
                pass
            time.sleep(1)
        log.warning("Rollout acknowledgement timeout; continuing asynchronously")

    def patch_resources(
        self,
        name: str,
        namespace: str,
        kind: str,
        cpu_cores: float,
        mem_mib: int,
        dry_run: bool = False
    ) -> None:
        """Apply resources using Dual-Patch logic (Pod + Workload)."""
        millicores = self._validate_cpu(cpu_cores)
        mem_mib = self._validate_memory(mem_mib)

        # Standard K8s unit conversion
        req_cpu, req_mem = f"{millicores}m", f"{mem_mib}Mi"
        
        # --- ACTION A: PATCH LIVE PODS (In-Place / No Restart) ---
        # This targets the actual running Linux processes.
        try:
            # We list pods in the namespace to find those owned by this workload
            pods = client.CoreV1Api().list_namespaced_pod(namespace, label_selector=f"app={name}")
            for pod in pods.items:
                # KEP-1287: Patching the pod spec resources field directly
                pod_patch = {
                    "spec": {
                        "containers": [{
                            "name": config.CONTAINER_NAME,
                            "resources": {
                                "requests": {"cpu": req_cpu, "memory": req_mem},
                                "limits": {"cpu": req_cpu, "memory": req_mem}
                            }
                        }]
                    }
                }
                # Using patch_namespaced_pod_status or patch_namespaced_pod depending on K8s version
                client.CoreV1Api().patch_namespaced_pod(pod.metadata.name, namespace, body=pod_patch)
                log.info(f"Zero-Downtime: In-place resized Pod {pod.metadata.name}")
        except Exception as e:
            log.error(f"In-place resize failed: {e}. Falling back to RollingUpdate.")

        # --- ACTION B: PATCH WORKLOAD (Persistence) ---
        # This ensures that if the pod ever restarts, it uses the new values.
        workload_patch = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": config.CONTAINER_NAME,
                            "resources": {
                                "requests": {"cpu": req_cpu, "memory": req_mem},
                                "limits": {"cpu": req_cpu, "memory": req_mem}
                            }
                        }]
                    }
                }
            }
        }
        
        try:
            if kind == "StatefulSet":
                self.apps_v1.patch_namespaced_stateful_set(name, namespace, body=workload_patch)
            else:
                self.apps_v1.patch_namespaced_deployment(name, namespace, body=workload_patch)
            log.info(f"Action B: Updated {kind} {name} metadata.")
        except ApiException as e:
            log.error(f"Workload patch failed: {e}")
