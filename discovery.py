# discovery.py
import logging
from typing import List, NamedTuple, Set, Tuple, Optional

from kubernetes import client
from kubernetes.client.exceptions import ApiException
from kubernetes.client.models import V1ObjectMeta

log = logging.getLogger("vpa.discovery")


class ScalableTarget(NamedTuple):
    name: str
    kind: str
    namespace: str


class DiscoveryResult(NamedTuple):
    """
    Explicit discovery outcome.

    Allows callers to distinguish:
    - empty but healthy discovery
    - partial discovery
    - global fail-closed failure
    """
    targets: List[ScalableTarget]
    partial_discovery: bool
    statefulset_failed: bool
    deployment_failed: bool
    unexpected_failure: bool


class WorkloadKind:
    __slots__ = ()

    DEPLOYMENT = "Deployment"
    STATEFULSET = "StatefulSet"


class DiscoveryEngine:
    """
    Authorized workload discovery.

    Authorization is strict and fail-closed.
    Availability may degrade explicitly via partial discovery.
    """

    LABEL_SELECTOR = "ai-vpa.io/enabled=true"

    REQUIRED_ANNOTATION_KEY = "ai-vpa.io/managed"
    REQUIRED_BOOLEAN_VALUE = "true"

    # Conservative page size to avoid large API payloads
    PAGE_LIMIT = 200

    def __init__(self) -> None:
        self.apps_v1 = client.AppsV1Api()

    def _is_explicitly_authorized(self, metadata: V1ObjectMeta) -> bool:
        annotations = metadata.annotations or {}
        return annotations.get(self.REQUIRED_ANNOTATION_KEY) == self.REQUIRED_BOOLEAN_VALUE

    def _discover_statefulsets(
        self,
        namespace: str,
        targets: List[ScalableTarget],
        seen: Set[Tuple[str, str, str]],
    ) -> bool:
        """
        Discover StatefulSets with pagination.

        Returns True if all pages are fetched successfully.
        Returns False if a handled ApiException occurs.
        Unexpected exceptions propagate.
        """
        continue_token: Optional[str] = None

        while True:
            try:
                resp = self.apps_v1.list_namespaced_stateful_set(
                    namespace=namespace,
                    label_selector=self.LABEL_SELECTOR,
                    limit=self.PAGE_LIMIT,
                    _continue=continue_token,
                )
            except ApiException as e:
                log.error(
                    "Kubernetes API error during StatefulSet discovery",
                    extra={
                        "exception_type": type(e).__name__,
                        "namespace": namespace,
                        "operation": "list_namespaced_stateful_set",
                        "status": e.status,
                        "reason": e.reason,
                        "partial_discovery": True,
                    },
                )
                return False

            for item in resp.items:
                if not self._is_explicitly_authorized(item.metadata):
                    continue

                key = (WorkloadKind.STATEFULSET, item.metadata.name, namespace)
                if key in seen:
                    continue

                seen.add(key)
                targets.append(
                    ScalableTarget(
                        name=item.metadata.name,
                        kind=WorkloadKind.STATEFULSET,
                        namespace=namespace,
                    )
                )

            continue_token = resp.metadata._continue
            if not continue_token:
                break

        return True

    def _discover_deployments(
        self,
        namespace: str,
        targets: List[ScalableTarget],
        seen: Set[Tuple[str, str, str]],
    ) -> bool:
        """
        Discover Deployments with pagination.

        Returns True if all pages are fetched successfully.
        Returns False if a handled ApiException occurs.
        Unexpected exceptions propagate.
        """
        continue_token: Optional[str] = None

        while True:
            try:
                resp = self.apps_v1.list_namespaced_deployment(
                    namespace=namespace,
                    label_selector=self.LABEL_SELECTOR,
                    limit=self.PAGE_LIMIT,
                    _continue=continue_token,
                )
            except ApiException as e:
                log.error(
                    "Kubernetes API error during Deployment discovery",
                    extra={
                        "exception_type": type(e).__name__,
                        "namespace": namespace,
                        "operation": "list_namespaced_deployment",
                        "status": e.status,
                        "reason": e.reason,
                        "partial_discovery": True,
                    },
                )
                return False

            for item in resp.items:
                if not self._is_explicitly_authorized(item.metadata):
                    continue

                key = (WorkloadKind.DEPLOYMENT, item.metadata.name, namespace)
                if key in seen:
                    continue

                seen.add(key)
                targets.append(
                    ScalableTarget(
                        name=item.metadata.name,
                        kind=WorkloadKind.DEPLOYMENT,
                        namespace=namespace,
                    )
                )

            continue_token = resp.metadata._continue
            if not continue_token:
                break

        return True

    def get_authorized_workloads(self, namespace: str) -> DiscoveryResult:
        """
        Orchestrates discovery while enforcing explicit failure semantics.

        Known API failures → partial discovery
        Unknown failures   → global fail-closed
        """
        targets: List[ScalableTarget] = []
        seen: Set[Tuple[str, str, str]] = set()

        statefulset_failed = False
        deployment_failed = False
        unexpected_failure = False

        try:
            if not self._discover_statefulsets(namespace, targets, seen):
                statefulset_failed = True

            if not self._discover_deployments(namespace, targets, seen):
                deployment_failed = True

        except Exception as e:
            unexpected_failure = True
            log.error(
                "Unexpected error during authorized workload discovery; failing closed",
                extra={
                    "exception_type": type(e).__name__,
                    "namespace": namespace,
                    "operation": "authorized_workload_discovery",
                    "partial_discovery": False,
                    "global_failure": True,
                },
                exc_info=True,
            )
            return DiscoveryResult(
                targets=[],
                partial_discovery=False,
                statefulset_failed=False,
                deployment_failed=False,
                unexpected_failure=True,
            )

        partial_discovery = statefulset_failed or deployment_failed

        log.info(
            "Authorized workload discovery complete",
            extra={
                "namespace": namespace,
                "authorized_count": len(targets),
                "partial_discovery": partial_discovery,
                "statefulset_failed": statefulset_failed,
                "deployment_failed": deployment_failed,
                "unexpected_failure": False,
            },
        )

        return DiscoveryResult(
            targets=targets,
            partial_discovery=partial_discovery,
            statefulset_failed=statefulset_failed,
            deployment_failed=deployment_failed,
            unexpected_failure=False,
        )
