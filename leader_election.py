# leader_election.py
import os
import uuid
import random
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
import config

from kubernetes import client
from kubernetes.client.rest import ApiException

log = logging.getLogger("vpa.leader")


class LeaderElection:
    """
    Production-grade Kubernetes Lease-based leader election.

    Optional polish applied:
    - Immediate leadership confirmation on create
    - Local fencing token initialized on create
    - Explicitly documented backoff leadership behavior

    Correctness model and public API are unchanged.
    """

    def __init__(self, lease_name: str, namespace: str):
        self.lease_name = lease_name
        self.namespace = namespace
        self.holder_id = config.POD_NAME  # <--- Integrated with Downward API

        self.lease_duration = 15
        self.renew_deadline_ratio = 0.7

        self.coordination_v1 = client.CoordinationV1Api()

        self._is_leader = False
        self._fencing_token = 0

        # Confirmation timing (env preferred)
        confirm_seconds = int(os.getenv("LEADER_CONFIRM_INTERVAL", "5"))
        self._confirm_interval = timedelta(seconds=max(1, confirm_seconds))
        self._last_confirm_time = None

        # Non-blocking backoff
        self._backoff_base = 0.2
        self._backoff_max = 5.0
        self._backoff_attempts = 0
        self._next_retry_time = None

        # Cached Lease (reduces API load)
        self._cached_lease: Optional[client.V1Lease] = None

    def check_leadership(self) -> bool:
        """
        The only method that decides leadership.

        Backoff behavior (explicitly documented):
        - During backoff, we return the last known leadership state.
        - This is safe and consistent with Kubernetes leader-election:
          transient API failures do not immediately revoke leadership.
        """
        now = datetime.now(timezone.utc)

        if self._next_retry_time and now < self._next_retry_time:
            return self._is_leader

        lease = self._cached_lease
        if lease is None or self._needs_reconfirmation(now):
            lease = self._read_lease()
            if lease is None:
                return self._is_leader

        return self._try_update_lease(lease, now)

    def _read_lease(self) -> Optional[client.V1Lease]:
        """
        Reads the Lease or attempts creation.

        Always returns Optional[V1Lease].
        Never makes leadership decisions.
        """
        try:
            lease = self.coordination_v1.read_namespaced_lease(
                self.lease_name, self.namespace
            )
            self._cached_lease = lease
            return lease

        except ApiException as e:
            if e.status == 404:
                return self._try_create_lease(datetime.now(timezone.utc))

            log.error(f"Lease read failed: {e}")
            self._lose_leadership("read error")
            self._schedule_backoff()
            return None

    def _try_create_lease(self, now: datetime) -> Optional[client.V1Lease]:
        """
        Race-safe Lease creation.

        Optional improvement:
        - Successful create is treated as confirmed leadership.
        - Safe because create is linearizable and conflict-guarded.
        """
        lease = client.V1Lease(
            metadata=client.V1ObjectMeta(
                name=self.lease_name,
                namespace=self.namespace,
            ),
            spec=client.V1LeaseSpec(
                holder_identity=self.holder_id,
                lease_duration_seconds=self.lease_duration,
                acquire_time=now,
                renew_time=now,
                lease_transitions=1,
            ),
        )

        try:
            created = self.coordination_v1.create_namespaced_lease(
                self.namespace, lease
            )
            self._cached_lease = created

            # Immediate confirmed leadership on create
            self._fencing_token = created.spec.lease_transitions or 1
            self._confirm_transition(True, "acquired (create)")
            self._reset_backoff()
            self._last_confirm_time = now

            return created

        except ApiException as e:
            if e.status == 409:
                return None

            log.error(f"Lease create failed: {e}")
            self._schedule_backoff()
            return None

    def _try_update_lease(self, lease: client.V1Lease, now: datetime) -> bool:
        """
        Acquire or renew leadership using optimistic locking.

        NOTE:
        - Cached Lease staleness is acceptable.
        - Leadership loss may be detected only during:
          * reconfirmation
          * renew window
          * update attempt
        """
        spec = lease.spec
        holder = spec.holder_identity
        renew_time = spec.renew_time or spec.acquire_time

        if renew_time is None:
            renew_time = now - timedelta(seconds=self.lease_duration)

        elapsed = (now - renew_time).total_seconds()
        expired = elapsed >= self.lease_duration
        renew_threshold = self.lease_duration * self.renew_deadline_ratio
        within_window = elapsed >= renew_threshold

        if holder != self.holder_id and not expired:
            self._confirm_transition(False, f"held by {holder}")
            return False

        if (
            holder == self.holder_id
            and not expired
            and not within_window
            and not self._needs_reconfirmation(now)
        ):
            return True

        acquiring = holder != self.holder_id or expired

        transitions = spec.lease_transitions or 0
        if acquiring:
            spec.lease_transitions = transitions + 1

        spec.holder_identity = self.holder_id
        spec.renew_time = now
        spec.lease_duration_seconds = self.lease_duration

        try:
            updated = self.coordination_v1.replace_namespaced_lease(
                self.lease_name,
                self.namespace,
                lease,
            )
            self._cached_lease = updated
            self._fencing_token = updated.spec.lease_transitions or transitions
            self._confirm_transition(True, "confirmed by update")
            self._reset_backoff()
            self._last_confirm_time = now
            return True

        except ApiException as e:
            if e.status == 409:
                self._confirm_transition(False, "update conflict")
            else:
                log.error(f"Lease update failed: {e}")
                self._confirm_transition(False, "update error")

            self._schedule_backoff()
            return False

    def _needs_reconfirmation(self, now: datetime) -> bool:
        if self._last_confirm_time is None:
            return True
        return now - self._last_confirm_time >= self._confirm_interval

    def _confirm_transition(self, is_leader: bool, reason: str):
        if self._is_leader != is_leader:
            if is_leader:
                log.info(
                    f"Leadership gained ({reason}), fencing-token={self._fencing_token}"
                )
            else:
                log.warning(f"Leadership lost ({reason})")
        self._is_leader = is_leader

    def _schedule_backoff(self):
        self._backoff_attempts += 1
        delay = min(
            self._backoff_max,
            self._backoff_base * (2 ** self._backoff_attempts),
        )
        delay *= 1 + random.random() * 0.2
        self._next_retry_time = datetime.now(timezone.utc) + timedelta(seconds=delay)

    def _reset_backoff(self):
        self._backoff_attempts = 0
        self._next_retry_time = None

    def _lose_leadership(self, reason: str):
        self._confirm_transition(False, reason)

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    @property
    def fencing_token(self) -> int:
        """
        Fencing token backed by spec.leaseTransitions.

        Tradeoff (explicit):
        - Other controllers may modify this field.
        - Acceptable for controller-level fencing.
        - CRDs are recommended for strict fencing guarantees.
        """
        return self._fencing_token
