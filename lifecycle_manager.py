# lifecycle_manager.py
import os
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Callable, Dict, Any

from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

SUPPORTED_FORMAT_VERSION = 3

log = logging.getLogger("vpa.lifecycle")


class SecurityError(RuntimeError):
    """Integrity or trust violation."""


class TransientFailure(RuntimeError):
    """Temporary failure: IO, fsync, incomplete write."""


class PersistentFailure(SecurityError):
    """Permanent trust failure: signature, hash, schema."""


class ModelLifecycleManager:
    """
    Final hardened lifecycle manager.

    Trust model:
    - Filesystem is hostile
    - Only signed manifests establish trust
    - state.json is advisory cache only
    """

    STATE_FILE = "state.json"
    TEMP_SUFFIXES = (".tmp", ".partial", ".download")
    MAX_ROLLBACK_DEPTH = 2

    def __init__(
        self,
        model_root: Path,
        public_key_pem: bytes,
        on_load_sync: Callable[[int], None],
        post_load_check: Callable[[], None],
        debounce_seconds: float = 2.0,
        backoff_seconds: float = 30.0,
    ):
        self.model_root = model_root.resolve(strict=True)
        self.releases_dir = (self.model_root / "releases").resolve(strict=True)
        self.current_symlink = self.model_root / "current"
        self.state_path = self.model_root / self.STATE_FILE

        if not self.releases_dir.exists():
            raise SecurityError("releases/ directory missing")

        self.public_key: Ed25519PublicKey = load_pem_public_key(public_key_pem)
        self.on_load_sync = on_load_sync
        self.post_load_check = post_load_check

        self.debounce_seconds = debounce_seconds
        self.backoff_seconds = backoff_seconds

        self._lock = threading.Lock()
        self._last_attempt = 0.0
        self._backoff_until = 0.0

        self._state = self._load_state_advisory()

        if self._state.get("active_path"):
            self._activate_path(self._state["active_path"])

    # ------------------------------------------------------------------
    # STATE (ADVISORY ONLY)
    # ------------------------------------------------------------------

    def _load_state_advisory(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return {}

    def _write_state_atomic(self, state: Dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # NOTE:
        # We intentionally omit fsync() on the parent directory.
        # This provides crash safety for process failures, but not
        # full power-loss durability. This is an accepted tradeoff.
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    def check_for_updates(self) -> None:
        now = time.monotonic()

        with self._lock:
            if now < self._backoff_until:
                return
            if now - self._last_attempt < self.debounce_seconds:
                return

            self._last_attempt = now

            try:
                target = self._resolve_current_symlink()
                if self._state.get("active_path") == str(target):
                    return

                self._activate_path(str(target))

            except TransientFailure as e:
                self._backoff_until = now + self.backoff_seconds
                log.warning("Transient activation failure: %s", e)
                self._attempt_rollback()

            except PersistentFailure as e:
                self._backoff_until = now + (self.backoff_seconds * 10)
                log.error("Persistent trust failure: %s", e)
                self._attempt_rollback()

    # ------------------------------------------------------------------
    # ACTIVATION PIPELINE
    # ------------------------------------------------------------------

    def _activate_path(self, path_str: str) -> None:
        model_path = Path(path_str).resolve(strict=True)

        self._enforce_containment(model_path)
        self._reject_temporary(model_path)

        manifest = self._load_and_verify_manifest(model_path)

        try:
            fd = os.open(model_path, os.O_RDONLY)
        except OSError as e:
            raise TransientFailure(f"I/O error opening model: {e}")

        try:
            actual_hash = self._hash_fd(fd)
            if actual_hash != manifest["sha256"]:
                raise PersistentFailure("Model hash mismatch")

            self.on_load_sync(fd)
            self.post_load_check()

            self._commit_activation(model_path, manifest, actual_hash)

        finally:
            os.close(fd)

    # ------------------------------------------------------------------
    # COMMIT
    # ------------------------------------------------------------------

    def _commit_activation(
        self,
        model_path: Path,
        manifest: Dict[str, Any],
        checksum: str,
    ) -> None:
        self._atomic_symlink_swap(model_path)

        new_state = {
            "active_path": str(model_path),
            "version": manifest["version"],
            "checksum": checksum,
            "activated_at": time.time(),
            "rollback": self._bounded_rollback_chain(),
        }

        self._write_state_atomic(new_state)
        self._state = new_state

        log.info("Activated model version=%s", manifest["version"])

    # ------------------------------------------------------------------
    # ROLLBACK
    # ------------------------------------------------------------------

    def _attempt_rollback(self) -> None:
        chain = self._state.get("rollback", [])
        if not chain:
            log.critical("No rollback available; remaining fail-closed")
            return

        candidate = chain[-1]
        try:
            self._activate_path(candidate["active_path"])
            log.warning("Rollback successful")

        except PersistentFailure as e:
            log.critical("Rollback failed permanently: %s", e)

        except TransientFailure as e:
            log.critical("Rollback failed transiently: %s", e)

    def _bounded_rollback_chain(self) -> list:
        prev_chain = list(self._state.get("rollback", []))
        prev_chain = prev_chain[-(self.MAX_ROLLBACK_DEPTH - 1):]
        prev_chain.append(self._state)
        return prev_chain

    # ------------------------------------------------------------------
    # MANIFEST VERIFICATION
    # ------------------------------------------------------------------

    def _load_and_verify_manifest(self, model_path: Path) -> Dict[str, Any]:
        manifest_path = model_path.with_suffix(".manifest.json")
        sig_path = manifest_path.with_suffix(".json.sig")

        if not manifest_path.exists() or not sig_path.exists():
            raise SecurityError("Missing manifest or signature")

        raw = manifest_path.read_bytes()
        sig = sig_path.read_bytes()

        try:
            self.public_key.verify(sig, raw)
        except InvalidSignature:
            raise PersistentFailure("Manifest signature invalid")

        data = json.loads(raw)

        required = {
            "version",
            "sha256",
            "filename",
            "filesize",
            "build_ts",
            "format_version",
        }
        if not required.issubset(data):
            raise SecurityError("Manifest incomplete")

        if data["format_version"] != SUPPORTED_FORMAT_VERSION:
            raise PersistentFailure(
                f"Incompatible model format_version="
                f"{data['format_version']} "
                f"(expected {SUPPORTED_FORMAT_VERSION})"
            )

        if data["filename"] != model_path.name:
            raise SecurityError("Filename mismatch")

        if model_path.stat().st_size != data["filesize"]:
            raise SecurityError("Filesize mismatch")

        return data

    # ------------------------------------------------------------------
    # FILESYSTEM SAFETY
    # ------------------------------------------------------------------

    def _resolve_current_symlink(self) -> Path:
        if not self.current_symlink.is_symlink():
            raise SecurityError("current is not a symlink")
        return self.current_symlink.resolve(strict=True)

    def _atomic_symlink_swap(self, target: Path) -> None:
        tmp = self.current_symlink.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()

        tmp.symlink_to(target)
        os.replace(tmp, self.current_symlink)

    def _enforce_containment(self, path: Path) -> None:
        try:
            path.relative_to(self.releases_dir)
        except ValueError:
            raise SecurityError("Model path escapes releases directory")

    def _reject_temporary(self, path: Path) -> None:
        if path.name.endswith(self.TEMP_SUFFIXES):
            raise SecurityError("Temporary artifact rejected")

    # ------------------------------------------------------------------
    # HASHING
    # ------------------------------------------------------------------

    def _hash_fd(self, fd: int) -> str:
        hasher = hashlib.sha256()
        os.lseek(fd, 0, os.SEEK_SET)
        while chunk := os.read(fd, 8192):
            hasher.update(chunk)
        os.lseek(fd, 0, os.SEEK_SET)
        return hasher.hexdigest()
