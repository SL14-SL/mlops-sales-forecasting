from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from ..monitoring.serving import set_serving_readiness
from .serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
)

BundleLoader = Callable[[], ServingBundle]


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested without an active bundle."""


@dataclass(frozen=True)
class ReloadResult:
    """Result of an attempted serving-bundle reload."""

    success: bool
    previous_release_id: str | None
    active_release_id: str | None
    error: str | None = None


class ModelManager:
    """Manage the currently active serving bundle."""

    def __init__(
        self,
        bundle_loader: BundleLoader,
    ) -> None:
        self._bundle_loader = bundle_loader
        self._bundle: ServingBundle | None = None
        self._last_reload_error: str | None = None
        self._state_lock = RLock()
        self._reload_lock = RLock()
        set_serving_readiness(False)

    @property
    def ready(self) -> bool:
        """Return whether a serving bundle is currently available."""
        with self._state_lock:
            return self._bundle is not None

    @property
    def active_release_id(self) -> str | None:
        """Return the active release identifier, if available."""
        with self._state_lock:
            if self._bundle is None:
                return None

            return self._bundle.release_id

    @property
    def last_reload_error(self) -> str | None:
        """Return the most recent reload error."""
        with self._state_lock:
            return self._last_reload_error

    def get_bundle(self) -> ServingBundle:
        """Return the active bundle or raise a readiness error."""
        with self._state_lock:
            if self._bundle is None:
                raise ModelNotReadyError(
                    "No serving bundle is currently loaded."
                )

            return self._bundle

    def load_initial(self) -> ServingBundle:
        """Load the initial bundle and propagate failures."""
        with self._reload_lock:
            try:
                bundle = self._bundle_loader()
                validate_serving_bundle(bundle)
            except Exception as exc:
                with self._state_lock:
                    self._last_reload_error = str(exc)

                set_serving_readiness(False)
                raise

            with self._state_lock:
                self._bundle = bundle
                self._last_reload_error = None

            set_serving_readiness(True)
            return bundle

    def replace_bundle(
        self,
        bundle: ServingBundle,
    ) -> ServingBundle:
        """Atomically replace the active bundle."""
        validate_serving_bundle(bundle)

        with self._state_lock:
            self._bundle = bundle
            self._last_reload_error = None

        set_serving_readiness(True)
        return bundle

    def reload(self) -> ReloadResult:
        """Attempt to reload while preserving a working bundle."""
        with self._reload_lock:
            with self._state_lock:
                previous_release_id = (
                    self._bundle.release_id
                    if self._bundle is not None
                    else None
                )

            try:
                candidate_bundle = self._bundle_loader()
                validate_serving_bundle(candidate_bundle)
            except Exception as exc:
                error = str(exc)

                with self._state_lock:
                    self._last_reload_error = error
                    active_release_id = (
                        self._bundle.release_id
                        if self._bundle is not None
                        else None
                    )

                set_serving_readiness(
                    active_release_id is not None
                )

                return ReloadResult(
                    success=False,
                    previous_release_id=previous_release_id,
                    active_release_id=active_release_id,
                    error=error,
                )

            with self._state_lock:
                self._bundle = candidate_bundle
                self._last_reload_error = None

            set_serving_readiness(True)

            return ReloadResult(
                success=True,
                previous_release_id=previous_release_id,
                active_release_id=candidate_bundle.release_id,
            )