"""Safety state machine shared by the VisionPilot desktop dashboard and its tests.

This module deliberately has no Qt dependency so the arm/calibration gate can be
verified without opening a camera or touching the OS mouse.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DashboardSafetyGate:
    """Keep desktop arming dependent on a usable backend and valid profile."""

    desktop_available: bool = False
    profile_valid: bool = False
    calibration_active: bool = False
    armed: bool = False

    @property
    def can_arm(self) -> bool:
        return (
            self.desktop_available
            and self.profile_valid
            and not self.calibration_active
            and not self.armed
        )

    def set_desktop_available(self, available: bool) -> None:
        self.desktop_available = bool(available)
        if not self.desktop_available:
            self.armed = False
            self.calibration_active = False

    def set_profile_valid(self, valid: bool) -> None:
        self.profile_valid = bool(valid)
        if not self.profile_valid:
            self.armed = False

    def begin_calibration(self) -> bool:
        if not self.desktop_available or self.armed or self.calibration_active:
            return False
        self.calibration_active = True
        return True

    def finish_calibration(self, *, valid_profile: bool) -> None:
        self.calibration_active = False
        self.profile_valid = bool(valid_profile)

    def request_arm(self) -> bool:
        """Toggle arming; return the resulting state.

        A failed arm request leaves the state disarmed. Calling this while
        armed always disarms, which makes the UI action safe and predictable.
        """
        if self.armed:
            self.armed = False
            return False
        if self.can_arm:
            self.armed = True
        return self.armed

    def disarm(self) -> None:
        self.armed = False

    def emergency_stop(self) -> None:
        self.disarm()
