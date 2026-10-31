import unittest

from visionpilot.dashboard_state import DashboardSafetyGate


class DashboardSafetyGateTests(unittest.TestCase):
    def test_starts_disarmed_and_cannot_arm_without_backend_or_profile(self):
        gate = DashboardSafetyGate()
        self.assertFalse(gate.armed)
        self.assertFalse(gate.can_arm)
        self.assertFalse(gate.request_arm())
        self.assertFalse(gate.armed)

    def test_requires_valid_profile_and_backend(self):
        gate = DashboardSafetyGate()
        gate.set_profile_valid(True)
        self.assertFalse(gate.request_arm())
        gate.set_desktop_available(True)
        self.assertTrue(gate.request_arm())
        self.assertTrue(gate.armed)

    def test_calibration_blocks_arm_until_finished(self):
        gate = DashboardSafetyGate(desktop_available=True, profile_valid=True)
        self.assertTrue(gate.begin_calibration())
        self.assertFalse(gate.request_arm())
        gate.finish_calibration(valid_profile=True)
        self.assertTrue(gate.request_arm())

    def test_profile_invalidation_disarms(self):
        gate = DashboardSafetyGate(
            desktop_available=True, profile_valid=True, armed=True
        )
        gate.set_profile_valid(False)
        self.assertFalse(gate.armed)
        self.assertFalse(gate.can_arm)

    def test_calibration_cannot_start_when_armed_or_without_backend(self):
        gate = DashboardSafetyGate()
        self.assertFalse(gate.begin_calibration())
        gate.set_desktop_available(True)
        gate.set_profile_valid(True)
        self.assertTrue(gate.request_arm())
        self.assertFalse(gate.begin_calibration())

    def test_duplicate_calibration_start_is_rejected(self):
        gate = DashboardSafetyGate(desktop_available=True)
        self.assertTrue(gate.begin_calibration())
        self.assertFalse(gate.begin_calibration())

    def test_emergency_stop_always_disarms(self):
        gate = DashboardSafetyGate(
            desktop_available=True, profile_valid=True, armed=True
        )
        gate.emergency_stop()
        self.assertFalse(gate.armed)


if __name__ == "__main__":
    unittest.main()
