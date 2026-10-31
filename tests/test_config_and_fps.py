import unittest

from visionpilot.config import VisionPilotConfig
from visionpilot.fps import FPSCounter


class ConfigTests(unittest.TestCase):
    def test_default_config_is_single_hand(self):
        config = VisionPilotConfig()
        self.assertEqual(config.max_hands, 1)
        self.assertEqual(config.camera_index, 0)

    def test_rejects_two_hands_in_phase_one(self):
        with self.assertRaises(ValueError):
            VisionPilotConfig(max_hands=2)

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            VisionPilotConfig(detection_confidence=1.5)


class FPSTests(unittest.TestCase):
    def test_fps_is_calculated(self):
        counter = FPSCounter(smoothing=1.0)
        self.assertEqual(counter.tick(now=10.0), 0.0)
        self.assertAlmostEqual(counter.tick(now=10.1), 10.0)

    def test_non_positive_time_delta_does_not_crash(self):
        counter = FPSCounter()
        counter.tick(now=10.0)
        self.assertEqual(counter.tick(now=10.0), 0.0)


if __name__ == "__main__":
    unittest.main()
