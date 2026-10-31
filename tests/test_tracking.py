import math
import unittest

from visionpilot.filters import OneEuroFilter
from visionpilot.landmarks import HandObservation, Landmark
from visionpilot.tracking import (
    HandTracker,
    MovementDirection,
    Point2D,
    TrackingConfig,
    TrackingPhase,
)


def make_observation(x=0.5, y=0.5, handedness="Right"):
    points = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    points[8] = Landmark(x, y, 0.0)
    return HandObservation(tuple(points), handedness, 0.98)


class OneEuroFilterTests(unittest.TestCase):
    def test_first_sample_is_immediate(self):
        filt = OneEuroFilter()
        self.assertAlmostEqual(filt.filter(0.42, 1.0), 0.42)

    def test_sudden_change_is_smoothed(self):
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        filt.filter(0.0, 1.0)
        value = filt.filter(1.0, 1.1)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

    def test_rejects_bad_samples_and_timestamps(self):
        filt = OneEuroFilter()
        with self.assertRaises(ValueError):
            filt.filter(float("nan"), 1.0)
        filt.filter(0.0, 1.0)
        with self.assertRaises(ValueError):
            filt.filter(0.1, 1.0)


class HandTrackerTests(unittest.TestCase):
    def test_extracts_normalized_index_fingertip(self):
        result = HandTracker().update(make_observation(0.61, 0.29), timestamp=1.0)
        self.assertEqual(result.phase, TrackingPhase.TRACKING)
        self.assertEqual(result.index_tip, Point2D(0.61, 0.29))
        self.assertTrue(result.action_ready)
        self.assertEqual(result.handedness, "Right")

    def test_detects_rightward_motion(self):
        tracker = HandTracker()
        tracker.update(make_observation(0.30, 0.50), timestamp=1.0)
        result = tracker.update(make_observation(0.55, 0.50), timestamp=1.1)
        self.assertGreater(result.velocity.x, 0.0)
        self.assertGreater(result.speed, 0.0)
        self.assertEqual(result.direction, MovementDirection.RIGHT)

    def test_detects_diagonal_motion(self):
        tracker = HandTracker()
        tracker.update(make_observation(0.30, 0.70), timestamp=1.0)
        result = tracker.update(make_observation(0.55, 0.45), timestamp=1.1)
        self.assertEqual(result.direction, MovementDirection.UP_RIGHT)

    def test_searching_before_first_detection(self):
        result = HandTracker().update(None, timestamp=1.0)
        self.assertEqual(result.phase, TrackingPhase.SEARCHING)
        self.assertIsNone(result.index_tip)
        self.assertFalse(result.action_ready)

    def test_missed_frame_suppresses_live_coordinate_immediately(self):
        tracker = HandTracker()
        tracker.update(make_observation(), timestamp=1.0)
        result = tracker.update(None, timestamp=1.05)
        self.assertEqual(result.phase, TrackingPhase.OCCLUDED)
        self.assertIsNone(result.index_tip)
        self.assertIsNotNone(result.last_known_position)
        self.assertFalse(result.action_ready)

    def test_reacquires_inside_grace_period(self):
        tracker = HandTracker(TrackingConfig(dropout_grace_period=0.35))
        tracker.update(make_observation(0.40, 0.50), timestamp=1.0)
        tracker.update(None, timestamp=1.10)
        result = tracker.update(make_observation(0.44, 0.50), timestamp=1.20)
        self.assertEqual(result.phase, TrackingPhase.TRACKING)
        self.assertTrue(result.action_ready)
        self.assertGreaterEqual(result.index_tip.x, 0.40)
        self.assertLessEqual(result.index_tip.x, 0.44)

    def test_prolonged_loss_resets_filter_before_reacquisition(self):
        tracker = HandTracker(TrackingConfig(dropout_grace_period=0.2))
        tracker.update(make_observation(0.20, 0.30), timestamp=1.0)
        lost = tracker.update(None, timestamp=1.25)
        self.assertEqual(lost.phase, TrackingPhase.LOST)
        self.assertIsNone(lost.index_tip)
        recovered = tracker.update(make_observation(0.80, 0.70), timestamp=1.30)
        self.assertEqual(recovered.index_tip, Point2D(0.80, 0.70))
        self.assertEqual(recovered.velocity, Point2D(0.0, 0.0))

    def test_clamps_coordinates_and_rejects_non_finite_values(self):
        tracker = HandTracker()
        result = tracker.update(make_observation(1.2, -0.2), timestamp=1.0)
        self.assertEqual(result.index_tip, Point2D(1.0, 0.0))
        with self.assertRaises(ValueError):
            HandTracker().update(make_observation(math.nan, 0.5), timestamp=1.0)

    def test_rejects_non_increasing_timestamps(self):
        tracker = HandTracker()
        tracker.update(make_observation(), timestamp=1.0)
        with self.assertRaises(ValueError):
            tracker.update(make_observation(), timestamp=1.0)

    def test_validates_tracking_configuration(self):
        with self.assertRaises(ValueError):
            TrackingConfig(dropout_grace_period=-0.1)
        with self.assertRaises(ValueError):
            TrackingConfig(min_cutoff=0.0)
        with self.assertRaises(ValueError):
            TrackingConfig(beta=float("nan"))


if __name__ == "__main__":
    unittest.main()
