import unittest

from visionpilot.gestures import Gesture, GestureClassifier, GestureConfig, GestureEvent
from visionpilot.landmarks import HandObservation, Landmark


def hand_pose(kind: str, *, index_y=0.30, middle_y=0.25):
    points = [Landmark(0.50, 0.58, 0.0) for _ in range(21)]
    points[0] = Landmark(0.55, 0.90, 0.0)
    specs = {
        "index": (5, 6, 8, 0.40, 0.65, 0.50, index_y),
        "middle": (9, 10, 12, 0.50, 0.63, 0.46, middle_y),
        "ring": (13, 14, 16, 0.60, 0.65, 0.50, 0.30),
        "pinky": (17, 18, 20, 0.70, 0.67, 0.53, 0.32),
    }
    extended = {
        "open": {"index", "middle", "ring", "pinky"},
        "two": {"index", "middle"},
        "point": {"index"},
        "fist": set(),
        "pinch": {"index"},
    }[kind]
    for name, (mcp, pip, tip, x, mcp_y, pip_y, tip_y) in specs.items():
        points[mcp] = Landmark(x, mcp_y, 0.0)
        points[pip] = Landmark(x, pip_y, 0.0)
        points[tip] = (
            Landmark(x, tip_y, 0.0)
            if name in extended
            else Landmark(x + 0.06, pip_y + 0.08, 0.0)
        )
    if kind == "open":
        points[2] = Landmark(0.34, 0.60, 0.0)
        points[3] = Landmark(0.27, 0.52, 0.0)
        points[4] = Landmark(0.18, 0.43, 0.0)
    elif kind == "pinch":
        points[2] = Landmark(0.34, 0.60, 0.0)
        points[3] = Landmark(0.36, 0.43, 0.0)
        points[4] = Landmark(0.405, index_y + 0.005, 0.0)
    else:
        points[2] = Landmark(0.34, 0.60, 0.0)
        points[3] = Landmark(0.38, 0.64, 0.0)
        points[4] = Landmark(0.43, 0.62, 0.0)
    return HandObservation(tuple(points), "Right", 0.98)


class GestureClassifierTests(unittest.TestCase):
    def stable(self, classifier, observation, start=1.0):
        result = None
        for timestamp in (start, start + 0.04, start + 0.08):
            result = classifier.update(observation, timestamp=timestamp)
        return result

    def test_open_palm(self):
        result = self.stable(GestureClassifier(), hand_pose("open"))
        self.assertEqual(result.gesture, Gesture.OPEN_PALM)
        self.assertEqual(result.event, GestureEvent.STARTED)

    def test_two_finger_scroll_gesture(self):
        result = self.stable(GestureClassifier(), hand_pose("two"))
        self.assertEqual(result.gesture, Gesture.TWO_FINGER)

    def test_index_point(self):
        result = self.stable(GestureClassifier(), hand_pose("point"))
        self.assertEqual(result.gesture, Gesture.INDEX_POINT)

    def test_fist(self):
        result = self.stable(GestureClassifier(), hand_pose("fist"))
        self.assertEqual(result.gesture, Gesture.FIST)

    def test_pinch(self):
        result = self.stable(GestureClassifier(), hand_pose("pinch"))
        self.assertEqual(result.gesture, Gesture.PINCH)

    def test_pinch_thresholds_allow_bent_index_without_confusing_fist(self):
        config = GestureConfig(
            pinch_enter_ratio=0.45,
            pinch_release_ratio=0.55,
            pinch_min_index_angle=90.0,
        )
        result = self.stable(GestureClassifier(config), hand_pose("pinch"))
        self.assertEqual(result.gesture, Gesture.PINCH)
        self.assertEqual(GestureClassifier().config.pinch_enter_ratio, 0.45)

    def test_pinch_with_partially_bent_index_finger(self):
        observation = hand_pose("pinch")
        points = list(observation.landmarks)
        points[8] = Landmark(0.48, 0.44, 0.0)
        points[4] = Landmark(0.485, 0.44, 0.0)
        result = self.stable(GestureClassifier(), HandObservation(tuple(points)))
        self.assertEqual(result.gesture, Gesture.PINCH)

    def test_debounce_prevents_single_frame_switch(self):
        result = GestureClassifier().update(hand_pose("open"), timestamp=1.0)
        self.assertEqual(result.gesture, Gesture.NONE)
        self.assertEqual(result.event, GestureEvent.NONE)

    def test_active_gesture_ends_immediately_when_hand_disappears(self):
        classifier = GestureClassifier()
        self.stable(classifier, hand_pose("pinch"))
        result = classifier.update(None, timestamp=1.12)
        self.assertEqual(result.gesture, Gesture.NONE)
        self.assertEqual(result.event, GestureEvent.ENDED)
        self.assertFalse(result.action_ready)

    def test_pinch_hold_duration(self):
        classifier = GestureClassifier()
        self.stable(classifier, hand_pose("pinch"))
        result = classifier.update(hand_pose("pinch"), timestamp=1.20)
        self.assertGreaterEqual(result.pinch_duration, 0.0)
        self.assertEqual(result.gesture, Gesture.PINCH)

    def test_pinch_release_preserves_hold_duration(self):
        classifier = GestureClassifier()
        self.stable(classifier, hand_pose("pinch"))
        classifier.update(hand_pose("point"), timestamp=1.12)
        classifier.update(hand_pose("point"), timestamp=1.16)
        result = classifier.update(hand_pose("point"), timestamp=1.20)
        self.assertEqual(result.gesture, Gesture.INDEX_POINT)
        self.assertEqual(result.event, GestureEvent.CHANGED)
        self.assertGreaterEqual(result.pinch_duration, 0.10)

    def test_action_readiness_drops_as_pose_changes(self):
        classifier = GestureClassifier()
        self.stable(classifier, hand_pose("pinch"))
        self.assertTrue(classifier.update(hand_pose("pinch"), timestamp=1.12).action_ready)
        transitioning = classifier.update(hand_pose("point"), timestamp=1.16)
        self.assertEqual(transitioning.gesture, Gesture.PINCH)
        self.assertFalse(transitioning.action_ready)

    def test_two_finger_scroll_delta_positive_when_hand_moves_down(self):
        classifier = GestureClassifier()
        self.stable(classifier, hand_pose("two", index_y=0.30, middle_y=0.25))
        result = classifier.update(hand_pose("two", index_y=0.36, middle_y=0.31), timestamp=1.12)
        self.assertGreater(result.scroll_delta_y, 0.0)

    def test_scroll_delta_is_capped(self):
        classifier = GestureClassifier()
        self.stable(classifier, hand_pose("two"))
        result = classifier.update(hand_pose("two", index_y=0.70, middle_y=0.65), timestamp=1.12)
        self.assertLessEqual(abs(result.scroll_delta_y), GestureConfig().max_scroll_delta)

    def test_rejects_bad_timestamp(self):
        classifier = GestureClassifier()
        classifier.update(hand_pose("open"), timestamp=1.0)
        with self.assertRaises(ValueError):
            classifier.update(hand_pose("open"), timestamp=1.0)

    def test_rejects_invalid_landmark_count(self):
        classifier = GestureClassifier()
        with self.assertRaises(ValueError):
            classifier.update(HandObservation(tuple([Landmark(0, 0, 0)] * 20)), timestamp=1.0)

    def test_config_threshold_order(self):
        with self.assertRaises(ValueError):
            GestureConfig(pinch_enter_ratio=0.5, pinch_release_ratio=0.4)


if __name__ == "__main__":
    unittest.main()
