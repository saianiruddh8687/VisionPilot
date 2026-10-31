from types import SimpleNamespace
import unittest

from visionpilot.landmarks import extract_hand_observation


def make_result(label="Left", count=21):
    points = [SimpleNamespace(x=i / 20, y=(20 - i) / 20, z=0.01 * i) for i in range(count)]
    category = SimpleNamespace(category_name=label, score=0.97)
    return SimpleNamespace(hand_landmarks=[points], handedness=[[category]])


class LandmarkExtractionTests(unittest.TestCase):
    def test_extracts_all_21_landmarks(self):
        observation = extract_hand_observation(make_result())
        self.assertIsNotNone(observation)
        self.assertEqual(len(observation.landmarks), 21)
        self.assertEqual(observation.handedness, "Left")
        self.assertAlmostEqual(observation.handedness_score, 0.97)

    def test_no_hand_returns_none(self):
        result = SimpleNamespace(hand_landmarks=[], handedness=[])
        self.assertIsNone(extract_hand_observation(result))

    def test_mirrored_input_corrects_handedness(self):
        observation = extract_hand_observation(make_result("Left"), mirrored_input=True)
        self.assertEqual(observation.handedness, "Right")

    def test_rejects_unexpected_landmark_count(self):
        with self.assertRaises(ValueError):
            extract_hand_observation(make_result(count=20))


if __name__ == "__main__":
    unittest.main()
