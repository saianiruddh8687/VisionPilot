import tempfile
import unittest
from pathlib import Path

from visionpilot.calibration import (
    CALIBRATION_TARGETS,
    CalibrationError,
    CalibrationPair,
    CalibrationProfile,
    CalibrationSession,
    CalibratedScreenMapper,
    fit_calibration,
    load_profile,
    save_profile,
)
from visionpilot.tracking import Point2D


class CalibrationFitTests(unittest.TestCase):
    def identity_pairs(self):
        return [CalibrationPair(camera=target, screen=target) for _, target in CALIBRATION_TARGETS]

    def test_fits_identity_homography(self):
        profile = fit_calibration(self.identity_pairs(), 1920, 1080)
        self.assertLess(profile.rms_error, 1e-8)
        mapped = profile.transform(Point2D(0.23, 0.71))
        self.assertAlmostEqual(mapped.x, 0.23, places=6)
        self.assertAlmostEqual(mapped.y, 0.71, places=6)

    def test_fits_offset_and_scale_mapping(self):
        pairs = [
            CalibrationPair(
                camera=Point2D(0.1 + target.x * 0.8, 0.1 + target.y * 0.8),
                screen=target,
            )
            for _label, target in CALIBRATION_TARGETS
        ]
        profile = fit_calibration(pairs, 1920, 1080)
        mapped = profile.transform(Point2D(0.1 + 0.4 * 0.8, 0.1 + 0.6 * 0.8))
        self.assertAlmostEqual(mapped.x, 0.4, places=5)
        self.assertAlmostEqual(mapped.y, 0.6, places=5)

    def test_rejects_too_few_points(self):
        with self.assertRaises(CalibrationError):
            fit_calibration(self.identity_pairs()[:3], 1920, 1080)

    def test_rejects_degenerate_points(self):
        pairs = [
            CalibrationPair(Point2D(v, v), Point2D(v, v))
            for v in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
        with self.assertRaises(CalibrationError):
            fit_calibration(pairs, 1920, 1080)

    def test_profile_round_trip(self):
        profile = fit_calibration(self.identity_pairs(), 1920, 1080)
        restored = CalibrationProfile.from_dict(profile.to_dict())
        self.assertEqual(restored, profile)
        self.assertEqual(restored.camera_index, 0)
        self.assertEqual(restored.camera_width, 1280)
        self.assertTrue(restored.mirror_preview)

    def test_save_and_load_profile(self):
        profile = fit_calibration(self.identity_pairs(), 1920, 1080)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            save_profile(profile, path)
            self.assertEqual(load_profile(path), profile)


class CalibrationSessionTests(unittest.TestCase):
    def test_requires_stable_samples_before_capture(self):
        session = CalibrationSession(1920, 1080, min_samples=5, max_samples=8)
        session.observe(Point2D(0.1, 0.1))
        with self.assertRaises(CalibrationError):
            session.capture()

    def test_captures_all_five_targets_and_binds_camera_metadata(self):
        session = CalibrationSession(
            1920, 1080, min_samples=5, max_samples=8,
            camera_index=1, camera_width=640, camera_height=480, mirror_preview=False,
        )
        for _label, target in CALIBRATION_TARGETS:
            for _ in range(5):
                session.observe(target)
            self.assertTrue(session.can_capture)
            completed = session.capture()
        self.assertTrue(completed)
        self.assertTrue(session.complete)
        self.assertEqual(session.progress, 5)
        self.assertIsNotNone(session.profile)
        self.assertEqual(session.profile.camera_index, 1)
        self.assertEqual((session.profile.camera_width, session.profile.camera_height), (640, 480))
        self.assertFalse(session.profile.mirror_preview)
        mapped = session.profile.transform(Point2D(0.4, 0.6))
        self.assertAlmostEqual(mapped.x, 0.4, places=5)
        self.assertAlmostEqual(mapped.y, 0.6, places=5)

    def test_unstable_samples_are_not_accepted(self):
        session = CalibrationSession(1920, 1080, min_samples=5, max_samples=8)
        for i in range(8):
            session.observe(Point2D(0.2 + (0.1 if i % 2 else 0), 0.2))
        self.assertFalse(session.can_capture)

    def test_missing_detection_clears_stability_buffer(self):
        session = CalibrationSession(1920, 1080, min_samples=5, max_samples=8)
        for _ in range(5):
            session.observe(Point2D(0.2, 0.2))
        session.observe(None)
        self.assertEqual(session.sample_count, 0)
        self.assertFalse(session.can_capture)

    def test_degenerate_final_capture_resets_session_for_retry(self):
        session = CalibrationSession(1920, 1080, min_samples=5, max_samples=8)
        for _label, _target in CALIBRATION_TARGETS:
            for _ in range(5):
                session.observe(Point2D(0.5, 0.5))
            if session.progress < 4:
                session.capture()
        for _ in range(5):
            session.observe(Point2D(0.5, 0.5))
        with self.assertRaises(CalibrationError):
            session.capture()
        self.assertEqual(session.progress, 0)
        self.assertEqual(session.current_target[0], "TOP LEFT")

    def test_cancel_stops_session(self):
        session = CalibrationSession(1920, 1080, min_samples=5, max_samples=8)
        session.cancel()
        self.assertTrue(session.cancelled)
        self.assertIsNone(session.current_target)
        session.observe(Point2D(0.2, 0.2))
        self.assertEqual(session.sample_count, 0)


class CalibratedMapperTests(unittest.TestCase):
    def test_maps_to_screen_and_clamps_edges(self):
        pairs = [CalibrationPair(camera=t, screen=t) for _, t in CALIBRATION_TARGETS]
        profile = fit_calibration(pairs, 1920, 1080)
        mapper = CalibratedScreenMapper(1920, 1080, profile, edge_margin=0.02)
        self.assertEqual(mapper.map(Point2D(0.5, 0.5)), (960, 540))
        x, y = mapper.map(Point2D(-2, 2))
        self.assertGreater(x, 0)
        self.assertLess(x, 1919)
        self.assertGreater(y, 0)
        self.assertLess(y, 1079)

    def test_calibration_targets_expand_to_safe_screen_area(self):
        pairs = [CalibrationPair(camera=t, screen=t) for _, t in CALIBRATION_TARGETS]
        profile = fit_calibration(pairs, 1920, 1080)
        mapper = CalibratedScreenMapper(1920, 1080, profile, edge_margin=0.005)

        top_left = mapper.map(Point2D(0.05, 0.05))
        bottom_right = mapper.map(Point2D(0.95, 0.95))

        self.assertEqual(top_left, (round(0.005 * 1919), round(0.005 * 1079)))
        self.assertEqual(bottom_right, (round(0.995 * 1919), round(0.995 * 1079)))

    def test_rejects_changed_screen_resolution(self):
        profile = fit_calibration(
            [CalibrationPair(camera=t, screen=t) for _, t in CALIBRATION_TARGETS],
            1920, 1080,
        )
        with self.assertRaises(CalibrationError):
            CalibratedScreenMapper(1280, 720, profile)


if __name__ == "__main__":
    unittest.main()
