import unittest
from types import SimpleNamespace

from visionpilot.desktop import (
    DesktopControlConfig,
    DesktopControlEngine,
    ScreenMapper,
)
from visionpilot.gestures import Gesture, GestureEvent, GestureUpdate
from visionpilot.tracking import Point2D


class FakeMouse:
    def __init__(self, size=(1000, 800)):
        self._size = size
        self.moves = []
        self.clicks = 0
        self.presses = 0
        self.releases = 0
        self.scrolls = []

    @property
    def screen_size(self):
        return self._size

    def move_to(self, x, y):
        self.moves.append((x, y))

    def click_left(self):
        self.clicks += 1

    def press_left(self):
        self.presses += 1

    def release_left(self):
        self.releases += 1

    def scroll(self, dx, dy):
        self.scrolls.append((dx, dy))


def tracking(x=0.5, y=0.5, ready=True):
    return SimpleNamespace(action_ready=ready, index_tip=Point2D(x, y))


def gesture(
    active=Gesture.INDEX_POINT,
    candidate=None,
    event=GestureEvent.NONE,
    previous=Gesture.NONE,
    duration=0.0,
    scroll=0.0,
    detected=True,
    ready=True,
):
    candidate = active if candidate is None else candidate
    return GestureUpdate(
        gesture=active,
        candidate=candidate,
        event=event,
        previous_gesture=previous,
        hand_detected=detected,
        stable_frames=3,
        pinch_duration=duration,
        scroll_delta_y=scroll,
        timestamp=1.0,
    ) if ready else GestureUpdate(
        gesture=active,
        candidate=candidate,
        event=event,
        previous_gesture=previous,
        hand_detected=detected,
        stable_frames=1,
        pinch_duration=duration,
        scroll_delta_y=scroll,
        timestamp=1.0,
    )


class ScreenMapperTests(unittest.TestCase):
    def test_maps_center_and_safe_edges(self):
        mapper = ScreenMapper(1000, 800, edge_margin=0.05)
        self.assertEqual(mapper.map(Point2D(0.5, 0.5)), (500, 400))
        x0, y0 = mapper.map(Point2D(-2.0, -1.0))
        x1, y1 = mapper.map(Point2D(2.0, 2.0))
        self.assertGreater(x0, 0)
        self.assertGreater(y0, 0)
        self.assertLess(x1, 999)
        self.assertLess(y1, 799)

    def test_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            ScreenMapper(1, 800)


class DesktopControlTests(unittest.TestCase):
    def setUp(self):
        self.mouse = FakeMouse()
        self.engine = DesktopControlEngine(self.mouse)

    def arm(self):
        self.assertTrue(self.engine.toggle_enabled())

    def test_starts_disarmed_and_ignores_actions(self):
        self.engine.update(tracking(), gesture())
        self.assertFalse(self.engine.enabled)
        self.assertEqual(self.mouse.moves, [])
        self.assertEqual(self.mouse.clicks, 0)

    def test_index_point_moves_cursor_when_armed(self):
        self.arm()
        self.engine.update(tracking(0.5, 0.5), gesture())
        self.assertEqual(self.mouse.moves[-1], (500, 400))

    def test_quick_pinch_release_clicks_once(self):
        self.arm()
        self.engine.update(
            tracking(0.5, 0.5),
            gesture(Gesture.PINCH, event=GestureEvent.STARTED, duration=0.05),
        )
        self.engine.update(
            tracking(0.5, 0.5),
            gesture(
                Gesture.INDEX_POINT, event=GestureEvent.CHANGED,
                previous=Gesture.PINCH, duration=0.20,
            ),
        )
        self.assertEqual(self.mouse.clicks, 1)
        self.assertEqual(self.mouse.presses, 0)

    def test_long_pinch_starts_drag_and_release_releases_button(self):
        self.arm()
        self.engine.update(
            tracking(0.5, 0.5),
            gesture(Gesture.PINCH, event=GestureEvent.STARTED, duration=0.1),
        )
        self.engine.update(
            tracking(0.6, 0.6),
            gesture(Gesture.PINCH, duration=0.5),
        )
        self.assertEqual(self.mouse.presses, 1)
        self.assertEqual(self.mouse.moves[-1], (600, 600))
        self.engine.update(
            tracking(0.7, 0.7),
            gesture(
                Gesture.INDEX_POINT, event=GestureEvent.CHANGED,
                previous=Gesture.PINCH, duration=0.8,
            ),
        )
        self.assertEqual(self.mouse.releases, 1)
        self.assertEqual(self.mouse.clicks, 0)

    def test_open_palm_pauses_and_index_point_resumes(self):
        self.arm()
        self.engine.update(tracking(), gesture(Gesture.OPEN_PALM))
        self.assertIn("PAUSED", self.engine.status)
        previous_moves = len(self.mouse.moves)
        self.engine.update(tracking(0.6, 0.6), gesture(Gesture.INDEX_POINT))
        self.assertEqual(len(self.mouse.moves), previous_moves + 1)
        self.assertIn("cursor tracking", self.engine.status)

    def test_two_finger_scroll_is_converted_to_wheel_ticks(self):
        self.arm()
        self.engine.update(tracking(), gesture(Gesture.TWO_FINGER, scroll=-0.10))
        self.assertTrue(self.mouse.scrolls)
        self.assertGreater(self.mouse.scrolls[-1][1], 0)

    def test_tracking_loss_disarms_and_releases_held_button(self):
        self.arm()
        self.engine.update(
            tracking(), gesture(Gesture.PINCH, duration=0.6)
        )
        self.assertEqual(self.mouse.presses, 1)
        self.engine.update(
            tracking(ready=False),
            gesture(Gesture.NONE, detected=False, ready=False),
        )
        self.assertFalse(self.engine.enabled)
        self.assertEqual(self.mouse.releases, 1)
        self.assertIn("TRACKING LOST", self.engine.status)

    def test_emergency_stop_disarms_and_releases_button(self):
        self.arm()
        self.engine.update(tracking(), gesture(Gesture.PINCH, duration=0.6))
        self.engine.emergency_stop()
        self.assertFalse(self.engine.enabled)
        self.assertEqual(self.mouse.releases, 1)
        self.assertIn("EMERGENCY STOP", self.engine.status)

    def test_manual_disarm_releases_button(self):
        self.arm()
        self.engine.update(tracking(), gesture(Gesture.PINCH, duration=0.6))
        self.assertFalse(self.engine.toggle_enabled())
        self.assertEqual(self.mouse.releases, 1)

    def test_control_error_fails_closed(self):
        class BrokenMouse(FakeMouse):
            def move_to(self, x, y):
                raise RuntimeError("test backend failure")
        engine = DesktopControlEngine(BrokenMouse())
        engine.toggle_enabled()
        engine.update(tracking(), gesture())
        self.assertFalse(engine.enabled)
        self.assertIn("CONTROL ERROR", engine.status)

    def test_shutdown_releases_button(self):
        self.arm()
        self.engine.update(tracking(), gesture(Gesture.PINCH, duration=0.6))
        self.engine.close()
        self.assertFalse(self.engine.enabled)
        self.assertEqual(self.mouse.releases, 1)


if __name__ == "__main__":
    unittest.main()
