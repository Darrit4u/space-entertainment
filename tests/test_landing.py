import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from landing import run_landing


class LandingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.body = SimpleNamespace(surface_gravity=1.63)
        self.ctrl = SimpleNamespace(throttle=0.0, sas=True)
        self.ap = MagicMock()
        self.ap.error = 0.5
        engine = lambda name, active: SimpleNamespace(
            part=SimpleNamespace(name=name), active=active, has_fuel=True,
        )
        self.vessel = SimpleNamespace(
            control=self.ctrl, auto_pilot=self.ap,
            orbit=SimpleNamespace(body=self.body, periapsis_altitude=14500),
            situation=SimpleNamespace(name="orbiting"),
            thrust=0.0, mass=3086, available_thrust=60000,
            available_torque=((10000, 10000, 10000), (-10000, -10000, -10000)),
            surface_velocity_reference_frame="velocity_frame",
            resources=SimpleNamespace(amount=lambda name: 1000),
            parts=SimpleNamespace(
                engines=[engine("liquidEngine3.v2", True),
                         engine("smallRadialEngine.v2", False),
                         engine("smallRadialEngine.v2", False)],
                all=[SimpleNamespace(name="externalTankRound", resources=SimpleNamespace(amount=lambda name: 10)) for _ in range(4)],
                decouplers=[SimpleNamespace(decoupled=False)], reaction_wheels=[object()],
            ),
        )
        self.sc = MagicMock()
        self.sc.active_vessel = self.vessel
        self.sc.bodies = {"Mun": self.body}
        type(self.sc).ut = PropertyMock(side_effect=itertools.count(100, 0.1))
        self.conn = SimpleNamespace(space_center=self.sc, krpc=SimpleNamespace(paused=False))
        self.logger_patch = patch("landing.FlightLogger")
        self.log = self.logger_patch.start().return_value
        self.addCleanup(self.logger_patch.stop)
        self.log.error = None
        self.print_patch = patch("builtins.print")
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def test_check_only_never_takes_control(self):
        self.conn.krpc.paused = True
        self.assertEqual(run_landing(self.conn), "preflight_passed")
        self.ap.engage.assert_not_called()
        self.ap.disengage.assert_not_called()
        self.assertTrue(self.ctrl.sas)
        self.log.close.assert_called_once()

    def test_rejects_wrong_body_and_nonzero_throttle(self):
        self.vessel.orbit.body = object()
        with self.assertRaisesRegex(ValueError, "Mun orbit"):
            run_landing(self.conn, prepare=True)
        self.vessel.orbit.body = self.body
        self.ctrl.throttle = 0.3
        with self.assertRaisesRegex(ValueError, "throttle to zero"):
            run_landing(self.conn, prepare=True)
        self.assertEqual(self.ctrl.throttle, 0.3)  # Rejection does not take control.
        self.ap.engage.assert_not_called()

    def test_rejects_missing_propellant_and_insufficient_thrust(self):
        self.vessel.parts.engines[1].has_fuel = False
        with self.assertRaisesRegex(ValueError, "propellant"):
            run_landing(self.conn)
        self.vessel.parts.engines[1].has_fuel = True
        self.vessel.available_thrust = 1
        with self.assertRaisesRegex(ValueError, "insufficient"):
            run_landing(self.conn)

    def test_paused_game_rejected_before_engaging(self):
        self.conn.krpc.paused = True
        with self.assertRaisesRegex(ValueError, "Resume KSP"):
            run_landing(self.conn, prepare=True)
        self.ap.engage.assert_not_called()

    def test_success_releases_controls(self):
        with patch("landing.time.sleep"), patch("landing.time.monotonic", side_effect=[0, 0.1, 0.2, 0.3]):
            self.assertEqual(run_landing(self.conn, prepare=True), "ready_for_deorbit")
        self.assertEqual(self.ctrl.throttle, 0)
        self.assertTrue(self.ctrl.sas)
        self.ap.disengage.assert_called_once()
        self.log.close.assert_called_once()

    def test_keyboard_interrupt_stops_thrust_and_autopilot(self):
        self.ap.error = 90
        with patch("landing.time.sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_landing(self.conn, prepare=True)
        self.assertEqual(self.ctrl.throttle, 0)
        self.ap.disengage.assert_called_once()
        self.log.log.assert_any_call(event="landing_interrupted")
        self.log.close.assert_called_once()

    def test_orientation_timeout_is_bounded(self):
        with patch("landing.time.monotonic", side_effect=[0, 1]):
            with self.assertRaisesRegex(TimeoutError, "wall-clock"):
                run_landing(self.conn, prepare=True, timeout=0.5)
        self.ap.disengage.assert_called_once()
        self.assertEqual(self.ctrl.throttle, 0)

    def test_stalled_ut_aborts(self):
        type(self.sc).ut = PropertyMock(return_value=100)
        with patch("landing.time.monotonic", side_effect=[0, 6]):
            with self.assertRaisesRegex(TimeoutError, "stopped advancing"):
                run_landing(self.conn, prepare=True)
        self.ap.disengage.assert_called_once()

    def test_autopilot_failure_still_cleans_up(self):
        self.ap.engage.side_effect = RuntimeError("RPC error")
        with self.assertRaisesRegex(RuntimeError, "RPC error"):
            run_landing(self.conn, prepare=True)
        self.ap.disengage.assert_called_once()
        self.assertEqual(self.ctrl.throttle, 0)
        self.log.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
