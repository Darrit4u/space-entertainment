import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from landing import brake_to_descent, descent_acceleration, descend_to_low_altitude, finish_touchdown, run_landing


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
        self.log.vessel = self.vessel
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

    def test_brake_mode_stops_at_descent_handoff(self):
        with patch("landing.time.sleep"), patch("landing.brake_to_descent", return_value="ready_for_vertical_descent") as braking:
            self.assertEqual(run_landing(self.conn, brake=True), "ready_for_vertical_descent")
        braking.assert_called_once_with(self.conn, self.vessel, self.log, 900.0)
        self.assertEqual(self.ctrl.throttle, 0)
        self.assertTrue(self.conn.krpc.paused)
        self.assertEqual(self.sc.rails_warp_factor, 0)
        self.ap.disengage.assert_called_once()

    def test_brake_error_restores_zero_throttle_and_pauses(self):
        with patch("landing.time.sleep"), patch("landing.brake_to_descent", side_effect=RuntimeError("reserve depleted")):
            with self.assertRaisesRegex(RuntimeError, "reserve depleted"):
                run_landing(self.conn, brake=True)
        self.assertEqual(self.ctrl.throttle, 0)
        self.assertTrue(self.conn.krpc.paused)
        self.log.close.assert_called_once()

    def test_full_landing_resumes_and_passes_the_detached_vessel_between_stages(self):
        self.conn.krpc.paused = True
        self.vessel.parts.all += [SimpleNamespace(name="sGripPad") for _ in range(4)]
        self.vessel.parts.all += [SimpleNamespace(name="other") for _ in range(47)]
        module = SimpleNamespace(**vars(self.vessel))
        module.parts = SimpleNamespace(all=[object() for _ in range(51)])
        order = []

        def braking(*args, **kwargs):
            self.assertFalse(self.conn.krpc.paused)
            self.assertFalse(kwargs["checkpoint"])
            self.log.vessel = module
            order.append("brake")

        def descent(conn, vessel, recorder, timeout, **kwargs):
            self.assertIs(vessel, module)
            self.assertFalse(self.conn.krpc.paused)
            self.assertFalse(kwargs["checkpoint"])
            order.append("descend")

        def touchdown(conn, vessel, recorder, timeout):
            self.assertIs(vessel, module)
            order.append("touchdown")
            self.log.phase = "landed"
            return "landed"

        with patch("landing.time.sleep"), patch("landing.brake_to_descent", side_effect=braking), patch("landing.descend_to_low_altitude", side_effect=descent), patch("landing.finish_touchdown", side_effect=touchdown):
            self.assertEqual(run_landing(self.conn, full=True), "landed")
        self.assertEqual(order, ["brake", "descend", "touchdown"])
        self.assertTrue(self.conn.krpc.paused)
        self.assertFalse(self.ctrl.sas)
        self.assertEqual(self.ctrl.throttle, 0)

    def test_full_landing_stops_if_the_module_lost_parts(self):
        self.vessel.parts.all += [SimpleNamespace(name="sGripPad") for _ in range(4)]
        self.vessel.parts.all += [SimpleNamespace(name="other") for _ in range(47)]
        with patch("landing.time.sleep"), patch("landing.brake_to_descent"), patch("landing.descend_to_low_altitude") as descent:
            with self.assertRaisesRegex(RuntimeError, "51 intact"):
                run_landing(self.conn, full=True)
        descent.assert_not_called()
        self.assertTrue(self.conn.krpc.paused)
        self.assertEqual(self.ctrl.throttle, 0)

    def test_invalid_braking_timeout_does_not_start(self):
        with self.assertRaises(ValueError):
            run_landing(self.conn, brake=True, brake_timeout=float("nan"))
        self.ap.engage.assert_not_called()

    def test_empty_terrier_switches_to_module_and_saves_handoff(self):
        self.body.reference_frame = "surface"
        self.body.non_rotating_reference_frame = "inertial"
        self.body.gravitational_parameter = 6.513839752e10
        self.body.equatorial_radius = 200000
        self.vessel.orbit.radius = 224000
        self.vessel.orbit.periapsis_altitude = -5001
        self.vessel.vacuum_specific_impulse = 345
        self.vessel.dry_mass = 2213
        self.vessel.velocity = lambda frame: (0, 500, 0)
        flight = SimpleNamespace(surface_altitude=8000, horizontal_speed=100, vertical_speed=-5, speed=100)
        self.vessel.flight = lambda frame: flight
        self.vessel.parts.controlling = SimpleNamespace(vessel=self.vessel)
        radial = self.vessel.parts.engines[1:]

        def separate():
            self.vessel.parts.engines = radial
            self.vessel.available_thrust = 32000
            self.vessel.mass = 1793
            self.vessel.dry_mass = 1353

        decoupler = MagicMock()
        decoupler.decouple.side_effect = separate
        self.vessel.parts.decouplers = [decoupler]

        def phase_transition(**kwargs):
            if kwargs.get("phase") == "main_braking":
                self.vessel.available_thrust = 0
            if kwargs.get("phase") == "landing_braking":
                flight.horizontal_speed = 1
            return {"event": kwargs.get("event")}

        self.log.log.side_effect = phase_transition
        with tempfile.TemporaryDirectory(prefix="braking-test-", dir=Path(__file__).parent) as directory:
            self.log.filename = Path(directory) / "flight.csv"
            with patch("landing.time.sleep"):
                self.assertEqual(brake_to_descent(self.conn, self.vessel, self.log, 30), "ready_for_vertical_descent")
            self.assertTrue(self.log.filename.with_suffix(".handoff.json").exists())
        decoupler.decouple.assert_called_once()
        self.assertTrue(all(e.active for e in radial))
        self.assertEqual(self.ctrl.throttle, 0)
        self.assertTrue(self.conn.krpc.paused)
        self.sc.save.assert_called_once()

        # The identical handoff condition must continue in integrated mode.
        self.conn.krpc.paused = False
        self.sc.save.reset_mock()
        with patch("landing.time.sleep"):
            self.assertEqual(brake_to_descent(self.conn, self.vessel, self.log, 30, checkpoint=False), "ready_for_vertical_descent")
        self.assertFalse(self.conn.krpc.paused)
        self.sc.save.assert_not_called()

    def test_descent_mode_accepts_detached_module_and_cleans_up(self):
        self.body.reference_frame = "surface"
        self.vessel.situation.name = "sub_orbital"
        self.vessel.parts.engines = self.vessel.parts.engines[1:]
        for engine in self.vessel.parts.engines:
            engine.active = True
        self.vessel.parts.decouplers = []
        self.vessel.flight = lambda frame: SimpleNamespace(surface_altitude=6000, horizontal_speed=5, vertical_speed=-7)
        with patch("landing.descend_to_low_altitude", return_value="ready_for_touchdown") as descent, patch("landing.time.sleep"):
            self.assertEqual(run_landing(self.conn, descend=True), "ready_for_touchdown")
        descent.assert_called_once_with(self.conn, self.vessel, self.log, 600.0)
        self.assertEqual(self.ctrl.throttle, 0)
        self.assertTrue(self.conn.krpc.paused)
        self.ap.disengage.assert_called_once()

    def test_descent_rejects_intact_configuration(self):
        self.body.reference_frame = "surface"
        self.vessel.situation.name = "sub_orbital"
        self.vessel.flight = lambda frame: SimpleNamespace(surface_altitude=6000, horizontal_speed=5, vertical_speed=-7)
        with self.assertRaisesRegex(ValueError, "two active radial"):
            run_landing(self.conn, descend=True)
        self.ap.engage.assert_not_called()

    def test_low_altitude_handoff_uses_rotated_body_velocity(self):
        self.body.reference_frame = "body_surface"
        self.body.gravitational_parameter = 6.513839752e10
        self.vessel.orbit.radius = 200040
        self.vessel.surface_reference_frame = "vessel_surface"
        self.vessel.velocity = lambda frame: (-1, 0, 0) if frame == "body_surface" else (0, 0, 0)
        self.vessel.flight = lambda frame: SimpleNamespace(surface_altitude=40, vertical_speed=-1)
        self.sc.transform_direction.return_value = (-1, 0.01, 0)
        self.log.log.return_value = {"altitude": 39.8, "vertical_speed": -1.3}
        with tempfile.TemporaryDirectory(prefix="descent-test-", dir=Path(__file__).parent) as directory:
            self.log.filename = Path(directory) / "flight.csv"
            with patch("landing.time.sleep"):
                self.assertEqual(descend_to_low_altitude(self.conn, self.vessel, self.log, 30), "ready_for_touchdown")
            self.assertTrue(self.log.filename.with_suffix(".handoff.json").exists())
        self.sc.transform_direction.assert_called_once_with((-1, 0, 0), "body_surface", "vessel_surface")
        self.assertEqual(self.ctrl.throttle, 0)
        self.assertTrue(self.conn.krpc.paused)

        self.conn.krpc.paused = False
        self.ctrl.throttle = .3
        self.sc.save.reset_mock()
        with patch("landing.time.sleep"):
            self.assertEqual(descend_to_low_altitude(self.conn, self.vessel, self.log, 30, checkpoint=False), "ready_for_touchdown")
        self.assertEqual(self.ctrl.throttle, .3)
        self.assertFalse(self.conn.krpc.paused)
        self.sc.save.assert_not_called()

    def test_vertical_thrust_is_not_cut_by_heading_error(self):
        self.body.reference_frame = "body_surface"
        self.body.gravitational_parameter = 6.513839752e10
        self.vessel.orbit.radius = 201000
        self.vessel.surface_reference_frame = "vessel_surface"
        self.vessel.velocity = lambda frame: (-40, 0, 0)
        self.vessel.direction = lambda frame: (1, 0, 0)
        self.vessel.flight = lambda frame: SimpleNamespace(surface_altitude=1000, vertical_speed=-40)
        self.vessel.dry_mass = 2213
        self.sc.transform_direction.return_value = (-40, 0, 0)
        self.ap.error = 180
        with patch("landing.time.sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                descend_to_low_altitude(self.conn, self.vessel, self.log, 30)
        self.assertGreater(self.ctrl.throttle, 0)
        self.assertEqual(self.log.thrust_up_component, 1)

    def test_fast_descent_aborts_above_the_static_safety_floor(self):
        self.body.reference_frame = "body_surface"
        self.body.gravitational_parameter = 6.513839752e10
        self.vessel.orbit.radius = 200100
        self.vessel.surface_reference_frame = "vessel_surface"
        self.vessel.velocity = lambda frame: (-60, 0, 0)
        self.vessel.flight = lambda frame: SimpleNamespace(surface_altitude=100, vertical_speed=-60)
        self.sc.transform_direction.return_value = (-60, 0, 0)
        with self.assertRaisesRegex(RuntimeError, "emergency altitude"):
            descend_to_low_altitude(self.conn, self.vessel, self.log, 30)
        self.assertEqual(self.ctrl.throttle, 0)

    def test_touchdown_mode_accepts_low_detached_module_and_restores_tuning(self):
        self.body.reference_frame = "surface"
        self.vessel.situation.name = "sub_orbital"
        self.vessel.parts.engines = self.vessel.parts.engines[1:]
        for engine in self.vessel.parts.engines:
            engine.active = True
        self.vessel.parts.decouplers = []
        self.vessel.parts.all += [SimpleNamespace(name="sGripPad") for _ in range(4)]
        self.vessel.flight = lambda frame: SimpleNamespace(surface_altitude=38, horizontal_speed=.05, vertical_speed=-1.76)
        self.ap.time_to_peak = (3, 3, 3)
        with patch("landing.finish_touchdown", return_value="landed"), patch("landing.time.sleep"):
            self.assertEqual(run_landing(self.conn, touchdown=True), "landed")
        self.assertEqual(self.ctrl.throttle, 0)
        self.assertTrue(self.conn.krpc.paused)
        self.assertEqual(self.ap.time_to_peak, (3, 3, 3))


class TouchdownVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="touchdown-test-", dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.body = SimpleNamespace(reference_frame="body", gravitational_parameter=6.513839752e10)
        self.flight = SimpleNamespace(surface_altitude=3.9, vertical_speed=-.2)
        self.ctrl = SimpleNamespace(throttle=0)
        self.ap = MagicMock()
        self.vessel = SimpleNamespace(
            control=self.ctrl, auto_pilot=self.ap, surface_reference_frame="surface",
            reference_frame="vessel", situation=SimpleNamespace(name="sub_orbital"),
            orbit=SimpleNamespace(body=self.body, radius=202600),
            parts=SimpleNamespace(all=[object() for _ in range(51)]),
            thrust=0, mass=1567, dry_mass=1353, available_thrust=32000,
            flight=lambda frame:self.flight,
            velocity=lambda frame:(-.2, 0, 0),
            angular_velocity=lambda frame:(0, 0, 0),
        )
        self.sc = MagicMock()
        self.sc.bodies = {"Mun":self.body}
        self.sc.active_vessel = self.vessel
        type(self.sc).ut = PropertyMock(side_effect=itertools.count(100, .1))
        self.conn = SimpleNamespace(space_center=self.sc, krpc=SimpleNamespace(paused=False))
        self.log = MagicMock()
        self.log.error = None
        self.log.filename = Path(self.directory.name) / "flight.csv"
        self.pad_up = -1
        self.sc.transform_direction.side_effect = lambda vector, source, target: (self.pad_up,0,0) if source=="vessel" else ((0,0,0) if self.vessel.situation.name=="landed" else (-.2,0,0))

        def event(**kwargs):
            if kwargs.get("event")=="contact_detected":
                # Keep reaction-wheel damping through the first contact.
                self.ap.disengage.assert_not_called()
            if kwargs.get("event")=="engines_off_for_turn":
                self.vessel.situation.name="landed"
                self.flight.surface_altitude=.9
                self.flight.vertical_speed=0
            return {"event":kwargs.get("event")}

        self.log.log.side_effect = event
        self.print_patch = patch("builtins.print")
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def test_contact_requires_settling_and_preserves_all_parts(self):
        with patch("landing.time.sleep"):
            self.assertEqual(finish_touchdown(self.conn,self.vessel,self.log,30),"landed")
        self.assertEqual(len(self.vessel.parts.all),51)
        self.assertTrue(self.conn.krpc.paused)
        self.sc.save.assert_called_once()
        self.ap.disengage.assert_called_once()
        self.assertFalse(self.ctrl.sas)
        self.assertEqual(self.ap.time_to_peak, (1, 1, 1))
        self.assertTrue(self.log.filename.with_suffix(".touchdown.json").exists())
        import json
        result = json.loads(self.log.filename.with_suffix(".touchdown.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(result["passive_stable_duration_s"], 30)
        self.log.log.assert_any_call(phase="landed",event="touchdown_confirmed")
        self.log.log.assert_any_call(phase="touchdown_passive",event="attitude_control_released")

    def test_instability_after_control_release_is_not_accepted(self):
        original_event = self.log.log.side_effect

        def event(**kwargs):
            if kwargs.get("event") == "attitude_control_released":
                self.pad_up = 1
            return original_event(**kwargs)

        self.log.log.side_effect = event
        with patch("landing.time.sleep"), patch("landing.time.monotonic", side_effect=itertools.count(0, .01)):
            with self.assertRaises(TimeoutError):
                finish_touchdown(self.conn, self.vessel, self.log, 2)
        self.ap.disengage.assert_called_once()
        self.sc.save.assert_not_called()

    def test_wrong_contact_side_is_not_accepted_as_landing(self):
        self.pad_up=1
        with patch("landing.time.sleep"),patch("landing.time.monotonic",side_effect=[0,.1,.2,10]):
            with self.assertRaises(TimeoutError):
                finish_touchdown(self.conn,self.vessel,self.log,2)
        self.sc.save.assert_not_called()

    def test_uncertain_contact_speed_is_not_accepted(self):
        # A long gap before detecting contact makes the speed bound too large.
        type(self.sc).ut = PropertyMock(side_effect=itertools.count(100, 4))
        with patch("landing.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "Contact speed exceeded"):
                finish_touchdown(self.conn, self.vessel, self.log, 30)
        self.sc.save.assert_not_called()

    def test_lost_part_is_rejected(self):
        self.log.log.side_effect=lambda **kwargs:self.vessel.parts.all.pop()
        with self.assertRaisesRegex(RuntimeError,"part was lost"):
            finish_touchdown(self.conn,self.vessel,self.log,30)
        self.sc.save.assert_not_called()

    def test_premature_contact_is_rejected(self):
        self.vessel.situation.name="landed"
        with self.assertRaisesRegex(RuntimeError,"Unexpected contact"):
            finish_touchdown(self.conn,self.vessel,self.log,30)
        self.sc.save.assert_not_called()


class DescentControlTests(unittest.TestCase):
    def test_constant_descent_requires_gravity_compensation(self):
        target, acceleration = descent_acceleration(40, (-1, 0, 0), 1.6, 0)
        self.assertEqual(target, -1)
        self.assertEqual(acceleration, (1.6, 0, 0))

    def test_high_altitude_slow_descent_allows_freefall(self):
        target, acceleration = descent_acceleration(6000, (-7, 2, 3), 1.6, 0)
        self.assertEqual(target, -50)
        self.assertEqual(acceleration, (0, 0, 0))

    def test_overspeed_brakes_and_horizontal_force_opposes_motion(self):
        import math
        _, acceleration = descent_acceleration(1000, (-80, 80, -60), 1.6, 0)
        self.assertGreater(acceleration[0], 1.6)
        self.assertLess(acceleration[1], 0)
        self.assertGreater(acceleration[2], 0)
        self.assertLessEqual(math.hypot(*acceleration[1:]), 0.2 * acceleration[0] + 1e-10)

    def test_profile_slows_and_derivative_damps_acceleration(self):
        targets = [descent_acceleration(h, (-1, 0, 0), 1.6, 0)[0] for h in (6000, 1000, 100, 40)]
        self.assertEqual(targets, sorted(targets))
        _, rising = descent_acceleration(40, (-1, 0, 0), 1.6, 2)
        _, falling = descent_acceleration(40, (-1, 0, 0), 1.6, -2)
        self.assertLess(rising[0], falling[0])


if __name__ == "__main__":
    unittest.main()
