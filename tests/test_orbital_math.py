import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from math_logic import OrbitalMath
from mission import SlimAutopilot


ROOT = Path(__file__).resolve().parents[1]


class OrbitalMathTests(unittest.TestCase):
    def test_wait_reaches_target_for_both_relative_rate_signs_and_wraparound(self):
        for initial, target, ship_rate, target_rate in (
            (90, 110, .003, .00005), (350, 10, .00005, .003),
            (10, 350, .003, .00005), (-10, 370, .003, .00005),
        ):
            with self.subTest(initial=initial, target=target, rate=ship_rate):
                phi, goal = math.radians(initial), math.radians(target)
                wait = OrbitalMath.time_to_phase_angle(goal, phi, ship_rate, target_rate)
                self.assertGreaterEqual(wait, 0)
                self.assertLess(wait, math.tau / abs(ship_rate-target_rate))
                error = math.remainder(phi + (target_rate-ship_rate)*wait - goal, math.tau)
                self.assertAlmostEqual(error, 0)

    def test_equal_rates_and_already_aligned_angles(self):
        self.assertEqual(OrbitalMath.time_to_phase_angle(math.tau, 0, 1, 1), 0)
        with self.assertRaisesRegex(ValueError, "Equal angular rates"):
            OrbitalMath.time_to_phase_angle(1, 0, 1, 1)

    def test_phase_uses_xz_plane_and_direction_of_motion(self):
        ship = (1, 0, 0)
        target = (0, 0, 2)
        self.assertAlmostEqual(OrbitalMath.current_phase_angle(ship, target, (-1, 0, 0)), math.pi/2)
        self.assertAlmostEqual(OrbitalMath.current_phase_angle(ship, target, (1, 0, 0)), 3*math.pi/2)

    def test_target_plane_can_be_tilted_and_ship_can_be_out_of_plane(self):
        # Target's plane is x-y, rather than the default x-z equator.
        self.assertAlmostEqual(OrbitalMath.current_phase_angle((1, 0, 7), (0, 2, 0), (-3, 1, 0)), math.pi/2)

    def test_degenerate_projection_and_radial_motion_are_rejected(self):
        for ship, target, velocity in (
            ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
            ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 0, 0), (0, 0, 1)),
            ((1, 0, 0), (1, 0, 0), (2, 0, 0)),
        ):
            with self.subTest(ship=ship, target=target):
                with self.assertRaises(ValueError):
                    OrbitalMath.current_phase_angle(ship, target, velocity)

    def test_plane_angle_distinguishes_coplanar_inclined_and_opposite_motion(self):
        r, v = (1, 0, 0), (0, 0, 1)
        self.assertAlmostEqual(OrbitalMath.relative_inclination(r, v, r, v), 0)
        self.assertAlmostEqual(OrbitalMath.relative_inclination(r, v, r, (0, 1, 0)), math.pi/2)
        self.assertAlmostEqual(OrbitalMath.relative_inclination(r, v, r, (0, 0, -1)), math.pi)
        with self.assertRaises(ValueError):
            OrbitalMath.relative_inclination(r, r, r, v)

    def test_hohmann_impulses_match_vis_viva_and_intercept_angle(self):
        mu, r1, r2 = 3.5316e12, 706324.6897133552, 12e6
        a = (r1+r2)/2
        dv1, dv2 = OrbitalMath.hohmann_delta_v(mu, r1, r2)
        self.assertAlmostEqual(dv1, math.sqrt(mu*(2/r1-1/a))-math.sqrt(mu/r1))
        self.assertAlmostEqual(dv2, math.sqrt(mu/r2)-math.sqrt(mu*(2/r2-1/a)))
        duration = OrbitalMath.hohmann_transfer_time(mu, r1, r2)
        phase = OrbitalMath.hohmann_phase_angle(mu, r1, r2)
        self.assertAlmostEqual(phase+OrbitalMath.angular_velocity(mu, r2)*duration, math.pi)
        self.assertEqual(OrbitalMath.hohmann_delta_v(mu, r1, r1), (0, 0))

    def test_burn_respects_rocket_equation_and_mass_flow(self):
        duration, fuel = OrbitalMath.burn_duration(837.0572366180314, 4653, 60000, 345, 1567)
        self.assertAlmostEqual(duration, 57.50688956679306)
        self.assertAlmostEqual(345*9.80665*math.log(4653/(4653-fuel)), 837.0572366180314)
        self.assertAlmostEqual(fuel/duration, 60000/(345*9.80665))
        self.assertEqual(OrbitalMath.burn_duration(0, 4653, 60000, 345, 0), (0, 0))

    def test_burn_cannot_spend_unreachable_fuel_or_protected_reserve(self):
        with self.assertRaisesRegex(ValueError, "reserve"):
            OrbitalMath.burn_duration(1800, 4653, 60000, 345, 1567)
        duration, _ = OrbitalMath.burn_duration(837, 4653, 60000, 345, 1567)
        half_thrust, _ = OrbitalMath.burn_duration(837, 4653, 30000, 345, 1567)
        self.assertAlmostEqual(half_thrust, 2*duration)

    def test_nonfinite_or_nonphysical_inputs_are_rejected(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    OrbitalMath.hohmann_delta_v(3.5e12, value, 12e6)
        for args in ((1, 0, 60000, 345, 0), (1, 4653, 0, 345, 1000), (float('nan'), 4653, 60000, 345, 1000)):
            with self.assertRaises(ValueError):
                OrbitalMath.burn_duration(*args)
        with self.assertRaises(ValueError):
            OrbitalMath.time_to_phase_angle(float('inf'), 0, 1, 2)

    def test_recorded_snapshot_reproduces_audited_geometry_and_burn(self):
        x = json.loads((ROOT/'docs/experiments/transfer-inspection.json').read_text(encoding='utf-8'))
        d, a = x['inspection'], x['analysis']
        r, v, rm, vm = (d[k] for k in ('position_inertial_m', 'velocity_inertial_m_s', 'mun_position_inertial_m', 'mun_velocity_inertial_m_s'))
        self.assertAlmostEqual(math.degrees(OrbitalMath.current_phase_angle(r, rm, vm)), a['phase_projected_in_mun_plane_deg'])
        self.assertAlmostEqual(math.degrees(OrbitalMath.relative_inclination(r, v, rm, vm)), a['relative_plane_angle_deg'])
        self.assertAlmostEqual(OrbitalMath.burn_duration(a['ideal_transfer_delta_v_m_s'], d['mass_kg'], 60000, 345, 1567)[0], a['ideal_burn_duration_s'])

    def test_offline_cli_marks_the_snapshot_as_an_unverified_candidate(self):
        result = subprocess.run([sys.executable, str(ROOT/'plan_transfer.py')], cwd=ROOT, check=True, text=True, capture_output=True)
        plan = json.loads(result.stdout)
        self.assertFalse(plan['encounter_verified'])
        self.assertFalse(plan['automatic_burn_authorized'])
        self.assertTrue(plan['requires_spatial_encounter_prediction'])
        self.assertGreater(plan['central_propellant_after_burn_kg'], 433)
        self.assertAlmostEqual(plan['candidate_node_ut']-plan['candidate_burn_start_ut'], plan['constant_thrust_burn_duration_s']/2)


class LegacyPlanningGuardsTests(unittest.TestCase):
    def test_inclined_snapshot_cannot_set_an_automatic_launch_time(self):
        x = json.loads((ROOT/'docs/experiments/transfer-inspection.json').read_text(encoding='utf-8'))['inspection']
        autopilot = SlimAutopilot.__new__(SlimAutopilot)  # No connection or controls.
        autopilot.kerbin = object()
        autopilot.vessel = SimpleNamespace(orbit=SimpleNamespace(body=autopilot.kerbin), position=lambda frame:x['position_inertial_m'], velocity=lambda frame:x['velocity_inertial_m_s'])
        autopilot.mun = SimpleNamespace(position=lambda frame:x['mun_position_inertial_m'], velocity=lambda frame:x['mun_velocity_inertial_m_s'])
        autopilot.mu = x['kerbin']['mu_m3_s2']
        autopilot.r1, autopilot.r2 = x['orbit']['semi_major_axis_m'], x['mun']['semi_major_axis_m']
        autopilot.orbital_rf = object()
        autopilot.logger = MagicMock()
        autopilot.launch_ut = 123  # A stale candidate must be cleared as well.
        with self.assertRaisesRegex(ValueError, '3D encounter'):
            autopilot.phase_1_compute_launch_time()
        self.assertIsNone(autopilot.launch_ut)

    def test_transfer_without_a_launch_estimate_does_not_create_a_node(self):
        autopilot = SlimAutopilot.__new__(SlimAutopilot)
        autopilot.launch_ut = None
        with self.assertRaisesRegex(ValueError, 'planning is incomplete'):
            autopilot.phase_2_transfer()

    def test_legacy_burn_rejects_a_late_window_or_insufficient_accessible_fuel(self):
        for liquid_fuel, oxidizer, now, error in ((180, 220, 90, 'start has passed'), (45, 55, 0, 'reserve')):
            with self.subTest(liquid_fuel=liquid_fuel, now=now):
                autopilot = SlimAutopilot.__new__(SlimAutopilot)
                autopilot.launch_ut = 100
                autopilot.mu, autopilot.r1, autopilot.r2 = 3.5316e12, 706324.69, 12e6
                engine = SimpleNamespace(part=SimpleNamespace(name='liquidEngine3.v2'), active=True, vacuum_specific_impulse=345,
                    propellants=[SimpleNamespace(name='LiquidFuel', total_resource_available=liquid_fuel), SimpleNamespace(name='Oxidizer', total_resource_available=oxidizer)])
                control = MagicMock()
                autopilot.vessel = SimpleNamespace(parts=SimpleNamespace(engines=[engine]), mass=4653, available_thrust=60000, control=control)
                autopilot.sc = SimpleNamespace(ut=now)
                autopilot.logger = MagicMock()
                with self.assertRaisesRegex(ValueError, error):
                    autopilot.phase_2_transfer()
                control.add_node.assert_not_called()


if __name__ == '__main__':
    unittest.main()
