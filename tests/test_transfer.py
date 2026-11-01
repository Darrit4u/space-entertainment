import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

from math_logic import OrbitalMath
from transfer import execute_transfer, fly_node


class LambertTests(unittest.TestCase):
    def test_published_vallado_example_in_si_units(self):
        # Official poliastro documentation, Curtis example 5.2.
        v0, v1 = OrbitalMath.lambert(398600.4418e9, (5000e3, 10000e3, 2100e3), (-14600e3, 2500e3, 7000e3), 3600)
        for actual, expected in zip(v0, (-5992.49503, 1925.36671, 3245.63805)):
            self.assertAlmostEqual(actual, expected, delta=.02)
        for actual, expected in zip(v1, (-3312.45851, -4196.61901, -385.28906)):
            self.assertAlmostEqual(actual, expected, delta=.02)

    def test_circular_quarter_orbit_and_long_arc(self):
        mu, radius = 3.5316e12, 700000
        speed = math.sqrt(mu/radius)
        period = math.tau*math.sqrt(radius**3/mu)
        for duration, long_path, sign in ((period/4, False, 1), (3*period/4, True, -1)):
            v0, v1 = OrbitalMath.lambert(mu, (radius,0,0), (0,0,radius), duration, long_path=long_path)
            self.assertAlmostEqual(v0[0], 0, delta=.001)
            self.assertAlmostEqual(v0[2], sign*speed, delta=.001)
            self.assertAlmostEqual(v1[0], -sign*speed, delta=.001)

    def test_invalid_and_collinear_geometry(self):
        for r0, r1, duration in (((0,0,0),(1,0,0),1), ((1,0,0),(-1,0,0),1), ((1,0,0),(0,0,1),0)):
            with self.assertRaises(ValueError):
                OrbitalMath.lambert(1, r0, r1, duration)


class TransferControlTests(unittest.TestCase):
    def setUp(self):
        self.body = SimpleNamespace(non_rotating_reference_frame='inertial')
        self.sc = MagicMock()
        self.sc.bodies = {'Kerbin':self.body, 'Mun':object()}
        self.ctrl = MagicMock()
        self.ctrl.nodes, self.ctrl.throttle, self.ctrl.sas = [], 0, False
        self.node = self.ctrl.add_node.return_value
        self.node.ut, self.node.delta_v = 200, 10
        self.node.remaining_burn_vector.return_value = (10,0,0)
        self.vessel = SimpleNamespace(control=self.ctrl, auto_pilot=MagicMock(), thrust=0, mass=4653, available_thrust=60000,
            orbit=SimpleNamespace(body=self.body), direction=lambda frame:(-1,0,0),
            parts=SimpleNamespace(all=[SimpleNamespace(name='fuelTank',mass=2250,dry_mass=250)]+[object() for _ in range(54)]))
        self.sc.active_vessel = self.vessel
        self.conn = SimpleNamespace(space_center=self.sc, krpc=SimpleNamespace(paused=True))
        self.plan = dict(ut=200,prograde=10,normal=0,radial=0)
        self.log_patch = patch('transfer.FlightLogger')
        self.log = self.log_patch.start().return_value
        self.log.error = None
        self.addCleanup(self.log_patch.stop)

    def test_bad_prediction_releases_controls_and_removes_own_node(self):
        with patch('transfer.encounter',return_value=None), patch('transfer.time.sleep'):
            with self.assertRaisesRegex(ValueError,'safe Mun encounter'):
                execute_transfer(self.conn,self.vessel,self.plan)
        self.assertEqual(self.ctrl.throttle,0)
        self.assertTrue(self.conn.krpc.paused)
        self.node.remove.assert_called_once()
        self.log.close.assert_called_once()

    def test_keyboard_interrupt_stops_thrust_and_closes_telemetry(self):
        with patch('transfer.encounter',return_value={'periapsis_altitude_m':50000}), patch('transfer.fly_node',side_effect=KeyboardInterrupt), patch('transfer.time.sleep'):
            with self.assertRaises(KeyboardInterrupt):
                execute_transfer(self.conn,self.vessel,self.plan)
        self.assertEqual(self.ctrl.throttle,0)
        self.assertTrue(self.conn.krpc.paused)
        self.vessel.auto_pilot.disengage.assert_called_once()
        self.log.close.assert_called_once()

    def test_wrong_thrust_direction_keeps_throttle_zero(self):
        type(self.sc).ut = PropertyMock(side_effect=[0,0,200])
        warp = PropertyMock(return_value=0)
        type(self.sc).rails_warp_factor = warp
        def sleep(seconds):
            if seconds==.05:
                self.assertFalse(any(call.args for call in warp.mock_calls))
                self.assertEqual(self.ctrl.throttle,0)
                raise KeyboardInterrupt
        with patch('transfer.encounter',return_value={'periapsis_altitude_m':50000}), patch('transfer.time.sleep',side_effect=sleep):
            with self.assertRaises(KeyboardInterrupt):
                execute_transfer(self.conn,self.vessel,self.plan)
        self.assertTrue(self.conn.krpc.paused)
        self.log.log.assert_any_call(event='transfer_interrupted')

    def test_mun_maneuver_uses_the_overridden_inertial_frame(self):
        mun=SimpleNamespace(non_rotating_reference_frame='mun_inertial')
        self.vessel.orbit.body=mun
        type(self.sc).ut=PropertyMock(side_effect=[0,0,200])
        def direction(frame):
            self.assertEqual(frame,'mun_inertial')
            return (-1,0,0)
        self.vessel.direction=direction
        with patch('transfer.time.sleep',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                fly_node(self.conn,self.vessel,self.node,self.log,math.inf,body=mun,prefix='lower_periapsis')
        self.assertEqual(self.vessel.auto_pilot.reference_frame,'mun_inertial')
        self.node.remaining_burn_vector.assert_called_with('mun_inertial')


if __name__ == '__main__':
    unittest.main()
