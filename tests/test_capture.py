import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from capture import execute_capture
from math_logic import OrbitalMath


class CaptureMathTests(unittest.TestCase):
    def test_capture_speed_gives_the_target_orbital_energy(self):
        mu, rp, ra, eccentricity = 6.513839752e10, 267934, 300000, 1.32118
        delta = OrbitalMath.capture_delta_v(mu,rp,eccentricity,ra)
        incoming = math.sqrt(mu*(1+eccentricity)/rp)
        self.assertGreater(delta,0)
        self.assertAlmostEqual((incoming-delta)**2/2-mu/rp,-mu/(rp+ra))

    def test_parabolic_arrival_and_circular_target(self):
        mu, r = 6.513839752e10, 270000
        self.assertAlmostEqual(OrbitalMath.capture_delta_v(mu,r,1,r),(math.sqrt(2)-1)*math.sqrt(mu/r))

    def test_invalid_capture_parameters(self):
        for args in ((0,1,1,2),(1,0,1,2),(1,2,.9,3),(1,2,1,1),(float('nan'),1,1,2)):
            with self.assertRaises(ValueError):
                OrbitalMath.capture_delta_v(*args)


class CaptureControlTests(unittest.TestCase):
    def setUp(self):
        self.kerbin = SimpleNamespace(name='Kerbin')
        self.mun = SimpleNamespace(name='Mun',non_rotating_reference_frame='mun_inertial',gravitational_parameter=6.513839752e10,equatorial_radius=200000)
        self.orbit = SimpleNamespace(body=self.kerbin,periapsis=267900,periapsis_altitude=67900,time_to_periapsis=1000,eccentricity=1.32,radius=267900,apoapsis_altitude=100000,next_orbit=None,period=3700)
        self.ctrl = SimpleNamespace(nodes=[],throttle=0,sas=False)
        tank = SimpleNamespace(name='fuelTank',mass=1216.13,dry_mass=250)
        module = [SimpleNamespace(name='externalTankRound',resources=SimpleNamespace(amount=lambda name:{'LiquidFuel':9.9,'Oxidizer':12.1}[name])) for _ in range(4)]
        engines = [SimpleNamespace(part=SimpleNamespace(name=n),active=n=='liquidEngine3.v2') for n in ('liquidEngine3.v2','smallRadialEngine.v2','smallRadialEngine.v2')]
        self.speed = math.sqrt(self.mun.gravitational_parameter*(1+self.orbit.eccentricity)/self.orbit.periapsis)
        self.vessel = SimpleNamespace(control=self.ctrl,auto_pilot=MagicMock(),orbit=self.orbit,available_thrust=60000,thrust=0,mass=3619.13,
            resources=SimpleNamespace(amount=lambda name:1000),parts=SimpleNamespace(all=[tank]+module+[SimpleNamespace(name='other') for _ in range(50)],engines=engines),
            velocity=lambda frame:(self.speed,0,0),direction=lambda frame:(-1,0,0))
        self.sc = MagicMock()
        self.sc.bodies = {'Kerbin':self.kerbin,'Mun':self.mun}
        self.sc.active_vessel = self.vessel
        self.sc.maximum_rails_warp_factor = 6
        type(self.sc).ut = PropertyMock(side_effect=itertools.count(0,1))
        self.conn = SimpleNamespace(space_center=self.sc,krpc=SimpleNamespace(paused=True))
        self.log_patch = patch('capture.FlightLogger')
        self.logger_class = self.log_patch.start()
        self.log = self.logger_class.return_value
        self.log.error = None
        self.log.filename = Path(__file__).resolve().parents[1]/'logs'/'mock-capture.csv'
        self.addCleanup(self.log_patch.stop)
        self.encounter_patch = patch('capture.encounter',return_value={'periapsis_altitude_m':67900})
        self.encounter_patch.start()
        self.addCleanup(self.encounter_patch.stop)
        self.print_patch = patch('builtins.print')
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def test_preflight_rejects_wrong_body_before_taking_control(self):
        self.orbit.body = self.mun
        with self.assertRaisesRegex(ValueError,'Kerbin'):
            execute_capture(self.conn,self.vessel)
        self.vessel.auto_pilot.engage.assert_not_called()
        self.logger_class.assert_not_called()

    def test_unexpected_soi_change_aborts_and_pauses(self):
        self.log.log.side_effect = lambda **kwargs:setattr(self.orbit,'body',object())
        with patch('capture.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'Unexpected SOI'):
                execute_capture(self.conn,self.vessel)
        self.assertTrue(self.conn.krpc.paused)
        self.assertEqual(self.ctrl.throttle,0)
        self.log.close.assert_called_once()

    def test_too_little_time_to_periapsis_rejects_capture(self):
        self.orbit.time_to_periapsis = 20
        self.log.log.side_effect = lambda **kwargs:setattr(self.orbit,'body',self.mun)
        with patch('capture.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'Insufficient time'):
                execute_capture(self.conn,self.vessel)
        self.vessel.auto_pilot.engage.assert_not_called()
        self.assertTrue(self.conn.krpc.paused)

    def test_wrong_direction_keeps_throttle_zero_and_interrupt_is_cleaned_up(self):
        self.log.log.side_effect = lambda **kwargs:setattr(self.orbit,'body',self.mun)
        self.vessel.direction = lambda frame:(1,0,0)
        type(self.sc).ut = PropertyMock(side_effect=[0,0,1000])
        calls = 0
        def sleep(seconds):
            nonlocal calls
            if seconds == .05:
                calls += 1
                if calls == 2:
                    self.assertEqual(self.ctrl.throttle,0)
                    raise KeyboardInterrupt
        with patch('capture.time.sleep',side_effect=sleep):
            with self.assertRaises(KeyboardInterrupt):
                execute_capture(self.conn,self.vessel)
        self.assertTrue(self.conn.krpc.paused)
        self.assertEqual(self.vessel.auto_pilot.reference_frame,'mun_inertial')
        self.log.close.assert_called_once()

    def test_success_requires_actual_mun_entry_and_30_second_unpowered_verification(self):
        target_energy = -self.mun.gravitational_parameter/(self.orbit.periapsis+300000)
        def event(**kwargs):
            if kwargs.get('event') == 'capture_coast_start':
                self.orbit.body = self.mun
            if kwargs.get('event') == 'capture_burn_start':
                self.speed = math.sqrt(2*(target_energy+self.mun.gravitational_parameter/self.orbit.radius))
                self.orbit.eccentricity = (300000-self.orbit.periapsis)/(300000+self.orbit.periapsis)
            return {'speed':self.speed,'radius':self.orbit.radius,'body':'Mun'}
        self.log.log.side_effect = event
        with patch('capture.time.sleep'):
            result = execute_capture(self.conn,self.vessel)
        self.assertEqual(result['outcome'],'mun_captured')
        self.assertGreaterEqual(result['verification_duration_s'],30)
        self.assertLess(result['specific_energy_j_kg'],0)
        self.assertEqual(result['parts'],55)
        self.assertAlmostEqual(result['module_propellant_kg'],440)
        self.assertEqual(self.ctrl.throttle,0)
        self.assertFalse(self.ctrl.sas)
        self.assertTrue(self.conn.krpc.paused)
        self.sc.save.assert_called_once()
        self.log.log.assert_any_call(phase='mun_approach',event='actual_mun_soi_entry')


if __name__ == '__main__':
    unittest.main()
