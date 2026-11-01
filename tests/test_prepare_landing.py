import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from math_logic import OrbitalMath
from prepare_landing import prepare_landing_orbit


class ApsisMathTests(unittest.TestCase):
    def test_both_lowering_burns_reach_the_requested_opposite_apsis(self):
        mu = 6.513839752e10
        for radius, before, after in ((300256,267920,215000),(215000,300256,225000)):
            delta = OrbitalMath.apsis_change_delta_v(mu,radius,before,after)
            self.assertLess(delta,0)
            speed = math.sqrt(mu*(2/radius-2/(radius+before)))+delta
            energy = speed*speed/2-mu/radius
            self.assertAlmostEqual(2*(-mu/(2*energy))-radius,after)
        self.assertEqual(OrbitalMath.apsis_change_delta_v(mu,215000,225000,225000),0)

    def test_invalid_apsis_inputs(self):
        for radius in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                OrbitalMath.apsis_change_delta_v(1,radius,1,2)


class PrepareLandingTests(unittest.TestCase):
    def setUp(self):
        self.mun = SimpleNamespace(equatorial_radius=200000,gravitational_parameter=6.513839752e10)
        self.orbit = SimpleNamespace(body=self.mun,eccentricity=.057,apoapsis=300256,periapsis=267920,apoapsis_altitude=100256,periapsis_altitude=67920,time_to_apoapsis=1800,time_to_periapsis=1600,next_orbit=None)
        self.ctrl = MagicMock()
        self.ctrl.nodes,self.ctrl.throttle,self.ctrl.sas = [],0,False
        self.tank = SimpleNamespace(name='fuelTank',mass=963.95,dry_mass=250)
        module = [SimpleNamespace(name='externalTankRound',resources=SimpleNamespace(amount=lambda name:{'LiquidFuel':9.9,'Oxidizer':12.1}[name])) for _ in range(4)]
        self.vessel = SimpleNamespace(control=self.ctrl,auto_pilot=MagicMock(),orbit=self.orbit,thrust=0,mass=3366.95,parts=SimpleNamespace(all=[self.tank]+module+[SimpleNamespace(name='other') for _ in range(50)]))
        self.sc = MagicMock()
        self.sc.bodies = {'Mun':self.mun}
        self.sc.active_vessel = self.vessel
        type(self.sc).ut = PropertyMock(side_effect=itertools.count(0,1))
        self.conn = SimpleNamespace(space_center=self.sc,krpc=SimpleNamespace(paused=True))
        self.nodes = []
        def add_node(ut,prograde):
            node = SimpleNamespace(ut=ut,orbit=SimpleNamespace(body=self.mun,periapsis_altitude=15000,apoapsis_altitude=100000 if not self.nodes else 25000,eccentricity=.1),remove=MagicMock())
            self.nodes.append(node)
            return node
        self.ctrl.add_node.side_effect = add_node
        self.log_patch = patch('prepare_landing.FlightLogger')
        self.log = self.log_patch.start().return_value
        self.log.error = None
        self.log.filename = Path(__file__).resolve().parents[1]/'logs'/'mock-preparation.csv'
        self.log.log.return_value = {'periapsis':15000,'apoapsis':25000}
        self.addCleanup(self.log_patch.stop)
        self.preflight_patch = patch('prepare_landing.run_landing',return_value='preflight_passed')
        self.preflight = self.preflight_patch.start()
        self.addCleanup(self.preflight_patch.stop)

    def test_two_burns_use_mun_frame_and_finish_with_a_paused_landing_preflight(self):
        order = []
        def fly(conn,vessel,node,recorder,deadline,**kwargs):
            self.assertIs(kwargs['body'],self.mun)
            order.append(kwargs['prefix'])
            self.conn.krpc.paused = False
            self.orbit.periapsis = 215000
            self.orbit.periapsis_altitude = 15000
            if len(order)==2:
                self.orbit.apoapsis,self.orbit.apoapsis_altitude = 225000,25000
        with patch('prepare_landing.fly_node',side_effect=fly),patch('prepare_landing.time.sleep'):
            result=prepare_landing_orbit(self.conn,self.vessel)
        self.assertEqual(order,['lower_periapsis','lower_apoapsis'])
        self.assertEqual(result['outcome'],'landing_orbit_ready')
        self.assertGreaterEqual(result['verification_duration_s'],30)
        self.assertAlmostEqual(result['module_propellant_kg'],440)
        self.assertEqual(self.preflight.call_count,2)
        self.assertTrue(self.conn.krpc.paused)
        self.assertFalse(self.ctrl.sas)
        self.sc.save.assert_called_once()
        self.assertTrue(all(n.remove.call_count==1 for n in self.nodes))

    def test_keyboard_interrupt_releases_controls_and_own_node(self):
        with patch('prepare_landing.fly_node',side_effect=KeyboardInterrupt),patch('prepare_landing.time.sleep'):
            with self.assertRaises(KeyboardInterrupt):
                prepare_landing_orbit(self.conn,self.vessel)
        self.assertEqual(self.ctrl.throttle,0)
        self.assertTrue(self.conn.krpc.paused)
        self.nodes[0].remove.assert_called_once()
        self.log.close.assert_called_once()
        self.sc.save.assert_not_called()

    def test_unsafe_predicted_periapsis_is_rejected_before_burning(self):
        original = self.ctrl.add_node.side_effect
        def unsafe(*args,**kwargs):
            n=original(*args,**kwargs)
            n.orbit.periapsis_altitude=1000
            return n
        self.ctrl.add_node.side_effect=unsafe
        with patch('prepare_landing.fly_node') as fly,patch('prepare_landing.time.sleep'):
            with self.assertRaisesRegex(ValueError,'unsafe orbit'):
                prepare_landing_orbit(self.conn,self.vessel)
        fly.assert_not_called()
        self.assertTrue(self.conn.krpc.paused)
        self.nodes[0].remove.assert_called_once()


if __name__ == '__main__':
    unittest.main()
