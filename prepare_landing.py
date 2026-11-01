"""Lower the captured Mun orbit to 15 x 25 km and check the landing interface."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import krpc

from landing import run_landing
from logger import FlightLogger
from math_logic import OrbitalMath
from transfer import fly_node


def prepare_landing_orbit(conn, vessel, timeout=900):
    sc, ctrl, ap = conn.space_center, vessel.control, vessel.auto_pilot
    mun = sc.bodies['Mun']
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Preparation timeout must be finite and positive')
    if not conn.krpc.paused or vessel.orbit.body != mun or vessel.orbit.eccentricity >= 1 or ctrl.nodes or len(vessel.parts.all) != 55:
        raise ValueError('Start preparation from the paused intact captured Mun orbit')
    # Existing landing preflight is read-only; preserve its existing engine checks.
    if run_landing(conn) != 'preflight_passed':
        raise ValueError('Landing configuration preflight failed')
    tank = next(p for p in vessel.parts.all if p.name=='fuelTank')
    tanks = [p for p in vessel.parts.all if p.name=='externalTankRound']
    if tank.mass-tank.dry_mass <= 433 or abs(sum((p.resources.amount('LiquidFuel')+p.resources.amount('Oxidizer'))*5 for p in tanks)-440) > .01:
        raise ValueError('Landing propellant reserves are unavailable')
    recorder = FlightLogger(vessel,sc)
    recorder.start()
    deadline = time.monotonic()+timeout
    old_roll, old_sas = ap.target_roll, ctrl.sas
    nodes = []
    burns = []
    success = False
    try:
        for name, timing, burn_radius, opposite_before, opposite_after in (
            ('lower_periapsis', 'time_to_apoapsis', 'apoapsis', 'periapsis', mun.equatorial_radius+15000),
            ('lower_apoapsis', 'time_to_periapsis', 'periapsis', 'apoapsis', mun.equatorial_radius+25000),
        ):
            orbit = vessel.orbit
            delta = OrbitalMath.apsis_change_delta_v(mun.gravitational_parameter,getattr(orbit,burn_radius),getattr(orbit,opposite_before),opposite_after)
            if delta >= 0:
                raise ValueError('This preparation expects two retrograde lowering burns')
            node = ctrl.add_node(sc.ut+getattr(orbit,timing),prograde=delta)
            nodes.append(node)
            predicted = node.orbit
            if predicted.body != mun or predicted.periapsis_altitude < 14000 or predicted.eccentricity >= 1:
                raise ValueError('Lowering node predicts an unsafe orbit')
            record = {'phase':name,'ut':node.ut,'prograde_m_s':delta,'predicted_periapsis_m':predicted.periapsis_altitude,'predicted_apoapsis_m':predicted.apoapsis_altitude,'mass_before_kg':vessel.mass}
            recorder.log(phase=name+'_planning',event='lowering_node_planned')
            fly_node(conn,vessel,node,recorder,deadline,body=mun,prefix=name)
            node.remove()
            nodes.remove(node)
            record.update(mass_after_kg=vessel.mass,actual_periapsis_m=vessel.orbit.periapsis_altitude,actual_apoapsis_m=vessel.orbit.apoapsis_altitude)
            burns.append(record)
            if not 14000 < vessel.orbit.periapsis_altitude < 16000:
                raise RuntimeError('Actual lowered periapsis is outside 14-16 km')
        ap.disengage()
        ctrl.sas = False
        recorder.attitude_control_active = False
        start = last_ut = sc.ut
        progress = time.monotonic()
        recorder.log(phase='low_orbit_verify',event='lowered_orbit_coast_start')
        while True:
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError('Low-orbit preparation timed out')
            if recorder.error is not None:
                raise RuntimeError('Preparation telemetry failed') from recorder.error
            ut = sc.ut
            if ut < last_ut:
                raise RuntimeError('KSP time moved backwards')
            if ut > last_ut:
                progress = now
            elif now-progress > 5:
                raise TimeoutError('KSP time stopped advancing')
            last_ut = ut
            orbit = vessel.orbit
            if conn.krpc.paused or sc.active_vessel != vessel or orbit.body != mun or len(vessel.parts.all) != 55:
                raise RuntimeError('Pause, vessel, body or parts changed during verification')
            if not 14000 < orbit.periapsis_altitude < 16000 or not 24000 < orbit.apoapsis_altitude < 26000 or orbit.eccentricity >= 1 or orbit.next_orbit is not None:
                raise RuntimeError('Final orbit is outside the landing preparation corridor')
            if ctrl.throttle != 0 or vessel.thrust > 1 or ctrl.sas or tank.mass-tank.dry_mass < 433:
                raise RuntimeError('Preparation verification requires unpowered motion and preserved reserve')
            if ut-start >= 30:
                break
            time.sleep(.1)
        module_fuel = sum((p.resources.amount('LiquidFuel')+p.resources.amount('Oxidizer'))*5 for p in tanks)
        if abs(module_fuel-440) > .01:
            raise RuntimeError('Module propellant was consumed during preparation')
        conn.krpc.paused = True
        readiness = run_landing(conn)
        if readiness != 'preflight_passed':
            raise RuntimeError('Final landing preflight failed')
        final = recorder.log(phase='landing_orbit_ready',event='landing_orbit_confirmed')
        name = 'SLIM_Landing_Ready_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        sc.save(name)
        success = True
        return {'outcome':'landing_orbit_ready','save':name,'telemetry':recorder.filename.relative_to(Path(__file__).parent).as_posix(),'burns':burns,'final':final,'parts':len(vessel.parts.all),'central_propellant_kg':tank.mass-tank.dry_mass,'module_propellant_kg':module_fuel,'verification_duration_s':ut-start,'landing_preflight':readiness,'limitation':'No deorbit, braking-stage separation, descent or touchdown performed in this trial'}
    except BaseException as error:
        if recorder.error is None:
            recorder.log(event='preparation_interrupted' if isinstance(error,KeyboardInterrupt) else 'preparation_error')
        raise
    finally:
        try:
            ctrl.throttle = 0
            sc.rails_warp_factor = sc.physics_warp_factor = 0
            if not conn.krpc.paused:
                time.sleep(.2)
            conn.krpc.paused = True
        finally:
            try:
                try:
                    ap.disengage()
                finally:
                    ap.target_roll = old_roll
                    ctrl.sas = False if success else old_sas
                    for node in nodes:
                        node.remove()
            finally:
                recorder.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True,help='New result JSON; automatically loads SLIM_Mun_Captured_v1')
    args = parser.parse_args()
    with args.result.open('x',encoding='utf-8') as output:
        source=Path('saves/SLIM_Mun_Captured_v1.sfs')
        result={'baseline':'SLIM_Mun_Captured_v1','baseline_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'code_sha256_lf':{n:hashlib.sha256(Path(n).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for n in ('prepare_landing.py','math_logic.py','transfer.py','landing.py','logger.py')}}
        previous=set(Path('logs').glob('flight_*.csv'))
        c=None
        try:
            expected=json.loads(Path('docs/experiments/mun-capture-summary.json').read_text(encoding='utf-8'))['selected_checkpoint']
            if result['baseline_sha256'] != expected['sha256']:
                raise ValueError('Captured-orbit source checksum changed')
            c=krpc.connect(name='SLIM prepare low Mun orbit',rpc_port=5000,stream_port=5001)
            c.space_center.load('SLIM_Mun_Captured_v1');c.krpc.paused=True;time.sleep(.2)
            vessel=c.space_center.active_vessel
            if abs(vessel.mass-3366.953125)>.1:
                raise ValueError('Captured-orbit initial mass mismatch')
            result.update(prepare_landing_orbit(c,vessel))
        except BaseException as error:
            result['error']=repr(error);traceback.print_exc()
        finally:
            if c is not None:
                try:
                    v=c.space_center.active_vessel
                    result['cleanup_state']={'paused':c.krpc.paused,'throttle':v.control.throttle,'thrust_N':v.thrust,'parts':len(v.parts.all),'body':v.orbit.body.name,'mass_kg':v.mass,'sas':v.control.sas}
                except Exception as error:
                    result['cleanup_read_error']=repr(error)
                c.close()
            result['logs']=[p.as_posix() for p in set(Path('logs').glob('flight_*.csv'))-previous]
            json.dump(result,output,ensure_ascii=False,indent=2);output.write('\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if result.get('outcome')!='landing_orbit_ready':
            raise SystemExit(1)
