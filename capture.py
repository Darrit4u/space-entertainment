"""Coast to actual Mun SOI entry and capture near periapsis; no low-orbit descent."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import krpc

from logger import FlightLogger
from math_logic import OrbitalMath
from transfer import encounter


def execute_capture(conn, vessel, timeout=900):
    sc, ctrl, ap = conn.space_center, vessel.control, vessel.auto_pilot
    kerbin, mun = sc.bodies['Kerbin'], sc.bodies['Mun']
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Capture timeout must be finite and positive')
    if not conn.krpc.paused or vessel.orbit.body != kerbin or ctrl.nodes or len(vessel.parts.all) != 55:
        raise ValueError('Start capture from the paused intact transfer handoff at Kerbin')
    if ctrl.throttle != 0 or vessel.thrust > 1 or vessel.available_thrust <= 0:
        raise ValueError('Zero throttle/thrust and an available engine are required')
    engines = vessel.parts.engines
    if len(engines) != 3 or not all(e.active == (e.part.name == 'liquidEngine3.v2') for e in engines):
        raise ValueError('Capture requires only the active Terrier')
    tanks = [p for p in vessel.parts.all if p.name == 'externalTankRound']
    if len(tanks) != 4 or abs(sum((p.resources.amount('LiquidFuel')+p.resources.amount('Oxidizer'))*5 for p in tanks)-440) > .01:
        raise ValueError('Protect all four full landing-module tanks')
    tank = next(p for p in vessel.parts.all if p.name == 'fuelTank')
    if tank.mass-tank.dry_mass <= 433 or vessel.resources.amount('ElectricCharge') <= 0:
        raise ValueError('Insufficient central propellant or electric charge')
    prediction = encounter(vessel.orbit, mun)
    if prediction is None or not 30000 < prediction['periapsis_altitude_m'] < 80000:
        raise ValueError('Expected a safe upcoming Mun encounter')
    old_sas, old_roll = ctrl.sas, ap.target_roll
    recorder = FlightLogger(vessel, sc)
    recorder.start()
    deadline = time.monotonic()+timeout
    last_ut = sc.ut
    progress = last_print = time.monotonic()
    phase = 'coast'
    entry = plan = None
    verify_start = None
    success = False
    try:
        ctrl.throttle = 0
        ctrl.sas = False
        ap.disengage()
        recorder.attitude_control_active = False
        sc.physics_warp_factor = 0
        conn.krpc.paused = False
        recorder.log(phase='mun_coast', event='capture_coast_start')
        while True:
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError('Capture exceeded its wall-clock timeout')
            if recorder.error is not None:
                raise RuntimeError('Capture telemetry failed') from recorder.error
            if conn.krpc.paused or sc.active_vessel != vessel or len(vessel.parts.all) != 55:
                raise RuntimeError('Pause, active vessel or part count changed during capture')
            if tank.mass-tank.dry_mass < 433 or vessel.available_thrust <= 0:
                raise RuntimeError('Protected central reserve or available thrust was lost')
            ut = sc.ut
            if ut < last_ut:
                raise RuntimeError('KSP time moved backwards during capture')
            if ut > last_ut:
                progress = now
            elif now-progress > 5:
                raise TimeoutError('KSP time stopped advancing during capture')
            last_ut = ut
            orbit = vessel.orbit
            body = orbit.body
            if phase == 'coast' and body == mun:
                sc.rails_warp_factor = 0
                rp = orbit.periapsis
                if not 30000 < orbit.periapsis_altitude < 80000 or not 0 < orbit.time_to_periapsis < 30000:
                    raise RuntimeError('Actual Mun entry has an unsafe periapsis or timing')
                dv = OrbitalMath.capture_delta_v(mun.gravitational_parameter, rp, orbit.eccentricity, mun.equatorial_radius+100000)
                duration, fuel = OrbitalMath.burn_duration(dv, vessel.mass, vessel.available_thrust, 345, tank.mass-tank.dry_mass-433)
                entry = {'ut':ut, 'body':body.name, 'periapsis_altitude_m':orbit.periapsis_altitude, 'eccentricity':orbit.eccentricity, 'time_to_periapsis_s':orbit.time_to_periapsis, 'mass_kg':vessel.mass}
                plan = {'periapsis_ut':ut+orbit.time_to_periapsis, 'ideal_delta_v_m_s':dv, 'ideal_duration_s':duration, 'ideal_propellant_kg':fuel, 'target_apoapsis_altitude_m':100000, 'target_specific_energy_j_kg':-mun.gravitational_parameter/(rp+mun.equatorial_radius+100000)}
                plan['burn_start_ut'] = plan['periapsis_ut']-duration/2
                if plan['burn_start_ut'] <= ut+60:
                    raise RuntimeError('Insufficient time after SOI entry for capture preparation')
                phase = 'approach'
                ap.reference_frame = mun.non_rotating_reference_frame
                ap.target_roll = float('nan')
                ap.engage()
                recorder.attitude_control_active = True
                recorder.log(phase='mun_approach', event='actual_mun_soi_entry')
                print(f"Actual Mun entry: periapsis {entry['periapsis_altitude_m']/1000:.3f} km, capture estimate {dv:.2f} m/s", flush=True)
            if phase == 'coast':
                if body != kerbin:
                    raise RuntimeError('Unexpected SOI transition before Mun')
                remaining = orbit.time_to_soi_change
                if not math.isfinite(remaining) or remaining <= 0 or encounter(orbit, mun) is None:
                    raise RuntimeError('Upcoming Mun encounter was lost')
                wanted = 6 if remaining > 5000 else (5 if remaining > 800 else (3 if remaining > 120 else (2 if remaining > 20 else 0)))
                sc.rails_warp_factor = min(wanted, sc.maximum_rails_warp_factor)
            else:
                if body != mun:
                    raise RuntimeError('Mun SOI was lost before capture confirmation')
                if orbit.periapsis_altitude < 30000:
                    raise RuntimeError('Capture periapsis fell below the safety corridor')
                velocity = vessel.velocity(mun.non_rotating_reference_frame)
                speed = math.hypot(*velocity)
                radius = orbit.radius
                energy = speed*speed/2-mun.gravitational_parameter/radius
                target_speed = math.sqrt(max(0, 2*(plan['target_specific_energy_j_kg']+mun.gravitational_parameter/radius)))
                remaining = max(0, speed-target_speed)
                lead = plan['burn_start_ut']-ut
                if phase in ('approach', 'burn'):
                    direction = tuple(-v/speed for v in velocity)
                    ap.target_direction = direction
                    alignment = sum(a*b for a,b in zip(vessel.direction(mun.non_rotating_reference_frame), direction))
                    recorder.attitude_error_deg = math.degrees(math.acos(max(-1,min(1,alignment))))
                    if phase == 'approach' and lead > 0:
                        wanted = 5 if lead > 3000 else (4 if lead > 180 else (2 if lead > 60 else 0))
                        sc.rails_warp_factor = min(wanted, sc.maximum_rails_warp_factor)
                    else:
                        # kRPC resets throttle on EVERY RailsWarpFactor write,
                        # including writing zero when warp is already off.
                        if sc.rails_warp_factor != 0:
                            sc.rails_warp_factor = 0
                        if sc.physics_warp_factor != 0:
                            sc.physics_warp_factor = 0
                        if phase == 'approach':
                            if lead < -10:
                                raise RuntimeError('Capture burn window missed before alignment')
                            if alignment >= math.cos(math.radians(1)):
                                phase = 'burn'
                                recorder.log(phase='mun_capture_burn', event='capture_burn_start')
                        if phase == 'burn' and energy < 0 and remaining <= .1:
                            ctrl.throttle = 0
                            time.sleep(.2)
                            if ctrl.throttle != 0 or vessel.thrust > 1:
                                raise RuntimeError('Capture thrust did not stop')
                            ap.disengage()
                            ctrl.sas = False
                            recorder.attitude_control_active = False
                            verify_start = sc.ut
                            phase = 'verify'
                            recorder.log(phase='mun_capture_verify', event='capture_burn_complete')
                        else:
                            ctrl.throttle = min(1,max(.005,remaining/(vessel.available_thrust/vessel.mass))) if phase=='burn' and alignment >= math.cos(math.radians(3)) else 0
                if phase == 'verify':
                    if energy >= 0 or orbit.eccentricity >= 1 or not 90000 < orbit.apoapsis_altitude < 110000 or not 30000 < orbit.periapsis_altitude < 85000 or orbit.next_orbit is not None:
                        raise RuntimeError('Post-burn orbit is not the accepted stable Mun capture')
                    if ctrl.throttle != 0 or vessel.thrust > 1 or ctrl.sas:
                        raise RuntimeError('Capture verification requires unpowered uncontrolled coasting')
                    if ut-verify_start >= 30:
                        module_fuel = sum((p.resources.amount('LiquidFuel')+p.resources.amount('Oxidizer'))*5 for p in tanks)
                        if abs(module_fuel-440) > .01:
                            raise RuntimeError('Landing-module propellant was consumed during capture')
                        conn.krpc.paused = True
                        final = recorder.log(phase='mun_captured', event='mun_capture_confirmed')
                        name = 'SLIM_Mun_Captured_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                        sc.save(name)
                        success = True
                        return {'outcome':'mun_captured','save':name,'telemetry':recorder.filename.relative_to(Path(__file__).parent).as_posix(),'entry':entry,'capture_plan':plan,'final':final,'specific_energy_j_kg':final['speed']**2/2-mun.gravitational_parameter/final['radius'],'eccentricity':orbit.eccentricity,'period_s':orbit.period,'parts':len(vessel.parts.all),'central_propellant_kg':tank.mass-tank.dry_mass,'module_propellant_kg':module_fuel,'verification_duration_s':ut-verify_start,'next_orbit':None}
            if now-last_print > 10:
                print(f'capture {phase}: body {body.name}, remaining {remaining:.2f}, UT {ut:.1f}',flush=True)
                last_print = now
            time.sleep(.05)
    except BaseException as error:
        if recorder.error is None:
            recorder.log(event='capture_interrupted' if isinstance(error,KeyboardInterrupt) else 'capture_error')
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
            finally:
                recorder.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True,help='New result JSON; loads SLIM_Transfer_Ready_v1')
    args = parser.parse_args()
    with args.result.open('x',encoding='utf-8') as output:
        baseline = Path('saves/SLIM_Transfer_Ready_v1.sfs')
        result = {'baseline':'SLIM_Transfer_Ready_v1','baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),'code_sha256_lf':{n:hashlib.sha256(Path(n).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for n in ('capture.py','math_logic.py','logger.py','transfer.py')}}
        previous_logs = set(Path('logs').glob('flight_*.csv'))
        c = None
        try:
            expected = json.loads(Path('docs/experiments/transfer-burn-summary.json').read_text(encoding='utf-8'))['selected_checkpoint']
            if result['baseline_sha256'] != expected['sha256']:
                raise ValueError('Capture baseline checksum changed')
            c = krpc.connect(name='SLIM actual Mun entry and capture',rpc_port=5000,stream_port=5001)
            c.space_center.load('SLIM_Transfer_Ready_v1')
            c.krpc.paused = True
            time.sleep(.2)
            v = c.space_center.active_vessel
            if abs(v.mass-3619.131591796875) > .1:
                raise ValueError('Capture initial mass does not match the fixed handoff')
            result.update(execute_capture(c,v))
        except BaseException as error:
            result['error'] = repr(error)
            traceback.print_exc()
        finally:
            if c is not None:
                try:
                    v=c.space_center.active_vessel
                    result['cleanup_state']={'paused':c.krpc.paused,'throttle':v.control.throttle,'thrust_N':v.thrust,'parts':len(v.parts.all),'body':v.orbit.body.name,'mass_kg':v.mass,'sas':v.control.sas}
                except Exception as error:
                    result['cleanup_read_error']=repr(error)
                c.close()
            result['logs']=[p.as_posix() for p in set(Path('logs').glob('flight_*.csv'))-previous_logs]
            json.dump(result,output,ensure_ascii=False,indent=2)
            output.write('\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if result.get('outcome')!='mun_captured':
            raise SystemExit(1)
