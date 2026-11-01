"""Spatial Kerbin-Mun transfer planning and bounded burns; no Mun capture."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import krpc
import numpy as np

from logger import FlightLogger
from math_logic import OrbitalMath


def encounter(orbit, mun):
    """Read the first Mun patch, rather than infer encounter from apoapsis."""
    for _ in range(4):
        if orbit.body == mun:
            return {"periapsis_altitude_m": orbit.periapsis_altitude, "periapsis_radius_m": orbit.periapsis, "eccentricity": orbit.eccentricity}
        orbit = orbit.next_orbit
        if orbit is None:
            return None
    return None


def refine_node(node, mun, deadline, target=50000):
    """Minimum-norm local correction of three node components using KSP patches."""
    for _ in range(10):
        if time.monotonic() > deadline:
            raise TimeoutError("Encounter refinement timed out")
        prediction = encounter(node.orbit, mun)
        if prediction is None:
            return None
        error = target-prediction["periapsis_altitude_m"]
        if abs(error) < 3000:
            return prediction
        values = [node.prograde, node.normal, node.radial]
        gradient = []
        for name, value in zip(("prograde", "normal", "radial"), values):
            setattr(node, name, value+.5)
            time.sleep(.03)
            perturbed = encounter(node.orbit, mun)
            setattr(node, name, value)
            gradient.append(0 if perturbed is None else (perturbed["periapsis_altitude_m"]-prediction["periapsis_altitude_m"])/.5)
        g = np.asarray(gradient)
        if np.linalg.norm(g) < 1:
            return None
        step = error*g/np.dot(g,g)
        step *= min(1, 10/max(np.linalg.norm(step), 1e-12))
        for name, value, delta in zip(("prograde", "normal", "radial"), values, step):
            setattr(node, name, float(value+delta))
        time.sleep(.04)
    return None


def find_transfer(conn, vessel, timeout=120):
    sc = conn.space_center
    kerbin, mun = sc.bodies['Kerbin'], sc.bodies['Mun']
    if not conn.krpc.paused or vessel.orbit.body != kerbin or vessel.control.nodes:
        raise ValueError("Plan on a paused Kerbin orbit without existing maneuver nodes")
    rf, orbit = kerbin.non_rotating_reference_frame, vessel.orbit
    deadline = time.monotonic()+timeout
    now, mu = sc.ut, kerbin.gravitational_parameter
    candidates = []
    # Real ephemerides in the same inertial frame; no coplanar phase assumption.
    for index in range(73):
        if time.monotonic() > deadline:
            raise TimeoutError("Spatial candidate search timed out")
        departure = now+180+index*orbit.period/24
        r = np.asarray(orbit.position_at(departure, rf))
        v = (np.asarray(orbit.position_at(departure+.5, rf))-np.asarray(orbit.position_at(departure-.5, rf)))
        for duration in range(20000, 40001, 2500):
            arrival = departure+duration
            rm = np.asarray(mun.orbit.position_at(arrival, rf))
            vm = (np.asarray(mun.orbit.position_at(arrival+.5, rf))-np.asarray(mun.orbit.position_at(arrival-.5, rf)))
            for long_path in (False, True):
                try:
                    vd, va = map(np.asarray, OrbitalMath.lambert(mu, r, rm, duration, long_path=long_path))
                    relative = va-vm
                    vinf = np.linalg.norm(relative)
                    rp = mun.equatorial_radius+50000
                    impact_parameter = math.sqrt(rp*rp+2*mun.gravitational_parameter*rp/vinf**2)
                    b = rm-relative*np.dot(rm,relative)/np.dot(relative,relative)
                    b /= np.linalg.norm(b)
                    goal = rm+b*impact_parameter
                    vd, va = map(np.asarray, OrbitalMath.lambert(mu, r, goal, duration, long_path=long_path))
                    delta = vd-v
                    cost = np.linalg.norm(delta)
                    capture = math.sqrt(vinf**2+2*mun.gravitational_parameter/rp)-math.sqrt(mun.gravitational_parameter/rp)
                    if cost+capture+40 < 1389 and cost < 1100:
                        candidates.append({"ut": departure, "delta": delta.tolist(), "delta_v_m_s": float(cost), "arrival_estimate_ut": arrival, "goal_inertial_m": goal.tolist(), "long_path": long_path, "rank": float(cost+capture)})
                except ValueError:
                    continue
    node = None
    try:
        for candidate in sorted(candidates, key=lambda c:c['rank'])[:12]:
            if time.monotonic() > deadline:
                raise TimeoutError("Patched-conic validation timed out")
            if node is not None:
                node.remove()
            node = vessel.control.add_node(candidate['ut'])
            vector = sc.transform_direction(tuple(candidate['delta']), rf, node.orbital_reference_frame)
            node.prograde, node.normal, node.radial = vector[1], vector[2], -vector[0]
            time.sleep(.08)
            prediction = refine_node(node, mun, deadline)
            if prediction is None or not 30000 < prediction['periapsis_altitude_m'] < 80000:
                continue
            if node.orbit.periapsis_altitude < 70000:
                continue
            candidate.update(prograde=node.prograde, normal=node.normal, radial=node.radial, predicted_encounter=prediction, candidate_count=len(candidates), planned_delta_v_m_s=node.delta_v)
            OrbitalMath.burn_duration(node.delta_v, vessel.mass, vessel.available_thrust, 345, 1567)
            return candidate
        raise ValueError(f"No safe Mun encounter within the fuel budget ({len(candidates)} Lambert candidates)")
    finally:
        if node is not None:
            node.remove()


def fly_node(conn, vessel, node, recorder, deadline, *, body=None, prefix='transfer'):
    """Align, wait with coast-only warp, then close the actual burn-vector error."""
    sc, ctrl, ap = conn.space_center, vessel.control, vessel.auto_pilot
    body = sc.bodies['Kerbin'] if body is None else body
    rf = body.non_rotating_reference_frame
    tank = next(p for p in vessel.parts.all if p.name == 'fuelTank')
    duration, _ = OrbitalMath.burn_duration(node.delta_v, vessel.mass, vessel.available_thrust, 345, max(0, tank.mass-tank.dry_mass-433))
    start_ut = node.ut-duration/2
    if start_ut <= sc.ut+30:
        raise ValueError('Insufficient lead time for a bounded node alignment')
    ap.reference_frame = rf
    ap.target_roll = float('nan')
    ap.engage()
    recorder.attitude_control_active = True
    last_ut = sc.ut
    progress = time.monotonic()
    last_print = progress
    burning = False
    initial_remaining = node.delta_v
    conn.krpc.paused = False
    recorder.log(phase=prefix+'_wait', event='node_alignment_start')
    while True:
        now = time.monotonic()
        if now > deadline:
            raise TimeoutError('Transfer execution exceeded its wall-clock limit')
        if recorder.error is not None:
            raise RuntimeError('Transfer telemetry failed') from recorder.error
        if conn.krpc.paused or sc.active_vessel != vessel or vessel.orbit.body != body:
            raise RuntimeError('Pause, vessel or body changed during the transfer burn')
        if len(vessel.parts.all) != 55:
            raise RuntimeError('A part was lost during transfer')
        if tank.mass-tank.dry_mass < 433 or vessel.available_thrust <= 0:
            raise RuntimeError('Protected central reserve or available thrust was lost')
        ut = sc.ut
        if ut < last_ut:
            raise RuntimeError('KSP time moved backwards')
        if ut > last_ut:
            progress = now
        elif now-progress > 5:
            raise TimeoutError('KSP time stopped advancing')
        last_ut = ut
        vector = np.asarray(node.remaining_burn_vector(rf))
        remaining = float(np.linalg.norm(vector))
        if not math.isfinite(remaining) or remaining > initial_remaining+10:
            raise RuntimeError('Maneuver residual diverged')
        if burning and remaining <= .15:
            ctrl.throttle = 0
            time.sleep(.2)
            if ctrl.throttle != 0 or vessel.thrust > 1:
                raise RuntimeError('Thrust did not stop after the maneuver')
            recorder.log(event=prefix+'_burn_complete')
            return
        direction = vector/max(remaining, 1e-12)
        ap.target_direction = tuple(direction)
        alignment = float(np.dot(vessel.direction(rf), direction))
        recorder.attitude_error_deg = math.degrees(math.acos(max(-1, min(1, alignment))))
        lead = start_ut-ut
        if lead > 0:
            ctrl.throttle = 0
            sc.rails_warp_factor = 4 if lead > 180 else (2 if lead > 60 else 0)
        else:
            # Rewriting RailsWarpFactor=0 also cuts throttle in kRPC 0.5.4.
            if sc.rails_warp_factor != 0:
                sc.rails_warp_factor = 0
            if sc.physics_warp_factor != 0:
                sc.physics_warp_factor = 0
            if not burning:
                if lead < -10:
                    raise RuntimeError('Burn window missed before alignment')
                if alignment >= math.cos(math.radians(1)):
                    burning = True
                    recorder.log(phase=prefix+'_burn', event='burn_start')
            ctrl.throttle = min(1, max(.005, remaining/(vessel.available_thrust/vessel.mass))) if burning and alignment >= math.cos(math.radians(3)) else 0
        if now-last_print > 10:
            print(f'{prefix}: lead {lead:.1f} s, remaining {remaining:.3f} m/s, alignment {recorder.attitude_error_deg:.2f} deg', flush=True)
            last_print = now
        time.sleep(.05)


def execute_transfer(conn, vessel, plan, timeout=900):
    sc, ctrl, ap = conn.space_center, vessel.control, vessel.auto_pilot
    if not conn.krpc.paused or ctrl.nodes or vessel.orbit.body != sc.bodies['Kerbin'] or len(vessel.parts.all) != 55:
        raise ValueError('Execute from the paused intact Kerbin baseline without nodes')
    if ctrl.throttle != 0 or vessel.thrust != 0:
        raise ValueError('Zero throttle and thrust required before transfer')
    old_sas, old_roll = ctrl.sas, ap.target_roll
    recorder = FlightLogger(vessel, sc)
    recorder.start()
    owned = []
    success = False
    deadline = time.monotonic()+timeout
    try:
        node = ctrl.add_node(plan['ut'], prograde=plan['prograde'], normal=plan['normal'], radial=plan['radial'])
        owned.append(node)
        prediction = encounter(node.orbit, sc.bodies['Mun'])
        if prediction is None or not 30000 < prediction['periapsis_altitude_m'] < 80000:
            raise ValueError('Recreated node no longer predicts a safe Mun encounter')
        fly_node(conn, vessel, node, recorder, deadline)
        node.remove()
        owned.remove(node)
        actual = encounter(vessel.orbit, sc.bodies['Mun'])
        if actual is None:
            raise RuntimeError('Actual trajectory after the burn misses Mun')
        correction = None
        if not 30000 < actual['periapsis_altitude_m'] < 80000:
            node = ctrl.add_node(sc.ut+300)
            owned.append(node)
            prediction = refine_node(node, sc.bodies['Mun'], min(deadline,time.monotonic()+30))
            if prediction is None or node.delta_v > 20:
                raise RuntimeError('A safe correction within 20 m/s could not be planned')
            correction = {'ut':node.ut, 'delta_v_m_s':node.delta_v, 'prediction':prediction}
            recorder.log(phase='transfer_correction', event='correction_planned')
            fly_node(conn, vessel, node, recorder, deadline)
            node.remove()
            owned.remove(node)
            actual = encounter(vessel.orbit, sc.bodies['Mun'])
        if actual is None or not 30000 < actual['periapsis_altitude_m'] < 80000:
            raise RuntimeError('Actual post-burn orbit lacks the accepted Mun periapsis')
        ap.disengage()
        recorder.attitude_control_active = False
        ctrl.sas = False
        conn.krpc.paused = True
        final = recorder.log(phase='transfer_ready', event='safe_mun_approach_confirmed')
        save_name = 'SLIM_Transfer_Ready_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        sc.save(save_name)
        success = True
        return {'outcome':'transfer_ready', 'save':save_name, 'telemetry':recorder.filename.relative_to(Path(__file__).parent).as_posix(), 'actual_trajectory_prediction':actual, 'correction':correction, 'final':final, 'parts':len(vessel.parts.all), 'central_propellant_kg':next(p for p in vessel.parts.all if p.name=='fuelTank').mass-250, 'time_to_soi_change_s':vessel.orbit.time_to_soi_change, 'limitation':'Mun encounter is predicted from the actual orbit; SOI entry and capture are the next iteration'}
    except BaseException as error:
        if recorder.error is None:
            recorder.log(event='transfer_interrupted' if isinstance(error, KeyboardInterrupt) else 'transfer_error')
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
                ap.disengage()
                ap.target_roll = old_roll
                ctrl.sas = False if success else old_sas
                for node in owned:
                    node.remove()
            finally:
                recorder.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', type=Path, help='New plan JSON; loads baseline and plans while paused')
    mode.add_argument('--execute', type=Path, help='Existing plan JSON to execute from the fixed baseline')
    parser.add_argument('--result', type=Path, help='New execution result JSON')
    args = parser.parse_args()
    if args.execute and not args.result:
        parser.error('--execute requires a new --result file')
    with (args.plan or args.result).open('x', encoding='utf-8') as output:
        previous_logs = set(Path('logs').glob('flight_*.csv'))
        c = krpc.connect(name='SLIM spatial transfer planning', rpc_port=5000, stream_port=5001)
        try:
            c.space_center.load('SLIM_Transfer_Start_v1')
            c.krpc.paused = True
            time.sleep(.2)
            v = c.space_center.active_vessel
            assert len(v.parts.all) == 55 and abs(v.mass-4653) < .01 and v.control.throttle == 0 and v.thrust == 0
            if args.plan:
                plan = find_transfer(c, v)
                plan.update(baseline='SLIM_Transfer_Start_v1', baseline_sha256=hashlib.sha256(Path('saves/SLIM_Transfer_Start_v1.sfs').read_bytes()).hexdigest(), planning_ut=c.space_center.ut, state='predicted_encounter_only')
            else:
                plan = json.loads(args.execute.read_text(encoding='utf-8'))
                if hashlib.sha256(Path('saves/SLIM_Transfer_Start_v1.sfs').read_bytes()).hexdigest() != plan['baseline_sha256']:
                    raise ValueError('Baseline checksum changed')
                plan = execute_transfer(c, v, plan)
                plan['input_plan_sha256_lf'] = hashlib.sha256(args.execute.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            plan['code_sha256_lf'] = {name:hashlib.sha256(Path(name).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for name in ('transfer.py','math_logic.py')}
            json.dump(plan, output, ensure_ascii=False, indent=2)
            output.write('\n')
            print(json.dumps(plan, ensure_ascii=False), flush=True)
        except BaseException as error:
            current = c.space_center.active_vessel
            logs = set(Path('logs').glob('flight_*.csv'))-previous_logs
            json.dump({'error':repr(error), 'paused':c.krpc.paused, 'throttle':current.control.throttle, 'thrust_N':current.thrust, 'parts':len(current.parts.all), 'mass_kg':current.mass, 'telemetry':[p.as_posix() for p in logs], 'code_sha256_lf':{name:hashlib.sha256(Path(name).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for name in ('transfer.py','math_logic.py')}}, output, ensure_ascii=False, indent=2)
            raise
        finally:
            c.close()
