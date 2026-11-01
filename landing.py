"""Prepare the Mun landing independently of the Kerbin transfer.

--prepare aligns without firing engines. --brake deorbits and removes the main
orbital velocity, then pauses and saves the handoff for vertical descent.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import time

import krpc

from logger import FlightLogger


def brake_to_descent(conn, vessel, recorder, timeout):
    """Use the Terrier, separate its empty stage, finish braking on the radial engines."""
    sc = conn.space_center
    body = sc.bodies["Mun"]
    ctrl = vessel.control
    ap = vessel.auto_pilot
    flight = vessel.flight(body.reference_frame)
    r = vessel.orbit.radius
    mu = body.gravitational_parameter
    target_radius = body.equatorial_radius - 5000
    target_speed = math.sqrt(mu * (2 / r - 2 / (r + target_radius)))
    speed = math.sqrt(sum(x * x for x in vessel.velocity(body.non_rotating_reference_frame)))
    planned_dv = max(0.0, speed - target_speed)
    available_dv = vessel.vacuum_specific_impulse * 9.80665 * math.log(vessel.mass / vessel.dry_mass)
    if available_dv < speed + 150:
        raise ValueError("Insufficient ideal delta-v for braking and a 150 m/s allowance")
    print(f"Deorbit estimate: {planned_dv:.1f} m/s; target periapsis -5000 m; ideal reserve {available_dv:.1f} m/s", flush=True)
    recorder.log(phase="deorbit", event="burn_start")
    phase = "deorbit"
    started = phase_started = last_progress = last_print = time.monotonic()
    last_ut = sc.ut
    sc.rails_warp_factor = 0
    sc.physics_warp_factor = 0
    while True:
        now = time.monotonic()
        if now - started > timeout or (phase != "coast" and now - phase_started > 120):
            raise TimeoutError(f"{phase}: braking phase exceeded its wall-clock timeout")
        if recorder.error is not None:
            raise RuntimeError("Braking telemetry failed") from recorder.error
        if conn.krpc.paused or sc.active_vessel != vessel or vessel.orbit.body != body:
            raise RuntimeError("Pause, vessel or central body changed during braking")
        ut = sc.ut
        if ut < last_ut:
            raise RuntimeError("KSP time moved backwards during braking")
        if ut > last_ut:
            last_progress = now
        if now - last_progress > 5:
            raise TimeoutError("KSP universal time stopped advancing during braking")
        last_ut = ut
        altitude = flight.surface_altitude
        horizontal = flight.horizontal_speed
        vertical = flight.vertical_speed
        if vessel.situation.name == "landed" or altitude < 1500:
            raise RuntimeError("Reached the 1500 m safety floor before the descent handoff")
        if phase == "deorbit":
            if vessel.orbit.periapsis_altitude <= -5000:
                ctrl.throttle = 0.0
                recorder.log(phase="deorbit_coast", event="burn_end")
                phase = "coast"
                phase_started = now
            elif ap.error < 5:
                ctrl.throttle = 0.15 if vessel.orbit.periapsis_altitude > 1000 else 0.05
            else:
                ctrl.throttle = 0.0
        elif phase == "coast":
            # Coarse coast at 10x only above 12 km radar altitude. No thrust in warp.
            wanted_warp = 2 if altitude > 12000 else 0
            if sc.rails_warp_factor != wanted_warp:
                sc.rails_warp_factor = wanted_warp
            if altitude <= 8000 and vertical < 0:
                sc.rails_warp_factor = 0
                phase = "braking"
                phase_started = now
                recorder.log(phase="main_braking", event="burn_start")
        else:
            if phase == "braking" and vessel.available_thrust <= 0:
                # Terrier cannot draw from the four module tanks in this craft.
                ctrl.throttle = 0.0
                time.sleep(0.2)  # KSP applies control inputs on a physics update.
                if ctrl.throttle != 0:
                    raise RuntimeError("Zero throttle was not applied before separation")
                recorder.log(event="braking_stage_empty")
                control_part = vessel.parts.controlling
                vessel.parts.decouplers[0].decouple()
                vessel = control_part.vessel
                recorder.vessel = vessel
                ctrl = vessel.control
                ap = vessel.auto_pilot
                flight = vessel.flight(body.reference_frame)
                sc.active_vessel = vessel
                if len(vessel.parts.engines) != 2 or any(
                    e.part.name != "smallRadialEngine.v2" for e in vessel.parts.engines
                ):
                    raise RuntimeError("Unexpected vessel after braking-stage separation")
                for engine in vessel.parts.engines:
                    engine.active = True
                ap.reference_frame = vessel.surface_velocity_reference_frame
                ap.target_direction = (0, -1, 0)
                ap.engage()
                time.sleep(0.2)
                phase = "landing_braking"
                phase_started = now
                recorder.log(phase="landing_braking", event="braking_stage_separated")
            if horizontal <= 5 and -20 <= vertical <= 0:
                ctrl.throttle = 0.0
                sc.rails_warp_factor = 0
                time.sleep(0.2)
                if ctrl.throttle != 0 or vessel.thrust > 1:
                    raise RuntimeError("Thrust did not stop before saving the descent handoff")
                conn.krpc.paused = True
                row = recorder.log(phase="descent_handoff", event="main_braking_complete")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                save_name = f"SLIM_Descent_Ready_{stamp}"
                sc.save(save_name)
                summary = {
                    "save": save_name, "planned_deorbit_delta_v_m_s": planned_dv,
                    "telemetry": str(recorder.filename), "handoff": row,
                    "landing_module_propellant_kg": sum(
                        (p.resources.amount("LiquidFuel") + p.resources.amount("Oxidizer")) * 5
                        for p in vessel.parts.all if p.name == "externalTankRound"
                    ),
                }
                with recorder.filename.with_suffix(".handoff.json").open("x", encoding="utf-8") as file:
                    json.dump(summary, file, ensure_ascii=False, indent=2)
                print(f"Handoff saved: {save_name}; altitude {altitude:.0f} m, horizontal {horizontal:.2f} m/s, vertical {vertical:.2f} m/s", flush=True)
                return "ready_for_vertical_descent"
            ctrl.throttle = min(1.0, max(0.05, flight.speed / 60)) if ap.error < 5 else 0.0
        if phase != "coast":
            if vessel.available_thrust <= 0 and phase != "braking":
                raise RuntimeError("No available thrust during braking")
            reserve = sum(
                (p.resources.amount("LiquidFuel") + p.resources.amount("Oxidizer")) * 5
                for p in vessel.parts.all if p.name == "externalTankRound"
            )
            if reserve < 100:
                raise RuntimeError("Landing-module propellant fell below the 100 kg reserve")
        if now - last_print >= 5:
            print(f"{phase}: altitude {altitude:.0f} m, horizontal {horizontal:.1f} m/s, vertical {vertical:.1f} m/s", flush=True)
            last_print = now
        time.sleep(0.1)


def run_landing(conn, *, prepare=False, timeout=30.0, brake=False, brake_timeout=900.0):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    if not math.isfinite(brake_timeout) or brake_timeout <= 0:
        raise ValueError("Braking timeout must be finite and positive")
    prepare = prepare or brake
    sc = conn.space_center
    vessel = sc.active_vessel
    ctrl = vessel.control
    ap = vessel.auto_pilot
    recorder = None
    owns_controls = False
    original_sas = ctrl.sas
    try:
        recorder = FlightLogger(vessel, sc)
        print(f"Telemetry: {recorder.filename}", flush=True)
        recorder.log(phase="landing_preflight", event="check_start")
        recorder.start()

        if vessel.orbit.body != sc.bodies["Mun"]:
            raise ValueError("Landing preparation requires a Mun orbit")
        if vessel.situation.name != "orbiting" or vessel.orbit.periapsis_altitude <= 0:
            raise ValueError("Start from the intact vessel in SLIM_Landing_Start_v1")
        if ctrl.throttle != 0 or vessel.thrust > 0:
            raise ValueError("Set throttle to zero before starting landing preparation")
        engines = vessel.parts.engines
        main = [e for e in engines if e.part.name == "liquidEngine3.v2"]
        radial = [e for e in engines if e.part.name == "smallRadialEngine.v2"]
        if len(engines) != 3 or len(main) != 1 or len(radial) != 2:
            raise ValueError("Expected one Terrier and two radial landing engines")
        if not main[0].active or any(e.active for e in radial):
            raise ValueError("Terrier must be active and both radial engines inactive")
        if not all(e.has_fuel for e in engines):
            raise ValueError("An engine has no accessible propellant")
        tanks = [p for p in vessel.parts.all if p.name == "externalTankRound"]
        if len(tanks) != 4 or any(
            p.resources.amount("LiquidFuel") <= 0 or p.resources.amount("Oxidizer") <= 0
            for p in tanks
        ):
            raise ValueError("All four landing-module tanks must contain both propellants")
        decouplers = vessel.parts.decouplers
        if len(decouplers) != 1 or decouplers[0].decoupled:
            raise ValueError("The braking stage must still be attached")
        if vessel.resources.amount("ElectricCharge") <= 0:
            raise ValueError("No electric charge for attitude control")
        if not vessel.parts.reaction_wheels or not all(
            any(abs(torque) > 0 for torque in axis)
            for axis in zip(*vessel.available_torque)
        ):
            raise ValueError("Attitude control torque is unavailable")
        if vessel.available_thrust <= vessel.mass * vessel.orbit.body.surface_gravity:
            raise ValueError("Available thrust is insufficient to support the vessel on Mun")
        recorder.log(event="preflight_passed")
        print("Preflight passed: Mun orbit, attached stage, propellant, thrust and attitude control.", flush=True)
        if not prepare:
            return "preflight_passed"
        if conn.krpc.paused:
            raise ValueError("Resume KSP before --prepare; --check works while paused")

        # Own the controls only after the read-only checks have passed.
        owns_controls = True
        ctrl.throttle = 0.0
        recorder.log(phase="landing_orientation", event="alignment_start")
        ap.reference_frame = vessel.surface_velocity_reference_frame
        ap.target_direction = (0, -1, 0)
        ap.engage()
        started = time.monotonic()
        last_ut = sc.ut
        last_progress = started
        settled_samples = 0
        while True:
            now = time.monotonic()
            if now - started >= timeout:
                raise TimeoutError("Landing orientation exceeded its wall-clock timeout")
            if recorder.error is not None:
                raise RuntimeError("Landing telemetry failed") from recorder.error
            if sc.active_vessel != vessel or vessel.orbit.body != sc.bodies["Mun"]:
                raise RuntimeError("Active vessel or central body changed; aborting preparation")
            if conn.krpc.paused:
                raise RuntimeError("KSP paused during orientation; preparation aborted")
            ut = sc.ut
            if ut < last_ut:
                raise RuntimeError("KSP time moved backwards; restart from the saved baseline")
            if ut > last_ut:
                last_progress = now
            if now - last_progress >= min(5.0, timeout):
                raise TimeoutError("KSP universal time stopped advancing")
            last_ut = ut
            error = ap.error  # kRPC reports the angular error in degrees.
            settled_samples = settled_samples + 1 if math.isfinite(error) and error < 2.0 else 0
            if settled_samples >= 3:
                break
            time.sleep(0.1)
        recorder.log(event="ready_for_deorbit")
        if brake:
            return brake_to_descent(conn, vessel, recorder, brake_timeout)
        print("Orientation ready. Engines were not fired; deorbit is the next iteration.", flush=True)
        return "ready_for_deorbit"
    except BaseException as error:
        if recorder is not None and recorder.error is None:
            try:
                recorder.log(event="landing_interrupted" if isinstance(error, KeyboardInterrupt) else "landing_error")
            except Exception as logging_error:
                print(f"Could not record abort: {logging_error}", flush=True)
        raise
    finally:
        # Each cleanup is attempted even if a previous cleanup fails.
        try:
            if owns_controls:
                if brake:
                    ctrl = recorder.vessel.control
                    ap = recorder.vessel.auto_pilot
                ctrl.throttle = 0.0
                if brake:
                    sc.rails_warp_factor = 0
                    sc.physics_warp_factor = 0
                    if conn.krpc.paused and ctrl.throttle != 0:
                        conn.krpc.paused = False
                    if not conn.krpc.paused:
                        time.sleep(0.2)
                    conn.krpc.paused = True
        finally:
            try:
                if owns_controls:
                    try:
                        ap.disengage()
                    finally:
                        ctrl.sas = original_sas
            finally:
                if recorder is not None:
                    recorder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only preflight checks (default)")
    mode.add_argument("--prepare", action="store_true", help="Align retrograde, then release controls; no engine burns")
    mode.add_argument("--brake", action="store_true", help="Deorbit, brake, separate the empty stage, finish on radial engines, then pause and save")
    parser.add_argument("--timeout", type=float, default=30, help="Maximum orientation wait in wall-clock seconds")
    parser.add_argument("--brake-timeout", type=float, default=900, help="Maximum deorbit/coast/braking duration in wall-clock seconds")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not math.isfinite(args.brake_timeout) or args.brake_timeout <= 0:
        parser.error("--brake-timeout must be finite and positive")
    conn = krpc.connect(name="SLIM landing preparation", rpc_port=5000, stream_port=5001)
    try:
        run_landing(conn, prepare=args.prepare, timeout=args.timeout, brake=args.brake, brake_timeout=args.brake_timeout)
    except KeyboardInterrupt:
        parser.exit(130, "Landing preparation interrupted; cleanup attempted.\n")
    except Exception as error:
        parser.exit(1, f"Landing preparation failed: {error}\n")
    finally:
        conn.close()
