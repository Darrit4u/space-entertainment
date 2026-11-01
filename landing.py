"""Prepare the Mun landing independently of the Kerbin transfer.

--prepare aligns without firing engines. --brake deorbits and removes the main
orbital velocity, then pauses and saves the handoff for vertical descent.
--descend continues from the detached module and stops above the surface.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import time

import krpc

from logger import FlightLogger


def descent_acceleration(altitude, velocity, gravity, measured_vertical_acceleration):
    """Return target vertical speed and thrust acceleration in the surface frame."""
    target = -min(50.0, max(1.0, math.sqrt(0.7 * max(altitude - 40.0, 0.0))))
    up = max(0.0, gravity + 0.6 * (target - velocity[0]) - 0.2 * measured_vertical_acceleration)
    north, east = -0.5 * velocity[1], -0.5 * velocity[2]
    horizontal = math.hypot(north, east)
    limit = 0.2 * up  # At most atan(0.2) = 11.3 degrees from vertical.
    if horizontal > limit:
        scale = limit / horizontal
        north *= scale
        east *= scale
    return target, (up, north, east)


def descend_to_low_altitude(conn, vessel, recorder, timeout):
    """Fly the detached module down to about 40 m, without attempting contact."""
    sc = conn.space_center
    body = sc.bodies["Mun"]
    ctrl, ap = vessel.control, vessel.auto_pilot
    surface_rf = vessel.surface_reference_frame
    flight = vessel.flight(body.reference_frame)
    sc.rails_warp_factor = 0
    sc.physics_warp_factor = 0
    ap.reference_frame = surface_rf
    ap.target_direction = (1, 0, 0)  # Surface-frame x is up; vessel thrust is forward.
    ap.engage()
    recorder.log(phase="vertical_descent", event="descent_start")
    started = last_progress = last_print = time.monotonic()
    last_ut = sc.ut
    previous_vertical = flight.vertical_speed
    while True:
        now = time.monotonic()
        if now - started > timeout:
            raise TimeoutError("Vertical descent exceeded its wall-clock timeout")
        if recorder.error is not None:
            raise RuntimeError("Vertical descent telemetry failed") from recorder.error
        if conn.krpc.paused or sc.active_vessel != vessel or vessel.orbit.body != body:
            raise RuntimeError("Pause, vessel or body changed during vertical descent")
        ut = sc.ut
        if ut < last_ut:
            raise RuntimeError("KSP time moved backwards during vertical descent")
        dt = ut - last_ut
        if dt <= 0:
            if now - last_progress > 5:
                raise TimeoutError("KSP universal time stopped advancing during descent")
            time.sleep(0.1)
            continue
        last_progress = now
        altitude = flight.surface_altitude
        # A vessel-centred frame moves with the vessel: its raw velocity is zero.
        # Rotate the body-relative velocity as a direction, without translating it.
        velocity = sc.transform_direction(vessel.velocity(body.reference_frame), body.reference_frame, surface_rf)
        horizontal = math.hypot(velocity[1], velocity[2])
        if altitude < 25 or vessel.situation.name == "landed":
            raise RuntimeError("Reached the 25 m safety floor before the low-altitude handoff")
        gravity = body.gravitational_parameter / vessel.orbit.radius ** 2
        braking_acceleration = max(0.1, vessel.available_thrust / vessel.mass - gravity)
        emergency_height = 25 + 2 * abs(velocity[0]) + velocity[0] ** 2 / (2 * braking_acceleration)
        if velocity[0] < -10 and altitude < emergency_height:
            raise RuntimeError("Descent entered the speed-dependent emergency altitude")
        if altitude <= 43 and abs(velocity[0]) <= 1.5 and horizontal <= 0.2:
            ctrl.throttle = 0.0
            time.sleep(0.2)
            if ctrl.throttle != 0 or vessel.thrust > 1:
                raise RuntimeError("Thrust did not stop before the low-altitude handoff")
            conn.krpc.paused = True
            row = recorder.log(phase="touchdown_handoff", event="vertical_descent_complete")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            save_name = f"SLIM_Touchdown_Ready_{stamp}"
            sc.save(save_name)
            with recorder.filename.with_suffix(".handoff.json").open("x", encoding="utf-8") as file:
                json.dump({"save": save_name, "telemetry": str(recorder.filename), "handoff": row}, file, ensure_ascii=False, indent=2)
            print(f"Low-altitude handoff saved: {save_name}; altitude {row['altitude']:.2f} m, vertical {row['vertical_speed']:.2f} m/s", flush=True)
            return "ready_for_touchdown"
        if vessel.mass - vessel.dry_mass < 50 or vessel.available_thrust <= 0:
            raise RuntimeError("Insufficient propellant reserve or available thrust during descent")
        measured_acceleration = (velocity[0] - previous_vertical) / dt
        target, acceleration = descent_acceleration(altitude, velocity, gravity, measured_acceleration)
        recorder.target_vertical_speed = target
        magnitude = math.sqrt(sum(a * a for a in acceleration))
        ap.target_direction = tuple(a / magnitude for a in acceleration) if magnitude > 0 else (1, 0, 0)
        actual_up = vessel.direction(surface_rf)[0]
        recorder.attitude_error_deg = ap.error
        recorder.thrust_up_component = actual_up
        recorder.commanded_up_acceleration = acceleration[0]
        # Only the actual thrust direction affects vertical force. A roll/heading
        # error reported by AutoPilot must not remove gravity compensation.
        ctrl.throttle = min(1.0, vessel.mass * acceleration[0] / (vessel.available_thrust * actual_up)) if actual_up > 0.7 else 0.0
        previous_vertical, last_ut = velocity[0], ut
        if now - last_print >= 10:
            print(f"descent: altitude {altitude:.0f} m, target {target:.1f}, vertical {velocity[0]:.1f}, horizontal {horizontal:.2f}, fuel {vessel.mass-vessel.dry_mass:.0f} kg, thrust-up {actual_up:.3f}, AP-error {recorder.attitude_error_deg:.1f}", flush=True)
            last_print = now
        time.sleep(0.1)


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


def run_landing(conn, *, prepare=False, timeout=30.0, brake=False, brake_timeout=900.0, descend=False, descent_timeout=600.0):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    if not math.isfinite(brake_timeout) or brake_timeout <= 0:
        raise ValueError("Braking timeout must be finite and positive")
    if not math.isfinite(descent_timeout) or descent_timeout <= 0:
        raise ValueError("Descent timeout must be finite and positive")
    if descend and (prepare or brake):
        raise ValueError("Choose either vertical descent or preparation/braking")
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
        if not descend and (vessel.situation.name != "orbiting" or vessel.orbit.periapsis_altitude <= 0):
            raise ValueError("Start from the intact vessel in SLIM_Landing_Start_v1")
        if descend:
            initial_flight = vessel.flight(vessel.orbit.body.reference_frame)
            if vessel.situation.name != "sub_orbital" or not 100 < initial_flight.surface_altitude < 15000:
                raise ValueError("Start descent from SLIM_Descent_Ready_v1 above 100 m")
            if initial_flight.horizontal_speed > 10 or abs(initial_flight.vertical_speed) > 30:
                raise ValueError("Initial velocities are outside the vertical-descent envelope")
        if ctrl.throttle != 0 or vessel.thrust > 0:
            raise ValueError("Set throttle to zero before starting landing preparation")
        engines = vessel.parts.engines
        main = [e for e in engines if e.part.name == "liquidEngine3.v2"]
        radial = [e for e in engines if e.part.name == "smallRadialEngine.v2"]
        if descend:
            if len(engines) != 2 or len(radial) != 2 or not all(e.active for e in radial):
                raise ValueError("Descent requires two active radial engines and no Terrier")
        else:
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
        if not descend and (len(decouplers) != 1 or decouplers[0].decoupled):
            raise ValueError("The braking stage must still be attached")
        if descend and decouplers:
            raise ValueError("Separate the braking stage before starting vertical descent")
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
        print("Preflight passed: Mun, expected configuration, propellant, thrust and attitude control.", flush=True)
        if not prepare and not descend:
            return "preflight_passed"
        if conn.krpc.paused:
            raise ValueError("Resume KSP before --prepare; --check works while paused")

        # Own the controls only after the read-only checks have passed.
        owns_controls = True
        ctrl.throttle = 0.0
        if descend:
            return descend_to_low_altitude(conn, vessel, recorder, descent_timeout)
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
                if brake or descend:
                    ctrl = recorder.vessel.control
                    ap = recorder.vessel.auto_pilot
                ctrl.throttle = 0.0
                if brake or descend:
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
    mode.add_argument("--descend", action="store_true", help="Continue from the detached module to about 40 m, then pause and save; no contact")
    parser.add_argument("--timeout", type=float, default=30, help="Maximum orientation wait in wall-clock seconds")
    parser.add_argument("--brake-timeout", type=float, default=900, help="Maximum deorbit/coast/braking duration in wall-clock seconds")
    parser.add_argument("--descent-timeout", type=float, default=600, help="Maximum vertical-descent duration in wall-clock seconds")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not math.isfinite(args.brake_timeout) or args.brake_timeout <= 0:
        parser.error("--brake-timeout must be finite and positive")
    if not math.isfinite(args.descent_timeout) or args.descent_timeout <= 0:
        parser.error("--descent-timeout must be finite and positive")
    conn = krpc.connect(name="SLIM landing preparation", rpc_port=5000, stream_port=5001)
    try:
        run_landing(conn, prepare=args.prepare, timeout=args.timeout, brake=args.brake, brake_timeout=args.brake_timeout, descend=args.descend, descent_timeout=args.descent_timeout)
    except KeyboardInterrupt:
        parser.exit(130, "Landing preparation interrupted; cleanup attempted.\n")
    except Exception as error:
        parser.exit(1, f"Landing preparation failed: {error}\n")
    finally:
        conn.close()
