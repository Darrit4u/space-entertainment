"""Prepare the Mun landing independently of the Kerbin transfer.

At this iteration --prepare aligns the intact vessel without firing engines.
Deorbit, powered descent and touchdown are implemented in subsequent iterations.
"""

import argparse
import math
import time

import krpc

from logger import FlightLogger


def run_landing(conn, *, prepare=False, timeout=30.0):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
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
                ctrl.throttle = 0.0
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
    parser.add_argument("--timeout", type=float, default=30, help="Maximum orientation wait in wall-clock seconds")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    conn = krpc.connect(name="SLIM landing preparation", rpc_port=5000, stream_port=5001)
    try:
        run_landing(conn, prepare=args.prepare, timeout=args.timeout)
    except KeyboardInterrupt:
        parser.exit(130, "Landing preparation interrupted; cleanup attempted.\n")
    except Exception as error:
        parser.exit(1, f"Landing preparation failed: {error}\n")
    finally:
        conn.close()
