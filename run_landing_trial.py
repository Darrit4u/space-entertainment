"""Load the fixed Mun-orbit baseline and record one complete landing trial."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import krpc

from landing import run_landing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="New JSON result file; existing files are never overwritten")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("x", encoding="utf-8") as output:
        result = {
            "baseline": "SLIM_Landing_Start_v1",
            "baseline_sha256": hashlib.sha256((root / "saves/SLIM_Landing_Start_v1.sfs").read_bytes()).hexdigest(),
            "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("landing.py", "logger.py", "run_landing_trial.py")},
        }
        conn = None
        before = set((root / "logs").glob("flight_*.csv"))
        try:
            conn = krpc.connect(name="SLIM full landing experiment", rpc_port=5000, stream_port=5001)
            sc = conn.space_center
            sc.load(result["baseline"])
            conn.krpc.paused = True
            time.sleep(0.2)
            vessel = sc.active_vessel
            result["initial"] = {"ut": sc.ut, "parts": len(vessel.parts.all), "mass_kg": vessel.mass, "apoapsis_m": vessel.orbit.apoapsis_altitude, "periapsis_m": vessel.orbit.periapsis_altitude}
            if len(vessel.parts.all) != 55 or abs(vessel.mass - 3085.957) > 0.1:
                raise ValueError("Loaded baseline does not match the fixed 55-part configuration")
            result["outcome"] = run_landing(conn, full=True)
        except Exception as error:
            result["error"] = repr(error)
            traceback.print_exc()
        finally:
            if conn is not None:
                try:
                    vessel = conn.space_center.active_vessel
                    flight = vessel.flight(vessel.orbit.body.reference_frame)
                    result["final"] = {"ut": sc.ut, "situation": vessel.situation.name, "parts": len(vessel.parts.all), "mass_kg": vessel.mass, "dry_mass_kg": vessel.dry_mass, "altitude_m": flight.surface_altitude, "vertical_speed_m_s": flight.vertical_speed, "horizontal_speed_m_s": flight.horizontal_speed, "throttle": vessel.control.throttle, "thrust_N": vessel.thrust, "sas": vessel.control.sas, "paused": conn.krpc.paused}
                except Exception as error:
                    result["final_read_error"] = repr(error)
                conn.close()
            logs = set((root / "logs").glob("flight_*.csv")) - before
            if logs:
                log = max(logs, key=lambda path: path.stat().st_mtime)
                rows = list(csv.DictReader(log.open(encoding="utf-8")))
                result["log"] = log.relative_to(root).as_posix()
                result["events"] = [{key: row[key] for key in ("ut", "t", "phase", "event", "altitude", "vertical_speed", "horizontal_speed", "mass", "pad_up_component", "angular_speed", "attitude_control_active")} for row in rows if row["event"] != "sample"]
                touchdown = log.with_suffix(".touchdown.json")
                if touchdown.exists():
                    result["touchdown"] = json.loads(touchdown.read_text(encoding="utf-8"))
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(json.dumps(result.get("final", result), ensure_ascii=False), flush=True)
        if result.get("outcome") != "landed":
            raise SystemExit(1)
