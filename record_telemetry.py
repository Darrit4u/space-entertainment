"""Read-only kRPC recording for testing telemetry or observing a manual flight."""

import argparse
import math
import time

import krpc

from logger import FlightLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10, help="Wall-clock recording duration in seconds")
    parser.add_argument("--phase", default="telemetry_check", help="Label stored in the CSV")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    conn = krpc.connect(name="SLIM telemetry only", rpc_port=5000, stream_port=5001)
    try:
        recorder = FlightLogger(conn.space_center.active_vessel, conn.space_center)
        try:
            print(f"Telemetry: {recorder.filename}", flush=True)
            recorder.log(phase=args.phase, event="recording_start")
            recorder.start()
            try:
                time.sleep(args.duration)
            except KeyboardInterrupt:
                recorder.log(event="recording_interrupted")
            else:
                recorder.log(event="recording_complete")
        finally:
            recorder.close()
    finally:
        conn.close()
