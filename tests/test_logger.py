import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from logger import FlightLogger


class FlightLoggerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="logger-test-", dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "flight.csv"
        body = SimpleNamespace(
            name="Mun", reference_frame="surface", non_rotating_reference_frame="inertial",
            gravitational_parameter=6.5138398e10, equatorial_radius=200000,
            rotational_period=138984, surface_gravity=1.629,
            orbit=SimpleNamespace(semi_major_axis=12000000),
        )
        self.sc = SimpleNamespace(ut=100.0, bodies={"Kerbin": body, "Mun": body})
        flight = SimpleNamespace(
            surface_altitude=1000, mean_altitude=2000, latitude=1, longitude=2,
            speed=7, vertical_speed=-3, horizontal_speed=6,
        )
        self.vessel = SimpleNamespace(
            name="SLIM", mass=10, dry_mass=5, thrust=20,
            available_thrust=40, specific_impulse=345,
            control=SimpleNamespace(throttle=0.5),
            situation=SimpleNamespace(name="orbiting"),
            resources=SimpleNamespace(amount=lambda name: 10),
            flight=lambda frame: flight,
            velocity=lambda frame: (3, 4, 0) if frame == "inertial" else (0, 0, 7),
            position=lambda frame: (202000, 0, 0),
            orbit=SimpleNamespace(
                body=body, radius=202000, semi_major_axis=210000,
                eccentricity=0.01, inclination=0.1,
                apoapsis_altitude=12000, periapsis_altitude=8000,
            ),
        )

    def test_ut_pause_and_warp_integral(self):
        logger = FlightLogger(self.vessel, self.sc, self.path)
        try:
            self.assertEqual(logger.log()["delta_v"], 0)
            self.sc.ut += 60  # UT can advance much faster than wall time.
            row = logger.log()
            self.assertEqual(row["t"], 60)
            self.assertEqual(row["sample_dt"], 60)
            self.assertEqual(row["delta_v"], 120)
            self.assertEqual(logger.log()["delta_v"], 120)  # Paused UT.
            self.vessel.thrust = 0
            self.sc.ut += 5
            self.assertEqual(logger.log()["delta_v"], 125)
        finally:
            logger.close()

    def test_frames_phases_and_initial_conditions_are_saved(self):
        logger = FlightLogger(self.vessel, self.sc, self.path)
        try:
            for phase in ("planning", "transfer", "mun_coast", "capture", "low_orbit", "landing"):
                logger.log(phase=phase, event="phase_start")
            logger.target_vertical_speed = -2
            row = logger.log()
            self.assertEqual(row["speed"], 5)
            self.assertEqual(row["surface_speed"], 7)
            self.assertEqual(row["altitude"], 1000)
            self.assertEqual(row["mean_altitude"], 2000)
            self.assertEqual(row["target_vertical_speed"], -2)
            # Flushed records must be readable before close, even if a run crashes.
            with self.path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[-1]["phase"], "landing")
            metadata = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["initial_telemetry"]["ut"], 100)
            self.assertEqual(metadata["initial_velocity_inertial_m_s"], [3, 4, 0])
        finally:
            logger.close()

    def test_existing_experiment_is_never_overwritten(self):
        self.path.write_text("earlier experiment", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            FlightLogger(self.vessel, self.sc, self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "earlier experiment")

    def test_loading_earlier_save_is_rejected(self):
        logger = FlightLogger(self.vessel, self.sc, self.path)
        try:
            self.sc.ut = 99
            with self.assertRaisesRegex(ValueError, "time moved backwards"):
                logger.log()
        finally:
            logger.close()

    def test_soi_switch_uses_new_central_body_frame(self):
        logger = FlightLogger(self.vessel, self.sc, self.path)
        try:
            self.vessel.orbit.body = SimpleNamespace(
                name="Kerbin", reference_frame="kerbin_surface",
                non_rotating_reference_frame="kerbin_inertial",
            )
            self.vessel.velocity = lambda frame: (5, 12, 0) if frame == "kerbin_inertial" else (0, 0, 0)
            self.sc.ut += 1
            row = logger.log(phase="transfer")
            self.assertEqual(row["body"], "Kerbin")
            self.assertEqual(row["speed"], 13)
            self.assertEqual(row["sample_dt"], 1)
        finally:
            logger.close()

    def test_background_recording_covers_waits_and_closes(self):
        logger = FlightLogger(self.vessel, self.sc, self.path, interval=0.01)
        logger.start()
        # Synchronize with a successful sample instead of relying on a fixed sleep.
        import time
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.path.open(encoding="utf-8", newline="") as file:
                if len(list(csv.DictReader(file))) >= 2:
                    break
            time.sleep(0.01)
        logger.close()
        with self.path.open(encoding="utf-8", newline="") as file:
            self.assertGreaterEqual(len(list(csv.DictReader(file))), 2)
        self.assertFalse(logger._thread.is_alive())
        self.assertTrue(logger.file.closed)

    def test_background_failure_is_reported_on_close(self):
        logger = FlightLogger(self.vessel, self.sc, self.path, interval=0.01)
        self.sc.ut = 99
        with contextlib.redirect_stdout(io.StringIO()):
            logger.start()
            self.assertTrue(logger._stop.wait(2))
        with self.assertRaisesRegex(RuntimeError, "CSV is incomplete"):
            logger.close()
        self.assertTrue(logger.file.closed)


if __name__ == "__main__":
    unittest.main()
