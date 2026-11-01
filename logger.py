import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading


class FlightLogger:
    """Record telemetry without changing vessel controls; elapsed time is KSP UT."""

    def __init__(self, vessel, space_center, filename=None, interval=0.2):
        if interval <= 0:
            raise ValueError("Logging interval must be positive")
        self.vessel = vessel
        self.sc = space_center
        self.interval = interval
        if filename is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            filename = Path(__file__).parent / "logs" / f"flight_{stamp}.csv"
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.start_ut = self.sc.ut
        self.last_ut = None
        self.last_acceleration = 0.0
        self.delta_v = 0.0
        self.phase = "initial"
        self.target_vertical_speed = None
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self.file = self.filename.open("x", newline="", encoding="utf-8")
        fields = [
            "t", "ut", "phase", "event", "body", "situation",
            "altitude", "mean_altitude", "radius", "latitude", "longitude",
            "speed", "surface_speed", "vertical_speed", "horizontal_speed",
            "target_vertical_speed", "mass", "dry_mass", "throttle", "thrust",
            "available_thrust", "specific_impulse", "liquid_fuel", "oxidizer",
            "electric_charge", "apoapsis", "periapsis", "delta_v", "sample_dt",
        ]
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        try:
            initial = self.log(event="run_start")
            body = vessel.orbit.body
            metadata = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "vessel": vessel.name,
                "sampling_wall_interval_s": interval,
                "time_basis": "KSP universal time (seconds)",
                "speed_frame": "current body's non_rotating_reference_frame",
                "surface_speed_frame": "current body's reference_frame",
                "altitude_basis": "surface_altitude; mean_altitude is logged separately",
                "delta_v_method": "trapezoidal integral of thrust/mass over KSP UT",
                "initial_telemetry": initial,
                "initial_position_inertial_m": vessel.position(body.non_rotating_reference_frame),
                "initial_velocity_inertial_m_s": vessel.velocity(body.non_rotating_reference_frame),
                "initial_orbit": {
                    "semi_major_axis_m": vessel.orbit.semi_major_axis,
                    "eccentricity": vessel.orbit.eccentricity,
                    "inclination_rad": vessel.orbit.inclination,
                },
                "bodies": {},
            }
            for name in ("Kerbin", "Mun"):
                celestial_body = self.sc.bodies[name]
                metadata["bodies"][name] = {
                    "mu_m3_s2": celestial_body.gravitational_parameter,
                    "equatorial_radius_m": celestial_body.equatorial_radius,
                    "rotational_period_s": celestial_body.rotational_period,
                    "surface_gravity_m_s2": celestial_body.surface_gravity,
                }
            metadata["bodies"]["Mun"]["orbit_semi_major_axis_m"] = self.sc.bodies["Mun"].orbit.semi_major_axis
            with self.filename.with_suffix(".json").open("x", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)
        except BaseException:
            self.file.close()
            raise

    def start(self):
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("Create a new logger for each run")
        self._thread = threading.Thread(target=self._record, name="flight-telemetry", daemon=True)
        self._thread.start()

    def _record(self):
        try:
            while not self._stop.wait(self.interval):
                self.log()
        except Exception as error:
            self.error = error
            self._stop.set()
            print(f"Telemetry recording failed: {error}", flush=True)

    def log(self, phase=None, event="sample"):
        with self._lock:
            if self.error is not None:
                raise RuntimeError("Telemetry recording failed") from self.error
            if phase is not None:
                self.phase = phase
            now = self.sc.ut
            dt = 0.0 if self.last_ut is None else now - self.last_ut
            if dt < 0:
                raise ValueError("KSP time moved backwards; start a new log after loading a save")
            orbit = self.vessel.orbit
            body = orbit.body
            flight = self.vessel.flight(body.reference_frame)
            thrust = self.vessel.thrust
            mass = self.vessel.mass
            if mass <= 0:
                raise ValueError("Vessel mass must be positive")
            acceleration = thrust / mass
            delta_v = self.delta_v + 0.5 * (self.last_acceleration + acceleration) * dt
            resources = self.vessel.resources
            row = {
                "t": now - self.start_ut, "ut": now,
                "phase": self.phase, "event": event, "body": body.name,
                "situation": self.vessel.situation.name,
                "altitude": flight.surface_altitude, "mean_altitude": flight.mean_altitude,
                "radius": orbit.radius, "latitude": flight.latitude, "longitude": flight.longitude,
                "speed": math.sqrt(sum(x * x for x in self.vessel.velocity(body.non_rotating_reference_frame))),
                "surface_speed": flight.speed, "vertical_speed": flight.vertical_speed,
                "horizontal_speed": flight.horizontal_speed,
                "target_vertical_speed": self.target_vertical_speed,
                "mass": mass, "dry_mass": self.vessel.dry_mass,
                "throttle": self.vessel.control.throttle, "thrust": thrust,
                "available_thrust": self.vessel.available_thrust,
                "specific_impulse": self.vessel.specific_impulse,
                "liquid_fuel": resources.amount("LiquidFuel"),
                "oxidizer": resources.amount("Oxidizer"),
                "electric_charge": resources.amount("ElectricCharge"),
                "apoapsis": orbit.apoapsis_altitude, "periapsis": orbit.periapsis_altitude,
                "delta_v": delta_v, "sample_dt": dt,
            }
            self.writer.writerow(row)
            self.file.flush()
            self.last_ut = now
            self.last_acceleration = acceleration
            self.delta_v = delta_v
            return row

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Telemetry worker did not stop; check the kRPC connection")
        self.file.close()
        if self.error is not None:
            raise RuntimeError("Telemetry recording failed; the CSV is incomplete") from self.error
