import krpc
import math
import time

from logger import FlightLogger
from math_logic import OrbitalMath


class SlimAutopilot:

    def __init__(self):
        self.conn = krpc.connect(name='SLIM OOP Autopilot', rpc_port=5000, stream_port=5001)
        self.sc = self.conn.space_center
        self.vessel = self.sc.active_vessel

        self.kerbin = self.sc.bodies['Kerbin']
        self.mun = self.sc.bodies['Mun']

        self.orbital_rf = self.kerbin.non_rotating_reference_frame
        self.surface_rf = self.vessel.surface_reference_frame

        self.ap = self.vessel.auto_pilot
        self.ctrl = self.vessel.control

        # Constants
        self.mu = self.kerbin.gravitational_parameter
        # Circular-transfer approximation uses the measured initial orbit.
        self.r1 = self.vessel.orbit.semi_major_axis
        self.r2 = self.mun.orbit.semi_major_axis

        self.phase_target = OrbitalMath.required_phase_angle(self.r1, self.r2)
        self.phase_tolerance = math.radians(1)

        # Landing PID
        self.KP = 0.04
        self.KD = 0.2

        self.launch_ut: float = 0.0
        self.logger = FlightLogger(self.vessel, self.sc)

    # --------------------------------------------------
    # PHASE 1 — PHASE ANGLE WAIT
    # --------------------------------------------------
    def phase_1_compute_launch_time(self):
        if self.vessel.orbit.body != self.kerbin:
            raise ValueError('The full mission must start from a Kerbin orbit')
        self.logger.log(phase='planning', event='phase_start')
        print('[Phase 1] Computing correct launch UT')

        r_ship = self.r1
        r_mun = self.r2

        omega_ship = OrbitalMath.angular_velocity(self.mu, r_ship)
        omega_mun = OrbitalMath.angular_velocity(self.mu, r_mun)

        phi_target = OrbitalMath.hohmann_phase_angle(
            self.mu, r_ship, r_mun
        )
        ship_pos = self.vessel.position(self.orbital_rf)
        mun_pos = self.mun.position(self.orbital_rf)

        phi_current = OrbitalMath.current_phase_angle(
            ship_pos, mun_pos
        )

        t_wait = OrbitalMath.time_to_phase_angle(
            phi_target, phi_current,
            omega_ship, omega_mun
        )

        self.launch_ut = self.sc.ut + t_wait

        print(
            f'[Phase 1] Launch in {t_wait / 60:.1f} min '
            f'at UT={self.launch_ut:.0f}'
        )

    # --------------------------------------------------
    # PHASE 2 — TRANSFER
    # --------------------------------------------------
    def phase_2_transfer(self):
        self.logger.log(phase='transfer', event='phase_start')
        print('[Phase 2] Executing timed transfer burn')

        dv1, _ = OrbitalMath.hohmann_delta_v(
            self.mu, self.r1, self.r2
        )

        node = self.vessel.control.add_node(
            ut=self.launch_ut,
            prograde=dv1
        )
        self.ap.engage()
        self.ap.reference_frame = node.reference_frame
        self.ap.target_direction = (0, 1, 0)
        self.ap.wait()

        print('[Phase 2] Waiting for burn time')

        while self.sc.ut < self.launch_ut - 20:
            time.sleep(0.2)

        self.logger.log(event='burn_start')
        self.ctrl.throttle = 1.0
        while node.remaining_delta_v > 15:
            time.sleep(0.1)

        self.ctrl.throttle = 0.0
        self.logger.log(event='burn_end')
        node.remove()

        print('[Phase 2] Transfer burn completed')

    # --------------------------------------------------
    # PHASE 3 — MUN CAPTURE
    # --------------------------------------------------
    def phase_3_mun_capture(self):
        self.logger.log(phase='mun_coast', event='phase_start')
        print('[Phase 3] Mun SOI capture')

        # --- 1. Wait for Mun SOI ---
        while self.vessel.orbit.body != self.mun:
            time.sleep(1.0)

        print('[Phase 3] Entered Mun SOI')
        self.logger.log(phase='capture', event='phase_start')

        mu_mun = self.mun.gravitational_parameter

        # Mun-centered reference frame
        mun_rf = self.mun.reference_frame

        # Target periapsis altitude for capture
        target_pe = self.mun.equatorial_radius + 15_000  # 15 km

        self.ap.engage()
        self.ap.reference_frame = mun_rf
        self.ap.target_direction = (0, 1, 0)  # retrograde

        print('[Phase 3] Waiting for periapsis')

        while self.vessel.orbit.periapsis > target_pe:
            time.sleep(0.5)

        print('[Phase 3] Performing capture burn')

        while True:
            r = self.vessel.orbit.radius
            v_vec = self.vessel.velocity(mun_rf)
            v = math.sqrt(sum(c * c for c in v_vec))

            specific_energy = v ** 2 / 2 - mu_mun / r

            if specific_energy < -100:  # safely bound
                break

            self.ctrl.throttle = 0.4
            time.sleep(0.1)

        self.ctrl.throttle = 0.0

        print('[Phase 3] Mun capture complete')

    # --------------------------------------------------
    # PHASE 4 — LOW ORBIT
    # --------------------------------------------------
    def phase_4_low_orbit(self):
        self.logger.log(phase='low_orbit', event='phase_start')
        print('[Phase 4] Transition to low Mun orbit')

        vessel = self.vessel
        mun = self.mun
        sc = self.sc

        # Mun-centered reference frame
        mun_rf = mun.reference_frame

        # Target low orbit altitude
        target_alt = 17_000  # 17 km

        # Engage autopilot to maintain prograde in Mun frame
        ap = vessel.auto_pilot
        ap.engage()
        ap.reference_frame = mun_rf
        ap.target_direction = (0, 1, 0)  # prograde
        ap.wait()

        # Flight telemetry in Mun frame
        flight = vessel.flight(mun_rf)

        print('[Phase 4] Circularizing and lowering orbit to target altitude')

        while True:
            # Current altitude above Mun
            alt = flight.surface_altitude

            # Relative velocity to Mun (vector)
            v_vec = vessel.velocity(mun_rf)
            v_rel = math.sqrt(sum(c * c for c in v_vec))

            # Mun circular orbit velocity at current radius
            r = mun.equatorial_radius + alt
            v_circ = math.sqrt(mun.gravitational_parameter / r)

            # Delta-v required to circularize
            delta_v = v_rel - v_circ

            # Adjust throttle based on delta-v (simple proportional control)
            Kp = 0.2
            throttle = max(0.0, min(Kp * delta_v, 1.0))
            vessel.control.throttle = throttle

            # Debug output
            print(f'ALT={alt:6.0f} m | V_rel={v_rel:6.2f} m/s | '
                  f'V_circ={v_circ:6.2f} m/s | ΔV={delta_v:6.2f} m/s | Thr={throttle:.2f}')

            # Stop when close to target altitude and velocity
            if abs(alt - target_alt) < 50 and abs(delta_v) < 1.0:
                vessel.control.throttle = 0.0
                print('[Phase 4] Low Mun orbit achieved')
                break

            time.sleep(0.1)

    def phase_5_landing(self):
        self.logger.log(phase='landing', event='phase_start')
        print('[Phase 5] SLIM-style powered descent (API CORRECT)')

        vessel = self.vessel
        mun = self.mun
        sc = self.sc

        # --- Correct reference frames ---
        surf_rf = mun.reference_frame
        surf_vel_rf = vessel.surface_velocity_reference_frame  # ✅ FIX

        flight = vessel.flight(surf_rf)

        # --- Autopilot: retrograde to surface velocity ---
        ap = vessel.auto_pilot
        ap.engage()
        ap.reference_frame = surf_vel_rf
        ap.target_direction = (0, -1, 0)  # retrograde to surface velocity
        ap.wait()

        # --- PID gains ---
        Kp = 0.6
        Kd = 0.25

        prev_error = 0.0
        prev_time = sc.ut

        # --- Ignite engine ---
        vessel.control.throttle = 0.15
        time.sleep(0.5)

        max_thrust = vessel.max_thrust
        g = mun.surface_gravity

        print(f'[Phase 5] Max thrust: {max_thrust:.0f} N')

        while True:
            alt = flight.surface_altitude
            v_vert = flight.vertical_speed  # +up, -down

            # --- SLIM vertical velocity profile ---
            if alt > 5000:
                v_target = -50.0
            elif alt > 1000:
                v_target = -10.0
            elif alt > 100:
                v_target = -2.0
            else:
                v_target = -0.3

            self.logger.target_vertical_speed = v_target

            # --- PID ---
            error = v_target - v_vert

            now = sc.ut
            dt = max(now - prev_time, 1e-3)
            d_error = (error - prev_error) / dt

            prev_error = error
            prev_time = now

            # --- Required vertical acceleration ---
            a_cmd = g + Kp * error + Kd * d_error
            a_cmd = max(a_cmd, 0.0)

            # --- Acceleration → throttle ---
            mass = vessel.mass
            throttle = (a_cmd * mass) / max_thrust
            throttle = max(0.1, min(throttle, 1.0))

            vessel.control.throttle = throttle

            # --- Debug ---
            # print(
            #     f'ALT={alt:6.0f} m | '
            #     f'Vv={v_vert:6.2f} | '
            #     f'Tgt={v_target:5.1f} | '
            #     f'Thr={throttle:.2f}'
            # )

            # --- Touchdown ---
            if alt < 0.8 and abs(v_vert) < 0.5:
                vessel.control.throttle = 0.0
                self.logger.log(event='touchdown_condition')
                ap.disengage()
                print('[Phase 5] Touchdown confirmed')
                break

            time.sleep(0.05)

    # --------------------------------------------------
    # RUN MISSION
    # --------------------------------------------------
    def run(self):
        print(f'Telemetry: {self.logger.filename}')
        self.logger.start()
        try:
            self.phase_1_compute_launch_time()
            self.phase_2_transfer()
            self.phase_3_mun_capture()
            self.phase_4_low_orbit()
            self.phase_5_landing()
            self.logger.log(event='mission_complete')
        finally:
            # Stop thrust even if the flight or telemetry raises an exception.
            try:
                self.ctrl.throttle = 0.0
            finally:
                try:
                    self.logger.close()
                finally:
                    self.conn.close()


# ======================================================
# MAIN
# ======================================================
if __name__ == '__main__':
    slim = SlimAutopilot()
    slim.run()
