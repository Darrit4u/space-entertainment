import math


class OrbitalMath:
    """SI units and radians; circular Hohmann estimates do not prove an encounter."""

    @staticmethod
    def lambert(mu, departure, arrival, duration, *, long_path=False):
        """Zero-revolution universal-variable Lambert solution, returning v0, v1.

        Bounded bisection of the time-of-flight equation, not ODE integration.
        Equations: https://docs.poliastro.space/en/stable/autoapi/poliastro/core/iod/index.html
        Collinear geometry is excluded because it does not define a unique plane.
        """
        if not math.isfinite(mu) or mu <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("mu and transfer duration must be finite and positive")
        if any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in (departure, arrival)):
            raise ValueError("Positions must have three finite components")
        r0, r1 = map(OrbitalMath.vector_norm, (departure, arrival))
        if min(r0, r1) == 0:
            raise ValueError("Positions must be nonzero")
        cosine = max(-1.0, min(1.0, OrbitalMath.dot(departure, arrival) / (r0*r1)))
        if abs(cosine) > 1-1e-12:
            raise ValueError("Collinear Lambert geometry is undefined")
        a = math.sqrt(r0*r1*(1+cosine)) * (-1 if long_path else 1)
        lower, upper = -4*math.pi**2, 4*math.pi**2-1e-6
        for _ in range(100):
            z = (lower+upper)/2
            if abs(z) < 1e-5:
                c = .5-z/24+z*z/720-z*z*z/40320
                s = 1/6-z/120+z*z/5040-z*z*z/362880
            elif z > 0:
                q = math.sqrt(z)
                c, s = (1-math.cos(q))/z, (q-math.sin(q))/q**3
            else:
                q = math.sqrt(-z)
                c, s = (math.cosh(q)-1)/(-z), (math.sinh(q)-q)/q**3
            y = r0+r1+a*(z*s-1)/math.sqrt(c)
            if y <= 0:
                lower = z
                continue
            time_of_flight = ((y/c)**1.5*s+a*math.sqrt(y))/math.sqrt(mu)
            if abs(time_of_flight-duration) < 1e-5:
                f, g, gd = 1-y/r0, a*math.sqrt(y/mu), 1-y/r1
                return (tuple((b-f*x)/g for x,b in zip(departure, arrival)),
                        tuple((gd*b-x)/g for x,b in zip(departure, arrival)))
            if time_of_flight < duration:
                lower = z
            else:
                upper = z
        raise ValueError("Lambert time-of-flight bisection did not converge")

    @staticmethod
    def hohmann_delta_v(mu, r1, r2):
        """
        Signed prograde impulses between circular coplanar orbits.
        The second impulse circularizes around the SAME body, not around Mun.
        """
        v1 = OrbitalMath.angular_velocity(mu, r1) * r1
        v2 = OrbitalMath.angular_velocity(mu, r2) * r2
        dv1 = v1 * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
        dv2 = v2 * (1 - math.sqrt(2 * r1 / (r1 + r2)))
        return dv1, dv2

    @staticmethod
    def vector_norm(v):
        return math.hypot(*v)

    @staticmethod
    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    @staticmethod
    def hohmann_phase_angle(mu, r1, r2):
        """
        Target lead angle in the direction of motion, for a circular transfer.
        """
        t_transfer = OrbitalMath.hohmann_transfer_time(mu, r1, r2)
        omega_target = OrbitalMath.angular_velocity(mu, r2)
        return (math.pi - omega_target * t_transfer) % math.tau

    @staticmethod
    def hohmann_transfer_time(mu, r1, r2):
        """Half the transfer ellipse period; radii are from the body's centre."""
        if not all(math.isfinite(r) and r > 0 for r in (r1, r2)):
            raise ValueError("Orbital radii must be finite and positive")
        return math.pi / OrbitalMath.angular_velocity(mu, (r1 + r2) / 2)

    @staticmethod
    def angular_velocity(mu: float, r: float) -> float:
        """
        Compute angular velocity for a circular orbit.

        Parameters
        ----------
        mu : float
            Gravitational parameter of central body (m^3 / s^2)
        r : float
            Orbital radius from center of body (m)

        Returns
        -------
        float
            Angular velocity (rad / s)
        """
        if not all(math.isfinite(x) and x > 0 for x in (mu, r)):
            raise ValueError("mu and orbital radius must be finite and positive")
        return math.sqrt(mu / r ** 3)

    @staticmethod
    def current_phase_angle(ship_pos, target_pos, target_velocity):
        """
        Target lead over the projected ship position in the TARGET orbital plane.
        All vectors must use the same body-centred inertial frame. The tangent
        follows target_velocity, so handedness and retrograde motion are explicit.
        Projection alone is not an encounter prediction for inclined orbits.
        """
        if any(len(v) != 3 or not all(math.isfinite(x) for x in v)
               for v in (ship_pos, target_pos, target_velocity)):
            raise ValueError("State vectors must have three finite components")
        radius = OrbitalMath.vector_norm(target_pos)
        if radius == 0:
            raise ValueError("Target position must be nonzero")
        radial = tuple(x / radius for x in target_pos)
        radial_speed = OrbitalMath.dot(target_velocity, radial)
        tangent = tuple(v - radial_speed * r for v, r in zip(target_velocity, radial))
        tangent_speed = OrbitalMath.vector_norm(tangent)
        if tangent_speed <= 1e-12 * OrbitalMath.vector_norm(target_velocity):
            raise ValueError("Target motion must define an orbital plane")
        tangent = tuple(x / tangent_speed for x in tangent)
        x, y = OrbitalMath.dot(ship_pos, radial), OrbitalMath.dot(ship_pos, tangent)
        if math.hypot(x, y) <= 1e-12 * OrbitalMath.vector_norm(ship_pos):
            raise ValueError("Ship projection in the target plane is undefined")
        return (-math.atan2(y, x)) % math.tau

    @staticmethod
    def relative_inclination(ship_pos, ship_velocity, target_pos, target_velocity):
        """Angle between oriented orbital planes; pi means opposite motion."""
        vectors = (ship_pos, ship_velocity, target_pos, target_velocity)
        if any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError("State vectors must have three finite components")
        normals = []
        for r, v in ((ship_pos, ship_velocity), (target_pos, target_velocity)):
            h = (r[1]*v[2]-r[2]*v[1], r[2]*v[0]-r[0]*v[2], r[0]*v[1]-r[1]*v[0])
            length = OrbitalMath.vector_norm(h)
            if length == 0:
                raise ValueError("Position and velocity must define an orbital plane")
            normals.append(tuple(x / length for x in h))
        return math.acos(max(-1.0, min(1.0, OrbitalMath.dot(*normals))))

    @staticmethod
    def time_to_phase_angle(
        phi_target, phi_current,
        omega_ship, omega_target
    ):
        """Next window for CONSTANT rates, with dphi/dt=omega_target-omega_ship."""
        if not all(math.isfinite(x) for x in (phi_target, phi_current, omega_ship, omega_target)):
            raise ValueError("Phase angles and angular rates must be finite")
        if abs(math.remainder(phi_target - phi_current, math.tau)) <= 1e-12:
            return 0.0
        rate = omega_target - omega_ship
        if rate == 0:
            raise ValueError("Equal angular rates cannot close a nonzero phase gap")
        distance = (phi_target-phi_current) % math.tau if rate > 0 else (phi_current-phi_target) % math.tau
        return distance / abs(rate)

    @staticmethod
    def burn_duration(delta_v, mass, thrust, specific_impulse, available_propellant):
        """Return (seconds, kg) for constant thrust/Isp and positive delta-v.

        available_propellant must include only engine-reachable fuel; the caller
        can subtract a protected reserve. This is not a trajectory integration.
        """
        if not all(math.isfinite(x) for x in (delta_v, mass, thrust, specific_impulse, available_propellant)):
            raise ValueError("Burn inputs must be finite")
        if delta_v < 0 or mass <= 0 or thrust <= 0 or specific_impulse <= 0 or not 0 <= available_propellant < mass:
            raise ValueError("Invalid delta-v, mass, thrust, Isp or available propellant")
        exhaust_speed = specific_impulse * 9.80665
        required = -mass * math.expm1(-delta_v / exhaust_speed)
        if required > available_propellant and not math.isclose(required, available_propellant, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("Burn exceeds engine-reachable propellant or protected reserve")
        return required * exhaust_speed / thrust, required

    @staticmethod
    def capture_delta_v(mu, periapsis_radius, incoming_eccentricity, target_apoapsis_radius):
        """Ideal periapsis retro-burn from a hyperbola/parabola to an ellipse."""
        if not all(math.isfinite(x) for x in (mu, periapsis_radius, incoming_eccentricity, target_apoapsis_radius)):
            raise ValueError("Capture parameters must be finite")
        if mu <= 0 or periapsis_radius <= 0 or incoming_eccentricity < 1 or target_apoapsis_radius < periapsis_radius:
            raise ValueError("Capture requires an incoming unbound orbit and apoapsis >= periapsis")
        arrival_speed = math.sqrt(mu*(1+incoming_eccentricity)/periapsis_radius)
        target_speed = math.sqrt(mu*(2/periapsis_radius-2/(periapsis_radius+target_apoapsis_radius)))
        return arrival_speed-target_speed

    @staticmethod
    def apsis_change_delta_v(mu, burn_radius, opposite_radius_before, opposite_radius_after):
        """Signed tangential impulse at an apsis, preserving that apsis radius."""
        if not all(math.isfinite(x) and x > 0 for x in (mu,burn_radius,opposite_radius_before,opposite_radius_after)):
            raise ValueError("Apsis radii and mu must be finite and positive")
        before = math.sqrt(mu*(2/burn_radius-2/(burn_radius+opposite_radius_before)))
        after = math.sqrt(mu*(2/burn_radius-2/(burn_radius+opposite_radius_after)))
        return after-before

