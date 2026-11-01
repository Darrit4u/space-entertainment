import math


class OrbitalMath:
    """All orbital mechanics formulas are isolated here"""

    @staticmethod
    def hohmann_delta_v(mu, r1, r2):
        """
        Returns ΔV1 and ΔV2 for a Hohmann-like transfer
        """
        dv1 = math.sqrt(mu / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
        dv2 = math.sqrt(mu / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2)))
        return dv1, dv2

    @staticmethod
    def vector_norm(v):
        return math.sqrt(sum(x * x for x in v))

    @staticmethod
    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    @staticmethod
    def hohmann_phase_angle(mu, r1, r2):
        """
        True target phase angle for Hohmann transfer
        """
        a = (r1 + r2) / 2
        t_transfer = math.pi * math.sqrt(a ** 3 / mu)
        omega_target = math.sqrt(mu / r2 ** 3)
        return math.pi - omega_target * t_transfer

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
        return math.sqrt(mu / r ** 3)

    @staticmethod
    def orbital_angle(position):
        """
        Angular position in orbital plane (rad)
        """
        x, y, _ = position
        return math.atan2(y, x)

    @staticmethod
    def current_phase_angle(ship_pos, target_pos):
        """
        Phase angle target-ship (rad), normalized to [0, 2π]
        """
        theta_ship = OrbitalMath.orbital_angle(ship_pos)
        theta_target = OrbitalMath.orbital_angle(target_pos)

        phase = theta_target - theta_ship
        return (phase + 2 * math.pi) % (2 * math.pi)

    @staticmethod
    def time_to_phase_angle(
        phi_target, phi_current,
        omega_ship, omega_target
    ):
        dphi = (phi_target - phi_current + 2*math.pi) % (2*math.pi)
        domega = omega_ship - omega_target
        return dphi / domega

    @staticmethod
    def required_phase_angle(r1, r2):
        """
        Approximate optimal phase angle for Mun transfer
        Result ~ 44–47 degrees
        """
        return math.pi * (1 - math.sqrt((2 * r2) / (r1 + r2)))

