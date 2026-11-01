import time

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(0, 1)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0:
            return 0.0

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.prev_error = error
        self.prev_time = now

        return max(self.min_out, min(self.max_out, output))
