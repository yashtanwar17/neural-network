import math
import random
import tkinter as tk


def fade_color(hex_color: str, alpha: float) -> str:
    """Fades a #RRGGBB color toward pure black (#000000)."""
    t = max(0.0, min(1.0, alpha))
    r = int(int(hex_color[1:3], 16) * t)
    g = int(int(hex_color[3:5], 16) * t)
    b = int(int(hex_color[5:7], 16) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class RocketLandingEnv:
    """
    2D Lunar Rocket Landing Environment.

    Action (3 values): [left_thruster, right_thruster, up_thruster] in {0, 1} or [0.0..1.0]
    State  (8 values): [x, y, vx, vy, angle, angular_vel, pad_x, dx_to_pad]
    """

    def __init__(self, randomize_pad: bool = False):
        self.randomize_pad = randomize_pad

        # Physics parameters (x in [-1, 1], y in [0, 1.45])
        self.dt = 0.02
        self.gravity = 0.45
        self.main_thrust = 1.15
        self.side_thrust = 0.22
        self.side_torque = 2.80
        self.linear_damping = 0.995
        self.angular_damping = 0.97

        # Landing zone & touchdown thresholds
        self.pad_width = 0.24
        self.pad_x = 0.0
        self.max_steps = 500
        self.max_landing_vy = 0.28
        self.max_landing_vx = 0.22
        self.max_landing_angle = 0.22

        # Window state
        self.width = 900
        self.height = 600
        self.root = None
        self.canvas = None
        self.closed = False

        self.terrain = []
        self.particles = []
        self.last_action = [0.0, 0.0, 0.0]
        self.status = "FLYING"  # FLYING | LANDED | HARD LANDING | CRASHED

        self.reset()

    def _build_terrain(self):
        left_flag = self.pad_x - self.pad_width / 2.0
        right_flag = self.pad_x + self.pad_width / 2.0

        left_xs = [x for x in (-1.10, -0.78, -0.45, -0.24) if x < left_flag - 0.05] + [left_flag]
        left_pts = [
            (x, 0.0 if x == left_flag else random.uniform(-0.03, 0.14))
            for x in left_xs
        ]

        right_xs = [right_flag] + [x for x in (0.24, 0.48, 0.80, 1.10) if x > right_flag + 0.05]
        right_pts = [
            (x, 0.0 if x == right_flag else random.uniform(-0.08, 0.09))
            for x in right_xs
        ]

        self.terrain = left_pts + right_pts

    def _terrain_height(self, wx: float) -> float:
        if wx <= self.terrain[0][0]:
            return self.terrain[0][1]
        if wx >= self.terrain[-1][0]:
            return self.terrain[-1][1]

        for i in range(len(self.terrain) - 1):
            x1, y1 = self.terrain[i]
            x2, y2 = self.terrain[i + 1]
            if x1 <= wx <= x2:
                t = (wx - x1) / max(1e-6, x2 - x1)
                return y1 + t * (y2 - y1)
        return 0.0

    def reset(self) -> list[float]:
        self.pad_x = random.uniform(-0.42, 0.42) if self.randomize_pad else 0.0
        self._build_terrain()

        self.x = random.uniform(-0.50, 0.50)
        self.y = random.uniform(1.12, 1.30)
        self.vx = random.uniform(-0.12, 0.12)
        self.vy = random.uniform(-0.06, 0.0)
        self.angle = random.uniform(-0.18, 0.18)
        self.angular_vel = random.uniform(-0.08, 0.08)

        self.steps = 0
        self.last_action = [0.0, 0.0, 0.0]
        self.status = "FLYING"
        self.particles.clear()
        self.prev_shaping = self._shaping()
        return self.get_state()

    def get_state(self) -> list[float]:
        return [
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.angle,
            self.angular_vel,
            self.pad_x,
            self.x - self.pad_x,
        ]

    def _shaping(self) -> float:
        dist = math.hypot(self.x - self.pad_x, self.y)
        speed = math.hypot(self.vx, self.vy)
        return -110.0 * dist - 65.0 * speed - 85.0 * abs(self.angle)

    def _emit_particles(self, left: float, right: float, up: float):
        if self.canvas is None or self.steps % 2 != 0:
            return

        sin_a = math.sin(self.angle)
        cos_a = math.cos(self.angle)

        if up > 0.1:
            spread = random.uniform(-0.14, 0.14)
            speed = random.uniform(0.55, 0.85)
            self.particles.append({
                "x": self.x + sin_a * 0.01,
                "y": self.y - cos_a * 0.01,
                "vx": self.vx + math.sin(self.angle + spread) * speed,
                "vy": self.vy - math.cos(self.angle + spread) * speed,
                "life": 1.0,
                "decay": 0.042,
                "r": 3.0,
                "color": "#f07178",
            })

        if left > 0.1:
            self.particles.append({
                "x": self.x - cos_a * 0.03 - sin_a * 0.03,
                "y": self.y - sin_a * 0.03 + cos_a * 0.03,
                "vx": self.vx - cos_a * 0.55,
                "vy": self.vy - sin_a * 0.55 + random.uniform(-0.04, 0.04),
                "life": 1.0,
                "decay": 0.075,
                "r": 2.2,
                "color": "#89ddff",
            })

        if right > 0.1:
            self.particles.append({
                "x": self.x + cos_a * 0.03 - sin_a * 0.03,
                "y": self.y + sin_a * 0.03 + cos_a * 0.03,
                "vx": self.vx + cos_a * 0.55,
                "vy": self.vy + sin_a * 0.55 + random.uniform(-0.04, 0.04),
                "life": 1.0,
                "decay": 0.075,
                "r": 2.2,
                "color": "#89ddff",
            })

    def _step_particles(self):
        alive = []
        for p in self.particles:
            p["x"] += p["vx"] * self.dt
            p["y"] += p["vy"] * self.dt
            p["life"] -= p["decay"]
            if p["life"] > 0.0 and p["y"] > self._terrain_height(p["x"]):
                alive.append(p)
        self.particles = alive

    def step(self, action: list[float]):
        left = max(0.0, min(1.0, float(action[0])))
        right = max(0.0, min(1.0, float(action[1])))
        up = max(0.0, min(1.0, float(action[2])))
        self.last_action = [left, right, up]
        self.steps += 1

        sin_a = math.sin(self.angle)
        cos_a = math.cos(self.angle)

        ax = -sin_a * (up * self.main_thrust)
        ay = cos_a * (up * self.main_thrust) - self.gravity

        net_side = left - right
        ax += cos_a * (net_side * self.side_thrust)
        ay += sin_a * (net_side * self.side_thrust)
        angular_acc = -net_side * self.side_torque

        self.vx = (self.vx + ax * self.dt) * self.linear_damping
        self.vy = (self.vy + ay * self.dt) * self.linear_damping
        self.angular_vel = (self.angular_vel + angular_acc * self.dt) * self.angular_damping

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.angle += self.angular_vel * self.dt

        self._emit_particles(left, right, up)

        shaping = self._shaping()
        reward = (shaping - self.prev_shaping) - 0.12 * up - 0.03 * (left + right)
        self.prev_shaping = shaping

        done = False
        info = {"landed": False}
        ground_y = self._terrain_height(self.x)

        if abs(self.x) > 1.05 or self.y > 1.60 or abs(self.angle) > (math.pi / 2.0):
            reward -= 100.0
            done = True
            self.status = "CRASHED"
        elif self.steps >= self.max_steps:
            reward -= 50.0
            done = True
            self.status = "TIMEOUT"
        elif self.y <= ground_y:
            self.y = ground_y
            done = True

            between_flags = abs(self.x - self.pad_x) <= (self.pad_width / 2.0)
            soft_landing = abs(self.vy) < self.max_landing_vy and abs(self.vx) < self.max_landing_vx
            upright = abs(self.angle) < self.max_landing_angle

            if between_flags and soft_landing and upright:
                reward += 160.0
                info["landed"] = True
                self.status = "LANDED"
            elif between_flags and upright:
                reward += 15.0
                self.status = "HARD LANDING"
            else:
                reward -= 100.0
                self.status = "CRASHED"

        return self.get_state(), reward, done, info

    def _init_window(self):
        self.root = tk.Tk()
        self.root.title("Lunar Lander")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root, width=self.width, height=self.height, bg="#000000", highlightthickness=0
        )
        self.canvas.pack()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.focus_force()

    def _on_close(self):
        self.closed = True
        if self.root is not None:
            self.root.destroy()

    def render(self):
        if self.closed:
            return
        if self.root is None:
            self._init_window()

        self._step_particles()
        self.canvas.delete("all")

        pad_screen_y = self.height - 95
        scale_x = self.width / 2.0
        scale_y = (pad_screen_y - 45) / 1.40

        def to_screen(wx: float, wy: float) -> tuple[float, float]:
            return self.width / 2.0 + wx * scale_x, pad_screen_y - wy * scale_y

        # Exhaust particles
        for p in self.particles:
            sx, sy = to_screen(p["x"], p["y"])
            col = fade_color(p["color"], p["life"])
            r = p["r"] * (0.45 + 0.55 * p["life"])
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=col, outline="")

        # Moon surface
        pts = [to_screen(wx, wy) for wx, wy in self.terrain]
        moon_poly = [(0, self.height)] + pts + [(self.width, self.height)]
        self.canvas.create_polygon(moon_poly, fill="#f8fafc", outline="")

        # Two flags marking the landing zone
        p1 = to_screen(self.pad_x - self.pad_width / 2.0, 0.0)
        p2 = to_screen(self.pad_x + self.pad_width / 2.0, 0.0)
        for fx, fy in (p1, p2):
            self.canvas.create_line(fx, fy, fx, fy - 46, fill="#ffffff", width=2)
            self.canvas.create_polygon(
                fx + 1, fy - 46, fx + 25, fy - 38, fx + 1, fy - 30, fill="#e5c100", outline=""
            )

        # Lander
        rx, ry = to_screen(self.x, self.y)
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        def rot(lx: float, ly: float) -> tuple[float, float]:
            return rx + (lx * cos_a - ly * sin_a), ry - (lx * sin_a + ly * cos_a)

        # Legs
        self.canvas.create_polygon(rot(-13, 15), rot(-17, 14), rot(-21, 0), rot(-17, 0), fill="#7c6ce8", outline="")
        self.canvas.create_polygon(rot(13, 15), rot(17, 14), rot(21, 0), rot(17, 0), fill="#7c6ce8", outline="")

        # Body
        self.canvas.create_polygon(rot(-14, 5), rot(14, 5), rot(11, 27), rot(-11, 27), fill="#7c6ce8", outline="")

        # End status label
        if self.status != "FLYING":
            color = "#4ade80" if self.status == "LANDED" else ("#facc15" if self.status == "HARD LANDING" else "#f87171")
            self.canvas.create_text(
                self.width / 2, self.height * 0.35, text=self.status, fill=color, font=("Consolas", 20, "bold")
            )

        self.root.update_idletasks()
        self.root.update()


def run_manual_sim():
    env = RocketLandingEnv(randomize_pad=False)
    env.render()

    pressed = set()
    done = False

    def on_key_press(event):
        nonlocal done
        key = event.keysym.lower()
        pressed.add(key)
        if key == "r":
            env.reset()
            done = False
        elif key == "p":
            env.randomize_pad = not env.randomize_pad
            env.reset()
            done = False

    def on_key_release(event):
        pressed.discard(event.keysym.lower())

    env.root.bind("<KeyPress>", on_key_press)
    env.root.bind("<KeyRelease>", on_key_release)

    def loop():
        nonlocal done
        if env.closed:
            return

        left = 1 if ("left" in pressed or "a" in pressed) else 0
        right = 1 if ("right" in pressed or "d" in pressed) else 0
        up = 1 if ("up" in pressed or "w" in pressed) else 0

        if not done:
            _, _, done, _ = env.step([left, right, up])

        env.render()
        env.root.after(20, loop)

    loop()
    env.root.mainloop()


if __name__ == "__main__":
    run_manual_sim()
