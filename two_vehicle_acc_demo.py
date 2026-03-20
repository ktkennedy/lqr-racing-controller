"""
Two-Vehicle Adaptive Cruise Control Demo - Real-time Interactive Simulation
USAFE Racing Team

Simulates two vehicles on an oval track:
  - Lead vehicle (orange): constant 60 km/h with pure pursuit steering
  - Ego vehicle (red): LQR lateral controller + ACC longitudinal controller

ACC Modes:
  CRUISE  - No lead vehicle in range (>100m), cruise at set speed
  FOLLOW  - Lead detected, maintaining 2s time gap + 10m min distance
  BRAKE   - Emergency braking when gap is closing too fast

Run: python3 demo/two_vehicle_acc_demo.py
Requirements: numpy, matplotlib, scipy

Controls:
  Space    - Pause / Resume
  Up/Dn    - Ego cruise speed ±10 km/h
  L/K      - Lead vehicle speed ±5 km/h
  R        - Reset simulation
  Q/Esc    - Quit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import solve_discrete_are

# ── Vehicle Parameters ──────────────────────────────────────────────
WHEELBASE = 2.72
LF = 1.58
LR = 1.59
MASS = 1700.0
DT = 0.04          # 25 Hz
LAG_TAU = 0.20     # actuator lag
DELAY_SEC = 0.20
DELAY_STEP = int(DELAY_SEC / DT)  # = 5
PREVIEW_STEP = 50

IZ = LF * LR * MASS
CF = MASS * (LF / (LF + LR)) * 0.5 * 9.81 * 0.165 * 180 / np.pi
CR = MASS * (LR / (LF + LR)) * 0.5 * 9.81 * 0.165 * 180 / np.pi

SIGMA1 = 2.0 * (CF + CR)
SIGMA2 = -2.0 * (LF * CF - LR * CR)
SIGMA3 = -2.0 * (LF**2 * CF + LR**2 * CR)

# ── ACC Parameters (from acc_test.cpp) ──────────────────────────────
ACC_Q_DIS = 50.0   # distance error weight
ACC_Q_VEL = 20.0   # velocity error weight
ACC_R     = 1.0    # control effort weight
ACC_LAG_TAU = 0.6  # longitudinal lag
TIME_GAP  = 1.0    # seconds (racing but safe)
MIN_DIST  = 8.0    # meters
ACC_RANGE = 120.0  # detection range (m)
ACC_MAX_ACCEL = 3.0   # m/s²
ACC_MAX_DECEL = -7.0  # m/s² (strong braking capability)

# ── Pure Pursuit Parameters ─────────────────────────────────────────
LOOKAHEAD_DIST = 15.0  # m

# ── ACC Mode Constants ──────────────────────────────────────────────
MODE_CRUISE = 0
MODE_FOLLOW = 1
MODE_BRAKE  = 2
MODE_NAMES  = ['CRUISE', 'FOLLOW', 'BRAKE']
MODE_COLORS = ['#00ff88', '#ffcc00', '#ff4444']


# ── Oval Track ──────────────────────────────────────────────────────
def generate_oval_track(straight_len=300, radius=80, ds=0.5):
    """Generate an oval track (x, y, yaw, curvature) with smooth transitions."""
    points = []
    # Bottom straight
    for s in np.arange(0, straight_len, ds):
        points.append((s, 0, 0, 0))
    # Right turn (semicircle)
    cx, cy = straight_len, radius
    for th in np.arange(-np.pi / 2, np.pi / 2, ds / radius):
        x = cx + radius * np.cos(th)
        y = cy + radius * np.sin(th)
        yaw = th + np.pi / 2
        points.append((x, y, yaw, 1.0 / radius))
    # Top straight (reverse direction)
    for s in np.arange(0, straight_len, ds):
        points.append((straight_len - s, 2 * radius, np.pi, 0))
    # Left turn (semicircle)
    cx2, cy2 = 0, radius
    for th in np.arange(np.pi / 2, 3 * np.pi / 2, ds / radius):
        x = cx2 + radius * np.cos(th)
        y = cy2 + radius * np.sin(th)
        yaw = th + np.pi / 2
        points.append((x, y, yaw, 1.0 / radius))
    return np.array(points)


def find_closest_index(track, x, y):
    dx = track[:, 0] - x
    dy = track[:, 1] - y
    return int(np.argmin(dx**2 + dy**2))


def compute_errors(track, idx, x, y, yaw):
    tx, ty, tyaw = track[idx, 0], track[idx, 1], track[idx, 2]
    dx = x - tx
    dy = y - ty
    ey = -dx * np.sin(tyaw) + dy * np.cos(tyaw)
    epsi = yaw - tyaw
    while epsi > np.pi:
        epsi -= 2 * np.pi
    while epsi < -np.pi:
        epsi += 2 * np.pi
    return ey, epsi


def track_distance_between(track, idx_from, idx_to):
    """Approximate arc-length distance along track from idx_from to idx_to (forward)."""
    n = len(track)
    total = 0.0
    i = idx_from
    while i != idx_to:
        j = (i + 1) % n
        dx = track[j, 0] - track[i, 0]
        dy = track[j, 1] - track[i, 1]
        total += np.sqrt(dx**2 + dy**2)
        i = j
        if total > 600.0:  # safety cap
            break
    return total


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# ── LQR Lateral Controller (from lqr_realtime.py) ───────────────────
def get_weights(speed_kmh):
    if speed_kmh > 108:
        return 0.8, 1.0, 1.3, 0.8, 7800.0
    elif speed_kmh > 100:
        return 1.0, 1.0, 1.5, 1.0, 7500.0
    elif speed_kmh > 85:
        return 1.0, 1.5, 2.5, 2.5, 7500.0
    elif speed_kmh > 78:
        return 4.2, 2.0, 3.0, 3.0, 7000.0
    elif speed_kmh > 55:
        return 2.0, 1.0, 6.0, 1.0, 6000.0
    elif speed_kmh > 30:
        return 4.0, 3.0, 7.0, 1.0, 4500.0
    elif speed_kmh > 20:
        return 3.0, 5.0, 7.0, 1.0, 2500.0
    else:
        return 5.0, 5.0, 10.0, 1.0, 2000.0


def build_lateral_matrices(vel):
    """Build discrete LQR matrices for lateral control (5-state with actuator lag)."""
    vel = max(vel, 1.5)

    Abaroc = np.zeros((4, 4))
    Abaroc[0, 1] = 1.0
    Abaroc[1, 1] = -SIGMA1 / (MASS * vel)
    Abaroc[1, 2] = SIGMA1 / MASS
    Abaroc[1, 3] = SIGMA2 / (MASS * vel)
    Abaroc[2, 3] = 1.0
    Abaroc[3, 1] = SIGMA2 / (IZ * vel)
    Abaroc[3, 2] = -SIGMA2 / IZ
    Abaroc[3, 3] = SIGMA3 / (IZ * vel)

    Bbaroc = np.zeros((4, 1))
    Bbaroc[1, 0] = 2 * CF / MASS
    Bbaroc[3, 0] = 2 * LF * CF / IZ

    Abarc = np.zeros((5, 5))
    Abarc[:4, :4] = Abaroc
    Abarc[:4, 4:5] = Bbaroc
    Abarc[4, 4] = -1.0 / LAG_TAU

    Bbarc = np.zeros((5, 1))
    Bbarc[4, 0] = 1.0 / LAG_TAU

    I5 = np.eye(5)
    Ad_inv = np.linalg.inv(I5 - DT * 0.5 * Abarc)
    Ad = Ad_inv @ (I5 + DT * 0.5 * Abarc)
    Bd = Ad_inv @ (DT * Bbarc)
    return Ad, Bd


def compute_lqr_lateral_gain(Ad, Bd, Q, R):
    try:
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
        return K
    except Exception:
        return np.zeros((1, 5))


# ── ACC LQR Longitudinal Controller (from acc_test.cpp) ─────────────
def build_acc_matrices():
    """
    ACC state: [distance_error, velocity_error, acceleration_cmd]
    Bilinear discretization of 3-state longitudinal model.
    """
    # Continuous-time system
    # d/dt [dist_err, vel_err, a] = Ac * [dist_err, vel_err, a] + Bc * u
    Ac = np.array([
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  0.0, -1.0 / ACC_LAG_TAU],
    ])
    Bc = np.array([[0.0], [0.0], [1.0 / ACC_LAG_TAU]])

    # Bilinear (Tustin) discretization
    I3 = np.eye(3)
    Ad_inv = np.linalg.inv(I3 - DT * 0.5 * Ac)
    Ad = Ad_inv @ (I3 + DT * 0.5 * Ac)
    Bd = Ad_inv @ (DT * Bc)
    return Ad, Bd


def compute_acc_gain():
    Ad, Bd = build_acc_matrices()
    Q = np.diag([ACC_Q_DIS, ACC_Q_VEL, 0.0])
    R = np.array([[ACC_R]])
    try:
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
        return K
    except Exception:
        return np.zeros((1, 3))


# ── Pure Pursuit Steering for Lead Vehicle ──────────────────────────
def pure_pursuit_steer(track, x, y, yaw, vx):
    """Simple pure pursuit: find lookahead point, compute required steer angle."""
    idx = find_closest_index(track, x, y)
    n = len(track)
    lookahead = max(LOOKAHEAD_DIST, vx * 1.0)

    # Walk forward on track until lookahead distance
    dist = 0.0
    target_idx = idx
    for _ in range(n):
        j = (target_idx + 1) % n
        seg = np.sqrt((track[j, 0] - track[target_idx, 0])**2 +
                      (track[j, 1] - track[target_idx, 1])**2)
        dist += seg
        target_idx = j
        if dist >= lookahead:
            break

    tx, ty = track[target_idx, 0], track[target_idx, 1]
    # Transform to vehicle frame
    dx = tx - x
    dy = ty - y
    local_x =  dx * np.cos(yaw) + dy * np.sin(yaw)
    local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)

    if abs(local_x) < 0.01:
        return 0.0

    # Pure pursuit curvature: 2*local_y / L^2
    L = np.sqrt(local_x**2 + local_y**2)
    curvature = 2.0 * local_y / (L**2)
    delta = np.arctan(WHEELBASE * curvature)
    return np.clip(delta, -0.5, 0.5)


# ── Vehicle State ────────────────────────────────────────────────────
class LeadVehicle:
    """Lead vehicle: pure pursuit steering, variable speed (curves + periodic)."""
    def __init__(self, track, start_idx, speed_kmh=60.0):
        self.track = track
        self.start_idx = start_idx
        self.reset(speed_kmh)

    def reset(self, speed_kmh=60.0):
        idx = self.start_idx
        self.x = self.track[idx, 0]
        self.y = self.track[idx, 1]
        self.yaw = self.track[idx, 2]
        self.vx = speed_kmh / 3.6
        self.delta_actual = 0.0
        self.base_speed_kmh = speed_kmh  # base speed setting
        self.target_speed_kmh = speed_kmh
        self.t = 0.0  # internal timer for speed variation

    def step(self):
        self.t += DT

        # Variable speed pattern:
        # 1) Slow down in curves (curvature-based)
        idx = find_closest_index(self.track, self.x, self.y)
        curv = abs(self.track[idx, 3])
        if curv > 0.005:
            curve_speed = self.base_speed_kmh * 0.65  # 65% speed in tight curves
        elif curv > 0.001:
            curve_speed = self.base_speed_kmh * 0.8   # 80% in mild curves
        else:
            curve_speed = self.base_speed_kmh          # full speed on straights

        # 2) Periodic speed variation (sinusoidal, ±12 km/h, period ~15s)
        periodic = 12.0 * np.sin(2 * np.pi * self.t / 15.0)

        # 3) Braking events (every ~25s, moderate slowdown for ~4s)
        brake_cycle = self.t % 25.0
        if 10.0 < brake_cycle < 14.0:
            brake_factor = 0.7  # 70% speed (gradual, not instant)
        else:
            brake_factor = 1.0

        self.target_speed_kmh = max(25.0, (curve_speed + periodic) * brake_factor)

        # Steering
        delta_cmd = pure_pursuit_steer(self.track, self.x, self.y, self.yaw, self.vx)
        self.delta_actual += DT / LAG_TAU * (delta_cmd - self.delta_actual)

        # Speed tracking (simple proportional)
        target_vx = self.target_speed_kmh / 3.6
        self.vx += np.clip(target_vx - self.vx, ACC_MAX_DECEL * DT, ACC_MAX_ACCEL * DT)
        self.vx = max(0.5, self.vx)

        # Bicycle kinematics
        beta = np.arctan(LR / WHEELBASE * np.tan(self.delta_actual))
        self.x += self.vx * np.cos(self.yaw + beta) * DT
        self.y += self.vx * np.sin(self.yaw + beta) * DT
        self.yaw += self.vx / LR * np.sin(beta) * DT
        while self.yaw > np.pi:
            self.yaw -= 2 * np.pi
        while self.yaw < -np.pi:
            self.yaw += 2 * np.pi


class EgoVehicle:
    """Ego vehicle: LQR lateral + ACC longitudinal."""
    def __init__(self, track):
        self.track = track
        self.acc_K = compute_acc_gain()
        self.reset()

    def reset(self, cruise_speed_kmh=80.0):
        self.x = self.track[0, 0]
        self.y = self.track[0, 1]
        self.yaw = self.track[0, 2]
        self.vx = 5.0  # start slow
        self.delta_actual = 0.0
        self.cruise_speed_kmh = cruise_speed_kmh

        # ACC state
        self.acc_mode = MODE_CRUISE
        self.acc_state = np.zeros(3)  # [dist_err, vel_err, a_cmd]
        self.accel_cmd = 0.0

        # Lateral controller state
        self.prev_ey = 0.0
        self.prev_epsi = 0.0

        # History
        self.hist_t = []
        self.hist_x = []
        self.hist_y = []
        self.hist_speed = []
        self.hist_distance = []
        self.hist_accel = []
        self.hist_mode = []

    def step(self, t, lead_x, lead_y, lead_vx):
        # ── ACC Longitudinal Control ──
        dist = euclidean_distance(self.x, self.y, lead_x, lead_y)
        desired_dist = self.vx * TIME_GAP + MIN_DIST
        dist_err = desired_dist - dist          # positive = too close
        vel_err = self.vx - lead_vx             # positive = ego faster

        if dist > ACC_RANGE:
            # CRUISE mode: simple speed tracking
            self.acc_mode = MODE_CRUISE
            speed_err = self.vx - self.cruise_speed_kmh / 3.6
            self.accel_cmd = np.clip(-2.0 * speed_err, ACC_MAX_DECEL, ACC_MAX_ACCEL)
        else:
            # === Emergency brake: override LQR if too close ===
            if dist < MIN_DIST * 1.5:
                # Critical: near collision, full brake immediately
                self.accel_cmd = ACC_MAX_DECEL
                self.acc_mode = MODE_BRAKE
                self.acc_state[2] = self.accel_cmd
            elif dist < desired_dist * 0.8:
                # Close: strong proportional brake (bypass LQR lag)
                closeness = 1.0 - dist / (desired_dist * 0.8)
                brake_force = ACC_MAX_DECEL * closeness
                brake_force += -4.0 * max(0, vel_err)  # velocity-proportional braking
                self.accel_cmd = np.clip(brake_force, ACC_MAX_DECEL, -0.5)
                self.acc_mode = MODE_BRAKE
                self.acc_state[2] = self.accel_cmd
            else:
                # Normal ACC: LQR control
                self.acc_state[0] = dist_err
                self.acc_state[1] = vel_err

                u_acc = float((-self.acc_K @ self.acc_state).item())
                u_acc = np.clip(u_acc, ACC_MAX_DECEL, ACC_MAX_ACCEL)

                # Update accel state via lag
                self.acc_state[2] += DT / ACC_LAG_TAU * (u_acc - self.acc_state[2])
                self.accel_cmd = self.acc_state[2]

                # Classify mode
                if dist_err > 3.0 or vel_err > 1.5:
                    self.acc_mode = MODE_BRAKE
                else:
                    self.acc_mode = MODE_FOLLOW

            # Override: if far from lead, blend toward cruise speed
            if dist_err < -15.0:
                speed_err = self.vx - self.cruise_speed_kmh / 3.6
                cruise_accel = np.clip(-2.0 * speed_err, ACC_MAX_DECEL, ACC_MAX_ACCEL)
                blend = np.clip((-dist_err - 15.0) / 30.0, 0.0, 1.0)
                self.accel_cmd = blend * cruise_accel + (1.0 - blend) * self.accel_cmd
                if blend > 0.7:
                    self.acc_mode = MODE_CRUISE

        # Integrate velocity
        self.vx += self.accel_cmd * DT

        # Hard safety: if closing in, never go faster than lead
        if dist < desired_dist and self.vx > lead_vx:
            # Blend toward lead speed proportional to how close we are
            ratio = max(0.0, dist / desired_dist)  # 0=touching, 1=at desired
            max_speed = lead_vx + (self.vx - lead_vx) * ratio
            self.vx = min(self.vx, max_speed)

        # Absolute safety: if within MIN_DIST, match or go slower than lead
        if dist < MIN_DIST:
            self.vx = min(self.vx, lead_vx * 0.9)

        self.vx = max(0.5, min(self.vx, 50.0))

        # ── LQR Lateral Control ──
        idx = find_closest_index(self.track, self.x, self.y)
        ey, epsi = compute_errors(self.track, idx, self.x, self.y, self.yaw)

        ey_dot = (ey - self.prev_ey) / DT
        epsi_dot = (epsi - self.prev_epsi) / DT

        speed_kmh = self.vx * 3.6
        q_ey, q_eydot, q_epsi, q_epsidot, r_w = get_weights(speed_kmh)
        Q = np.diag([q_ey, q_eydot, q_epsi, q_epsidot, 0.0])
        R = np.array([[r_w]])

        Ad, Bd = build_lateral_matrices(self.vx)
        K = compute_lqr_lateral_gain(Ad, Bd, Q, R)

        state = np.array([[ey], [ey_dot], [epsi], [epsi_dot], [self.delta_actual]])
        delta_cmd = float((-K @ state).item())

        # Steering rate limit
        rate_limit = 0.5
        diff = delta_cmd - self.delta_actual
        if abs(diff) / DT > rate_limit:
            delta_cmd = self.delta_actual + np.sign(diff) * rate_limit * DT
        delta_cmd = np.clip(delta_cmd, -0.5, 0.5)

        # Actuator lag
        self.delta_actual += DT / LAG_TAU * (delta_cmd - self.delta_actual)

        # ── Bicycle model update ──
        beta = np.arctan(LR / WHEELBASE * np.tan(self.delta_actual))
        self.x += self.vx * np.cos(self.yaw + beta) * DT
        self.y += self.vx * np.sin(self.yaw + beta) * DT
        self.yaw += self.vx / LR * np.sin(beta) * DT
        while self.yaw > np.pi:
            self.yaw -= 2 * np.pi
        while self.yaw < -np.pi:
            self.yaw += 2 * np.pi

        self.prev_ey = ey
        self.prev_epsi = epsi

        # ── Log ──
        self.hist_t.append(t)
        self.hist_x.append(self.x)
        self.hist_y.append(self.y)
        self.hist_speed.append(speed_kmh)
        self.hist_distance.append(dist)
        self.hist_accel.append(self.accel_cmd)
        self.hist_mode.append(self.acc_mode)

        max_hist = 1500
        if len(self.hist_t) > max_hist:
            for h in [self.hist_t, self.hist_x, self.hist_y, self.hist_speed,
                      self.hist_distance, self.hist_accel, self.hist_mode]:
                del h[0]


# ── Simulation Container ─────────────────────────────────────────────
class Simulation:
    def __init__(self):
        self.track = generate_oval_track()
        n = len(self.track)

        # Lead vehicle starts 40m ahead of ego (index offset)
        # Ego starts at index 0, lead at index ~80 (40m / 0.5m ds = 80 steps)
        lead_start_idx = 80
        self.lead = LeadVehicle(self.track, lead_start_idx, speed_kmh=60.0)
        self.ego = EgoVehicle(self.track)

        self.paused = False
        self.t = 0.0

        # Lead history for trail
        self.lead_hist_x = []
        self.lead_hist_y = []
        self.lead_hist_speed = []

    def reset(self):
        self.lead.reset(speed_kmh=self.lead.target_speed_kmh)
        self.ego.reset(cruise_speed_kmh=self.ego.cruise_speed_kmh)
        self.t = 0.0
        self.lead_hist_x.clear()
        self.lead_hist_y.clear()
        self.lead_hist_speed.clear()

    def step(self):
        if self.paused:
            return
        self.lead.step()
        self.ego.step(self.t, self.lead.x, self.lead.y, self.lead.vx)
        self.t += DT

        self.lead_hist_x.append(self.lead.x)
        self.lead_hist_y.append(self.lead.y)
        self.lead_hist_speed.append(self.lead.vx * 3.6)

        max_hist = 1500
        if len(self.lead_hist_x) > max_hist:
            del self.lead_hist_x[0]
            del self.lead_hist_y[0]
            del self.lead_hist_speed[0]


# ── Real-time Visualization ──────────────────────────────────────────
def main():
    sim = Simulation()

    # Dark theme colors (matching lqr_realtime.py)
    dark_bg    = '#1a1a2e'
    panel_bg   = '#16213e'
    text_color = '#eee'
    grid_color = '#2a2a4a'
    track_color = '#3a3a5a'
    ref_color   = '#4a90d9'
    ego_color   = '#e94560'    # red
    lead_color  = '#ff8c00'    # orange

    fig = plt.figure(figsize=(16, 9), facecolor=dark_bg)
    fig.canvas.manager.set_window_title('Two-Vehicle ACC Demo - USAFE Racing')

    # Layout: track (left, 2 cols) + 4 plots (right, 2 cols)
    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.35,
                          left=0.05, right=0.98, top=0.91, bottom=0.06)
    ax_track = fig.add_subplot(gs[:, :2])
    ax_dist  = fig.add_subplot(gs[0, 2:])
    ax_speed = fig.add_subplot(gs[1, 2:])
    ax_accel = fig.add_subplot(gs[2, 2:])
    ax_mode  = fig.add_subplot(gs[3, 2:])

    for ax in [ax_track, ax_dist, ax_speed, ax_accel, ax_mode]:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=text_color, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

    fig.suptitle('Two-Vehicle ACC Demo — USAFE Racing  |  LQR Lateral + ACC Longitudinal',
                 fontsize=12, color=text_color, fontweight='bold', y=0.97)

    track = sim.track

    # ── Track View ──
    ax_track.plot(track[:, 0], track[:, 1], '-',
                  color=track_color, lw=18, alpha=0.5, solid_capstyle='round')
    ax_track.plot(track[:, 0], track[:, 1], '--',
                  color=ref_color, lw=1.0, alpha=0.6)
    ax_track.set_aspect('equal')
    ax_track.set_title('Track View', color=text_color, fontsize=10, pad=5)
    ax_track.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)

    # Lead vehicle: trail + marker
    lead_trail, = ax_track.plot([], [], '-', color=lead_color, lw=2.0, alpha=0.7)
    lead_dot, = ax_track.plot([], [], 's', color=lead_color, markersize=9,
                               zorder=6, label='Lead')

    # Ego vehicle: trail + marker
    ego_trail, = ax_track.plot([], [], '-', color=ego_color, lw=2.0, alpha=0.8)
    ego_dot, = ax_track.plot([], [], 'o', color='#ff6b6b', markersize=8,
                              zorder=7, label='Ego')

    ax_track.legend(loc='lower right', fontsize=8, facecolor=panel_bg,
                    edgecolor=grid_color, labelcolor=text_color)

    # Info text overlays on track view
    info_text = ax_track.text(
        0.02, 0.97, '', transform=ax_track.transAxes,
        color='#00ff88', fontsize=9, fontweight='bold',
        va='top', fontfamily='monospace')
    mode_track_text = ax_track.text(
        0.02, 0.78, '', transform=ax_track.transAxes,
        color='#ffcc00', fontsize=10, fontweight='bold',
        va='top', fontfamily='monospace')
    time_text = ax_track.text(
        0.98, 0.97, '', transform=ax_track.transAxes,
        color=text_color, fontsize=9, fontweight='bold',
        va='top', ha='right', fontfamily='monospace')
    status_text = ax_track.text(
        0.5, 0.02, '', transform=ax_track.transAxes,
        color='#ff6b6b', fontsize=9, fontweight='bold',
        va='bottom', ha='center', fontfamily='monospace')

    # ── Distance plot ──
    ax_dist.set_title('Distance to Lead Vehicle (m)', color=text_color, fontsize=9, pad=3)
    ax_dist.axhline(MIN_DIST, color='#ff4444', lw=0.8, alpha=0.6, linestyle='--',
                    label=f'Min dist ({MIN_DIST}m)')
    ax_dist.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)
    ax_dist.legend(loc='upper right', fontsize=7, facecolor=panel_bg,
                   edgecolor=grid_color, labelcolor=text_color)
    dist_line, = ax_dist.plot([], [], '-', color='#00bbff', lw=1.5)
    desired_dist_line, = ax_dist.plot([], [], '--', color='#ffcc00', lw=1.0,
                                       alpha=0.7, label='Desired dist')

    # ── Speed comparison plot ──
    ax_speed.set_title('Speed Comparison (km/h)', color=text_color, fontsize=9, pad=3)
    ax_speed.set_ylim(0, 120)
    ax_speed.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)
    ego_speed_line, = ax_speed.plot([], [], '-', color=ego_color, lw=1.5, label='Ego')
    lead_speed_line, = ax_speed.plot([], [], '-', color=lead_color, lw=1.5, label='Lead')
    cruise_line, = ax_speed.plot([], [], '--', color='#888', lw=0.8, alpha=0.6, label='Ego cruise')
    ax_speed.legend(loc='upper right', fontsize=7, facecolor=panel_bg,
                    edgecolor=grid_color, labelcolor=text_color)

    # ── Accel / control effort plot ──
    ax_accel.set_title('ACC Control Effort (m/s²)', color=text_color, fontsize=9, pad=3)
    ax_accel.axhline(0, color=ref_color, lw=0.8, alpha=0.5)
    ax_accel.set_ylim(-6, 4)
    ax_accel.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)
    accel_line, = ax_accel.plot([], [], '-', color='#ff8844', lw=1.5)

    # ── ACC mode indicator ──
    ax_mode.set_title('ACC Mode', color=text_color, fontsize=9, pad=3)
    ax_mode.set_ylim(-0.5, 2.5)
    ax_mode.set_yticks([0, 1, 2])
    ax_mode.set_yticklabels(['CRUISE', 'FOLLOW', 'BRAKE'],
                             color=text_color, fontsize=8)
    ax_mode.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)
    ax_mode.set_xlabel('Time (s)', color=text_color, fontsize=8)
    mode_line, = ax_mode.plot([], [], 's', color='#00ff88', markersize=2)

    # Help text
    help_str = ('[Space] Pause  [Up/Dn] Ego cruise speed ±10 km/h  '
                '[L/K] Lead speed ±5 km/h  [R] Reset  [Q] Quit')
    fig.text(0.5, 0.005, help_str, ha='center', color='#666',
             fontsize=7, fontfamily='monospace')

    # ── Keyboard handler ──
    def on_key(event):
        if event.key == ' ':
            sim.paused = not sim.paused
        elif event.key == 'up':
            sim.ego.cruise_speed_kmh = min(130, sim.ego.cruise_speed_kmh + 10)
        elif event.key == 'down':
            sim.ego.cruise_speed_kmh = max(20, sim.ego.cruise_speed_kmh - 10)
        elif event.key == 'l':
            sim.lead.target_speed_kmh = min(110, sim.lead.target_speed_kmh + 5)
        elif event.key == 'k':
            sim.lead.target_speed_kmh = max(20, sim.lead.target_speed_kmh - 5)
        elif event.key == 'r':
            sim.reset()
        elif event.key in ('q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plot_window = 30.0

    def animate(_frame):
        steps_per_frame = max(1, int(1.0 / (30 * DT)))
        for _ in range(steps_per_frame):
            sim.step()

        if len(sim.ego.hist_t) < 2:
            return []

        # ── Track view update ──
        trail_n = 250
        lead_trail.set_data(sim.lead_hist_x[-trail_n:], sim.lead_hist_y[-trail_n:])
        lead_dot.set_data([sim.lead.x], [sim.lead.y])
        ego_trail.set_data(sim.ego.hist_x[-trail_n:], sim.ego.hist_y[-trail_n:])
        ego_dot.set_data([sim.ego.x], [sim.ego.y])

        # Auto-zoom following ego
        margin = 70
        cx, cy = sim.ego.x, sim.ego.y
        ax_track.set_xlim(cx - margin * 1.8, cx + margin * 1.8)
        ax_track.set_ylim(cy - margin, cy + margin)

        # Info text
        dist = sim.ego.hist_distance[-1] if sim.ego.hist_distance else 0.0
        mode = sim.ego.acc_mode
        mode_name = MODE_NAMES[mode]
        mode_color = MODE_COLORS[mode]

        info_text.set_text(
            f"Ego:  {sim.ego.hist_speed[-1]:.0f} km/h  (cruise {sim.ego.cruise_speed_kmh:.0f})\n"
            f"Lead: {sim.lead.vx * 3.6:.0f} km/h  (set {sim.lead.target_speed_kmh:.0f})\n"
            f"Dist: {dist:.1f} m"
        )
        mode_track_text.set_text(f"ACC: {mode_name}")
        mode_track_text.set_color(mode_color)
        time_text.set_text(f"t = {sim.t:.1f}s")

        if sim.paused:
            status_text.set_text('PAUSED')
        else:
            status_text.set_text('')

        # ── Rolling time window ──
        t_arr = sim.ego.hist_t
        t_min = max(0.0, t_arr[-1] - plot_window)
        t_max = t_arr[-1] + 0.5

        # Distance plot
        ax_dist.set_xlim(t_min, t_max)
        dist_arr = sim.ego.hist_distance
        dist_line.set_data(t_arr, dist_arr)

        # Desired distance line (ego_speed * time_gap + min_dist)
        desired = [v / 3.6 * TIME_GAP + MIN_DIST for v in sim.ego.hist_speed]
        desired_dist_line.set_data(t_arr, desired)
        desired_dist_line.set_label('Desired dist')

        window_dist = [d for t, d in zip(t_arr, dist_arr) if t >= t_min]
        window_des  = [d for t, d in zip(t_arr, desired)  if t >= t_min]
        if window_dist:
            all_d = window_dist + window_des
            ax_dist.set_ylim(max(0, min(all_d) - 10), max(all_d) + 20)

        # Speed plot
        ax_speed.set_xlim(t_min, t_max)
        ego_speed_line.set_data(t_arr, sim.ego.hist_speed)
        if sim.lead_hist_speed:
            lead_t = sim.ego.hist_t  # same length window
            n_lead = min(len(sim.lead_hist_speed), len(t_arr))
            lead_speed_line.set_data(t_arr[-n_lead:], sim.lead_hist_speed[-n_lead:])
        # Cruise speed reference line
        cruise_val = sim.ego.cruise_speed_kmh
        cruise_line.set_data([t_min, t_max], [cruise_val, cruise_val])
        window_spd = [s for t, s in zip(t_arr, sim.ego.hist_speed) if t >= t_min]
        if window_spd:
            lo = max(0, min(window_spd) - 10)
            hi = max(window_spd) + 10
            ax_speed.set_ylim(lo, hi)

        # Accel plot
        ax_accel.set_xlim(t_min, t_max)
        accel_line.set_data(t_arr, sim.ego.hist_accel)
        window_acc = [a for t, a in zip(t_arr, sim.ego.hist_accel) if t >= t_min]
        if window_acc:
            lo = min(window_acc) - 0.5
            hi = max(window_acc) + 0.5
            ax_accel.set_ylim(min(lo, -1.0), max(hi, 1.0))

        # Mode plot (colored scatter)
        ax_mode.set_xlim(t_min, t_max)
        modes = sim.ego.hist_mode
        colors_scatter = [MODE_COLORS[m] for m in modes]
        mode_line.set_data(t_arr, modes)
        # Color the points by mode
        for coll in list(ax_mode.collections):
            coll.remove()
        t_window = [(t, m) for t, m in zip(t_arr, modes) if t >= t_min]
        if t_window:
            tw, mw = zip(*t_window)
            col = [MODE_COLORS[m] for m in mw]
            ax_mode.scatter(tw, mw, c=col, s=4, zorder=5)

        return [lead_trail, lead_dot, ego_trail, ego_dot,
                dist_line, desired_dist_line,
                ego_speed_line, lead_speed_line, cruise_line,
                accel_line, mode_line]

    anim = animation.FuncAnimation(
        fig, animate, interval=33, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    print("Two-Vehicle ACC Demo - USAFE Racing")
    print("  Lead vehicle (orange): pure pursuit steering, constant speed")
    print("  Ego vehicle  (red):    LQR lateral + ACC longitudinal")
    print()
    print("ACC Parameters:")
    print(f"  Time gap: {TIME_GAP}s  |  Min distance: {MIN_DIST}m  |  Range: {ACC_RANGE}m")
    print(f"  Q_dis={ACC_Q_DIS}  Q_vel={ACC_Q_VEL}  R={ACC_R}  lag_tau={ACC_LAG_TAU}")
    print()
    print("Controls:")
    print("  [Space]  Pause / Resume")
    print("  [Up/Dn]  Ego cruise speed ±10 km/h")
    print("  [L/K]    Lead vehicle speed ±5 km/h")
    print("  [R]      Reset simulation")
    print("  [Q/Esc]  Quit")
    print()
    print("Starting simulation...")
    main()
