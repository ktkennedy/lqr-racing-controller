"""
LQR Preview Controller - Real-time Interactive Simulation
USAFE Racing Team

Run: python3 lqr_realtime.py
Requirements: numpy, matplotlib, scipy

Controls:
  Space  - Pause / Resume
  Up/Dn  - Increase / Decrease target speed
  R      - Reset simulation
  Q/Esc  - Quit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import solve_discrete_are

# ── Vehicle Parameters (from preview_ctrl.cpp) ──────────────────────
WHEELBASE = 2.72
LF = 1.58
LR = 1.59
MASS = 1700.0
DT = 0.04          # 20 Hz
LAG_TAU = 0.14

IZ = LF * LR * MASS
CF = MASS * (LF / (LF + LR)) * 0.5 * 9.81 * 0.165 * 180 / np.pi
CR = MASS * (LR / (LF + LR)) * 0.5 * 9.81 * 0.165 * 180 / np.pi

SIGMA1 = 2.0 * (CF + CR)
SIGMA2 = -2.0 * (LF * CF - LR * CR)
SIGMA3 = -2.0 * (LF**2 * CF + LR**2 * CR)


# ── Speed-scheduled Q/R weights ─────────────────────────────────────
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


# ── Bicycle model + LQR ─────────────────────────────────────────────
def build_state_matrices(vel):
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


def compute_lqr_gain(Ad, Bd, Q, R):
    try:
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
        return K
    except Exception:
        return np.zeros((1, 5))


# ── Oval track ───────────────────────────────────────────────────────
def generate_oval_track(straight_len=300, radius=80, ds=0.5):
    points = []
    for s in np.arange(0, straight_len, ds):
        points.append((s, 0, 0, 0))
    cx, cy = straight_len, radius
    for th in np.arange(-np.pi / 2, np.pi / 2, ds / radius):
        x = cx + radius * np.cos(th)
        y = cy + radius * np.sin(th)
        yaw = th + np.pi / 2
        points.append((x, y, yaw, 1.0 / radius))
    for s in np.arange(0, straight_len, ds):
        points.append((straight_len - s, 2 * radius, np.pi, 0))
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
    return np.argmin(dx**2 + dy**2)


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


# ── Simulation State ─────────────────────────────────────────────────
class SimState:
    def __init__(self, track):
        self.track = track
        self.reset()

    def reset(self):
        self.x = self.track[0, 0]
        self.y = self.track[0, 1] + 1.5   # 1.5m lateral offset
        self.yaw = self.track[0, 2]
        self.vx = 20.0  # m/s
        self.delta_actual = 0.0
        self.t = 0.0
        self.step = 0
        self.prev_ey = 0.0
        self.prev_epsi = 0.0
        self.paused = False
        self.speed_offset = 0  # km/h user offset

        # History buffers (rolling window)
        self.max_hist = 1500  # 60s at 20Hz
        self.hist_t = []
        self.hist_x = []
        self.hist_y = []
        self.hist_ey = []
        self.hist_epsi = []
        self.hist_speed = []
        self.hist_delta = []
        self.hist_Q_ey = []
        self.hist_R_w = []

    def step_sim(self):
        if self.paused:
            return

        idx = find_closest_index(self.track, self.x, self.y)
        curv = self.track[idx, 3]

        # Speed profile
        base_target = 60 if curv > 0.001 else 110
        target_speed_kmh = max(20, base_target + self.speed_offset)
        target_speed = target_speed_kmh / 3.6
        if self.vx < target_speed:
            self.vx = min(self.vx + 2.0 * DT, target_speed)
        else:
            self.vx = max(self.vx - 3.0 * DT, target_speed)

        speed_kmh = self.vx * 3.6
        ey, epsi = compute_errors(self.track, idx, self.x, self.y, self.yaw)

        if self.step > 0:
            ey_dot = (ey - self.prev_ey) / DT
            epsi_dot = (epsi - self.prev_epsi) / DT
        else:
            ey_dot = 0.0
            epsi_dot = 0.0

        q_ey, q_eydot, q_epsi, q_epsidot, r_w = get_weights(speed_kmh)
        Q = np.diag([q_ey, q_eydot, q_epsi, q_epsidot, 0.0])
        R = np.array([[r_w]])

        Ad, Bd = build_state_matrices(self.vx)
        K = compute_lqr_gain(Ad, Bd, Q, R)

        state = np.array([[ey], [ey_dot], [epsi], [epsi_dot], [self.delta_actual]])
        delta_cmd = float((-K @ state).item())

        # Steering rate limit
        rate_limit = 0.5
        delta_rate = np.clip((delta_cmd - self.delta_actual) / DT, -rate_limit, rate_limit)
        delta_cmd = self.delta_actual + delta_rate * DT
        delta_cmd = np.clip(delta_cmd, -0.5, 0.5)

        # Actuator lag
        self.delta_actual += DT / LAG_TAU * (delta_cmd - self.delta_actual)

        # Bicycle model update
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
        self.t += DT
        self.step += 1

        # Append history
        self.hist_t.append(self.t)
        self.hist_x.append(self.x)
        self.hist_y.append(self.y)
        self.hist_ey.append(ey)
        self.hist_epsi.append(np.degrees(epsi))
        self.hist_speed.append(speed_kmh)
        self.hist_delta.append(np.degrees(self.delta_actual))
        self.hist_Q_ey.append(q_ey)
        self.hist_R_w.append(r_w)

        # Trim history
        if len(self.hist_t) > self.max_hist:
            for h in [self.hist_t, self.hist_x, self.hist_y, self.hist_ey,
                       self.hist_epsi, self.hist_speed, self.hist_delta,
                       self.hist_Q_ey, self.hist_R_w]:
                del h[0]


# ── Real-time Visualization ──────────────────────────────────────────
def main():
    track = generate_oval_track()
    sim = SimState(track)

    # Dark theme
    dark_bg = '#1a1a2e'
    panel_bg = '#16213e'
    accent = '#e94560'
    text_color = '#eee'
    grid_color = '#2a2a4a'
    track_color = '#3a3a5a'
    ref_color = '#4a90d9'

    fig = plt.figure(figsize=(16, 9), facecolor=dark_bg)
    fig.canvas.manager.set_window_title('LQR Controller - USAFE Racing (Interactive)')

    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3,
                          left=0.05, right=0.98, top=0.91, bottom=0.06)
    ax_track = fig.add_subplot(gs[:, :2])
    ax_ey    = fig.add_subplot(gs[0, 2:])
    ax_epsi  = fig.add_subplot(gs[1, 2:])
    ax_speed = fig.add_subplot(gs[2, 2:])

    for ax in [ax_track, ax_ey, ax_epsi, ax_speed]:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=text_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

    fig.suptitle('LQR Preview Controller — USAFE Racing (Real-time)',
                 fontsize=14, color=text_color, fontweight='bold', y=0.97)

    # Track background
    ax_track.plot(track[:, 0], track[:, 1], '-', color=track_color, lw=18, alpha=0.5, solid_capstyle='round')
    ax_track.plot(track[:, 0], track[:, 1], '--', color=ref_color, lw=1.2, alpha=0.7)
    ax_track.set_aspect('equal')
    ax_track.set_title('Track View', color=text_color, fontsize=11, pad=6)
    ax_track.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)

    trail_line, = ax_track.plot([], [], '-', color=accent, lw=2, alpha=0.8)
    car_dot, = ax_track.plot([], [], 'o', color='#ff6b6b', markersize=8, zorder=5)
    speed_text = ax_track.text(0.02, 0.96, '', transform=ax_track.transAxes,
                               color='#00ff88', fontsize=11, fontweight='bold',
                               va='top', fontfamily='monospace')
    weight_text = ax_track.text(0.02, 0.88, '', transform=ax_track.transAxes,
                                color='#ffcc00', fontsize=9, va='top', fontfamily='monospace')
    time_text = ax_track.text(0.98, 0.96, '', transform=ax_track.transAxes,
                              color=text_color, fontsize=10, fontweight='bold',
                              va='top', ha='right', fontfamily='monospace')
    status_text = ax_track.text(0.5, 0.02, '', transform=ax_track.transAxes,
                                color='#ff6b6b', fontsize=10, fontweight='bold',
                                va='bottom', ha='center', fontfamily='monospace')

    # Right-side plots
    ax_ey.set_title('Lateral Error (m)', color=text_color, fontsize=10, pad=4)
    ax_ey.set_ylim(-3, 3)
    ax_ey.axhline(0, color=ref_color, lw=0.8, alpha=0.5)
    ax_ey.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)
    ey_line, = ax_ey.plot([], [], '-', color='#00ff88', lw=1.5)

    ax_epsi.set_title('Heading Error (deg)', color=text_color, fontsize=10, pad=4)
    ax_epsi.set_ylim(-10, 10)
    ax_epsi.axhline(0, color=ref_color, lw=0.8, alpha=0.5)
    ax_epsi.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)
    epsi_line, = ax_epsi.plot([], [], '-', color='#ff6b6b', lw=1.5)

    ax_speed.set_title('Speed (km/h)', color=text_color, fontsize=10, pad=4)
    ax_speed.set_ylim(0, 140)
    ax_speed.grid(True, color=grid_color, alpha=0.3, linewidth=0.5)
    ax_speed.set_xlabel('Time (s)', color=text_color, fontsize=9)
    speed_line, = ax_speed.plot([], [], '-', color='#ffcc00', lw=1.5)

    # Help text
    help_str = '[Space] Pause  [Up/Dn] Speed  [R] Reset  [Q] Quit'
    fig.text(0.5, 0.01, help_str, ha='center', color='#666', fontsize=8, fontfamily='monospace')

    # Keyboard handler
    def on_key(event):
        if event.key == ' ':
            sim.paused = not sim.paused
        elif event.key == 'up':
            sim.speed_offset += 10
        elif event.key == 'down':
            sim.speed_offset = max(-40, sim.speed_offset - 10)
        elif event.key == 'r':
            sim.reset()
        elif event.key in ('q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Time window for rolling plots (seconds)
    plot_window = 30.0

    def animate(_frame):
        # Run multiple sim steps per frame to match real-time at ~30fps display
        steps_per_frame = max(1, int(1.0 / (30 * DT)))
        for _ in range(steps_per_frame):
            sim.step_sim()

        if len(sim.hist_t) < 2:
            return []

        # Trail on track (last 300 points)
        trail_n = 300
        trail_x = sim.hist_x[-trail_n:]
        trail_y = sim.hist_y[-trail_n:]
        trail_line.set_data(trail_x, trail_y)
        car_dot.set_data([sim.hist_x[-1]], [sim.hist_y[-1]])

        # Auto-zoom
        margin = 60
        cx, cy = sim.hist_x[-1], sim.hist_y[-1]
        ax_track.set_xlim(cx - margin * 1.8, cx + margin * 1.8)
        ax_track.set_ylim(cy - margin, cy + margin)

        # Info
        speed_text.set_text(f"Speed: {sim.hist_speed[-1]:.0f} km/h")
        weight_text.set_text(f"Q_ey={sim.hist_Q_ey[-1]:.1f}  R={sim.hist_R_w[-1]:.0f}")
        time_text.set_text(f"t = {sim.t:.1f}s")
        if sim.paused:
            status_text.set_text('PAUSED')
        elif sim.speed_offset != 0:
            status_text.set_text(f'Speed offset: {sim.speed_offset:+d} km/h')
        else:
            status_text.set_text('')

        # Rolling time window for right-side plots
        t_arr = sim.hist_t
        t_min = max(0, t_arr[-1] - plot_window)
        t_max = t_arr[-1] + 1.0

        ax_ey.set_xlim(t_min, t_max)
        ey_line.set_data(t_arr, sim.hist_ey)
        # Auto-scale Y
        window_ey = [e for t, e in zip(t_arr, sim.hist_ey) if t >= t_min]
        if window_ey:
            max_ey = max(abs(min(window_ey)), abs(max(window_ey)), 0.3) * 1.3
            ax_ey.set_ylim(-max_ey, max_ey)

        ax_epsi.set_xlim(t_min, t_max)
        epsi_line.set_data(t_arr, sim.hist_epsi)
        window_epsi = [e for t, e in zip(t_arr, sim.hist_epsi) if t >= t_min]
        if window_epsi:
            max_epsi = max(abs(min(window_epsi)), abs(max(window_epsi)), 1.0) * 1.3
            ax_epsi.set_ylim(-max_epsi, max_epsi)

        ax_speed.set_xlim(t_min, t_max)
        speed_line.set_data(t_arr, sim.hist_speed)
        window_spd = [s for t, s in zip(t_arr, sim.hist_speed) if t >= t_min]
        if window_spd:
            ax_speed.set_ylim(max(0, min(window_spd) - 10), max(window_spd) + 10)

        return [trail_line, car_dot, ey_line, epsi_line, speed_line]

    anim = animation.FuncAnimation(fig, animate, interval=33, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    print("LQR Preview Controller - Real-time Simulation")
    print("Controls: [Space] Pause  [Up/Down] Speed  [R] Reset  [Q] Quit")
    print("Starting...")
    main()
