"""
VISTA — Animation renderer: passive vs Digital PID, same speed bump.
Renders an MP4 (and GIF) directly from the sampled-data simulation,
side by side, with exaggerated displacements and a live accel trace.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.patches import Rectangle, Circle

from vista import (QuarterCar, SemiActiveDamper, DamperParams,
                   PassiveController, DigitalPID, PIDGains, SpeedBump, simulate)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ---- simulate both vehicles on the identical road (two bumps) --------------
car = QuarterCar()
bump1 = SpeedBump(height=0.05, length=0.5, speed_kmh=20.0, t_start=1.0)
bump2 = SpeedBump(height=0.05, length=0.5, speed_kmh=20.0, t_start=3.0)
road = lambda t: bump1(t) + bump2(t)
T_END, FPS, SPEED = 5.5, 30, 0.5          # render at 0.5x slow motion
rp = simulate(car, SemiActiveDamper(DamperParams(2000, 2000, 1e-4)),
              PassiveController(2000), road, T_END)
rd = simulate(car, SemiActiveDamper(DamperParams(500, 3000, 0.015)),
              DigitalPID(PIDGains(kp=6000, ki=0, kd=20, tf=0.010, f_max=4000)),
              road, T_END, label="Digital PID")

frames = int(T_END / SPEED * FPS)
t_frames = np.linspace(0.0, T_END, frames)

def at(r, t):
    i = min(int(t / 1e-3), len(r.t) - 1)
    return (r.z_s[i], r.z_u[i], r.z_r[i], r.a_s[i], r.c_damper[i])

# ---- figure ----------------------------------------------------------------
BG, PANEL, INK, DIM = "#0d1622", "#13202f", "#d7e3ee", "#7c93a8"
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor(BG)
axL = fig.add_axes([0.04, 0.30, 0.44, 0.62]); axR = fig.add_axes([0.52, 0.30, 0.44, 0.62])
axT = fig.add_axes([0.06, 0.06, 0.88, 0.17])
for ax in (axL, axR):
    ax.set_facecolor(BG); ax.set_xlim(0, 10); ax.set_ylim(-1.2, 6.2)
    ax.axis("off")
axT.set_facecolor(PANEL)
axT.set_xlim(0, T_END); axT.set_ylim(-13, 13)
axT.tick_params(colors=DIM, labelsize=8)
for s in axT.spines.values(): s.set_color("#22374d")
axT.set_ylabel("body accel [m/s²]", color=DIM, fontsize=9)
axL.set_title("PASSIVE  (c = 2000 Ns/m)", color="#9fb4c7", fontsize=13, family="monospace")
axR.set_title("DIGITAL PID  (semi-active, 1 kHz ECU)", color="#ffd166", fontsize=13, family="monospace")
fig.text(0.5, 0.965, "VISTA — two speed bumps (t=1 s, t=3 s), 0.5× slow motion, displacements ×12",
         color=DIM, fontsize=10, ha="center", family="monospace")

EX = 12 * 10  # m -> axis units, exaggerated

def build_rig(ax, accent):
    road_line, = ax.plot([], [], color="#39536f", lw=2)
    tire = Circle((5, 0), 0.55, fill=False, ec="#5aa9e6", lw=4); ax.add_patch(tire)
    mu = Rectangle((4.0, 0), 2.0, 0.5, fc="#22374d", ec="#5aa9e6"); ax.add_patch(mu)
    spring, = ax.plot([], [], color="#69d58c", lw=2.2)
    damp_rod, = ax.plot([], [], color="#c8b08a", lw=3)
    damp_cyl = Rectangle((5.5, 0), 0.5, 1.0, fill=False, ec="#c8b08a", lw=2.5)
    ax.add_patch(damp_cyl)
    ms = Rectangle((3.2, 0), 3.6, 1.6, fc="#1c3350", ec=accent, lw=1.5); ax.add_patch(ms)
    label = ax.text(5, 0, "m\u209b", color=accent, ha="center", fontsize=12, family="monospace")
    return dict(road=road_line, tire=tire, mu=mu, spring=spring,
                rod=damp_rod, cyl=damp_cyl, ms=ms, label=label)

rigL = build_rig(axL, "#9fb4c7")
rigR = build_rig(axR, "#ffd166")
trP, = axT.plot([], [], color="#54687c", lw=1.4, label="passive")
trD, = axT.plot([], [], color="#ff5c5c", lw=1.6, label="Digital PID")
axT.legend(loc="upper right", fontsize=8, facecolor=PANEL, edgecolor="#22374d",
           labelcolor=INK)
cursor = axT.axvline(0, color=DIM, lw=0.8, ls="--")

xs_road = np.linspace(0, 10, 160)
V = 20 / 3.6

def spring_pts(x0, y_bot, y_top, coils=6, w=0.35):
    ys = np.linspace(y_bot, y_top, coils * 2 + 2)
    xs = np.full_like(ys, x0)
    xs[1:-1] += np.where(np.arange(len(ys) - 2) % 2 == 0, w, -w)
    return xs, ys

def update_rig(rig, r, t, ax):
    zs, zu, zr, a, c = at(r, t)
    # road ribbon: future to the right of the wheel (x=5)
    zr_line = [road((t + (xx - 5) * 0.09 / V)) for xx in xs_road]
    rig["road"].set_data(xs_road, np.array(zr_line) * EX * 0.1)
    y_r, y_u, y_s = zr * EX * 0.1, 1.0 + zu * EX * 0.1, 4.2 + zs * EX * 0.1
    rig["tire"].center = (5, y_r + 0.55)
    rig["mu"].set_y(y_u); rig["mu"].set_x(4.0)
    sx, sy = spring_pts(4.6, y_u + 0.5, y_s)
    rig["spring"].set_data(sx, sy)
    mid = (y_u + 0.5 + y_s) / 2
    rig["cyl"].set_y(mid - 0.5)
    rig["rod"].set_data([5.75, 5.75, np.nan, 5.75, 5.75],
                        [y_u + 0.5, mid - 0.2, np.nan, mid + 0.2, y_s])
    k = (c - 500) / 2500
    rig["cyl"].set_edgecolor((0.55 + 0.45 * k, 0.55 - 0.25 * k, 0.45 - 0.25 * k))
    rig["ms"].set_y(y_s)
    rig["label"].set_position((5, y_s + 0.7))

def init():
    return []

def animate(f):
    t = t_frames[f]
    update_rig(rigL, rp, t, axL)
    update_rig(rigR, rd, t, axR)
    sel = rp.t <= t
    trP.set_data(rp.t[sel], rp.a_s[sel])
    trD.set_data(rd.t[sel], rd.a_s[sel])
    cursor.set_xdata([t, t])
    return []

ani = manim.FuncAnimation(fig, animate, frames=frames, init_func=init, blit=False)
mp4 = os.path.join(OUT, "vista_passive_vs_pid.mp4")
ani.save(mp4, writer=manim.FFMpegWriter(fps=FPS, bitrate=2400))
print("MP4 saved")
gif = os.path.join(OUT, "vista_passive_vs_pid.gif")
ani.save(gif, writer=manim.PillowWriter(fps=15))
print("GIF saved")
