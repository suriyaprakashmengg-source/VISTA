"""
VISTA v1.30 — Stage 13: Half-Car Model
======================================
Per-corner deployment of the quarter-car controller library on the
4-DOF pitch-plane model. The wheelbase delay makes the speed bump a
two-impact event (front axle, then rear 468 ms later at 20 km/h),
exciting the pitch mode — invisible to any quarter-car model.

Comparison: Passive / Continuous Skyhook (c_sky=8000) / Hybrid (α=0.70)
on speed bump and medium random road.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vista import (HalfCar, SemiActiveDamper, DamperParams,
                   PassiveController, ContinuousSkyhook, HybridSkyGround,
                   SpeedBump, RandomRoad, simulate_half, compute_half_metrics,
                   improvement)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
REPORT = []
def log(s=""):
    print(s); REPORT.append(s)

hc = HalfCar()
T_END = 8.0
bump = SpeedBump(height=0.05, length=0.5, speed_kmh=20.0, t_start=1.0)
medium = RandomRoad(T_END, roughness=2e-3, seed=42)

sad = lambda: SemiActiveDamper(DamperParams(c_min=500, c_max=3000, tau_valve=0.015))
pdm = lambda: SemiActiveDamper(DamperParams(c_min=2000, c_max=2000, tau_valve=1e-4))

C_SKY, C_GND, ALPHA = 8000, 4000, 0.70

def controllers(kind):
    if kind == "passive":
        return pdm(), pdm(), PassiveController(2000), PassiveController(2000), "Passive"
    if kind == "sky":
        return sad(), sad(), ContinuousSkyhook(C_SKY), ContinuousSkyhook(C_SKY), \
               "Continuous Skyhook (per corner)"
    if kind == "hybrid":
        return (sad(), sad(), HybridSkyGround(ALPHA, C_SKY, C_GND),
                HybridSkyGround(ALPHA, C_SKY, C_GND), "Hybrid α=0.70 (per corner)")

log("=" * 78)
log(f"STAGE 13 — HALF-CAR MODEL (heave {hc.p.heave_hz:.2f} Hz, "
    f"pitch {hc.p.pitch_hz:.2f} Hz, axle delay {hc.p.axle_delay*1e3:.0f} ms)")
log("=" * 78)

all_runs = {}
for road, name, tev in [(bump, "SPEED BUMP", 1.0), (medium, "MEDIUM ROAD", 0.5)]:
    log(f"\n--- {name} ---")
    log(f"{'Controller':<34} {'heaveRMS':>9} {'dH%':>6} {'pitchRMS':>9} {'dP%':>6} "
        f"{'pkPitch°':>9} {'travel mm':>10} {'tireRMS mm':>11}")
    runs = []
    base = None
    for kind in ("passive", "sky", "hybrid"):
        df, dr, cf, cr, lbl = controllers(kind)
        r = simulate_half(hc, df, dr, cf, cr, road, T_END, label=lbl)
        m = compute_half_metrics(r, tev)
        if base is None:
            base = m
        runs.append((r, m))
        log(f"{lbl:<34} {m.rms_heave_acc:>9.4f} "
            f"{improvement(base.rms_heave_acc, m.rms_heave_acc):>+6.1f} "
            f"{m.rms_pitch_acc:>9.4f} "
            f"{improvement(base.rms_pitch_acc, m.rms_pitch_acc):>+6.1f} "
            f"{m.peak_pitch_deg:>9.3f} {m.max_travel*1e3:>10.1f} "
            f"{m.rms_tire*1e3:>11.3f}")
    all_runs[name] = runs

# ---------------------------------------------------------------------------
# Plot: bump — heave, pitch, damping (shows the two-impact pitch event)
# ---------------------------------------------------------------------------
colors = ["#888888", "#1f77b4", "#d62728"]
runs = all_runs["SPEED BUMP"]
fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
for (r, _), c in zip(runs, colors):
    sel = r.t <= 5.0
    ax[0].plot(r.t[sel], r.a_z[sel], color=c, lw=1.2, label=r.label)
    ax[1].plot(r.t[sel], np.rad2deg(r.theta[sel]), color=c, lw=1.2)
    ax[2].plot(r.t[sel], r.a_th[sel], color=c, lw=1.2)
    ax[3].plot(r.t[sel], r.c_f[sel], color=c, lw=1.0)
    ax[3].plot(r.t[sel], r.c_r[sel], color=c, lw=1.0, ls="--", alpha=0.7)
r0 = runs[0][0]; sel = r0.t <= 5.0
ax[1].plot(r0.t[sel], r0.z_rf[sel] * 20, "k:", lw=0.8, alpha=0.5,
           label="road (front, scaled)")
ax[1].plot(r0.t[sel], r0.z_rr[sel] * 20, "k--", lw=0.8, alpha=0.5,
           label="road (rear, scaled)")
ax[0].set_ylabel("Heave accel [m/s²]"); ax[0].legend(fontsize=9)
ax[0].set_title("VISTA v1.30 — Half-Car on Speed Bump: "
                "two-impact event excites pitch (front hit, rear hit +468 ms)")
ax[1].set_ylabel("Pitch angle [deg]"); ax[1].legend(fontsize=8)
ax[2].set_ylabel("Pitch accel [rad/s²]")
ax[3].set_ylabel("Damping F(–)/R(--) [Ns/m]"); ax[3].set_xlabel("Time [s]")
for a in ax: a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v130_halfcar_bump.png"), dpi=140)
plt.close(fig)

with open(os.path.join(OUT, "vista_v130_results.txt"), "w") as f:
    f.write("VISTA v1.30 — Stage 13 (Half-Car) Results\n" + "\n".join(REPORT) + "\n")
print("\nSaved Stage 13 plots + results to ./output")
