"""
VISTA v1.10 — Stages 9-11: Continuous Skyhook, Groundhook, Hybrid
=================================================================
1. Stage 9 : sweep c_sky for continuous skyhook, compare vs 2-state
             skyhook and Digital PID (should closely match the PID).
2. Stage 10: sweep c_gnd for groundhook, evaluate tire deflection
             (dynamic tire load proxy) on a rough road.
3. Stage 11: sweep alpha in the hybrid controller and map the
             comfort <-> road-holding Pareto front.

All controllers run inside the unchanged v1.00 sampled-data framework.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vista import (QuarterCar, SemiActiveDamper, DamperParams,
                   PassiveController, SkyhookController, ContinuousSkyhook,
                   Groundhook, HybridSkyGround, DigitalPID, PIDGains,
                   SpeedBump, RandomRoad, simulate, compute_metrics, improvement)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
REPORT = []

def log(s=""):
    print(s); REPORT.append(s)

car = QuarterCar()
T_END = 8.0
bump = SpeedBump(height=0.05, length=0.5, speed_kmh=20.0, t_start=1.0)
rough = RandomRoad(T_END, roughness=4e-3, seed=99)   # road-holding scenario
medium = RandomRoad(T_END, roughness=2e-3, seed=42)  # mixed scenario

passive_damper = SemiActiveDamper(DamperParams(c_min=2000, c_max=2000, tau_valve=1e-4))
sad = lambda: SemiActiveDamper(DamperParams(c_min=500, c_max=3000, tau_valve=0.015))

pid_gains = PIDGains(kp=6000, ki=0, kd=20, tf=0.010, f_max=4000)

# Baselines
res_pass_b = simulate(car, passive_damper, PassiveController(2000), bump, T_END)
m_pass_b = compute_metrics(res_pass_b, 1.0)
res_pass_r = simulate(car, passive_damper, PassiveController(2000), rough, T_END)
m_pass_r = compute_metrics(res_pass_r, 0.5)
res_pass_m = simulate(car, passive_damper, PassiveController(2000), medium, T_END)
m_pass_m = compute_metrics(res_pass_m, 0.5)

# ===========================================================================
# STAGE 9 — Continuous Skyhook: c_sky sweep
# ===========================================================================
log("=" * 74)
log("STAGE 9 — CONTINUOUS SKYHOOK: c_sky sweep (speed bump, vs passive)")
log("=" * 74)
log(f"{'c_sky':>7} {'RMS acc':>9} {'dRMS%':>7} {'peak':>8} {'dPk%':>7} "
    f"{'travel mm':>10} {'tire mm':>8} {'settle s':>9}")
sky_sweep = []
for c_sky in [1000, 2000, 3000, 4000, 6000, 8000, 12000]:
    r = simulate(car, sad(), ContinuousSkyhook(c_sky), bump, T_END)
    m = compute_metrics(r, 1.0)
    sky_sweep.append((c_sky, m))
    log(f"{c_sky:>7} {m.rms_accel:>9.4f} {improvement(m_pass_b.rms_accel, m.rms_accel):>+7.1f} "
        f"{m.peak_accel:>8.3f} {improvement(m_pass_b.peak_accel, m.peak_accel):>+7.1f} "
        f"{m.max_travel*1e3:>10.1f} {m.max_tire_defl*1e3:>8.1f} {m.settling_time:>9.2f}")

# pick best by same balanced cost as v1.00
def cost_balanced(m, base):
    return (0.25 * m.rms_accel / base.rms_accel
            + 0.15 * m.peak_accel / base.peak_accel
            + 0.20 * m.max_travel / base.max_travel
            + 0.10 * m.max_tire_defl / base.max_tire_defl
            + 0.30 * m.settling_time / max(base.settling_time, 1e-6))

c_sky_best = min(sky_sweep, key=lambda e: cost_balanced(e[1], m_pass_b))[0]
log(f"\nBest c_sky (balanced cost): {c_sky_best} Ns/m")

# Stage 9 comparison: 2-state skyhook vs continuous skyhook vs Digital PID
log("\nStage 9 cross-check on speed bump:")
for ctrl in [SkyhookController(500, 3000), ContinuousSkyhook(c_sky_best),
             DigitalPID(pid_gains)]:
    r = simulate(car, sad(), ctrl, bump, T_END)
    m = compute_metrics(r, 1.0)
    log(f"  {r.label:<28} RMS {m.rms_accel:.4f} ({improvement(m_pass_b.rms_accel, m.rms_accel):+.1f}%)  "
        f"peak {m.peak_accel:.3f} ({improvement(m_pass_b.peak_accel, m.peak_accel):+.1f}%)  "
        f"settle {m.settling_time:.2f}s")

# ===========================================================================
# STAGE 10 — Groundhook: c_gnd sweep (rough road = road-holding scenario)
# ===========================================================================
log("\n" + "=" * 74)
log("STAGE 10 — GROUNDHOOK: c_gnd sweep (rough road, vs passive)")
log("=" * 74)
log(f"{'c_gnd':>7} {'RMStire':>9} {'dTire%':>8} {'maxTire':>9} {'RMS acc':>9} {'dRMS%':>7}")
gnd_sweep = []
for c_gnd in [1000, 2000, 3000, 4000, 6000, 8000, 12000]:
    r = simulate(car, sad(), Groundhook(c_gnd), rough, T_END)
    m = compute_metrics(r, 0.5)
    gnd_sweep.append((c_gnd, m))
    log(f"{c_gnd:>7} {m.rms_tire_defl*1e3:>9.3f} "
        f"{improvement(m_pass_r.rms_tire_defl, m.rms_tire_defl):>+8.1f} "
        f"{m.max_tire_defl*1e3:>9.2f} {m.rms_accel:>9.4f} "
        f"{improvement(m_pass_r.rms_accel, m.rms_accel):>+7.1f}")

best_tire = min(m.rms_tire_defl for _, m in gnd_sweep)
c_gnd_best = min(c for c, m in gnd_sweep if m.rms_tire_defl <= 1.01 * best_tire)
log(f"\nBest c_gnd (smallest gain within 1% of min RMS tire deflection): {c_gnd_best} Ns/m")

# ===========================================================================
# STAGE 11 — Hybrid: alpha sweep -> Pareto front (medium road)
# ===========================================================================
log("\n" + "=" * 74)
log("STAGE 11 — HYBRID SKY-GROUND: alpha sweep (medium road)")
log("=" * 74)
log(f"{'alpha':>6} {'RMS acc':>9} {'dRMS%':>7} {'RMStire':>9} {'dTire%':>8} {'travel mm':>10}")
alphas = np.round(np.linspace(0.0, 1.0, 11), 2)
pareto = []
for a in alphas:
    ctrl = HybridSkyGround(alpha=float(a), c_sky=c_sky_best, c_gnd=c_gnd_best)
    r = simulate(car, sad(), ctrl, medium, T_END)
    m = compute_metrics(r, 0.5)
    pareto.append((a, m))
    log(f"{a:>6.2f} {m.rms_accel:>9.4f} {improvement(m_pass_m.rms_accel, m.rms_accel):>+7.1f} "
        f"{m.rms_tire_defl*1e3:>9.3f} {improvement(m_pass_m.rms_tire_defl, m.rms_tire_defl):>+8.1f} "
        f"{m.max_travel*1e3:>10.2f}")

# --- Pareto filtering: keep only non-dominated points ----------------------
acc = np.array([m.rms_accel for _, m in pareto])
tire = np.array([m.rms_tire_defl for _, m in pareto])
nondom = []
for i in range(len(pareto)):
    dominated = any((acc[j] <= acc[i] and tire[j] <= tire[i] and
                     (acc[j] < acc[i] or tire[j] < tire[i])) for j in range(len(pareto)))
    if not dominated:
        nondom.append(i)
log(f"\nNon-dominated alphas (true Pareto set): "
    f"{[float(pareto[i][0]) for i in nondom]}")

# knee point among non-dominated points: closest to utopia (normalized)
acc_nd, tire_nd = acc[nondom], tire[nondom]
accn = (acc_nd - acc_nd.min()) / max(acc_nd.max() - acc_nd.min(), 1e-12)
tiren = (tire_nd - tire_nd.min()) / max(tire_nd.max() - tire_nd.min(), 1e-12)
knee_i = nondom[int(np.argmin(np.hypot(accn, tiren)))]
alpha_knee = pareto[knee_i][0]
log(f"\nKnee point of Pareto front: alpha = {alpha_knee:.2f} "
    f"(RMS acc {acc[knee_i]:.4f}, RMS tire {tire[knee_i]*1e3:.3f} mm)")

# Also validate knee tuning on the bump
r_knee_b = simulate(car, sad(), HybridSkyGround(alpha_knee, c_sky_best, c_gnd_best), bump, T_END)
m_knee_b = compute_metrics(r_knee_b, 1.0)
log(f"Knee tuning on speed bump: RMS {m_knee_b.rms_accel:.4f} "
    f"({improvement(m_pass_b.rms_accel, m_knee_b.rms_accel):+.1f}%), "
    f"tire {m_knee_b.max_tire_defl*1e3:.1f} mm "
    f"({improvement(m_pass_b.max_tire_defl, m_knee_b.max_tire_defl):+.1f}%), "
    f"settle {m_knee_b.settling_time:.2f}s")

# ===========================================================================
# PLOTS
# ===========================================================================
# --- Pareto front -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
dom = [i for i in range(len(pareto)) if i not in nondom]
ax.scatter(tire[dom] * 1e3, acc[dom], color="#bbbbbb", s=40, zorder=2,
           label="Dominated tunings")
nd_sorted = sorted(nondom, key=lambda i: tire[i])
ax.plot(tire[nd_sorted] * 1e3, acc[nd_sorted], "-o", color="#1f77b4", lw=1.8,
        ms=7, zorder=3, label="Pareto front (non-dominated)")
for (a, m) in pareto:
    ax.annotate(f"α={a:.1f}", (m.rms_tire_defl * 1e3, m.rms_accel),
                textcoords="offset points", xytext=(8, 4), fontsize=8)
ax.scatter([tire[knee_i] * 1e3], [acc[knee_i]], s=160, facecolors="none",
           edgecolors="#d62728", lw=2, zorder=4, label=f"Knee (α={alpha_knee:.2f})")
ax.scatter([m_pass_m.rms_tire_defl * 1e3], [m_pass_m.rms_accel], marker="s", s=90,
           color="#888888", zorder=4, label="Passive (2000 Ns/m)")
# reference: pure controllers at extremes are alpha=1 / alpha=0 already on curve
ax.set_xlabel("RMS tire deflection [mm]  →  worse road holding")
ax.set_ylabel("RMS body acceleration [m/s²]  →  worse comfort")
ax.set_title("VISTA v1.10 — Comfort vs Road-Holding Pareto Front\n"
             f"Hybrid Skyhook-Groundhook, α sweep (c_sky={c_sky_best}, c_gnd={c_gnd_best}), medium road")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v110_pareto_front.png"), dpi=140)
plt.close(fig)

# --- Stage 9 time-domain comparison on bump ---------------------------------
runs = []
for ctrl in [PassiveController(2000), SkyhookController(500, 3000),
             ContinuousSkyhook(c_sky_best), DigitalPID(pid_gains)]:
    d = passive_damper if isinstance(ctrl, PassiveController) else sad()
    runs.append(simulate(car, d, ctrl, bump, T_END))

colors = {"Passive": "#888888", "Skyhook (optimized)": "#1f77b4",
          "Continuous Skyhook": "#2ca02c", "Digital PID (v1.00)": "#d62728"}
fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for r in runs:
    sel = r.t <= 4.0
    c = colors.get(r.label, "#9467bd")
    ax[0].plot(r.t[sel], r.a_s[sel], label=r.label, color=c, lw=1.2)
    ax[1].plot(r.t[sel], r.travel[sel] * 1e3, color=c, lw=1.2)
    ax[2].plot(r.t[sel], r.c_damper[sel], color=c, lw=1.0)
ax[0].set_ylabel("Body accel [m/s²]"); ax[0].legend(fontsize=9)
ax[0].set_title("Stage 9 — Two-state vs Continuous Skyhook vs Digital PID (speed bump)")
ax[1].set_ylabel("Susp. travel [mm]")
ax[2].set_ylabel("Damping [Ns/m]"); ax[2].set_xlabel("Time [s]")
for a in ax: a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v110_stage9_comparison.png"), dpi=140)
plt.close(fig)

# --- Stage 10 groundhook effect on tire deflection (rough road) --------------
r_gnd = simulate(car, sad(), Groundhook(c_gnd_best), rough, T_END)
r_sky_r = simulate(car, sad(), ContinuousSkyhook(c_sky_best), rough, T_END)
fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
for r, c in [(res_pass_r, "#888888"), (r_sky_r, "#2ca02c"), (r_gnd, "#d62728")]:
    sel = (r.t >= 2.0) & (r.t <= 5.0)
    ax[0].plot(r.t[sel], r.tire_defl[sel] * 1e3, color=c, lw=1.0, label=r.label)
    ax[1].plot(r.t[sel], r.a_s[sel], color=c, lw=1.0)
ax[0].set_ylabel("Tire deflection [mm]"); ax[0].legend(fontsize=9)
ax[0].set_title(f"Stage 10 — Groundhook (c_gnd={c_gnd_best}) on rough road: "
                "road holding vs comfort")
ax[1].set_ylabel("Body accel [m/s²]"); ax[1].set_xlabel("Time [s]")
for a in ax: a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v110_stage10_groundhook.png"), dpi=140)
plt.close(fig)

with open(os.path.join(OUT, "vista_v110_results.txt"), "w") as f:
    f.write("VISTA v1.10 — Stages 9-11 Results\n" + "\n".join(REPORT) + "\n")
with open(os.path.join(OUT, "v110_tunings.json"), "w") as f:
    json.dump({"c_sky": c_sky_best, "c_gnd": c_gnd_best,
               "alpha_knee": float(alpha_knee)}, f, indent=2)
print("\nSaved Stage 9-11 plots + results to ./output")
