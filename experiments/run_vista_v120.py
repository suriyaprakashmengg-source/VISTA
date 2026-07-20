"""
VISTA v1.20 — Stage 12: Clipped LQR
===================================
1. Sweep q_tire (road-holding weight) to trace the LQR trade-off curve
   on the medium random road.
2. Overlay it on the Stage-11 hybrid Pareto front (same road, same
   framework) — does optimal control beat the tuned heuristic?
3. Full-state vs measurable-subset LQR: quantify the cost of not being
   able to measure tire deflection in a real vehicle.
4. Validate the chosen LQR on the speed bump.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vista import (QuarterCar, SemiActiveDamper, DamperParams,
                   PassiveController, ContinuousSkyhook, HybridSkyGround,
                   DigitalPID, PIDGains, LQRController, LQRWeights, design_lqr,
                   SpeedBump, RandomRoad, simulate, compute_metrics, improvement)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
REPORT = []
def log(s=""):
    print(s); REPORT.append(s)

car = QuarterCar()
T_END = 8.0
bump = SpeedBump(height=0.05, length=0.5, speed_kmh=20.0, t_start=1.0)
medium = RandomRoad(T_END, roughness=2e-3, seed=42)

passive_damper = SemiActiveDamper(DamperParams(c_min=2000, c_max=2000, tau_valve=1e-4))
sad = lambda: SemiActiveDamper(DamperParams(c_min=500, c_max=3000, tau_valve=0.015))

res_pass_m = simulate(car, passive_damper, PassiveController(2000), medium, T_END)
m_pass_m = compute_metrics(res_pass_m, 0.5)
res_pass_b = simulate(car, passive_damper, PassiveController(2000), bump, T_END)
m_pass_b = compute_metrics(res_pass_b, 1.0)

C_SKY, C_GND = 8000, 4000  # Stage 9/10 tunings

# ===========================================================================
# 1. LQR front: sweep q_tire (full-state, clipped)
# ===========================================================================
log("=" * 76)
log("STAGE 12 — CLIPPED LQR: q_tire sweep (medium road, full-state)")
log("=" * 76)
log(f"{'q_tire':>9} {'RMS acc':>9} {'dRMS%':>7} {'RMStire':>9} {'dTire%':>8} "
    f"{'travel mm':>10} {'K (rounded)':>40}")
q_tires = np.logspace(3, 7.5, 12)
lqr_front = []
for qt in q_tires:
    w = LQRWeights(q_acc=1.0, q_travel=1e4, q_tire=qt, r=1e-8)
    K = design_lqr(car.p, w)
    ctrl = LQRController(K, mode="full", label=f"LQR qt={qt:.0e}")
    r = simulate(car, sad(), ctrl, medium, T_END)
    m = compute_metrics(r, 0.5)
    lqr_front.append((qt, K, m))
    log(f"{qt:>9.1e} {m.rms_accel:>9.4f} {improvement(m_pass_m.rms_accel, m.rms_accel):>+7.1f} "
        f"{m.rms_tire_defl*1e3:>9.3f} {improvement(m_pass_m.rms_tire_defl, m.rms_tire_defl):>+8.1f} "
        f"{m.max_travel*1e3:>10.2f} {str(np.round(K.ravel(), 0)):>40}")

# ===========================================================================
# 2. Hybrid front (recomputed, identical conditions) + overlay
# ===========================================================================
alphas = np.round(np.linspace(0.0, 1.0, 11), 2)
hyb_front = []
for a in alphas:
    r = simulate(car, sad(), HybridSkyGround(float(a), C_SKY, C_GND), medium, T_END)
    hyb_front.append((a, compute_metrics(r, 0.5)))

def nondominated(points):
    """points: list of (acc, tire) tuples -> indices of non-dominated."""
    idx = []
    for i, (ai, ti) in enumerate(points):
        if not any((aj <= ai and tj <= ti and (aj < ai or tj < ti))
                   for j, (aj, tj) in enumerate(points)):
            idx.append(i)
    return idx

lqr_pts = [(m.rms_accel, m.rms_tire_defl) for _, _, m in lqr_front]
hyb_pts = [(m.rms_accel, m.rms_tire_defl) for _, m in hyb_front]
lqr_nd = nondominated(lqr_pts)
hyb_nd = nondominated(hyb_pts)

# Dominance check: for each hybrid ND point, does some LQR point dominate it?
log("\nFront comparison (medium road):")
n_dominated = 0
for i in hyb_nd:
    ah, th = hyb_pts[i]
    dom = any(al <= ah and tl <= th and (al < ah or tl < th) for al, tl in lqr_pts)
    n_dominated += dom
log(f"  Hybrid non-dominated points dominated by an LQR point: "
    f"{n_dominated}/{len(hyb_nd)}")

# ===========================================================================
# 3. Full-state vs measurable LQR at a mid-front weight
# ===========================================================================
qt_sel = 1e6
w_sel = LQRWeights(q_acc=1.0, q_travel=1e4, q_tire=qt_sel, r=1e-8)
K_sel = design_lqr(car.p, w_sel)
log(f"\nSelected weight q_tire={qt_sel:.0e}, K = {np.round(K_sel.ravel(), 0)}")
log(f"{'variant':<22} {'RMS acc':>9} {'dRMS%':>7} {'RMStire':>9} {'dTire%':>8}")
sens_rows = {}
for mode in ("full", "measurable"):
    r = simulate(car, sad(), LQRController(K_sel, mode=mode), medium, T_END)
    m = compute_metrics(r, 0.5)
    sens_rows[mode] = m
    log(f"{'LQR ' + mode:<22} {m.rms_accel:>9.4f} "
        f"{improvement(m_pass_m.rms_accel, m.rms_accel):>+7.1f} "
        f"{m.rms_tire_defl*1e3:>9.3f} "
        f"{improvement(m_pass_m.rms_tire_defl, m.rms_tire_defl):>+8.1f}")
gap = improvement(sens_rows['measurable'].rms_accel, sens_rows['full'].rms_accel)
log(f"  Sensing gap (full vs measurable, RMS acc): {gap:+.1f}%")

# ===========================================================================
# 4. Bump validation vs the calibration menu
# ===========================================================================
log("\nSpeed bump cross-validation:")
log(f"{'Controller':<28} {'RMS acc':>9} {'dRMS%':>7} {'peak':>8} "
    f"{'tire mm':>8} {'settle s':>9}")
for ctrl, damper in [
    (PassiveController(2000), passive_damper),
    (DigitalPID(PIDGains(kp=6000, ki=0, kd=20, tf=0.010, f_max=4000)), sad()),
    (HybridSkyGround(0.70, C_SKY, C_GND), sad()),
    (LQRController(K_sel, "full"), sad()),
    (LQRController(K_sel, "measurable"), sad()),
]:
    r = simulate(car, damper, ctrl, bump, T_END)
    m = compute_metrics(r, 1.0)
    log(f"{r.label:<28} {m.rms_accel:>9.4f} "
        f"{improvement(m_pass_b.rms_accel, m.rms_accel):>+7.1f} {m.peak_accel:>8.3f} "
        f"{m.max_tire_defl*1e3:>8.1f} {m.settling_time:>9.2f}")

# ===========================================================================
# Plot: front overlay
# ===========================================================================
fig, ax = plt.subplots(figsize=(9.5, 7))
hx = np.array([hyb_pts[i][1] for i in hyb_nd]) * 1e3
hy = np.array([hyb_pts[i][0] for i in hyb_nd])
o = np.argsort(hx)
ax.plot(hx[o], hy[o], "-o", color="#1f77b4", lw=1.8, ms=6,
        label="Hybrid Sky-Ground front (α sweep)")
lx = np.array([lqr_pts[i][1] for i in lqr_nd]) * 1e3
ly = np.array([lqr_pts[i][0] for i in lqr_nd])
o = np.argsort(lx)
ax.plot(lx[o], ly[o], "-s", color="#d62728", lw=1.8, ms=6,
        label="Clipped LQR front (q_tire sweep)")
ax.scatter([sens_rows['measurable'].rms_tire_defl * 1e3],
           [sens_rows['measurable'].rms_accel], marker="D", s=90,
           color="#ff7f0e", zorder=4, label="LQR measurable-only sensing")
ax.scatter([m_pass_m.rms_tire_defl * 1e3], [m_pass_m.rms_accel], marker="s",
           s=90, color="#888888", zorder=4, label="Passive (2000 Ns/m)")
ax.set_xlabel("RMS tire deflection [mm]  →  worse road holding")
ax.set_ylabel("RMS body acceleration [m/s²]  →  worse comfort")
ax.set_title("VISTA v1.20 — Clipped LQR vs Hybrid Skyhook-Groundhook\n"
             "Trade-off fronts, medium random road, identical framework")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v120_lqr_vs_hybrid_front.png"), dpi=140)
plt.close(fig)

with open(os.path.join(OUT, "vista_v120_results.txt"), "w") as f:
    f.write("VISTA v1.20 — Stage 12 (Clipped LQR) Results\n" + "\n".join(REPORT) + "\n")
with open(os.path.join(OUT, "v120_lqr_design.json"), "w") as f:
    json.dump({"q_acc": 1.0, "q_travel": 1e4, "q_tire": qt_sel, "r": 1e-8,
               "K": [float(k) for k in K_sel.ravel()]}, f, indent=2)
print("\nSaved Stage 12 plots + results to ./output")
