"""
VISTA v1.40 — Gap-1 Study: Virtual Sensing (Kalman Observer + LQG)
==================================================================
Stage-12 finding: road-holding-weighted LQR collapses without the
unmeasurable tire-deflection state (27-point RMS-comfort swing at
q_tire = 1e6). This study asks: how much of that gap does a Kalman
observer recover, using only noisy production sensors?

Variants (medium random road, identical excitation):
  A. Passive baseline
  B. LQR full-state, noiseless      -> theoretical upper bound
  C. LQR truncated, noisy sensors   -> the production collapse
  D. LQG (Kalman + LQR), noisy      -> the proposed fix

All at the road-holding design point q_tire = 1e6, then swept across
the full weight range to compare complete trade-off fronts.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vista import (QuarterCar, SemiActiveDamper, DamperParams,
                   PassiveController, LQRController, LQRWeights, design_lqr,
                   LQGController, SensorNoise, design_kalman,
                   SpeedBump, RandomRoad, simulate, compute_metrics, improvement)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
REPORT = []
def log(s=""):
    print(s); REPORT.append(s)

car = QuarterCar()
T_END = 8.0
medium = RandomRoad(T_END, roughness=2e-3, seed=42)
noise = SensorNoise()          # 0.3 mm travel, 0.02 m/s body, 0.05 m/s wheel

pd = SemiActiveDamper(DamperParams(c_min=2000, c_max=2000, tau_valve=1e-4))
sad = lambda: SemiActiveDamper(DamperParams(c_min=500, c_max=3000, tau_valve=0.015))

m_pass = compute_metrics(simulate(car, pd, PassiveController(2000), medium, T_END), 0.5)

L = design_kalman(car.p, noise, q_road=1e-2)
log("=" * 78)
log("VISTA v1.40 — GAP-1: KALMAN OBSERVER + LQG")
log("=" * 78)
log(f"Sensors: travel (σ=0.3 mm), v_body (σ=0.02 m/s), v_wheel (σ=0.05 m/s)")
log(f"Observability rank 4/4 — tire deflection recoverable in principle")

# ===========================================================================
# 1. Head-to-head at the road-holding design point (q_tire = 1e6)
# ===========================================================================
QT = 1e6
K = design_lqr(car.p, LQRWeights(q_acc=1.0, q_travel=1e4, q_tire=QT, r=1e-8))

def dh(m): return improvement(m_pass.rms_accel, m.rms_accel)
def dt_(m): return improvement(m_pass.rms_tire_defl, m.rms_tire_defl)

log(f"\n--- Head-to-head at q_tire = {QT:.0e} (medium road) ---")
log(f"{'Variant':<42} {'RMS acc':>9} {'dRMS%':>7} {'RMStire':>9} {'dTire%':>8}")

r_full = simulate(car, sad(), LQRController(K, "full", "LQR full-state (noiseless bound)"),
                  medium, T_END)
m_full = compute_metrics(r_full, 0.5)
log(f"{'LQR full-state (noiseless bound)':<42} {m_full.rms_accel:>9.4f} {dh(m_full):>+7.1f} "
    f"{m_full.rms_tire_defl*1e3:>9.3f} {dt_(m_full):>+8.1f}")

r_tr = simulate(car, sad(), LQRController(K, "measurable", "LQR truncated (noisy sensors)"),
                medium, T_END, sensor_noise=noise)
m_tr = compute_metrics(r_tr, 0.5)
log(f"{'LQR truncated (noisy sensors)':<42} {m_tr.rms_accel:>9.4f} {dh(m_tr):>+7.1f} "
    f"{m_tr.rms_tire_defl*1e3:>9.3f} {dt_(m_tr):>+8.1f}")

lqg = LQGController(K, L, car.p, label="LQG Kalman+LQR (noisy sensors)")
r_lqg = simulate(car, sad(), lqg, medium, T_END, sensor_noise=noise)
m_lqg = compute_metrics(r_lqg, 0.5)
log(f"{'LQG Kalman+LQR (noisy sensors)':<42} {m_lqg.rms_accel:>9.4f} {dh(m_lqg):>+7.1f} "
    f"{m_lqg.rms_tire_defl*1e3:>9.3f} {dt_(m_lqg):>+8.1f}")

rec = 100.0 * (dh(m_lqg) - dh(m_tr)) / (dh(m_full) - dh(m_tr))
log(f"\nGap recovery (RMS comfort, truncated -> full): {rec:.0f}%")
log(f"  truncated {dh(m_tr):+.1f}%  ->  LQG {dh(m_lqg):+.1f}%  (bound {dh(m_full):+.1f}%)")

# ===========================================================================
# 2. Estimation quality: true vs estimated tire deflection
# ===========================================================================
est = np.array(lqg.est_log)
n = min(len(est), len(r_lqg.t))
t_e, true_td, est_td = r_lqg.t[:n], r_lqg.tire_defl[:n], est[:n, 2]
err_rms = np.sqrt(np.mean((true_td - est_td)[t_e >= 0.5] ** 2))
sig_rms = np.sqrt(np.mean(true_td[t_e >= 0.5] ** 2))
log(f"\nTire-deflection estimation: RMS error {err_rms*1e3:.3f} mm "
    f"on a {sig_rms*1e3:.3f} mm RMS signal "
    f"({100*(1-err_rms/sig_rms):.0f}% of signal RMS explained)")

fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
sel = (t_e >= 2.0) & (t_e <= 4.0)
ax[0].plot(t_e[sel], true_td[sel] * 1e3, color="#333333", lw=1.2, label="True tire deflection")
ax[0].plot(t_e[sel], est_td[sel] * 1e3, color="#d62728", lw=1.0, ls="--",
           label="Kalman estimate (noisy sensors)")
ax[0].set_ylabel("Tire deflection [mm]"); ax[0].legend(fontsize=9)
ax[0].set_title("VISTA v1.40 — Virtual sensing: estimating the unmeasurable state")
ax[1].plot(t_e[sel], (true_td[sel] - est_td[sel]) * 1e3, color="#1f77b4", lw=1.0)
ax[1].set_ylabel("Estimation error [mm]"); ax[1].set_xlabel("Time [s]")
for a in ax: a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v140_estimation_quality.png"), dpi=140)
plt.close(fig)

# ===========================================================================
# 3. Full fronts: sweep q_tire for all three variants
# ===========================================================================
log(f"\n--- Front sweep over q_tire (medium road) ---")
log(f"{'q_tire':>9} | {'full dRMS%':>10} {'dTire%':>7} | {'trunc dRMS%':>11} {'dTire%':>7} | "
    f"{'LQG dRMS%':>9} {'dTire%':>7}")
q_tires = np.logspace(3, 7, 9)
fronts = {"full": [], "trunc": [], "lqg": []}
for qt in q_tires:
    Kq = design_lqr(car.p, LQRWeights(q_acc=1.0, q_travel=1e4, q_tire=qt, r=1e-8))
    mf = compute_metrics(simulate(car, sad(), LQRController(Kq, "full"), medium, T_END), 0.5)
    mt = compute_metrics(simulate(car, sad(), LQRController(Kq, "measurable"), medium,
                                  T_END, sensor_noise=noise), 0.5)
    ml = compute_metrics(simulate(car, sad(), LQGController(Kq, L, car.p), medium,
                                  T_END, sensor_noise=noise), 0.5)
    fronts["full"].append(mf); fronts["trunc"].append(mt); fronts["lqg"].append(ml)
    log(f"{qt:>9.1e} | {dh(mf):>+10.1f} {dt_(mf):>+7.1f} | {dh(mt):>+11.1f} {dt_(mt):>+7.1f} | "
        f"{dh(ml):>+9.1f} {dt_(ml):>+7.1f}")

fig, ax = plt.subplots(figsize=(9.5, 7))
styles = {"full": ("LQR full-state (noiseless bound)", "#333333", "-", "o"),
          "trunc": ("LQR truncated (noisy sensors)", "#ff7f0e", "--", "^"),
          "lqg": ("LQG Kalman+LQR (noisy sensors)", "#d62728", "-", "s")}
for key, (lbl, c, ls, mk) in styles.items():
    acc = np.array([m.rms_accel for m in fronts[key]])
    tire = np.array([m.rms_tire_defl for m in fronts[key]]) * 1e3
    ax.plot(tire, acc, ls, marker=mk, color=c, lw=1.6, ms=6, label=lbl)
ax.scatter([m_pass.rms_tire_defl * 1e3], [m_pass.rms_accel], marker="s", s=90,
           color="#888888", zorder=4, label="Passive (2000 Ns/m)")
ax.set_xlabel("RMS tire deflection [mm]  →  worse road holding")
ax.set_ylabel("RMS body acceleration [m/s²]  →  worse comfort")
ax.set_title("VISTA v1.40 — Closing the sensing gap with a Kalman observer\n"
             "q_tire sweep, medium road, identical noise realization")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v140_lqg_fronts.png"), dpi=140)
plt.close(fig)

with open(os.path.join(OUT, "vista_v140_results.txt"), "w") as f:
    f.write("VISTA v1.40 — Gap-1 (Kalman Observer + LQG) Results\n"
            + "\n".join(REPORT) + "\n")
print("\nSaved Gap-1 plots + results to ./output")
