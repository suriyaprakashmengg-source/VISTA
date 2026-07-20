"""
VISTA v1.00 — Digital Suspension Control Architecture
=====================================================
Main experiment script:

  1. Grid search over digital PID gains (Kp, Ki, Kd) with a balanced
     multi-objective cost (comfort + travel + settling), evaluated on
     the speed bump — the scenario that exposed the old PID's failure.
  2. Three-way comparison (Passive / Optimized Skyhook / Digital PID)
     on speed bump AND random road.
  3. Plots + metrics tables saved to ./output.
"""

import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vista import (QuarterCar, SemiActiveDamper, DamperParams,
                   PassiveController, SkyhookController, DigitalPID, PIDGains,
                   SpeedBump, RandomRoad, simulate, compute_metrics, improvement)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

car = QuarterCar()
bump = SpeedBump(height=0.05, length=0.5, speed_kmh=20.0, t_start=1.0)
T_END = 8.0

# Passive baseline uses a fixed-coefficient damper (no valve dynamics needed)
passive_damper = SemiActiveDamper(DamperParams(c_min=2000, c_max=2000, tau_valve=1e-4))
sa_damper = lambda: SemiActiveDamper(DamperParams(c_min=500, c_max=3000, tau_valve=0.015))

# ---------------------------------------------------------------------------
# 1. Reference runs on the bump
# ---------------------------------------------------------------------------
res_pass_b = simulate(car, passive_damper, PassiveController(2000), bump, T_END)
res_sky_b = simulate(car, sa_damper(), SkyhookController(500, 3000), bump, T_END)
m_pass_b = compute_metrics(res_pass_b, t_event=1.0)
m_sky_b = compute_metrics(res_sky_b, t_event=1.0)

# ---------------------------------------------------------------------------
# 2. PID gain grid search (balanced multi-objective cost)
# ---------------------------------------------------------------------------
def cost_comfort(m, base):
    """Comfort-weighted cost (still bounds travel & settling)."""
    return (0.40 * m.rms_accel / base.rms_accel
            + 0.20 * m.peak_accel / base.peak_accel
            + 0.15 * m.max_travel / base.max_travel
            + 0.10 * m.max_tire_defl / base.max_tire_defl
            + 0.15 * m.settling_time / max(base.settling_time, 1e-6))

def cost_balanced(m, base):
    """Balanced cost: settling and travel penalized as hard as comfort.
    This is the objective that rejects the Stage-5 'floaty' tuning."""
    return (0.25 * m.rms_accel / base.rms_accel
            + 0.15 * m.peak_accel / base.peak_accel
            + 0.20 * m.max_travel / base.max_travel
            + 0.10 * m.max_tire_defl / base.max_tire_defl
            + 0.30 * m.settling_time / max(base.settling_time, 1e-6))

KP = [500, 1000, 2000, 3000, 4000, 6000]
KI = [0, 500, 2000, 5000, 10000]
KD = [0, 20, 50, 100, 200]

evals = []
for kp, ki, kd in itertools.product(KP, KI, KD):
    pid = DigitalPID(PIDGains(kp=kp, ki=ki, kd=kd, tf=0.010, f_max=4000))
    r = simulate(car, sa_damper(), pid, bump, T_END)
    m = compute_metrics(r, t_event=1.0)
    evals.append(((kp, ki, kd), m))

def best_by(costfn):
    scored = sorted(evals, key=lambda e: costfn(e[1], m_pass_b))
    return scored[:5]

top_c = best_by(cost_comfort)
top_b = best_by(cost_balanced)
print("Top 5 COMFORT tunings (kp, ki, kd | RMS | travel mm | settle s):")
for g, m in top_c:
    print(f"  {g}  {m.rms_accel:.4f}  {m.max_travel*1e3:.1f}  {m.settling_time:.2f}")
print("Top 5 BALANCED tunings (kp, ki, kd | RMS | travel mm | settle s):")
for g, m in top_b:
    print(f"  {g}  {m.rms_accel:.4f}  {m.max_travel*1e3:.1f}  {m.settling_time:.2f}")

(kp_c, ki_c, kd_c), _ = top_c[0]
(best_kp, best_ki, best_kd), _ = top_b[0]
gains_comfort = PIDGains(kp=kp_c, ki=ki_c, kd=kd_c, tf=0.010, f_max=4000)
best_gains = PIDGains(kp=best_kp, ki=best_ki, kd=best_kd, tf=0.010, f_max=4000)

# ---------------------------------------------------------------------------
# 3. Final comparison — bump + random road
# ---------------------------------------------------------------------------
rough = RandomRoad(t_end=T_END, roughness=2e-3, seed=42)

same_tuning = (kp_c, ki_c, kd_c) == (best_kp, best_ki, best_kd)

def run_all(road, t_event):
    out = []
    rp = simulate(car, passive_damper, PassiveController(2000), road, T_END)
    rs = simulate(car, sa_damper(), SkyhookController(500, 3000), road, T_END)
    out += [(rp, compute_metrics(rp, t_event)), (rs, compute_metrics(rs, t_event))]
    if not same_tuning:
        rc = simulate(car, sa_damper(), DigitalPID(gains_comfort), road, T_END,
                      label=f"Digital PID Comfort ({kp_c},{ki_c},{kd_c})")
        out.append((rc, compute_metrics(rc, t_event)))
    rd = simulate(car, sa_damper(), DigitalPID(best_gains), road, T_END,
                  label=f"Digital PID ({best_kp},{best_ki},{best_kd})")
    out.append((rd, compute_metrics(rd, t_event)))
    return out

runs_bump = run_all(bump, 1.0)
runs_rough = run_all(rough, 0.5)

def table(runs, title):
    base = runs[0][1]
    lines = [f"\n=== {title} ===",
             f"{'Controller':<38}{'RMS acc':>9}{'Peak acc':>10}{'Travel mm':>11}{'Tire mm':>9}{'Settle s':>9}"]
    for r, m in runs:
        lines.append(f"{r.label:<38}{m.rms_accel:>9.4f}{m.peak_accel:>10.3f}"
                     f"{m.max_travel*1e3:>11.1f}{m.max_tire_defl*1e3:>9.1f}{m.settling_time:>9.2f}")
    for r, m in runs[1:]:
        lines.append(f"  -> {r.label} vs passive: RMS {improvement(base.rms_accel, m.rms_accel):+.1f}%, "
                     f"peak {improvement(base.peak_accel, m.peak_accel):+.1f}%, "
                     f"travel {improvement(base.max_travel, m.max_travel):+.1f}%, "
                     f"tire {improvement(base.max_tire_defl, m.max_tire_defl):+.1f}%, "
                     f"settling {improvement(base.settling_time, m.settling_time):+.1f}%")
    return "\n".join(lines)

report = table(runs_bump, "SPEED BUMP (metrics from t=1.0 s)")
report += "\n" + table(runs_rough, "RANDOM ROAD (metrics from t=0.5 s)")
print(report)
with open(os.path.join(OUT, "vista_v100_results.txt"), "w") as f:
    f.write(f"VISTA v1.00 — Digital Control Architecture — Results\n"
            f"Best PID gains: Kp={best_kp}, Ki={best_ki}, Kd={best_kd}, Tf=10 ms, "
            f"F_max=4000 N, Ts=1 ms, valve tau=15 ms\n" + report + "\n")

# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
def color(lbl):
    if lbl.startswith("Passive"): return "#888888"
    if lbl.startswith("Skyhook"): return "#1f77b4"
    if "Comfort" in lbl: return "#2ca02c"
    return "#d62728"

def plot_comparison(runs, fname, title, tmax=None):
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for r, _ in runs:
        sel = slice(None) if tmax is None else r.t <= tmax
        ax[0].plot(r.t[sel], r.a_s[sel], label=r.label, color=color(r.label), lw=1.2)
        ax[1].plot(r.t[sel], r.z_s[sel] * 1e3, color=color(r.label), lw=1.2)
        ax[2].plot(r.t[sel], r.travel[sel] * 1e3, color=color(r.label), lw=1.2)
        ax[3].plot(r.t[sel], r.c_damper[sel], color=color(r.label), lw=1.0)
    r0 = runs[0][0]
    sel = slice(None) if tmax is None else r0.t <= tmax
    ax[1].plot(r0.t[sel], r0.z_r[sel] * 1e3, "k--", lw=0.8, alpha=0.5, label="road")
    ax[0].set_ylabel("Body accel [m/s²]"); ax[0].legend(loc="upper right", fontsize=9)
    ax[1].set_ylabel("Body disp [mm]")
    ax[2].set_ylabel("Susp. travel [mm]")
    ax[3].set_ylabel("Damping [Ns/m]"); ax[3].set_xlabel("Time [s]")
    ax[0].set_title(title)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)

plot_comparison(runs_bump, "v100_bump_comparison.png",
                "VISTA v1.00 — Speed Bump: Passive vs Skyhook vs Digital PID", tmax=5.0)
plot_comparison(runs_rough, "v100_random_road_comparison.png",
                "VISTA v1.00 — Random Road: Passive vs Skyhook vs Digital PID")

# Force tracking plot: requested vs achieved force (shows passivity clipping)
rd = runs_bump[-1][0]
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
sel = (rd.t >= 0.9) & (rd.t <= 3.0)
ax[0].plot(rd.t[sel], rd.f_request[sel], label="PID force request", color="#d62728", lw=1.0)
ax[0].plot(rd.t[sel], rd.f_damper[sel], label="Achieved damper force", color="#2ca02c", lw=1.0)
ax[0].set_ylabel("Force [N]"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title("Digital PID: force request vs achievable force (passivity clipping)")
ax[1].plot(rd.t[sel], rd.c_damper[sel], color="#9467bd", lw=1.0)
ax[1].set_ylabel("Valve damping [Ns/m]"); ax[1].set_xlabel("Time [s]"); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v100_force_tracking.png"), dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Multi-road robustness validation (v0.92 methodology, applied to v1.00)
# ---------------------------------------------------------------------------
road_set = {
    "Speed bump": (bump, 1.0),
    "Smooth road": (RandomRoad(T_END, roughness=1e-3, seed=7), 0.5),
    "Medium road": (RandomRoad(T_END, roughness=2e-3, seed=42), 0.5),
    "Rough road":  (RandomRoad(T_END, roughness=4e-3, seed=99), 0.5),
}
rob_lines = ["\n=== MULTI-ROAD ROBUSTNESS: Digital PID vs Passive (RMS improvement) ==="]
for name, (road, tev) in road_set.items():
    rp = simulate(car, passive_damper, PassiveController(2000), road, T_END)
    rd_ = simulate(car, sa_damper(), DigitalPID(best_gains), road, T_END)
    mp, md = compute_metrics(rp, tev), compute_metrics(rd_, tev)
    rob_lines.append(f"{name:<14} RMS {improvement(mp.rms_accel, md.rms_accel):+6.1f}%   "
                     f"peak {improvement(mp.peak_accel, md.peak_accel):+6.1f}%   "
                     f"tire {improvement(mp.max_tire_defl, md.max_tire_defl):+6.1f}%")
rob = "\n".join(rob_lines)
print(rob)
with open(os.path.join(OUT, "vista_v100_results.txt"), "a") as f:
    f.write(rob + "\n")

with open(os.path.join(OUT, "best_gains.json"), "w") as f:
    json.dump({"kp": best_kp, "ki": best_ki, "kd": best_kd,
               "tf": 0.010, "f_max": 4000, "ts": 1e-3, "tau_valve": 0.015}, f, indent=2)

print("\nSaved plots + results to ./output")
