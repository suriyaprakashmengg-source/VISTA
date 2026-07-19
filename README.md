# VISTA — Virtual Suspension Test Bench

**Semi-active suspension control simulation platform in Python** — quarter-car and half-car vehicle models, an industrially representative sampled-data control architecture, and a full controller design arc from digital PID to clipped LQR, benchmarked honestly against a validated passive baseline.


## What this project demonstrates

- **Physically correct actuator modeling.** Controllers request *force*, never a damping coefficient. A dedicated semi-active damper layer applies the clipped-optimal law (passivity/dissipativity constraint) with first-order valve dynamics (τ = 15 ms), exactly as a production CDC/MR damper behaves.
- **Real ECU architecture.** Discrete controllers execute at 1 kHz with zero-order hold; the continuous plant integrates with fixed-step RK4 between samples. Digital PID with proper difference equations, derivative filtering on the measurement, and back-calculation anti-windup that treats passivity clipping as actuator saturation.
- **A complete controller design arc behind one interface.** Digital PID → continuous skyhook → groundhook → hybrid blending → clipped LQR, all implementing `Controller.update(measurements, dt) -> force` and all compared inside the identical framework.
- **Honest engineering findings**, including negative results:
  - A sign-convention bug (controller unintentionally running *anti*-skyhook) was caught during v1.00 bring-up **because** the force-based architecture made it diagnosable — the old coefficient-based design hid this error class entirely.
  - Groundhook adds only ~1.5 % road holding over a well-chosen passive damper: the damper envelope, not the control law, limits road holding.
  - **Clipped LQR does not beat the tuned skyhook-groundhook hybrid.** Passivity clipping voids LQR's optimality guarantee (derived for a fully active actuator); the two trade-off fronts essentially coincide.
  - **The production sensing gap is a road-holding limit, not a comfort limit.** Removing the unmeasurable tire-deflection state costs nothing for comfort-weighted LQR designs but is catastrophic (27-point swing) for road-holding-weighted ones.
- **Virtual sensing (LQG).** A steady-state Kalman observer estimating the full state from three noisy production sensors, wrapped with the LQR gain as certainty-equivalence LQG. Result: sensor noise, not sensing topology, dominates comfort performance — LQG recovers the comfort-to-mid front to within a few points of the noiseless full-state bound. But tire deflection proved *practically unobservable*: its 0.25 mm RMS signal sits below the 0.3 mm travel-sensor noise floor, and no filter tuning across four decades explains more than ~21% of it. Comfort control is noise-limited (software-fixable); road-holding control is information-limited (needs preview or new sensors).
- **Model scaling.** The 4-DOF half-car (heave, pitch, wheelbase road delay) reuses the entire quarter-car controller library unchanged via decentralized per-corner control — and per-corner skyhook damps pitch as effectively as heave with zero pitch-specific logic.

## Headline results (speed bump vs optimized passive baseline)

| Controller | RMS accel | Peak accel | Settling |
|---|---|---|---|
| Two-state skyhook (optimized) | +8.4 % | **−12.0 % (worse)** | +0.6 % |
| Digital PID (Kp=6000, Kd=20) | **+42.5 %** | **+52.7 %** | −7.6 % |
| Hybrid α=0.70 (knee point) | +16.3 % | +16.0 % | **+33 %** |
| Clipped LQR (full state) | +18.4 % | +13.3 % | +17 % |

Half-car, per-corner continuous skyhook: heave RMS +32.1 %, pitch RMS +33.0 %, peak pitch 0.41° vs 0.74° passive.

## Repository layout

```
vista/              Core package
  vehicle.py        Quarter-car plant (force-input interface)
  halfcar.py        4-DOF pitch-plane half-car
  damper.py         Semi-active actuator: passivity clip + valve dynamics
  controllers.py    Controller interface + PID/skyhook/groundhook/hybrid
  lqr.py            State-space model, Riccati design, clipped LQR
  simulation.py     Sampled-data loop (quarter car)
  halfsim.py        Sampled-data loop (half car, per-corner ECU channels)
  roads.py          Speed bump, filtered-noise random roads
  metrics.py        Comfort / road-holding / settling metrics
experiments/        One runnable study per version milestone
docs/report/        13-page technical report (PDF + Word)
docs/figures/       All result plots
results/            Metrics tables and tuning files per version
```

## Live interactive demo

`docs/demo/index.html` is a real-time quarter-car bench that runs in any browser with zero dependencies: fixed-step RK4 physics paced to the wall clock, the 1 kHz ECU loop with passivity clipping and valve lag, an animated rig with scrolling road, an oscilloscope panel, and a live passive "shadow vehicle" driving the identical road so the RMS improvement readout is always honest. Switch controllers and drop speed bumps while it runs.

`docs/demo3d/index.html` is the same physics rendered as a **3D virtual model**: rolling wheel, compressing coil spring, telescoping damper that glows as the valve firms, deformable road surface, orbit camera. Same 1 kHz ECU, same honest passive-shadow comparison.

To host it free on GitHub Pages: repo Settings → Pages → deploy from branch `main`, folder `/docs` — the demo is then live at `https://YOU.github.io/vista-suspension/demo/`.

## Quick start

```bash
pip install -r requirements.txt
python experiments/run_vista_v100.py   # digital PID design study
python experiments/run_vista_v110.py   # skyhook/groundhook/hybrid + Pareto front
python experiments/run_vista_v120.py   # clipped LQR vs hybrid
python experiments/run_vista_v130.py   # half-car with per-corner control
```

Each script prints its metrics tables and writes plots to `experiments/output/`.

## Documentation

The full engineering narrative — modeling, architecture, every design study with figures, and the key findings — is in [`docs/report/VISTA_Technical_Report.pdf`](docs/report/VISTA_Technical_Report.pdf). Version-by-version results live in [`results/`](results/) and [`CHANGELOG.md`](CHANGELOG.md).

## Roadmap

Full-car model (7-DOF, roll dynamics) · interactive parameter dashboard · model-predictive control (the constraint handling that clipping performs implicitly is what MPC optimizes explicitly).

## License

MIT
