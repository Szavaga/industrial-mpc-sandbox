# Industrial MPC Sandbox

A full-stack process control simulator built around an **open-loop unstable exothermic CSTR** — the kind of reactor found in industrial polymerization, nitration, and specialty chemical processes. The platform implements and compares three MPC formulations, two state estimators, and a complete system identification pipeline, all running in real time through a live React dashboard.

> Built as a portfolio project to demonstrate applied process control engineering at an industrial level.

---

## Why This Project

Most control textbooks simulate stable, forgiving systems. This one doesn't. The CSTR operates at an unstable steady state — without the controller, temperature runs away within seconds. Every algorithm here has to actually work, or the simulation breaks.

---

## Control Algorithms

### Nonlinear MPC (NMPC)
- Full Arrhenius ODE embedded directly in the optimizer (Gekko / IPOPT)
- Prediction horizon N = 40 steps
- Handles the open-loop instability natively — no linearization needed
- Economic MPC mode: switches objective from setpoint tracking to profit maximization

### Linear MPC
- Jacobian linearization around the operating steady state
- Deviation-space formulation
- Faster solve times, useful for comparison against NMPC

### Identified MPC
- Hot-swappable: replace the physics model with a data-driven ARX model
- Fitted from closed-loop PRBS excitation data in real time
- Demonstrates model-plant mismatch effects

### Feedforward Compensation
- Measured disturbance injected directly into the MPC predictor
- IAE split metric: compares performance with feedforward ON vs OFF

---

## State Estimation

### Kalman Filter (KF)
- Discrete linear KF in deviation space
- Lightweight, runs every simulation tick
- Tracks noisy concentration and temperature measurements

### Moving Horizon Estimator (MHE)
- Nonlinear, constrained estimation over a receding horizon
- Supports both L1 and L2 norm objectives (robust vs. smooth)
- Hard constraints on state estimates — physically meaningful results even under sensor faults
- Configurable horizon, measurement noise weights, and model error weights

---

## System Identification Pipeline

1. **PRBS Excitation** — pseudo-random binary sequence injected into feed flow
2. **ARX Fitting** — least-squares identification of discrete ARX model
3. **State-Space Realization** — conversion to SS form for MPC compatibility
4. **Validation** — step response comparison between identified and physics model
5. **Hot Swap** — identified model loads into MPC without stopping the simulation

---

## Simulation Features

| Feature | Detail |
|---|---|
| Reactor modes | Single CSTR / Two CSTRs in series |
| Disturbance injection | Feed concentration and temperature disturbances |
| Sensor faults | Bias injection on CA or T measurement |
| Noise | Configurable Gaussian noise on both measurements |
| Emergency Shutdown | Override to minimum feed + maximum cooling |
| Runaway detection | Automatic flag + alarm when T exceeds threshold |
| Constraint handling | Hard limits enforced in MPC and MHE |

---

## What Gets Tracked Per Step

True hidden state · filtered estimate · raw noisy measurement · control inputs (F, Tc) · setpoints · constraint violations · MPC solve status · N-step predicted trajectory · Kalman gain · IAE (total + FF split) · runaway flags · alarm log

---

## Tech Stack

**Backend** — Python, FastAPI, WebSocket, Gekko (IPOPT), NumPy

**Frontend** — React 18, Vite, Recharts, Tailwind CSS

**API** — REST endpoints for control, config, export + real-time WebSocket simulation loop

---

## Dashboard

Three-column layout with live 300-point rolling charts for CA, T, feed flow, and coolant temperature. Each chart shows the filtered estimate, raw measurement, setpoint, MPC predicted trajectory, and constraint bands simultaneously. All controller and estimator parameters are tunable live without restarting the simulation.

---

## Planned Extensions

- [ ] KPI API — settling time, overshoot, control effort metrics
- [ ] Offset-Free MPC — integrating disturbance model for zero steady-state offset
- [ ] Extended Kalman Filter (EKF)
- [ ] Save / Load simulation scenarios
- [ ] NARX nonlinear system identification
- [ ] Tube MPC — robust MPC with explicit uncertainty sets

---

## Getting Started

```bash
# Backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## About

Built by a chemical engineering student interested in the intersection of process control, optimization, and modern software engineering. Open to process/plant engineering roles and internships.
