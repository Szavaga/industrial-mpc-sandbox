"""
Szimulációs állapot kezelése — Single CSTR vagy Dual CSTR in Series.

Automatikusan alkalmazkodik a modell dimenziójához:
  - CSTRModel:     n=2 állapot, m=2 bemenet
  - DualCSTRModel: n=4 állapot, m=3 bemenet
"""

from collections import deque
from typing import List, Optional, Union
import numpy as np
from system_model import CSTRModel, DualCSTRModel, DEFAULT_MODEL
from kalman_filter import DiscreteKalmanFilter


MAX_HISTORY = 500

# Mérési zaj skálázás: sigma=1 → CA: ±0.02 mol/L, T: ±2 K  (egységpáronként)
_NOISE_SCALE_SINGLE = np.array([0.02, 2.0])
_NOISE_SCALE_DUAL   = np.array([0.02, 2.0, 0.02, 2.0])


class SimulationState:
    def __init__(self, model=None, dt: float = 1.0):
        self.model = model or DEFAULT_MODEL
        self.dt    = dt
        self._mhe: Optional[object] = None   # MHEEstimator instance, if active
        self.estimator_type: str = 'KF'      # 'KF' | 'MHE'
        self._mhe_residuals = np.zeros(2)
        self._mhe_success   = False
        self._init_kf()
        self._reset_state()

    # ── KF inicializálás ──────────────────────────────────────────────────────
    def _init_kf(self):
        n = len(self.model.x_ss)
        A_c, B_c = self.model.linearize()
        if n == 2:
            Q_proc = np.diag([1e-4, 10.0])
        else:
            Q_proc = np.diag([1e-4, 10.0, 1e-4, 10.0])

        self._kf = DiscreteKalmanFilter(
            A_c=A_c, B_c=B_c,
            dt=self.dt,
            x_ss=self.model.x_ss,
            u_ss=self.model.u_ss,
            Q_proc=Q_proc,
        )
        self._noise_scale = (
            _NOISE_SCALE_DUAL if n == 4 else _NOISE_SCALE_SINGLE
        )
        # Zajok és szenzorhibák a modell dimenziójában
        self.noise_sigma = 0.0
        self.sensor_bias = np.zeros(n)

    # ── Visszaállítás ─────────────────────────────────────────────────────────
    def _reset_state(self):
        n = len(self.model.x_ss)
        self.x      = self.model.x_ss.copy()
        self.u      = self.model.u_ss.copy()
        self.sp     = self.model.x_ss.copy()
        self.time   = 0.0
        self.y_meas = self.x.copy()
        self.x_hat  = self.x.copy()
        self._kf.reset(self.x)
        if n == 2:
            self._kf.P = np.diag([0.01, 100.0])
        else:
            self._kf.P = np.diag([0.01, 100.0, 0.01, 100.0])
        self._history: deque = deque(maxlen=MAX_HISTORY)
        self.iae_ca:   float = 0.0
        self.iae_temp: float = 0.0
        # Split IAE: accumulated separately when FF is ON vs OFF
        self.iae_ff_on_ca:    float = 0.0
        self.iae_ff_on_temp:  float = 0.0
        self.iae_ff_off_ca:   float = 0.0
        self.iae_ff_off_temp: float = 0.0
        self._time_ff_on:     float = 0.0   # total simulation time with FF enabled
        self._time_ff_off:    float = 0.0   # total simulation time with FF disabled
        # Propagation delay tracker: ring buffer of R1 outputs (CA1,T1)
        self._prop_buf: deque = deque(maxlen=120)   # ~120 s @ dt=1
        self._append_history(constraint_violations={}, mpc_success=True)

    # ── Modellváltás (SINGLE ↔ SERIES) ───────────────────────────────────────
    def switch_model(self, new_model, preserve_noise: bool = True):
        old_sigma = self.noise_sigma
        self.model = new_model
        self._init_kf()
        if preserve_noise:
            self.noise_sigma = old_sigma
        self._reset_state()

    # ── Zajos mérés + Kalman-szűrés / MHE ───────────────────────────────────
    def observe(self) -> np.ndarray:
        n   = len(self.model.x_ss)
        eff = self.noise_sigma * self._noise_scale
        noise = np.random.randn(n) * eff if self.noise_sigma > 0 else np.zeros(n)
        self.y_meas = np.clip(
            self.x + noise + self.sensor_bias,
            self.model.x_min, self.model.x_max,
        )
        if (self.estimator_type == 'MHE'
                and self._mhe is not None
                and n == 2):   # MHE only for single CSTR
            x_hat, ok, res = self._mhe.update(
                self.y_meas[:2], self.u[:2]
            )
            self.x_hat          = np.clip(x_hat, self.model.x_min, self.model.x_max)
            self._mhe_success   = ok
            self._mhe_residuals = res
            # Also keep KF in sync (for seamless KF↔MHE switching)
            self._kf.x_hat = self.x_hat.copy()
        else:
            self.x_hat = self._kf.step(self.y_meas, self.u, eff)
            self.x_hat = np.clip(self.x_hat, self.model.x_min, self.model.x_max)
        return self.x_hat.copy()

    # ── Estimator selector ─────────────────────────────────────────────────────
    def set_estimator(self, estimator_type: str, mhe=None):
        """Switch between 'KF' and 'MHE'.  mhe: MHEEstimator instance or None."""
        self.estimator_type = estimator_type
        if mhe is not None:
            self._mhe = mhe
        if estimator_type == 'KF':
            # Sync KF from current x_hat so switch is bumpless
            self._kf.reset(self.x_hat)

    # ── Szimulációs lépés ─────────────────────────────────────────────────────
    def step(self, u_opt: np.ndarray, disturbances: np.ndarray, mpc_success: bool = True,
             ff_enabled: bool = False) -> dict:
        n = len(self.model.x_ss)
        # Zavarás dimenziójának igazítása
        d = np.zeros(n)
        d[:min(len(disturbances), n)] = disturbances[:min(len(disturbances), n)]

        u_clipped  = np.clip(u_opt, self.model.u_min, self.model.u_max)
        violations = self.model.check_constraint_violations(self.x, u_clipped, self.u)
        x_next     = self.model.rk4_step(self.x, u_clipped, self.dt, d)

        self.u = u_clipped.copy()
        self.x = x_next.copy()
        self.time += self.dt

        # IAE — az elsőre vonatkoztatva (CA1, T1)
        e_ca   = abs(self.x_hat[0] - self.sp[0]) * self.dt
        e_temp = abs(self.x_hat[1] - self.sp[1]) * self.dt
        self.iae_ca   += e_ca
        self.iae_temp += e_temp
        # Split IAE by FF mode for performance comparison
        if ff_enabled:
            self.iae_ff_on_ca   += e_ca
            self.iae_ff_on_temp += e_temp
            self._time_ff_on    += self.dt
        else:
            self.iae_ff_off_ca   += e_ca
            self.iae_ff_off_temp += e_temp
            self._time_ff_off    += self.dt

        # Propagation delay: R1 kimenetének mentése
        if n == 4:
            self._prop_buf.append({
                "time": round(self.time, 1),
                "ca1":  round(float(self.x[0]), 4),
                "t1":   round(float(self.x[1]), 2),
            })

        self._append_history(constraint_violations=violations, mpc_success=mpc_success)
        return self._current_snapshot(violations)

    # ── Belső segédek ─────────────────────────────────────────────────────────
    def _append_history(self, constraint_violations: dict, mpc_success: bool):
        self._history.append({
            "time":    round(self.time, 3),
            "x":       self.x.tolist(),
            "y_meas":  self.y_meas.tolist(),
            "x_hat":   self.x_hat.tolist(),
            "u":       self.u.tolist(),
            "sp":      self.sp.tolist(),
            "violations": list(constraint_violations.keys()),
            "mpc_ok":  mpc_success,
        })

    def _current_snapshot(self, violations: dict) -> dict:
        n = len(self.model.x_ss)
        snap = {
            "time":                   round(self.time, 3),
            "states":                 self.x_hat.tolist(),
            "states_true":            self.x.tolist(),
            "states_raw":             self.y_meas.tolist(),
            "control":                self.u.tolist(),
            "setpoints":              self.sp.tolist(),
            "constraint_violations":  violations,
            "kalman_gain":            self._kf.gain_diag,
            "approaching_runaway":    self.model.is_approaching_runaway(self.x),
            "is_runaway":             self.model.is_runaway(self.x),
            "estimator_type":         self.estimator_type,
            "mhe_success":            bool(self._mhe_success),
            "mhe_residuals":          [round(float(r), 5) for r in self._mhe_residuals],
            "iae_ca":                 round(self.iae_ca,   4),
            "iae_temp":               round(self.iae_temp, 2),
            "iae_ff_on_ca":           round(self.iae_ff_on_ca,    4),
            "iae_ff_on_temp":         round(self.iae_ff_on_temp,  2),
            "iae_ff_off_ca":          round(self.iae_ff_off_ca,   4),
            "iae_ff_off_temp":        round(self.iae_ff_off_temp, 2),
            "time_ff_on":             round(self._time_ff_on,  1),
            "time_ff_off":            round(self._time_ff_off, 1),
            "reactor_mode":           "SERIES" if n == 4 else "SINGLE",
        }
        if n == 4:
            snap["prop_delay_buf"] = list(self._prop_buf)[-10:]  # last 10 pts
        return snap

    # ── Nyilvános API ─────────────────────────────────────────────────────────
    def reset(self, x0: List[float] = None, u0: List[float] = None):
        self._reset_state()
        if x0 is not None:
            self.x = np.clip(np.array(x0, dtype=float), self.model.x_min, self.model.x_max)
        if u0 is not None:
            self.u = np.clip(np.array(u0, dtype=float), self.model.u_min, self.model.u_max)
        self._kf.reset(self.x)
        self.y_meas = self.x.copy()
        self.x_hat  = self.x.copy()
        self._mhe_residuals = np.zeros(2)
        self._mhe_success   = False
        if self._mhe is not None:
            self._mhe.reset(self.x[:2])

    def get_history(self) -> List[dict]:
        return list(self._history)

    @property
    def is_at_steady_state(self) -> bool:
        dx = self.x - self.model.x_ss
        return bool(np.linalg.norm(dx) < 0.01)
