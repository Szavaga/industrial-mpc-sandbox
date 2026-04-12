"""
Model Predictive Control — két üzemmóddal.

  controller_type = "NONLINEAR" (alapértelmezett):
    Arrhenius-alapú nemlineáris ODE-k, NMPC, instabil SS stabilizálás.

  controller_type = "LINEAR":
    A munkapont körüli Jacobian linearizáció (fix A, B mátrixok).
    Deviációs térben:
      dz/dt = A_c · z + B_c · v     (z = x − x_ss, v = u − u_ss)
    Pontosan egyezik az NMPC-vel a munkapontban, de nagy excursionoknál
    eltér — Arrhenius exponenciális nem közelíthető lineárisan messze SS-től.

  Economic Mode (csak NONLINEAR-ban érdemes):
    max  Σ [ product_price · (F/60)·CA − feedstock_cost · (F/60)·CAf
            − energy_cost · (300−Tc)² · 0.001 ]
"""

import uuid
import numpy as np
from gekko import GEKKO
from system_model import CSTRModel, DualCSTRModel


class MPCConfig:
    def __init__(self):
        self.prediction_horizon: int   = 40
        self.control_horizon:    int   = 10
        self.Q: np.ndarray             = np.diag([50.0, 0.2])
        self.R: np.ndarray             = np.diag([0.001, 0.01])
        self.dt: float                 = 1.0
        # Controller type
        self.controller_type: str      = "NONLINEAR"   # "LINEAR" | "NONLINEAR"
        # Feedforward: MPC sees measured disturbance in prediction model
        self.feedforward_enabled: bool = False
        # Identified model in LINEAR mode
        self.use_identified_model: bool = False
        # Economic mode
        self.economic_mode:  bool  = False
        self.product_price:  float = 10.0
        self.feedstock_cost: float = 2.0
        self.energy_cost:    float = 0.5

    def update(self, data: dict):
        if "prediction_horizon" in data:
            self.prediction_horizon = int(data["prediction_horizon"])
        if "control_horizon" in data:
            self.control_horizon = int(data["control_horizon"])
        if "Q00" in data:
            self.Q[0, 0] = float(data["Q00"])
        if "Q11" in data:
            self.Q[1, 1] = float(data["Q11"])
        if "R00" in data:
            self.R[0, 0] = float(data["R00"])
        if "R11" in data:
            self.R[1, 1] = float(data["R11"])
        if "dt" in data:
            self.dt = float(data["dt"])
        if "controller_type" in data:
            v = str(data["controller_type"]).upper()
            if v in ("LINEAR", "NONLINEAR"):
                self.controller_type = v
        if "feedforward_enabled" in data:
            self.feedforward_enabled = bool(data["feedforward_enabled"])
        if "use_identified_model" in data:
            self.use_identified_model = bool(data["use_identified_model"])
        if "economic_mode" in data:
            self.economic_mode = bool(data["economic_mode"])
        if "product_price" in data:
            self.product_price = float(data["product_price"])
        if "feedstock_cost" in data:
            self.feedstock_cost = float(data["feedstock_cost"])
        if "energy_cost" in data:
            self.energy_cost = float(data["energy_cost"])


class MPCController:
    def __init__(self, model, config: MPCConfig = None):
        self.model  = model
        self.config = config or MPCConfig()
        self._A_c, self._B_c = model.linearize()
        # Identified discrete model (hot-swappable)
        self._A_d_id:        np.ndarray = None
        self._B_d_id:        np.ndarray = None
        self._C_d_id:        np.ndarray = None
        self._use_identified: bool      = False
        self._id_source:      str       = "jacobian"

    def set_model(self, new_model):
        self.model = new_model
        self._A_c, self._B_c = new_model.linearize()

    def set_linear_model(
        self,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray = None,
        source: str = "identified",
    ) -> None:
        """Hot-swap the discrete linear matrices used in _compute_linear_discrete()."""
        n_aug = A_d.shape[0]
        na    = n_aug // 2
        self._A_d_id = A_d.copy()
        self._B_d_id = B_d.copy()
        if C_d is not None:
            self._C_d_id = C_d.copy()
        else:
            # Default: first state of each na-block → output
            C = np.zeros((2, n_aug))
            C[0, 0]  = 1.0
            C[1, na] = 1.0
            self._C_d_id = C
        self._use_identified = True
        self._id_source      = source

    def clear_identified_model(self) -> None:
        self._A_d_id         = None
        self._B_d_id         = None
        self._C_d_id         = None
        self._use_identified = False
        self._id_source      = "jacobian"

    # ── Fő belépési pont ──────────────────────────────────────────────────────
    def compute(
        self,
        x0:           np.ndarray,
        setpoints:    np.ndarray,
        u_prev:       np.ndarray,
        disturbances: np.ndarray = None,
    ) -> tuple[np.ndarray, dict, bool]:
        if disturbances is None:
            disturbances = np.zeros(len(self.model.x_ss))
        # Feedforward: pass measured disturbances into prediction model only if enabled.
        # When disabled the MPC is blind to disturbances and reacts only via state feedback.
        d_pred = disturbances if self.config.feedforward_enabled else np.zeros_like(disturbances)
        if isinstance(self.model, DualCSTRModel):
            return self._compute_series(x0, setpoints, u_prev, d_pred)
        if self.config.controller_type == "LINEAR":
            # Use data-driven identified model if available and toggled on
            if (self._use_identified
                    and self.config.use_identified_model
                    and self._A_d_id is not None):
                return self._compute_linear_discrete(x0, setpoints, u_prev, d_pred)
            return self._compute_linear(x0, setpoints, u_prev, d_pred)
        return self._compute_nonlinear(x0, setpoints, u_prev, d_pred)

    # ── NMPC (Nemlineáris) ────────────────────────────────────────────────────
    def _compute_nonlinear(self, x0, setpoints, u_prev, disturbances):
        cfg = self.config
        mdl = self.model
        N   = cfg.prediction_horizon
        dt  = cfg.dt

        sid = str(uuid.uuid4()).replace("-", "")[:8]
        m   = GEKKO(remote=False, name=f"nmpc_{sid}")
        m.time = [dt * k for k in range(N + 1)]

        k0_s         = mdl.k0
        EoverR       = mdl.EoverR
        Caf          = mdl.Caf
        Tf           = mdl.Tf
        mdelH_rho_Cp = mdl.mdelH_rho_Cp
        UA_rho_Cp_V  = mdl.UA_rho_Cp_V
        V            = mdl.V
        d0, d1       = float(disturbances[0]), float(disturbances[1])

        T_ub = 395.0 if cfg.economic_mode else float(mdl.x_max[1])
        CA = m.Var(value=float(x0[0]), lb=float(mdl.x_min[0]), ub=float(mdl.x_max[0]), name="CA")
        T  = m.Var(value=float(x0[1]), lb=float(mdl.x_min[1]), ub=T_ub,               name="T")

        F  = m.MV(value=float(u_prev[0]), lb=float(mdl.u_min[0]), ub=float(mdl.u_max[0]), name="F")
        Tc = m.MV(value=float(u_prev[1]), lb=float(mdl.u_min[1]), ub=float(mdl.u_max[1]), name="Tc")
        F.STATUS  = 1;  Tc.STATUS  = 1
        F.DCOST   = float(cfg.R[0, 0]);  Tc.DCOST  = float(cfg.R[1, 1])
        F.DMAX    = float(mdl.du_max[0]); Tc.DMAX  = float(mdl.du_max[1])
        F.MV_STEP_HOR  = cfg.control_horizon
        Tc.MV_STEP_HOR = cfg.control_horizon

        cv_CA = m.CV(value=float(x0[0]), name="cv_CA")
        cv_T  = m.CV(value=float(x0[1]), name="cv_T")
        cv_CA.STATUS = 1;  cv_T.STATUS = 1
        cv_CA.SP  = float(setpoints[0]);  cv_T.SP  = float(setpoints[1])
        cv_CA.WSP = float(cfg.Q[0, 0]);   cv_T.WSP = float(cfg.Q[1, 1])
        cv_CA.TAU = 30.0;                 cv_T.TAU = 40.0

        k   = m.Intermediate(k0_s * m.exp(-EoverR / T))
        q_V = m.Intermediate(F / 60.0 / V)
        m.Equation(CA.dt() == q_V*(Caf - CA) - k*CA + d0)
        m.Equation(T.dt()  == q_V*(Tf  - T)  + mdelH_rho_Cp*k*CA - UA_rho_Cp_V*(T - Tc) + d1)
        m.Equation(cv_CA == CA)
        m.Equation(cv_T  == T)

        if cfg.economic_mode:
            cv_CA.STATUS = 0;  cv_T.STATUS = 0
            profit_rate = (cfg.product_price  * (F / 60.0) * CA
                           - cfg.feedstock_cost * (F / 60.0) * Caf
                           - cfg.energy_cost    * ((300.0 - Tc) ** 2) * 0.001)
            m.Obj(-profit_rate)

        m.options.IMODE   = 6
        m.options.CV_TYPE = 2
        m.options.NODES   = 2
        m.options.SOLVER  = 3
        m.options.MAX_ITER = 200

        try:
            m.solve(disp=False)
            u_opt = np.array([float(F.NEWVAL), float(Tc.NEWVAL)])
            predicted = {
                "time": list(m.time),
                "CA":   [float(v) for v in CA.value],
                "T":    [float(v) for v in T.value],
                "u1":   [float(v) for v in F.value],
                "u2":   [float(v) for v in Tc.value],
            }
            success = True
        except Exception as exc:
            print(f"[NMPC] Gekko sikertelen: {exc}")
            u_opt = np.clip(u_prev.copy(), mdl.u_min, mdl.u_max)
            predicted = {}
            success = False
        finally:
            m.cleanup()

        return u_opt, predicted, success

    # ── Lineáris MPC ──────────────────────────────────────────────────────────
    def _compute_linear(self, x0, setpoints, u_prev, disturbances):
        """
        Linearizált MPC — fix Jacobian a munkapontnál (CA_ss=0.5, T_ss=350 K).

        Deviációs tér:  z = x − x_ss,  v = u − u_ss
          dz/dt = A_c · z + B_c · v + d

        Pontossági korlát: az Arrhenius exponenciális k(T)=k0·exp(−Ea/RT)
        1. rendű Taylor-sorral közelítve — nagy T-eltérésnél (>20 K) a modell
        szisztematikusan alul/felülbecsüli a reakciósebességet.
        """
        cfg  = self.config
        mdl  = self.model
        N    = cfg.prediction_horizon
        dt   = cfg.dt
        A_c  = self._A_c
        B_c  = self._B_c

        x_ss = mdl.x_ss.copy()   # [CA_ss, T_ss]
        u_ss = mdl.u_ss.copy()   # [F_ss, Tc_ss]

        d0, d1 = float(disturbances[0]), float(disturbances[1])

        sid = str(uuid.uuid4()).replace("-", "")[:8]
        m   = GEKKO(remote=False, name=f"lmpc_{sid}")
        m.time = [dt * k for k in range(N + 1)]

        # Deviációs állapotok
        z0_init = float(x0[0]) - float(x_ss[0])
        z1_init = float(x0[1]) - float(x_ss[1])

        # Korlátok deviációs térben
        z0_min = float(mdl.x_min[0]) - float(x_ss[0])
        z0_max = float(mdl.x_max[0]) - float(x_ss[0])
        z1_min = float(mdl.x_min[1]) - float(x_ss[1])
        z1_max = float(mdl.x_max[1]) - float(x_ss[1])

        z0 = m.Var(value=z0_init, lb=z0_min, ub=z0_max, name="z_CA")
        z1 = m.Var(value=z1_init, lb=z1_min, ub=z1_max, name="z_T")

        # MV-k deviációs térben (v = u − u_ss)
        v0_init = float(u_prev[0]) - float(u_ss[0])
        v1_init = float(u_prev[1]) - float(u_ss[1])
        v0_min  = float(mdl.u_min[0]) - float(u_ss[0])
        v0_max  = float(mdl.u_max[0]) - float(u_ss[0])
        v1_min  = float(mdl.u_min[1]) - float(u_ss[1])
        v1_max  = float(mdl.u_max[1]) - float(u_ss[1])

        v0 = m.MV(value=v0_init, lb=v0_min, ub=v0_max, name="v_F")
        v1 = m.MV(value=v1_init, lb=v1_min, ub=v1_max, name="v_Tc")
        v0.STATUS = 1;  v1.STATUS = 1
        v0.DCOST  = float(cfg.R[0, 0]);  v1.DCOST  = float(cfg.R[1, 1])
        v0.DMAX   = float(mdl.du_max[0]); v1.DMAX  = float(mdl.du_max[1])
        v0.MV_STEP_HOR = cfg.control_horizon
        v1.MV_STEP_HOR = cfg.control_horizon

        # CV-k eredeti térben: ca = z0 + CA_ss
        ca_abs = m.CV(value=float(x0[0]), name="cv_CA")
        t_abs  = m.CV(value=float(x0[1]), name="cv_T")
        ca_abs.STATUS = 1;  t_abs.STATUS = 1
        ca_abs.SP  = float(setpoints[0]);  t_abs.SP  = float(setpoints[1])
        ca_abs.WSP = float(cfg.Q[0, 0]);   t_abs.WSP = float(cfg.Q[1, 1])
        ca_abs.TAU = 30.0;                 t_abs.TAU = 40.0

        # Lineáris ODE-k (deviációs tér, zavarás hozzáadva)
        # dz0/dt = A[0,0]*z0 + A[0,1]*z1 + B[0,0]*v0 + B[0,1]*v1 + d0
        # dz1/dt = A[1,0]*z0 + A[1,1]*z1 + B[1,0]*v0 + B[1,1]*v1 + d1
        m.Equation(z0.dt() == (float(A_c[0,0])*z0 + float(A_c[0,1])*z1
                               + float(B_c[0,0])*v0 + float(B_c[0,1])*v1 + d0))
        m.Equation(z1.dt() == (float(A_c[1,0])*z0 + float(A_c[1,1])*z1
                               + float(B_c[1,0])*v0 + float(B_c[1,1])*v1 + d1))

        # CV = eredeti állapot = deviáció + SS
        m.Equation(ca_abs == z0 + float(x_ss[0]))
        m.Equation(t_abs  == z1 + float(x_ss[1]))

        m.options.IMODE   = 6
        m.options.CV_TYPE = 2
        m.options.NODES   = 2
        m.options.SOLVER  = 3
        m.options.MAX_ITER = 200

        try:
            m.solve(disp=False)
            # MV értékek visszakonvertálása eredeti térbe
            F_opt  = float(v0.NEWVAL) + float(u_ss[0])
            Tc_opt = float(v1.NEWVAL) + float(u_ss[1])
            u_opt  = np.clip([F_opt, Tc_opt], mdl.u_min, mdl.u_max)

            # Trajektória: eredeti térbe visszakonvertálva
            ca_traj = [float(v) + float(x_ss[0]) for v in z0.value]
            t_traj  = [float(v) + float(x_ss[1]) for v in z1.value]
            f_traj  = [float(v) + float(u_ss[0]) for v in v0.value]
            tc_traj = [float(v) + float(u_ss[1]) for v in v1.value]

            predicted = {
                "time": list(m.time),
                "CA":   ca_traj,
                "T":    t_traj,
                "u1":   f_traj,
                "u2":   tc_traj,
            }
            success = True
        except Exception as exc:
            print(f"[LMPC] Gekko sikertelen: {exc}")
            u_opt = np.clip(u_prev.copy(), mdl.u_min, mdl.u_max)
            predicted = {}
            success = False
        finally:
            m.cleanup()

        return u_opt, predicted, success

    # ── Dual CSTR in Series — 4-state NMPC ────────────────────────────────────
    def _compute_series(self, x0, setpoints, u_prev, disturbances):
        """
        4-state NMPC: x=[CA1,T1,CA2,T2], u=[F,Tc1,Tc2].
        setpoints=[SP_CA1, SP_T1, SP_CA2, SP_T2].
        """
        cfg = self.config
        mdl = self.model
        N   = cfg.prediction_horizon
        dt  = cfg.dt

        sid = str(uuid.uuid4()).replace("-", "")[:8]
        m   = GEKKO(remote=False, name=f"series_{sid}")
        m.time = [dt * k for k in range(N + 1)]

        k0_s = mdl.k0;  EoverR = mdl.EoverR
        Caf = mdl.Caf;   Tf = mdl.Tf
        mH  = mdl.mdelH / (mdl.rho * mdl.Cp)
        ua1 = mdl.UA1 / (mdl.rho * mdl.Cp * mdl.V1)
        ua2 = mdl.UA2 / (mdl.rho * mdl.Cp * mdl.V2)
        V1  = mdl.V1;    V2 = mdl.V2
        d   = disturbances

        CA1 = m.Var(value=float(x0[0]), lb=float(mdl.x_min[0]), ub=float(mdl.x_max[0]))
        T1  = m.Var(value=float(x0[1]), lb=float(mdl.x_min[1]), ub=float(mdl.x_max[1]))
        CA2 = m.Var(value=float(x0[2]), lb=float(mdl.x_min[2]), ub=float(mdl.x_max[2]))
        T2  = m.Var(value=float(x0[3]), lb=float(mdl.x_min[3]), ub=float(mdl.x_max[3]))

        F   = m.MV(value=float(u_prev[0]), lb=float(mdl.u_min[0]), ub=float(mdl.u_max[0]))
        Tc1 = m.MV(value=float(u_prev[1]), lb=float(mdl.u_min[1]), ub=float(mdl.u_max[1]))
        Tc2 = m.MV(value=float(u_prev[2]), lb=float(mdl.u_min[2]), ub=float(mdl.u_max[2]))

        Q00, Q11 = float(cfg.Q[0, 0]), float(cfg.Q[1, 1])
        R00, R11 = float(cfg.R[0, 0]), float(cfg.R[1, 1])
        for mv, rc, dm in [(F, R00, float(mdl.du_max[0])),
                           (Tc1, R11, float(mdl.du_max[1])),
                           (Tc2, R11, float(mdl.du_max[2]))]:
            mv.STATUS = 1;  mv.DCOST = rc;  mv.DMAX = dm
            mv.MV_STEP_HOR = cfg.control_horizon

        sp = setpoints
        cv_CA1 = m.CV(value=float(x0[0]));  cv_T1 = m.CV(value=float(x0[1]))
        cv_CA2 = m.CV(value=float(x0[2]));  cv_T2 = m.CV(value=float(x0[3]))
        for cv, sp_val, w, tau in [
            (cv_CA1, float(sp[0]), Q00, 30.0),
            (cv_T1,  float(sp[1]), Q11, 40.0),
            (cv_CA2, float(sp[2]), Q00, 35.0),
            (cv_T2,  float(sp[3]), Q11, 45.0),
        ]:
            cv.STATUS = 1;  cv.SP = sp_val;  cv.WSP = w;  cv.TAU = tau

        k1  = m.Intermediate(k0_s * m.exp(-EoverR / T1))
        k2  = m.Intermediate(k0_s * m.exp(-EoverR / T2))
        q1  = m.Intermediate(F / 60.0 / V1)
        q2  = m.Intermediate(F / 60.0 / V2)

        m.Equation(CA1.dt() == q1*(Caf - CA1) - k1*CA1 + float(d[0]))
        m.Equation(T1.dt()  == q1*(Tf  - T1)  + mH*k1*CA1 - ua1*(T1-Tc1) + float(d[1]))
        d2 = float(d[2]) if len(d) > 2 else 0.0
        d3 = float(d[3]) if len(d) > 3 else 0.0
        m.Equation(CA2.dt() == q2*(CA1 - CA2) - k2*CA2 + d2)
        m.Equation(T2.dt()  == q2*(T1  - T2)  + mH*k2*CA2 - ua2*(T2-Tc2) + d3)

        m.Equation(cv_CA1 == CA1);  m.Equation(cv_T1 == T1)
        m.Equation(cv_CA2 == CA2);  m.Equation(cv_T2 == T2)

        if cfg.economic_mode:
            for cv in [cv_CA1, cv_T1, cv_CA2, cv_T2]:
                cv.STATUS = 0
            profit = (cfg.product_price  * (F/60.0) * CA2
                      - cfg.feedstock_cost * (F/60.0) * Caf
                      - cfg.energy_cost    * ((300.0-Tc1)**2 + (300.0-Tc2)**2) * 0.0005)
            m.Obj(-profit)

        m.options.IMODE = 6;  m.options.CV_TYPE = 2
        m.options.NODES = 2;  m.options.SOLVER  = 3
        m.options.MAX_ITER = 200

        try:
            m.solve(disp=False)
            u_opt = np.array([float(F.NEWVAL), float(Tc1.NEWVAL), float(Tc2.NEWVAL)])
            predicted = {
                "time": list(m.time),
                "CA":   [float(v) for v in CA1.value],
                "T":    [float(v) for v in T1.value],
                "CA2":  [float(v) for v in CA2.value],
                "T2":   [float(v) for v in T2.value],
                "u1":   [float(v) for v in F.value],
                "u2":   [float(v) for v in Tc1.value],
                "u3":   [float(v) for v in Tc2.value],
            }
            success = True
        except Exception as exc:
            print(f"[SERIES-NMPC] Gekko sikertelen: {exc}")
            u_opt = np.clip(u_prev.copy(), mdl.u_min, mdl.u_max)
            predicted = {}
            success = False
        finally:
            m.cleanup()

        return u_opt, predicted, success

    # ── ARX-Identified Linear MPC (discrete state-space, n_aug-dimensional) ──
    def _compute_linear_discrete(self, x0, setpoints, u_prev, disturbances):
        """
        Linear MPC using the ARX-identified discrete state-space model.

        State: z in R^(2*na) — deviation-space, block structure:
               z = [dCA(k-1), dCA(k-2), dT(k-1), dT(k-2)]   (for na=2)

        Pseudo-continuous representation for Gekko IMODE=6:
               dz/dt = (A_d - I)/dt * z + B_d/dt * v
        """
        cfg   = self.config
        mdl   = self.model
        A_d   = self._A_d_id
        B_d   = self._B_d_id
        C_d   = self._C_d_id
        n_aug = A_d.shape[0]
        na    = n_aug // 2
        dt    = cfg.dt
        N     = cfg.prediction_horizon

        x_ss = mdl.x_ss.copy()
        u_ss = mdl.u_ss.copy()

        A_c_eff = (A_d - np.eye(n_aug)) / dt
        B_c_eff = B_d / dt

        z0 = np.zeros(n_aug)
        z0[0]  = float(x0[0]) - float(x_ss[0])
        z0[na] = float(x0[1]) - float(x_ss[1])

        v0_init = float(u_prev[0]) - float(u_ss[0])
        v1_init = float(u_prev[1]) - float(u_ss[1])
        v0_min  = float(mdl.u_min[0]) - float(u_ss[0])
        v0_max  = float(mdl.u_max[0]) - float(u_ss[0])
        v1_min  = float(mdl.u_min[1]) - float(u_ss[1])
        v1_max  = float(mdl.u_max[1]) - float(u_ss[1])

        sid = str(uuid.uuid4()).replace("-", "")[:8]
        m   = GEKKO(remote=False, name=f"lmpc_id_{sid}")
        m.time = [dt * k for k in range(N + 1)]

        z_vars = []
        for i in range(n_aug):
            if i == 0:
                lb = float(mdl.x_min[0]) - float(x_ss[0])
                ub = float(mdl.x_max[0]) - float(x_ss[0])
            elif i == na:
                lb = float(mdl.x_min[1]) - float(x_ss[1])
                ub = float(mdl.x_max[1]) - float(x_ss[1])
            else:
                lb, ub = -1e3, 1e3
            z_vars.append(m.Var(value=float(z0[i]), lb=lb, ub=ub, name=f"z{i}"))

        v0 = m.MV(value=v0_init, lb=v0_min, ub=v0_max, name="v_F_id")
        v1 = m.MV(value=v1_init, lb=v1_min, ub=v1_max, name="v_Tc_id")
        v0.STATUS = 1;  v1.STATUS = 1
        v0.DCOST  = float(cfg.R[0, 0]);  v1.DCOST  = float(cfg.R[1, 1])
        v0.DMAX   = float(mdl.du_max[0]); v1.DMAX  = float(mdl.du_max[1])
        v0.MV_STEP_HOR = cfg.control_horizon
        v1.MV_STEP_HOR = cfg.control_horizon

        ca_abs = m.CV(value=float(x0[0]), name="cv_CA_id")
        t_abs  = m.CV(value=float(x0[1]), name="cv_T_id")
        ca_abs.STATUS = 1;  t_abs.STATUS = 1
        ca_abs.SP  = float(setpoints[0]);  t_abs.SP  = float(setpoints[1])
        ca_abs.WSP = float(cfg.Q[0, 0]);   t_abs.WSP = float(cfg.Q[1, 1])
        ca_abs.TAU = 30.0;                 t_abs.TAU = 40.0

        v_in = [v0, v1]
        for i in range(n_aug):
            rhs = (sum(float(A_c_eff[i, j]) * z_vars[j] for j in range(n_aug))
                   + sum(float(B_c_eff[i, k]) * v_in[k] for k in range(2)))
            m.Equation(z_vars[i].dt() == rhs)

        ca_expr = sum(float(C_d[0, j]) * z_vars[j] for j in range(n_aug)) + float(x_ss[0])
        t_expr  = sum(float(C_d[1, j]) * z_vars[j] for j in range(n_aug)) + float(x_ss[1])
        m.Equation(ca_abs == ca_expr)
        m.Equation(t_abs  == t_expr)

        m.options.IMODE   = 6
        m.options.CV_TYPE = 2
        m.options.NODES   = 2
        m.options.SOLVER  = 3
        m.options.MAX_ITER = 200

        try:
            m.solve(disp=False)
            F_opt  = float(v0.NEWVAL) + float(u_ss[0])
            Tc_opt = float(v1.NEWVAL) + float(u_ss[1])
            u_opt  = np.clip([F_opt, Tc_opt], mdl.u_min, mdl.u_max)

            ca_traj = [
                sum(float(C_d[0, j]) * float(z_vars[j].value[k]) for j in range(n_aug)) + float(x_ss[0])
                for k in range(N + 1)
            ]
            t_traj = [
                sum(float(C_d[1, j]) * float(z_vars[j].value[k]) for j in range(n_aug)) + float(x_ss[1])
                for k in range(N + 1)
            ]
            f_traj  = [float(v0.value[k]) + float(u_ss[0]) for k in range(N + 1)]
            tc_traj = [float(v1.value[k]) + float(u_ss[1]) for k in range(N + 1)]

            predicted = {
                "time": list(m.time),
                "CA":   ca_traj,
                "T":    t_traj,
                "u1":   f_traj,
                "u2":   tc_traj,
            }
            success = True
        except Exception as exc:
            print(f"[LMPC-ID] Gekko sikertelen: {exc}")
            u_opt    = np.clip(u_prev.copy(), mdl.u_min, mdl.u_max)
            predicted = {}
            success  = False
        finally:
            m.cleanup()

        return u_opt, predicted, success
