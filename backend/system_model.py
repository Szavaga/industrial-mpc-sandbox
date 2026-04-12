"""
Nemlineáris CSTR (Continuous Stirred Tank Reactor) — valódi fizikai modell.

Állapotok:
  x[0] = CA  [mol/L]   — reaktor-koncentráció (A komponens)
  x[1] = T   [K]       — reaktor hőmérséklete

Beavatkozók:
  u[0] = F   [L/min]   — tápáram-sebesség
  u[1] = Tc  [K]       — hűtővíz hőmérséklete

Fizikai törvények:
  Arrhenius:      k(T) = k0 · exp(−Ea/R / T)
  Anyagmérleg:    V · dCA/dt = F·(CAf − CA)/60 − V·k(T)·CA
  Energiamérleg:  ρ·Cp·V·dT/dt = ρ·Cp·F·(Tf − T)/60
                                 + (−ΔH)·V·k(T)·CA
                                 − UA·(T − Tc)

Idő [s]-ban, F [L/min]-ban (ezért F/60 az egyenletekben).

Megjegyzés: A stacionárius pont (CA_ss=0.5, T_ss=350 K, Tc_ss=300 K)
nyílt körös INSTABIL — az NMPC aktívan stabilizálja.
Ez szemléletes demo: leállítva a szabályozót, a reaktor "megszalad".
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class CSTRModel:
    # ── Fizikai paraméterek (Gekko CSTR benchmark) ──────────────────────────
    V:      float = 100.0        # Reaktortérfogat [L]
    Caf:    float = 1.0          # Tápkoncentráció [mol/L]
    Tf:     float = 350.0        # Tápáram hőmérséklete [K]
    rho:    float = 1000.0       # Sűrűség [g/L]
    Cp:     float = 0.239        # Fajhő [J/(g·K)]
    mdelH:  float = 5.0e4        # Reakcióhő [J/mol] (exoterm → pozitív)
    k0:     float = 7.2e10/60.0  # Arrhenius pre-exp. [1/s]
    EoverR: float = 8750.0       # Ea/R [K]
    UA:     float = 5.0e4/60.0   # UA hőátadás [J/(s·K)]

    # ── Stacionárius üzempont ────────────────────────────────────────────────
    # Ellenőrzés (lásd system_model_verify.py):
    #   dCA/dt = (100/60/100)*(1−0.5) − k(350)*0.5 = 1/120 − 1/120 = 0 ✓
    #   dT/dt  = (100/60/100)*(350−350) + 209.2*(1/60)*0.5 − (UA/ρCpV)*(350−300) = 0 ✓
    x_ss:   np.ndarray = field(default_factory=lambda: np.array([0.5,   350.0]))
    u_ss:   np.ndarray = field(default_factory=lambda: np.array([100.0, 300.0]))

    # ── Fizikai korlátok ─────────────────────────────────────────────────────
    x_min:  np.ndarray = field(default_factory=lambda: np.array([0.02, 300.0]))
    x_max:  np.ndarray = field(default_factory=lambda: np.array([0.98, 430.0]))
    u_min:  np.ndarray = field(default_factory=lambda: np.array([50.0,  250.0]))
    u_max:  np.ndarray = field(default_factory=lambda: np.array([200.0, 350.0]))
    du_max: np.ndarray = field(default_factory=lambda: np.array([20.0,  5.0]))

    # ── Változó nevek ────────────────────────────────────────────────────────
    state_names: List[str] = field(default_factory=lambda: [
        "Concentration CA [mol/L]", "Temperature T [K]"
    ])
    input_names: List[str] = field(default_factory=lambda: [
        "Feed Flow F [L/min]", "Coolant Temp Tc [K]"
    ])

    # ── Instabilitás küszöbök ────────────────────────────────────────────────
    T_danger:  float = 400.0   # K — sárga figyelmeztetés
    T_runaway: float = 420.0   # K — piros: szabadfutás

    # ── Odvezett konstansok ──────────────────────────────────────────────────
    @property
    def mdelH_rho_Cp(self) -> float:
        """(−ΔH) / (ρ·Cp) [K·L/mol] — az energiamérlegben szereplő arány"""
        return self.mdelH / (self.rho * self.Cp)

    @property
    def UA_rho_Cp_V(self) -> float:
        """UA / (ρ·Cp·V) [1/s] — hőeltávolítási időállandó"""
        return self.UA / (self.rho * self.Cp * self.V)

    # ── Arrhenius sebességi együttható ───────────────────────────────────────
    def k_arrhenius(self, T: float) -> float:
        """k(T) = k0 · exp(−EoverR / T)  [1/s]"""
        return self.k0 * np.exp(-self.EoverR / max(float(T), 200.0))

    # ── Nemlineáris állapot-derivált f(x, u, d) ──────────────────────────────
    def f(self, x: np.ndarray, u: np.ndarray, d: np.ndarray = None) -> np.ndarray:
        """
        Folytonos idejű ODE.

          dCA/dt = F/(60·V)·(CAf − CA) − k(T)·CA + d[0]
          dT/dt  = F/(60·V)·(Tf  − T)  + (ΔH/ρCp)·k(T)·CA
                 − (UA/ρCpV)·(T − Tc) + d[1]
        """
        if d is None:
            d = np.zeros(2)
        CA, T_r = float(x[0]), float(x[1])
        F,  Tc  = float(u[0]), float(u[1])

        q_V = F / 60.0 / self.V          # [1/s]
        k   = self.k_arrhenius(T_r)

        dCA = q_V * (self.Caf - CA) - k * CA                         + d[0]
        dT  = q_V * (self.Tf  - T_r) + self.mdelH_rho_Cp * k * CA \
              - self.UA_rho_Cp_V * (T_r - Tc)                        + d[1]
        return np.array([dCA, dT])

    # ── RK4 integráció ───────────────────────────────────────────────────────
    def rk4_step(
        self, x: np.ndarray, u: np.ndarray, dt: float, d: np.ndarray = None
    ) -> np.ndarray:
        if d is None:
            d = np.zeros(2)
        k1 = self.f(x,               u, d)
        k2 = self.f(x + 0.5*dt*k1,  u, d)
        k3 = self.f(x + 0.5*dt*k2,  u, d)
        k4 = self.f(x + dt*k3,       u, d)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return np.clip(x_next, self.x_min, self.x_max)

    # ── Kényszersértések ellenőrzése ─────────────────────────────────────────
    def check_constraint_violations(
        self, x: np.ndarray, u: np.ndarray, u_prev: np.ndarray
    ) -> dict:
        violations = {}
        for i, name in enumerate(self.state_names):
            if x[i] < self.x_min[i]:
                violations[f"x{i}_low"]  = {
                    "variable": name, "type": "lower_bound",
                    "value": float(x[i]), "limit": float(self.x_min[i]),
                }
            if x[i] > self.x_max[i]:
                violations[f"x{i}_high"] = {
                    "variable": name, "type": "upper_bound",
                    "value": float(x[i]), "limit": float(self.x_max[i]),
                }
        for i, name in enumerate(self.input_names):
            du = abs(u[i] - u_prev[i])
            if du > self.du_max[i]:
                violations[f"du{i}"] = {
                    "variable": name, "type": "rate_limit",
                    "value": float(du), "limit": float(self.du_max[i]),
                }
        return violations

    # ── Linearizált Jacobian (KF és oktatási célra) ──────────────────────────
    def linearize(
        self, x0: np.ndarray = None, u0: np.ndarray = None, eps: float = 1e-5
    ):
        """
        Numerikus Jacobian ∂f/∂x (A_c) és ∂f/∂u (B_c) az adott ponton.
        Alapértelmezetten a stacionárius üzempont körül linearizál.
        """
        x0 = (x0 if x0 is not None else self.x_ss).copy()
        u0 = (u0 if u0 is not None else self.u_ss).copy()
        d0 = np.zeros(2)
        f0 = self.f(x0, u0, d0)
        n, m = len(x0), len(u0)
        A_c = np.zeros((n, n))
        B_c = np.zeros((n, m))
        for i in range(n):
            xp = x0.copy(); xp[i] += eps
            A_c[:, i] = (self.f(xp, u0, d0) - f0) / eps
        for j in range(m):
            up = u0.copy(); up[j] += eps
            B_c[:, j] = (self.f(x0, up, d0) - f0) / eps
        return A_c, B_c

    # ── Instabilitás jelzők ──────────────────────────────────────────────────
    def is_approaching_runaway(self, x: np.ndarray) -> bool:
        return float(x[1]) > self.T_danger

    def is_runaway(self, x: np.ndarray) -> bool:
        return float(x[1]) > self.T_runaway


# ─── Dual CSTR in Series ─────────────────────────────────────────────────────

class DualCSTRModel:
    """
    Két egyforma CSTR sorba kapcsolva.  Állapot x=[CA1,T1,CA2,T2], bemenet u=[F,Tc1,Tc2].

    R1: tápáram betáplálás (CAf, Tf) → kimenet (CA1, T1)
    R2: R1 kimenete mint betáplálás  → kimenet (CA2, T2)

    Mindkét reaktorban azonos Arrhenius-kinetika.  V2 = V/2 (kisebb, gyorsabb reaktor).
    """

    def __init__(self):
        base = CSTRModel()
        # Megosztott fizika
        self.k0       = base.k0
        self.EoverR   = base.EoverR
        self.Caf      = base.Caf
        self.Tf       = base.Tf
        self.rho      = base.rho
        self.Cp       = base.Cp
        self.mdelH    = base.mdelH
        self.V1       = base.V       # 100 L
        self.V2       = base.V / 2   # 50 L
        self.UA1      = base.UA
        self.UA2      = base.UA / 2
        self.T_danger  = base.T_danger
        self.T_runaway = base.T_runaway

        # Levezetett skalárok
        self._mH  = self.mdelH / (self.rho * self.Cp)   # K·L/mol
        self._ua1 = self.UA1 / (self.rho * self.Cp * self.V1)
        self._ua2 = self.UA2 / (self.rho * self.Cp * self.V2)

        # Fizikai korlátok — x=[CA1,T1,CA2,T2]
        self.x_min = np.array([0.02, 300.0, 0.02, 300.0])
        self.x_max = np.array([0.98, 430.0, 0.98, 430.0])
        # u=[F, Tc1, Tc2]
        self.u_min  = np.array([ 50.0, 250.0, 250.0])
        self.u_max  = np.array([200.0, 350.0, 350.0])
        self.du_max = np.array([20.0,    5.0,   5.0])

        # Stacionárius pontok (numerikusan keresve, közelítő értékek)
        # R1 SS = szimpla CSTR SS
        ca1_ss, t1_ss = float(base.x_ss[0]), float(base.x_ss[1])
        # R2 SS: CA2_ss és T2_ss amikor bement = (CA1_ss, T1_ss)
        ca2_ss, t2_ss = self._find_r2_ss(ca1_ss, t1_ss,
                                          float(base.u_ss[0]), float(base.u_ss[1]))
        self.x_ss = np.array([ca1_ss, t1_ss, ca2_ss, t2_ss])
        self.u_ss = np.array([float(base.u_ss[0]), float(base.u_ss[1]),
                              float(base.u_ss[1])])   # Tc2_ss ≈ Tc1_ss

        self.state_names = [
            "R1 — CA1 [mol/L]", "R1 — T1 [K]",
            "R2 — CA2 [mol/L]", "R2 — T2 [K]",
        ]
        self.input_names = [
            "Feed Flow F [L/min]", "Coolant R1 Tc1 [K]", "Coolant R2 Tc2 [K]",
        ]

    def k_arr(self, T: float) -> float:
        return self.k0 * np.exp(-self.EoverR / max(float(T), 200.0))

    def _find_r2_ss(self, ca1, t1, F, tc2):
        """Newton iteration for R2 steady state given R1 output."""
        q2 = F / 60.0 / self.V2
        # Solve:  dCA2/dt = q2*(ca1-CA2) - k(T2)*CA2 = 0
        #         dT2/dt  = q2*(t1-T2) + mH*k(T2)*CA2 - ua2*(T2-tc2) = 0
        # Start from R1 SS values
        x = np.array([ca1 * 0.7, t1 + 5.0])
        for _ in range(200):
            ca2, t2 = x
            k = self.k_arr(t2)
            f0 = q2*(ca1 - ca2) - k*ca2
            f1 = q2*(t1  - t2)  + self._mH*k*ca2 - self._ua2*(t2 - tc2)
            fx = np.array([f0, f1])
            if np.linalg.norm(fx) < 1e-12:
                break
            # Jacobian
            dkdT = self.k_arr(t2 + 1e-4) - k  # finite diff / 1e-4
            J = np.array([
                [-(q2 + k),       -dkdT*ca2            ],
                [ self._mH*k,      -(q2+self._ua2) + self._mH*dkdT*ca2],
            ])
            try:
                dx = np.linalg.solve(J, -fx)
            except np.linalg.LinAlgError:
                break
            x = x + dx
        return float(x[0]), float(x[1])

    def f(self, x: np.ndarray, u: np.ndarray, d: np.ndarray = None) -> np.ndarray:
        """4-dim ODE: x=[CA1,T1,CA2,T2], u=[F,Tc1,Tc2], d=[d_ca1,d_t1,d_ca2,d_t2]"""
        if d is None:
            d = np.zeros(4)
        CA1, T1, CA2, T2 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        F, Tc1, Tc2 = float(u[0]), float(u[1]), float(u[2])

        q1 = F / 60.0 / self.V1
        q2 = F / 60.0 / self.V2
        k1 = self.k_arr(T1)
        k2 = self.k_arr(T2)

        dCA1 = q1*(self.Caf - CA1) - k1*CA1             + d[0]
        dT1  = q1*(self.Tf  - T1)  + self._mH*k1*CA1 - self._ua1*(T1 - Tc1) + d[1]
        dCA2 = q2*(CA1 - CA2)      - k2*CA2             + d[2]
        dT2  = q2*(T1  - T2)       + self._mH*k2*CA2 - self._ua2*(T2 - Tc2) + d[3]
        return np.array([dCA1, dT1, dCA2, dT2])

    def rk4_step(self, x, u, dt, d=None):
        if d is None:
            d = np.zeros(4)
        k1 = self.f(x,               u, d)
        k2 = self.f(x + 0.5*dt*k1,  u, d)
        k3 = self.f(x + 0.5*dt*k2,  u, d)
        k4 = self.f(x + dt*k3,       u, d)
        return np.clip(x + (dt/6.0)*(k1+2*k2+2*k3+k4), self.x_min, self.x_max)

    def linearize(self, x0=None, u0=None, eps=1e-5):
        x0 = (x0 if x0 is not None else self.x_ss).copy()
        u0 = (u0 if u0 is not None else self.u_ss).copy()
        d0 = np.zeros(4)
        f0 = self.f(x0, u0, d0)
        n, m = len(x0), len(u0)
        A_c = np.zeros((n, n))
        B_c = np.zeros((n, m))
        for i in range(n):
            xp = x0.copy(); xp[i] += eps
            A_c[:, i] = (self.f(xp, u0, d0) - f0) / eps
        for j in range(m):
            up = u0.copy(); up[j] += eps
            B_c[:, j] = (self.f(x0, up, d0) - f0) / eps
        return A_c, B_c

    def check_constraint_violations(self, x, u, u_prev):
        violations = {}
        for i in range(4):
            if x[i] < self.x_min[i]:
                violations[f"x{i}_low"]  = {"variable": self.state_names[i],
                    "type": "lower_bound", "value": float(x[i]), "limit": float(self.x_min[i])}
            if x[i] > self.x_max[i]:
                violations[f"x{i}_high"] = {"variable": self.state_names[i],
                    "type": "upper_bound", "value": float(x[i]), "limit": float(self.x_max[i])}
        for i in range(3):
            du = abs(u[i] - u_prev[i])
            if du > self.du_max[i]:
                violations[f"du{i}"] = {"variable": self.input_names[i],
                    "type": "rate_limit", "value": float(du), "limit": float(self.du_max[i])}
        return violations

    def is_approaching_runaway(self, x):
        return float(x[1]) > self.T_danger or float(x[3]) > self.T_danger

    def is_runaway(self, x):
        return float(x[1]) > self.T_runaway or float(x[3]) > self.T_runaway


# Singletons
DEFAULT_MODEL      = CSTRModel()
DEFAULT_DUAL_MODEL = DualCSTRModel()
