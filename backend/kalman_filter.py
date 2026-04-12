"""
Diszkrét Kalman-szűrő 2-dimenziós állapotbecsléshez (deviációs koordinátákban).

Modell (deviációs):
  x_dev(k+1) = A_d * x_dev(k) + B_d * u_dev(k) + w(k),  w ~ N(0, Q_proc)
  y_dev(k)   = x_dev(k)                          + v(k),  v ~ N(0, R_meas)

ahol A_d = I + A_c*dt, B_d = B_c*dt  (elsőrendű Euler-diszkretizáció),
H = I (teljes állapot mérhető).

Deviációs koordináta: x_dev = x - x_ss,  u_dev = u - u_ss
"""

import numpy as np


class DiscreteKalmanFilter:
    def __init__(
        self,
        A_c: np.ndarray,
        B_c: np.ndarray,
        dt: float,
        x_ss: np.ndarray,
        u_ss: np.ndarray,
        Q_proc: np.ndarray = None,
    ):
        n = A_c.shape[0]
        self.n = n
        # Euler-diszkretizáció (deviációs tér)
        self.A_d = np.eye(n) + A_c * dt
        self.B_d = B_c * dt
        self.x_ss = x_ss.copy()
        self.u_ss = u_ss.copy()
        # Folyamatzaj-kovariancia (modellbizonytalanság)
        self.Q = Q_proc if Q_proc is not None else np.eye(n) * 0.1
        # KF állapot (deviációs)
        self.x_hat_dev = np.zeros(n)
        self.P = np.eye(n) * 10.0   # kezdeti bizonytalanság
        self.K = np.zeros((n, n))    # Kalman-erősítés mátrix

    # ── Visszaállítás ──────────────────────────────────────────────────────────
    def reset(self, x0: np.ndarray):
        """KF visszaállítása adott (abszolút) kezdeti állapotból."""
        self.x_hat_dev = (x0 - self.x_ss).copy()
        self.P = np.eye(self.n) * 10.0
        self.K = np.zeros((self.n, self.n))

    # ── Egy KF-lépés ──────────────────────────────────────────────────────────
    def step(
        self,
        y_meas:    np.ndarray,
        u_prev:    np.ndarray,
        sigma_vec, # float or array-like — mérési zaj szórása állapotonként
    ) -> np.ndarray:
        """
        Előrejelzés + frissítés egy lépésben.

        Args:
            y_meas:    zajos mérés abszolút koordinátában  (n,)
            u_prev:    az ELŐZŐ lépésben alkalmazott beavatkozójel (abszolút)
            sigma_vec: skaláris vagy (n,) tömb — mérési zaj szórása állapotonként
                       Például: sigma=2 → sigma_vec=[0.04, 4.0] for CA/T

        Returns:
            x̂(k|k): frissített állapotbecslés abszolút koordinátában
        """
        sv  = np.atleast_1d(sigma_vec)
        if sv.size == 1:
            sv = np.repeat(sv, self.n)
        var = np.maximum(sv ** 2, 1e-6)
        R   = np.diag(var)

        u_dev = u_prev - self.u_ss
        y_dev = y_meas  - self.x_ss

        # ── Előrejelzés ──
        x_pred = self.A_d @ self.x_hat_dev + self.B_d @ u_dev
        P_pred = self.A_d @ self.P @ self.A_d.T + self.Q

        # ── Frissítés (H = I) ──
        S = P_pred + R                              # S = P⁻ + R
        self.K = P_pred @ np.linalg.inv(S)
        innov  = y_dev - x_pred                     # innováció
        self.x_hat_dev = x_pred + self.K @ innov
        self.P = (np.eye(self.n) - self.K) @ P_pred

        return self.x_hat_dev + self.x_ss           # abszolút becslés

    # ── Kalman-erősítés diagonálisa ──────────────────────────────────────────
    @property
    def gain_diag(self):
        """K mátrix diagonális elemei [K[0,0], K[1,1]], értékkészlet ≈ [0..1]."""
        return [float(np.clip(self.K[i, i], 0.0, 1.0)) for i in range(self.n)]
