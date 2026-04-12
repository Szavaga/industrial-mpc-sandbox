"""
System Identification for CSTR — PRBS + ARX + State-Space conversion.

Workflow:
  1. generate_prbs()          — LFSR-based excitation signal
  2. fit_arx()                — MIMO ARX via least squares (deviation space)
  3. arx_to_ss()              — Observable companion state-space form
  4. sim_step_response*()     — Unit step responses for 3 models (ARX, Jacobian, True)

The identified discrete A_d, B_d can be hot-swapped into MPCController._compute_linear_discrete()
as a data-driven replacement for the Jacobian-linearized matrices.
"""

import numpy as np
from scipy.linalg import expm


# ── LFSR tap tables for maximal-length PRBS sequences ────────────────────────
_LFSR_TAPS = {
    5: [5, 3],
    6: [6, 5],
    7: [7, 6],
    8: [8, 6, 5, 4],
}


def generate_prbs(n_bits: int, clock_period_steps: int, seed_offset: int = 0) -> np.ndarray:
    """
    Generate a PRBS (Pseudo-Random Binary Sequence) using a maximal-length LFSR.

    Parameters
    ----------
    n_bits          : LFSR length. Period = 2^n_bits - 1 symbols.
                      Supported: 5, 6, 7, 8.
    clock_period_steps : Each symbol repeated this many times (ZOH hold).
    seed_offset     : Rotate the raw bit sequence by this amount before ZOH.
                      Use different offsets for F and Tc to avoid correlation.

    Returns
    -------
    np.ndarray, shape = ((2**n_bits - 1) * clock_period_steps,)
    Values in {-1.0, +1.0}.
    """
    if n_bits not in _LFSR_TAPS:
        raise ValueError(f"n_bits must be one of {list(_LFSR_TAPS.keys())}, got {n_bits}")

    taps = _LFSR_TAPS[n_bits]
    period = (1 << n_bits) - 1   # 2^n_bits - 1

    reg = [1] * n_bits
    bits = []
    for _ in range(period):
        bits.append(reg[-1])
        feedback = 0
        for t in taps:
            feedback ^= reg[t - 1]
        reg = [feedback] + reg[:-1]

    # Rotate by seed_offset for decorrelation between channels
    offset = seed_offset % period
    bits = bits[offset:] + bits[:offset]

    # Map {0,1} → {-1, +1} and apply ZOH hold
    seq = np.array(bits, dtype=float) * 2.0 - 1.0   # {0,1} → {-1,+1}
    return np.repeat(seq, clock_period_steps)


# ── ARX identification ────────────────────────────────────────────────────────

def fit_arx(
    y_data: np.ndarray,
    u_data: np.ndarray,
    na: int = 2,
    nb: int = 2,
) -> dict | None:
    """
    Fit a MIMO ARX model via ordinary least squares.

    Model (for each output i):
        y_i(k) = -a1*y_i(k-1) - ... - a_na*y_i(k-na)
               +  b1_1*u1(k-1) + ... + b_nb_1*u1(k-nb)
               +  b1_2*u2(k-1) + ... + b_nb_2*u2(k-nb)

    Parameters
    ----------
    y_data : (N, 2)  — deviation-space outputs  [dCA, dT]
    u_data : (N, 2)  — deviation-space inputs   [dF, dTc]
    na, nb : ARX polynomial orders

    Returns
    -------
    dict with keys:
        theta_ca, theta_t  — np.ndarray (na + 2*nb,)
        fit_pct_ca, fit_pct_t — float, MATLAB NRMSE convention ∈ [0,100]
        na, nb             — int (echoed back)
    None if data is insufficient or lstsq fails.
    """
    N = len(y_data)
    n_skip = max(na, nb) + 5   # discard transient + ARX lag
    n_params = na + 2 * nb     # a1..ana + b_F_1..nb + b_Tc_1..nb

    if N - n_skip < n_params + 10:
        return None   # not enough data

    # Build regression matrix (N - n_skip rows)
    n_rows = N - n_skip
    Phi = np.zeros((n_rows, n_params))
    Y0  = np.zeros(n_rows)
    Y1  = np.zeros(n_rows)

    for i, k in enumerate(range(n_skip, N)):
        row = []
        # AR part: -y(k-1..na)
        for lag in range(1, na + 1):
            row.append(-y_data[k - lag, 0])   # CA part first (fit for CA)
        # X part: u1(k-1..nb), u2(k-1..nb)
        for lag in range(1, nb + 1):
            row.append(u_data[k - lag, 0])    # dF lags
        for lag in range(1, nb + 1):
            row.append(u_data[k - lag, 1])    # dTc lags
        Phi[i] = row
        Y0[i]  = y_data[k, 0]   # dCA
        Y1[i]  = y_data[k, 1]   # dT

    # Solve for CA
    theta_ca, _, _, _ = np.linalg.lstsq(Phi, Y0, rcond=None)
    # Solve for T  (same Phi structure, different AR part)
    # For T output, the AR regressors should use y_T history, not y_CA history.
    # Rebuild Phi for T:
    Phi_T = Phi.copy()
    for i, k in enumerate(range(n_skip, N)):
        for lag_idx, lag in enumerate(range(1, na + 1)):
            Phi_T[i, lag_idx] = -y_data[k - lag, 1]   # T history

    theta_t, _, _, _ = np.linalg.lstsq(Phi_T, Y1, rcond=None)

    # Fit quality (MATLAB NRMSE)
    y0_hat = Phi   @ theta_ca
    y1_hat = Phi_T @ theta_t

    def _fit_pct(y, y_hat):
        denom = np.linalg.norm(y - np.mean(y))
        if denom < 1e-12:
            return 0.0
        return float(np.clip(100.0 * (1.0 - np.linalg.norm(y - y_hat) / denom), 0.0, 100.0))

    return {
        "theta_ca":   theta_ca,
        "theta_t":    theta_t,
        "fit_pct_ca": _fit_pct(Y0, y0_hat),
        "fit_pct_t":  _fit_pct(Y1, y1_hat),
        "na":         na,
        "nb":         nb,
    }


# ── ARX → discrete state-space (observable companion form) ───────────────────

def arx_to_ss(
    theta_ca: np.ndarray,
    theta_t:  np.ndarray,
    na: int,
    nb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert two MISO ARX models to a block-diagonal MIMO discrete state-space.

    State convention for each MISO block (na states):
        z_i = [y_i(k-1), y_i(k-2), ..., y_i(k-na)]^T

    Companion update (ZOH-discrete):
        z_i(k+1) = A_miso @ z_i(k) + B_miso @ u(k)
        y_i(k)   = C_miso @ z_i(k)   where C_miso = [1, 0, ..., 0]

    The nb input lags are embedded into the state update rows (rows 0..nb-1).

    Returns
    -------
    A_d : (2*na × 2*na)  block-diagonal
    B_d : (2*na × 2)
    C_d : (2  × 2*na)
    """
    n_u = 2   # number of inputs

    def _build_miso_ss(theta: np.ndarray):
        # theta = [a1..ana, b_u1_1..nb, b_u2_1..nb]
        a_coeff = theta[:na]          # AR coefficients (already negated in Phi, so +a here)
        b_u1    = theta[na:na+nb]     # lags for input 1
        b_u2    = theta[na+nb:]       # lags for input 2

        # A_miso: observable companion form
        A_m = np.zeros((na, na))
        A_m[0, :] = -a_coeff          # first row: -a1, -a2, ..., -ana
        if na > 1:
            A_m[1:, :-1] = np.eye(na - 1)   # shift: z_i(k-1) → z_i(k-2), etc.

        # B_miso: input lags embedded in state rows
        B_m = np.zeros((na, n_u))
        for lag_idx in range(nb):
            if lag_idx < na:
                B_m[lag_idx, 0] = b_u1[lag_idx]
                B_m[lag_idx, 1] = b_u2[lag_idx]

        C_m = np.zeros((1, na))
        C_m[0, 0] = 1.0

        return A_m, B_m, C_m

    A_ca, B_ca, C_ca = _build_miso_ss(theta_ca)
    A_t,  B_t,  C_t  = _build_miso_ss(theta_t)

    # Block-diagonal assembly
    A_d = np.block([[A_ca, np.zeros((na, na))],
                    [np.zeros((na, na)), A_t]])
    B_d = np.vstack([B_ca, B_t])
    C_d = np.block([[C_ca, np.zeros((1, na))],
                    [np.zeros((1, na)), C_t]])

    return A_d, B_d, C_d


# ── Step response simulations ─────────────────────────────────────────────────

def _step_response_dict(dCA_dF, dCA_dTc, dT_dF, dT_dTc, dt) -> list[dict]:
    n = len(dCA_dF)
    return [
        {
            "time":     round(i * dt, 1),
            "dCA_dF":   float(dCA_dF[i]),
            "dCA_dTc":  float(dCA_dTc[i]),
            "dT_dF":    float(dT_dF[i]),
            "dT_dTc":   float(dT_dTc[i]),
        }
        for i in range(n)
    ]


def sim_step_response(
    A_d: np.ndarray,
    B_d: np.ndarray,
    C_d: np.ndarray,
    n_steps: int = 60,
    dt: float = 1.0,
) -> list[dict]:
    """Simulate discrete state-space step responses for unit steps in u1 and u2."""
    n_aug = A_d.shape[0]

    def _sim(u_step: np.ndarray) -> np.ndarray:
        """Returns (n_steps, 2) output trajectory for a unit step in u_step."""
        z = np.zeros(n_aug)
        outs = np.zeros((n_steps, 2))
        for k in range(n_steps):
            outs[k] = (C_d @ z).flatten()
            z = A_d @ z + B_d @ u_step
        return outs

    out_dF  = _sim(np.array([1.0, 0.0]))   # unit step in F
    out_dTc = _sim(np.array([0.0, 1.0]))   # unit step in Tc

    return _step_response_dict(
        dCA_dF  = out_dF[:, 0],
        dCA_dTc = out_dTc[:, 0],
        dT_dF   = out_dF[:, 1],
        dT_dTc  = out_dTc[:, 1],
        dt=dt,
    )


def sim_step_response_jacobian(
    A_c: np.ndarray,
    B_c: np.ndarray,
    dt: float,
    n_steps: int = 60,
) -> list[dict]:
    """
    Discretize the Jacobian (continuous A_c, B_c) via matrix exponential,
    then simulate step responses.
    """
    n = A_c.shape[0]
    # ZOH discretization via expm of augmented matrix
    M = np.zeros((n + n, n + n))
    M[:n, :n] = A_c
    M[:n, n:] = B_c
    Mexp      = expm(M * dt)
    A_d_jac   = Mexp[:n, :n]
    B_d_jac   = Mexp[:n, n:]
    C_d_jac   = np.eye(n)

    return sim_step_response(A_d_jac, B_d_jac, C_d_jac, n_steps=n_steps, dt=dt)


def sim_step_response_true(model, dt: float, n_steps: int = 60) -> list[dict]:
    """
    Simulate true nonlinear model step responses around SS.
    Step size: 1 L/min for F, 1 K for Tc.
    """
    x_ss = model.x_ss.copy()
    u_ss = model.u_ss.copy()

    def _sim_nl(u_perturb: np.ndarray) -> np.ndarray:
        x = x_ss.copy()
        u = np.clip(u_ss + u_perturb, model.u_min, model.u_max)
        outs = np.zeros((n_steps, len(x_ss)))
        for k in range(n_steps):
            outs[k] = x - x_ss   # deviation
            x = model.rk4_step(x, u, dt, d=np.zeros(len(x_ss)))
        return outs

    out_dF  = _sim_nl(np.array([1.0, 0.0]))
    out_dTc = _sim_nl(np.array([0.0, 1.0]))

    return _step_response_dict(
        dCA_dF  = out_dF[:, 0],
        dCA_dTc = out_dTc[:, 0],
        dT_dF   = out_dF[:, 1],
        dT_dTc  = out_dTc[:, 1],
        dt=dt,
    )
