"""
Industrial MPC Sandbox — FastAPI Backend  (NMPC / Nonlinear CSTR edition)

WebSocket (elsődleges csatorna):
  WS  /ws                 — Valós idejű szimuláció (push + parancsok)

REST (segéd):
  GET  /api/model-info    — Fizikai paraméterek és SS értékek
  POST /api/reset         — Szimuláció visszaállítása
  GET  /api/state         — Debug: aktuális állapot snapshot
  GET  /api/export        — History letöltése CSV-ként
  GET  /api/events        — Eseménynapló (riasztások)

WebSocket üzenetprotokoll
  Client → Server:  { "cmd": "start" | "stop" | "reset" | "setpoints" |
                              "disturbance" | "disturbance_clear" | "config" |
                              "noise" | "sensor_fault" | "sensor_fault_clear" |
                              "economic_config" | "esd" | "esd_clear" | "tick_rate" |
                              "feedforward" | "sysid_start" | "sysid_use_model",
                       "data": { ... } }
  Server → Client:  { "type": "state", ..., "new_alarms": [...] }
                  | { "type": "reset_done" }
                  | { "type": "error", "message": "..." }
"""

import asyncio
import csv
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np

from system_model import DEFAULT_MODEL, DEFAULT_DUAL_MODEL
from mpc_controller import MPCController, MPCConfig
from simulation_state import SimulationState
from sysid import (generate_prbs, fit_arx, arx_to_ss,
                   sim_step_response, sim_step_response_jacobian, sim_step_response_true)
from mhe_estimator import MHEEstimator, MHEConfig


# ---------------------------------------------------------------------------
app = FastAPI(
    title="Industrial NMPC Sandbox API",
    description="Nemlineáris CSTR szimulátor NMPC-vel (Arrhenius + instabil SS)",
    version="4.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mpc_config     = MPCConfig()
mpc_controller = MPCController(model=DEFAULT_MODEL, config=mpc_config)
sim_state      = SimulationState(model=DEFAULT_MODEL, dt=mpc_config.dt)
_disturbances  = np.zeros(2)
_esd_active    = False
_reactor_mode  = "SINGLE"   # "SINGLE" | "SERIES"
_event_log: list = []           # szerver-szintű eseménynapló (max 200 bejegyzés)
_sysid_result:          dict  = None   # last successful fit result
_sysid_identified_model: tuple = None  # (A_d, B_d, C_d) numpy arrays

_mhe_config = MHEConfig()   # live MHE configuration (hot-swappable)
_mhe_estimator: MHEEstimator | None = None   # singleton MHE instance

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nmpc_worker")


# ---------------------------------------------------------------------------
# Eseménylogger
# ---------------------------------------------------------------------------

def _log_event(event_type: str, description: str, level: str, sim_time: float = 0.0) -> dict:
    """
    Esemény naplózása globálisan.
    level: 'critical' | 'warning' | 'info'
    """
    entry = {
        "time":        round(sim_time, 1),
        "type":        event_type,
        "description": description,
        "level":       level,
    }
    _event_log.append(entry)
    if len(_event_log) > 200:
        _event_log.pop(0)
    return entry


# ---------------------------------------------------------------------------
# SysID helper (runs in executor — pure numpy, no asyncio)
# ---------------------------------------------------------------------------

def _run_sysid_fit(y_arr: np.ndarray, u_arr: np.ndarray, cfg: dict) -> dict | None:
    """
    Fits ARX model, converts to state-space, computes three step responses.
    Returns a JSON-serializable dict, or None on failure.
    """
    na = int(cfg.get("na", 2))
    nb = int(cfg.get("nb", 2))

    arx = fit_arx(y_arr, u_arr, na=na, nb=nb)
    if arx is None:
        return None

    A_d, B_d, C_d = arx_to_ss(arx["theta_ca"], arx["theta_t"], na, nb)

    A_c, B_c = DEFAULT_MODEL.linearize()
    sr_arx  = sim_step_response(A_d, B_d, C_d, n_steps=60, dt=mpc_config.dt)
    sr_jac  = sim_step_response_jacobian(A_c, B_c, dt=mpc_config.dt, n_steps=60)
    sr_true = sim_step_response_true(DEFAULT_MODEL, dt=mpc_config.dt, n_steps=60)

    return {
        "fit_pct_ca":    float(arx["fit_pct_ca"]),
        "fit_pct_t":     float(arx["fit_pct_t"]),
        "na":            na,
        "nb":            nb,
        "A_d":           A_d.tolist(),
        "B_d":           B_d.tolist(),
        "C_d":           C_d.tolist(),
        "step_response": {
            "arx":      sr_arx,
            "jacobian": sr_jac,
            "true":     sr_true,
        },
    }


# ---------------------------------------------------------------------------
# Pydantic sémák
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    setpoints: Optional[List[float]] = None


class SetpointRequest(BaseModel):
    ca_sp:   float = Field(..., ge=0.05, le=0.95)
    temp_sp: float = Field(..., ge=305.0, le=425.0)


class DisturbanceRequest(BaseModel):
    d_ca:   float = Field(0.0)
    d_temp: float = Field(0.0)
    duration_steps: int = Field(10, ge=1, le=200)


class MPCConfigRequest(BaseModel):
    prediction_horizon: Optional[int]   = Field(None, ge=5, le=80)
    control_horizon:    Optional[int]   = Field(None, ge=1, le=30)
    Q00: Optional[float] = Field(None, ge=0.1)
    Q11: Optional[float] = Field(None, ge=0.01)
    R00: Optional[float] = Field(None, ge=1e-4)
    R11: Optional[float] = Field(None, ge=1e-4)


class ResetRequest(BaseModel):
    x0: Optional[List[float]] = None
    u0: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/step")
async def step(req: StepRequest):
    global _disturbances
    if req.setpoints is not None:
        if len(req.setpoints) != 2:
            raise HTTPException(400, "Pontosan 2 setpointot kell megadni")
        sim_state.sp = np.array(req.setpoints, dtype=float)
    sp    = sim_state.sp.copy()
    x_hat = sim_state.observe()
    u_opt, predicted_traj, mpc_ok = mpc_controller.compute(
        x0=x_hat, setpoints=sp, u_prev=sim_state.u, disturbances=_disturbances,
    )
    result = sim_state.step(u_opt=u_opt, disturbances=_disturbances, mpc_success=mpc_ok)
    return {**result, "mpc_success": mpc_ok, "predicted_trajectory": predicted_traj,
            "active_disturbances": _disturbances.tolist()}


@app.post("/api/setpoints")
async def update_setpoints(req: SetpointRequest):
    sim_state.sp = np.array([req.ca_sp, req.temp_sp])
    return {"setpoints": sim_state.sp.tolist()}


@app.post("/api/disturbance")
async def inject_disturbance(req: DisturbanceRequest):
    global _disturbances
    _disturbances = np.array([req.d_ca, req.d_temp])
    return {"disturbances": _disturbances.tolist()}


@app.post("/api/disturbance/clear")
async def clear_disturbance():
    global _disturbances
    _disturbances = np.zeros(2)
    return {"disturbances": _disturbances.tolist()}


@app.post("/api/config")
async def update_config(req: MPCConfigRequest):
    update_data = {}
    for field_name in ("prediction_horizon", "control_horizon", "Q00", "Q11", "R00", "R11"):
        val = getattr(req, field_name, None)
        if val is not None:
            update_data[field_name] = val
    mpc_config.update(update_data)
    sim_state.dt = mpc_config.dt
    return {
        "prediction_horizon": mpc_config.prediction_horizon,
        "control_horizon":    mpc_config.control_horizon,
        "Q": mpc_config.Q.tolist(),
        "R": mpc_config.R.tolist(),
        "dt": mpc_config.dt,
    }


@app.post("/api/reset")
async def reset(req: ResetRequest = None):
    global _disturbances, _esd_active
    _disturbances = np.zeros(2)
    _esd_active   = False
    x0 = req.x0 if req and req.x0 else None
    u0 = req.u0 if req and req.u0 else None
    sim_state.reset(x0=x0, u0=u0)
    return {"message": "Visszaállítva", "time": sim_state.time,
            "states": sim_state.x.tolist(), "setpoints": sim_state.sp.tolist()}


@app.get("/api/state")
async def get_state():
    return {
        "time":      sim_state.time,
        "states":    sim_state.x_hat.tolist(),
        "control":   sim_state.u.tolist(),
        "setpoints": sim_state.sp.tolist(),
        "history":   sim_state.get_history(),
        "mpc_config": {
            "prediction_horizon": mpc_config.prediction_horizon,
            "control_horizon":    mpc_config.control_horizon,
            "Q": mpc_config.Q.tolist(),
            "R": mpc_config.R.tolist(),
            "dt": mpc_config.dt,
        },
        "active_disturbances": _disturbances.tolist(),
    }


@app.get("/api/model-info")
async def model_info():
    m = DEFAULT_MODEL
    return {
        "system_name": "Nonlinear CSTR (Exothermic A→B)",
        "description":  "Arrhenius-kinetikájú nemlineáris CSTR — nyílt körös instabil SS",
        "state_equation": "dCA/dt = F/(60V)*(CAf−CA) − k(T)*CA   |   "
                          "dT/dt = F/(60V)*(Tf−T) + (ΔH/ρCp)*k(T)*CA − (UA/ρCpV)*(T−Tc)",
        "arrhenius": f"k(T) = {m.k0:.3e} · exp(−{m.EoverR:.0f}/T)",
        "states": [
            {"index": 0, "name": m.state_names[0], "ss": float(m.x_ss[0]),
             "min": float(m.x_min[0]), "max": float(m.x_max[0]), "unit": "mol/L"},
            {"index": 1, "name": m.state_names[1], "ss": float(m.x_ss[1]),
             "min": float(m.x_min[1]), "max": float(m.x_max[1]), "unit": "K"},
        ],
        "inputs": [
            {"index": 0, "name": m.input_names[0], "ss": float(m.u_ss[0]),
             "min": float(m.u_min[0]), "max": float(m.u_max[0]), "dmax": float(m.du_max[0]), "unit": "L/min"},
            {"index": 1, "name": m.input_names[1], "ss": float(m.u_ss[1]),
             "min": float(m.u_min[1]), "max": float(m.u_max[1]), "dmax": float(m.du_max[1]), "unit": "K"},
        ],
        "T_danger":  m.T_danger,
        "T_runaway": m.T_runaway,
        "note_stability": "A T_ss=350 K, CA_ss=0.5 üzempont nyílt körös INSTABIL (Jacobian λ>0). "
                          "Az NMPC aktívan stabilizálja.",
    }


@app.get("/api/export", summary="Szimulációs history CSV letöltés")
async def export_csv():
    history = sim_state.get_history()
    if not history:
        raise HTTPException(404, "Nincs historikus adat — indítsd el a szimulációt")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "time_s",
        "ca_true_mol_L", "temp_true_K",
        "ca_filtered_mol_L", "temp_filtered_K",
        "ca_raw_mol_L", "temp_raw_K",
        "F_flow_L_min", "Tc_K",
        "sp_ca_mol_L", "sp_temp_K",
        "mpc_ok",
    ])
    for h in history:
        writer.writerow([
            h["time"],
            h["x"][0],     h["x"][1],
            h["x_hat"][0], h["x_hat"][1],
            h["y_meas"][0], h["y_meas"][1],
            h["u"][0],     h["u"][1],
            h["sp"][0],    h["sp"][1],
            int(h["mpc_ok"]),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cstr_simulation_history.csv"},
    )


@app.get("/api/events", summary="Eseménynapló lekérdezése")
async def get_events(limit: int = 50):
    return {"events": _event_log[-limit:], "total": len(_event_log)}


@app.get("/api/sysid/model", summary="Azonosított lineáris modell mátrixok")
async def get_sysid_model():
    if _sysid_result is None:
        raise HTTPException(404, "Nincs azonosított modell — futtass SysID tesztet először")
    return {
        "A_d":        _sysid_result["A_d"],
        "B_d":        _sysid_result["B_d"],
        "C_d":        _sysid_result["C_d"],
        "fit_pct_ca": _sysid_result["fit_pct_ca"],
        "fit_pct_t":  _sysid_result["fit_pct_t"],
        "na":         _sysid_result["na"],
        "nb":         _sysid_result["nb"],
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def simulation_ws(websocket: WebSocket):
    global _disturbances, _esd_active
    await websocket.accept()

    sess = {
        "running":           False,
        "tick_interval":     mpc_config.dt,
        "last_runaway":      False,
        "last_approaching":  False,
        "pending_alarms":    [],    # alarmok amik a következő state üzenetbe kerülnek
        # SysID state
        "sysid_active":  False,
        "sysid_step":    0,
        "sysid_total":   0,
        "sysid_prbs_F":  [],
        "sysid_prbs_Tc": [],
        "sysid_buf_u":   [],
        "sysid_buf_y":   [],
        "sysid_config":  {},
    }

    async def sim_loop():
        global _sysid_result, _sysid_identified_model
        loop = asyncio.get_event_loop()
        while True:
            t0 = loop.time()
            if sess["running"]:
                x_hat  = sim_state.observe()
                u_prev = sim_state.u.copy()
                sp     = sim_state.sp.copy()
                d      = _disturbances.copy()

                # ESD: NMPC felülírása — min áramlás + min hűtővíz hőmérséklet (max hűtés)
                if _esd_active:
                    u_opt = sim_state.model.u_min.copy()
                    pred = {}
                    ok   = True
                elif sess["sysid_active"]:
                    # SysID: override u_opt with PRBS excitation
                    step_idx = sess["sysid_step"]
                    u_opt = np.clip(
                        np.array([sess["sysid_prbs_F"][step_idx],
                                  sess["sysid_prbs_Tc"][step_idx]]),
                        sim_state.model.u_min[:2], sim_state.model.u_max[:2],
                    )
                    pred = {}
                    ok   = True
                    # Record deviation-space I/O
                    sess["sysid_buf_u"].append([
                        float(u_opt[0]) - float(DEFAULT_MODEL.u_ss[0]),
                        float(u_opt[1]) - float(DEFAULT_MODEL.u_ss[1]),
                    ])
                    sess["sysid_buf_y"].append([
                        float(x_hat[0]) - float(DEFAULT_MODEL.x_ss[0]),
                        float(x_hat[1]) - float(DEFAULT_MODEL.x_ss[1]),
                    ])
                    sess["sysid_step"] += 1
                    # Send progress
                    await websocket.send_json({
                        "type":  "sysid_progress",
                        "step":  sess["sysid_step"],
                        "total": sess["sysid_total"],
                    })
                else:
                    u_opt, pred, ok = await loop.run_in_executor(
                        _executor,
                        lambda: mpc_controller.compute(x_hat, sp, u_prev, d),
                    )

                ff_active = mpc_config.feedforward_enabled and bool(np.any(d != 0))
                result = sim_state.step(u_opt, d, ok, ff_enabled=mpc_config.feedforward_enabled)

                # SysID: runaway abort
                if sess["sysid_active"] and result.get("is_runaway", False):
                    sess["sysid_active"] = False
                    entry = _log_event("SYSID_ERR",
                        "SysID aborted — thermal runaway during excitation",
                        "critical", result["time"])
                    sess["pending_alarms"].append(entry)
                    await websocket.send_json({
                        "type": "sysid_error",
                        "message": "Aborted — thermal runaway detected during ID test",
                    })

                # SysID: collection complete → async fit
                if sess["sysid_active"] and sess["sysid_step"] >= sess["sysid_total"]:
                    sess["sysid_active"] = False
                    y_arr = np.array(sess["sysid_buf_y"])
                    u_arr = np.array(sess["sysid_buf_u"])
                    sysid_cfg = dict(sess["sysid_config"])

                    fit_result = await loop.run_in_executor(
                        _executor,
                        lambda: _run_sysid_fit(y_arr, u_arr, sysid_cfg),
                    )
                    if fit_result is not None:
                        _sysid_result = fit_result
                        _sysid_identified_model = (
                            np.array(fit_result["A_d"]),
                            np.array(fit_result["B_d"]),
                            np.array(fit_result["C_d"]),
                        )
                        entry = _log_event("SYSID_OK",
                            f"ARX fit complete: CA={fit_result['fit_pct_ca']:.1f}%, T={fit_result['fit_pct_t']:.1f}%",
                            "info", result["time"])
                        sess["pending_alarms"].append(entry)
                        # Send result (step_response contains Python floats, safe for JSON)
                        await websocket.send_json({
                            "type":          "sysid_result",
                            "fit_pct_ca":    fit_result["fit_pct_ca"],
                            "fit_pct_t":     fit_result["fit_pct_t"],
                            "na":            fit_result["na"],
                            "nb":            fit_result["nb"],
                            "step_response": fit_result["step_response"],
                        })
                    else:
                        await websocket.send_json({
                            "type":    "sysid_error",
                            "message": "ARX fit failed — insufficient data or numerical error",
                        })

                # ── Riasztás detekció ──────────────────────────────────────
                new_alarms = list(sess["pending_alarms"])
                sess["pending_alarms"] = []

                approaching = result.get("approaching_runaway", False)
                runaway     = result.get("is_runaway", False)

                if runaway and not sess["last_runaway"]:
                    entry = _log_event(
                        "RUNAWAY",
                        f"Thermal runaway! T={sim_state.x[1]:.1f} K",
                        "critical", result["time"],
                    )
                    new_alarms.append(entry)
                elif not runaway and sess["last_runaway"]:
                    entry = _log_event(
                        "RUNAWAY_CLR", "System exited runaway zone", "info", result["time"],
                    )
                    new_alarms.append(entry)

                if approaching and not sess["last_approaching"] and not runaway:
                    entry = _log_event(
                        "APPROACHING",
                        f"T={sim_state.x[1]:.1f} K — approaching danger zone",
                        "warning", result["time"],
                    )
                    new_alarms.append(entry)
                elif not approaching and sess["last_approaching"] and not runaway:
                    entry = _log_event(
                        "APPROACHING_CLR", "Temperature back in safe zone", "info", result["time"],
                    )
                    new_alarms.append(entry)

                sess["last_runaway"]   = runaway
                sess["last_approaching"] = approaching

                await websocket.send_json({
                    "type":                  "state",
                    **result,
                    "mpc_success":           ok,
                    "predicted_trajectory":  pred if pred else {},
                    "active_disturbances":   d.tolist(),
                    "noise_sigma":           sim_state.noise_sigma,
                    "esd_active":            _esd_active,
                    "economic_mode":         mpc_config.economic_mode,
                    "controller_type":       mpc_config.controller_type,
                    "reactor_mode":          _reactor_mode,
                    "feedforward_enabled":   mpc_config.feedforward_enabled,
                    "ff_active":             ff_active,
                    "estimator_type":        sim_state.estimator_type,
                    "mhe_success":           sim_state._mhe_success,
                    "mhe_residuals":         sim_state._mhe_residuals.tolist(),
                    "sysid_active":          sess["sysid_active"],
                    "sysid_identified":      _sysid_result is not None,
                    "use_identified_model":  mpc_config.use_identified_model,
                    "linear_model_source":   mpc_controller._id_source,
                    "new_alarms":            new_alarms,
                })

            elapsed   = loop.time() - t0
            wait_time = max(0.0, sess["tick_interval"] - elapsed)
            await asyncio.sleep(wait_time)

    async def recv_loop():
        global _disturbances, _esd_active, _reactor_mode, _sysid_result, _sysid_identified_model
        async for raw in websocket.iter_json():
            cmd  = raw.get("cmd")
            data = raw.get("data", {})

            if cmd == "start":
                sess["running"] = True

            elif cmd == "stop":
                sess["running"] = False

            elif cmd == "reset":
                sess["running"]          = False
                sess["last_runaway"]     = False
                sess["last_approaching"] = False
                sess["pending_alarms"]   = []
                _disturbances = np.zeros(2)
                _esd_active   = False
                sim_state.reset(x0=data.get("x0"), u0=data.get("u0"))
                await websocket.send_json({
                    "type":      "reset_done",
                    "time":      0.0,
                    "states":    sim_state.x.tolist(),
                    "setpoints": sim_state.sp.tolist(),
                })

            elif cmd == "setpoints":
                if _reactor_mode == "SERIES":
                    sim_state.sp = np.array([
                        float(data.get("ca",   sim_state.sp[0])),
                        float(data.get("temp", sim_state.sp[1])),
                        float(data.get("ca2",  sim_state.sp[2])),
                        float(data.get("temp2",sim_state.sp[3])),
                    ])
                else:
                    sim_state.sp = np.array([
                        float(data.get("ca",   sim_state.sp[0])),
                        float(data.get("temp", sim_state.sp[1])),
                    ])

            elif cmd == "disturbance":
                if _reactor_mode == "SERIES":
                    _disturbances = np.array([
                        float(data.get("d_ca",   0.0)),
                        float(data.get("d_temp", 0.0)),
                        0.0, 0.0,
                    ])
                else:
                    _disturbances = np.array([
                        float(data.get("d_ca",   0.0)),
                        float(data.get("d_temp", 0.0)),
                    ])

            elif cmd == "disturbance_clear":
                _disturbances = np.zeros(4 if _reactor_mode == "SERIES" else 2)

            elif cmd == "noise":
                sim_state.noise_sigma = max(0.0, float(data.get("sigma", 0.0)))

            elif cmd == "sensor_fault":
                biasCA = float(data.get("bias_level", 0.0))
                biasT  = float(data.get("bias_temp",  0.0))
                sim_state.sensor_bias = np.array([biasCA, biasT])
                entry = _log_event(
                    "SENSOR_FAULT",
                    f"Sensor bias injected: CA={biasCA:+.3f} mol/L, T={biasT:+.1f} K",
                    "warning", sim_state.time,
                )
                sess["pending_alarms"].append(entry)

            elif cmd == "sensor_fault_clear":
                sim_state.sensor_bias = np.zeros(2)
                entry = _log_event(
                    "SENSOR_CLR", "Sensor fault cleared", "info", sim_state.time,
                )
                sess["pending_alarms"].append(entry)

            elif cmd == "config":
                mpc_config.update(data)
                sess["tick_interval"] = mpc_config.dt
                if "controller_type" in data:
                    ct = str(data["controller_type"]).upper()
                    entry = _log_event(
                        "CTRL_SWITCH",
                        f"Controller switched to {ct}",
                        "info", sim_state.time,
                    )
                    sess["pending_alarms"].append(entry)

            elif cmd == "economic_config":
                mpc_config.update({
                    "economic_mode":  data.get("economic_mode",  mpc_config.economic_mode),
                    "product_price":  data.get("product_price",  mpc_config.product_price),
                    "feedstock_cost": data.get("feedstock_cost", mpc_config.feedstock_cost),
                    "energy_cost":    data.get("energy_cost",    mpc_config.energy_cost),
                })

            elif cmd == "esd":
                _esd_active = True
                entry = _log_event(
                    "ESD",
                    "Emergency Shutdown activated — feed minimized, cooling maximized",
                    "critical", sim_state.time,
                )
                sess["pending_alarms"].append(entry)

            elif cmd == "esd_clear":
                _esd_active = False
                entry = _log_event(
                    "ESD_CLR", "Emergency Shutdown cleared — resuming NMPC", "info", sim_state.time,
                )
                sess["pending_alarms"].append(entry)

            elif cmd == "feedforward":
                enabled = bool(data.get("enabled", False))
                mpc_config.feedforward_enabled = enabled
                entry = _log_event(
                    "FF_ON" if enabled else "FF_OFF",
                    f"Feedforward {'ENABLED — MPC sees measured disturbance' if enabled else 'DISABLED — MPC blind to disturbance'}",
                    "info", sim_state.time,
                )
                sess["pending_alarms"].append(entry)

            elif cmd == "sysid_start":
                if sess["sysid_active"]:
                    pass   # ignore if already running
                else:
                    n_bits      = int(data.get("n_bits",       7))
                    clock_steps = int(data.get("clock_period", 3))
                    amp_F       = float(data.get("amp_F",  10.0))
                    amp_Tc      = float(data.get("amp_Tc",   5.0))
                    na          = int(data.get("na", 2))
                    nb          = int(data.get("nb", 2))

                    prbs_F  = generate_prbs(n_bits, clock_steps, seed_offset=0)
                    prbs_Tc = generate_prbs(n_bits, clock_steps, seed_offset=len(prbs_F) // 3)
                    total   = len(prbs_F)

                    u_ss = DEFAULT_MODEL.u_ss
                    sess["sysid_prbs_F"]  = (float(u_ss[0]) + amp_F  * prbs_F).tolist()
                    sess["sysid_prbs_Tc"] = (float(u_ss[1]) + amp_Tc * prbs_Tc).tolist()
                    sess["sysid_buf_u"]   = []
                    sess["sysid_buf_y"]   = []
                    sess["sysid_step"]    = 0
                    sess["sysid_total"]   = total
                    sess["sysid_config"]  = {
                        "n_bits": n_bits, "clock_period": clock_steps,
                        "amp_F":  amp_F,  "amp_Tc":       amp_Tc,
                        "na":     na,     "nb":           nb,
                    }
                    sess["sysid_active"] = True

                    entry = _log_event("SYSID_START",
                        f"SysID started: PRBS n={n_bits}, T_clk={clock_steps}s, N={total} steps",
                        "info", sim_state.time)
                    sess["pending_alarms"].append(entry)

            elif cmd == "sysid_use_model":
                use_id = bool(data.get("use_identified", False))
                mpc_config.update({"use_identified_model": use_id})
                if use_id and _sysid_identified_model is not None:
                    A_d, B_d, C_d = _sysid_identified_model
                    mpc_controller.set_linear_model(A_d, B_d, C_d, source="identified")
                else:
                    mpc_controller.clear_identified_model()
                entry = _log_event(
                    "SYSID_MODEL",
                    f"Linear MPC model: {'ARX-identified' if use_id else 'Jacobian'}",
                    "info", sim_state.time,
                )
                sess["pending_alarms"].append(entry)

            elif cmd == "reactor_mode":
                new_mode = str(data.get("mode", "SINGLE")).upper()
                if new_mode in ("SINGLE", "SERIES") and new_mode != _reactor_mode:
                    _reactor_mode = new_mode
                    _disturbances = np.zeros(4 if new_mode == "SERIES" else 2)
                    new_model = DEFAULT_DUAL_MODEL if new_mode == "SERIES" else DEFAULT_MODEL
                    sim_state.switch_model(new_model)
                    mpc_controller.set_model(new_model)
                    # ESD uses correct u_min for new model
                    entry = _log_event(
                        "MODE_SWITCH",
                        f"Reactor mode switched to {new_mode}",
                        "info", sim_state.time,
                    )
                    sess["pending_alarms"].append(entry)
                    await websocket.send_json({
                        "type":         "reset_done",
                        "time":         0.0,
                        "states":       sim_state.x.tolist(),
                        "setpoints":    sim_state.sp.tolist(),
                        "reactor_mode": new_mode,
                    })

            elif cmd == "estimator":
                global _mhe_config, _mhe_estimator
                est_type = str(data.get("type", "KF")).upper()
                if est_type == "MHE" and _reactor_mode == "SINGLE":
                    # Build or reconfigure MHE estimator
                    new_cfg = MHEConfig(
                        horizon  = int(data.get("horizon",  _mhe_config.horizon)),
                        R_ca     = float(data.get("R_ca",   _mhe_config.R_ca)),
                        R_t      = float(data.get("R_t",    _mhe_config.R_t)),
                        wmodel   = float(data.get("wmodel", _mhe_config.wmodel)),
                        ev_type  = int(data.get("ev_type",  _mhe_config.ev_type)),
                    )
                    if _mhe_estimator is None:
                        _mhe_estimator = MHEEstimator(DEFAULT_MODEL, new_cfg, mpc_config.dt)
                        _mhe_estimator.warmup()
                    else:
                        _mhe_estimator.reconfigure(new_cfg)
                    _mhe_config = new_cfg
                    sim_state.set_estimator('MHE', _mhe_estimator)
                    entry = _log_event(
                        "EST_MHE",
                        f"Estimator: MHE (N={new_cfg.horizon}, ev={new_cfg.ev_type})",
                        "info", sim_state.time,
                    )
                elif est_type == "KF":
                    sim_state.set_estimator('KF')
                    entry = _log_event(
                        "EST_KF", "Estimator: Linear Kalman Filter", "info", sim_state.time,
                    )
                else:
                    entry = None
                if entry:
                    sess["pending_alarms"].append(entry)

            elif cmd == "tick_rate":
                ms = float(data.get("ms", 600))
                sess["tick_interval"] = max(0.1, ms / 1000.0)

    sim_task  = asyncio.create_task(sim_loop())
    recv_task = asyncio.create_task(recv_loop())

    try:
        done, pending = await asyncio.wait(
            [sim_task, recv_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception() if not task.cancelled() else None
            if exc and not isinstance(exc, WebSocketDisconnect):
                print(f"[WS] task exception: {exc}")
    except Exception as e:
        print(f"[WS] session error: {e}")
        sim_task.cancel()
        recv_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
