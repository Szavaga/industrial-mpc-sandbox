/**
 * Backend API kliens.
 * Vite proxy: /api → http://localhost:8000/api
 */

const BASE = '/api'

async function request(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) opts.body = JSON.stringify(body)
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API ${method} ${path} → ${res.status}: ${text}`)
  }
  return res.json()
}

const get  = (path)       => request('GET',  path)
const post = (path, body) => request('POST', path, body)

export const api = {
  /** Egy szimulációs időlépés (MPC + RK4). */
  step: (setpoints) =>
    post('/step', setpoints ? { setpoints } : {}),

  /** Setpointok frissítése futás közben. */
  setSetpoints: (level_sp, temp_sp) =>
    post('/setpoints', { level_sp, temp_sp }),

  /** Zavarás bevitele. */
  injectDisturbance: (d_level, d_temp, duration_steps = 20) =>
    post('/disturbance', { d_level, d_temp, duration_steps }),

  /** Zavarás törlése. */
  clearDisturbance: () =>
    post('/disturbance/clear'),

  /** MPC konfiguráció módosítása. */
  updateConfig: (cfg) =>
    post('/config', cfg),

  /** Szimuláció nullázása. */
  reset: (x0, u0) =>
    post('/reset', { x0: x0 ?? null, u0: u0 ?? null }),

  /** Teljes állapot és historikus adatok. */
  getState: () =>
    get('/state'),

  /** Rendszermodell paraméterei (A, B, korlátok). */
  getModelInfo: () =>
    get('/model-info'),
}
