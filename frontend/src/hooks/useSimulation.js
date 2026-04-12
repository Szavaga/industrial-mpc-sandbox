/**
 * useSimulation — WebSocket alapú szimulációs hook
 *
 * v4: Economic Mode + ESD + Alarm Journal + CSV Export
 *   - states[0] = CA [mol/L], states[1] = T [K]
 *   - control[0] = F [L/min], control[1] = Tc [K]
 *   - states = x̂ (KF-becslés), states_raw = y (zajos mérés)
 *   - economic_mode: profitmaximilizálás tracking helyett
 *   - esd: vészleállítás (min flow + max cooling)
 *   - alarmLog: riasztások listája (max 50)
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../lib/api'

const CHART_MAX_POINTS = 300
const WS_URL     = `ws://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8000/ws`
const API_BASE   = `http://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8000`

export function useSimulation() {
  const wsRef       = useRef(null)
  const mountedRef  = useRef(true)
  const runningRef  = useRef(false)

  // Refs for stale-closure prevention in handleMessage
  const reactorModeRef = useRef('SINGLE')
  // Ref for economic cfg (stale closure prevention in handleMessage)
  const economicCfgRef = useRef({
    economicMode: false,
    productPrice:  10.0,
    feedstockCost: 2.0,
    energyCost:    0.5,
  })

  const [wsStatus, setWsStatus]             = useState('connecting')
  const [running, setRunningState]          = useState(false)
  const [history, setHistory]               = useState([])
  const [currentState, setCurrentState]     = useState(null)
  const [violations, setViolations]         = useState({})
  const [mpcSuccess, setMpcSuccess]         = useState(true)
  const [predTraj, setPredTraj]             = useState(null)
  const [modelInfo, setModelInfo]           = useState(null)
  const [distActive, setDistActive]         = useState(false)
  const [approachingRunaway, setApproachingRunaway] = useState(false)
  const [isRunaway, setIsRunaway]           = useState(false)

  const [setpoints,  setSetpointsState]  = useState({ ca: 0.5, temp: 350.0 })
  const [setpoints2, setSetpoints2State] = useState({ ca: 0.453, temp: 329.2 })

  const [mpcCfg, setMpcCfg] = useState({
    prediction_horizon: 40,
    control_horizon:    10,
    Q00: 50.0, Q11: 0.2,
    R00: 0.001, R11: 0.01,
  })

  // ── Zaj & szenzor-hiba állapot ────────────────────────────────────────────
  const [noiseSigma,        setNoiseSigmaState]   = useState(0.0)
  const [sensorFaultActive, setSensorFaultActive] = useState(false)
  const [kalmanGain,        setKalmanGain]        = useState([0, 0])

  // ── Gazdasági üzemmód ─────────────────────────────────────────────────────
  const [economicCfg, setEconomicCfgState] = useState({
    economicMode: false,
    productPrice:  10.0,
    feedstockCost: 2.0,
    energyCost:    0.5,
  })

  // ── Controller type & IAE ─────────────────────────────────────────────────
  const [controllerType, setControllerType] = useState('NONLINEAR')
  const [iae, setIae] = useState({ ca: 0, temp: 0 })

  // ── Reactor mode ──────────────────────────────────────────────────────────
  const [reactorMode, setReactorMode] = useState('SINGLE')   // 'SINGLE' | 'SERIES'

  // ── ESD ───────────────────────────────────────────────────────────────────
  const [esdActive, setEsdActive] = useState(false)

  // ── SysID ─────────────────────────────────────────────────────────────────
  const [sysidActive,         setSysidActive]         = useState(false)
  const [sysidProgress,       setSysidProgress]       = useState({ step: 0, total: 0 })
  const [sysidResult,         setSysidResult]         = useState(null)
  const [sysidIdentified,     setSysidIdentified]     = useState(false)
  const [useIdentifiedModel,  setUseIdentifiedModel]  = useState(false)
  const [linearModelSource,   setLinearModelSource]   = useState('jacobian')

  // ── MHE estimator ─────────────────────────────────────────────────────────
  const [estimatorType,  setEstimatorType]  = useState('KF')
  const [mheSuccess,     setMheSuccess]     = useState(true)
  const [mheResiduals,   setMheResiduals]   = useState({ ca: 0, t: 0 })

  // ── Feedforward ───────────────────────────────────────────────────────────
  const [feedforwardEnabled, setFeedforwardEnabled] = useState(false)
  const [ffActive,           setFfActive]           = useState(false)
  // Split IAE metrics: rates per second in FF-ON vs FF-OFF mode
  const [iaeByMode, setIaeByMode] = useState({
    ff_on_ca:    0, ff_on_temp:  0,
    ff_off_ca:   0, ff_off_temp: 0,
    time_ff_on:  0, time_ff_off: 0,
  })

  // ── Riasztásnapló ─────────────────────────────────────────────────────────
  const [alarmLog, setAlarmLog] = useState([])

  // Keep refs in sync
  useEffect(() => { reactorModeRef.current = reactorMode }, [reactorMode])
  useEffect(() => { economicCfgRef.current = economicCfg }, [economicCfg])

  // ── WS küldés ────────────────────────────────────────────────────────────────
  const send = useCallback((cmd, data) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data !== undefined ? { cmd, data } : { cmd }))
    }
  }, [])

  // ── Bejövő üzenetek ──────────────────────────────────────────────────────
  const handleMessage = useCallback((msg) => {
    if (msg.type === 'state') {
      setCurrentState(msg)
      setMpcSuccess(msg.mpc_success ?? true)
      setViolations(msg.constraint_violations ?? {})
      setApproachingRunaway(msg.approaching_runaway ?? false)
      setIsRunaway(msg.is_runaway ?? false)
      setEsdActive(msg.esd_active ?? false)
      if (msg.controller_type)     setControllerType(msg.controller_type)
      if (msg.reactor_mode)        setReactorMode(msg.reactor_mode)
      if (msg.feedforward_enabled != null) setFeedforwardEnabled(msg.feedforward_enabled)
      if (msg.ff_active != null)           setFfActive(msg.ff_active)
      if (msg.estimator_type != null)      setEstimatorType(msg.estimator_type)
      if (msg.mhe_success != null)         setMheSuccess(msg.mhe_success)
      if (msg.mhe_residuals?.length === 2)
        setMheResiduals({ ca: msg.mhe_residuals[0], t: msg.mhe_residuals[1] })
      if (msg.sysid_active != null)        setSysidActive(msg.sysid_active)
      if (msg.sysid_identified != null)    setSysidIdentified(msg.sysid_identified)
      if (msg.use_identified_model != null) setUseIdentifiedModel(msg.use_identified_model)
      if (msg.linear_model_source != null) setLinearModelSource(msg.linear_model_source)
      if (msg.iae_ca   != null) setIae(prev => ({ ...prev, ca:   msg.iae_ca   }))
      if (msg.iae_temp != null) setIae(prev => ({ ...prev, temp: msg.iae_temp }))
      if (msg.iae_ff_on_ca != null) {
        setIaeByMode({
          ff_on_ca:    msg.iae_ff_on_ca    ?? 0,
          ff_on_temp:  msg.iae_ff_on_temp  ?? 0,
          ff_off_ca:   msg.iae_ff_off_ca   ?? 0,
          ff_off_temp: msg.iae_ff_off_temp ?? 0,
          time_ff_on:  msg.time_ff_on  ?? 0,
          time_ff_off: msg.time_ff_off ?? 0,
        })
      }

      if (msg.kalman_gain?.length === 2) setKalmanGain(msg.kalman_gain)
      if (msg.predicted_trajectory?.CA?.length)  setPredTraj(msg.predicted_trajectory)

      // Riasztásnapló frissítése
      if (msg.new_alarms?.length) {
        setAlarmLog(prev => [...prev, ...msg.new_alarms].slice(-50))
      }

      const cv        = msg.constraint_violations ?? {}
      const hasViol   = Object.keys(cv).length > 0
      const viol_ca   = !!(cv.x0_low || cv.x0_high)
      const viol_temp = !!(cv.x1_low || cv.x1_high)

      const raw = msg.states_raw ?? msg.states

      // Profit számítás
      const cfg = economicCfgRef.current
      const F  = msg.control[0]
      const isSeries = (reactorModeRef.current === 'SERIES')
      // In series use R2 output (CA2) for profit, else R1 (CA1)
      const CA_profit = isSeries ? (msg.states[2] ?? msg.states[0]) : msg.states[0]
      const Tc = msg.control[1]
      const flowRate   = F / 60.0
      const profitPerS = cfg.productPrice  * flowRate * CA_profit
                       - cfg.feedstockCost * flowRate * 1.0
                       - cfg.energyCost    * Math.pow(300.0 - Tc, 2) * 0.001
      const profit = profitPerS * 3600

      // R2 fields (only valid in SERIES, otherwise undefined)
      const hasR2 = isSeries && msg.states.length >= 4
      if (hasR2) {
        setSetpoints2State({ ca: +msg.setpoints[2].toFixed(3), temp: +msg.setpoints[3].toFixed(1) })
      }

      setHistory(prev => {
        const point = {
          time:     +msg.time.toFixed(1),
          ca:       +msg.states[0].toFixed(4),
          temp:     +msg.states[1].toFixed(2),
          ca_raw:   +raw[0].toFixed(4),
          temp_raw: +raw[1].toFixed(2),
          sp_ca:    +msg.setpoints[0].toFixed(3),
          sp_temp:  +msg.setpoints[1].toFixed(1),
          f_flow:   +msg.control[0].toFixed(2),
          tc:       +msg.control[1].toFixed(2),
          tc2:      msg.control[2] != null ? +msg.control[2].toFixed(2) : undefined,
          viol:     hasViol,
          viol_ca,
          viol_temp,
          profit:   +profit.toFixed(0),
          ff_active:    msg.ff_active    ?? false,
          sysid_active: msg.sysid_active ?? false,
          mhe_res_ca:   msg.mhe_residuals?.[0] ?? 0,
          mhe_res_t:    msg.mhe_residuals?.[1] ?? 0,
          // R2 fields
          ca2:      hasR2 ? +msg.states[2].toFixed(4)   : undefined,
          temp2:    hasR2 ? +msg.states[3].toFixed(2)   : undefined,
          sp_ca2:   hasR2 ? +msg.setpoints[2].toFixed(3): undefined,
          sp_temp2: hasR2 ? +msg.setpoints[3].toFixed(1): undefined,
        }
        const next = [...prev, point]
        return next.length > CHART_MAX_POINTS ? next.slice(-CHART_MAX_POINTS) : next
      })

    } else if (msg.type === 'reset_done') {
      setHistory([])
      setCurrentState(null)
      setViolations({})
      setPredTraj(null)
      setDistActive(false)
      setRunningState(false)
      runningRef.current = false
      setKalmanGain([0, 0])
      setApproachingRunaway(false)
      setIsRunaway(false)
      setEsdActive(false)
      setFfActive(false)
      setIae({ ca: 0, temp: 0 })
      setIaeByMode({ ff_on_ca: 0, ff_on_temp: 0, ff_off_ca: 0, ff_off_temp: 0, time_ff_on: 0, time_ff_off: 0 })
      setSysidActive(false)
      setSysidProgress({ step: 0, total: 0 })
      setSysidResult(null)
      if (msg.reactor_mode) setReactorMode(msg.reactor_mode)
    } else if (msg.type === 'sysid_progress') {
      setSysidActive(true)
      setSysidProgress({ step: msg.step, total: msg.total })
    } else if (msg.type === 'sysid_result') {
      setSysidActive(false)
      setSysidProgress({ step: 0, total: 0 })
      setSysidResult({
        fit_pct_ca:    msg.fit_pct_ca,
        fit_pct_t:     msg.fit_pct_t,
        na:            msg.na,
        nb:            msg.nb,
        step_response: msg.step_response,
      })
      setSysidIdentified(true)
    } else if (msg.type === 'sysid_error') {
      setSysidActive(false)
      setSysidProgress({ step: 0, total: 0 })
    }
  }, [])

  // ── WebSocket + auto-reconnect ──────────────────────────────────────────
  useEffect(() => {
    mountedRef.current = true
    let reconnectTimer = null

    const connect = () => {
      if (!mountedRef.current) return
      setWsStatus('connecting')
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return }
        setWsStatus('connected')
      }
      ws.onmessage = (event) => {
        try { handleMessage(JSON.parse(event.data)) }
        catch (e) { console.error('[ws] parse error:', e) }
      }
      ws.onclose = () => {
        if (!mountedRef.current) return
        setWsStatus('disconnected')
        reconnectTimer = setTimeout(connect, 2000)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    api.getModelInfo()
      .then(info => { if (mountedRef.current) setModelInfo(info) })
      .catch(console.warn)

    return () => {
      mountedRef.current = false
      clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [handleMessage])

  // ── Kontroll függvények ──────────────────────────────────────────────────
  const setRunning = useCallback((valOrFn) => {
    const next = typeof valOrFn === 'function' ? valOrFn(runningRef.current) : valOrFn
    runningRef.current = next
    setRunningState(next)
    send(next ? 'start' : 'stop')
  }, [send])

  const updateSetpoints = useCallback((ca, temp) => {
    const sp = { ca: +ca, temp: +temp }
    setSetpointsState(sp)
    send('setpoints', sp)
  }, [send])

  const updateSetpoints2 = useCallback((ca2, temp2) => {
    setSetpoints2State(prev => ({ ...prev, ca: +ca2, temp: +temp2 }))
    // send full setpoint command including R1 current values (server merges missing keys)
    send('setpoints', { ca2: +ca2, temp2: +temp2 })
  }, [send])

  const updateReactorMode = useCallback((mode) => {
    setReactorMode(mode)
    reactorModeRef.current = mode
    send('reactor_mode', { mode })
  }, [send])

  const updateMpcCfg = useCallback((cfg) => {
    setMpcCfg(cfg)
    send('config', {
      prediction_horizon: cfg.prediction_horizon,
      control_horizon:    cfg.control_horizon,
      Q00: cfg.Q00, Q11: cfg.Q11,
      R00: cfg.R00, R11: cfg.R11,
    })
  }, [send])

  const updateControllerType = useCallback((type) => {
    setControllerType(type)
    send('config', { controller_type: type })
  }, [send])

  const injectDisturbance = useCallback((dCA, dTemp, durationSteps) => {
    send('disturbance', { d_ca: dCA, d_temp: dTemp })
    setDistActive(true)
    setTimeout(() => {
      send('disturbance_clear')
      setDistActive(false)
    }, durationSteps * (mpcCfg._dt_ms ?? 1000))
  }, [send, mpcCfg])

  const updateNoise = useCallback((sigma) => {
    setNoiseSigmaState(sigma)
    send('noise', { sigma })
  }, [send])

  const injectSensorFault = useCallback((biasLevel, biasTemp) => {
    setSensorFaultActive(true)
    send('sensor_fault', { bias_level: biasLevel, bias_temp: biasTemp })
  }, [send])

  const clearSensorFault = useCallback(() => {
    setSensorFaultActive(false)
    send('sensor_fault_clear')
  }, [send])

  // ── Gazdasági konfiguráció ─────────────────────────────────────────────
  const updateEconomicCfg = useCallback((cfg) => {
    setEconomicCfgState(cfg)
    send('economic_config', {
      economic_mode:  cfg.economicMode,
      product_price:  cfg.productPrice,
      feedstock_cost: cfg.feedstockCost,
      energy_cost:    cfg.energyCost,
    })
  }, [send])

  // ── Feedforward ──────────────────────────────────────────────────────────
  const updateFeedforward = useCallback((enabled) => {
    setFeedforwardEnabled(enabled)
    send('feedforward', { enabled })
  }, [send])

  // ── MHE estimator ────────────────────────────────────────────────────
  const updateEstimator = useCallback((type, config = {}) => {
    setEstimatorType(type)
    send('estimator', { type, ...config })
  }, [send])

  // ── SysID ────────────────────────────────────────────────────────────
  const startSysId = useCallback((config) => {
    send('sysid_start', config)
  }, [send])

  const updateUseIdentified = useCallback((useIdentified) => {
    setUseIdentifiedModel(useIdentified)
    send('sysid_use_model', { use_identified: useIdentified })
  }, [send])

  // ── ESD ───────────────────────────────────────────────────────────────
  const esd = useCallback(() => {
    setEsdActive(true)
    send('esd')
  }, [send])

  const esdClear = useCallback(() => {
    setEsdActive(false)
    send('esd_clear')
  }, [send])

  // ── CSV export ────────────────────────────────────────────────────────
  const downloadCsv = useCallback(() => {
    window.open(`${API_BASE}/api/export`, '_blank')
  }, [])

  const reset = useCallback(() => {
    runningRef.current = false
    setRunningState(false)
    setDistActive(false)
    setSensorFaultActive(false)
    setEsdActive(false)
    send('reset')
  }, [send])

  return {
    wsStatus,
    running, setRunning,
    history, currentState,
    violations, mpcSuccess, predTraj,
    modelInfo,
    setpoints, updateSetpoints,
    mpcCfg, updateMpcCfg,
    distActive, injectDisturbance,
    noiseSigma, updateNoise,
    sensorFaultActive, injectSensorFault, clearSensorFault,
    kalmanGain,
    approachingRunaway, isRunaway,
    economicCfg, updateEconomicCfg,
    esdActive, esd, esdClear,
    alarmLog,
    downloadCsv,
    controllerType, updateControllerType,
    iae,
    reactorMode, updateReactorMode,
    setpoints2, updateSetpoints2,
    feedforwardEnabled, updateFeedforward, ffActive, iaeByMode,
    estimatorType, mheSuccess, mheResiduals, updateEstimator,
    sysidActive, sysidProgress, sysidResult, sysidIdentified,
    useIdentifiedModel, linearModelSource,
    startSysId, updateUseIdentified,
    reset,
  }
}
