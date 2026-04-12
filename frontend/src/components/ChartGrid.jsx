import { useMemo, useRef } from 'react'
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ReferenceArea,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis,
} from 'recharts'

// ─── Profit számítás ─────────────────────────────────────────────────────────
function computeProfit(state, econCfg) {
  if (!state || !econCfg) return null
  const F  = state.control[0]   // L/min
  const CA = state.states[0]    // mol/L (KF)
  const Tc = state.control[1]   // K
  const flowRate  = F / 60.0
  const product   = econCfg.productPrice  * flowRate * CA
  const feed      = econCfg.feedstockCost * flowRate * 1.0   // Caf = 1.0
  const cooling   = econCfg.energyCost    * Math.pow(300.0 - Tc, 2) * 0.001
  return (product - feed - cooling) * 3600  // $/hr
}

// ─── Segédfüggvények ──────────────────────────────────────────────────────────

function getViolationRanges(data, violKey) {
  const ranges = []
  let start = null
  for (const p of data) {
    if (p[violKey] && start === null) start = p.time
    if (!p[violKey] && start !== null) { ranges.push([start, p.time]); start = null }
  }
  if (start !== null && data.length > 0) ranges.push([start, data.at(-1).time])
  return ranges
}

function getSpChangeEvents(data, spKey) {
  const events = []
  for (let i = 1; i < data.length; i++) {
    if (Math.abs(data[i][spKey] - data[i - 1][spKey]) > 0.001) {
      events.push({ time: data[i].time, val: data[i][spKey] })
    }
  }
  return events
}

/** Pillantnyi tracking cost: J = Q_CA*(CA−spCA)² + Q_T*(T−spT)² */
function computeCost(state, cfg) {
  if (!state) return null
  const eCA = state.states[0] - state.setpoints[0]
  const eT  = state.states[1] - state.setpoints[1]
  return cfg.Q00 * eCA * eCA + cfg.Q11 * eT * eT
}

/** history + predikció összefűzése */
function buildChartData(history, predTraj, pvKey, predKey, spKey, rawKey) {
  const hist = history.map(h => ({
    time: h.time,
    pv:   h[pvKey],
    sp:   h[spKey],
    raw:  rawKey ? h[rawKey] : undefined,
  }))
  if (!predTraj?.[predKey]?.length) return hist
  const last     = hist.at(-1)
  const baseTime = last?.time ?? 0
  const sp       = last?.sp ?? 0
  const pred     = predTraj.time.map((relT, i) => ({
    time: +(baseTime + relT).toFixed(2),
    pv:   i === 0 ? predTraj[predKey][i] : undefined,
    sp,
    pred: predTraj[predKey][i],
  }))
  return [...hist, ...pred.slice(1)]
}

// ─── Tooltip ──────────────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-900 border border-slate-600 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-400 mb-1.5">t = {label?.toFixed?.(1) ?? label} s</p>
      {payload.map(p => p.value != null && (
        <p key={p.dataKey} style={{ color: p.color }} className="tabular-nums">
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value} {unit}
        </p>
      ))}
    </div>
  )
}

// ─── Folyamat-változó grafikon ─────────────────────────────────────────────────
function ProcessChart({
  data, title, unit, color, predColor, spColor = '#4ade80',
  yMin, yMax, cMin, cMax, violKey, spKey, hasNoise,
  dangerLine, runawayLine,
}) {
  const visibleData = useMemo(() => data.slice(-120), [data])

  const violRanges = useMemo(
    () => getViolationRanges(visibleData, violKey), [visibleData, violKey]
  )
  const spEvents = useMemo(
    () => getSpChangeEvents(visibleData, spKey), [visibleData, spKey]
  )

  const predClr = predColor ?? `${color}77`
  const yDomain = [d => Math.min(d, yMin ?? Infinity) - 0.01,
                   d => Math.max(d, yMax ?? -Infinity) + 0.01]

  return (
    <div className="card flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">{title}</span>
        <span className="text-xs text-slate-500">[{unit}]</span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={visibleData} margin={{ top: 6, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="time" type="number" domain={['auto', 'auto']}
            tickFormatter={v => v.toFixed(0)}
            tick={{ fill: '#64748b', fontSize: 10 }}
            label={{ value: 't (s)', position: 'insideBottomRight', offset: 0, fill: '#475569', fontSize: 10 }} />
          <YAxis domain={yDomain} tick={{ fill: '#64748b', fontSize: 10 }} width={48} />
          <Tooltip content={<ChartTooltip unit={unit} />} />
          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} iconType="circle" iconSize={8} />

          {/* Fizikai korlátok */}
          {cMin !== undefined && (
            <ReferenceLine y={cMin} stroke="#ef4444" strokeDasharray="4 2"
              label={{ value: 'min', fill: '#ef4444', fontSize: 9 }} />
          )}
          {cMax !== undefined && (
            <ReferenceLine y={cMax} stroke="#ef4444" strokeDasharray="4 2"
              label={{ value: 'max', fill: '#ef4444', fontSize: 9 }} />
          )}

          {/* Veszély / runaway vonalak (T charton) */}
          {dangerLine !== undefined && (
            <ReferenceLine y={dangerLine} stroke="#f59e0b" strokeDasharray="6 3" strokeWidth={1.5}
              label={{ value: '⚠ Danger', fill: '#f59e0b', fontSize: 9, position: 'insideTopLeft' }} />
          )}
          {runawayLine !== undefined && (
            <ReferenceLine y={runawayLine} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={2}
              label={{ value: '🔥 Runaway', fill: '#ef4444', fontSize: 9, position: 'insideTopLeft' }} />
          )}

          {/* Kényszersértés sávok */}
          {violRanges.map(([x1, x2], i) => (
            <ReferenceArea key={`viol-${i}`} x1={x1} x2={x2}
              fill="#ef444418" stroke="#ef444450" strokeWidth={1} />
          ))}

          {/* Setpoint-változás vonalak */}
          {spEvents.map((e, i) => (
            <ReferenceLine key={`sp-${i}`} x={e.time}
              stroke="#4ade80" strokeDasharray="5 3" strokeWidth={1.5}
              label={{ value: `→${e.val.toFixed(3)}`, position: 'insideTopRight',
                       fill: '#4ade80', fontSize: 9, fontWeight: 'bold' }} />
          ))}

          {/* Setpoint vonal */}
          <Line dataKey="sp" name="SP" stroke={spColor}
            strokeDasharray="6 3" strokeWidth={1.5}
            dot={false} activeDot={false} connectNulls />

          {/* Predikciós vonal */}
          <Line dataKey="pred" name="Predicted (NMPC)" stroke={predClr}
            strokeDasharray="3 3" strokeWidth={1.5}
            dot={false} activeDot={false} />

          {/* Nyers zajos jel */}
          {hasNoise && (
            <Line dataKey="raw" name="Raw (Noisy)" stroke={`${color}55`}
              strokeDasharray="2 3" strokeWidth={1}
              dot={false} activeDot={false} />
          )}

          {/* PV / Filtered (KF) */}
          <Line dataKey="pv" name={hasNoise ? 'Filtered (KF)' : 'PV'} stroke={color} strokeWidth={2}
            dot={false} activeDot={{ r: 4, fill: color }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Dual-reactor process chart (R1 + optional R2) ───────────────────────────
function DualProcessChart({
  data, history, title, unit, color, color2, predColor, spColor = '#4ade80',
  yMin, yMax, cMin, cMax, violKey, spKey, hasNoise,
  dangerLine, runawayLine, showR2, pvKey2, spKey2,
}) {
  const visibleData = useMemo(() => data.slice(-120), [data])
  const r2data = useMemo(() => {
    if (!showR2) return []
    return history.slice(-120).map(h => ({ time: h.time, r2: h[pvKey2], sp2: h[spKey2] }))
  }, [history, showR2, pvKey2, spKey2])

  const violRanges = useMemo(() => getViolationRanges(visibleData, violKey), [visibleData, violKey])
  const spEvents   = useMemo(() => getSpChangeEvents(visibleData, spKey),    [visibleData, spKey])
  const predClr    = predColor ?? `${color}77`
  const yDomain    = [d => Math.min(d, yMin ?? Infinity) - 0.01,
                      d => Math.max(d, yMax ?? -Infinity) + 0.01]

  // Merge R2 into visibleData by time index
  const merged = useMemo(() => {
    if (!showR2 || !r2data.length) return visibleData
    const r2map = {}
    r2data.forEach(p => { r2map[p.time] = p })
    return visibleData.map(p => ({ ...p, r2: r2map[p.time]?.r2, sp2: r2map[p.time]?.sp2 }))
  }, [visibleData, r2data, showR2])

  return (
    <div className="card flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">{title}</span>
        <span className="text-xs text-slate-500">[{unit}]</span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={merged} margin={{ top: 6, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="time" type="number" domain={['auto', 'auto']}
            tickFormatter={v => v.toFixed(0)} tick={{ fill: '#64748b', fontSize: 10 }}
            label={{ value: 't (s)', position: 'insideBottomRight', offset: 0, fill: '#475569', fontSize: 10 }} />
          <YAxis domain={yDomain} tick={{ fill: '#64748b', fontSize: 10 }} width={48} />
          <Tooltip content={<ChartTooltip unit={unit} />} />
          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} iconType="circle" iconSize={8} />

          {cMin !== undefined && <ReferenceLine y={cMin} stroke="#ef4444" strokeDasharray="4 2"
            label={{ value: 'min', fill: '#ef4444', fontSize: 9 }} />}
          {cMax !== undefined && <ReferenceLine y={cMax} stroke="#ef4444" strokeDasharray="4 2"
            label={{ value: 'max', fill: '#ef4444', fontSize: 9 }} />}
          {dangerLine !== undefined && <ReferenceLine y={dangerLine} stroke="#f59e0b"
            strokeDasharray="6 3" strokeWidth={1.5}
            label={{ value: '⚠ Danger', fill: '#f59e0b', fontSize: 9, position: 'insideTopLeft' }} />}
          {runawayLine !== undefined && <ReferenceLine y={runawayLine} stroke="#ef4444"
            strokeDasharray="4 4" strokeWidth={2}
            label={{ value: '🔥 Runaway', fill: '#ef4444', fontSize: 9, position: 'insideTopLeft' }} />}

          {violRanges.map(([x1, x2], i) => (
            <ReferenceArea key={`viol-${i}`} x1={x1} x2={x2}
              fill="#ef444418" stroke="#ef444450" strokeWidth={1} />
          ))}
          {spEvents.map((e, i) => (
            <ReferenceLine key={`sp-${i}`} x={e.time} stroke="#4ade80"
              strokeDasharray="5 3" strokeWidth={1.5}
              label={{ value: `→${e.val.toFixed(3)}`, position: 'insideTopRight',
                       fill: '#4ade80', fontSize: 9, fontWeight: 'bold' }} />
          ))}

          <Line dataKey="sp" name="SP (R1)" stroke={spColor}
            strokeDasharray="6 3" strokeWidth={1.5} dot={false} activeDot={false} connectNulls />
          <Line dataKey="pred" name="Predicted" stroke={predClr}
            strokeDasharray="3 3" strokeWidth={1.5} dot={false} activeDot={false} />
          {hasNoise && <Line dataKey="raw" name="Raw (Noisy)" stroke={`${color}55`}
            strokeDasharray="2 3" strokeWidth={1} dot={false} activeDot={false} />}
          <Line dataKey="pv" name={hasNoise ? 'R1 (KF)' : 'R1'} stroke={color}
            strokeWidth={2} dot={false} activeDot={{ r: 4, fill: color }} />

          {showR2 && <>
            <Line dataKey="sp2" name="SP (R2)" stroke={`${color2}88`}
              strokeDasharray="4 2" strokeWidth={1} dot={false} activeDot={false} connectNulls />
            <Line dataKey="r2" name="R2" stroke={color2}
              strokeDasharray="5 2" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: color2 }} />
          </>}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Beavatkozójel grafikon ───────────────────────────────────────────────────
function ControlChart({ data, dataKey, title, unit, color, cMin, cMax, ffRanges, sysidRanges }) {
  const visibleData = useMemo(() => data.slice(-120), [data])
  const hasFf    = ffRanges?.length > 0
  const hasSysid = sysidRanges?.length > 0
  return (
    <div className="card flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">{title}</span>
          {hasFf && (
            <span className="text-[10px] px-1 py-0.5 rounded font-bold
                             bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
              ⚡FF
            </span>
          )}
          {hasSysid && (
            <span className="text-[10px] px-1 py-0.5 rounded font-bold
                             bg-amber-500/15 text-amber-400 border border-amber-500/30">
              ⚡ID
            </span>
          )}
        </div>
        <span className="text-xs text-slate-500">[{unit}]</span>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={visibleData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="time" type="number" domain={['auto', 'auto']}
            tickFormatter={v => v.toFixed(0)} tick={{ fill: '#64748b', fontSize: 10 }} />
          <YAxis domain={[cMin - 1, cMax + 1]} tick={{ fill: '#64748b', fontSize: 10 }} width={48} />
          <Tooltip content={<ChartTooltip unit={unit} />} />
          {/* Cyan bands = MPC acting with FF knowledge (proactive compensation) */}
          {ffRanges?.map(([x1, x2], i) => (
            <ReferenceArea key={`ff-${i}`} x1={x1} x2={x2}
              fill="#06b6d425" stroke="#06b6d450" strokeWidth={1} />
          ))}
          {/* Amber bands = PRBS ID test excitation active */}
          {sysidRanges?.map(([x1, x2], i) => (
            <ReferenceArea key={`id-${i}`} x1={x1} x2={x2}
              fill="#f59e0b18" stroke="#f59e0b40" strokeWidth={1} />
          ))}
          <ReferenceLine y={cMin} stroke="#ef4444" strokeDasharray="4 2"
            label={{ value: `min`, fill: '#ef4444', fontSize: 9 }} />
          <ReferenceLine y={cMax} stroke="#ef4444" strokeDasharray="4 2"
            label={{ value: `max`, fill: '#ef4444', fontSize: 9 }} />
          <Area dataKey={dataKey} name="MV" stroke={color} fill={`${color}22`}
            strokeWidth={2} dot={false} activeDot={{ r: 4, fill: color }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Fázissík — CA vs T ───────────────────────────────────────────────────────
function PhasePlaneChart({ history, setpoints, modelInfo }) {
  const allTraj    = useMemo(() => history.slice(-150).map(p => ({ x: p.ca, y: p.temp })), [history])
  const recentTraj = useMemo(() => allTraj.slice(-30), [allTraj])
  const currentPos = allTraj.at(-1)

  const spData  = setpoints ? [{ x: setpoints.ca, y: setpoints.temp }] : []
  const ssData  = modelInfo ? [{ x: modelInfo.states[0].ss, y: modelInfo.states[1].ss }] : []

  const xDomain = [modelInfo?.states[0]?.min ?? 0.02, modelInfo?.states[0]?.max ?? 0.98]
  const yDomain = [modelInfo?.states[1]?.min ?? 300,  modelInfo?.states[1]?.max ?? 430]

  const T_danger  = modelInfo?.T_danger  ?? 400
  const T_runaway = modelInfo?.T_runaway ?? 420

  const PhaseTip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const p = payload[0].payload
    return (
      <div className="bg-slate-900 border border-slate-600 rounded-lg p-2 text-xs shadow-xl">
        <p style={{ color: '#60a5fa' }}>CA = {p.x?.toFixed(4)} mol/L</p>
        <p style={{ color: '#fb923c' }}>T  = {p.y?.toFixed(1)} K</p>
      </div>
    )
  }

  return (
    <div className="card flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
          Phase Plane — CA vs T (Nonlinear Trajectory)
        </span>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span><span className="inline-block w-3 h-0.5 bg-blue-400 mr-1" />Trajectory</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-full bg-orange-400 mr-1" />Current</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-400 mr-1" />SP</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-full bg-slate-500 mr-1" />SS</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />

          <XAxis type="number" dataKey="x" name="CA" domain={xDomain}
            tick={{ fill: '#64748b', fontSize: 10 }}
            label={{ value: 'CA (mol/L)', position: 'insideBottom', offset: -18,
                     fill: '#475569', fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="T" domain={yDomain}
            tick={{ fill: '#64748b', fontSize: 10 }} width={50}
            label={{ value: 'T (K)', angle: -90, position: 'insideLeft',
                     offset: 12, fill: '#475569', fontSize: 11 }} />
          <ZAxis range={[20, 20]} />
          <Tooltip content={<PhaseTip />} />

          {/* Érvényes tartomány */}
          <ReferenceArea
            x1={xDomain[0]} x2={xDomain[1]} y1={yDomain[0]} y2={yDomain[1]}
            fill="#3b82f608" stroke="#334155" strokeDasharray="6 3" strokeWidth={1} />

          {/* Veszélyzóna (T > T_danger): sárga sáv */}
          <ReferenceArea
            x1={xDomain[0]} x2={xDomain[1]} y1={T_danger} y2={yDomain[1]}
            fill="#f59e0b10" stroke="#f59e0b30" strokeWidth={1} />

          {/* Runaway zóna (T > T_runaway): piros sáv */}
          <ReferenceArea
            x1={xDomain[0]} x2={xDomain[1]} y1={T_runaway} y2={yDomain[1]}
            fill="#ef444420" stroke="#ef444450" strokeWidth={1} />

          {/* Teljes trajektória (halvány) */}
          <Scatter name="History" data={allTraj} fill="#60a5fa30"
            line={{ stroke: '#60a5fa40', strokeWidth: 1 }}
            lineType="joint"
            shape={({ cx, cy }) => <circle cx={cx} cy={cy} r={1.5} fill="#60a5fa33" />} />

          {/* Utolsó 30 pont (erős kék) */}
          <Scatter name="Recent" data={recentTraj} fill="#60a5fa"
            line={{ stroke: '#60a5fa', strokeWidth: 2 }}
            lineType="joint"
            shape={({ cx, cy }) => <circle cx={cx} cy={cy} r={2} fill="#60a5fa88" />} />

          {/* Stacionárius pont (szürke X) */}
          <Scatter name="SS" data={ssData} fill="#64748b"
            shape={({ cx, cy }) => (
              <g>
                <line x1={cx-7} y1={cy} x2={cx+7} y2={cy} stroke="#64748b" strokeWidth={2} />
                <line x1={cx} y1={cy-7} x2={cx} y2={cy+7} stroke="#64748b" strokeWidth={2} />
              </g>
            )} />

          {/* Setpoint (zöld célkereszt) */}
          <Scatter name="SP" data={spData} fill="#4ade80"
            shape={({ cx, cy }) => (
              <g>
                <circle cx={cx} cy={cy} r={11} fill="none" stroke="#4ade80" strokeWidth={1.5} strokeOpacity={0.5} />
                <circle cx={cx} cy={cy} r={5}  fill="none" stroke="#4ade80" strokeWidth={1.5} />
                <circle cx={cx} cy={cy} r={2}  fill="#4ade80" />
              </g>
            )} />

          {/* Aktuális pozíció */}
          {currentPos && (
            <Scatter name="Current" data={[currentPos]} fill="#f97316"
              shape={({ cx, cy }) => (
                <g>
                  <circle cx={cx} cy={cy} r={14} fill="#f9731615" stroke="#f9731640" strokeWidth={1} />
                  <circle cx={cx} cy={cy} r={6}  fill="#f97316" stroke="#fff" strokeWidth={1.5} />
                </g>
              )} />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── KPI kártya ────────────────────────────────────────────────────────────────
function KpiCard({ label, value, unit, color, alarm, subtitle }) {
  return (
    <div className={`card flex flex-col gap-1 border transition-colors duration-300
      ${alarm ? 'border-red-500 bg-red-950/40' : 'border-slate-700'}`}>
      <span className="label mb-0">{label}</span>
      <span className={`value-display ${color}`}>
        {value ?? '—'}
        <span className="text-sm font-normal text-slate-400 ml-1">{unit}</span>
      </span>
      {subtitle && <span className="text-xs text-slate-600">{subtitle}</span>}
    </div>
  )
}

function CostKpi({ state, mpcCfg }) {
  const J = state ? computeCost(state, mpcCfg) : null
  const color = J == null ? 'text-slate-500'
    : J < 0.5  ? 'text-emerald-400'
    : J < 5.0  ? 'text-amber-400'
    :              'text-red-400'
  const grade = J == null ? '—'
    : J < 0.5  ? 'Excellent'
    : J < 5.0  ? 'Converging'
    :              'Large error'
  return (
    <KpiCard label="Cost J(k)" value={J != null ? J.toFixed(3) : null}
      unit="" color={color} subtitle={grade} />
  )
}

// ─── Profit KPI ───────────────────────────────────────────────────────────────
function ProfitKpi({ profit, economicMode }) {
  const color = profit == null ? 'text-slate-500'
    : profit > 10000  ? 'text-emerald-400'
    : profit > 0      ? 'text-amber-400'
    :                   'text-red-400'
  const label = economicMode ? 'Profit (Eco)' : 'Profit ($/hr)'
  const display = profit != null ? `${(profit / 1000).toFixed(1)}k` : '—'
  return (
    <div className={`card flex flex-col gap-1 border transition-colors duration-300 ${
      economicMode ? 'border-emerald-600/40 bg-emerald-950/20' : 'border-slate-700'
    }`}>
      <span className="label mb-0">{label}</span>
      <span className={`value-display ${color}`}>
        {display}
        <span className="text-sm font-normal text-slate-400 ml-1">$/hr</span>
      </span>
      {economicMode && (
        <span className="text-xs text-emerald-500">Economic NMPC aktív</span>
      )}
    </div>
  )
}

// ─── Alarm Journal ────────────────────────────────────────────────────────────
function AlarmJournal({ alarmLog }) {
  const endRef = useRef(null)
  const recent = alarmLog.slice(-10).reverse()

  return (
    <div className="card p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold uppercase tracking-widest text-slate-300">
          Alarm Journal
        </span>
        <span className="text-xs text-slate-600">{alarmLog.length} total</span>
      </div>
      {recent.length === 0 ? (
        <p className="text-xs text-slate-600 text-center py-2">Nincs riasztás</p>
      ) : (
        <div className="flex flex-col gap-1 max-h-48 overflow-y-auto" ref={endRef}>
          {recent.map((alarm, i) => (
            <div key={i} className={`flex items-center gap-2 text-xs rounded px-2 py-1.5 ${
              alarm.level === 'critical'
                ? 'bg-red-950/60 text-red-300 border border-red-800/60'
                : alarm.level === 'warning'
                ? 'bg-amber-950/50 text-amber-300 border border-amber-800/60'
                : 'bg-slate-800/60 text-slate-400 border border-slate-700/60'
            }`}>
              <span className="font-bold tabular-nums text-slate-500 w-14 shrink-0">
                {alarm.time?.toFixed ? alarm.time.toFixed(0) : alarm.time}s
              </span>
              <span className={`font-bold w-28 shrink-0 ${
                alarm.level === 'critical' ? 'text-red-400' :
                alarm.level === 'warning'  ? 'text-amber-400' : 'text-slate-500'
              }`}>{alarm.type}</span>
              <span className="truncate flex-1">{alarm.description}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Tank vizualizáció ────────────────────────────────────────────────────────
function TankViz({ currentState, reactorMode }) {
  const isSeries = reactorMode === 'SERIES'
  const ca1 = currentState?.states[0]
  const t1  = currentState?.states[1]
  const ca2 = isSeries ? currentState?.states[2] : null
  const t2  = isSeries ? currentState?.states[3] : null
  const f   = currentState?.control[0]

  const TankBlock = ({ label, ca, temp, color, small }) => (
    <div className={`flex flex-col items-center gap-1 ${small ? 'w-20' : 'w-24'}`}>
      <span className="text-xs font-bold text-slate-400">{label}</span>
      <div className={`relative rounded-lg border-2 ${color} flex flex-col items-center justify-center
                       ${small ? 'w-16 h-14' : 'w-20 h-16'} bg-slate-900/60`}>
        <span className="text-xs tabular-nums font-bold text-blue-300">
          {ca != null ? ca.toFixed(3) : '—'}
        </span>
        <span className="text-[10px] text-slate-500">mol/L</span>
        <span className="text-xs tabular-nums font-bold text-orange-300 mt-0.5">
          {temp != null ? temp.toFixed(1) : '—'} K
        </span>
      </div>
    </div>
  )

  const Arrow = ({ label }) => (
    <div className="flex flex-col items-center justify-center gap-0.5 px-1">
      <span className="text-[10px] text-teal-500">{label}</span>
      <span className="text-teal-400 text-lg leading-none">→</span>
    </div>
  )

  return (
    <div className="card p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold uppercase tracking-widest text-slate-300">
          Process Flow Diagram
        </span>
        <span className="text-xs text-slate-500">{isSeries ? 'SERIES' : 'SINGLE'}</span>
      </div>
      <div className="flex items-center justify-center gap-1 py-1">
        {/* Feed */}
        <div className="flex flex-col items-center gap-0.5">
          <span className="text-[10px] text-slate-600">CAf=1.0</span>
          <span className="text-teal-400 text-sm">→</span>
        </div>

        <TankBlock label="R1" ca={ca1} temp={t1} color="border-blue-500/60" />

        {isSeries ? (
          <>
            <Arrow label={f != null ? `${f.toFixed(0)}L/m` : ''} />
            <TankBlock label="R2" ca={ca2} temp={t2} color="border-teal-500/60" small />
          </>
        ) : null}

        {/* Product */}
        <div className="flex flex-col items-center gap-0.5">
          <span className="text-[10px] text-slate-600">Product</span>
          <span className="text-emerald-400 text-sm">→</span>
        </div>
      </div>
      {isSeries && (
        <div className="mt-1 text-center text-xs text-slate-600">
          Τ₁ = V₁/F·60 = {f ? (100/f*60).toFixed(0) : '—'}s &nbsp;|&nbsp;
          Τ₂ = V₂/F·60 = {f ? (50/f*60).toFixed(0) : '—'}s
          <span className="ml-2 text-teal-700">(propagation delay)</span>
        </div>
      )}
    </div>
  )
}

// ─── Runaway figyelmeztetés sáv ───────────────────────────────────────────────
function RunawayAlert({ approaching, runaway }) {
  if (!approaching && !runaway) return null
  return (
    <div className={`rounded-lg border p-3 flex items-center gap-3
      ${runaway
        ? 'bg-red-950/60 border-red-500 text-red-300'
        : 'bg-amber-950/60 border-amber-500 text-amber-300'
      }`}>
      <span className="text-xl">{runaway ? '🔥' : '⚠️'}</span>
      <div>
        <p className="text-sm font-bold">
          {runaway ? 'RUNAWAY — Thermal Runaway!' : 'DANGER — Approaching Runaway Zone'}
        </p>
        <p className="text-xs opacity-75">
          {runaway
            ? 'T > 420 K — a reakció szabadfutásba ment. Indíts reset-et!'
            : 'T > 400 K — hűtési kapacitás határán. Az NMPC kompenzál.'}
        </p>
      </div>
    </div>
  )
}

// ─── MHE maradék hiba chart ───────────────────────────────────────────────────
function MheResidualsChart({ history }) {
  const data = useMemo(
    () => history.slice(-120).map(h => ({
      time: h.time,
      ca:   h.mhe_res_ca ?? 0,
      t:    h.mhe_res_t  ?? 0,
    })),
    [history],
  )
  return (
    <div className="card p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
          MHE — Maradék hiba  |y − ŷ|
        </span>
        <div className="flex items-center gap-3 text-[10px] text-slate-500">
          <span><span className="inline-block w-3 h-0.5 bg-blue-400 mr-1" />CA [mol/L]</span>
          <span><span className="inline-block w-3 h-0.5 bg-orange-400 mr-1" />T [K]</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="time" type="number" domain={['auto', 'auto']}
            tickFormatter={v => v.toFixed(0)} tick={{ fill: '#64748b', fontSize: 10 }} />
          <YAxis tick={{ fill: '#64748b', fontSize: 10 }} width={52} />
          <Tooltip content={<ChartTooltip unit="" />} />
          <Line dataKey="ca" name="|CA err|" stroke="#60a5fa" strokeWidth={1.5}
            dot={false} activeDot={{ r: 3 }} />
          <Line dataKey="t"  name="|T err|"  stroke="#fb923c" strokeWidth={1.5}
            dot={false} activeDot={{ r: 3 }} />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-[10px] text-slate-600 mt-1 text-center">
        Nagy csúcs = szenzor-outlier. L1 módban az MHE elnyomja — a KF "elvándorol".
      </p>
    </div>
  )
}

// ─── SysID lépésválasz-összehasonlítás (2×2 grid) ────────────────────────────
function StepLine({ data, dataKey, name, color, strokeDasharray }) {
  return (
    <Line
      data={data} dataKey={dataKey} name={name}
      stroke={color} strokeWidth={1.5}
      strokeDasharray={strokeDasharray}
      dot={false} activeDot={false} connectNulls
    />
  )
}

function StepSubChart({ title, arx, jacobian, true: trueData, dataKey, unit }) {
  const merged = useMemo(() => {
    if (!arx?.length) return []
    return arx.map((p, i) => ({
      time:     p.time,
      arx:      p[dataKey],
      jacobian: jacobian?.[i]?.[dataKey],
      true:     trueData?.[i]?.[dataKey],
    }))
  }, [arx, jacobian, trueData, dataKey])

  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider text-center">
        {title}
      </span>
      <ResponsiveContainer width="100%" height={140}>
        <ComposedChart data={merged} margin={{ top: 4, right: 4, bottom: 0, left: -16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="time" type="number" domain={['auto', 'auto']}
            tickFormatter={v => v.toFixed(0)} tick={{ fill: '#64748b', fontSize: 9 }} />
          <YAxis tick={{ fill: '#64748b', fontSize: 9 }} width={44} />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null
              return (
                <div className="bg-slate-900 border border-slate-600 rounded p-2 text-[10px] shadow-xl">
                  <p className="text-slate-400 mb-1">t = {label?.toFixed?.(1)}s</p>
                  {payload.map(p => p.value != null && (
                    <p key={p.dataKey} style={{ color: p.color }} className="tabular-nums">
                      {p.name}: {p.value.toFixed(5)} {unit}
                    </p>
                  ))}
                </div>
              )
            }}
          />
          <StepLine data={merged} dataKey="true"     name="True"     color="#60a5fa" />
          <StepLine data={merged} dataKey="arx"      name="ARX-ID"   color="#34d399" strokeDasharray="5 3" />
          <StepLine data={merged} dataKey="jacobian" name="Jacobian" color="#a78bfa" strokeDasharray="2 4" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

function SysIdStepResponseChart({ sysidResult }) {
  const { step_response } = sysidResult
  const arx      = step_response?.arx      ?? []
  const jacobian = step_response?.jacobian ?? []
  const trueResp = step_response?.true     ?? []

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
          SysID — Step Response Comparison
        </span>
        <div className="flex items-center gap-3 text-[10px] text-slate-500">
          <span><span className="inline-block w-4 h-0.5 bg-blue-400 mr-1" />True</span>
          <span className="inline-block w-4 border-t-2 border-dashed border-emerald-400 mr-1" />
          <span className="text-emerald-400">ARX-ID</span>
          <span className="inline-block w-4 border-t-2 border-dotted border-violet-400 mr-1" />
          <span className="text-violet-400">Jacobian</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <StepSubChart title="dCA / dF"  dataKey="dCA_dF"  unit="mol/L"
          arx={arx} jacobian={jacobian} true={trueResp} />
        <StepSubChart title="dCA / dTc" dataKey="dCA_dTc" unit="mol/L"
          arx={arx} jacobian={jacobian} true={trueResp} />
        <StepSubChart title="dT / dF"   dataKey="dT_dF"   unit="K"
          arx={arx} jacobian={jacobian} true={trueResp} />
        <StepSubChart title="dT / dTc"  dataKey="dT_dTc"  unit="K"
          arx={arx} jacobian={jacobian} true={trueResp} />
      </div>
      <div className="mt-2 text-[10px] text-slate-600 text-center">
        Egységugrás (ΔF=1 L/min, ΔTc=1 K) | ARX(na={sysidResult.na}, nb={sysidResult.nb})
        | CA fit: {sysidResult.fit_pct_ca?.toFixed(1)}% | T fit: {sysidResult.fit_pct_t?.toFixed(1)}%
      </div>
    </div>
  )
}

// ─── Fő export ────────────────────────────────────────────────────────────────
export function ChartGrid({ sim }) {
  const {
    history, predTraj, currentState, violations, modelInfo,
    setpoints, mpcCfg, noiseSigma, approachingRunaway, isRunaway,
    economicCfg, alarmLog, esdActive, controllerType, iae, reactorMode,
    ffActive, feedforwardEnabled,
    sysidActive, sysidResult,
    estimatorType, mheSuccess,
  } = sim

  const isLinear  = controllerType === 'LINEAR'
  const isSeries  = reactorMode === 'SERIES'

  // Compute time-ranges where FF was active (for cyan bands on control charts)
  const ffRanges = useMemo(() => {
    const ranges = []
    let start = null
    const vis = history.slice(-120)
    for (const p of vis) {
      if (p.ff_active && start === null) start = p.time
      if (!p.ff_active && start !== null) { ranges.push([start, p.time]); start = null }
    }
    if (start !== null && vis.length > 0) ranges.push([start, vis.at(-1).time])
    return ranges
  }, [history])

  // Compute time-ranges where SysID PRBS test was active (for amber bands)
  const sysidRanges = useMemo(() => {
    const ranges = []
    let start = null
    const vis = history.slice(-120)
    for (const p of vis) {
      if (p.sysid_active && start === null) start = p.time
      if (!p.sysid_active && start !== null) { ranges.push([start, p.time]); start = null }
    }
    if (start !== null && vis.length > 0) ranges.push([start, vis.at(-1).time])
    return ranges
  }, [history])

  const hasNoise    = (noiseSigma ?? 0) > 0
  const T_danger    = modelInfo?.T_danger  ?? 400
  const T_runaway   = modelInfo?.T_runaway ?? 420

  const caData   = useMemo(
    () => buildChartData(history, predTraj, 'ca',   'CA', 'sp_ca',   'ca_raw'),
    [history, predTraj],
  )
  const tempData = useMemo(
    () => buildChartData(history, predTraj, 'temp', 'T',  'sp_temp', 'temp_raw'),
    [history, predTraj],
  )

  const profit   = useMemo(
    () => computeProfit(currentState, economicCfg),
    [currentState, economicCfg],
  )

  const violCA   = !!(violations?.x0_low || violations?.x0_high)
  const violTemp = !!(violations?.x1_low || violations?.x1_high)

  return (
    <div className="relative flex flex-col gap-4">
      {/* Emergency flash overlay */}
      {isRunaway && (
        <div className="absolute inset-0 bg-red-950/25 animate-pulse rounded-xl
                        pointer-events-none z-0" />
      )}
      {esdActive && !isRunaway && (
        <div className="absolute inset-0 bg-amber-950/20 animate-pulse rounded-xl
                        pointer-events-none z-0" />
      )}

      <div className="relative z-10 flex flex-col gap-4">
        {/* ESD aktív banner */}
        {esdActive && (
          <div className="rounded-lg border border-amber-500 bg-amber-950/60 p-3
                          flex items-center gap-3 text-amber-300">
            <span className="text-xl">🛑</span>
            <div>
              <p className="text-sm font-bold">EMERGENCY SHUTDOWN AKTÍV</p>
              <p className="text-xs opacity-75">F = F_min, Tc = Tc_min (max hűtés) — NMPC felülírva</p>
            </div>
          </div>
        )}

        {/* Feedforward aktív banner */}
        {ffActive && (
          <div className="rounded-lg border border-cyan-500/60 bg-cyan-950/40 px-4 py-2.5
                          flex items-center gap-3 text-cyan-300 animate-pulse">
            <span className="text-lg">⚡</span>
            <div>
              <p className="text-sm font-bold">FEEDFORWARD AKTÍV — Proaktív kompenzáció</p>
              <p className="text-xs opacity-75">
                Az MPC látja a zavarást és már most mozgatja a szelepeket — mielőtt a PV eltér a SP-től.
                Figyeld a kék sávokat az MV grafikonokon!
              </p>
            </div>
          </div>
        )}

        {/* ID TEST ACTIVE banner */}
        {sysidActive && (
          <div className="rounded-lg border border-amber-500/60 bg-amber-950/40 px-4 py-2.5
                          flex items-center gap-3 text-amber-300 animate-pulse">
            <span className="text-lg">⚡</span>
            <div>
              <p className="text-sm font-bold">ID TEST AKTÍV — PRBS Excitation</p>
              <p className="text-xs opacity-75">
                A PRBS gerjesztő jel fut — az MPC nem irányít. Sárga sávok jelzik az ID periódust az MV grafikonokon.
              </p>
            </div>
          </div>
        )}

        {/* Runaway figyelmeztetés */}
        <RunawayAlert approaching={approachingRunaway} runaway={isRunaway} />

        {/* LINEAR MODE badge */}
        {isLinear && (
          <div className="flex items-center gap-3 rounded-lg border border-violet-500/50
                          bg-violet-950/30 px-4 py-2.5">
            <span className="inline-flex items-center gap-1.5 rounded-md bg-violet-600/80
                             px-2 py-0.5 text-xs font-bold text-white tracking-wider">
              ⚠ LINEAR MODE active
            </span>
            <span className="text-xs text-violet-300">
              Fix Jacobian (SS körüli) — nagy excursionoknál pontatlan
            </span>
          </div>
        )}

        {/* MHE active badge */}
        {estimatorType === 'MHE' && (
          <div className={`flex items-center gap-3 rounded-lg border px-4 py-2
                          ${mheSuccess
                            ? 'border-indigo-500/50 bg-indigo-950/30'
                            : 'border-red-500/40 bg-red-950/20 animate-pulse'}`}>
            <span className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-bold text-white tracking-wider ${
              mheSuccess ? 'bg-indigo-600/80' : 'bg-red-600/80'
            }`}>
              {mheSuccess ? '⬡ MHE active' : '⬡ MHE failed'}
            </span>
            <span className="text-xs text-indigo-300">
              Nonlineáris IPOPT becslő — x̂ = argmin Σ|y−ŷ|² s.t. ODE, CA∈[0,1], T∈[300,500]
            </span>
          </div>
        )}

        {/* KPI sor */}
        <div className="grid grid-cols-2 sm:grid-cols-6 gap-3">
          <KpiCard label="CA (x̂)"
            value={currentState?.states[0]?.toFixed(4)}
            unit="mol/L" color="text-blue-400" alarm={violCA} />
          <KpiCard label="T (x̂)"
            value={currentState?.states[1]?.toFixed(1)}
            unit="K" color={isRunaway ? 'text-red-400' : approachingRunaway ? 'text-amber-400' : 'text-orange-400'}
            alarm={violTemp} />
          <KpiCard label="Feed Flow F"
            value={currentState?.control[0]?.toFixed(1)}
            unit="L/min" color="text-cyan-400" />
          <KpiCard label="Coolant Tc"
            value={currentState?.control[1]?.toFixed(1)}
            unit="K" color="text-violet-400" />
          <CostKpi state={currentState} mpcCfg={mpcCfg} />
          <ProfitKpi profit={profit} economicMode={economicCfg?.economicMode} />
        </div>

        {/* IAE sor */}
        <div className="grid grid-cols-2 gap-3">
          <div className={`card flex flex-col gap-1 border ${
            isLinear ? 'border-violet-700/60' : 'border-slate-700'
          }`}>
            <span className="label mb-0">IAE — CA</span>
            <span className="text-lg font-bold tabular-nums text-blue-400">
              {(iae?.ca ?? 0).toFixed(3)}
              <span className="text-xs font-normal text-slate-500 ml-1">mol·s/L</span>
            </span>
            <span className="text-xs text-slate-600">∫|CA_hat − SP_CA| dt</span>
          </div>
          <div className={`card flex flex-col gap-1 border ${
            isLinear ? 'border-violet-700/60' : 'border-slate-700'
          }`}>
            <span className="label mb-0">IAE — T</span>
            <span className="text-lg font-bold tabular-nums text-orange-400">
              {(iae?.temp ?? 0).toFixed(1)}
              <span className="text-xs font-normal text-slate-500 ml-1">K·s</span>
            </span>
            <span className="text-xs text-slate-600">∫|T_hat − SP_T| dt</span>
          </div>
        </div>

        {/* SysID lépésválasz összehasonlítás */}
        {sysidResult && (
          <SysIdStepResponseChart sysidResult={sysidResult} />
        )}

        {/* Tank vizualizáció */}
        <TankViz currentState={currentState} reactorMode={reactorMode} />

        {/* Állapotváltozók */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <DualProcessChart
            data={caData} history={history}
            title={isSeries
              ? (hasNoise ? 'CA — R1 (KF) / R2 / SP' : 'Concentration CA — R1/R2/SP/Prediction')
              : (hasNoise ? 'CA — Raw / KF-Filtered / SP' : 'Concentration CA — PV / SP / Prediction')}
            unit="mol/L" color="#60a5fa" color2="#2dd4bf"
            yMin={0.02} yMax={0.98}
            cMin={modelInfo?.states[0]?.min} cMax={modelInfo?.states[0]?.max}
            violKey="viol_ca" spKey="sp_ca" hasNoise={hasNoise}
            showR2={isSeries} pvKey2="ca2" spKey2="sp_ca2"
          />
          <DualProcessChart
            data={tempData} history={history}
            title={isSeries
              ? (hasNoise ? 'T — R1 (KF) / R2 / SP' : 'Temperature T — R1/R2/SP/Prediction')
              : (hasNoise ? 'T — Raw / KF-Filtered / SP' : 'Temperature T — PV / SP / Prediction')}
            unit="K" color="#fb923c" predColor="#fbbf2477" color2="#f472b6"
            yMin={300} yMax={430}
            cMin={modelInfo?.states[1]?.min} cMax={modelInfo?.states[1]?.max}
            violKey="viol_temp" spKey="sp_temp" hasNoise={hasNoise}
            dangerLine={T_danger} runawayLine={T_runaway}
            showR2={isSeries} pvKey2="temp2" spKey2="sp_temp2"
          />
        </div>

        {/* Beavatkozójelek */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ControlChart data={history} dataKey="f_flow"
            title="MV1 — Feed Flow F" unit="L/min" color="#22d3ee"
            cMin={modelInfo?.inputs[0]?.min ?? 50} cMax={modelInfo?.inputs[0]?.max ?? 200}
            ffRanges={ffRanges} sysidRanges={sysidRanges} />
          <ControlChart data={history} dataKey="tc"
            title={isSeries ? 'MV2 — Coolant Tc1 (R1)' : 'MV2 — Coolant Temperature Tc'}
            unit="K" color="#a78bfa"
            cMin={modelInfo?.inputs[1]?.min ?? 250} cMax={modelInfo?.inputs[1]?.max ?? 350}
            ffRanges={ffRanges} sysidRanges={sysidRanges} />
        </div>
        {isSeries && (
          <ControlChart data={history} dataKey="tc2"
            title="MV3 — Coolant Tc2 (R2)" unit="K" color="#34d399"
            cMin={250} cMax={350} ffRanges={ffRanges} sysidRanges={sysidRanges} />
        )}

        {/* Fázissík */}
        {history.length > 2 && (
          <PhasePlaneChart
            history={history}
            setpoints={setpoints}
            modelInfo={modelInfo}
          />
        )}

        {/* MHE residuals chart */}
        {estimatorType === 'MHE' && history.length > 2 && (
          <MheResidualsChart history={history} />
        )}

        {/* Alarm Journal */}
        <AlarmJournal alarmLog={alarmLog ?? []} />
      </div>
    </div>
  )
}
