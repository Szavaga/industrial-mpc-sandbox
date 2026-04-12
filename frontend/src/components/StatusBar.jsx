import { Play, Square, RotateCcw, Cpu, AlertTriangle, CheckCircle2, Zap, Wifi, WifiOff, Loader2 } from 'lucide-react'

const WS_STYLES = {
  connected:    { icon: Wifi,    color: 'text-emerald-400', label: 'WS' },
  connecting:   { icon: Loader2, color: 'text-amber-400 animate-spin', label: 'WS…' },
  disconnected: { icon: WifiOff, color: 'text-red-400',     label: 'WS OFF' },
}

export function StatusBar({ sim }) {
  const { running, setRunning, reset, currentState, mpcSuccess, violations, distActive, wsStatus } = sim
  const ws = WS_STYLES[wsStatus] ?? WS_STYLES.disconnected
  const WsIcon = ws.icon
  const violCount = Object.keys(violations).length
  const t = currentState?.time?.toFixed(1) ?? '0.0'

  return (
    <header className="flex items-center justify-between px-5 py-3
                       bg-slate-900 border-b border-slate-700 shrink-0 z-10">
      {/* Bal: cím */}
      <div className="flex items-center gap-3">
        <Cpu className="text-blue-400" size={22} />
        <div>
          <h1 className="text-sm font-bold text-slate-100 leading-none">
            Industrial MPC Sandbox
          </h1>
          <span className="text-xs text-slate-500">2×2 CSTR — MIMO Model Predictive Control</span>
        </div>
      </div>

      {/* Közép: státuszok */}
      <div className="flex items-center gap-6 text-xs">
        {/* WebSocket kapcsolat */}
        <div className={`flex items-center gap-1.5 font-semibold ${ws.color}`}>
          <WsIcon size={14} />
          {ws.label}
        </div>
        {/* Idő */}
        <div className="flex items-center gap-1.5 text-slate-400">
          <span className="label mb-0">t =</span>
          <span className="tabular-nums text-slate-200 font-bold">{t} s</span>
        </div>

        {/* MPC státusz */}
        <div className={`flex items-center gap-1.5 font-semibold
          ${mpcSuccess ? 'text-emerald-400' : 'text-amber-400'}`}>
          {mpcSuccess
            ? <><CheckCircle2 size={14} /> MPC OK</>
            : <><AlertTriangle size={14} /> MPC FALLBACK</>}
        </div>

        {/* Kényszersértés */}
        {violCount > 0 && (
          <div className="flex items-center gap-1.5 text-red-400 font-semibold animate-pulse-fast">
            <AlertTriangle size={14} />
            {violCount} VIOLATION{violCount > 1 ? 'S' : ''}
          </div>
        )}

        {/* Zavarás */}
        {distActive && (
          <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
            <Zap size={14} />
            DISTURBANCE ACTIVE
          </div>
        )}

        {/* Futás jelző */}
        <div className={`flex items-center gap-1.5 font-semibold
          ${running ? 'text-emerald-400' : 'text-slate-500'}`}>
          <span className={`w-2 h-2 rounded-full ${running ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
          {running ? 'RUNNING' : 'STOPPED'}
        </div>
      </div>

      {/* Jobb: vezérlő gombok */}
      <div className="flex items-center gap-2">
        <button
          className="btn-primary flex items-center gap-2"
          onClick={() => setRunning(r => !r)}
        >
          {running
            ? <><Square size={14} /> Stop</>
            : <><Play  size={14} /> Start</>}
        </button>
        <button
          className="btn-ghost flex items-center gap-2"
          onClick={reset}
        >
          <RotateCcw size={14} /> Reset
        </button>
      </div>
    </header>
  )
}
