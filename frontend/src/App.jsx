import { useSimulation } from './hooks/useSimulation'
import { StatusBar } from './components/StatusBar'
import { Sidebar }   from './components/Sidebar'
import { ChartGrid } from './components/ChartGrid'

export default function App() {
  const sim = useSimulation()

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-surface">
      <StatusBar sim={sim} />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar sim={sim} />
        <main className="flex-1 p-4 overflow-y-auto">
          {sim.history.length === 0 && !sim.running ? (
            <EmptyState onStart={() => sim.setRunning(true)} />
          ) : (
            <ChartGrid sim={sim} />
          )}
        </main>
      </div>
    </div>
  )
}

function EmptyState({ onStart }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
      <div className="text-6xl select-none">⚙️</div>
      <div>
        <h2 className="text-xl font-bold text-slate-200 mb-2">
          Industrial MPC Sandbox
        </h2>
        <p className="text-sm text-slate-500 max-w-md">
          2×2 MIMO CSTR szimulator — szint és hőmérséklet szabályozás
          bemenő áramlással és hűtővízzel. Model Predictive Control
          (Gekko / IPOPT) valós idejű optimalizálással.
        </p>
      </div>
      <div className="flex flex-col items-center gap-2">
        <button className="btn-primary px-8 py-3 text-base" onClick={onStart}>
          ▶ Szimuláció indítása
        </button>
        <p className="text-xs text-slate-600">
          Bal oldalt állítsd be a setpointokat és az MPC paramétereket
        </p>
      </div>
      <div className="grid grid-cols-3 gap-4 text-xs text-slate-600 max-w-sm">
        {[
          ['N = 15', 'Prediction Horizon'],
          ['M = 5',  'Control Horizon'],
          ['dt = 0.5s', 'Időlépés'],
          ['Q = diag(8,12)', 'Tracking súly'],
          ['R = diag(0.5,0.5)', 'Input költség'],
          ['IPOPT', 'MPC Solver'],
        ].map(([v, l]) => (
          <div key={l} className="card p-2 text-center">
            <p className="font-bold text-slate-400">{v}</p>
            <p className="text-slate-600">{l}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
