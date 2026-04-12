import { useState } from 'react'
import { Sliders, Target, Zap, ChevronDown, ChevronUp, Info, Radio, AlertTriangle,
         DollarSign, ShieldAlert, Download, Cpu, FlaskConical, Activity, ScanLine,
         BrainCircuit } from 'lucide-react'

// ─── Csúszka segédkomponens ───────────────────────────────────────────────────
function Slider({ label, value, min, max, step = 0.1, unit, color = 'accent-blue-500', onChange }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <span className="text-xs text-slate-400">{label}</span>
        <span className={`text-xs font-bold tabular-nums ${color}`}>
          {Number(value).toFixed(step < 0.1 ? 3 : step < 1 ? 2 : 1)} {unit}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, #3b82f6 ${pct}%, #334155 ${pct}%)`,
          accentColor: '#3b82f6',
        }}
      />
    </div>
  )
}

// ─── Összecsukható szekció ────────────────────────────────────────────────────
function Section({ title, icon: Icon, defaultOpen = true, children, accent }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`card p-3 ${accent ? 'border-red-500/40' : ''}`}>
      <button
        className="flex w-full items-center justify-between mb-0"
        onClick={() => setOpen(o => !o)}
      >
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-300">
          <Icon size={13} className="text-slate-400" />
          {title}
        </div>
        {open ? <ChevronUp size={13} className="text-slate-500" />
               : <ChevronDown size={13} className="text-slate-500" />}
      </button>
      {open && <div className="mt-3 flex flex-col gap-3">{children}</div>}
    </div>
  )
}

// ─── Szám-input ───────────────────────────────────────────────────────────────
function NumInput({ label, value, min, max, step = 1, onChange }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs text-slate-400 flex-1">{label}</span>
      <input
        type="number" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        className="input-num w-20 text-right py-1"
      />
    </div>
  )
}

// ─── Reactor mode selector ───────────────────────────────────────────────────
function ReactorModePanel({ sim }) {
  const { reactorMode, updateReactorMode } = sim
  const isSeries = reactorMode === 'SERIES'
  return (
    <div className={`card p-3 border-2 transition-colors duration-300 ${
      isSeries ? 'border-teal-500/50 bg-teal-950/20' : 'border-slate-700'
    }`}>
      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-300 mb-3">
        <FlaskConical size={13} className="text-slate-400" />
        Configuration
      </div>
      <div className="flex rounded-lg overflow-hidden border border-slate-700 text-xs font-semibold">
        <button
          onClick={() => updateReactorMode('SINGLE')}
          className={`flex-1 py-2 transition-colors duration-150 ${
            !isSeries ? 'bg-slate-500 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          SINGLE
        </button>
        <button
          onClick={() => updateReactorMode('SERIES')}
          className={`flex-1 py-2 transition-colors duration-150 ${
            isSeries ? 'bg-teal-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          SERIES
        </button>
      </div>
      <p className={`text-xs mt-2 ${isSeries ? 'text-teal-400' : 'text-slate-500'}`}>
        {isSeries
          ? '2×CSTR sorba — R1→R2, 4 állapot, 3 bemenet'
          : '1×CSTR — 2 állapot, 2 bemenet'}
      </p>
      {isSeries && (
        <div className="mt-1 text-xs text-slate-600">
          V₁=100L, V₂=50L • közös F áramlás
        </div>
      )}
    </div>
  )
}

// ─── R2 setpoint panel (csak SERIES módban) ──────────────────────────────────
function Setpoint2Panel({ sim }) {
  const { setpoints2, updateSetpoints2 } = sim
  return (
    <Section title="R2 Setpoints" icon={Target} defaultOpen={true} accent={false}>
      <Slider
        label="CA2 Setpoint" value={setpoints2.ca} min={0.1} max={0.7} step={0.01}
        unit="mol/L" color="text-teal-400"
        onChange={v => updateSetpoints2(v, setpoints2.temp)}
      />
      <Slider
        label="T2 Setpoint" value={setpoints2.temp} min={310} max={400} step={1}
        unit="K" color="text-cyan-400"
        onChange={v => updateSetpoints2(setpoints2.ca, v)}
      />
      <div className="text-xs text-slate-600 -mt-1">
        R2 SS ≈ CA=0.453 mol/L, T=329 K
      </div>
    </Section>
  )
}

// ─── Controller switch ────────────────────────────────────────────────────────
const LINEAR_TOOLTIP = `Lineáris MPC — Jacobian-alapú közelítés

A kontroller a munkapont (CA=0.5 mol/L, T=350 K)
körül számolt fix A és B mátrixokat használja:

  dz/dt = A·z + B·v   (z = x−x_ss, v = u−u_ss)

PONTOSSÁGI KORLÁT:
Az Arrhenius-egyenlet k(T) = k₀·exp(−Ea/RT)
exponenciálisan nemlineáris. A linearizáció csak
a munkapont közelében (ΔT < ~15 K) érvényes.
Nagy szétérésnél (pl. T → 400 K) a lineáris
modell szisztematikusan alulbecsüli a reakció-
sebességet → az NMPC jobban teljesít.`

function ControllerSwitchPanel({ sim }) {
  const { controllerType, updateControllerType, iae } = sim
  const isLinear = controllerType === 'LINEAR'
  const [showTip, setShowTip] = useState(false)

  return (
    <div className={`card p-3 border-2 transition-colors duration-300 ${
      isLinear ? 'border-violet-500/50 bg-violet-950/20' : 'border-blue-500/40 bg-blue-950/10'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-300">
          <Cpu size={13} className="text-slate-400" />
          Control Logic
        </div>
        <button
          title="Miért pontatlan a lineáris mód?"
          onClick={() => setShowTip(v => !v)}
          className="text-slate-500 hover:text-slate-300 transition-colors"
        >
          <Info size={13} />
        </button>
      </div>

      {/* Toggle */}
      <div className="flex rounded-lg overflow-hidden border border-slate-700 text-xs font-semibold">
        <button
          onClick={() => updateControllerType('NONLINEAR')}
          className={`flex-1 py-2 transition-colors duration-150 ${
            !isLinear
              ? 'bg-blue-600 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          NMPC
        </button>
        <button
          onClick={() => updateControllerType('LINEAR')}
          className={`flex-1 py-2 transition-colors duration-150 ${
            isLinear
              ? 'bg-violet-600 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          Linear
        </button>
      </div>

      {/* Mode description */}
      <p className={`text-xs mt-2 ${isLinear ? 'text-violet-400' : 'text-blue-400'}`}>
        {isLinear
          ? 'Fix Jacobian (A,B) — SS körüli linearizáció'
          : 'Arrhenius ODE — valódi nemlineáris dinamika'}
      </p>

      {/* IAE metrics */}
      <div className="mt-2 border-t border-slate-700 pt-2 flex flex-col gap-1">
        <p className="text-xs text-slate-500 font-semibold">IAE (Integral Absolute Error)</p>
        <div className="flex justify-between text-xs tabular-nums">
          <span className="text-slate-500">∫|CA−SP| dt</span>
          <span className="text-blue-400 font-bold">{(iae?.ca ?? 0).toFixed(2)}</span>
        </div>
        <div className="flex justify-between text-xs tabular-nums">
          <span className="text-slate-500">∫|T−SP| dt</span>
          <span className="text-orange-400 font-bold">{(iae?.temp ?? 0).toFixed(1)}</span>
        </div>
      </div>

      {/* Educational tooltip */}
      {showTip && (
        <pre className="mt-2 text-[10px] text-slate-400 bg-slate-900 rounded p-2
                        whitespace-pre-wrap leading-relaxed border border-slate-700">
          {LINEAR_TOOLTIP}
        </pre>
      )}
    </div>
  )
}

// ─── Estimator panel (KF vs MHE) ─────────────────────────────────────────────
const MHE_TOOLTIP = `Moving Horizon Estimation (MHE)

A Kalman-szűrő nemlineáris, optimalizáció-alapú
verziója — az NMPC "dual"-ja.

NMPC : jövőbe optimalizál (predikciós horizont)
MHE  : múltba optimalizál (becslési ablak)
Mindkettő: Gekko + IPOPT ugyanazzal a solver-rel.

MHE ELŐNYÖK a KF-fel szemben:
  • Nonlineáris ODE modell (pontos Arrhenius-dinamika,
    nem linearizált közelítés)
  • Fizikai korlátok keményen érvényesítve:
    CA ≥ 0.001, T ∈ [300, 500] K
  • L1 norma → outlier-robusztus becslés:
    egyetlen kiugró mérés nem viszi félre az állapotot!
    (KF nem tudja ezt)

MHE KONFIGURÁCIÓ:
  Horizon N  — ablakhossz (több adat = pontosabb, lassabb)
  R(CA), R(T)— mérési zajvariancia (magasabb = mérés megbízhatatlan)
  WMODEL     — modelltámogatás (magas = simább becslés)
  L1 norma   — érzéketlenség szenzorhibára

VIZUÁLIS BIZONYÍTÉK:
  Inject sensor fault → figyelj MHE vs KF x̂ divergenciára!
  L1 módban a kiugró mérés szinte egyáltalán nem veri ki
  a becslést — a KF azonnal "elcsúszik".`

function EstimatorPanel({ sim }) {
  const {
    estimatorType, mheSuccess, mheResiduals,
    updateEstimator, controllerType, reactorMode,
  } = sim
  const isMHE    = estimatorType === 'MHE'
  const isSeries = reactorMode === 'SERIES'
  const [showTip,  setShowTip]  = useState(false)
  const [horizon,  setHorizon]  = useState(10)
  const [rCa,      setRCa]      = useState(0.001)
  const [rT,       setRT]       = useState(0.04)
  const [wmodel,   setWmodel]   = useState(0.1)
  const [evType,   setEvType]   = useState(2)

  const handleSwitch = (type) => {
    if (type === 'MHE') {
      updateEstimator('MHE', { horizon, R_ca: rCa, R_t: rT, wmodel, ev_type: evType })
    } else {
      updateEstimator('KF')
    }
  }

  const handleApply = () => {
    if (isMHE) {
      updateEstimator('MHE', { horizon, R_ca: rCa, R_t: rT, wmodel, ev_type: evType })
    }
  }

  return (
    <div className={`card p-3 border-2 transition-colors duration-300 ${
      isMHE
        ? mheSuccess ? 'border-indigo-500/50 bg-indigo-950/20' : 'border-red-500/40 bg-red-950/10'
        : 'border-slate-700'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-300">
          <BrainCircuit size={13} className="text-slate-400" />
          Estimator
        </div>
        <div className="flex items-center gap-1.5">
          {isMHE && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold border ${
              mheSuccess
                ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
                : 'bg-red-500/20 text-red-300 border-red-500/40 animate-pulse'
            }`}>
              {mheSuccess ? '✓ IPOPT' : '✗ failed'}
            </span>
          )}
          <button
            title="MHE vs KF összehasonlítás"
            onClick={() => setShowTip(v => !v)}
            className="text-slate-500 hover:text-slate-300 transition-colors"
          >
            <Info size={13} />
          </button>
        </div>
      </div>

      {/* KF / MHE toggle */}
      <div className={`flex rounded-lg overflow-hidden border text-xs font-semibold ${
        isSeries ? 'border-slate-800 opacity-50' : 'border-slate-700'
      }`}>
        <button
          onClick={() => !isSeries && handleSwitch('KF')}
          className={`flex-1 py-2 transition-colors duration-150 ${
            !isMHE
              ? 'bg-slate-500 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          KF
        </button>
        <button
          onClick={() => !isSeries && handleSwitch('MHE')}
          className={`flex-1 py-2 transition-colors duration-150 ${
            isMHE
              ? 'bg-indigo-600 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          MHE
        </button>
      </div>

      <p className={`text-xs mt-2 ${isMHE ? 'text-indigo-400' : 'text-slate-500'}`}>
        {isMHE
          ? 'Nonlineáris IPOPT — fizikai korlátok, L1/L2 norma'
          : 'Lineáris Kalman-szűrő — Jacobian-linearizált modell'}
      </p>

      {isSeries && (
        <div className="text-[10px] text-slate-600 mt-1">MHE csak SINGLE módban</div>
      )}

      {/* MHE config (only when MHE active) */}
      {isMHE && (
        <div className="mt-3 border-t border-slate-700 pt-2 flex flex-col gap-2">
          <p className="text-xs text-slate-500 font-semibold">MHE konfiguráció</p>

          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-slate-400">Horizon N</span>
            <input
              type="number" min={3} max={30} value={horizon}
              onChange={e => setHorizon(+e.target.value)}
              className="input-num w-16 text-right py-0.5 text-xs"
            />
          </div>

          <Slider label="R(CA)" value={rCa} min={0.0001} max={0.05} step={0.0001}
            unit="" color="text-blue-400" onChange={setRCa} />
          <Slider label="R(T)"  value={rT}  min={0.001}  max={1.0}  step={0.001}
            unit="" color="text-orange-400" onChange={setRT} />
          <Slider label="WMODEL" value={wmodel} min={0.001} max={1.0} step={0.001}
            unit="" color="text-indigo-400" onChange={setWmodel} />

          {/* L2 / L1 toggle */}
          <div className="flex rounded-lg overflow-hidden border border-slate-700 text-xs font-semibold">
            <button
              onClick={() => setEvType(2)}
              className={`flex-1 py-1.5 transition-colors duration-150 ${
                evType === 2 ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              L2 Gaussian
            </button>
            <button
              onClick={() => setEvType(1)}
              className={`flex-1 py-1.5 transition-colors duration-150 ${
                evType === 1 ? 'bg-amber-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              L1 Robust
            </button>
          </div>

          <button
            className="btn-ghost text-xs py-1 mt-0.5 border border-indigo-500/40 text-indigo-400 hover:bg-indigo-950/30"
            onClick={handleApply}
          >
            Apply config
          </button>
        </div>
      )}

      {/* Residuals */}
      {isMHE && (
        <div className="mt-2 border-t border-slate-700 pt-2 flex flex-col gap-1">
          <p className="text-xs text-slate-500 font-semibold">Maradék hiba |y − ŷ|</p>
          <div className="flex justify-between text-xs tabular-nums">
            <span className="text-slate-400">CA</span>
            <span className={`font-bold ${mheResiduals.ca < 0.01 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {mheResiduals.ca.toFixed(5)} mol/L
            </span>
          </div>
          <div className="flex justify-between text-xs tabular-nums">
            <span className="text-slate-400">T</span>
            <span className={`font-bold ${mheResiduals.t < 0.5 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {mheResiduals.t.toFixed(3)} K
            </span>
          </div>
        </div>
      )}

      {/* Educational tooltip */}
      {showTip && (
        <pre className="mt-2 text-[10px] text-slate-400 bg-slate-900 rounded p-2
                        whitespace-pre-wrap leading-relaxed border border-slate-700">
          {MHE_TOOLTIP}
        </pre>
      )}
    </div>
  )
}

// ─── Feedforward panel ────────────────────────────────────────────────────────
const FF_TOOLTIP = `Mért zavarás előrecsatolt kompenzálása (MDFF)

A klasszikus visszacsatolásos MPC csak az
állapoteltérés (x̂ − SP) alapján avatkozik be.

FEEDFORWARD ON:
Az MPC a predikciós modelljébe beleszámolja a
mért zavarást (d_CA, d_T) az egész horizonton.
→ Proaktív: a szelepek mozognak MIELŐTT a PV
  (CA, T) eltér a setpointtól!

FEEDFORWARD OFF:
Az MPC "vak" a zavarásra — csak x̂ eltérés után
reagál (feedback-only).
→ Reaktív: elkésett kompenzáció.

VIZUÁLIS BIZONYÍTÉK:
Inject disturbance-t mindkét módban, és figyeld:
  • FF ON : az MV-k (F, Tc) azonnal elmozdulnak
  • FF OFF: az MV-k csak t+τ után, ahol τ ≈ 60s

TELJESÍTMÉNY:
Az IAE delta a %-os javulást mutatja FF ON vs OFF
(csak azonos ideig futtatott módok esetén érvényes).`

function FeedforwardPanel({ sim }) {
  const { feedforwardEnabled, updateFeedforward, ffActive, iaeByMode, distActive } = sim
  const [showTip, setShowTip] = useState(false)

  // Average IAE rate [error·s / simulation-second] for each mode
  const rateOn  = iaeByMode.time_ff_on  > 5 ? iaeByMode.ff_on_ca  / iaeByMode.time_ff_on  : null
  const rateOff = iaeByMode.time_ff_off > 5 ? iaeByMode.ff_off_ca / iaeByMode.time_ff_off : null
  const deltaPct = rateOn != null && rateOff != null && rateOff > 0
    ? ((rateOff - rateOn) / rateOff * 100)
    : null

  return (
    <div className={`card p-3 border-2 transition-colors duration-300 ${
      feedforwardEnabled ? 'border-cyan-500/50 bg-cyan-950/20' : 'border-slate-700'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-300">
          <Activity size={13} className="text-slate-400" />
          Feedforward
        </div>
        <div className="flex items-center gap-1.5">
          {/* Live FF-active indicator */}
          {ffActive && (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-bold
                             bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 animate-pulse">
              ⚡ PROAKTÍV
            </span>
          )}
          <button
            title="Hogyan működik a feedforward?"
            onClick={() => setShowTip(v => !v)}
            className="text-slate-500 hover:text-slate-300 transition-colors"
          >
            <Info size={13} />
          </button>
        </div>
      </div>

      {/* Toggle */}
      <div className="flex rounded-lg overflow-hidden border border-slate-700 text-xs font-semibold">
        <button
          onClick={() => updateFeedforward(false)}
          className={`flex-1 py-2 transition-colors duration-150 ${
            !feedforwardEnabled
              ? 'bg-slate-500 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          OFF
        </button>
        <button
          onClick={() => updateFeedforward(true)}
          className={`flex-1 py-2 transition-colors duration-150 ${
            feedforwardEnabled
              ? 'bg-cyan-600 text-white'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          ON
        </button>
      </div>

      <p className={`text-xs mt-2 ${feedforwardEnabled ? 'text-cyan-400' : 'text-slate-500'}`}>
        {feedforwardEnabled
          ? 'MPC látja a zavarást → proaktív előre-kompenzáció'
          : 'MPC vak a zavarásra → csak x̂ eltérésre reagál'}
      </p>

      {/* IAE Performance Delta */}
      <div className="mt-2 border-t border-slate-700 pt-2 flex flex-col gap-1">
        <p className="text-xs text-slate-500 font-semibold">IAE ráta összehasonlítás (CA)</p>
        <div className="flex justify-between text-xs tabular-nums">
          <span className="text-cyan-400">⚡ FF ON</span>
          <span className="font-bold text-cyan-300">
            {rateOn != null ? rateOn.toFixed(5) : '—'}
          </span>
        </div>
        <div className="flex justify-between text-xs tabular-nums">
          <span className="text-slate-500">✕ FF OFF</span>
          <span className="font-bold text-slate-400">
            {rateOff != null ? rateOff.toFixed(5) : '—'}
          </span>
        </div>
        {deltaPct != null ? (
          <div className={`text-xs font-bold text-center mt-0.5 px-2 py-0.5 rounded ${
            deltaPct > 0
              ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
              : 'bg-red-500/15 text-red-400 border border-red-500/30'
          }`}>
            FF javulás: {deltaPct > 0 ? '+' : ''}{deltaPct.toFixed(1)}%
          </div>
        ) : (
          <div className="text-xs text-slate-600 text-center mt-0.5">
            Tesztelj mindkét módban a delta-hoz
          </div>
        )}
      </div>

      {/* Educational tooltip */}
      {showTip && (
        <pre className="mt-2 text-[10px] text-slate-400 bg-slate-900 rounded p-2
                        whitespace-pre-wrap leading-relaxed border border-slate-700">
          {FF_TOOLTIP}
        </pre>
      )}
    </div>
  )
}

// ─── Setpoint panel ───────────────────────────────────────────────────────────
function SetpointPanel({ sim }) {
  const { setpoints, updateSetpoints } = sim
  return (
    <Section title="Setpoints" icon={Target}>
      <Slider
        label="CA Setpoint" value={setpoints.ca} min={0.1} max={0.9} step={0.01}
        unit="mol/L" color="text-blue-400"
        onChange={v => updateSetpoints(v, setpoints.temp)}
      />
      <Slider
        label="T Setpoint" value={setpoints.temp} min={315} max={415} step={1}
        unit="K" color="text-orange-400"
        onChange={v => updateSetpoints(setpoints.ca, v)}
      />
      <div className="text-xs text-slate-600 -mt-1">
        SS: CA=0.5 mol/L, T=350 K (instabil SS)
      </div>
    </Section>
  )
}

// ─── NMPC konfig panel ────────────────────────────────────────────────────────
function MpcConfigPanel({ sim }) {
  const { mpcCfg, updateMpcCfg } = sim
  const set = (key, val) => updateMpcCfg({ ...mpcCfg, [key]: val })

  return (
    <Section title="NMPC Configuration" icon={Sliders} defaultOpen={false}>
      <div className="text-xs text-slate-500 -mt-1 mb-1 flex items-center gap-1">
        <Info size={11} /> Arrhenius-alapú nemlineáris MPC
      </div>

      <div className="flex flex-col gap-2 pb-2 border-b border-slate-700">
        <NumInput label="Prediction Horizon (N)" value={mpcCfg.prediction_horizon}
          min={10} max={80} onChange={v => set('prediction_horizon', v)} />
        <NumInput label="Control Horizon (M)" value={mpcCfg.control_horizon}
          min={1} max={30} onChange={v => set('control_horizon', v)} />
      </div>

      <div className="flex flex-col gap-2 pb-2 border-b border-slate-700">
        <p className="text-xs text-slate-500 font-semibold">Q — Tracking súlyok</p>
        <Slider label="Q[CA]" value={mpcCfg.Q00} min={1} max={200} step={1}
          unit="" color="text-blue-400" onChange={v => set('Q00', v)} />
        <Slider label="Q[T]"  value={mpcCfg.Q11} min={0.01} max={5} step={0.01}
          unit="" color="text-orange-400" onChange={v => set('Q11', v)} />
      </div>

      <div className="flex flex-col gap-2">
        <p className="text-xs text-slate-500 font-semibold">R — Beavatkozás költség</p>
        <Slider label="R[F]"  value={mpcCfg.R00} min={0.0001} max={0.1} step={0.0005}
          unit="" color="text-cyan-400" onChange={v => set('R00', v)} />
        <Slider label="R[Tc]" value={mpcCfg.R11} min={0.001}  max={0.5} step={0.001}
          unit="" color="text-violet-400" onChange={v => set('R11', v)} />
      </div>
    </Section>
  )
}

// ─── Zavarás panel ────────────────────────────────────────────────────────────
function DisturbancePanel({ sim }) {
  const { injectDisturbance, distActive } = sim
  const [dCA,  setDCA]  = useState(0)
  const [dTemp, setDTemp] = useState(0)
  const [dur,   setDur]   = useState(10)

  return (
    <Section title="Disturbance Injection" icon={Zap} defaultOpen={false}>
      {distActive && (
        <div className="text-xs text-amber-400 font-bold animate-pulse flex items-center gap-1">
          <Zap size={11} /> Zavarás aktív...
        </div>
      )}
      <Slider
        label="d(CA)" value={dCA} min={-0.02} max={0.02} step={0.002}
        unit="mol/(L·s)" color="text-amber-400"
        onChange={setDCA}
      />
      <Slider
        label="d(T)" value={dTemp} min={-5} max={5} step={0.5}
        unit="K/s" color="text-amber-400"
        onChange={setDTemp}
      />
      <NumInput label="Időtartam (lépés)" value={dur} min={1} max={100} onChange={setDur} />
      <div className="flex gap-2 mt-1">
        <button className="btn-amber flex-1 flex items-center justify-center gap-1.5"
          onClick={() => injectDisturbance(dCA, dTemp, dur)}
          disabled={distActive || (dCA === 0 && dTemp === 0)}>
          <Zap size={13} /> Inject
        </button>
        <button className="btn-ghost px-3" onClick={() => { setDCA(0); setDTemp(0) }}>
          Reset
        </button>
      </div>
    </Section>
  )
}

// ─── Kalman-erősítés sáv ──────────────────────────────────────────────────────
function KalmanGainBar({ label, value, color }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex justify-between items-center">
        <span className="text-xs text-slate-400">{label}</span>
        <span className={`text-xs font-bold tabular-nums ${color}`}>{pct}%</span>
      </div>
      <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${color.replace('text-', 'bg-')}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ─── Zaj & becslés panel ──────────────────────────────────────────────────────
function NoiseEstimationPanel({ sim }) {
  const {
    noiseSigma, updateNoise,
    sensorFaultActive, injectSensorFault, clearSensorFault,
    kalmanGain,
  } = sim
  const [biasCA,   setBiasCA]   = useState(0)
  const [biasTemp, setBiasTemp] = useState(0)
  const noiseOn = noiseSigma > 0

  return (
    <Section title="Noise & Estimation" icon={Radio} defaultOpen={false}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-400">Mérési zaj (σ)</span>
        <button
          onClick={() => updateNoise(noiseOn ? 0 : 1.0)}
          className={`text-xs px-2 py-0.5 rounded font-semibold transition-colors ${
            noiseOn ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                    : 'bg-slate-700 text-slate-400 border border-slate-600'
          }`}
        >
          {noiseOn ? 'ON' : 'OFF'}
        </button>
      </div>
      <Slider
        label="σ (skálázott)" value={noiseSigma} min={0} max={5} step={0.1}
        unit="" color={noiseOn ? 'text-amber-400' : 'text-slate-500'}
        onChange={updateNoise}
      />
      <div className="text-xs text-slate-600 -mt-1">
        σ=1 → CA:±0.02 mol/L, T:±2 K
      </div>

      <div className="border-t border-slate-700 pt-2 flex flex-col gap-1.5">
        <div className="flex items-center gap-1 text-xs text-slate-500 font-semibold mb-0.5">
          <Info size={10} /> Kalman-erősítés (mérési bizalom)
        </div>
        <KalmanGainBar label="K[CA]" value={kalmanGain[0]} color="text-blue-400" />
        <KalmanGainBar label="K[T]"  value={kalmanGain[1]} color="text-orange-400" />
        <div className="text-xs text-slate-600">0%=modell, 100%=szenzor</div>
      </div>

      <div className="border-t border-slate-700 pt-2 flex flex-col gap-2">
        <div className="flex items-center gap-1 text-xs text-slate-400 font-semibold">
          <AlertTriangle size={11} className="text-red-400" /> Szenzor-hiba (offset)
        </div>
        {sensorFaultActive && (
          <div className="text-xs text-red-400 font-bold animate-pulse flex items-center gap-1">
            <AlertTriangle size={11} /> Szenzor-offset aktív!
          </div>
        )}
        <Slider
          label="CA offset" value={biasCA} min={-0.1} max={0.1} step={0.005}
          unit="mol/L" color="text-red-400" onChange={setBiasCA}
        />
        <Slider
          label="T offset" value={biasTemp} min={-15} max={15} step={0.5}
          unit="K" color="text-red-400" onChange={setBiasTemp}
        />
        <div className="flex gap-2">
          <button
            className="btn-amber flex-1 flex items-center justify-center gap-1.5 text-xs"
            onClick={() => injectSensorFault(biasCA, biasTemp)}
            disabled={sensorFaultActive || (biasCA === 0 && biasTemp === 0)}
          >
            <AlertTriangle size={12} /> Inject
          </button>
          <button
            className="btn-ghost px-3 text-xs"
            onClick={() => { clearSensorFault(); setBiasCA(0); setBiasTemp(0) }}
            disabled={!sensorFaultActive}
          >
            Clear
          </button>
        </div>
      </div>
    </Section>
  )
}

// ─── Kényszersértés badge-ek ──────────────────────────────────────────────────
function ViolationBadges({ violations }) {
  const entries = Object.values(violations)
  if (!entries.length) return null
  return (
    <div className="card border-red-500/60 bg-red-950/30 p-3">
      <p className="section-title text-red-400 border-red-800">Kényszersértések</p>
      <div className="flex flex-col gap-1.5">
        {entries.map((v, i) => (
          <div key={i} className="flex justify-between text-xs">
            <span className="text-red-300 font-semibold">{v.variable}</span>
            <span className="text-red-400 tabular-nums">
              {v.type === 'rate_limit'
                ? `Δu=${v.value?.toFixed(2)} > ${v.limit}`
                : `${v.type}: ${v.value?.toFixed(3)} / ${v.limit}`}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Gazdasági üzemmód panel ──────────────────────────────────────────────────
function EconomicPanel({ sim }) {
  const { economicCfg, updateEconomicCfg } = sim
  const set = (key, val) => updateEconomicCfg({ ...economicCfg, [key]: val })

  return (
    <Section title="Economic Mode" icon={DollarSign} defaultOpen={false}
             accent={economicCfg.economicMode}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-400">Üzemmód</span>
        <button
          onClick={() => set('economicMode', !economicCfg.economicMode)}
          className={`text-xs px-3 py-0.5 rounded font-semibold transition-colors ${
            economicCfg.economicMode
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
              : 'bg-slate-700 text-slate-400 border border-slate-600'
          }`}
        >
          {economicCfg.economicMode ? 'ECONOMIC' : 'TRACKING'}
        </button>
      </div>

      {economicCfg.economicMode && (
        <div className="text-xs text-emerald-400/80 -mt-1 mb-1 flex items-center gap-1">
          <Info size={10} /> Setpoint tracking kikapcsolva
        </div>
      )}

      <div className="flex flex-col gap-0.5 text-xs text-slate-600 -mt-1 mb-2 font-mono">
        <span>Profit = P·(F·CA) − F·(F·CAin) − E·f(Tc)</span>
      </div>

      <Slider
        label="Product Price (P)" value={economicCfg.productPrice}
        min={1} max={30} step={0.5} unit="$/mol" color="text-emerald-400"
        onChange={v => set('productPrice', v)}
      />
      <Slider
        label="Feedstock Cost (F)" value={economicCfg.feedstockCost}
        min={0.1} max={10} step={0.1} unit="$/mol" color="text-amber-400"
        onChange={v => set('feedstockCost', v)}
      />
      <Slider
        label="Energy Cost (E)" value={economicCfg.energyCost}
        min={0} max={5} step={0.1} unit="$/K²" color="text-red-400"
        onChange={v => set('energyCost', v)}
      />
      <div className="text-xs text-slate-600 -mt-1">
        f(Tc) = (300−Tc)² × 0.001 · E
      </div>
    </Section>
  )
}

// ─── ESD gomb ─────────────────────────────────────────────────────────────────
function EsdButton({ sim }) {
  const { esdActive, esd, esdClear, downloadCsv } = sim
  return (
    <div className="flex flex-col gap-2 mt-auto">
      {/* CSV Export */}
      <button
        className="btn-ghost flex items-center justify-center gap-2 text-xs py-1.5"
        onClick={downloadCsv}
      >
        <Download size={12} /> Export CSV
      </button>

      {/* ESD gomb */}
      {!esdActive ? (
        <button
          className="flex items-center justify-center gap-2 text-sm font-bold
                     py-3 rounded-lg border-2 border-red-600 bg-red-950/40
                     text-red-400 hover:bg-red-900/50 hover:border-red-500
                     transition-all duration-150 active:scale-95"
          onClick={esd}
        >
          <ShieldAlert size={16} />
          EMERGENCY SHUTDOWN
        </button>
      ) : (
        <div className="flex flex-col gap-1.5">
          <div className="text-xs text-center text-red-400 font-bold animate-pulse
                          flex items-center justify-center gap-1">
            <ShieldAlert size={12} /> ESD AKTÍV — F_min, Tc_min
          </div>
          <button
            className="flex items-center justify-center gap-2 text-xs font-semibold
                       py-2 rounded-lg border border-amber-500/60 bg-amber-950/40
                       text-amber-400 hover:bg-amber-900/50 transition-all duration-150"
            onClick={esdClear}
          >
            ESD törlése — NMPC visszaállítása
          </button>
        </div>
      )}
    </div>
  )
}

// ─── System Identification panel ─────────────────────────────────────────────
const SYSID_TOOLTIP = `System Identification — PRBS + ARX

A valóságban az ipari MPC-nek nincs analitikus
modellje. A mérnökök PRBS gerjesztő jelet
injektálnak, és ARX modellt illesztenek adatból.

PRBS (Pseudo-Random Binary Sequence):
  Maximális hosszúságú LFSR sorozat. Két ortogonális
  sorozat F-hez és Tc-hez (seed_offset meggátolja
  a korrelációt → reguláris regressziós mátrix).

ARX fit (Least Squares):
  y(k) = Σaᵢ·y(k-i) + Σbⱼ·u(k-j)
  Fit% = 100·(1 − ||y−ŷ|| / ||y−ȳ||)  [MATLAB NRMSE]

Modell hot-swap:
  Az azonosított A_d, B_d mátrixok felváltják a
  Jacobian-linearizált modellt a LINEAR MPC-ben.
  Így hasonlítható: Jacobian vs ARX-ID vs True.`

function FitBar({ pct, color }) {
  return (
    <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${Math.max(0, Math.min(100, pct ?? 0))}%` }}
      />
    </div>
  )
}

function SysIdPanel({ sim }) {
  const {
    running, controllerType,
    sysidActive, sysidProgress, sysidResult, sysidIdentified,
    useIdentifiedModel, linearModelSource,
    startSysId, updateUseIdentified,
  } = sim

  const [nBits,       setNBits]       = useState(7)
  const [clockPeriod, setClockPeriod] = useState(3)
  const [ampF,        setAmpF]        = useState(10)
  const [ampTc,       setAmpTc]       = useState(5)
  const [na,          setNa]          = useState(2)
  const [nb,          setNb]          = useState(2)
  const [showTip,     setShowTip]     = useState(false)

  const period      = (Math.pow(2, nBits) - 1) * clockPeriod
  const durationMin = (period / 60).toFixed(1)

  const isLinear     = controllerType === 'LINEAR'
  const canRun       = running && !sysidActive
  const progressPct  = sysidProgress.total > 0
    ? Math.round((sysidProgress.step / sysidProgress.total) * 100)
    : 0

  const handleRun = () => {
    startSysId({ n_bits: nBits, clock_period: clockPeriod, amp_F: ampF, amp_Tc: ampTc, na, nb })
  }

  return (
    <div className={`card p-3 border-2 transition-colors duration-300 ${
      sysidActive ? 'border-amber-500/60 bg-amber-950/15' :
      sysidIdentified ? 'border-emerald-500/40 bg-emerald-950/10' :
      'border-slate-700'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-300">
          <ScanLine size={13} className="text-slate-400" />
          System ID
        </div>
        <div className="flex items-center gap-1.5">
          {sysidActive && (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-bold
                             bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse">
              ⚡ FUTÓ
            </span>
          )}
          <button
            title="Hogyan működik a System ID?"
            onClick={() => setShowTip(v => !v)}
            className="text-slate-500 hover:text-slate-300 transition-colors"
          >
            <Info size={13} />
          </button>
        </div>
      </div>

      {/* PRBS config */}
      <div className="flex flex-col gap-2">
        <div className="flex gap-2">
          <div className="flex-1">
            <span className="text-[10px] text-slate-500 block mb-0.5">PRBS bits</span>
            <select
              value={nBits}
              onChange={e => setNBits(+e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded text-xs px-1.5 py-1 text-slate-300"
            >
              {[5, 6, 7, 8].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div className="flex-1">
            <span className="text-[10px] text-slate-500 block mb-0.5">Clock (lépés)</span>
            <input
              type="number" min={1} max={10} value={clockPeriod}
              onChange={e => setClockPeriod(+e.target.value)}
              className="input-num w-full text-right py-1 text-xs"
            />
          </div>
        </div>

        <div className="text-[10px] text-slate-500 -mt-1">
          Teszt hossza: {period} lépés ≈ {durationMin} perc
        </div>

        <Slider label="Amp F" value={ampF} min={1} max={30} step={1}
          unit="L/min" color="text-cyan-400" onChange={setAmpF} />
        <Slider label="Amp Tc" value={ampTc} min={1} max={15} step={1}
          unit="K" color="text-violet-400" onChange={setAmpTc} />

        <div className="flex gap-2">
          <div className="flex-1">
            <span className="text-[10px] text-slate-500 block mb-0.5">ARX na</span>
            <input
              type="number" min={1} max={4} value={na}
              onChange={e => setNa(+e.target.value)}
              className="input-num w-full text-right py-1 text-xs"
            />
          </div>
          <div className="flex-1">
            <span className="text-[10px] text-slate-500 block mb-0.5">ARX nb</span>
            <input
              type="number" min={1} max={4} value={nb}
              onChange={e => setNb(+e.target.value)}
              className="input-num w-full text-right py-1 text-xs"
            />
          </div>
        </div>
      </div>

      {/* Progress bar */}
      {sysidActive && (
        <div className="mt-2">
          <div className="flex justify-between text-[10px] text-amber-400 mb-0.5">
            <span>PRBS gyűjtés…</span>
            <span>{progressPct}%</span>
          </div>
          <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-amber-500 rounded-full transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Run button */}
      <button
        className={`w-full mt-3 py-2 rounded-lg text-xs font-bold transition-colors duration-150 flex items-center justify-center gap-1.5 ${
          canRun
            ? 'bg-amber-600 text-white hover:bg-amber-500 active:scale-95'
            : 'bg-slate-700 text-slate-500 cursor-not-allowed'
        }`}
        onClick={handleRun}
        disabled={!canRun}
        title={!running ? 'Indítsd el a szimulációt először' : sysidActive ? 'Teszt folyamatban…' : ''}
      >
        <ScanLine size={13} />
        {sysidActive ? 'Teszt futó…' : 'Run ID Test'}
      </button>

      {!running && (
        <div className="text-[10px] text-slate-600 text-center mt-1">
          Indítsd el a szimulációt a teszthez
        </div>
      )}

      {/* Fit results */}
      {sysidResult && (
        <div className="mt-3 border-t border-slate-700 pt-2 flex flex-col gap-1.5">
          <p className="text-xs text-slate-500 font-semibold">
            ARX Fit (na={sysidResult.na}, nb={sysidResult.nb})
          </p>
          <div>
            <div className="flex justify-between text-xs tabular-nums mb-0.5">
              <span className="text-slate-400">CA fit</span>
              <span className={`font-bold ${sysidResult.fit_pct_ca > 80 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {sysidResult.fit_pct_ca?.toFixed(1)}%
              </span>
            </div>
            <FitBar pct={sysidResult.fit_pct_ca} color={sysidResult.fit_pct_ca > 80 ? 'bg-emerald-500' : 'bg-amber-500'} />
          </div>
          <div>
            <div className="flex justify-between text-xs tabular-nums mb-0.5">
              <span className="text-slate-400">T fit</span>
              <span className={`font-bold ${sysidResult.fit_pct_t > 80 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {sysidResult.fit_pct_t?.toFixed(1)}%
              </span>
            </div>
            <FitBar pct={sysidResult.fit_pct_t} color={sysidResult.fit_pct_t > 80 ? 'bg-emerald-500' : 'bg-amber-500'} />
          </div>
        </div>
      )}

      {/* Use Identified Model toggle */}
      {sysidIdentified && (
        <div className="mt-3 border-t border-slate-700 pt-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-slate-400">Use Identified Model</span>
            {!isLinear && (
              <span className="text-[10px] text-slate-600" title="Csak LINEAR módban aktív">LINEAR only</span>
            )}
          </div>
          <div className={`flex rounded-lg overflow-hidden border text-xs font-semibold ${
            isLinear ? 'border-slate-700' : 'border-slate-800 opacity-50'
          }`}>
            <button
              onClick={() => isLinear && updateUseIdentified(false)}
              className={`flex-1 py-1.5 transition-colors duration-150 ${
                !useIdentifiedModel
                  ? 'bg-slate-500 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              Jacobian
            </button>
            <button
              onClick={() => isLinear && updateUseIdentified(true)}
              className={`flex-1 py-1.5 transition-colors duration-150 ${
                useIdentifiedModel
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              ARX-ID
            </button>
          </div>
          {useIdentifiedModel && (
            <p className="text-[10px] text-emerald-400 mt-1">
              Forrás: {linearModelSource}
            </p>
          )}
        </div>
      )}

      {/* Educational tooltip */}
      {showTip && (
        <pre className="mt-2 text-[10px] text-slate-400 bg-slate-900 rounded p-2
                        whitespace-pre-wrap leading-relaxed border border-slate-700">
          {SYSID_TOOLTIP}
        </pre>
      )}
    </div>
  )
}

// ─── Fő Sidebar export ────────────────────────────────────────────────────────
export function Sidebar({ sim }) {
  return (
    <aside className="w-64 shrink-0 flex flex-col gap-3 p-3 overflow-y-auto
                      bg-slate-950 border-r border-slate-800">
      <ReactorModePanel sim={sim} />
      <ControllerSwitchPanel sim={sim} />
      <EstimatorPanel sim={sim} />
      <FeedforwardPanel sim={sim} />
      <SysIdPanel sim={sim} />
      <SetpointPanel sim={sim} />
      {sim.reactorMode === 'SERIES' && <Setpoint2Panel sim={sim} />}
      <MpcConfigPanel sim={sim} />
      <EconomicPanel sim={sim} />
      <DisturbancePanel sim={sim} />
      <NoiseEstimationPanel sim={sim} />
      <ViolationBadges violations={sim.violations} />

      <EsdButton sim={sim} />

      {sim.modelInfo && (
        <div className="card p-3 mt-auto">
          <p className="section-title">Nemlineáris CSTR</p>
          <div className="text-xs text-slate-500 flex flex-col gap-0.5">
            <span className="font-mono text-slate-600">k(T)=k₀·exp(−Ea/RT)</span>
            <span className="font-mono text-slate-600 text-[10px]">
              k₀={sim.modelInfo.k0_str ?? '7.2e10/60 s⁻¹'}
            </span>
            <span className="text-slate-600">SS instabil (nyílt kör)</span>
            {sim.modelInfo.states?.map(s => (
              <span key={s.index} className="text-slate-600">
                x{s.index}: {s.name.split(' ')[0]} SS={s.ss} [{s.unit}]
              </span>
            ))}
          </div>
        </div>
      )}
    </aside>
  )
}
