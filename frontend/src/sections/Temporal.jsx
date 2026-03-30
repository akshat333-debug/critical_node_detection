import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts'

const COLORS = ['#6366f1', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#f97316', '#f43f5e']

function WeightEvolutionChart({ timeseries, metrics }) {
  if (!timeseries || timeseries.length === 0) return null

  return (
    <div className="chart-container">
      <h4>📈 CRITIC Weight Evolution Over Time</h4>
      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Observe how the relative importance of different centralities shifts as the network evolves.
      </p>
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={timeseries}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Snapshot (Time)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 'auto']} label={{ value: 'Weight', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
          <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} formatter={(v) => v.toFixed(3)} />
          <Legend />
          {metrics.map((m, i) => (
            <Line
              key={m}
              type="monotone"
              dataKey={m}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              name={m.charAt(0).toUpperCase() + m.slice(1)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function WeightComparisonChart({ comparison }) {
  if (!comparison) return null

  const data = Object.keys(comparison).map(m => ({
    metric: m.charAt(0).toUpperCase() + m.slice(1),
    static: comparison[m].static,
    adaptive: comparison[m].adaptive,
  }))

  return (
    <div className="chart-container">
      <h4>⚖️ Static vs Adaptive Weights</h4>
      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Comparing standard CRITIC weights on the current snapshot vs exponentially-decayed adaptive weights across all historical snapshots.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} formatter={(v) => v.toFixed(3)} />
          <Legend />
          <Bar dataKey="static" name="Static Result" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
          <Bar dataKey="adaptive" name="Adaptive (Decay)" fill="#10b981" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Temporal({ data, loading, onRun }) {
  const hasData = !!data

  return (
    <section className="section animate-in">
      <div className="section-header">
        <div className="step-badge">Step 4 / 8</div>
        <h2>⏳ Temporal & Evolving Networks</h2>
        <p>
          Real-world networks are rarely static; they evolve over time. This section models network snapshots, analyzes the "drift" of CRITIC weights, and identifies emerging critical nodes through exponential decay.
        </p>
      </div>

      <div className="theory-panel">
        <h4>📐 Theoretical Interpretation</h4>
        <p>
          Consider a communication network receiving new links and losing old ones. A centrality that provided high discriminating power at <code>t = 0</code> may become homogenous at <code>t = 5</code>. 
        </p>
        <p style={{ marginTop: '0.5rem' }}>
          We simulate this by evolving the graph across multiple snapshots. We apply <strong>Adaptive Weight Recalculation</strong> via an exponential decay function: recent snapshots contribute more to the final CRITIC weights, preventing old stale data from masking emerging critical hubs.
        </p>
      </div>

      {!hasData && (
        <div style={{ textAlign: 'center', padding: 'var(--space-2xl)' }}>
          <button className="btn btn-primary" onClick={onRun} disabled={loading}>
            {loading ? <><span className="spinner" /> Simulating network evolution...</> : '🚀 Run Temporal Analysis'}
          </button>
        </div>
      )}

      {hasData && (
        <div className="animate-in">
          {/* Main Chart */}
          <WeightEvolutionChart timeseries={data.weight_evolution?.timeseries} metrics={data.weight_evolution?.metrics} />

          {/* Drift Alerts */}
          {data.weight_evolution?.drift && (
            <div className="card" style={{ marginTop: 'var(--space-lg)' }}>
              <h4 style={{ marginBottom: '0.8rem' }}>⚠️ Weight Drift Alerts</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                Metrics whose influence has significantly changed between the first and last snapshot.
              </p>
              {Object.entries(data.weight_evolution.drift)
                .filter(([_, d]) => d.significant)
                .map(([m, d], i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', padding: '0.5rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span className={`insight-tag ${d.direction === 'increased' ? 'winner' : 'danger'}`}>
                      {d.direction === 'increased' ? '↗️ Increased' : '↘️ Decreased'}
                    </span>
                    <strong style={{ textTransform: 'capitalize' }}>{m}</strong>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      Shifted by {(d.absolute_change * 100).toFixed(1)}% (from {d.initial.toFixed(2)} to {d.final.toFixed(2)})
                    </span>
                  </div>
                ))}
              {Object.values(data.weight_evolution.drift).filter(d => d.significant).length === 0 && (
                <p style={{ fontSize: '0.9rem', color: 'var(--accent-emerald)' }}>✅ Network is stable. No significant weight drifts detected (all changes {'<'} 3%).</p>
              )}
            </div>
          )}

          <div style={{ height: 'var(--space-xl)' }} />

          {/* Comparison */}
          <WeightComparisonChart comparison={data.weight_comparison} />

          <div style={{ height: 'var(--space-xl)' }} />

          {/* Node Trajectories */}
          <div className="grid-2">
            <div className="card">
              <h4 style={{ color: 'var(--accent-emerald)', marginBottom: '1rem' }}>⭐ Rising Stars</h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.8rem' }}>
                Nodes not currently in the top set, but their rank trajectory is sharply improving.
              </p>
              {data.rising_stars?.length > 0 ? data.rising_stars.map((n, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
                  <span>Node {n.node}</span>
                  <span style={{ color: 'var(--accent-emerald)' }}>Trend: {n.trend} (Avg Rank: {n.avg_rank.toFixed(1)})</span>
                </div>
              )) : <span className="insight-tag info">No significant emerging nodes detected.</span>}
            </div>

            <div className="card">
              <h4 style={{ color: 'var(--accent-rose)', marginBottom: '1rem' }}>📉 Declining Importance</h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.8rem' }}>
                Nodes that were previously highly ranked but are consistently losing structural importance.
              </p>
              {data.declining?.length > 0 ? data.declining.map((n, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
                  <span>Node {n.node}</span>
                  <span style={{ color: 'var(--accent-rose)' }}>Trend: +{n.trend} (Avg Rank: {n.avg_rank.toFixed(1)})</span>
                </div>
              )) : <span className="insight-tag info">No significant declining nodes detected.</span>}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
