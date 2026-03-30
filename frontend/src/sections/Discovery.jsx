/**
 * Discovery Section — Step 2
 * Network visualization, centrality analysis, CRITIC weights, TOPSIS rankings.
 */
import { useState, useMemo, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, ScatterChart, Scatter, CartesianGrid, Legend
} from 'recharts'
import NetworkGraph from '../components/NetworkGraph'

const COLORS = ['#6366f1', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#f97316', '#f43f5e']

function SectionHeader({ step, title, description }) {
  return (
    <div className="section-header">
      <div className="step-badge">Step {step}</div>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  )
}

function MetricCard({ value, label }) {
  return (
    <div className="metric-card">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  )
}

function RankingsTable({ rankings, topK = 10 }) {
  const rows = rankings.slice(0, topK)
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Node</th>
          <th>TOPSIS Score</th>
          <th>Dist to Best</th>
          <th>Dist to Worst</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={r.node ?? r.index ?? i}>
            <td>
              <span className={`rank-badge ${i < 3 ? `rank-${i + 1}` : ''}`}>
                {r.rank}
              </span>
            </td>
            <td style={{ fontWeight: 600 }}>{r.node ?? r.index}</td>
            <td className="mono">{(r.closeness ?? 0).toFixed(4)}</td>
            <td className="mono" style={{ color: 'var(--text-muted)' }}>{(r.dist_to_best ?? 0).toFixed(4)}</td>
            <td className="mono" style={{ color: 'var(--text-muted)' }}>{(r.dist_to_worst ?? 0).toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function CritiqWeightsChart({ weights }) {
  const data = Object.entries(weights)
    .map(([k, v]) => ({ name: k.charAt(0).toUpperCase() + k.slice(1), value: v }))
    .sort((a, b) => b.value - a.value)

  return (
    <div className="grid-2">
      <div className="chart-container">
        <h4>CRITIC Weights (Bar)</h4>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis type="category" dataKey="name" tick={{ fill: '#e8ecf4', fontSize: 12 }} />
            <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-container">
        <h4>Weight Distribution</h4>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
              {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function OverlapChart({ comparison }) {
  const data = comparison.map(c => ({
    name: c.metric.charAt(0).toUpperCase() + c.metric.slice(1),
    overlap: c.overlap,
  }))
  return (
    <div className="chart-container">
      <h4>Top-10 Overlap: Single Metrics vs TOPSIS</h4>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} unit="%" />
          <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} />
          <Bar dataKey="overlap" fill="#6366f1" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function ExplanationCards({ explanations }) {
  if (!explanations?.length) return null
  return (
    <div>
      <h3 style={{ marginBottom: '1rem' }}>💡 Why Are These Nodes Critical?</h3>
      {explanations.map((exp, i) => (
        <div key={i} className="explanation-card">
          <div className="node-badge" style={{
            background: i === 0 ? 'var(--gradient-danger)' : i === 1 ? 'var(--gradient-accent)' : 'rgba(100,116,139,0.3)',
            color: 'white',
          }}>
            Node {exp.node} — Rank #{exp.rank}
          </div>
          <p style={{ fontSize: '0.88rem', marginBottom: '0.6rem' }}>{exp.criticality} (Top {exp.percentile?.toFixed(0)}%)</p>
          {exp.top_factors?.map((f, j) => (
            <div key={j} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', width: 90, textTransform: 'capitalize' }}>{f.metric}</span>
              <div className="progress-bar-track" style={{ flex: 1 }}>
                <div className="progress-bar-fill" style={{
                  width: `${f.percentile}%`,
                  background: COLORS[j % COLORS.length],
                }} />
              </div>
              <span className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-muted)', width: 55, textAlign: 'right' }}>
                w={f.weight?.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

export default function Discovery({ data, loading }) {
  const [colorMetric, setColorMetric] = useState('topsis_score')

  if (loading || !data) {
    return (
      <section className="section">
        <div className="loading-overlay">
          <div className="spinner" />
          <p>Computing centralities and CRITIC-TOPSIS rankings…</p>
        </div>
      </section>
    )
  }

  const { network_info, weights, rankings, critical_nodes, excluded_metrics, comparison, explanations, graph } = data

  return (
    <section className="section animate-in">
      <SectionHeader step="2 / 8" title="🔍 Network Discovery" description="Load the network, compute centrality measures, determine CRITIC weights, and produce TOPSIS rankings." />

      {/* ── Network Stats ───────────────────────────────────── */}
      <div className="grid-4" style={{ marginBottom: 'var(--space-xl)' }}>
        <MetricCard value={network_info.nodes} label="Nodes" />
        <MetricCard value={network_info.edges} label="Edges" />
        <MetricCard value={network_info.density?.toFixed(3)} label="Density" />
        <MetricCard value={network_info.avg_clustering?.toFixed(3)} label="Avg Clustering" />
      </div>

      {graph && (
        <div style={{ marginBottom: 'var(--space-xl)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
            <h3>🕸️ Network Structure</h3>
            <div className="select-wrapper">
              <select value={colorMetric} onChange={e => setColorMetric(e.target.value)}>
                <option value="topsis_score">Color by: TOPSIS Score</option>
                <option value="degree">Color by: Degree Centrality</option>
                <option value="betweenness">Color by: Betweenness Centrality</option>
                <option value="closeness">Color by: Closeness Centrality</option>
                <option value="pagerank">Color by: PageRank</option>
                <option value="kshell">Color by: K-Shell</option>
              </select>
            </div>
          </div>
          <NetworkGraph graphData={graph} criticalNodes={critical_nodes} colorMetric={colorMetric} />
        </div>
      )}

      {excluded_metrics?.length > 0 && (
        <div className="theory-panel" style={{ marginBottom: 'var(--space-lg)' }}>
          <h4>🧠 Adaptive Centrality Selection</h4>
          <p>Low-variance metrics excluded: <strong>{excluded_metrics.join(', ')}</strong>. These provide negligible discriminating power for this network topology.</p>
        </div>
      )}

      {/* ── CRITIC Weights ──────────────────────────────────── */}
      <div style={{ marginBottom: 'var(--space-xl)' }}>
        <h3 style={{ marginBottom: 'var(--space-md)' }}>⚖️ CRITIC Weight Analysis</h3>
        <div className="theory-panel">
          <h4>📐 Theoretical Interpretation</h4>
          <p>
            CRITIC assigns weights objectively: <code>w_j = C_j / ΣC_j</code> where 
            <code>C_j = σ_j × Σ(1 − r_jk)</code>. A centrality with <strong>high standard deviation</strong> (it discriminates well) 
            and <strong>low correlation</strong> with others (it provides unique info) receives higher weight.
            Unlike subjective weighting, this is entirely data-driven.
          </p>
        </div>
        <CritiqWeightsChart weights={weights} />
      </div>

      {/* ── Rankings Table ──────────────────────────────────── */}
      <div style={{ marginBottom: 'var(--space-xl)' }}>
        <h3 style={{ marginBottom: 'var(--space-md)' }}>🏆 TOPSIS Rankings</h3>
        <div className="theory-panel">
          <h4>📐 Theoretical Interpretation</h4>
          <p>
            Each node's score <code>C_i = D⁻ / (D⁺ + D⁻)</code> measures how close it is to the 
            ideal best (max on all weighted criteria) relative to the ideal worst. 
            A score near 1.0 means the node excels across <em>all</em> important centralities simultaneously.
          </p>
        </div>
        <div className="card" style={{ overflow: 'auto' }}>
          <RankingsTable rankings={rankings} />
        </div>
      </div>

      {/* ── Overlap Comparison ──────────────────────────────── */}
      <div style={{ marginBottom: 'var(--space-xl)' }}>
        <h3 style={{ marginBottom: 'var(--space-md)' }}>📊 Ranking Overlap Analysis</h3>
        <p style={{ marginBottom: 'var(--space-md)' }}>
          How much do single-metric top-10 lists agree with the multi-attribute TOPSIS top-10?
          Lower overlap confirms that <strong>no single metric alone captures the full picture</strong>.
        </p>
        <OverlapChart comparison={comparison} />
      </div>

      {/* ── Explainable AI ──────────────────────────────────── */}
      <ExplanationCards explanations={explanations} />
    </section>
  )
}
