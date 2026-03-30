/**
 * Robustness Section — Step 4
 * Rank Stability Plot, Robustness Curve (sensitivity analysis), and Adversarial grading.
 * Directly addresses professor's feedback.
 */
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, Legend, ErrorBar, LineChart, Line
} from 'recharts'

const COLORS = ['#6366f1', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#f97316', '#f43f5e']

function RankStabilityPlot({ confidenceIntervals }) {
  // Show top 15 nodes' mean rank with CI range
  const data = confidenceIntervals.slice(0, 15).map((ci, i) => ({
    node: `Node ${ci.node}`,
    mean_rank: ci.mean_rank,
    std_rank: ci.std_rank,
    ci_low: ci.ci_low,
    ci_high: ci.ci_high,
    range: ci.rank_range,
    color: ci.std_rank < 2 ? '#10b981' : ci.std_rank < 5 ? '#f59e0b' : '#f43f5e',
  }))

  return (
    <div className="chart-container">
      <h4>📊 Rank Stability Plot (Bootstrap Confidence Intervals)</h4>
      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Each bar shows a node's <strong>mean rank</strong> across {'>'}30 bootstrap iterations.
        The whisker (colored segment) shows the <strong>rank standard deviation</strong>.
        Green = highly stable (σ {'<'} 2), Yellow = moderate, Red = unstable.
      </p>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={data} layout="vertical" margin={{ left: 70 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Mean Rank (lower = more critical)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} />
          <YAxis type="category" dataKey="node" tick={{ fill: '#e8ecf4', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }}
            formatter={(v, name) => {
              if (name === 'mean_rank') return [`${v.toFixed(1)}`, 'Mean Rank']
              if (name === 'std_rank') return [`±${v.toFixed(2)}`, 'Rank Std Dev']
              return [v, name]
            }}
          />
          <Bar dataKey="mean_rank" radius={[0, 4, 4, 0]}>
            {data.map((d, i) => <Cell key={i} fill={d.color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function RobustnessCurve({ centralityImpact }) {
  const data = centralityImpact.map(ci => ({
    centrality: ci.removed_centrality.charAt(0).toUpperCase() + ci.removed_centrality.slice(1),
    impact: ci.impact,
    overlap: ci.top_10_overlap,
  })).sort((a, b) => b.impact - a.impact)

  return (
    <div className="chart-container">
      <h4>🔬 Robustness Curve: Centrality Removal Impact</h4>
      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        This chart shows the <strong>impact on rankings</strong> when each centrality metric is removed from
        the framework. Higher impact = the metric contributes significantly unique information.
        If all bars are low, the framework is robust; if one bar is very high, the ranking depends heavily on that metric.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="centrality" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Impact (% ranking change)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
          <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} formatter={(v) => [`${v.toFixed(1)}%`]} />
          <Bar dataKey="impact" radius={[4, 4, 0, 0]} minPointSize={4}>
            {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function TopKStabilityHeatmap({ topkStability }) {
  // Pivot and display as bar groups
  const metrics = [...new Set(topkStability.map(t => t.metric))]
  const ks = [...new Set(topkStability.map(t => t.k))].sort((a, b) => a - b)

  const data = metrics.map(m => {
    const row = { metric: m.charAt(0).toUpperCase() + m.slice(1) }
    topkStability.filter(t => t.metric === m).forEach(t => {
      row[`k=${t.k}`] = t.overlap
    })
    return row
  })

  return (
    <div className="chart-container">
      <h4>📈 Top-k Stability: TOPSIS vs Single Metrics at Different k</h4>
      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Each grouped bar shows the overlap (%) between TOPSIS top-k and a single metric's top-k
        at various values of k. Consistent overlap across k values means rankings are stable.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} unit="%" />
          <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} />
          <Legend />
          {ks.map((k, i) => (
            <Bar key={k} dataKey={`k=${k}`} fill={COLORS[i % COLORS.length]} radius={[2, 2, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function AdversarialGrade({ adversarial }) {
  const gradeColors = { A: 'var(--accent-emerald)', B: 'var(--accent-amber)', C: 'var(--accent-rose)', D: 'var(--accent-rose)' }
  return (
    <div className="card" style={{ textAlign: 'center' }}>
      <h4 style={{ marginBottom: '0.5rem' }}>🛡️ Adversarial Robustness Grade</h4>
      <div style={{ fontSize: '4rem', fontWeight: 900, color: gradeColors[adversarial.overall_grade] || 'var(--text-primary)' }}>
        {adversarial.overall_grade}
      </div>
      <p style={{ fontSize: '0.85rem' }}>Overall Vulnerability: {adversarial.overall_vulnerability?.toFixed(0)}%</p>

      <div style={{ marginTop: '1rem', textAlign: 'left' }}>
        <h4 style={{ marginBottom: '0.5rem', fontSize: '0.9rem' }}>Per-Node Robustness</h4>
        {adversarial.node_robustness?.map((nr, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.3rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
            <span>Node {nr.node}</span>
            <span className={`insight-tag ${nr.robustness_grade === 'A' ? 'winner' : nr.robustness_grade === 'B' ? 'warn' : 'danger'}`}>
              Grade {nr.robustness_grade} ({nr.vulnerability_score?.toFixed(0)}%)
            </span>
          </div>
        ))}
      </div>

      {adversarial.recommendations?.length > 0 && (
        <div style={{ marginTop: '1rem', textAlign: 'left' }}>
          <h4 style={{ marginBottom: '0.5rem', fontSize: '0.9rem' }}>Recommendations</h4>
          {adversarial.recommendations.map((r, i) => (
            <p key={i} style={{ fontSize: '0.82rem', marginBottom: '0.3rem' }}>• {r}</p>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Robustness({ data, loading, onRun }) {
  return (
    <section className="section animate-in">
      <div className="section-header">
        <div className="step-badge">Step 7 / 8</div>
        <h2>🛡️ Robustness & Stability Analysis</h2>
        <p>
          Verify that CRITIC-TOPSIS rankings are stable under perturbation, bootstrap resampling,
          and adversarial attacks. This section provides the <strong>Rank Stability Plot</strong> and <strong>Robustness Curve</strong>.
        </p>
      </div>

      <div className="theory-panel">
        <h4>📐 Theoretical Interpretation — Why Robustness Matters</h4>
        <p>
          A ranking method is only useful if it's <strong>stable</strong>: small changes to the network (edge noise,
          measurement error) should not drastically change the top critical nodes. We evaluate this through:
        </p>
        <ul style={{ paddingLeft: '1.2rem', marginTop: '0.5rem' }}>
          <li><strong>Bootstrap resampling</strong>: Repeatedly sample 80% of edges, recompute rankings, and measure rank variance.</li>
          <li><strong>Centrality removal</strong>: Remove each metric one-at-a-time and measure ranking change (sensitivity analysis).</li>
          <li><strong>Adversarial attacks</strong>: Add/remove strategic edges or fake nodes to test if rankings can be manipulated.</li>
        </ul>
      </div>

      {!data && (
        <div style={{ textAlign: 'center', padding: 'var(--space-2xl)' }}>
          <button className="btn btn-primary" onClick={onRun} disabled={loading}>
            {loading ? <><span className="spinner" /> Running robustness analysis (this takes ~30s)…</> : '🚀 Run Full Robustness Analysis'}
          </button>
        </div>
      )}

      {data && (
        <div className="animate-in">
          {/* Stability Metrics Summary */}
          <div className="grid-3" style={{ marginBottom: 'var(--space-xl)' }}>
            <div className="metric-card">
              <div className="metric-value">{data.uncertainty?.stability_metrics?.mean_rank_std?.toFixed(2)}</div>
              <div className="metric-label">Avg Rank Std Dev</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{data.uncertainty?.stability_metrics?.stable_nodes}</div>
              <div className="metric-label">Stable Nodes (σ{'<'}2)</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{data.uncertainty?.stability_metrics?.unstable_nodes}</div>
              <div className="metric-label">Unstable Nodes (σ≥5)</div>
            </div>
          </div>

          {/* Rank Stability Plot */}
          <RankStabilityPlot confidenceIntervals={data.uncertainty.confidence_intervals} />

          <div style={{ height: 'var(--space-xl)' }} />

          {/* Robustness Curve */}
          <RobustnessCurve centralityImpact={data.sensitivity.centrality_impact} />

          <div style={{ height: 'var(--space-xl)' }} />

          {/* Top-k Stability */}
          <TopKStabilityHeatmap topkStability={data.sensitivity.top_k_stability} />

          <div style={{ height: 'var(--space-xl)' }} />

          {/* Adversarial */}
          <div className="grid-2">
            <AdversarialGrade adversarial={data.adversarial} />
            <div className="card">
              <h4 style={{ marginBottom: '0.5rem' }}>🎯 High-Confidence Critical Nodes</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                Nodes with {'>'} 90% probability of being in the top-10 across all bootstrap iterations:
              </p>
              {data.uncertainty?.high_confidence_critical?.length > 0 ? (
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {data.uncertainty.high_confidence_critical.map((node, i) => (
                    <span key={i} className="insight-tag winner" style={{ padding: '0.4rem 0.8rem', fontSize: '0.9rem' }}>
                      Node {node}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="insight-tag warn">No nodes {'>'} 90% confident in top-10</span>
              )}

              <div style={{ marginTop: '1.5rem' }}>
                <h4 style={{ marginBottom: '0.5rem', fontSize: '0.9rem' }}>📊 Top-10 Probabilities</h4>
                {data.uncertainty?.top_k_probabilities?.slice(0, 10).map((p, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem' }}>
                    <span style={{ width: 60, fontSize: '0.82rem', color: 'var(--text-muted)' }}>Node {p.node}</span>
                    <div className="progress-bar-track" style={{ flex: 1 }}>
                      <div className="progress-bar-fill" style={{
                        width: `${(p.prob_top_k * 100)}%`,
                        background: p.prob_top_k >= 0.9 ? 'var(--accent-emerald)' : p.prob_top_k >= 0.5 ? 'var(--accent-amber)' : 'var(--accent-rose)',
                      }} />
                    </div>
                    <span className="mono" style={{ fontSize: '0.78rem', width: 45, textAlign: 'right' }}>
                      {(p.prob_top_k * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="theory-panel" style={{ marginTop: 'var(--space-lg)' }}>
            <h4>📐 Interpretation</h4>
            <p>
              The rank stability plot confirms which critical nodes are <strong>consistently identified</strong>
              regardless of edge perturbation. Nodes with low σ (green bars) are robustly critical —
              their importance is an intrinsic property of the network topology, not an artifact of specific edges.
              The robustness curve shows which centrality metrics contribute the most unique information;
              removing them causes the largest ranking disruption.
            </p>
          </div>
        </div>
      )}
    </section>
  )
}
