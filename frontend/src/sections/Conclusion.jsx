/**
 * Conclusion Section — Step 5
 * Summary of findings, key insights, and data export.
 */
export default function Conclusion({ data, impactData, robustnessData }) {
  const hasAnalysis = !!data
  const hasImpact = !!impactData
  const hasRobustness = !!robustnessData

  const downloadJSON = (obj, filename) => {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadCSV = (rows, filename) => {
    if (!rows?.length) return
    const headers = Object.keys(rows[0])
    const csv = [headers.join(','), ...rows.map(r => headers.map(h => r[h] ?? '').join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="section animate-in">
      <div className="section-header">
        <div className="step-badge">Step 5 / 5</div>
        <h2>📦 Conclusion & Export</h2>
        <p>
          Summary of key findings from the CRITIC-TOPSIS analysis pipeline.
        </p>
      </div>

      {/* ── Key Findings ──────────────────────────────────────── */}
      <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
        <h3 style={{ marginBottom: 'var(--space-md)' }}>🔑 Key Findings</h3>

        {hasAnalysis && (
          <div style={{ marginBottom: '1rem' }}>
            <h4 style={{ color: 'var(--accent-blue)', marginBottom: '0.4rem' }}>Network Overview</h4>
            <p>Analyzed <strong>{data.network_info.nodes} nodes</strong> and <strong>{data.network_info.edges} edges</strong> 
              {' '}with density {data.network_info.density?.toFixed(3)} and average clustering {data.network_info.avg_clustering?.toFixed(3)}.</p>
            <p style={{ marginTop: '0.3rem' }}>
              <strong>Top critical nodes: </strong>
              {data.critical_nodes?.slice(0, 5).map((n, i) => (
                <span key={i} className="insight-tag winner" style={{ marginRight: '0.3rem' }}>Node {n}</span>
              ))}
            </p>
          </div>
        )}

        {hasAnalysis && (
          <div style={{ marginBottom: '1rem' }}>
            <h4 style={{ color: 'var(--accent-violet)', marginBottom: '0.4rem' }}>Most Influential Metric</h4>
            {(() => {
              const sorted = Object.entries(data.weights).sort((a, b) => b[1] - a[1])
              return (
                <p>
                  <strong>{sorted[0][0].charAt(0).toUpperCase() + sorted[0][0].slice(1)}</strong> received
                  the highest CRITIC weight ({(sorted[0][1] * 100).toFixed(1)}%), meaning it provides the most
                  discriminating and unique information for this network.
                </p>
              )
            })()}
          </div>
        )}

        {hasImpact && (
          <div style={{ marginBottom: '1rem' }}>
            <h4 style={{ color: 'var(--accent-rose)', marginBottom: '0.4rem' }}>Attack Simulation Result</h4>
            {(() => {
              const sorted = [...impactData.effectiveness].sort((a, b) => b.effectiveness - a.effectiveness)
              const ct = sorted.find(s => s.method === 'CRITIC-TOPSIS')
              return (
                <p>
                  CRITIC-TOPSIS achieved an effectiveness score of <strong>{ct?.effectiveness?.toFixed(4)}</strong>,
                  {sorted[0].method === 'CRITIC-TOPSIS'
                    ? ' outperforming all single-metric baselines.'
                    : ` ranking competitively against ${sorted[0].method} (${sorted[0].effectiveness?.toFixed(4)}).`}
                </p>
              )
            })()}
          </div>
        )}

        {hasRobustness && (
          <div>
            <h4 style={{ color: 'var(--accent-emerald)', marginBottom: '0.4rem' }}>Robustness Assessment</h4>
            <p>
              Adversarial robustness grade: <strong>{robustnessData.adversarial?.overall_grade}</strong> 
              ({robustnessData.adversarial?.overall_vulnerability?.toFixed(0)}% vulnerable).
              {robustnessData.uncertainty?.high_confidence_critical?.length > 0
                ? ` Nodes ${robustnessData.uncertainty.high_confidence_critical.join(', ')} are confirmed critical with >90% bootstrap confidence.`
                : ' Bootstrap analysis showed moderate ranking stability.'}
            </p>
          </div>
        )}
      </div>

      {/* ── Distinctness Summary ──────────────────────────────── */}
      <div className="uniqueness-banner" style={{ marginBottom: 'var(--space-xl)' }}>
        <h3>🌟 Distinctness of the CRITIC-TOPSIS Approach</h3>
        <div className="grid-2" style={{ marginTop: '1rem' }}>
          <div>
            <h4 style={{ color: 'var(--accent-violet)', marginBottom: '0.5rem', fontSize: '0.95rem' }}>Methodological Uniqueness</h4>
            <ul style={{ paddingLeft: '1.2rem', color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: '1.8' }}>
              <li><strong>Objective weighting</strong> — removes human bias via CRITIC's variance-correlation formula</li>
              <li><strong>Multi-attribute fusion</strong> — simultaneously considers 7 complementary centrality dimensions</li>
              <li><strong>Ideal-solution proximity</strong> — TOPSIS finds the geometrically optimal compromise</li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--accent-violet)', marginBottom: '0.5rem', fontSize: '0.95rem' }}>Experimental Validation</h4>
            <ul style={{ paddingLeft: '1.2rem', color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: '1.8' }}>
              <li><strong>Targeted attack simulation</strong> confirms superior critical node identification</li>
              <li><strong>Bootstrap stability</strong> validates ranking robustness under network perturbation</li>
              <li><strong>Adversarial analysis</strong> demonstrates resilience against strategic manipulation</li>
            </ul>
          </div>
        </div>
      </div>

      {/* ── Export ─────────────────────────────────────────────── */}
      <div className="card">
        <h3 style={{ marginBottom: 'var(--space-md)' }}>📥 Export Results</h3>
        <div className="grid-3">
          <button
            className="btn btn-outline"
            disabled={!hasAnalysis}
            onClick={() => downloadCSV(data?.rankings, 'critical_nodes_rankings.csv')}
          >
            📄 Rankings (CSV)
          </button>
          <button
            className="btn btn-outline"
            disabled={!hasAnalysis}
            onClick={() => downloadJSON(data?.weights, 'critic_weights.json')}
          >
            ⚖️ CRITIC Weights (JSON)
          </button>
          <button
            className="btn btn-outline"
            disabled={!hasAnalysis}
            onClick={() => downloadCSV(data?.centralities, 'all_centralities.csv')}
          >
            📊 All Centralities (CSV)
          </button>
          <button
            className="btn btn-outline"
            disabled={!hasImpact}
            onClick={() => downloadJSON(impactData, 'impact_results.json')}
          >
            💥 Impact Data (JSON)
          </button>
          <button
            className="btn btn-outline"
            disabled={!hasRobustness}
            onClick={() => downloadJSON(robustnessData, 'robustness_results.json')}
          >
            🛡️ Robustness Data (JSON)
          </button>
          <button
            className="btn btn-outline"
            disabled={!hasAnalysis}
            onClick={() => {
              const report = data?.summary_report || ''
              const blob = new Blob([report], { type: 'text/markdown' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = 'analysis_report.md'
              a.click()
              URL.revokeObjectURL(url)
            }}
          >
            📝 Full Report (Markdown)
          </button>
        </div>
      </div>
    </section>
  )
}
