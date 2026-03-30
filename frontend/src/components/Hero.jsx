/**
 * Hero / Introduction section.
 * Explains WHY CRITIC-TOPSIS is unique and distinct vs naive single-metric approaches.
 */
export default function Hero({ onStart, loading }) {
  return (
    <section className="section animate-in">
      <div className="hero">
        <h1>Critical Node Detection</h1>
        <p className="subtitle">
          A multi-attribute decision-making framework that combines <strong>7 centrality measures</strong>,
          objectively weights them with <strong>CRITIC</strong>, and ranks nodes via <strong>TOPSIS</strong> —
          producing superior results to any single metric alone.
        </p>
        <button className="btn btn-primary" onClick={onStart} disabled={loading}>
          {loading ? <><span className="spinner" /> Analyzing network…</> : '🚀 Begin Analysis'}
        </button>
      </div>

      {/* ── Uniqueness & Distinctness Callout ────────────────────────────── */}
      <div className="uniqueness-banner animate-in" style={{ animationDelay: '0.15s' }}>
        <h3>🌟 What Makes This Approach Unique?</h3>
        <p style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>
          Traditional methods rely on a <strong>single centrality metric</strong> (e.g. degree, betweenness).
          But different metrics capture <em>different aspects</em> of importance — a bridge node may have low degree yet 
          high betweenness. Our framework solves this fundamental limitation:
        </p>

        <div className="distinction-grid">
          <div className="distinction-card weakness">
            <h4 style={{ color: 'var(--accent-rose)', marginBottom: '0.5rem' }}>❌ Single-Metric Ranking</h4>
            <ul style={{ paddingLeft: '1.2rem', color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.8' }}>
              <li>Captures only <strong>one aspect</strong> of node importance</li>
              <li>Degree ignores bridge roles; betweenness ignores hub status</li>
              <li>No objective way to decide <em>which</em> metric to use</li>
              <li>Ranking is metric-dependent and biased</li>
            </ul>
          </div>
          <div className="distinction-vs">VS</div>
          <div className="distinction-card strength">
            <h4 style={{ color: 'var(--accent-emerald)', marginBottom: '0.5rem' }}>✅ CRITIC-TOPSIS Framework</h4>
            <ul style={{ paddingLeft: '1.2rem', color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.8' }}>
              <li>Fuses <strong>7 complementary centralities</strong> simultaneously</li>
              <li>CRITIC assigns weights <strong>objectively</strong> from data variance & correlation</li>
              <li>TOPSIS finds the multi-dimensional compromise closest to the ideal</li>
              <li>Proven superior via targeted-attack experimental evaluation</li>
            </ul>
          </div>
        </div>
      </div>

      {/* ── Theory: Why Multi-Attribute? ──────────────────────────────────── */}
      <div className="theory-panel animate-in" style={{ animationDelay: '0.3s' }}>
        <h4>📐 Theoretical Interpretation</h4>
        <p>
          Consider a network where node <em>A</em> has the highest degree and node <em>B</em> has the highest
          betweenness centrality. A naive approach forces you to choose one. But <strong>CRITIC</strong> (CRiteria 
          Importance Through Intercriteria Correlation) calculates the <em>information content</em> of each metric:
        </p>
        <p style={{ margin: '0.7rem 0' }}>
          <code>C_j = σ_j × Σ(1 − r_jk)</code> — weight grows with high standard deviation (discriminating power)
          and low correlation with other criteria (unique information).
        </p>
        <p>
          Then <strong>TOPSIS</strong> ranks each node by its Euclidean proximity to the <em>ideal best</em> solution
          (max on all weighted criteria) and distance from the <em>ideal worst</em>.
          The result: <code>Closeness = D⁻ / (D⁺ + D⁻)</code>, a score ∈ [0, 1] where higher is better.
        </p>
      </div>

      {/* ── Methodology Steps ────────────────────────────────────────────── */}
      <div className="card animate-in" style={{ animationDelay: '0.45s' }}>
        <h3 style={{ marginBottom: '1rem' }}>📋 Methodology Pipeline</h3>
        <div className="grid-3" style={{ textAlign: 'center' }}>
          {[
            { num: 1, title: 'Compute Centralities', desc: 'Degree, Betweenness, Closeness, Eigenvector, PageRank, K-Shell, H-Index', color: 'var(--accent-blue)' },
            { num: 2, title: 'CRITIC Weighting', desc: 'Objectively determine weights using standard deviation × conflict (1 − correlation)', color: 'var(--accent-violet)' },
            { num: 3, title: 'TOPSIS Ranking', desc: 'Rank nodes by closeness to ideal best / distance from ideal worst', color: 'var(--accent-emerald)' },
          ].map(step => (
            <div key={step.num} className="metric-card">
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem', color: step.color, fontWeight: 800 }}>{step.num}</div>
              <div style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-heading)', marginBottom: '0.3rem' }}>{step.title}</div>
              <p style={{ fontSize: '0.82rem', lineHeight: '1.6' }}>{step.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
