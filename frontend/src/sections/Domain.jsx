import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

export default function Domain({ data, domains, loading, onRun, onFetchDomains }) {
  const [selectedDomain, setSelectedDomain] = useState('social')
  const hasData = !!data

  useEffect(() => {
    // Only fetch domains list if we don't have it yet and the user visits this tab
    if (!domains || Object.keys(domains).length === 0) {
      if (typeof onFetchDomains === 'function') {
        onFetchDomains()
      }
    }
  }, [domains, onFetchDomains])

  // Process data for charts
  const getRadarData = () => {
    if (!hasData || !data.comparison) return []
    return Object.keys(data.comparison)
      .filter(k => k !== 'overall_overlap_fraction')
      .map(metric => ({
        metric: metric.charAt(0).toUpperCase() + metric.slice(1),
        CRITIC: data.comparison[metric]?.critic_weight || 0,
        Empirical: data.comparison[metric]?.domain_weight || 0,
      }))
  }

  return (
    <section className="section animate-in">
      <div className="section-header">
        <div className="step-badge">Step 5 / 8</div>
        <h2>🧠 Domain Intelligence</h2>
        <p>
          Compare data-driven CRITIC weights against empirical domain knowledge profiles.
          Certain network types inherently value specific structural properties over others.
        </p>
      </div>

      <div className="theory-panel">
        <h4>📐 Theoretical Interpretation</h4>
        <p>
          While CRITIC objectively extracts weights from the purely topological variance matrix (data-driven), we often have
          <strong> a priori domain knowledge</strong>. For instance, in power grids, node capacity (Betweenness) is fundamentally
          more critical than reachability (Closeness).
        </p>
        <p style={{ marginTop: '0.5rem' }}>
          This section allows you to contrast the data-driven weights with pre-defined empirical profiles (Biological, Transport, Social, etc.).
        </p>
      </div>

      <div className="card" style={{ marginBottom: 'var(--space-2xl)' }}>
        <h4 style={{ marginBottom: '1rem' }}>Select Domain Profile</h4>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <select 
            value={selectedDomain} 
            onChange={e => setSelectedDomain(e.target.value)}
            disabled={loading}
            style={{ 
              padding: '0.6rem', 
              borderRadius: '8px', 
              background: 'var(--bg-tertiary)', 
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
              minWidth: '200px'
            }}
          >
            {domains && Object.keys(domains).map(d => (
              <option key={d} value={d}>
                {domains[d].name || d.toUpperCase()}
              </option>
            ))}
            {(!domains || Object.keys(domains).length === 0) && (
              <>
                <option value="social">Social Networks</option>
                <option value="infrastructure">Infrastructure</option>
                <option value="biological">Biological Networks</option>
              </>
            )}
          </select>
          
          <button 
            className="btn btn-primary" 
            onClick={() => onRun(selectedDomain)} 
            disabled={loading}
          >
            {loading ? <><span className="spinner" /> Analyzing Domain...</> : 'Evaluate Domain Profile'}
          </button>
        </div>

        {domains && domains[selectedDomain] && (
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '1rem' }}>
            {domains[selectedDomain].description}
          </p>
        )}
      </div>

      {hasData && (
        <div className="animate-in grid-2">
          {/* Radar Chart comparing Weights */}
          <div className="card">
            <h4 style={{ marginBottom: '1rem' }}>Spider Chart: CRITIC vs Empirical</h4>
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart outerRadius={110} data={getRadarData()}>
                <PolarGrid stroke="rgba(148,163,184,0.1)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 0.45]} tick={false} axisLine={false} />
                <Tooltip contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} formatter={(v) => v.toFixed(3)} />
                <Legend />
                <Radar name="CRITIC (Data-Driven)" dataKey="CRITIC" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.4} />
                <Radar name="Empirical (Prior Info)" dataKey="Empirical" stroke="#10b981" fill="#10b981" fillOpacity={0.4} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h4 style={{ marginBottom: '1rem' }}>Rank Overlap Breakdown</h4>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              How much difference does the domain profile make compared to completely objective CRITIC?
            </p>
            
            <div style={{ marginBottom: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Top-10 Rank Overlap</span>
                <strong style={{ color: 'var(--accent-blue)' }}>
                  {(data.comparison?.overall_overlap_fraction * 100).toFixed(1) || 0}%
                </strong>
              </div>
              <div style={{ height: '8px', background: 'var(--bg-tertiary)', borderRadius: '4px', overflow: 'hidden' }}>
                <div 
                  style={{ 
                    height: '100%', 
                    background: 'var(--accent-blue)', 
                    width: `${(data.comparison?.overall_overlap_fraction * 100) || 0}%`,
                    transition: 'width 1s ease-in-out'
                  }} 
                />
              </div>
            </div>

            <div style={{ padding: '1rem', background: 'rgba(99, 102, 241, 0.05)', border: '1px solid rgba(99, 102, 241, 0.1)', borderRadius: '12px' }}>
              <h5 style={{ color: 'var(--accent-blue)', marginBottom: '0.5rem' }}>Key Differences</h5>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                {Object.keys(data.comparison || {})
                  .filter(m => m !== 'overall_overlap_fraction' && data.comparison[m]?.critic_weight != null && Math.abs((data.comparison[m].critic_weight || 0) - (data.comparison[m].domain_weight || 0)) > 0.05)
                  .map(m => (
                    <div key={m} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.3rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
                      <span style={{ textTransform: 'capitalize' }}>{m}</span>
                      <span>
                        CRITIC: {(data.comparison[m]?.critic_weight ?? 0).toFixed(2)} → Emp: {(data.comparison[m]?.domain_weight ?? 0).toFixed(2)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
            
          </div>
        </div>
      )}
    </section>
  )
}
