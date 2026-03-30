import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts'

export default function Scale({ data, loading, onRun }) {
  const [model, setModel] = useState('erdos_renyi')
  const [maxNodes, setMaxNodes] = useState(1000)

  const hasData = data && Array.isArray(data) && data.length > 0

  // Process data for charts
  const getTimesData = () => {
    if (!hasData) return []
    return data.map(d => ({
      nodes: d.nodes,
      edges: d.edges,
      'Centralities (s)': d.centralities_time,
      'CRITIC (s)': d.critic_time,
      'TOPSIS (s)': d.topsis_time,
      'Total Base Process (s)': d.total_time
    }))
  }

  const getSlowestStepsData = () => {
    if (!hasData) return []
    const last = data[data.length - 1]
    return [
      { step: 'Centralities', time: last.centralities_time, fill: '#3b82f6' },
      { step: 'CRITIC Weights', time: last.critic_time, fill: '#fbbf24' },
      { step: 'TOPSIS Rank', time: last.topsis_time, fill: '#10b981' },
    ]
  }

  return (
    <section className="section animate-in">
      <div className="section-header">
        <div className="step-badge">Step 6 / 8</div>
        <h2>⏱️ Scalability & Big O Constraints</h2>
        <p>
          Demonstrate the framework's performance on increasingly large generated networks, 
          identifying computational bottlenecks in the CRITIC-TOPSIS pipeline.
        </p>
      </div>

      <div className="theory-panel">
        <h4>📐 Theoretical Interpretation</h4>
        <p>
          Network analysis is computationally expensive: standard betweenness centrality operates in <code>O(V * E)</code> time. 
          As network size grows to thousands of nodes, calculating all 7 centralities simultaneously becomes the dominant bottleneck. 
        </p>
        <p style={{ marginTop: '0.5rem' }}>
          This section allows us to empirically verify the polynomial time constraints of combining Multi-Attribute Decision Making (MADM) with complex graph topologies.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 'var(--space-2xl)' }}>
        <h4 style={{ marginBottom: '1rem' }}>Benchmark Parameters</h4>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>Graph Model</label>
            <select 
              value={model} 
              onChange={e => setModel(e.target.value)}
              disabled={loading}
              style={{ 
                padding: '0.6rem', borderRadius: '8px', 
                background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)',
                color: 'var(--text-primary)', minWidth: '200px'
              }}
            >
              <option value="erdos_renyi">Erdős-Rényi (Random)</option>
              <option value="barabasi_albert">Barabási-Albert (Scale-Free)</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>Max Nodes</label>
            <select 
              value={maxNodes} 
              onChange={e => setMaxNodes(Number(e.target.value))}
              disabled={loading}
              style={{ 
                padding: '0.6rem', borderRadius: '8px', 
                background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)',
                color: 'var(--text-primary)', minWidth: '150px'
              }}
            >
              <option value={500}>500 Nodes (Fast)</option>
              <option value={1000}>1,000 Nodes (Med)</option>
              <option value={2000}>2,000 Nodes (Slow)</option>
              <option value={3000}>3,000 Nodes (Very Slow)</option>
            </select>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'flex-end', height: '100%', paddingBottom: '0.2rem' }}>
            <button 
              className="btn btn-primary" 
              onClick={() => onRun({ model, max_size: maxNodes })} 
              disabled={loading}
              style={{ marginTop: 'auto' }}
            >
              {loading ? <><span className="spinner" /> Benchmarking O(N)...</> : 'Run Scale Benchmark'}
            </button>
          </div>
        </div>
      </div>

      {hasData && (
        <div className="animate-in grid-2">
          {/* Main Line Chart */}
          <div className="card" style={{ gridColumn: '1 / -1' }}>
            <h4 style={{ marginBottom: '0.5rem' }}>Computation Time vs Network Size</h4>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              Observe the polynomial scaling. As Nodes (N) naturally increases Edges (E), time grows non-linearly.
            </p>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={getTimesData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                <XAxis dataKey="nodes" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Network Size (Nodes)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Time (Seconds)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                <Tooltip 
                  contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} 
                  labelFormatter={(v) => `Nodes: ${v}`}
                  formatter={(v, name) => [v.toFixed(3) + 's', name]} 
                />
                <Legend />
                <Line type="monotone" dataKey="Total Base Process (s)" stroke="#8b5cf6" strokeWidth={3} />
                <Line type="monotone" dataKey="Centralities (s)" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Breakdown for largest network */}
          <div className="card">
            <h4 style={{ marginBottom: '1rem' }}>Bottleneck Breakdown at N={data[data.length - 1].nodes}</h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={getSlowestStepsData()} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis dataKey="step" type="category" tick={{ fill: '#e8ecf4', fontSize: 12 }} width={120} />
                <Tooltip 
                  contentStyle={{ background: '#1a1f35', border: '1px solid rgba(148,163,184,0.1)', borderRadius: 8, color: '#e8ecf4' }} 
                  formatter={(v) => v.toFixed(3) + 's'} 
                />
                <Bar dataKey="time" radius={[0, 4, 4, 0]} minPointSize={5}>
                  {getSlowestStepsData().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h4 style={{ color: 'var(--accent-blue)', marginBottom: '1rem' }}>Performance Summary</h4>
            <ul style={{ paddingLeft: '1.2rem', color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.8' }}>
              <li style={{ marginBottom: '0.5rem' }}>
                Largest network tested: <strong>{data[data.length - 1].nodes} nodes</strong> / {data[data.length - 1].edges} edges
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                Total computation time: <strong>{data[data.length - 1].total_time.toFixed(2)} seconds</strong>
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                <strong>Centralities Phase</strong> consumes roughly {((data[data.length - 1].centralities_time / Math.max(data[data.length - 1].total_time, 0.001)) * 100).toFixed(0)}% of the total runtime.
              </li>
              <li style={{ marginBottom: '0.5rem' }}>
                <strong>CRITIC + TOPSIS</strong> are extremely efficient O(M×N) operations on the computed matrices, resolving in fractions of a second.
              </li>
            </ul>
          </div>
        </div>
      )}
    </section>
  )
}
