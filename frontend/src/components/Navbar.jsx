/**
 * Sticky top navigation bar with section links and network selector.
 */
export default function Navbar({ sections, active, onNavigate, completed, network, networks, onNetworkChange }) {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <div className="nav-brand">
          <span style={{ fontSize: '1.5rem' }}>🔍</span>
          <span>Critical Node Detection</span>
        </div>

        <ul className="nav-links">
          {sections.map(s => (
            <li key={s.id}>
              <button
                className={`nav-link ${active === s.id ? 'active' : ''}`}
                onClick={() => onNavigate(s.id)}
              >
                {completed.has(s.id) && active !== s.id ? '✓ ' : ''}{s.label.split(' ').slice(1).join(' ')}
              </button>
            </li>
          ))}
        </ul>

        <div className="network-selector">
          <select value={network} onChange={e => onNetworkChange(e.target.value)}>
            {networks.map(n => (
              <option key={n.key} value={n.key}>{n.label}</option>
            ))}
          </select>
        </div>
      </div>
    </nav>
  )
}
