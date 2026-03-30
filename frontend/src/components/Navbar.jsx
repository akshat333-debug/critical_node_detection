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
