import { useState, useEffect, useCallback } from 'react'
import './index.css'
import { analyze, getImpact, getCascade, getRobustness, getTemporal, getDomain, getDomains, getScale } from './api'
import Hero from './components/Hero'
import Navbar from './components/Navbar'
import Discovery from './sections/Discovery'
import Impact from './sections/Impact'
import Robustness from './sections/Robustness'
import Temporal from './sections/Temporal'
import Domain from './sections/Domain'
import Scale from './sections/Scale'
import Conclusion from './sections/Conclusion'

const SECTIONS = [
  { id: 'intro',      label: '① Introduction',    icon: '🎯' },
  { id: 'discovery',  label: '② Discovery',       icon: '🔍' },
  { id: 'impact',     label: '③ Impact',          icon: '💥' },
  { id: 'temporal',   label: '④ Temporal',        icon: '⏳' },
  { id: 'domain',     label: '⑤ Domain Intel',    icon: '🧠' },
  { id: 'scale',      label: '⑥ Scale Profile',   icon: '⏱️' },
  { id: 'robustness', label: '⑦ Robustness',      icon: '🛡️' },
  { id: 'conclusion', label: '⑧ Conclusion',      icon: '📦' },
]

const NETWORKS = [
  { key: 'karate',        label: 'Karate Club (34 nodes)' },
  { key: 'les_miserables', label: 'Les Misérables (77 nodes)' },
  { key: 'florentine',    label: 'Florentine Families (15 nodes)' },
  { key: 'dolphins',      label: 'Dolphins (62 nodes)' },
  { key: 'usair',         label: 'US Air Infrastructure (332 nodes)' },
  { key: 'power_grid',    label: 'Western US Power Grid (4941 nodes)' },
]

export default function App() {
  const [activeSection, setActiveSection] = useState('intro')
  const [network, setNetwork] = useState('karate')
  const [analysisData, setAnalysisData] = useState(null)
  const [impactData, setImpactData] = useState(null)
  const [cascadeData, setCascadeData] = useState(null)
  const [temporalData, setTemporalData] = useState(null)
  const [domainData, setDomainData] = useState(null)
  const [scaleData, setScaleData] = useState(null)
  const [robustnessData, setRobustnessData] = useState(null)
  const [domainsList, setDomainsList] = useState(null)
  const [loading, setLoading] = useState({})
  const [completedSections, setCompletedSections] = useState(new Set())

  const markCompleted = useCallback((id) => {
    setCompletedSections(prev => new Set([...prev, id]))
  }, [])

  // Run initial analysis when network changes
  useEffect(() => {
    setLoading(prev => ({ ...prev, analyze: true }))
    setAnalysisData(null)
    setImpactData(null)
    setRobustnessData(null)
    setCompletedSections(new Set())

    analyze({ network, top_k: 10 })
      .then(data => {
        setAnalysisData(data)
        markCompleted('intro')
        markCompleted('discovery')
      })
      .catch(err => console.error('Analysis failed:', err))
      .finally(() => setLoading(prev => ({ ...prev, analyze: false })))
  }, [network, markCompleted])

  const runImpact = useCallback(() => {
    setLoading(prev => ({ ...prev, impact: true }))
    
    // Run both impact and cascade in parallel for this section
    Promise.all([
      getImpact({ network }),
      getCascade({ network, capacity_factor: 1.2, initial_fraction: 0.05 })
    ])
      .then(([impData, cascData]) => { 
        setImpactData(impData)
        setCascadeData(cascData)
        markCompleted('impact') 
      })
      .catch(err => console.error('Impact/Cascade failed:', err))
      .finally(() => setLoading(prev => ({ ...prev, impact: false })))
  }, [network, markCompleted])

  const runTemporal = useCallback(() => {
    setLoading(prev => ({ ...prev, temporal: true }))
    getTemporal({ network, n_snapshots: 5, volatility: 0.1, decay: 0.3 })
      .then(data => { setTemporalData(data); markCompleted('temporal') })
      .catch(err => console.error('Temporal failed:', err))
      .finally(() => setLoading(prev => ({ ...prev, temporal: false })))
  }, [network, markCompleted])

  const runDomain = useCallback((domainStr) => {
    setLoading(prev => ({ ...prev, domain: true }))
    getDomain({ network, domain: domainStr })
      .then(data => { setDomainData(data); markCompleted('domain') })
      .catch(err => console.error('Domain failed:', err))
      .finally(() => setLoading(prev => ({ ...prev, domain: false })))
  }, [network, markCompleted])

  const fetchDomains = useCallback(() => {
    getDomains()
      .then(data => { setDomainsList(data) })
      .catch(err => console.error('Domains fetch failed:', err))
  }, [])

  const runScale = useCallback((params) => {
    setLoading(prev => ({ ...prev, scale: true }))
    getScale(params)
      .then(data => { setScaleData(data); markCompleted('scale') })
      .catch(err => console.error('Scale failed:', err))
      .finally(() => setLoading(prev => ({ ...prev, scale: false })))
  }, [markCompleted])

  const runRobustness = useCallback(() => {
    setLoading(prev => ({ ...prev, robustness: true }))
    getRobustness({ network, n_bootstrap: 30, top_k: 10 })
      .then(data => { setRobustnessData(data); markCompleted('robustness') })
      .catch(err => console.error('Robustness failed:', err))
      .finally(() => setLoading(prev => ({ ...prev, robustness: false })))
  }, [network, markCompleted])

  return (
    <>
      <Navbar
        sections={SECTIONS}
        active={activeSection}
        onNavigate={setActiveSection}
        completed={completedSections}
        network={network}
        networks={NETWORKS}
        onNetworkChange={setNetwork}
      />

      <main className="app-container">
        {/* Pipeline Stepper */}
        <div className="pipeline-stepper">
          {SECTIONS.map((s, i) => (
            <div key={s.id} style={{ display: 'flex', alignItems: 'center' }}>
              <button
                className={`pipeline-step ${activeSection === s.id ? 'active' : ''} ${completedSections.has(s.id) && activeSection !== s.id ? 'completed' : ''}`}
                onClick={() => setActiveSection(s.id)}
              >
                {completedSections.has(s.id) && activeSection !== s.id ? '✓' : s.icon} {s.label.split(' ').slice(1).join(' ')}
              </button>
              {i < SECTIONS.length - 1 && <span className="pipeline-arrow">→</span>}
            </div>
          ))}
        </div>

        {/* Section: Introduction */}
        {activeSection === 'intro' && (
          <Hero onStart={() => setActiveSection('discovery')} loading={loading.analyze} />
        )}

        {/* Section: Discovery */}
        {activeSection === 'discovery' && (
          <Discovery data={analysisData} loading={loading.analyze} />
        )}

        {/* Section: Impact */}
        {activeSection === 'impact' && (
          <Impact
            data={impactData}
            cascadeData={cascadeData}
            loading={loading.impact}
            onRun={runImpact}
            analysisData={analysisData}
          />
        )}

        {/* Section: Temporal */}
        {activeSection === 'temporal' && (
          <Temporal
            data={temporalData}
            loading={loading.temporal}
            onRun={runTemporal}
          />
        )}

        {/* Section: Domain */}
        {activeSection === 'domain' && (
          <Domain
            data={domainData}
            domains={domainsList}
            loading={loading.domain}
            onRun={runDomain}
            onFetchDomains={fetchDomains}
          />
        )}

        {/* Section: Scale */}
        {activeSection === 'scale' && (
          <Scale
            data={scaleData}
            loading={loading.scale}
            onRun={runScale}
          />
        )}

        {/* Section: Robustness */}
        {activeSection === 'robustness' && (
          <Robustness
            data={robustnessData}
            loading={loading.robustness}
            onRun={runRobustness}
          />
        )}

        {/* Section: Conclusion */}
        {activeSection === 'conclusion' && (
          <Conclusion data={analysisData} impactData={impactData} robustnessData={robustnessData} />
        )}
      </main>

      <footer className="footer">
        <p>Critical Node Detection using CRITIC-TOPSIS Framework</p>
        <p style={{ marginTop: '0.3rem' }}>Pipeline: Introduction → Discovery → Impact → Temporal → Domain → Scale → Robustness → Conclusion</p>
      </footer>
    </>
  )
}
