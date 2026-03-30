import { useState, useEffect, useCallback } from 'react'
import './index.css'
import { analyze, getImpact, getRobustness } from './api'
import Hero from './components/Hero'
import Navbar from './components/Navbar'
import Discovery from './sections/Discovery'
import Impact from './sections/Impact'
import Robustness from './sections/Robustness'
import Conclusion from './sections/Conclusion'

const SECTIONS = [
  { id: 'intro',      label: '① Introduction',    icon: '🎯' },
  { id: 'discovery',  label: '② Discovery',       icon: '🔍' },
  { id: 'impact',     label: '③ Impact',          icon: '💥' },
  { id: 'robustness', label: '④ Robustness',      icon: '🛡️' },
  { id: 'conclusion', label: '⑤ Conclusion',      icon: '📦' },
]

const NETWORKS = [
  { key: 'karate',        label: 'Karate Club (34 nodes)' },
  { key: 'les_miserables', label: 'Les Misérables (77 nodes)' },
  { key: 'florentine',    label: 'Florentine Families (15 nodes)' },
  { key: 'dolphins',      label: 'Dolphins (62 nodes)' },
]

export default function App() {
  const [activeSection, setActiveSection] = useState('intro')
  const [network, setNetwork] = useState('karate')
  const [analysisData, setAnalysisData] = useState(null)
  const [impactData, setImpactData] = useState(null)
  const [robustnessData, setRobustnessData] = useState(null)
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
    getImpact({ network })
      .then(data => { setImpactData(data); markCompleted('impact') })
      .catch(err => console.error('Impact failed:', err))
      .finally(() => setLoading(prev => ({ ...prev, impact: false })))
  }, [network, markCompleted])

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
            loading={loading.impact}
            onRun={runImpact}
            analysisData={analysisData}
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
        <p style={{ marginTop: '0.3rem' }}>Pipeline: Introduction → Discovery → Impact → Robustness → Conclusion</p>
      </footer>
    </>
  )
}
