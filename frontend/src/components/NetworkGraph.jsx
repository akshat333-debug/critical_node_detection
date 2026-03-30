/**
 * Interactive Force-Directed Network Graph Visualization
 * Uses react-force-graph-2d for WebGL-accelerated rendering.
 * 
 * Features:
 * - Critical nodes highlighted with size + color + glow
 * - Hover tooltips with node metrics
 * - Color-by-metric selector
 * - Zoom/pan/drag
 */
import { useRef, useState, useCallback, useMemo, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

const METRIC_COLORS = {
  topsis_score: { high: '#f43f5e', low: '#1e293b', label: 'TOPSIS Score' },
  degree: { high: '#3b82f6', low: '#1e293b', label: 'Degree' },
  betweenness: { high: '#10b981', low: '#1e293b', label: 'Betweenness' },
  closeness: { high: '#f59e0b', low: '#1e293b', label: 'Closeness' },
  eigenvector: { high: '#8b5cf6', low: '#1e293b', label: 'Eigenvector' },
  pagerank: { high: '#06b6d4', low: '#1e293b', label: 'PageRank' },
  kshell: { high: '#f97316', low: '#1e293b', label: 'K-Shell' },
  hindex: { high: '#ec4899', low: '#1e293b', label: 'H-Index' },
}

function lerp(a, b, t) {
  return a + (b - a) * Math.max(0, Math.min(1, t))
}

function lerpColor(hex1, hex2, t) {
  const r1 = parseInt(hex1.slice(1, 3), 16), g1 = parseInt(hex1.slice(3, 5), 16), b1 = parseInt(hex1.slice(5, 7), 16)
  const r2 = parseInt(hex2.slice(1, 3), 16), g2 = parseInt(hex2.slice(3, 5), 16), b2 = parseInt(hex2.slice(5, 7), 16)
  const r = Math.round(lerp(r1, r2, t)), g = Math.round(lerp(g1, g2, t)), b = Math.round(lerp(b1, b2, t))
  return `rgb(${r},${g},${b})`
}

export default function NetworkGraph({ graphData, criticalNodes, colorMetric = 'topsis_score' }) {
  const fgRef = useRef()
  const [hoveredNode, setHoveredNode] = useState(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 500 })
  const containerRef = useRef()

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        setDimensions({ width: entry.contentRect.width, height: 500 })
      }
    })
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  // Compute metric ranges for normalization
  const metricRange = useMemo(() => {
    if (!graphData?.nodes?.length) return { min: 0, max: 1 }
    const vals = graphData.nodes.map(n => n[colorMetric] ?? 0).filter(v => isFinite(v))
    return { min: Math.min(...vals), max: Math.max(...vals) }
  }, [graphData, colorMetric])

  const criticalSet = useMemo(() => new Set(criticalNodes || []), [criticalNodes])

  // Custom node painter
  const paintNode = useCallback((node, ctx, globalScale) => {
    const isCritical = criticalSet.has(node.id)
    const metricVal = node[colorMetric] ?? 0
    const range = metricRange.max - metricRange.min
    const t = range > 0 ? (metricVal - metricRange.min) / range : 0.5

    const palette = METRIC_COLORS[colorMetric] || METRIC_COLORS.topsis_score
    const color = lerpColor(palette.low, palette.high, t)

    const baseSize = isCritical ? 6 : 3
    const size = baseSize / Math.sqrt(Math.max(globalScale, 0.3))

    // Glow for critical nodes
    if (isCritical) {
      ctx.beginPath()
      ctx.arc(node.x, node.y, size + 3, 0, 2 * Math.PI)
      ctx.fillStyle = `${palette.high}40`
      ctx.fill()
      
      ctx.beginPath()
      ctx.arc(node.x, node.y, size + 1.5, 0, 2 * Math.PI)
      ctx.fillStyle = `${palette.high}60`
      ctx.fill()
    }

    // Node circle
    ctx.beginPath()
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI)
    ctx.fillStyle = isCritical ? palette.high : color
    ctx.fill()

    // Border
    ctx.strokeStyle = isCritical ? '#ffffff' : 'rgba(148,163,184,0.3)'
    ctx.lineWidth = isCritical ? 1.5 / globalScale : 0.5 / globalScale
    ctx.stroke()

    // Label for critical nodes
    if (isCritical && globalScale > 0.6) {
      ctx.font = `${Math.max(10 / globalScale, 3)}px Inter, sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = '#e8ecf4'
      ctx.fillText(node.id, node.x, node.y + size + 2)
    }

    // Hover highlight
    if (hoveredNode === node.id) {
      ctx.beginPath()
      ctx.arc(node.x, node.y, size + 4, 0, 2 * Math.PI)
      ctx.strokeStyle = '#6366f1'
      ctx.lineWidth = 2 / globalScale
      ctx.stroke()
    }
  }, [colorMetric, metricRange, criticalSet, hoveredNode])

  const handleNodeHover = useCallback(node => {
    setHoveredNode(node?.id ?? null)
    if (containerRef.current) {
      containerRef.current.style.cursor = node ? 'pointer' : 'default'
    }
  }, [])

  const handleZoomToFit = useCallback(() => {
    fgRef.current?.zoomToFit(400, 40)
  }, [])

  if (!graphData?.nodes?.length) return null

  const palette = METRIC_COLORS[colorMetric] || METRIC_COLORS.topsis_score

  return (
    <div ref={containerRef} className="network-graph-container">
      {/* Controls */}
      <div className="graph-controls">
        <button className="graph-control-btn" onClick={handleZoomToFit} title="Fit to view">
          ⊞
        </button>
      </div>

      {/* Legend */}
      <div className="graph-legend">
        <div className="graph-legend-item">
          <span className="graph-legend-dot critical" /> Critical
        </div>
        <div className="graph-legend-item">
          <span className="graph-legend-dot regular" /> Regular
        </div>
        <div className="graph-legend-gradient">
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Low</span>
          <div className="graph-gradient-bar" style={{
            background: `linear-gradient(to right, ${palette.low}, ${palette.high})`
          }} />
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>High</span>
        </div>
      </div>

      {/* Tooltip */}
      {hoveredNode != null && (() => {
        const node = graphData.nodes.find(n => n.id === hoveredNode)
        if (!node) return null
        return (
          <div className="graph-tooltip">
            <div className="graph-tooltip-header">
              {node.is_critical && <span className="insight-tag winner" style={{ marginRight: '0.3rem', fontSize: '0.65rem' }}>Critical</span>}
              <strong>Node {node.id}</strong>
            </div>
            {node.rank != null && <div className="graph-tooltip-row">Rank: <strong>#{node.rank}</strong></div>}
            {node.topsis_score != null && <div className="graph-tooltip-row">TOPSIS: <strong>{node.topsis_score.toFixed(4)}</strong></div>}
            <div className="graph-tooltip-row">Degree: <strong>{node.degree}</strong></div>
            {node.betweenness != null && <div className="graph-tooltip-row">Betweenness: <strong>{node.betweenness.toFixed(4)}</strong></div>}
            {node.pagerank != null && <div className="graph-tooltip-row">PageRank: <strong>{node.pagerank.toFixed(4)}</strong></div>}
          </div>
        )
      })()}

      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width || 800}
        height={dimensions.height}
        backgroundColor="transparent"
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={(node, color, ctx) => {
          const size = criticalSet.has(node.id) ? 8 : 5
          ctx.fillStyle = color
          ctx.beginPath()
          ctx.arc(node.x, node.y, size, 0, 2 * Math.PI)
          ctx.fill()
        }}
        linkColor={() => 'rgba(148,163,184,0.12)'}
        linkWidth={0.5}
        onNodeHover={handleNodeHover}
        cooldownTicks={80}
        onEngineStop={() => fgRef.current?.zoomToFit(400, 40)}
        d3AlphaDecay={0.03}
        d3VelocityDecay={0.3}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        enableNodeDrag={true}
      />
    </div>
  )
}
