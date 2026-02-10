"""
Critical Node Detection - Interactive Web Interface
====================================================
Story-driven 6-tab pipeline: Discovery → Impact → Robustness → Evolution → Domain → Scale/Export

Run with: streamlit run app.py
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import io
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from centralities import compute_all_centralities
from critic import compute_critic_weights, normalize_minmax, normalize_zscore
from topsis import topsis_rank, get_critical_nodes
from evaluation import (compare_attack_methods, compute_attack_effectiveness,
                        get_ranking_from_centrality, get_ranking_from_topsis)
from data_loading import (load_karate_club, load_les_miserables, 
                          load_florentine_families, load_dolphins,
                          load_power_grid, load_usair,
                          create_barabasi_albert, create_erdos_renyi,
                          get_network_info)
from cascading_failure import simulate_cascading_failure, compare_cascade_methods, cascade_over_fractions
from sensitivity_analysis import sensitivity_to_normalization, sensitivity_to_centrality_removal, sensitivity_to_top_k
from real_world_datasets import generate_social_network, generate_infrastructure_network, generate_biological_network, get_network_characteristics
from temporal_analysis import temporal_prediction_summary, temporal_adaptive_summary, generate_temporal_snapshots, analyze_temporal_rankings
from scalability import benchmark_scalability, benchmark_single
from explainable_ai import explain_node, explain_top_k, generate_summary_report
from uncertainty import full_uncertainty_analysis, bootstrap_rankings
from domain_weights import domain_aware_analysis, get_available_domains, detect_network_domain
from adversarial import full_adversarial_analysis, test_robustness

# Page config
st.set_page_config(
    page_title="Critical Node Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .pipeline-step {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border-left: 4px solid #667eea;
        padding: 0.6rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔍 Critical Node Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CRITIC-TOPSIS Multi-Attribute Framework — Story-Driven Analysis Pipeline</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Network Selection & Settings
# ============================================================================

st.sidebar.header("📊 Network Selection")

upload_mode = st.sidebar.radio(
    "Network source:",
    ["Built-in networks", "📤 Upload custom network"],
    help="Upload your own network as an edge list"
)

G = None

if upload_mode == "📤 Upload custom network":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Upload edge list file:**")
    st.sidebar.caption("Format: one edge per line, separated by space/comma/tab")
    st.sidebar.caption("Example: `1 2` or `nodeA,nodeB`")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose file", 
        type=['txt', 'csv', 'edgelist'],
        help="Upload edge list file"
    )
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            edges = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                for sep in [',', '\t', ' ', ';']:
                    parts = line.split(sep)
                    if len(parts) >= 2:
                        edges.append((parts[0].strip(), parts[1].strip()))
                        break
            
            if edges:
                G = nx.Graph()
                G.add_edges_from(edges)
                G.name = f"Custom ({len(G.nodes())} nodes)"
                st.sidebar.success(f"✅ Loaded {len(G.nodes())} nodes, {len(G.edges())} edges")
            else:
                st.sidebar.error("Could not parse edges from file")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    else:
        st.sidebar.info("👆 Upload a file to analyze your own network")
        G = load_karate_club()
else:
    network_option = st.sidebar.selectbox(
        "Choose a network:",
        ["Karate Club (34 nodes)", "Les Miserables (77 nodes)", 
         "Florentine Families (15 nodes)", "Dolphins (62 nodes)",
         "Power Grid (4941 nodes)", "USAir (332 nodes)",
         "Barabási-Albert (custom)", "Erdős-Rényi (custom)"]
    )
    
    @st.cache_data
    def load_network(option, ba_n=100, ba_m=3, er_n=100, er_p=0.1):
        if "Karate" in option:
            return load_karate_club()
        elif "Les Miserables" in option:
            return load_les_miserables()
        elif "Florentine" in option:
            return load_florentine_families()
        elif "Dolphins" in option:
            return load_dolphins()
        elif "Power Grid" in option:
            return load_power_grid()
        elif "USAir" in option:
            return load_usair()
        elif "Barabási" in option:
            return create_barabasi_albert(ba_n, ba_m)
        else:
            return create_erdos_renyi(er_n, er_p)
    
    if "Barabási" in network_option:
        ba_n = st.sidebar.slider("Number of nodes", 20, 2000, 100)
        ba_m = st.sidebar.slider("Edges per new node", 1, 10, 3)
        G = load_network(network_option, ba_n=ba_n, ba_m=ba_m)
    elif "Erdős" in network_option:
        er_n = st.sidebar.slider("Number of nodes", 20, 2000, 100)
        er_p = st.sidebar.slider("Edge probability", 0.01, 0.5, 0.1)
        G = load_network(network_option, er_n=er_n, er_p=er_p)
    else:
        G = load_network(network_option)

if G is None:
    G = load_karate_club()

# ============================================================================
# SETTINGS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Top-k critical nodes", 5, 20, 10)

normalization_method = st.sidebar.selectbox(
    "Normalization method:",
    ["Min-Max (default)", "Z-Score", "Log (for skewed data)"],
    help="Choose how to normalize centrality values before CRITIC"
)

use_adaptive = st.sidebar.checkbox(
    "🧠 Adaptive centrality selection",
    value=True,
    help="Automatically remove low-variance metrics (like k-shell=0)"
)

variance_threshold = 0.01
if use_adaptive:
    variance_threshold = st.sidebar.slider(
        "Variance threshold",
        0.001, 0.1, 0.01,
        help="Metrics with normalized variance below this are excluded"
    )

# ============================================================================
# COMPUTE ANALYSIS
# ============================================================================

info = get_network_info(G)

def normalize_log(df):
    """Log normalization for skewed distributions."""
    df_log = df.copy()
    for col in df.columns:
        df_log[col] = np.log1p(df[col])
    return normalize_minmax(df_log)

@st.cache_data
def compute_analysis(_G, norm_method, adaptive, var_thresh):
    df = compute_all_centralities(_G, verbose=False)
    
    excluded_metrics = []
    if adaptive:
        df_norm_check = normalize_minmax(df)
        variances = df_norm_check.var()
        low_var = variances[variances < var_thresh].index.tolist()
        if low_var and len(low_var) < len(df.columns):
            excluded_metrics = low_var
            df = df.drop(columns=low_var)
    
    weights, _ = compute_critic_weights(df, normalization=norm_method.split()[0].lower(), verbose=False)
    results, _ = topsis_rank(df, weights, verbose=False)
    
    return df, weights, results, excluded_metrics

norm_key = normalization_method.split()[0].lower()
if "log" in normalization_method.lower():
    norm_key = "minmax"

df_centrality, weights, topsis_results, excluded_metrics = compute_analysis(
    G, normalization_method, use_adaptive, variance_threshold
)
critical_nodes = get_critical_nodes(topsis_results, top_k)

if excluded_metrics:
    st.sidebar.warning(f"⚠️ Excluded low-variance: {', '.join(excluded_metrics)}")

# ============================================================================
# PIPELINE PROGRESS INDICATOR
# ============================================================================

TAB_NAMES = [
    "🔍 Discovery",
    "💥 Impact",
    "🛡️ Robustness",
    "⏰ Temporal",
    "🎯 Domain Intel",
    "📦 Scale & Export"
]

TAB_STORIES = [
    "Load your network and uncover raw importance signals",
    "Test what happens when critical nodes fail",
    "Verify if rankings hold under uncertainty and attacks",
    "Track how critical nodes change over time",
    "Apply expert knowledge for your network type",
    "Validate speed and download everything"
]

# Pipeline steps info
st.markdown("""
<div style="display:flex; justify-content:center; gap:0.3rem; margin-bottom:0.5rem; flex-wrap:wrap;">
    <span style="background:#667eea; color:white; padding:0.25rem 0.6rem; border-radius:1rem; font-size:0.75rem;">① Discovery</span>
    <span style="color:#667eea; font-size:0.75rem; padding-top:0.25rem;">→</span>
    <span style="background:#764ba2; color:white; padding:0.25rem 0.6rem; border-radius:1rem; font-size:0.75rem;">② Impact</span>
    <span style="color:#764ba2; font-size:0.75rem; padding-top:0.25rem;">→</span>
    <span style="background:#f5576c; color:white; padding:0.25rem 0.6rem; border-radius:1rem; font-size:0.75rem;">③ Robustness</span>
    <span style="color:#f5576c; font-size:0.75rem; padding-top:0.25rem;">→</span>
    <span style="background:#4facfe; color:white; padding:0.25rem 0.6rem; border-radius:1rem; font-size:0.75rem;">④ Temporal</span>
    <span style="color:#4facfe; font-size:0.75rem; padding-top:0.25rem;">→</span>
    <span style="background:#43e97b; color:white; padding:0.25rem 0.6rem; border-radius:1rem; font-size:0.75rem;">⑤ Domain</span>
    <span style="color:#43e97b; font-size:0.75rem; padding-top:0.25rem;">→</span>
    <span style="background:#fa709a; color:white; padding:0.25rem 0.6rem; border-radius:1rem; font-size:0.75rem;">⑥ Export</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 6 TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(TAB_NAMES)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: NETWORK DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="pipeline-step">📍 <b>Step 1 / 6 — Discovery</b>: Load your network and uncover raw importance signals.</div>', unsafe_allow_html=True)
    
    # ── Row 1: Network stats ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", info['nodes'])
    c2.metric("Edges", info['edges'])
    c3.metric("Density", f"{info['density']:.3f}")
    c4.metric("Avg Clustering", f"{info['avg_clustering']:.3f}")
    
    if excluded_metrics:
        st.info(f"🧠 **Adaptive mode** excluded low-variance metrics: {', '.join(excluded_metrics)}")
    
    # ── Row 2: Network viz + Top nodes ──
    st.markdown("---")
    col_viz, col_rank = st.columns([2, 1])
    
    with col_viz:
        st.subheader("🕸️ Network Visualization")
        pos = nx.spring_layout(G, k=2/np.sqrt(G.number_of_nodes()), seed=42)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                line=dict(width=0.5, color='#888'), hoverinfo='none')
        
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_colors = ['red' if n in critical_nodes else 'lightblue' for n in G.nodes()]
        node_sizes = [20 if n in critical_nodes else 10 for n in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='white')),
            text=[str(n) if n in critical_nodes else '' for n in G.nodes()],
            textposition='top center', hoverinfo='text',
            hovertext=[f"Node: {n}<br>Rank: {int(topsis_results.loc[n, 'rank'])}" if n in topsis_results.index else f"Node: {n}" for n in G.nodes()]
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False, hovermode='closest',
                         margin=dict(b=0, l=0, r=0, t=0),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Red = Top critical nodes | 🔵 Blue = Regular nodes")
    
    with col_rank:
        st.subheader("🏆 Top Critical Nodes")
        top_df = topsis_results.head(top_k)[['closeness', 'rank']].reset_index()
        top_df.columns = ['Node', 'Score', 'Rank']
        top_df['Score'] = top_df['Score'].round(4)
        st.dataframe(top_df, use_container_width=True, hide_index=True)
    
    # ── Row 3: Centralities ──
    st.markdown("---")
    st.subheader("📊 Centrality Measures")
    
    centrality_info = {
        'degree': 'Number of connections',
        'betweenness': 'Bridge between communities',
        'closeness': 'How quickly reach all nodes',
        'eigenvector': 'Connected to important nodes',
        'pagerank': 'Random walk probability',
        'kshell': 'Core decomposition level',
        'hindex': 'Quality × quantity of neighbors'
    }
    
    available_metrics = [m for m in df_centrality.columns if m in centrality_info]
    selected_metric = st.selectbox("Select centrality to visualize:", available_metrics)
    st.info(f"**{selected_metric.capitalize()}**: {centrality_info.get(selected_metric, 'Importance measure')}")
    
    col_hist, col_bar = st.columns(2)
    with col_hist:
        fig = px.histogram(df_centrality, x=selected_metric, nbins=20,
                          title=f'{selected_metric.capitalize()} Distribution')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_bar:
        top_by = df_centrality[selected_metric].nlargest(10)
        fig = px.bar(x=top_by.values, y=[str(n) for n in top_by.index],
                    orientation='h', title=f'Top 10 by {selected_metric.capitalize()}')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, 
                         xaxis_title=selected_metric.capitalize(), yaxis_title='Node')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    corr = df_centrality.corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                   title='Centrality Correlations')
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Row 4: CRITIC Weights ──
    st.markdown("---")
    st.subheader("⚖️ CRITIC Weight Analysis")
    st.markdown(f"*Normalization: {normalization_method}* — Higher weight = more discriminating and unique information.")
    
    col_wbar, col_wpie = st.columns(2)
    with col_wbar:
        fig = px.bar(x=weights.values, y=weights.index, orientation='h',
                    title='CRITIC Weights', color=weights.values, 
                    color_continuous_scale='viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'},
                         xaxis_title='Weight', yaxis_title='Centrality', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_wpie:
        fig = px.pie(values=weights.values, names=weights.index, title='Weight Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    weight_df = pd.DataFrame({
        'Centrality': weights.index,
        'Weight': weights.values.round(4),
        'Percentage': (weights.values * 100).round(2)
    })
    weight_df['Percentage'] = weight_df['Percentage'].astype(str) + '%'
    st.dataframe(weight_df, use_container_width=True, hide_index=True)
    
    # ── Row 5: TOPSIS Rankings ──
    st.markdown("---")
    st.subheader("🏆 TOPSIS Rankings")
    
    col_tdist, col_tbar = st.columns(2)
    with col_tdist:
        fig = px.histogram(topsis_results, x='closeness', nbins=20,
                          title='TOPSIS Score Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with col_tbar:
        top_n = topsis_results.head(15)
        fig = px.bar(x=top_n['closeness'], y=[str(n) for n in top_n.index],
                    orientation='h', title='Top 15 Critical Nodes',
                    color=top_n['closeness'], color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder':'total ascending'},
                         xaxis_title='TOPSIS Score', yaxis_title='Node')
        st.plotly_chart(fig, use_container_width=True)
    
    # Overlap comparison
    comparison_data = []
    for metric in df_centrality.columns:
        metric_top = set(df_centrality[metric].nlargest(top_k).index)
        topsis_top = set(critical_nodes)
        overlap = len(metric_top & topsis_top) / top_k * 100
        comparison_data.append({'Metric': metric, 'Overlap (%)': overlap})
    comparison_df = pd.DataFrame(comparison_data)
    fig = px.bar(comparison_df, x='Metric', y='Overlap (%)',
                title=f'Top-{top_k} Overlap with TOPSIS',
                color='Overlap (%)', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Row 6: Explainable AI (top-5 auto) ──
    st.markdown("---")
    st.subheader("💡 Why Are These Nodes Critical?")
    
    node_to_explain = st.selectbox(
        "Select node to explain:", 
        critical_nodes[:min(10, len(critical_nodes))],
        format_func=lambda x: f"Node {x} (Rank #{int(topsis_results.loc[x, 'rank'])})"
    )
    
    exp = explain_node(node_to_explain, df_centrality, weights, topsis_results)
    st.markdown(exp['main_explanation'])
    
    st.markdown("**Contributing Factors:**")
    for factor in exp['top_factors']:
        pct = factor['percentile']
        st.progress(pct/100, text=f"**{factor['metric']}**: Top {100-pct:.0f}% (weight: {factor['weight']:.2f})")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: IMPACT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="pipeline-step">📍 <b>Step 2 / 6 — Impact</b>: Test what happens when critical nodes fail.</div>', unsafe_allow_html=True)
    
    # ── Section A: Targeted Attack Simulation ──
    st.subheader("💥 Targeted Attack Simulation")
    st.markdown("Simulate removing critical nodes and measure network damage. CRITIC-TOPSIS rankings are compared against single-metric strategies.")
    
    if st.button("🚀 Run Attack Simulation", type="primary"):
        with st.spinner("Simulating attacks..."):
            rankings = {'CRITIC-TOPSIS': get_ranking_from_topsis(topsis_results)}
            for col in ['degree', 'betweenness', 'closeness', 'pagerank']:
                if col in df_centrality.columns:
                    rankings[col] = get_ranking_from_centrality(df_centrality, col)
            
            attack_results = compare_attack_methods(G, rankings, verbose=False)
            effectiveness = compute_attack_effectiveness(attack_results)
        
        st.success("Simulation complete!")
        
        col_curve, col_eff = st.columns(2)
        
        with col_curve:
            fig = go.Figure()
            colors = {'CRITIC-TOPSIS': 'red', 'degree': 'blue', 
                     'betweenness': 'green', 'closeness': 'purple', 'pagerank': 'orange'}
            for method, df in attack_results.items():
                fig.add_trace(go.Scatter(
                    x=df['fraction_removed'] * 100, y=df['lcc_fraction'],
                    mode='lines+markers', name=method,
                    line=dict(color=colors.get(method, 'gray'), width=2)
                ))
            fig.update_layout(title='Network Fragmentation Under Attack',
                            xaxis_title='Nodes Removed (%)', yaxis_title='Largest Component (fraction)',
                            hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_eff:
            eff_sorted = effectiveness.sort_values('effectiveness', ascending=True)
            fig = px.bar(eff_sorted, x='effectiveness', y='method', orientation='h',
                        title='Attack Effectiveness (higher = better)',
                        color='method', color_discrete_map={'CRITIC-TOPSIS': 'red'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        winner = effectiveness.iloc[0]['method']
        winner_score = effectiveness.iloc[0]['effectiveness']
        if winner == 'CRITIC-TOPSIS':
            st.balloons()
            st.success(f"🎉 **CRITIC-TOPSIS wins!** Effectiveness: {winner_score:.4f}")
        else:
            topsis_score = effectiveness[effectiveness['method'] == 'CRITIC-TOPSIS']['effectiveness'].values[0]
            st.info(f"**Winner: {winner}** ({winner_score:.4f}) | CRITIC-TOPSIS: {topsis_score:.4f}")
    
    # ── Section B: Cascading Failure ──
    st.markdown("---")
    st.subheader("🌊 Cascading Failure Simulation")
    st.markdown("Simulate how failures **spread** through the network. Each node has capacity = initial_load × capacity_factor. When overloaded, it fails and causes further failures.")
    
    col_cap, col_frac = st.columns(2)
    with col_cap:
        capacity_factor = st.slider("Capacity factor", 1.0, 2.0, 1.2, 0.1,
                                   help="How much extra load each node can handle (1.0 = no tolerance)")
    with col_frac:
        initial_fraction = st.slider("Initial failure fraction", 0.02, 0.20, 0.05, 0.01,
                                    help="Fraction of top critical nodes to remove initially")
    
    if st.button("🌊 Run Cascade Simulation", type="primary"):
        with st.spinner("Simulating cascading failures..."):
            ranking = get_ranking_from_topsis(topsis_results)
            n_initial = max(1, int(G.number_of_nodes() * initial_fraction))
            cascade_result = simulate_cascading_failure(
                G, ranking[:n_initial], capacity_factor=capacity_factor, verbose=False)
        
        st.success("Cascade simulation complete!")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial Failures", cascade_result['initial_failures'])
        c2.metric("Cascade Iterations", cascade_result['cascade_iterations'])
        c3.metric("Total Failures", cascade_result['total_failures'])
        c4.metric("Survival Rate", f"{cascade_result['survival_rate']:.1%}")
        
        multiplier = cascade_result['total_failures'] / max(1, cascade_result['initial_failures'])
        cascade_extra = cascade_result['total_failures'] - cascade_result['initial_failures']
        
        if multiplier > 1.5:
            st.error(f"⚠️ **Cascade multiplier: {multiplier:.1f}x** — Initial failures caused {cascade_extra} additional failures!")
        else:
            st.info(f"ℹ️ Cascade multiplier: {multiplier:.1f}x (relatively contained)")
        
        st.metric("Final LCC Fraction", f"{cascade_result['lcc_fraction']:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ROBUSTNESS CHECK
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="pipeline-step">📍 <b>Step 3 / 6 — Robustness</b>: Verify if rankings hold under uncertainty and attacks.</div>', unsafe_allow_html=True)
    
    # ── Section A: Sensitivity Analysis ──
    st.subheader("🔬 Sensitivity Analysis")
    st.markdown("Test how robust the rankings are to parameter changes.")
    
    if st.button("🔬 Run Sensitivity Analysis", type="primary"):
        with st.spinner("Analyzing sensitivity..."):
            norm_sens = sensitivity_to_normalization(G)
            cent_impact = sensitivity_to_centrality_removal(G)
            topk_stab = sensitivity_to_top_k(G)
        
        st.success("Analysis complete!")
        
        st.markdown("#### Normalization Method Impact")
        st.dataframe(norm_sens, hide_index=True)
        
        st.markdown("#### Centrality Impact (When Removed)")
        st.caption("*Higher impact means this metric significantly affects rankings*")
        fig = px.bar(cent_impact, x='removed_centrality', y='impact',
                    title='Impact of Removing Each Centrality', color='impact',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Top-k Stability")
        pivot = topk_stab.pivot(index='metric', columns='k', values='overlap')
        fig = px.imshow(pivot, text_auto='.0f', color_continuous_scale='Blues',
                       title='TOPSIS Overlap (%) with Single Metrics at Different k')
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Section B: Uncertainty Quantification ──
    st.markdown("---")
    st.subheader("📊 Uncertainty Quantification")
    st.markdown("How confident are we in the rankings? Bootstrap-based confidence intervals.")
    
    n_bootstrap = st.slider("Bootstrap iterations", 20, 100, 50)
    
    if st.button("📊 Run Uncertainty Analysis", type="primary"):
        with st.spinner(f"Running {n_bootstrap} bootstrap iterations..."):
            unc_result = full_uncertainty_analysis(G, n_bootstrap=n_bootstrap, top_k=10)
        
        st.success("Uncertainty analysis complete!")
        
        st.markdown("#### 🎯 High-Confidence Critical Nodes (>90% in top-10)")
        hc = unc_result['high_confidence_critical']
        if hc:
            st.success(f"Nodes: {hc}")
        else:
            st.warning("No nodes are >90% confident in top-10")
        
        st.markdown("#### Top-10 Probabilities")
        prob_df = unc_result['top_k_probabilities'].head(10)
        fig = px.bar(prob_df, x='node', y='prob_top_k', 
                    title='Probability of Being in Top-10',
                    color='prob_top_k', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Stability Metrics")
        stab = unc_result['stability_metrics']
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Avg Rank Std", f"{stab['mean_rank_std']:.2f}")
        sc2.metric("Stable Nodes (σ<2)", stab['stable_nodes'])
        sc3.metric("Unstable Nodes (σ≥5)", stab['unstable_nodes'])
    
    # ── Section C: Adversarial Robustness ──
    st.markdown("---")
    st.subheader("🛡️ Adversarial Robustness Testing")
    st.markdown("Test if an attacker can manipulate node rankings by adding/removing edges or creating fake nodes.")
    
    if st.button("🛡️ Run Adversarial Analysis", type="primary"):
        with st.spinner("Testing attack vectors..."):
            adv_result = full_adversarial_analysis(G)
        
        st.success("Adversarial analysis complete!")
        
        grade = adv_result['overall_grade']
        vuln = adv_result['overall_vulnerability']
        
        if grade == 'A':
            st.success(f"**Overall Grade: {grade}** — Highly Robust! ({vuln:.0f}% vulnerable)")
        elif grade == 'B':
            st.warning(f"**Overall Grade: {grade}** — Moderately Robust ({vuln:.0f}% vulnerable)")
        else:
            st.error(f"**Overall Grade: {grade}** — Vulnerable to attacks ({vuln:.0f}% vulnerable)")
        
        st.markdown("#### Node Robustness")
        for nr in adv_result['node_robustness']:
            st.write(f"Node {nr['node']}: **Grade {nr['robustness_grade']}** ({nr['vulnerability_score']:.0f}% vulnerable)")
        
        st.markdown("#### Recommendations")
        for rec in adv_result['recommendations']:
            st.write(f"• {rec}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: TEMPORAL EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="pipeline-step">📍 <b>Step 4 / 6 — Temporal</b>: Track how critical nodes change over time.</div>', unsafe_allow_html=True)
    
    # ── Section A: Temporal Analysis ──
    st.subheader("⏰ Temporal Critical Node Prediction")
    st.markdown("Predict future critical nodes **and** track how CRITIC weights adapt over time.")
    
    col_snap, col_vol, col_adapt = st.columns(3)
    with col_snap:
        n_snapshots = st.slider("Number of time snapshots", 3, 10, 5)
    with col_vol:
        volatility = st.slider("Network volatility", 0.05, 0.30, 0.10)
    with col_adapt:
        use_adaptive_temporal = st.checkbox("🔄 Adaptive weights", value=True,
                                           help="Recalculate CRITIC weights per snapshot with exponential decay")
    
    if use_adaptive_temporal:
        decay_factor = st.slider("Decay factor (0=equal, 1=latest only)", 0.0, 1.0, 0.3, 0.1,
                                help="Higher values weight recent snapshots more heavily")
    
    if st.button("⏰ Run Temporal Analysis", type="primary"):
        if use_adaptive_temporal:
            with st.spinner("Running adaptive temporal analysis..."):
                temp_result = temporal_adaptive_summary(G, n_snapshots, volatility, decay=decay_factor)
        else:
            with st.spinner("Generating temporal snapshots..."):
                temp_result = temporal_prediction_summary(G, n_snapshots, volatility)
        
        st.success(f"Analyzed {n_snapshots} snapshots!")
        
        st.markdown("### 🌟 Rising Stars (Becoming More Critical)")
        if temp_result['rising_stars']:
            for rs in temp_result['rising_stars'][:5]:
                st.write(f"**Node {rs['node']}**: Trend {rs['trend']:.1f} (improving)")
        else:
            st.info("No rising stars detected")
        
        st.markdown("### 🏆 Stable Critical Nodes")
        if temp_result['stable_critical']:
            for sc in temp_result['stable_critical'][:5]:
                st.write(f"**Node {sc['node']}**: Stability σ={sc['stability']:.2f}")
        
        if use_adaptive_temporal and 'weight_evolution' in temp_result:
            st.markdown("---")
            st.markdown("### 📊 CRITIC Weight Evolution Over Time")
            
            weight_df = temp_result['weight_evolution']['weight_timeseries']
            fig = go.Figure()
            for col in weight_df.columns:
                fig.add_trace(go.Scatter(
                    x=list(range(len(weight_df))), y=weight_df[col],
                    mode='lines+markers', name=col.capitalize(), line=dict(width=2)
                ))
            fig.update_layout(title='How CRITIC Weights Change Across Snapshots',
                            xaxis_title='Snapshot (time)', yaxis_title='CRITIC Weight',
                            hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### ⚠️ Weight Drift Detection")
            drift = temp_result['weight_evolution']['drift']
            has_drift = False
            for metric, d in drift.items():
                if d['significant']:
                    has_drift = True
                    icon = "📈" if d['direction'] == 'increased' else "📉"
                    st.warning(f"{icon} **{metric}**: {d['direction']} by {d['absolute_change']:.4f} "
                              f"(from {d['initial']:.4f} → {d['final']:.4f})")
            if not has_drift:
                st.success("✅ No significant weight drift — weights are stable across time.")
            
            st.markdown("### 🔄 Static vs. Adaptive Weights")
            comp_data = []
            for m, c in temp_result['weight_comparison'].items():
                comp_data.append({
                    'Metric': m.capitalize(),
                    'Static CRITIC': round(c['static'], 4),
                    'Adaptive (temporal)': round(c['adaptive'], 4),
                    'Difference': round(c['diff'], 4)
                })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    
    # ── Section B: Real-World Network Generation ──
    st.markdown("---")
    st.subheader("🌐 Real-World Network Generation")
    st.markdown("Generate realistic networks with domain-specific structural properties for temporal studies.")
    
    domain_gen = st.selectbox("Select network domain:", ["Social", "Infrastructure", "Biological"])
    rw_nodes = st.slider("Network size", 50, 500, 200)
    
    if st.button("🌐 Generate & Analyze", type="primary"):
        with st.spinner(f"Generating {domain_gen} network..."):
            if domain_gen == "Social":
                G_rw = generate_social_network(rw_nodes)
            elif domain_gen == "Infrastructure":
                G_rw = generate_infrastructure_network(rw_nodes)
            else:
                G_rw = generate_biological_network(rw_nodes)
            
            chars = get_network_characteristics(G_rw)
            df_rw = compute_all_centralities(G_rw, verbose=False)
            weights_rw, _ = compute_critic_weights(df_rw, verbose=False)
            results_rw, _ = topsis_rank(df_rw, weights_rw, verbose=False)
        
        st.success(f"{domain_gen} network generated!")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Nodes", chars['nodes'])
        rc2.metric("Edges", chars['edges'])
        rc3.metric("Clustering", f"{chars['clustering']:.3f}")
        
        st.markdown("**Top 10 Critical Nodes:**")
        st.write(results_rw.head(10))
        
        st.markdown("**CRITIC Weights:**")
        fig = px.bar(x=weights_rw.values, y=weights_rw.index, orientation='h',
                    color=weights_rw.values, color_continuous_scale='viridis')
        fig.update_layout(xaxis_title='Weight', yaxis_title='Centrality',
                         yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: DOMAIN INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="pipeline-step">📍 <b>Step 5 / 6 — Domain Intelligence</b>: Apply expert knowledge for your network type.</div>', unsafe_allow_html=True)
    
    # ── Section A: Domain-Specific Weighting ──
    st.subheader("🎯 Domain-Specific Weighting")
    st.markdown("Use pre-trained weights optimized for your network type and compare with data-driven CRITIC weights.")
    
    domains = get_available_domains()
    domain_choice = st.selectbox(
        "Select domain profile:",
        list(domains.keys()),
        format_func=lambda x: f"{domains[x]['name']}"
    )
    st.info(f"**{domains[domain_choice]['name']}**: {domains[domain_choice]['description']}")
    
    if st.button("🎯 Apply Domain Weights", type="primary"):
        with st.spinner("Analyzing with domain weights..."):
            dom_result = domain_aware_analysis(G, domain=domain_choice)
        
        st.success(f"Applied {domain_choice} weights!")
        
        col_dw, col_dc = st.columns(2)
        with col_dw:
            st.markdown("#### Domain Weights")
            for m, w in dom_result['domain_weights'].items():
                st.write(f"**{m}**: {w:.2f}")
        with col_dc:
            st.markdown("#### Comparison with CRITIC")
            overlap = dom_result['comparison_with_critic']['overlap']
            st.metric("Top-10 Overlap", f"{overlap:.0%}")
            if dom_result['comparison_with_critic']['unique_to_domain']:
                st.write(f"Unique to domain: {dom_result['comparison_with_critic']['unique_to_domain']}")
            if dom_result['comparison_with_critic']['unique_to_critic']:
                st.write(f"Unique to CRITIC: {dom_result['comparison_with_critic']['unique_to_critic']}")
        
        st.markdown("#### Top 10 Critical Nodes (Domain Weights)")
        st.write(dom_result['results'].head(10))
    
    # ── Section B: Network Comparison ──
    st.markdown("---")
    st.subheader("📊 Network Comparison")
    st.markdown("Compare same analysis across multiple network types.")
    
    selected_networks = st.multiselect(
        "Select networks to compare:",
        ["Karate Club", "Social (synthetic)", "Infrastructure (synthetic)", "Biological (synthetic)"],
        default=["Karate Club", "Social (synthetic)"]
    )
    
    if st.button("📊 Run Comparison", type="primary") and selected_networks:
        comparison_results = []
        
        for net_name in selected_networks:
            with st.spinner(f"Analyzing {net_name}..."):
                if net_name == "Karate Club":
                    G_temp = load_karate_club()
                elif "Social" in net_name:
                    G_temp = generate_social_network(100)
                elif "Infrastructure" in net_name:
                    G_temp = generate_infrastructure_network(100)
                else:
                    G_temp = generate_biological_network(100)
                
                chars = get_network_characteristics(G_temp)
                df_temp = compute_all_centralities(G_temp, verbose=False)
                weights_temp, _ = compute_critic_weights(df_temp, verbose=False)
                
                comparison_results.append({
                    'Network': net_name,
                    'Nodes': chars['nodes'],
                    'Edges': chars['edges'],
                    'Density': f"{chars['density']:.3f}",
                    'Clustering': f"{chars['clustering']:.3f}",
                    'Top Weight': f"{weights_temp.idxmax()} ({weights_temp.max():.2f})"
                })
        
        st.success("Comparison complete!")
        st.dataframe(pd.DataFrame(comparison_results), hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: SCALE & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="pipeline-step">📍 <b>Step 6 / 6 — Scale & Export</b>: Validate speed and download everything.</div>', unsafe_allow_html=True)
    
    # ── Section A: Scalability Benchmark ──
    st.subheader("📏 Scalability Benchmark")
    st.markdown("Demonstrate CRITIC-TOPSIS performance on **large networks** and measure computation time.")
    
    col_bm, col_bs = st.columns(2)
    with col_bm:
        bench_model = st.selectbox("Network model:", ["Barabási-Albert", "Erdős-Rényi"])
    with col_bs:
        bench_max = st.selectbox("Max network size:", [1000, 2000, 5000], index=1)
    
    if bench_max == 1000:
        bench_sizes = [100, 250, 500, 750, 1000]
    elif bench_max == 2000:
        bench_sizes = [100, 250, 500, 1000, 1500, 2000]
    else:
        bench_sizes = [100, 500, 1000, 2000, 3000, 5000]
    
    st.caption(f"Will test sizes: {bench_sizes}")
    
    if st.button("📏 Run Scalability Benchmark", type="primary"):
        progress_bar = st.progress(0, text="Starting benchmark...")
        
        def update_progress(i, total):
            progress_bar.progress((i + 1) / total, text=f"Testing n={bench_sizes[i]}...")
        
        with st.spinner("Benchmarking..."):
            model_key = 'barabasi_albert' if 'Barabási' in bench_model else 'erdos_renyi'
            bench_df = benchmark_scalability(sizes=bench_sizes, model=model_key,
                                            progress_callback=update_progress)
        
        progress_bar.progress(1.0, text="Complete!")
        st.success(f"Benchmark complete for {len(bench_sizes)} network sizes!")
        
        scale_factor = bench_df.iloc[-1]['total_time'] / max(bench_df.iloc[0]['total_time'], 0.0001)
        
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Smallest", f"{bench_df.iloc[0]['n_nodes']} nodes",
                  f"{bench_df.iloc[0]['total_time']:.3f}s")
        bc2.metric("Largest", f"{bench_df.iloc[-1]['n_nodes']} nodes",
                  f"{bench_df.iloc[-1]['total_time']:.3f}s")
        bc3.metric("Scaling Factor", f"{scale_factor:.1f}x")
        
        st.markdown("#### ⏱️ Computation Time vs Network Size")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bench_df['n_nodes'], y=bench_df['centrality_time'],
                                mode='lines+markers', name='Centralities',
                                line=dict(width=2, color='#1f77b4')))
        fig.add_trace(go.Scatter(x=bench_df['n_nodes'], y=bench_df['critic_time'],
                                mode='lines+markers', name='CRITIC Weights',
                                line=dict(width=2, color='#ff7f0e')))
        fig.add_trace(go.Scatter(x=bench_df['n_nodes'], y=bench_df['topsis_time'],
                                mode='lines+markers', name='TOPSIS Ranking',
                                line=dict(width=2, color='#2ca02c')))
        fig.add_trace(go.Scatter(x=bench_df['n_nodes'], y=bench_df['total_time'],
                                mode='lines+markers', name='Total',
                                line=dict(width=3, color='red', dash='dash')))
        fig.update_layout(xaxis_title='Number of Nodes', yaxis_title='Time (seconds)',
                         hovermode='x unified', height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 📋 Detailed Timing Results")
        display_df = bench_df[['n_nodes', 'n_edges', 'centrality_time', 'critic_time', 'topsis_time', 'total_time']].copy()
        display_df.columns = ['Nodes', 'Edges', 'Centrality (s)', 'CRITIC (s)', 'TOPSIS (s)', 'Total (s)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### 🔍 Current Network Performance")
        current_bench = benchmark_single(G)
        st.info(f"**{getattr(G, 'name', 'Current')}** ({current_bench['n_nodes']} nodes, "
               f"{current_bench['n_edges']} edges): **{current_bench['total_time']:.4f}s** total")
    
    # ── Section B: Export ──
    st.markdown("---")
    st.subheader("📥 Export Results")
    st.markdown("Download your analysis results in various formats.")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("#### Critical Nodes (Rankings)")
        export_df = topsis_results.reset_index()
        export_df.columns = ['node', 'topsis_score', 'dist_to_best', 'dist_to_worst', 'rank']
        
        csv_data = export_df.to_csv(index=False)
        st.download_button("📄 Download CSV", csv_data, "critical_nodes.csv", "text/csv")
        
        json_data = export_df.to_json(orient='records', indent=2)
        st.download_button("📋 Download JSON", json_data, "critical_nodes.json", "application/json")
        
        st.markdown("---")
        st.markdown("#### Top Critical Nodes")
        st.code(f"Top {top_k} nodes: {critical_nodes[:top_k]}")
    
    with col_ex2:
        st.markdown("#### CRITIC Weights")
        weights_df = pd.DataFrame({'centrality': weights.index, 'weight': weights.values})
        st.download_button("📄 Download Weights CSV", weights_df.to_csv(index=False),
                          "critic_weights.csv", "text/csv")
        
        st.markdown("---")
        st.markdown("#### All Centralities")
        st.download_button("📊 Download All Centralities", df_centrality.to_csv(),
                          "centralities.csv", "text/csv")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p>Critical Node Detection using CRITIC-TOPSIS Framework</p>
    <p><em>Pipeline: Discovery → Impact → Robustness → Temporal → Domain → Scale & Export</em></p>
</div>
""", unsafe_allow_html=True)
