"""
Critical Node Detection - Interactive Web Interface
====================================================
A Streamlit-based UI for analyzing critical nodes in networks.

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
                          create_barabasi_albert, create_erdos_renyi,
                          get_network_info)
from cascading_failure import simulate_cascading_failure, compare_cascade_methods, cascade_over_fractions
from sensitivity_analysis import sensitivity_to_normalization, sensitivity_to_centrality_removal, sensitivity_to_top_k
from real_world_datasets import generate_social_network, generate_infrastructure_network, generate_biological_network, get_network_characteristics
from temporal_analysis import temporal_prediction_summary, generate_temporal_snapshots, analyze_temporal_rankings
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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .feature-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔍 Critical Node Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CRITIC-TOPSIS Multi-Attribute Framework</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Network Selection & Settings
# ============================================================================

st.sidebar.header("📊 Network Selection")

# NEW FEATURE: Custom network upload
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
            
            # Parse edges
            edges = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Try different separators
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
        # Default to Karate Club if no file uploaded
        G = load_karate_club()
else:
    network_option = st.sidebar.selectbox(
        "Choose a network:",
        ["Karate Club (34 nodes)", "Les Miserables (77 nodes)", 
         "Florentine Families (15 nodes)", "Dolphins (62 nodes)",
         "Barabási-Albert (custom)", "Erdős-Rényi (custom)"]
    )
    
    # Load network based on selection
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
        elif "Barabási" in option:
            return create_barabasi_albert(ba_n, ba_m)
        else:
            return create_erdos_renyi(er_n, er_p)
    
    # Additional parameters for synthetic networks
    if "Barabási" in network_option:
        ba_n = st.sidebar.slider("Number of nodes", 20, 500, 100)
        ba_m = st.sidebar.slider("Edges per new node", 1, 10, 3)
        G = load_network(network_option, ba_n=ba_n, ba_m=ba_m)
    elif "Erdős" in network_option:
        er_n = st.sidebar.slider("Number of nodes", 20, 500, 100)
        er_p = st.sidebar.slider("Edge probability", 0.01, 0.5, 0.1)
        G = load_network(network_option, er_n=er_n, er_p=er_p)
    else:
        G = load_network(network_option)

# Ensure G is loaded
if G is None:
    G = load_karate_club()

# ============================================================================
# SETTINGS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Top-k critical nodes", 5, 20, 10)

# NEW FEATURE: Normalization method selection
normalization_method = st.sidebar.selectbox(
    "Normalization method:",
    ["Min-Max (default)", "Z-Score", "Log (for skewed data)"],
    help="Choose how to normalize centrality values before CRITIC"
)

# NEW FEATURE: Adaptive centrality selection
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

# Network info
info = get_network_info(G)

# Custom normalization functions
def normalize_log(df):
    """Log normalization for skewed distributions."""
    df_log = df.copy()
    for col in df.columns:
        df_log[col] = np.log1p(df[col])
    return normalize_minmax(df_log)

# Compute data
@st.cache_data
def compute_analysis(_G, norm_method, adaptive, var_thresh):
    df = compute_all_centralities(_G, verbose=False)
    
    # Adaptive centrality selection
    excluded_metrics = []
    if adaptive:
        df_norm_check = normalize_minmax(df)
        variances = df_norm_check.var()
        low_var = variances[variances < var_thresh].index.tolist()
        if low_var and len(low_var) < len(df.columns):  # Keep at least some columns
            excluded_metrics = low_var
            df = df.drop(columns=low_var)
    
    # Apply selected normalization for CRITIC
    weights, _ = compute_critic_weights(df, normalization=norm_method.split()[0].lower(), verbose=False)
    results, _ = topsis_rank(df, weights, verbose=False)
    
    return df, weights, results, excluded_metrics

# Map normalization method
norm_key = normalization_method.split()[0].lower()
if "log" in normalization_method.lower():
    norm_key = "minmax"  # Use minmax for log preprocessing

df_centrality, weights, topsis_results, excluded_metrics = compute_analysis(
    G, normalization_method, use_adaptive, variance_threshold
)
critical_nodes = get_critical_nodes(topsis_results, top_k)

# Show excluded metrics if any
if excluded_metrics:
    st.sidebar.warning(f"⚠️ Excluded low-variance: {', '.join(excluded_metrics)}")

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15 = st.tabs([
    "📈 Overview", "📊 Centralities", "⚖️ CRITIC", 
    "🏆 Rankings", "💥 Attack", "🌊 Cascade", 
    "📥 Export", "🔬 Sensitivity", "📊 Compare", "🌐 Real-World",
    "⏰ Temporal", "💡 Explain", "📊 Uncertainty", "🎯 Domain", "🛡️ Adversarial"
])

# TAB 1: Overview
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", info['nodes'])
    with col2:
        st.metric("Edges", info['edges'])
    with col3:
        st.metric("Density", f"{info['density']:.3f}")
    with col4:
        st.metric("Avg Clustering", f"{info['avg_clustering']:.3f}")
    
    # Show active features
    features_active = []
    if use_adaptive:
        features_active.append("🧠 Adaptive")
    if "Z-Score" in normalization_method:
        features_active.append("📊 Z-Score")
    if "Log" in normalization_method:
        features_active.append("📈 Log Norm")
    
    if features_active:
        st.info(f"Active features: {' | '.join(features_active)}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🕸️ Network Visualization")
        
        # Create network visualization with Plotly
        pos = nx.spring_layout(G, k=2/np.sqrt(G.number_of_nodes()), seed=42)
        
        # Edge traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        )
        
        # Node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = ['red' if node in critical_nodes else 'lightblue' for node in G.nodes()]
        node_sizes = [20 if node in critical_nodes else 10 for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='white')),
            text=[str(n) if n in critical_nodes else '' for n in G.nodes()],
            textposition='top center',
            hoverinfo='text',
            hovertext=[f"Node: {n}<br>Rank: {int(topsis_results.loc[n, 'rank'])}" for n in G.nodes()]
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Red nodes = Top critical nodes")
    
    with col_right:
        st.subheader("🏆 Top Critical Nodes")
        top_df = topsis_results.head(top_k)[['closeness', 'rank']].reset_index()
        top_df.columns = ['Node', 'Score', 'Rank']
        top_df['Score'] = top_df['Score'].round(4)
        st.dataframe(top_df, use_container_width=True, hide_index=True)

# TAB 2: Centralities
with tab2:
    st.subheader("📊 Centrality Measures")
    
    if excluded_metrics:
        st.warning(f"**Adaptive mode excluded:** {', '.join(excluded_metrics)} (low variance)")
    
    st.markdown(f"Using **{len(df_centrality.columns)}** centrality measures:")
    
    # Centrality descriptions
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution histogram
        fig = px.histogram(df_centrality, x=selected_metric, nbins=20,
                          title=f'{selected_metric.capitalize()} Distribution')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top 10 by this metric
        top_by_metric = df_centrality[selected_metric].nlargest(10)
        fig = px.bar(x=top_by_metric.values, y=[str(n) for n in top_by_metric.index],
                    orientation='h', title=f'Top 10 by {selected_metric.capitalize()}')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, 
                         xaxis_title=selected_metric.capitalize(),
                         yaxis_title='Node')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("📈 Centrality Correlations")
    corr = df_centrality.corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                   title='Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: CRITIC Weights
with tab3:
    st.subheader("⚖️ CRITIC Weight Analysis")
    st.markdown(f"""
    **CRITIC** determines objective weights based on:
    - **Standard deviation**: Higher variance = more discriminating
    - **Correlation**: Lower correlation = more unique information
    
    *Normalization: {normalization_method}*
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weight bar chart
        fig = px.bar(x=weights.values, y=weights.index, orientation='h',
                    title='CRITIC Weights',
                    color=weights.values, color_continuous_scale='viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'},
                         xaxis_title='Weight', yaxis_title='Centrality',
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart
        fig = px.pie(values=weights.values, names=weights.index,
                    title='Weight Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Weight table
    st.subheader("📋 Weight Details")
    weight_df = pd.DataFrame({
        'Centrality': weights.index,
        'Weight': weights.values.round(4),
        'Percentage': (weights.values * 100).round(2)
    })
    weight_df['Percentage'] = weight_df['Percentage'].astype(str) + '%'
    st.dataframe(weight_df, use_container_width=True, hide_index=True)

# TAB 4: Rankings
with tab4:
    st.subheader("🏆 TOPSIS Rankings")
    st.markdown("Nodes ranked by closeness to ideal best and distance from ideal worst.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        fig = px.histogram(topsis_results, x='closeness', nbins=20,
                          title='TOPSIS Score Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top nodes bar chart  
        top_n = topsis_results.head(15)
        fig = px.bar(x=top_n['closeness'], y=[str(n) for n in top_n.index],
                    orientation='h', title=f'Top 15 Critical Nodes',
                    color=top_n['closeness'], color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder':'total ascending'},
                         xaxis_title='TOPSIS Score', yaxis_title='Node')
        st.plotly_chart(fig, use_container_width=True)
    
    # Compare with single metrics
    st.subheader("📊 Ranking Comparison")
    st.markdown("How TOPSIS compares to single-metric rankings:")
    
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

# TAB 5: Attack Simulation
with tab5:
    st.subheader("💥 Targeted Attack Simulation")
    st.markdown("Simulate removing critical nodes and measure network damage.")
    
    if st.button("🚀 Run Attack Simulation", type="primary"):
        with st.spinner("Simulating attacks..."):
            # Create rankings
            rankings = {'CRITIC-TOPSIS': get_ranking_from_topsis(topsis_results)}
            for col in ['degree', 'betweenness', 'closeness', 'pagerank']:
                if col in df_centrality.columns:
                    rankings[col] = get_ranking_from_centrality(df_centrality, col)
            
            attack_results = compare_attack_methods(G, rankings, verbose=False)
            effectiveness = compute_attack_effectiveness(attack_results)
        
        st.success("Simulation complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Attack curves
            fig = go.Figure()
            colors = {'CRITIC-TOPSIS': 'red', 'degree': 'blue', 
                     'betweenness': 'green', 'closeness': 'purple', 'pagerank': 'orange'}
            
            for method, df in attack_results.items():
                fig.add_trace(go.Scatter(
                    x=df['fraction_removed'] * 100, y=df['lcc_fraction'],
                    mode='lines+markers', name=method,
                    line=dict(color=colors.get(method, 'gray'), width=2)
                ))
            
            fig.update_layout(
                title='Network Fragmentation Under Attack',
                xaxis_title='Nodes Removed (%)',
                yaxis_title='Largest Component (fraction)',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Effectiveness comparison
            eff_sorted = effectiveness.sort_values('effectiveness', ascending=True)
            
            fig = px.bar(eff_sorted, x='effectiveness', y='method', orientation='h',
                        title='Attack Effectiveness (higher = better)',
                        color='method',
                        color_discrete_map={'CRITIC-TOPSIS': 'red'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'},
                            showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Winner announcement
        winner = effectiveness.iloc[0]['method']
        winner_score = effectiveness.iloc[0]['effectiveness']
        
        if winner == 'CRITIC-TOPSIS':
            st.balloons()
            st.success(f"🎉 **CRITIC-TOPSIS wins!** Effectiveness: {winner_score:.4f}")
        else:
            topsis_score = effectiveness[effectiveness['method'] == 'CRITIC-TOPSIS']['effectiveness'].values[0]
            st.info(f"**Winner: {winner}** ({winner_score:.4f}) | CRITIC-TOPSIS: {topsis_score:.4f}")

# TAB 6: Cascading Failure Simulation
with tab6:
    st.subheader("🌊 Cascading Failure Simulation")
    st.markdown("""
    Simulate how failures **spread** through the network, not just immediate removal.
    
    **Model**: Each node has capacity = initial_load × capacity_factor. When overloaded, it fails.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        capacity_factor = st.slider("Capacity factor", 1.0, 2.0, 1.2, 0.1,
                                   help="How much extra load each node can handle (1.0 = no tolerance)")
    with col2:
        initial_fraction = st.slider("Initial failure fraction", 0.02, 0.20, 0.05, 0.01,
                                    help="Fraction of top critical nodes to remove initially")
    
    if st.button("🌊 Run Cascade Simulation", type="primary"):
        with st.spinner("Simulating cascading failures..."):
            ranking = get_ranking_from_topsis(topsis_results)
            n_initial = max(1, int(G.number_of_nodes() * initial_fraction))
            cascade_result = simulate_cascading_failure(
                G, ranking[:n_initial],
                capacity_factor=capacity_factor, verbose=False
            )
        
        st.success("Cascade simulation complete!")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial Failures", cascade_result['initial_failures'])
        c2.metric("Cascade Iterations", cascade_result['cascade_iterations'])
        c3.metric("Total Failures", cascade_result['total_failures'])
        c4.metric("Survival Rate", f"{cascade_result['survival_rate']:.1%}")
        
        multiplier = cascade_result['total_failures'] / max(1, cascade_result['initial_failures'])
        cascade_extra = cascade_result['total_failures'] - cascade_result['initial_failures']
        
        if multiplier > 1.5:
            st.error(f"⚠️ **Cascade multiplier: {multiplier:.1f}x** - Initial failures caused {cascade_extra} additional failures!")
        else:
            st.info(f"ℹ️ Cascade multiplier: {multiplier:.1f}x (relatively contained)")
        
        st.metric("Final LCC Fraction", f"{cascade_result['lcc_fraction']:.1%}")

# TAB 7: Export Results
with tab7:
    st.subheader("📥 Export Results")
    st.markdown("Download your analysis results in various formats.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Critical Nodes")
        export_df = topsis_results.reset_index()
        export_df.columns = ['node', 'topsis_score', 'dist_to_best', 'dist_to_worst', 'rank']
        
        csv_data = export_df.to_csv(index=False)
        st.download_button("📄 Download CSV", csv_data, "critical_nodes.csv", "text/csv")
        
        json_data = export_df.to_json(orient='records', indent=2)
        st.download_button("📋 Download JSON", json_data, "critical_nodes.json", "application/json")
        
        st.markdown("---")
        st.markdown("### Top Critical Nodes")
        st.code(f"Top {top_k} nodes: {critical_nodes[:top_k]}")
    
    with col2:
        st.markdown("### CRITIC Weights")
        weights_df = pd.DataFrame({'centrality': weights.index, 'weight': weights.values})
        st.download_button("📄 Download Weights CSV", weights_df.to_csv(index=False), "critic_weights.csv", "text/csv")
        
        st.markdown("---")
        st.markdown("### All Centralities")
        st.download_button("📊 Download All Centralities", df_centrality.to_csv(), "centralities.csv", "text/csv")

# TAB 8: Sensitivity Analysis (NEW)
with tab8:
    st.subheader("🔬 Sensitivity Analysis")
    st.markdown("Test how robust the rankings are to parameter changes.")
    
    if st.button("🔬 Run Sensitivity Analysis", type="primary"):
        with st.spinner("Analyzing sensitivity..."):
            norm_sens = sensitivity_to_normalization(G)
            cent_impact = sensitivity_to_centrality_removal(G)
            topk_stab = sensitivity_to_top_k(G)
        
        st.success("Analysis complete!")
        
        st.markdown("### 1. Normalization Method Impact")
        st.dataframe(norm_sens, hide_index=True)
        
        st.markdown("### 2. Centrality Impact (if removed)")
        st.markdown("*Higher impact means this metric significantly affects rankings*")
        fig = px.bar(cent_impact, x='removed_centrality', y='impact',
                    title='Impact of Removing Each Centrality', color='impact', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 3. Top-k Stability")
        pivot = topk_stab.pivot(index='metric', columns='k', values='overlap')
        fig = px.imshow(pivot, text_auto='.0f', color_continuous_scale='Blues',
                       title='TOPSIS Overlap (%) with Single Metrics at Different k')
        st.plotly_chart(fig, use_container_width=True)

# TAB 9: Network Comparison (NEW)
with tab9:
    st.subheader("📊 Network Comparison Mode")
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

# TAB 10: Real-World Datasets (NEW)
with tab10:
    st.subheader("🌐 Real-World Network Types")
    st.markdown("Generate realistic networks based on different domains.")
    
    domain = st.selectbox("Select network domain:", ["Social", "Infrastructure", "Biological"])
    rw_nodes = st.slider("Network size", 50, 500, 200)
    
    if st.button("🌐 Generate & Analyze", type="primary"):
        with st.spinner(f"Generating {domain} network..."):
            if domain == "Social":
                G_rw = generate_social_network(rw_nodes)
            elif domain == "Infrastructure":
                G_rw = generate_infrastructure_network(rw_nodes)
            else:
                G_rw = generate_biological_network(rw_nodes)
            
            chars = get_network_characteristics(G_rw)
            df_rw = compute_all_centralities(G_rw, verbose=False)
            weights_rw, _ = compute_critic_weights(df_rw, verbose=False)
            results_rw, _ = topsis_rank(df_rw, weights_rw, verbose=False)
        
        st.success(f"{domain} network generated!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", chars['nodes'])
        col2.metric("Edges", chars['edges'])
        col3.metric("Clustering", f"{chars['clustering']:.3f}")
        
        st.markdown("**Top 10 Critical Nodes:**")
        st.write(results_rw.head(10))
        
        st.markdown("**CRITIC Weights:**")
        fig = px.bar(x=weights_rw.values, y=weights_rw.index, orientation='h', color=weights_rw.values)
        st.plotly_chart(fig, use_container_width=True)

# TAB 11: Temporal Analysis (NEW)
with tab11:
    st.subheader("⏰ Temporal Critical Node Prediction")
    st.markdown("Predict which nodes will become critical in the FUTURE.")
    
    col1, col2 = st.columns(2)
    with col1:
        n_snapshots = st.slider("Number of time snapshots", 3, 10, 5)
    with col2:
        volatility = st.slider("Network volatility", 0.05, 0.30, 0.10)
    
    if st.button("⏰ Run Temporal Analysis", type="primary"):
        with st.spinner("Generating temporal snapshots..."):
            result = temporal_prediction_summary(G, n_snapshots, volatility)
        
        st.success(f"Analyzed {n_snapshots} snapshots!")
        
        st.markdown("### 🌟 Rising Stars (Becoming More Critical)")
        if result['rising_stars']:
            for rs in result['rising_stars'][:5]:
                st.write(f"**Node {rs['node']}**: Trend {rs['trend']:.1f} (improving)")
        else:
            st.info("No rising stars detected")
        
        st.markdown("### 🏆 Stable Critical Nodes")
        if result['stable_critical']:
            for sc in result['stable_critical'][:5]:
                st.write(f"**Node {sc['node']}**: Stability σ={sc['stability']:.2f}")

# TAB 12: Explainable AI (NEW)
with tab12:
    st.subheader("💡 Explainable AI: Why is this node critical?")
    st.markdown("Natural language explanations for node importance.")
    
    node_to_explain = st.selectbox(
        "Select node to explain:", 
        critical_nodes[:min(10, len(critical_nodes))],
        format_func=lambda x: f"Node {x} (Rank #{int(topsis_results.loc[x, 'rank'])})"
    )
    
    if st.button("💡 Explain", type="primary"):
        exp = explain_node(node_to_explain, df_centrality, weights, topsis_results)
        st.markdown(exp['main_explanation'])
        
        st.markdown("### Contributing Factors")
        for factor in exp['top_factors']:
            pct = factor['percentile']
            st.progress(pct/100, text=f"**{factor['metric']}**: Top {100-pct:.0f}% (weight: {factor['weight']:.2f})")

# TAB 13: Uncertainty Quantification (NEW)
with tab13:
    st.subheader("📊 Uncertainty Quantification")
    st.markdown("How confident are we in the rankings? Bootstrap-based confidence intervals.")
    
    n_bootstrap = st.slider("Bootstrap iterations", 20, 100, 50)
    
    if st.button("📊 Run Uncertainty Analysis", type="primary"):
        with st.spinner(f"Running {n_bootstrap} bootstrap iterations..."):
            result = full_uncertainty_analysis(G, n_bootstrap=n_bootstrap, top_k=10)
        
        st.success("Uncertainty analysis complete!")
        
        st.markdown("### 🎯 High-Confidence Critical Nodes (>90% in top-10)")
        hc = result['high_confidence_critical']
        if hc:
            st.success(f"Nodes: {hc}")
        else:
            st.warning("No nodes are >90% confident in top-10")
        
        st.markdown("### 📊 Top-10 Probabilities")
        prob_df = result['top_k_probabilities'].head(10)
        fig = px.bar(prob_df, x='node', y='prob_top_k', 
                    title='Probability of Being in Top-10',
                    color='prob_top_k', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

# TAB 14: Domain-Specific (NEW)
with tab14:
    st.subheader("🎯 Domain-Specific Weighting")
    st.markdown("Use pre-trained weights optimized for your network type.")
    
    domains = get_available_domains()
    domain_choice = st.selectbox(
        "Select domain profile:",
        list(domains.keys()),
        format_func=lambda x: f"{domains[x]['name']}"
    )
    
    st.info(f"**{domains[domain_choice]['name']}**: {domains[domain_choice]['description']}")
    
    if st.button("🎯 Apply Domain Weights", type="primary"):
        with st.spinner("Analyzing with domain weights..."):
            result = domain_aware_analysis(G, domain=domain_choice)
        
        st.success(f"Applied {domain_choice} weights!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Domain Weights")
            for m, w in result['domain_weights'].items():
                st.write(f"**{m}**: {w:.2f}")
        with col2:
            st.markdown("### Comparison with CRITIC")
            overlap = result['comparison_with_critic']['overlap']
            st.metric("Top-10 Overlap", f"{overlap:.0%}")

# TAB 15: Adversarial Robustness (NEW)
with tab15:
    st.subheader("🛡️ Adversarial Robustness Testing")
    st.markdown("Test if an attacker can manipulate node rankings.")
    
    if st.button("🛡️ Run Adversarial Analysis", type="primary"):
        with st.spinner("Testing attack vectors..."):
            result = full_adversarial_analysis(G)
        
        st.success("Adversarial analysis complete!")
        
        grade = result['overall_grade']
        vuln = result['overall_vulnerability']
        
        if grade == 'A':
            st.success(f"**Overall Grade: {grade}** - Highly Robust! ({vuln:.0f}% vulnerable)")
        elif grade == 'B':
            st.warning(f"**Overall Grade: {grade}** - Moderately Robust ({vuln:.0f}% vulnerable)")
        else:
            st.error(f"**Overall Grade: {grade}** - Vulnerable to attacks ({vuln:.0f}% vulnerable)")
        
        st.markdown("### Node Robustness")
        for nr in result['node_robustness']:
            st.write(f"Node {nr['node']}: **Grade {nr['robustness_grade']}** ({nr['vulnerability_score']:.0f}% vulnerable)")
        
        st.markdown("### Recommendations")
        for rec in result['recommendations']:
            st.write(f"• {rec}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p>Critical Node Detection using CRITIC-TOPSIS Framework</p>
    <p><em>Advanced: Temporal Prediction | Explainable AI | Uncertainty | Domain Weights | Adversarial Robustness</em></p>
</div>
""", unsafe_allow_html=True)

