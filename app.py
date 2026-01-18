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
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from centralities import compute_all_centralities
from critic import compute_critic_weights
from topsis import topsis_rank, get_critical_nodes
from evaluation import (compare_attack_methods, compute_attack_effectiveness,
                        get_ranking_from_centrality, get_ranking_from_topsis)
from data_loading import (load_karate_club, load_les_miserables, 
                          load_florentine_families, load_dolphins,
                          create_barabasi_albert, create_erdos_renyi,
                          get_network_info)

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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔍 Critical Node Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CRITIC-TOPSIS Multi-Attribute Framework</p>', unsafe_allow_html=True)

# Sidebar - Network Selection
st.sidebar.header("📊 Network Selection")

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

# Settings
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Top-k critical nodes", 5, 20, 10)

# Network info
info = get_network_info(G)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "📊 Centralities", "⚖️ CRITIC Weights", 
    "🏆 Rankings", "💥 Attack Simulation"
])

# Compute data (cached)
@st.cache_data
def compute_analysis(_G):
    df = compute_all_centralities(_G, verbose=False)
    weights, _ = compute_critic_weights(df, verbose=False)
    results, _ = topsis_rank(df, weights, verbose=False)
    return df, weights, results

df_centrality, weights, topsis_results = compute_analysis(G)
critical_nodes = get_critical_nodes(topsis_results, top_k)

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
    st.markdown("Seven different measures of node importance:")
    
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
    
    selected_metric = st.selectbox("Select centrality to visualize:", list(centrality_info.keys()))
    st.info(f"**{selected_metric.capitalize()}**: {centrality_info[selected_metric]}")
    
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
    st.markdown("""
    **CRITIC** determines objective weights based on:
    - **Standard deviation**: Higher variance = more discriminating
    - **Correlation**: Lower correlation = more unique information
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
            rankings = {
                'CRITIC-TOPSIS': get_ranking_from_topsis(topsis_results),
                'degree': get_ranking_from_centrality(df_centrality, 'degree'),
                'betweenness': get_ranking_from_centrality(df_centrality, 'betweenness'),
                'closeness': get_ranking_from_centrality(df_centrality, 'closeness'),
                'pagerank': get_ranking_from_centrality(df_centrality, 'pagerank'),
            }
            
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
            colors_eff = ['red' if m == 'CRITIC-TOPSIS' else 'steelblue' 
                         for m in eff_sorted['method']]
            
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p>Critical Node Detection using CRITIC-TOPSIS Framework</p>
    <p>Built with NetworkX, Streamlit, and Plotly</p>
</div>
""", unsafe_allow_html=True)
