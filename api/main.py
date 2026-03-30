"""
Critical Node Detection — FastAPI Backend
==========================================
REST API for the CRITIC-TOPSIS critical node detection framework.
Decouples computation from the frontend.

Run with:  uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Add project root's `src/` to path so we can import the algorithm modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from centralities import compute_all_centralities
from critic import compute_critic_weights, normalize_minmax
from topsis import topsis_rank, get_critical_nodes
from evaluation import (
    compare_attack_methods,
    compute_attack_effectiveness,
    get_ranking_from_centrality,
    get_ranking_from_topsis,
)
from cascading_failure import simulate_cascading_failure, cascade_over_fractions
from sensitivity_analysis import (
    sensitivity_to_normalization,
    sensitivity_to_centrality_removal,
    sensitivity_to_top_k,
)
from uncertainty import full_uncertainty_analysis
from adversarial import full_adversarial_analysis
from explainable_ai import explain_node, explain_top_k, generate_summary_report
from data_loading import (
    load_karate_club,
    load_les_miserables,
    load_florentine_families,
    load_dolphins,
    load_power_grid,
    load_usair,
    create_barabasi_albert,
    create_erdos_renyi,
    get_network_info,
)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Critical Node Detection API",
    description="CRITIC-TOPSIS multi-attribute framework for identifying critical nodes in complex networks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request / response models ──────────────────────────────────────

class AnalyzeRequest(BaseModel):
    network: str = Field("karate", description="Built-in network key or 'custom'")
    edges: Optional[List[List[str]]] = Field(None, description="Edge list for custom networks")
    top_k: int = Field(10, ge=1, le=50)
    normalization: str = Field("minmax", pattern="^(minmax|zscore)$")
    adaptive: bool = True
    variance_threshold: float = Field(0.01, ge=0.001, le=0.1)


class ImpactRequest(BaseModel):
    network: str = "karate"
    edges: Optional[List[List[str]]] = None
    fractions: Optional[List[float]] = None


class CascadeRequest(BaseModel):
    network: str = "karate"
    edges: Optional[List[List[str]]] = None
    capacity_factor: float = Field(1.2, ge=1.0, le=2.0)
    initial_fraction: float = Field(0.05, ge=0.01, le=0.3)


class RobustnessRequest(BaseModel):
    network: str = "karate"
    edges: Optional[List[List[str]]] = None
    n_bootstrap: int = Field(30, ge=10, le=200)
    top_k: int = Field(10, ge=1, le=50)


# ── Helper: resolve a NetworkX graph from a request ─────────────────────────

NETWORK_LOADERS = {
    "karate": load_karate_club,
    "les_miserables": load_les_miserables,
    "florentine": load_florentine_families,
    "dolphins": load_dolphins,
    "power_grid": load_power_grid,
    "usair": load_usair,
}


def _resolve_graph(network: str, edges: Optional[List[List[str]]] = None) -> nx.Graph:
    if network == "custom" and edges:
        G = nx.Graph()
        G.add_edges_from([tuple(e) for e in edges])
        G.name = f"Custom ({G.number_of_nodes()} nodes)"
        return G

    loader = NETWORK_LOADERS.get(network)
    if loader is None:
        if network.startswith("ba_"):
            parts = network.split("_")
            n = int(parts[1]) if len(parts) > 1 else 100
            m = int(parts[2]) if len(parts) > 2 else 3
            return create_barabasi_albert(n, m)
        if network.startswith("er_"):
            parts = network.split("_")
            n = int(parts[1]) if len(parts) > 1 else 100
            p = float(parts[2]) if len(parts) > 2 else 0.1
            return create_erdos_renyi(n, p)
        raise HTTPException(status_code=400, detail=f"Unknown network: {network}")
    return loader()


def _run_pipeline(G: nx.Graph, normalization: str = "minmax",
                  adaptive: bool = True, variance_threshold: float = 0.01):
    """Core CRITIC-TOPSIS pipeline. Returns (df_centrality, weights, topsis_results, excluded)."""
    df = compute_all_centralities(G, verbose=False)

    excluded_metrics: List[str] = []
    if adaptive:
        df_norm_check = normalize_minmax(df)
        variances = df_norm_check.var()
        low_var = variances[variances < variance_threshold].index.tolist()
        if low_var and len(low_var) < len(df.columns):
            excluded_metrics = low_var
            df = df.drop(columns=low_var)

    weights, details = compute_critic_weights(df, normalization=normalization, verbose=False)
    results, topsis_details = topsis_rank(df, weights, verbose=False)
    return df, weights, results, excluded_metrics


# ── Utility: safely convert numpy/pandas types to JSON-serializable dicts ───

def _safe(obj):
    """Recursively convert numpy / pandas objects to plain Python."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/networks")
def list_networks():
    """Return a catalogue of available built-in networks."""
    catalogue = []
    for key, loader in NETWORK_LOADERS.items():
        try:
            G = loader()
            info = get_network_info(G)
            catalogue.append({"key": key, **_safe(info)})
        except Exception:
            catalogue.append({"key": key, "name": key, "error": True})
    return catalogue


# ── 1. Core analysis ────────────────────────────────────────────────────────

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Run CRITIC-TOPSIS and return centralities, weights, rankings, explanations."""
    G = _resolve_graph(req.network, req.edges)
    info = get_network_info(G)
    df, weights, results, excluded = _run_pipeline(
        G, req.normalization, req.adaptive, req.variance_threshold
    )
    critical = get_critical_nodes(results, req.top_k)

    # Explanations for top-k
    explanations = []
    for node in critical[:5]:
        exp = explain_node(node, df, weights, results)
        explanations.append(_safe(exp))

    # Correlation matrix for centralities
    corr = df.corr()

    # Overlap comparison
    comparison = []
    topsis_set = set(critical)
    for col in df.columns:
        metric_top = set(df[col].nlargest(req.top_k).index)
        overlap = len(metric_top & topsis_set) / req.top_k * 100
        comparison.append({"metric": col, "overlap": overlap})

    return _safe({
        "network_info": info,
        "centralities": df.reset_index().to_dict(orient="records"),
        "centrality_columns": list(df.columns),
        "weights": weights.to_dict(),
        "rankings": results.reset_index().rename(columns={"index": "node"}).to_dict(orient="records"),
        "critical_nodes": [int(n) if isinstance(n, (int, np.integer)) else str(n) for n in critical],
        "excluded_metrics": excluded,
        "correlation": corr.to_dict(),
        "comparison": comparison,
        "explanations": explanations,
        "summary_report": generate_summary_report(df, weights, results),
    })


# ── 2. Impact / Attack simulation ──────────────────────────────────────────

@app.post("/api/impact")
def impact(req: ImpactRequest):
    """Run targeted removal attacks and return collapse curves."""
    G = _resolve_graph(req.network, req.edges)
    df, weights, results, _ = _run_pipeline(G)

    rankings = {"CRITIC-TOPSIS": get_ranking_from_topsis(results)}
    for col in ["degree", "betweenness", "closeness", "pagerank"]:
        if col in df.columns:
            rankings[col] = get_ranking_from_centrality(df, col)

    fractions = req.fractions or [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    attack_results = compare_attack_methods(G, rankings, fractions=fractions, verbose=False)
    effectiveness = compute_attack_effectiveness(attack_results)

    curves = {}
    for method, adf in attack_results.items():
        curves[method] = {
            "fraction_removed": adf["fraction_removed"].tolist(),
            "lcc_fraction": adf["lcc_fraction"].tolist(),
            "efficiency": adf["efficiency"].tolist(),
        }

    return _safe({
        "curves": curves,
        "effectiveness": effectiveness.to_dict(orient="records"),
    })


# ── 3. Cascading failure ───────────────────────────────────────────────────

@app.post("/api/cascade")
def cascade(req: CascadeRequest):
    """Simulate cascading failures."""
    G = _resolve_graph(req.network, req.edges)
    _, _, results, _ = _run_pipeline(G)
    ranking = get_ranking_from_topsis(results)

    n_initial = max(1, int(G.number_of_nodes() * req.initial_fraction))
    result = simulate_cascading_failure(
        G, ranking[:n_initial],
        capacity_factor=req.capacity_factor,
        verbose=False,
    )

    # Also run cascade over multiple fractions for the curve
    fractions_df = cascade_over_fractions(G, ranking, capacity_factor=req.capacity_factor)

    return _safe({
        "single": {k: v for k, v in result.items() if k != "cascade_history"},
        "cascade_history": result.get("cascade_history", []),
        "fraction_curve": fractions_df.to_dict(orient="records"),
    })


# ── 4. Robustness (sensitivity + uncertainty + adversarial) ─────────────────

@app.post("/api/robustness")
def robustness(req: RobustnessRequest):
    """Full robustness analysis: sensitivity, bootstrap, adversarial."""
    G = _resolve_graph(req.network, req.edges)

    # Sensitivity
    norm_sens = sensitivity_to_normalization(G)
    cent_impact = sensitivity_to_centrality_removal(G)
    topk_stab = sensitivity_to_top_k(G)

    # Uncertainty
    unc = full_uncertainty_analysis(G, n_bootstrap=req.n_bootstrap, top_k=req.top_k)

    # Adversarial
    adv = full_adversarial_analysis(G)

    return _safe({
        "sensitivity": {
            "normalization": norm_sens.to_dict(orient="records"),
            "centrality_impact": cent_impact.to_dict(orient="records"),
            "top_k_stability": topk_stab.to_dict(orient="records"),
        },
        "uncertainty": {
            "confidence_intervals": unc["confidence_intervals"].to_dict(orient="records"),
            "top_k_probabilities": unc["top_k_probabilities"].to_dict(orient="records"),
            "stability_metrics": unc["stability_metrics"],
            "high_confidence_critical": unc["high_confidence_critical"],
        },
        "adversarial": {
            "overall_grade": adv["overall_grade"],
            "overall_vulnerability": adv["overall_vulnerability"],
            "node_robustness": adv["node_robustness"],
            "recommendations": adv["recommendations"],
        },
    })


# ── 5. Theory content (static) ─────────────────────────────────────────────

@app.get("/api/theory")
def theory():
    """Return the theoretical background content for the UI."""
    theory_path = PROJECT_ROOT / "docs" / "THEORY.md"
    if theory_path.exists():
        return {"content": theory_path.read_text()}
    return {"content": "Theory document not found."}


# ── Health check ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}
