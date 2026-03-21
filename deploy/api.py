"""
deploy/api.py
Purpose: FastAPI server for Cosmic-Net inference and explanation endpoints.
         Provides REST API for halo mass prediction with uncertainty quantification.
Inputs: HTTP requests with subhalo data
Outputs: JSON responses with predictions, uncertainties, and explanations
Config keys: api.host, api.port, api.checkpoint_path, api.cors_origins, api.rate_limit
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import load_model, CosmicNetGNN
from graph.graph_builder import GraphBuilder
from explain.explainer import CosmicNetExplainer, create_explainer
from data.loaders.base_loader import HaloData, SubhaloData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Pydantic Models for Request/Response
# ==============================================================================

class SubhaloInput(BaseModel):
    """Input schema for a single subhalo."""
    subhalo_id: int = Field(..., description="Unique ID for the subhalo")
    x: float = Field(..., description="X position in Mpc")
    y: float = Field(..., description="Y position in Mpc")
    z: float = Field(..., description="Z position in Mpc")
    vx: float = Field(0.0, description="X velocity in km/s")
    vy: float = Field(0.0, description="Y velocity in km/s")
    vz: float = Field(0.0, description="Z velocity in km/s")
    stellar_mass: float = Field(..., description="Stellar mass in M_sun")
    velocity_dispersion: float = Field(100.0, description="Velocity dispersion in km/s")
    half_mass_radius: float = Field(0.001, description="Half-mass radius in Mpc")
    metallicity: float = Field(0.02, description="Metallicity (dimensionless)")


class HaloInput(BaseModel):
    """Input schema for a halo prediction request."""
    cluster_id: str = Field(..., description="Unique cluster/halo identifier")
    subhalos: List[SubhaloInput] = Field(..., min_length=1, description="List of subhalos")


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    cluster_id: str
    M_halo_mean: float = Field(..., description="Mean predicted log10(M_halo / M_sun)")
    M_halo_std: float = Field(..., description="Standard deviation of prediction")
    confidence_95: List[float] = Field(..., description="95% confidence interval [low, high]")
    M_halo_linear: float = Field(..., description="Mean halo mass in M_sun (linear)")
    num_subhalos: int
    processing_time_ms: float


class ExplanationResponse(BaseModel):
    """Response schema for explanations."""
    cluster_id: str
    prediction: float
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_params: int


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    halos: List[HaloInput]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_halos: int
    total_processing_time_ms: float


# ==============================================================================
# Application State
# ==============================================================================

class AppState:
    """Global application state."""
    def __init__(self):
        self.model: Optional[CosmicNetGNN] = None
        self.config: Optional[Dict[str, Any]] = None
        self.device: Optional[torch.device] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.explainer: Optional[CosmicNetExplainer] = None
        self.is_ready: bool = False


app_state = AppState()


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="Cosmic-Net API",
    description="Physics-informed GNN for dark matter halo mass prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ==============================================================================
# Startup and Shutdown Events
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and initialize components on startup."""
    logger.info("Starting Cosmic-Net API...")

    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                app_state.config = yaml.safe_load(f)
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
            app_state.config = _get_default_config()

        # Setup device
        app_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {app_state.device}")

        # Load model
        api_config = app_state.config.get('api', {})
        checkpoint_path = api_config.get('checkpoint_path', 'outputs/checkpoints/best_model.pt')

        if Path(checkpoint_path).exists():
            app_state.model = load_model(checkpoint_path, app_state.config, app_state.device)
            logger.info(f"Model loaded from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            # Create untrained model for API structure validation
            from model.model import build_model
            app_state.model = build_model(app_state.config).to(app_state.device)
            logger.warning("Using untrained model - predictions will be meaningless")

        # Initialize graph builder
        app_state.graph_builder = GraphBuilder(app_state.config)

        # Initialize explainer
        app_state.explainer = create_explainer(
            app_state.model, app_state.config, app_state.device
        )

        # Configure CORS
        cors_origins = api_config.get('cors_origins', ['*'])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app_state.is_ready = True
        logger.info("Cosmic-Net API ready!")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Cosmic-Net API...")


# ==============================================================================
# Helper Functions
# ==============================================================================

def _get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        'seed': 42,
        'model': {
            'node_features': 4,
            'edge_features': 5,
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 3,
            'dropout': 0.2,
            'pooling': 'mean',
            'residual': True,
            'mc_dropout': True,
            'mc_samples': 50,
            'activation': 'leaky_relu'
        },
        'graph': {
            'method': 'radius',
            'radius_mpc': 2.0,
            'self_loops': True,
            'edge_features': ['distance', 'delta_v', 'cos_theta', 'mass_ratio', 'proj_sep']
        },
        'api': {
            'checkpoint_path': 'outputs/checkpoints/best_model.pt',
            'cors_origins': ['*']
        },
        'explain': {
            'method': 'pgexplainer',
            'top_k_nodes': 10,
            'top_k_edges': 20,
            'output_dir': 'outputs/explanations'
        }
    }


def _convert_input_to_halo(halo_input: HaloInput) -> HaloData:
    """Convert API input to HaloData object."""
    subhalos = []
    for sub in halo_input.subhalos:
        subhalo = SubhaloData(
            subhalo_id=sub.subhalo_id,
            position=np.array([sub.x, sub.y, sub.z], dtype=np.float32),
            velocity=np.array([sub.vx, sub.vy, sub.vz], dtype=np.float32),
            stellar_mass=sub.stellar_mass,
            velocity_dispersion=sub.velocity_dispersion,
            half_mass_radius=sub.half_mass_radius,
            metallicity=sub.metallicity
        )
        subhalos.append(subhalo)

    return HaloData(
        cluster_id=halo_input.cluster_id,
        subhalos=subhalos,
        halo_mass=1e12,  # Placeholder, will be predicted
        redshift=0.0
    )


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if app_state.is_ready else "initializing",
        model_loaded=app_state.model is not None,
        device=str(app_state.device) if app_state.device else "unknown",
        model_params=app_state.model.count_parameters() if app_state.model else 0
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return await root()


@app.post("/predict", response_model=PredictionResponse)
async def predict(halo_input: HaloInput):
    """
    Predict halo mass from subhalo data.

    Returns prediction with MC-Dropout uncertainty estimates.
    """
    import time
    start_time = time.time()

    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        # Convert input to HaloData
        halo_data = _convert_input_to_halo(halo_input)

        # Build graph
        graph = app_state.graph_builder.build_graph(halo_data)
        graph = graph.to(app_state.device)

        # Create batch
        from torch_geometric.data import Batch
        batch_data = Batch.from_data_list([graph])

        # Predict with uncertainty
        uncertainty = app_state.model.predict_with_uncertainty(batch_data)

        # Extract results
        mean_pred = float(uncertainty['mean'][0])
        std_pred = float(uncertainty['std'][0])
        conf_low = float(uncertainty['confidence_95_low'][0])
        conf_high = float(uncertainty['confidence_95_high'][0])

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            cluster_id=halo_input.cluster_id,
            M_halo_mean=mean_pred,
            M_halo_std=std_pred,
            confidence_95=[conf_low, conf_high],
            M_halo_linear=10 ** mean_pred,
            num_subhalos=len(halo_input.subhalos),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple halos.
    """
    import time
    start_time = time.time()

    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    predictions = []
    for halo_input in request.halos:
        try:
            pred = await predict(halo_input)
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Failed to predict {halo_input.cluster_id}: {e}")

    total_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_halos=len(request.halos),
        total_processing_time_ms=total_time
    )


@app.post("/explain", response_model=ExplanationResponse)
async def explain(halo_input: HaloInput):
    """
    Generate explanation for a halo prediction.

    Returns node and edge importance masks in Three.js-ready format.
    """
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        # Convert input to HaloData
        halo_data = _convert_input_to_halo(halo_input)

        # Build graph
        graph = app_state.graph_builder.build_graph(halo_data)
        graph = graph.to(app_state.device)

        # Generate explanation
        result = app_state.explainer.explain(graph)

        return ExplanationResponse(
            cluster_id=result.cluster_id,
            prediction=result.prediction,
            nodes=[
                {
                    'id': n.id,
                    'importance': n.importance,
                    'x': n.x,
                    'y': n.y,
                    'z': n.z,
                    'stellar_mass': n.stellar_mass
                }
                for n in result.nodes
            ],
            edges=[
                {
                    'source': e.source,
                    'target': e.target,
                    'importance': e.importance,
                    'distance': e.distance
                }
                for e in result.edges
            ],
            metadata=result.metadata
        )

    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Return current configuration."""
    if app_state.config is None:
        raise HTTPException(status_code=503, detail="Config not loaded")
    return app_state.config


@app.get("/model/info")
async def model_info():
    """Return model information."""
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        'parameters': app_state.model.count_parameters(),
        'num_layers': app_state.config.get('model', {}).get('num_layers', 3),
        'hidden_dim': app_state.config.get('model', {}).get('hidden_dim', 128),
        'pooling': app_state.config.get('model', {}).get('pooling', 'mean'),
        'mc_dropout': app_state.config.get('model', {}).get('mc_dropout', True),
        'device': str(app_state.device)
    }


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "deploy.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cosmic-Net API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)
