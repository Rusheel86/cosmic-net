from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Cosmic-Net API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model
    MODEL_PATH: str = "model_artifacts/best_model.pt"
    MODEL_HIDDEN_DIM: int = 64
    MODEL_NUM_LAYERS: int = 4
    MODEL_NUM_NODE_FEATURES: int = 6     # pos_x, pos_y, pos_z, stellar_mass, velocity_disp, metallicity
    MC_DROPOUT_PASSES: int = 50          # Monte Carlo uncertainty estimation

    # Graph construction
    KNN_K: int = 5                       # k-NN neighbours when building the graph

    # Physics
    VIRIAL_GREEN_THRESHOLD: float = 0.15  # |1 - Q| < 0.15 → green
    VIRIAL_AMBER_THRESHOLD: float = 0.35  # |1 - Q| < 0.35 → amber, else red

    # Symbolic equation (populated at startup from PySR output)
    # log(M_halo) = α·log(σ_vel) + β·metallicity + γ
    EQUATION_ALPHA: float = 2.31
    EQUATION_BETA: float = 0.87
    EQUATION_GAMMA: float = 11.42
    EQUATION_R2: float = 0.91
    EQUATION_LATEX: str = (
        r"\log(M_{\rm halo}) = 2.31\,\log(\sigma_{\rm vel}) "
        r"+ 0.87\,Z + 11.42"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
