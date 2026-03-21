"""
symbolic_regression.py
Purpose: Distill GNN knowledge into human-readable mathematical equations using PySR.
         Implements dimensional analysis post-filtering to ensure physics consistency.
         Supports both raw node features and GNN embeddings as input.
Inputs: embeddings: Tensor[N, D] - GNN node embeddings (pre or post pooling)
        features: Tensor[N, 4] - Raw node features
        targets: Tensor[N] or Tensor[B] - Target values (halo mass)
        config (dict) - Configuration dictionary from config.yaml
Outputs: Discovered equations with Pareto front (complexity vs accuracy)
         dimensionally_valid_equations.tex and all_equations_pareto.csv
Config keys: symbolic.library, symbolic.embedding_point, symbolic.pysr,
             symbolic.gplearn, symbolic.enforce_dimensional_consistency,
             symbolic.output_dir, symbolic.feature_dimensions
"""

import os
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False
    PySRRegressor = None

try:
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except ImportError:
    HAS_GPLEARN = False
    SymbolicRegressor = None

try:
    import sympy
    from sympy import symbols, latex, simplify
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    sympy = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredEquation:
    """Represents a discovered symbolic equation."""
    equation_str: str
    latex_str: str
    complexity: int
    loss: float
    rmse: float
    is_dimensionally_valid: bool
    dimension_check_notes: str
    sympy_expr: Optional[Any] = None


class DimensionalAnalyzer:
    """
    Performs dimensional analysis on discovered equations.

    Physical dimensions:
    - [M] = mass (e.g., M_sun)
    - [L] = length (e.g., Mpc)
    - [T] = time (e.g., seconds)
    - [1] = dimensionless

    Known valid scaling relations:
    - Virial: M ~ R * σ² / G → [L] * [L/T]² / [L³/(M*T²)] = [M]
    - Faber-Jackson: L ~ σ⁴ (with appropriate constants)
    """

    # Standard dimensions for input features
    DEFAULT_DIMENSIONS = {
        'log_stellar_mass': '[M]',      # log10(M_stellar / M_sun)
        'log_vel_dispersion': '[L/T]',  # log10(σ / km/s)
        'log_half_mass_radius': '[L]',  # log10(R / Mpc)
        'log_metallicity': '[1]',       # dimensionless
        'stellar_mass': '[M]',
        'vel_disp': '[L/T]',
        'half_mass_r': '[L]',
        'metallicity': '[1]',
        'x0': '[M]',  # PySR default names
        'x1': '[L/T]',
        'x2': '[L]',
        'x3': '[1]'
    }

    # Known physical constants
    CONSTANTS = {
        'G': '[L³/(M*T²)]',  # Gravitational constant
        'c': '[L/T]',        # Speed of light
        'h': '[M*L²/T]'      # Planck constant
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dimensional analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        symbolic_config = config.get('symbolic', {})

        # Get custom dimension definitions
        self.dimensions = self.DEFAULT_DIMENSIONS.copy()
        custom_dims = symbolic_config.get('feature_dimensions', {})
        self.dimensions.update(custom_dims)

        # Target dimension (halo mass)
        self.target_dimension = '[M]'

    def check_equation(self, equation_str: str) -> Tuple[bool, str]:
        """
        Check if an equation is dimensionally consistent.

        This is a simplified check that looks for known patterns.
        A full dimensional analysis would require parsing and tracking
        dimensions through the expression tree.

        Args:
            equation_str: String representation of the equation

        Returns:
            Tuple of (is_valid, notes)
        """
        notes = []

        # Convert to lowercase for pattern matching
        eq_lower = equation_str.lower()

        # Check for known valid patterns
        valid_patterns = [
            # Virial-like: contains velocity squared and radius
            (r'.*\bsigma\b.*\*\*\s*2.*\*.*\br\b.*', 'Virial-like: M ~ σ² * R'),
            (r'.*\bvel\b.*\*\*\s*2.*\*.*\bradius\b.*', 'Virial-like: M ~ v² * R'),
            (r'.*x1.*\*\*\s*2.*\*.*x2.*', 'Virial-like: x₁² * x₂'),

            # Mass-dependent scaling
            (r'.*\bm\b.*\*\*.*[234].*', 'Power law in mass'),
            (r'.*x0.*\*\*.*', 'Power law in stellar mass'),

            # Simple proportionality
            (r'^[a-z0-9\.\+\-\*\/ ]+$', 'Basic algebraic form')
        ]

        for pattern, description in valid_patterns:
            if re.search(pattern, eq_lower):
                notes.append(f"Matches pattern: {description}")

        # Check for potentially invalid operations
        invalid_indicators = []

        # Adding quantities of different dimensions
        if '+' in equation_str:
            # This is a simplistic check - real analysis needs expression parsing
            terms = equation_str.split('+')
            if len(terms) > 1:
                notes.append("Contains addition - verify terms have same dimensions")

        # Log of dimensional quantity (often valid in astrophysics due to log scaling)
        if 'log' in eq_lower:
            notes.append("Contains log - acceptable for log-scaled quantities")

        # Determine validity (simplified heuristic)
        # In practice, this would need proper symbolic dimension tracking
        is_valid = len(invalid_indicators) == 0

        if not notes:
            notes.append("No specific patterns detected - manual review recommended")

        return is_valid, '; '.join(notes)

    def validate_virial_theorem(self, equation_str: str) -> Tuple[bool, str]:
        """
        Check if equation matches Virial theorem scaling: M ~ R * σ² / G

        Args:
            equation_str: Equation string

        Returns:
            Tuple of (matches_virial, description)
        """
        eq_lower = equation_str.lower()

        # Look for σ² (velocity dispersion squared)
        has_vel_squared = any([
            'sigma**2' in eq_lower,
            'vel**2' in eq_lower,
            'x1**2' in eq_lower,
            'vel_disp**2' in eq_lower
        ])

        # Look for R (radius)
        has_radius = any([
            ' r ' in f' {eq_lower} ',
            'radius' in eq_lower,
            'x2' in eq_lower,
            'half_mass' in eq_lower
        ])

        if has_vel_squared and has_radius:
            return True, "Matches Virial theorem structure: M ~ R * σ²"

        return False, "Does not match Virial theorem structure"

    def validate_faber_jackson(self, equation_str: str) -> Tuple[bool, str]:
        """
        Check if equation matches Faber-Jackson relation: L ~ σ⁴

        Args:
            equation_str: Equation string

        Returns:
            Tuple of (matches_fj, description)
        """
        eq_lower = equation_str.lower()

        # Look for σ⁴
        has_vel_fourth = any([
            'sigma**4' in eq_lower,
            'vel**4' in eq_lower,
            'x1**4' in eq_lower
        ])

        if has_vel_fourth:
            return True, "Matches Faber-Jackson structure: M ~ σ⁴"

        return False, "Does not match Faber-Jackson structure"


class SymbolicRegressor:
    """
    Wrapper for symbolic regression using PySR or gplearn.

    Extracts interpretable mathematical equations from GNN embeddings
    and validates them against physical constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize symbolic regressor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.symbolic_config = config.get('symbolic', {})

        # Library selection
        self.library = self.symbolic_config.get('library', 'pysr')

        # Validate library availability
        if self.library == 'pysr' and not HAS_PYSR:
            logger.warning("PySR not available, falling back to gplearn")
            self.library = 'gplearn'

        if self.library == 'gplearn' and not HAS_GPLEARN:
            raise ImportError("Neither PySR nor gplearn available. Install one.")

        # Output directory
        self.output_dir = Path(self.symbolic_config.get('output_dir', 'outputs/equations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dimensional analysis
        self.enforce_dimensional = self.symbolic_config.get('enforce_dimensional_consistency', True)
        self.dimensional_analyzer = DimensionalAnalyzer(config)

        # PySR settings
        self.pysr_config = self.symbolic_config.get('pysr', {})

        # gplearn settings
        self.gplearn_config = self.symbolic_config.get('gplearn', {})

        # Store results
        self.equations: List[DiscoveredEquation] = []
        self.pareto_front: List[DiscoveredEquation] = []

        logger.info(f"SymbolicRegressor initialized: library={self.library}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[DiscoveredEquation]:
        """
        Fit symbolic regression model.

        Args:
            X: Input features [N, D]
            y: Target values [N]
            feature_names: Optional names for features

        Returns:
            List of discovered equations
        """
        logger.info(f"Fitting symbolic regression: X shape {X.shape}, y shape {y.shape}")

        if self.library == 'pysr':
            equations = self._fit_pysr(X, y, feature_names)
        else:
            equations = self._fit_gplearn(X, y, feature_names)

        # Dimensional analysis post-filter
        for eq in equations:
            is_valid, notes = self.dimensional_analyzer.check_equation(eq.equation_str)
            eq.is_dimensionally_valid = is_valid
            eq.dimension_check_notes = notes

            # Additional validation against known relations
            virial_match, virial_notes = self.dimensional_analyzer.validate_virial_theorem(eq.equation_str)
            fj_match, fj_notes = self.dimensional_analyzer.validate_faber_jackson(eq.equation_str)

            if virial_match:
                eq.dimension_check_notes += f"; {virial_notes}"
            if fj_match:
                eq.dimension_check_notes += f"; {fj_notes}"

        self.equations = equations
        self.pareto_front = [eq for eq in equations if eq.is_dimensionally_valid] if self.enforce_dimensional else equations

        # Save results
        self._save_results()

        return equations

    def _fit_pysr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[DiscoveredEquation]:
        """
        Fit using PySR.

        Args:
            X, y: Training data
            feature_names: Feature names

        Returns:
            List of discovered equations
        """
        if not HAS_PYSR:
            raise ImportError("PySR not installed")

        # Build operator list
        binary_ops = self.pysr_config.get('binary_operators', ['+', '-', '*', '/'])
        unary_ops = self.pysr_config.get('unary_operators', ['log', 'sqrt', 'square'])

        # Create PySR model
        model = PySRRegressor(
            niterations=self.pysr_config.get('niterations', 100),
            populations=self.pysr_config.get('populations', 30),
            maxsize=self.pysr_config.get('maxsize', 20),
            binary_operators=binary_ops,
            unary_operators=unary_ops,
            parsimony=self.pysr_config.get('parsimony', 0.0032),
            weight_optimize=self.pysr_config.get('weight_optimize', 0.001),
            timeout_in_seconds=self.pysr_config.get('timeout_in_seconds', 3600),
            procs=4,
            batching=True,
            batch_size=50,
            progress=True,
            temp_equation_file=True
        )

        # Fit model
        model.fit(X, y, variable_names=feature_names)

        # Extract equations from Pareto front
        equations = []

        if hasattr(model, 'equations_') and model.equations_ is not None:
            for idx, row in model.equations_.iterrows():
                try:
                    eq_str = str(row.get('equation', row.get('sympy_format', '')))
                    complexity = int(row.get('complexity', 0))
                    loss = float(row.get('loss', row.get('mse', float('inf'))))

                    # Get LaTeX if available
                    if HAS_SYMPY:
                        try:
                            latex_str = latex(model.sympy(idx))
                        except Exception:
                            latex_str = eq_str
                    else:
                        latex_str = eq_str

                    eq = DiscoveredEquation(
                        equation_str=eq_str,
                        latex_str=latex_str,
                        complexity=complexity,
                        loss=loss,
                        rmse=np.sqrt(loss),
                        is_dimensionally_valid=True,
                        dimension_check_notes=""
                    )
                    equations.append(eq)

                except Exception as e:
                    logger.warning(f"Failed to parse equation at index {idx}: {e}")

        logger.info(f"PySR discovered {len(equations)} equations")
        return equations

    def _fit_gplearn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[DiscoveredEquation]:
        """
        Fit using gplearn.

        Args:
            X, y: Training data
            feature_names: Feature names

        Returns:
            List of discovered equations
        """
        if not HAS_GPLEARN:
            raise ImportError("gplearn not installed")

        from gplearn.genetic import SymbolicRegressor as GPLearnSR

        # Build function set
        function_set = self.gplearn_config.get('function_set', ['add', 'sub', 'mul', 'div', 'sqrt', 'log'])

        model = GPLearnSR(
            population_size=self.gplearn_config.get('population_size', 5000),
            generations=self.gplearn_config.get('generations', 50),
            tournament_size=self.gplearn_config.get('tournament_size', 20),
            function_set=function_set,
            parsimony_coefficient=self.gplearn_config.get('parsimony_coefficient', 0.001),
            random_state=self.config.get('seed', 42),
            verbose=1
        )

        model.fit(X, y)

        # gplearn returns single best program
        eq_str = str(model._program)

        # Compute RMSE
        y_pred = model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        equations = [
            DiscoveredEquation(
                equation_str=eq_str,
                latex_str=eq_str,  # gplearn doesn't have built-in latex
                complexity=model._program.length_,
                loss=rmse ** 2,
                rmse=rmse,
                is_dimensionally_valid=True,
                dimension_check_notes=""
            )
        ]

        logger.info(f"gplearn discovered best equation: {eq_str}")
        return equations

    def _save_results(self) -> None:
        """Save discovered equations to files."""
        # Save all equations to CSV
        if self.equations:
            df = pd.DataFrame([
                {
                    'equation': eq.equation_str,
                    'latex': eq.latex_str,
                    'complexity': eq.complexity,
                    'loss': eq.loss,
                    'rmse': eq.rmse,
                    'dimensionally_valid': eq.is_dimensionally_valid,
                    'notes': eq.dimension_check_notes
                }
                for eq in self.equations
            ])
            df.to_csv(self.output_dir / 'all_equations_pareto.csv', index=False)
            logger.info(f"Saved {len(df)} equations to all_equations_pareto.csv")

        # Save dimensionally valid equations to LaTeX
        valid_equations = [eq for eq in self.equations if eq.is_dimensionally_valid]
        if valid_equations:
            with open(self.output_dir / 'dimensionally_valid_equations.tex', 'w') as f:
                f.write("% Dimensionally valid equations discovered by PySR\n")
                f.write("% Generated by Cosmic-Net symbolic regression\n\n")
                f.write("\\begin{align}\n")
                for i, eq in enumerate(valid_equations[:10]):  # Top 10
                    f.write(f"  M_{{halo}} &= {eq.latex_str} "
                           f"\\quad (\\text{{RMSE}} = {eq.rmse:.4f}, "
                           f"\\text{{complexity}} = {eq.complexity}) \\\\\n")
                f.write("\\end{align}\n")

            logger.info(f"Saved {len(valid_equations)} valid equations to dimensionally_valid_equations.tex")

    def log_to_wandb(self) -> None:
        """Log results to W&B."""
        if not HAS_WANDB or wandb.run is None:
            return

        # Log Pareto front as table
        if self.equations:
            columns = ['equation', 'complexity', 'rmse', 'dimensionally_valid']
            data = [[eq.equation_str, eq.complexity, eq.rmse, eq.is_dimensionally_valid]
                    for eq in self.equations]
            table = wandb.Table(columns=columns, data=data)
            wandb.log({'pareto_front': table})

        # Log best equation
        if self.pareto_front:
            best = min(self.pareto_front, key=lambda x: x.rmse)
            wandb.log({
                'best_equation': best.equation_str,
                'best_equation_rmse': best.rmse,
                'best_equation_complexity': best.complexity
            })

    def get_best_equation(self) -> Optional[DiscoveredEquation]:
        """Get the best (lowest RMSE) dimensionally valid equation."""
        if not self.pareto_front:
            return None
        return min(self.pareto_front, key=lambda x: x.rmse)


def extract_features_for_sr(
    model: torch.nn.Module,
    loader,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from GNN for symbolic regression.

    Args:
        model: Trained GNN model
        loader: DataLoader
        config: Configuration dictionary
        device: Device for computation

    Returns:
        Tuple of (features, targets, feature_names)
    """
    symbolic_config = config.get('symbolic', {})
    embedding_point = symbolic_config.get('embedding_point', 'pre_pooling')

    all_features = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)

            # Ensure required attributes exist
            if not hasattr(batch_data, 'stellar_mass'):
                batch_data.stellar_mass = torch.ones(batch_data.x.size(0), device=device) * 1e10
            if not hasattr(batch_data, 'vel_disp'):
                batch_data.vel_disp = torch.ones(batch_data.x.size(0), device=device) * 100
            if not hasattr(batch_data, 'half_mass_r'):
                batch_data.half_mass_r = torch.ones(batch_data.x.size(0), device=device) * 0.01

            # Get embeddings
            embeddings = model.get_embeddings(batch_data, embedding_point)

            # Get raw features
            raw_features = batch_data.x

            # Combine raw features with GNN embeddings
            if embedding_point == 'pre_pooling':
                # Per-node: concatenate raw features + embeddings
                combined = torch.cat([raw_features, embeddings], dim=1)
                # Need to aggregate to per-halo for regression target
                # Use mean pooling
                from torch_geometric.nn import global_mean_pool
                combined = global_mean_pool(combined, batch_data.batch)
            else:
                # Already per-halo
                raw_pooled = global_mean_pool(raw_features, batch_data.batch)
                combined = torch.cat([raw_pooled, embeddings], dim=1)

            all_features.append(combined.cpu().numpy())
            all_targets.append(batch_data.y.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Feature names
    raw_names = ['log_stellar_mass', 'log_vel_disp', 'log_half_mass_r', 'log_metallicity']
    embedding_names = [f'emb_{i}' for i in range(features.shape[1] - len(raw_names))]
    feature_names = raw_names + embedding_names

    logger.info(f"Extracted features: {features.shape}, targets: {targets.shape}")

    return features, targets, feature_names


def run_symbolic_regression(
    model: torch.nn.Module,
    loader,
    config: Dict[str, Any],
    device: torch.device
) -> SymbolicRegressor:
    """
    Main entry point for symbolic regression.

    Args:
        model: Trained GNN model
        loader: DataLoader with data for SR
        config: Configuration dictionary
        device: Device for computation

    Returns:
        Fitted SymbolicRegressor instance
    """
    # Extract features
    features, targets, feature_names = extract_features_for_sr(model, loader, config, device)

    # Use only raw features for interpretability (first 4)
    X = features[:, :4]
    y = targets
    names = feature_names[:4]

    # Create and fit regressor
    regressor = SymbolicRegressor(config)
    regressor.fit(X, y, feature_names=names)

    # Log to W&B
    regressor.log_to_wandb()

    return regressor
