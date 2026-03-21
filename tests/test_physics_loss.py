"""
test_physics_loss.py
Purpose: Unit tests for model/physics_loss.py
Tests: Virial theorem loss, gradient computation, batch handling, lambda weighting
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.physics_loss import (
    VirialLoss,
    PhysicsInformedLoss,
    compute_rmse,
    compute_mae,
    compute_r2_score,
    compute_scatter,
    MetricsComputer
)


@pytest.fixture
def config():
    """Default configuration for tests."""
    return {
        'physics': {
            'G': 4.302e-9,  # Mpc (km/s)^2 / M_sun
            'virial_coefficient': 2.0,
            'use_virial_loss': True
        },
        'training': {
            'virial_grad_clip': 1.0
        }
    }


@pytest.fixture
def virial_loss_fn(config):
    """Create VirialLoss instance."""
    return VirialLoss(config)


@pytest.fixture
def physics_loss_fn(config):
    """Create PhysicsInformedLoss instance."""
    return PhysicsInformedLoss(config)


def create_mock_batch(
    num_clusters: int = 4,
    nodes_per_cluster: int = 10,
    halo_mass_log: float = 12.0,
    stellar_mass: float = 1e10,
    vel_disp: float = 200.0,
    half_mass_r: float = 0.01
) -> Batch:
    """Create mock batch data for testing."""
    total_nodes = num_clusters * nodes_per_cluster

    # Create minimal batch structure
    batch = torch.repeat_interleave(
        torch.arange(num_clusters),
        nodes_per_cluster
    )

    # Physical quantities per node
    stellar_masses = torch.ones(total_nodes) * stellar_mass
    vel_disps = torch.ones(total_nodes) * vel_disp
    half_mass_rs = torch.ones(total_nodes) * half_mass_r

    # Create Batch object with required attributes
    data_list = []
    for i in range(num_clusters):
        data = Data(
            x=torch.randn(nodes_per_cluster, 4),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            y=torch.tensor([halo_mass_log])
        )
        data_list.append(data)

    batch_data = Batch.from_data_list(data_list)
    batch_data.stellar_mass = stellar_masses
    batch_data.vel_disp = vel_disps
    batch_data.half_mass_r = half_mass_rs

    return batch_data


class TestVirialLoss:
    """Tests for VirialLoss class."""

    def test_perfect_virial_state_low_loss(self, virial_loss_fn):
        """
        Test that a perfect virial state (2*KE = |PE|) returns near-zero loss.

        For virial equilibrium: 2 * KE = |PE|
        KE = 0.5 * M * sigma^2
        |PE| = G * M_halo^2 / R

        So: M * sigma^2 = G * M_halo^2 / R
        """
        # Design a virialized system
        G = 4.302e-9
        R = 0.5  # Mpc
        sigma = 300.0  # km/s
        M_stellar = 1e11  # M_sun per subhalo
        num_nodes = 10

        # For virial: 2 * (0.5 * N * M * sigma^2) = G * M_halo^2 / R
        # N * M * sigma^2 = G * M_halo^2 / R
        # M_halo = sqrt(N * M * sigma^2 * R / G)
        M_halo = np.sqrt(num_nodes * M_stellar * sigma ** 2 * R / G)
        log_M_halo = np.log10(M_halo)

        # Create batch
        batch_data = create_mock_batch(
            num_clusters=1,
            nodes_per_cluster=num_nodes,
            halo_mass_log=log_M_halo,
            stellar_mass=M_stellar,
            vel_disp=sigma,
            half_mass_r=R
        )

        predictions = torch.tensor([log_M_halo])

        loss, loss_dict = virial_loss_fn(predictions, batch_data, lambda_weight=1.0)

        # Virial ratio should be close to 1.0
        virial_ratio = loss_dict['virial_ratio_mean'].item()
        assert abs(virial_ratio - 1.0) < 0.1, f"Virial ratio should be ~1.0, got {virial_ratio}"

        # Loss should be small
        assert loss.item() < 0.01, f"Loss should be < 0.01 for virialized system, got {loss.item()}"

    def test_violated_virial_state_high_loss(self, virial_loss_fn):
        """Test that a non-virialized system returns significant loss."""
        # Create system with very wrong halo mass (10x too small)
        batch_data = create_mock_batch(
            num_clusters=1,
            nodes_per_cluster=10,
            halo_mass_log=11.0,  # Very small compared to virial expectation
            stellar_mass=1e11,
            vel_disp=500.0,  # High velocity dispersion
            half_mass_r=0.1
        )

        predictions = torch.tensor([11.0])  # log10(10^11) M_sun

        loss, loss_dict = virial_loss_fn(predictions, batch_data, lambda_weight=1.0)

        # Loss should be non-zero
        assert loss.item() > 0.01, f"Loss should be > 0.01 for non-virialized system"

    def test_loss_is_differentiable(self, virial_loss_fn):
        """Test that virial loss supports backpropagation."""
        batch_data = create_mock_batch(num_clusters=2, nodes_per_cluster=5)

        # Create predictions that require gradients
        predictions = torch.tensor([12.0, 12.5], requires_grad=True)

        loss, _ = virial_loss_fn(predictions, batch_data, lambda_weight=1.0)

        # Check that backward works
        loss.backward()

        # Gradient should exist and be finite
        assert predictions.grad is not None, "Gradient should exist"
        assert torch.isfinite(predictions.grad).all(), "Gradient should be finite"

    def test_batch_of_clusters(self, virial_loss_fn):
        """Test that loss works with multiple clusters in batch."""
        num_clusters = 8
        batch_data = create_mock_batch(
            num_clusters=num_clusters,
            nodes_per_cluster=15
        )

        predictions = torch.randn(num_clusters) + 12.0

        loss, loss_dict = virial_loss_fn(predictions, batch_data, lambda_weight=1.0)

        # Loss should be a scalar
        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

        # Should have computed virial ratio for each cluster
        assert 'virial_ratio_mean' in loss_dict

    def test_lambda_zero_returns_zero(self, virial_loss_fn):
        """Test that lambda=0 returns exactly zero loss."""
        batch_data = create_mock_batch(num_clusters=4)
        predictions = torch.randn(4) + 12.0

        loss, loss_dict = virial_loss_fn(predictions, batch_data, lambda_weight=0.0)

        assert loss.item() == 0.0, f"Loss with lambda=0 should be exactly 0, got {loss.item()}"

    def test_virial_loss_disabled(self, config):
        """Test that virial loss can be disabled via config."""
        config['physics']['use_virial_loss'] = False
        loss_fn = VirialLoss(config)

        batch_data = create_mock_batch(num_clusters=2)
        predictions = torch.randn(2) + 12.0

        loss, loss_dict = loss_fn(predictions, batch_data, lambda_weight=1.0)

        assert loss.item() == 0.0, "Disabled virial loss should return 0"
        assert loss_dict['virial_ratio_mean'].item() == 1.0


class TestPhysicsInformedLoss:
    """Tests for PhysicsInformedLoss class."""

    def test_total_loss_combines_mse_and_virial(self, physics_loss_fn):
        """Test that total loss is MSE + lambda * virial."""
        batch_data = create_mock_batch(num_clusters=4, nodes_per_cluster=10)
        predictions = torch.tensor([12.0, 12.5, 11.8, 12.2])
        targets = torch.tensor([12.1, 12.4, 11.9, 12.3])

        total_loss, loss_dict = physics_loss_fn(
            predictions, targets, batch_data, lambda_weight=1.0
        )

        mse = loss_dict['mse_loss'].item()
        virial = loss_dict['virial_loss'].item()

        # Total should be approximately MSE + virial
        expected_total = mse + virial
        assert abs(total_loss.item() - expected_total) < 1e-5

    def test_mse_only_computation(self, physics_loss_fn):
        """Test MSE-only computation for validation."""
        predictions = torch.tensor([12.0, 12.5, 11.8])
        targets = torch.tensor([12.1, 12.4, 11.9])

        mse = physics_loss_fn.compute_mse_only(predictions, targets)

        expected = ((predictions - targets) ** 2).mean()
        assert abs(mse.item() - expected.item()) < 1e-6

    def test_loss_dict_contains_all_components(self, physics_loss_fn):
        """Test that loss dict has all expected keys."""
        batch_data = create_mock_batch(num_clusters=2)
        predictions = torch.randn(2) + 12.0
        targets = torch.randn(2) + 12.0

        _, loss_dict = physics_loss_fn(predictions, targets, batch_data, lambda_weight=0.5)

        expected_keys = ['mse_loss', 'virial_loss', 'total_loss', 'lambda',
                         'virial_ratio_mean', 'ke_mean', 'pe_mean']
        for key in expected_keys:
            assert key in loss_dict, f"Missing key: {key}"


class TestMetrics:
    """Tests for metric computation functions."""

    def test_rmse(self):
        """Test RMSE computation."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])

        rmse = compute_rmse(preds, targets)
        expected = torch.sqrt(torch.mean((preds - targets) ** 2))

        assert abs(rmse.item() - expected.item()) < 1e-6

    def test_mae(self):
        """Test MAE computation."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])

        mae = compute_mae(preds, targets)
        expected = torch.mean(torch.abs(preds - targets))

        assert abs(mae.item() - expected.item()) < 1e-6

    def test_r2_perfect_prediction(self):
        """Test R² = 1 for perfect prediction."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        r2 = compute_r2_score(preds, targets)
        assert abs(r2.item() - 1.0) < 1e-6

    def test_r2_poor_prediction(self):
        """Test R² near 0 for poor prediction."""
        preds = torch.tensor([2.0, 2.0, 2.0])  # Predicting mean
        targets = torch.tensor([1.0, 2.0, 3.0])

        r2 = compute_r2_score(preds, targets)
        assert r2.item() < 0.1  # Should be near 0

    def test_scatter(self):
        """Test scatter (std of residuals) computation."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])

        scatter = compute_scatter(preds, targets)
        expected = (preds - targets).std()

        assert abs(scatter.item() - expected.item()) < 1e-6

    def test_metrics_computer_all(self):
        """Test MetricsComputer.compute_all returns all metrics."""
        preds = torch.randn(100) + 12.0
        targets = preds + torch.randn(100) * 0.1

        metrics = MetricsComputer.compute_all(preds, targets)

        expected_keys = ['mse', 'rmse', 'mae', 'r2', 'scatter']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
