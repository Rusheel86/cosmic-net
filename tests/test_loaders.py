"""
test_loaders.py
Purpose: Unit tests for all data loaders (synthetic, tng, camels, camels_hf)
Tests: Schema validation, data quality, no NaN/Inf, range checks
"""

import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders.base_loader import (
    BaseDataLoader,
    HaloData,
    SubhaloData,
    get_loader,
    HaloDataset
)
from data.loaders.synthetic_loader import SyntheticLoader


@pytest.fixture
def synthetic_config():
    """Configuration for synthetic data loader."""
    return {
        'seed': 42,
        'data': {
            'source': 'synthetic',
            'grouping': 'fof',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'batch_size': 32,
            'num_workers': 0,
            'synthetic': {
                'features_path': 'data/raw/test_synthetic_features.pt',
                'csv_path': 'data/raw/test_synthetic_preprocessed_data.csv'
            }
        }
    }


@pytest.fixture
def synthetic_loader(synthetic_config):
    """Create synthetic data loader instance."""
    return SyntheticLoader(synthetic_config)


class TestSubhaloData:
    """Tests for SubhaloData dataclass."""

    def test_subhalo_creation(self):
        """Test SubhaloData object creation."""
        subhalo = SubhaloData(
            subhalo_id=1,
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([100.0, -50.0, 25.0]),
            stellar_mass=1e10,
            velocity_dispersion=200.0,
            half_mass_radius=0.01,
            metallicity=0.02
        )

        assert subhalo.subhalo_id == 1
        assert subhalo.stellar_mass == 1e10

    def test_get_node_features(self):
        """Test node feature extraction returns Tensor[4]."""
        subhalo = SubhaloData(
            subhalo_id=0,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            stellar_mass=1e10,
            velocity_dispersion=200.0,
            half_mass_radius=0.01,
            metallicity=0.02
        )

        features = subhalo.get_node_features()

        # Should be [log_stellar_mass, log_vel_disp, log_half_mass_r, log_metallicity]
        assert features.shape == (4,)
        assert features.dtype == np.float32

        # Values should be log-scaled
        assert features[0] == pytest.approx(10.0, abs=0.01)  # log10(1e10)
        assert features[1] == pytest.approx(np.log10(200), abs=0.01)

    def test_node_features_no_nan(self):
        """Test that node features handle edge cases without NaN."""
        # Test with zero values (should use epsilon)
        subhalo = SubhaloData(
            subhalo_id=0,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            stellar_mass=0.0,  # Edge case
            velocity_dispersion=0.0,
            half_mass_radius=0.0,
            metallicity=0.0
        )

        features = subhalo.get_node_features()

        assert not np.any(np.isnan(features)), "Features should not contain NaN"
        assert not np.any(np.isinf(features)), "Features should not contain Inf"


class TestHaloData:
    """Tests for HaloData dataclass."""

    def test_halo_creation(self):
        """Test HaloData object creation."""
        subhalos = [
            SubhaloData(
                subhalo_id=i,
                position=np.random.randn(3),
                velocity=np.random.randn(3) * 100,
                stellar_mass=1e10 * (1 + 0.1 * i),
                velocity_dispersion=200.0,
                half_mass_radius=0.01,
                metallicity=0.02
            )
            for i in range(5)
        ]

        halo = HaloData(
            cluster_id='test_halo_001',
            subhalos=subhalos,
            halo_mass=1e13,
            redshift=0.0
        )

        assert halo.cluster_id == 'test_halo_001'
        assert halo.num_subhalos == 5
        assert halo.halo_mass == 1e13

    def test_log_halo_mass(self):
        """Test log halo mass property."""
        halo = HaloData(
            cluster_id='test',
            subhalos=[],
            halo_mass=1e12
        )

        assert halo.log_halo_mass == pytest.approx(12.0, abs=0.01)

    def test_get_node_features_shape(self):
        """Test get_node_features returns Tensor[N, 4]."""
        n_subhalos = 10
        subhalos = [
            SubhaloData(
                subhalo_id=i,
                position=np.random.randn(3),
                velocity=np.random.randn(3) * 100,
                stellar_mass=1e10,
                velocity_dispersion=200.0,
                half_mass_radius=0.01,
                metallicity=0.02
            )
            for i in range(n_subhalos)
        ]

        halo = HaloData(
            cluster_id='test',
            subhalos=subhalos,
            halo_mass=1e12
        )

        features = halo.get_node_features()
        assert features.shape == (n_subhalos, 4)

    def test_get_positions_shape(self):
        """Test get_positions returns Tensor[N, 3]."""
        n_subhalos = 7
        subhalos = [
            SubhaloData(
                subhalo_id=i,
                position=np.array([float(i), 0.0, 0.0]),
                velocity=np.zeros(3),
                stellar_mass=1e10,
                velocity_dispersion=200.0,
                half_mass_radius=0.01,
                metallicity=0.02
            )
            for i in range(n_subhalos)
        ]

        halo = HaloData(cluster_id='test', subhalos=subhalos, halo_mass=1e12)
        positions = halo.get_positions()

        assert positions.shape == (n_subhalos, 3)


class TestSyntheticLoader:
    """Tests for SyntheticLoader."""

    def test_synthetic_loader_no_network_access(self, synthetic_loader):
        """Test that synthetic loader works without network access."""
        # This should generate synthetic data without any API calls
        halos = synthetic_loader.load()

        assert len(halos) > 0, "Should generate at least some halos"

    def test_output_schema_matches_contract(self, synthetic_loader):
        """Test that output matches base_loader contract."""
        halos = synthetic_loader.load()

        for halo in halos[:10]:  # Test first 10
            # Check HaloData attributes
            assert isinstance(halo.cluster_id, str)
            assert len(halo.subhalos) > 0
            assert isinstance(halo.halo_mass, float)

            # Check node features shape: Tensor[N, 4]
            features = halo.get_node_features()
            assert features.shape[1] == 4, f"Expected 4 node features, got {features.shape[1]}"

            # Check target: log10(M_halo)
            target = halo.log_halo_mass
            assert isinstance(target, float)

    def test_no_nan_or_inf_values(self, synthetic_loader):
        """Test that no tensors contain NaN or Inf."""
        halos = synthetic_loader.load()

        for halo in halos:
            features = halo.get_node_features()
            positions = halo.get_positions()
            velocities = halo.get_velocities()

            assert not np.any(np.isnan(features)), f"NaN in features for {halo.cluster_id}"
            assert not np.any(np.isinf(features)), f"Inf in features for {halo.cluster_id}"
            assert not np.any(np.isnan(positions)), f"NaN in positions for {halo.cluster_id}"
            assert not np.any(np.isnan(velocities)), f"NaN in velocities for {halo.cluster_id}"

    def test_node_features_log_scaled_range(self, synthetic_loader):
        """Test that log-scaled features are in plausible range [-6, 16]."""
        halos = synthetic_loader.load()

        all_features = []
        for halo in halos:
            all_features.append(halo.get_node_features())

        all_features = np.concatenate(all_features, axis=0)

        # Log-scaled features should be roughly in [-6, 16]
        # log10(1e-6) = -6, log10(1e16) = 16
        assert all_features.min() > -10, f"Features too small: {all_features.min()}"
        assert all_features.max() < 20, f"Features too large: {all_features.max()}"

    def test_target_in_physical_range(self, synthetic_loader):
        """Test that target (log10 M_halo) is in range [10, 16]."""
        halos = synthetic_loader.load()

        for halo in halos:
            target = halo.log_halo_mass

            # Halo masses should be 10^10 to 10^16 M_sun
            assert 10.0 <= target <= 16.0, \
                f"Target {target} outside physical range [10, 16] for {halo.cluster_id}"

    def test_split_ratios_correct(self, synthetic_loader):
        """Test that train/val/test split ratios are respected."""
        halos = synthetic_loader.load()
        train, val, test = synthetic_loader.split_data(halos)

        total = len(halos)
        train_ratio = len(train) / total
        val_ratio = len(val) / total
        test_ratio = len(test) / total

        # Allow 5% tolerance due to integer rounding
        assert abs(train_ratio - 0.7) < 0.05, f"Train ratio off: {train_ratio}"
        assert abs(val_ratio - 0.15) < 0.05, f"Val ratio off: {val_ratio}"
        assert abs(test_ratio - 0.15) < 0.05, f"Test ratio off: {test_ratio}"

    def test_deterministic_with_seed(self, synthetic_config):
        """Test that same seed produces identical data."""
        loader1 = SyntheticLoader(synthetic_config)
        loader2 = SyntheticLoader(synthetic_config)

        halos1 = loader1.load()
        halos2 = loader2.load()

        # Should have same number of halos
        assert len(halos1) == len(halos2)

        # First halo should be identical
        h1, h2 = halos1[0], halos2[0]
        assert h1.cluster_id == h2.cluster_id
        assert np.allclose(h1.get_node_features(), h2.get_node_features())

    def test_get_statistics(self, synthetic_loader):
        """Test statistics computation."""
        synthetic_loader.load()
        stats = synthetic_loader.get_statistics()

        expected_keys = [
            'num_halos', 'total_subhalos', 'feature_mean', 'feature_std',
            'log_mass_mean', 'log_mass_std', 'subhalos_per_halo_mean'
        ]

        for key in expected_keys:
            assert key in stats, f"Missing statistic: {key}"


class TestGetLoader:
    """Tests for loader factory function."""

    def test_get_loader_synthetic(self, synthetic_config):
        """Test factory returns correct loader for synthetic source."""
        loader = get_loader(synthetic_config)
        assert isinstance(loader, SyntheticLoader)

    def test_get_loader_invalid_source(self, synthetic_config):
        """Test factory raises error for invalid source."""
        synthetic_config['data']['source'] = 'invalid_source'

        with pytest.raises(ValueError, match="Unknown data source"):
            get_loader(synthetic_config)


class TestHaloDataset:
    """Tests for HaloDataset PyTorch Dataset wrapper."""

    def test_dataset_len(self, synthetic_loader):
        """Test dataset length matches halos list."""
        halos = synthetic_loader.load()[:20]
        dataset = HaloDataset(halos)

        assert len(dataset) == len(halos)

    def test_dataset_getitem(self, synthetic_loader):
        """Test dataset indexing returns correct halo."""
        halos = synthetic_loader.load()[:20]
        dataset = HaloDataset(halos)

        for i in range(min(5, len(halos))):
            assert dataset[i] == halos[i]


# Skip TNG/CAMELS tests in CI (they require network/API keys)
@pytest.mark.skipif(
    not Path('.env').exists(),
    reason="Skipping API-dependent tests (no .env file)"
)
class TestTNGLoader:
    """Tests for TNG loader (requires API key)."""

    def test_tng_loader_initialization(self):
        """Test TNG loader initializes without error."""
        config = {
            'seed': 42,
            'data': {
                'source': 'tng',
                'tng': {
                    'base_url': 'https://www.tng-project.org/api/TNG100-1/',
                    'snapshot': 99,
                    'max_halos': 10,
                    'cache_dir': 'data/raw/test_tng_cache'
                }
            }
        }

        from data.loaders.tng_loader import TNGLoader
        loader = TNGLoader(config)
        assert loader.snapshot == 99


@pytest.mark.skipif(
    not Path('.env').exists(),
    reason="Skipping API-dependent tests"
)
class TestCAMELSLoader:
    """Tests for CAMELS loader."""
    pass  # Would contain similar tests for CAMELS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
