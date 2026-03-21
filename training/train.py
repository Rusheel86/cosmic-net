"""
train.py
Purpose: Main training loop for Cosmic-Net GNN with W&B logging, checkpointing,
         and support for cross-simulation generalization experiments.
Inputs: config (dict) - Configuration dictionary from config.yaml
        train_loader, val_loader, test_loader - PyG DataLoaders
Outputs: Trained model checkpoint, W&B logs, evaluation metrics
Config keys: training.epochs, training.learning_rate, training.optimizer,
             training.train_source, training.test_source, training.checkpoint_dir,
             wandb.enabled, wandb.project, wandb.entity
"""

import os
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

from dotenv import load_dotenv

from model.model import CosmicNetGNN, build_model
from model.physics_loss import PhysicsInformedLoss, MetricsComputer
from training.scheduler import (
    build_optimizer,
    build_lr_scheduler,
    LambdaScheduler,
    EarlyStopping,
    TrainingState,
    get_current_lr
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for Cosmic-Net GNN.

    Handles:
    - Training loop with physics-informed loss
    - Validation and evaluation
    - Checkpointing and early stopping
    - W&B logging
    - Cross-simulation generalization experiments
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: CosmicNetGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary from config.yaml
            model: The GNN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            device: Device to train on (auto-detect if None)
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.wandb_config = config.get('wandb', {})

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Model
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss function
        self.loss_fn = PhysicsInformedLoss(config)

        # Optimizer and schedulers
        self.optimizer = build_optimizer(self.model, config)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, config)
        self.lambda_scheduler = LambdaScheduler(config)

        # Early stopping
        patience = self.training_config.get('early_stopping_patience', 20)
        self.early_stopping = EarlyStopping(patience=patience)

        # Training state
        self.state = TrainingState(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            lambda_scheduler=self.lambda_scheduler,
            early_stopping=self.early_stopping
        )

        # Gradient clipping
        self.grad_clip = self.training_config.get('grad_clip', 1.0)

        # Checkpointing
        self.checkpoint_dir = Path(self.training_config.get('checkpoint_dir', 'outputs/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = self.training_config.get('save_every', 10)
        self.save_best = self.training_config.get('save_best', True)

        # Cross-simulation experiment detection
        train_source = self.training_config.get('train_source', 'synthetic')
        test_source = self.training_config.get('test_source', 'synthetic')
        self.is_cross_sim = train_source != test_source
        if self.is_cross_sim:
            logger.info(f"Cross-simulation experiment: train on {train_source}, test on {test_source}")

        # W&B initialization
        self.use_wandb = self.wandb_config.get('enabled', True) and HAS_WANDB
        self._init_wandb()

        logger.info(f"Trainer initialized on device: {self.device}")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        if not self.use_wandb:
            logger.info("W&B logging disabled")
            return

        try:
            # Get API key from environment
            api_key = os.environ.get('WANDB_API_KEY')
            if api_key:
                wandb.login(key=api_key)

            # Build run name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = self.wandb_config.get('run_name_prefix', 'cosmicnet')
            run_name = f"{prefix}_{timestamp}"

            # Tags
            tags = list(self.wandb_config.get('tags', ['gnn', 'physics-informed']))
            if self.is_cross_sim:
                tags.append('cross_sim')

            # Initialize W&B run
            wandb.init(
                project=self.wandb_config.get('project', 'cosmic-net'),
                entity=self.wandb_config.get('entity') or os.environ.get('WANDB_ENTITY'),
                name=run_name,
                config=self.config,
                tags=tags,
                reinit=True
            )

            # Watch model
            if self.wandb_config.get('watch_model', True):
                wandb.watch(
                    self.model,
                    log='all',
                    log_freq=self.wandb_config.get('watch_freq', 100)
                )

            logger.info(f"W&B initialized: {wandb.run.name}")

        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_virial = 0.0
        num_batches = 0

        lambda_value = self.lambda_scheduler.step(epoch)

        for batch_data in self.train_loader:
            batch_data = batch_data.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(batch_data)

            # Compute loss
            targets = batch_data.y
            loss, loss_dict = self.loss_fn(predictions, targets, batch_data, lambda_value)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss_dict['total_loss'].item()
            total_mse += loss_dict['mse_loss'].item()
            total_virial += loss_dict['virial_loss'].item()
            num_batches += 1

        # Average metrics
        metrics = {
            'train/loss': total_loss / max(num_batches, 1),
            'train/mse': total_mse / max(num_batches, 1),
            'train/virial_loss': total_virial / max(num_batches, 1),
            'train/lambda': lambda_value,
            'train/lr': get_current_lr(self.optimizer)
        }

        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader, prefix: str = 'val') -> Dict[str, float]:
        """
        Validate on a data loader.

        Args:
            loader: DataLoader to validate on
            prefix: Metric prefix (e.g., 'val' or 'test')

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        total_loss = 0.0
        total_mse = 0.0
        total_virial = 0.0
        num_batches = 0

        lambda_value = self.lambda_scheduler.get_lambda()

        for batch_data in loader:
            batch_data = batch_data.to(self.device)

            # Forward pass
            predictions, _ = self.model(batch_data)
            targets = batch_data.y

            # Compute loss
            loss, loss_dict = self.loss_fn(predictions, targets, batch_data, lambda_value)

            # Accumulate
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            total_loss += loss_dict['total_loss'].item()
            total_mse += loss_dict['mse_loss'].item()
            total_virial += loss_dict['virial_loss'].item()
            num_batches += 1

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Compute additional metrics
        detailed_metrics = MetricsComputer.compute_all(all_predictions, all_targets)

        metrics = {
            f'{prefix}/loss': total_loss / max(num_batches, 1),
            f'{prefix}/mse': total_mse / max(num_batches, 1),
            f'{prefix}/virial_loss': total_virial / max(num_batches, 1),
            f'{prefix}/rmse': detailed_metrics['rmse'],
            f'{prefix}/mae': detailed_metrics['mae'],
            f'{prefix}/r2': detailed_metrics['r2'],
            f'{prefix}/scatter': detailed_metrics['scatter']
        }

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary with training results
        """
        epochs = self.training_config.get('epochs', 200)

        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(self.val_loader, prefix='val')

            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}

            # LR scheduler step
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metrics['val/mse'])
                else:
                    self.lr_scheduler.step()

            # Track best model
            current_val_loss = val_metrics['val/mse']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                if self.save_best:
                    self._save_checkpoint('best_model.pt', epoch, metrics)

            # Early stopping
            if self.early_stopping.step(current_val_loss, epoch):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, metrics)

            # Log to W&B
            if self.use_wandb:
                # Add weight norms
                metrics.update({f'weights/{k}': v for k, v in self.model.get_weight_norms().items()})
                wandb.log(metrics, step=epoch)

            # Log to console
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train MSE: {train_metrics['train/mse']:.4f} | "
                f"Val MSE: {val_metrics['val/mse']:.4f} | "
                f"Val R²: {val_metrics['val/r2']:.4f} | "
                f"λ: {train_metrics['train/lambda']:.3f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Store losses
            self.state.train_losses.append(train_metrics['train/mse'])
            self.state.val_losses.append(val_metrics['val/mse'])
            self.state.epoch = epoch

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation MSE: {best_val_loss:.4f} at epoch {best_epoch}")

        # Final test evaluation
        test_metrics = {}
        if self.test_loader is not None:
            # Load best model
            best_checkpoint = self.checkpoint_dir / 'best_model.pt'
            if best_checkpoint.exists():
                self._load_checkpoint(best_checkpoint)

            test_metrics = self.validate(self.test_loader, prefix='test')
            logger.info(f"Test MSE: {test_metrics['test/mse']:.4f} | "
                       f"Test R²: {test_metrics['test/r2']:.4f}")

            if self.use_wandb:
                wandb.log(test_metrics)

        # Save final checkpoint
        final_metrics = {**val_metrics, **test_metrics}
        self._save_checkpoint('final_model.pt', epoch, final_metrics)

        # Close W&B
        if self.use_wandb:
            wandb.finish()

        return {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_test_metrics': test_metrics,
            'total_time': total_time
        }

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lambda_scheduler_state': self.lambda_scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint: {path}")

    @torch.no_grad()
    def evaluate_with_uncertainty(
        self,
        loader: DataLoader,
        n_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate with MC-Dropout uncertainty estimation.

        Args:
            loader: DataLoader to evaluate
            n_samples: Number of MC samples

        Returns:
            Dictionary with predictions, uncertainties, and metrics
        """
        all_means = []
        all_stds = []
        all_targets = []
        all_cluster_ids = []

        for batch_data in loader:
            batch_data = batch_data.to(self.device)

            # Get predictions with uncertainty
            uncertainty = self.model.predict_with_uncertainty(batch_data, n_samples)

            all_means.append(uncertainty['mean'].cpu())
            all_stds.append(uncertainty['std'].cpu())
            all_targets.append(batch_data.y.cpu())

            # Get cluster IDs if available
            if hasattr(batch_data, 'cluster_id'):
                all_cluster_ids.extend(batch_data.cluster_id)

        all_means = torch.cat(all_means)
        all_stds = torch.cat(all_stds)
        all_targets = torch.cat(all_targets)

        # Compute metrics
        metrics = MetricsComputer.compute_all(all_means, all_targets)

        # Calibration check: what fraction of targets fall within 95% CI?
        conf_low = all_means - 1.96 * all_stds
        conf_high = all_means + 1.96 * all_stds
        coverage = ((all_targets >= conf_low) & (all_targets <= conf_high)).float().mean()

        return {
            'predictions': all_means.numpy(),
            'uncertainties': all_stds.numpy(),
            'targets': all_targets.numpy(),
            'cluster_ids': all_cluster_ids,
            'metrics': metrics,
            'coverage_95': coverage.item()
        }


def train_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None
) -> Tuple[CosmicNetGNN, Dict[str, Any]]:
    """
    Main entry point for training.

    Args:
        config: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)

    Returns:
        Tuple of (trained_model, results_dict)
    """
    # Build model
    model = build_model(config)

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # Train
    results = trainer.train()

    return model, results


def run_ablation_study(
    config: Dict[str, Any],
    ablation_configs: List[Dict[str, Any]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None
) -> List[Dict[str, Any]]:
    """
    Run ablation study with multiple configuration variations.

    Args:
        config: Base configuration
        ablation_configs: List of config overrides for each ablation
        train_loader, val_loader, test_loader: Data loaders

    Returns:
        List of results for each ablation
    """
    results = []

    for i, ablation in enumerate(ablation_configs):
        # Merge configs
        run_config = {**config}
        for key, value in ablation.items():
            keys = key.split('.')
            d = run_config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

        logger.info(f"Running ablation {i+1}/{len(ablation_configs)}: {ablation}")

        # Train
        model, run_results = train_model(
            run_config, train_loader, val_loader, test_loader
        )

        results.append({
            'ablation': ablation,
            'results': run_results
        })

    return results
