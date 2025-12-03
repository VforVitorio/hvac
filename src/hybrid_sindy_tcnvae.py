"""
Hybrid Model combining SINDy Physics + TCN-VAE
Integrates physics-based equations discovered by PySINDy with
Temporal Convolutional VAE for HVAC digital twin modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from tcn_vae import TCNVAE, TCNVAEPredictor, vae_loss_function


@dataclass
class HybridConfig:
    """Configuration for hybrid SINDy + TCN-VAE model."""
    input_dim: int
    output_dim: int
    sequence_length: int

    # TCN-VAE parameters
    latent_dim: int = 64
    encoder_channels: List[int] = None
    decoder_channels: List[int] = None
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2

    # Physics parameters
    physics_weight: float = 0.3
    vae_beta: float = 0.1  # β-VAE weight

    # Training parameters
    learning_rate: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        """Set default values for channel lists."""
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128, 256]
        if self.decoder_channels is None:
            self.decoder_channels = [256, 128, 64]


class PhysicsInformedLayer(nn.Module):
    """Layer that enforces physical constraints on predictions."""

    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, output_format: str = '2d') -> torch.Tensor:
        """
        Apply physics constraints to predictions.

        Args:
            predictions: Model predictions (batch, channels, seq) or (batch, features)
            output_format: '2d' for (batch, features) or '3d' for (batch, channels, seq)

        Returns:
            Physics-constrained predictions
        """
        constrained = predictions.clone()

        if output_format == '3d':  # (batch, channels, sequence)
            # Temperature constraints (channels 0-3: -10°C to 50°C)
            constrained[:, :4, :] = torch.clamp(constrained[:, :4, :], min=-10, max=50)
            # Humidity constraints (channel 4: 0% to 100%)
            if constrained.shape[1] > 4:
                constrained[:, 4, :] = torch.clamp(constrained[:, 4, :], min=0, max=100)
            # Power constraints (channel 5: >= 0)
            if constrained.shape[1] > 5:
                constrained[:, 5, :] = torch.clamp(constrained[:, 5, :], min=0)
        else:  # 2d format (batch, features)
            # Temperature constraints
            constrained[:, :4] = torch.clamp(constrained[:, :4], min=-10, max=50)
            # Humidity constraints
            if constrained.shape[1] > 4:
                constrained[:, 4] = torch.clamp(constrained[:, 4], min=0, max=100)
            # Power constraints
            if constrained.shape[1] > 5:
                constrained[:, 5] = torch.clamp(constrained[:, 5], min=0)

        return constrained


class PhysicsSINDyNetwork(nn.Module):
    """Physics-based network using SINDy discovered equations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sequence_length: int,
        sindy_coefficients: Optional[np.ndarray] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length

        # Initialize with SINDy coefficients if available
        if sindy_coefficients is not None:
            # Use discovered physics
            self.physics_weights = nn.Parameter(
                torch.tensor(sindy_coefficients, dtype=torch.float32),
                requires_grad=True  # Allow fine-tuning
            )
        else:
            # Learnable physics parameters (when SINDy not available)
            self.physics_weights = nn.Parameter(
                torch.randn(output_dim, input_dim) * 0.01
            )

        # Temporal integration layer
        self.temporal_conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)

        # Nonlinear physics layer
        self.nonlinear_physics = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-based predictions.

        Args:
            x: Input (batch, input_dim, sequence_length)

        Returns:
            Physics predictions (batch, output_dim, sequence_length)
        """
        batch_size = x.size(0)

        # Apply temporal convolution
        x_temporal = self.temporal_conv(x)  # (batch, input_dim, seq_len)

        # Linear physics component (SINDy equations)
        # Reshape for matrix multiplication
        x_transposed = x_temporal.permute(0, 2, 1)  # (batch, seq_len, input_dim)

        # Apply physics weights
        linear_physics = torch.matmul(
            x_transposed,
            self.physics_weights.t()
        )  # (batch, seq_len, output_dim)

        # Nonlinear physics component
        nonlinear_physics = self.nonlinear_physics(x_transposed)  # (batch, seq_len, output_dim)

        # Combine linear and nonlinear
        physics_output = linear_physics + 0.1 * nonlinear_physics

        # Back to (batch, output_dim, seq_len) format
        return physics_output.permute(0, 2, 1)


class HybridSINDyTCNVAE(nn.Module):
    """
    Hybrid model combining:
    1. SINDy physics-based component
    2. TCN-VAE data-driven component
    3. Adaptive fusion mechanism
    """

    def __init__(
        self,
        config: HybridConfig,
        sindy_coefficients: Optional[np.ndarray] = None
    ):
        super().__init__()

        self.config = config

        # Physics component (SINDy)
        self.physics_network = PhysicsSINDyNetwork(
            config.input_dim,
            config.output_dim,
            config.sequence_length,
            sindy_coefficients
        )

        # Data-driven component (TCN-VAE)
        self.tcn_vae = TCNVAE(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            encoder_channels=config.encoder_channels,
            decoder_channels=config.decoder_channels,
            kernel_size=config.tcn_kernel_size,
            dropout=config.tcn_dropout,
            output_dim=config.output_dim
        )

        # Predictor from TCN-VAE latent space
        self.vae_predictor = TCNVAEPredictor(
            self.tcn_vae,
            config.output_dim,
            hidden_dims=[128, 64]
        )

        # Physics constraint layer
        self.physics_constraint = PhysicsInformedLayer()

        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(config.physics_weight))

        # Attention-based fusion (optional enhancement)
        self.fusion_attention = nn.Sequential(
            nn.Conv1d(config.output_dim * 2, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = True,
        use_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.

        Args:
            x: Input tensor (batch, input_dim, sequence_length)
            return_components: Whether to return individual components
            use_attention: Use attention-based fusion instead of simple weighted sum

        Returns:
            Dictionary containing:
                - 'hybrid': Fused prediction
                - 'physics': Physics-based prediction
                - 'tcn_vae': Data-driven prediction
                - 'reconstruction': VAE reconstruction
                - 'mu': Latent mean
                - 'logvar': Latent log variance
        """
        # Physics-based prediction
        physics_pred = self.physics_network(x)  # (batch, output_dim, seq_len)

        # TCN-VAE prediction
        vae_pred = self.vae_predictor(x, deterministic=False)  # (batch, output_dim, seq_len)

        # VAE reconstruction for auxiliary loss
        vae_recon, mu, logvar = self.tcn_vae(x)

        # Fusion
        if use_attention:
            # Attention-based fusion
            combined = torch.cat([physics_pred, vae_pred], dim=1)  # (batch, 2*output_dim, seq_len)
            attention_weights = self.fusion_attention(combined)  # (batch, 2, seq_len)

            # Apply attention weights
            hybrid_pred = (
                attention_weights[:, 0:1, :] * physics_pred +
                attention_weights[:, 1:2, :] * vae_pred
            )
        else:
            # Simple weighted fusion
            alpha = torch.sigmoid(self.alpha)  # Ensure alpha in [0, 1]
            hybrid_pred = alpha * physics_pred + (1 - alpha) * vae_pred

        # Apply physics constraints
        hybrid_pred = self.physics_constraint(hybrid_pred, output_format='3d')

        results = {
            'hybrid': hybrid_pred,
            'physics': physics_pred,
            'tcn_vae': vae_pred,
            'reconstruction': vae_recon,
            'mu': mu,
            'logvar': logvar,
            'alpha': torch.sigmoid(self.alpha).item()
        }

        if not return_components:
            return {'hybrid': hybrid_pred}

        return results

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified prediction interface.

        Args:
            x: Input tensor

        Returns:
            Hybrid predictions
        """
        results = self.forward(x, return_components=False)
        return results['hybrid']


class HybridModelTrainer:
    """Trainer for hybrid SINDy + TCN-VAE model."""

    def __init__(
        self,
        model: HybridSINDyTCNVAE,
        config: HybridConfig
    ):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        self.mse_loss = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'physics_loss': [],
            'vae_loss': [],
            'kl_div': []
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss combining data, physics, and VAE components.

        Args:
            outputs: Model outputs dictionary
            targets: Ground truth targets
            inputs: Input data

        Returns:
            Dictionary of loss components
        """
        # Main prediction loss
        prediction_loss = self.mse_loss(outputs['hybrid'], targets)

        # Physics consistency loss
        physics_loss = self.mse_loss(outputs['physics'], targets)

        # TCN-VAE loss
        vae_total_loss, vae_recon_loss, kl_div = vae_loss_function(
            outputs['reconstruction'],
            inputs,
            outputs['mu'],
            outputs['logvar'],
            beta=self.config.vae_beta
        )

        # Combined loss
        total_loss = (
            prediction_loss +
            self.config.physics_weight * physics_loss +
            0.1 * vae_total_loss  # VAE auxiliary loss
        )

        return {
            'total': total_loss,
            'prediction': prediction_loss,
            'physics': physics_loss,
            'vae': vae_total_loss,
            'kl_div': kl_div
        }

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'physics': 0.0,
            'vae': 0.0,
            'kl_div': 0.0
        }

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_x)

            # Compute losses
            losses = self.compute_loss(outputs, batch_y, batch_x)

            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

        # Average losses
        n_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        return epoch_losses

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'prediction': 0.0
        }

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.mse_loss(outputs['hybrid'], batch_y)

                val_losses['total'] += loss.item()
                val_losses['prediction'] += loss.item()

        n_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= n_batches

        return val_losses

    def train(self, train_loader, val_loader, epochs: int):
        """Complete training loop."""
        print(f"\n{'='*60}")
        print(f"Training Hybrid SINDy + TCN-VAE on {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_metrics['total'])
            self.history['physics_loss'].append(train_metrics['physics'])
            self.history['vae_loss'].append(train_metrics['vae'])
            self.history['kl_div'].append(train_metrics['kl_div'])

            self.scheduler.step(val_metrics['total'])

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_metrics['total']:.6f}")
                print(f"  Val Loss: {val_metrics['total']:.6f}")
                print(f"  Physics Loss: {train_metrics['physics']:.6f}")
                print(f"  VAE Loss: {train_metrics['vae']:.6f}")
                print(f"  KL Div: {train_metrics['kl_div']:.6f}")
                print(f"  Alpha: {self.model.alpha.item():.4f}\n")

    def save_model(self, filepath: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        print(f"✓ Model saved to {filepath}")


if __name__ == "__main__":
    print("Hybrid SINDy + TCN-VAE Model for HVAC Digital Twin")