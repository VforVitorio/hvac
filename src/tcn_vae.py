"""
TCN-VAE (Temporal Convolutional Network - Variational Autoencoder)
Architecture for capturing temporal dynamics in HVAC systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class CausalConv1d(nn.Module):
    """Causal 1D convolution for temporal modeling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Remove future information (causal padding)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TemporalBlock(nn.Module):
    """Residual temporal block with dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        # Residual connection
        return F.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder.
    Uses dilated causal convolutions to capture long-term dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        num_levels = len(hidden_channels)

        for i in range(num_levels):
            in_ch = input_dim if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            dilation = 2 ** i  # Exponentially increasing dilation

            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.output_channels = hidden_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, sequence_length)
        Returns:
            Encoded representation (batch, hidden_channels, sequence_length)
        """
        return self.network(x)


class TCNDecoder(nn.Module):
    """
    Temporal Convolutional Network Decoder.
    Reconstructs temporal sequences from latent representation.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: List[int],
        output_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        num_levels = len(hidden_channels)

        for i in range(num_levels):
            in_ch = latent_dim if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            dilation = 2 ** (num_levels - i - 1)  # Reverse dilation

            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.network = nn.Sequential(*layers)

        # Final output layer
        self.output_layer = nn.Conv1d(hidden_channels[-1], output_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent representation (batch, latent_dim, sequence_length)
        Returns:
            Reconstructed sequence (batch, output_dim, sequence_length)
        """
        out = self.network(z)
        return self.output_layer(out)


class TCNVAE(nn.Module):
    """
    Temporal Convolutional Network Variational Autoencoder.
    Learns latent representation of temporal HVAC dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_channels: List[int] = [64, 128, 256],
        decoder_channels: List[int] = [256, 128, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim or input_dim

        # Encoder
        self.encoder = TCNEncoder(
            input_dim,
            encoder_channels,
            kernel_size,
            dropout
        )

        # Latent space (mu and log_var)
        self.fc_mu = nn.Conv1d(encoder_channels[-1], latent_dim, 1)
        self.fc_logvar = nn.Conv1d(encoder_channels[-1], latent_dim, 1)

        # Decoder
        self.decoder = TCNDecoder(
            latent_dim,
            decoder_channels,
            self.output_dim,
            kernel_size,
            dropout
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: (batch, input_dim, sequence_length)

        Returns:
            mu, logvar: Latent distribution parameters
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.

        Args:
            z: Latent representation (batch, latent_dim, sequence_length)

        Returns:
            Reconstructed output (batch, output_dim, sequence_length)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through TCN-VAE.

        Args:
            x: Input tensor (batch, input_dim, sequence_length)

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

    def get_latent(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get latent representation.

        Args:
            x: Input tensor
            deterministic: If True, return mu; else sample from distribution

        Returns:
            Latent representation
        """
        mu, logvar = self.encode(x)
        if deterministic:
            return mu
        return self.reparameterize(mu, logvar)


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss function combining reconstruction and KL divergence.

    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence (β-VAE)

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    # KL(N(mu, sigma) || N(0, 1))
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = kl_div / x.size(0)  # Normalize by batch size

    # Total loss
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


class TCNVAEPredictor(nn.Module):
    """
    Predictor network that uses TCN-VAE latent representation.
    Maps latent space to output predictions.
    """

    def __init__(
        self,
        tcn_vae: TCNVAE,
        prediction_dim: int,
        hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()

        self.tcn_vae = tcn_vae
        self.prediction_dim = prediction_dim

        # Prediction network from latent space
        layers = []
        prev_dim = tcn_vae.latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Conv1d(prev_dim, prediction_dim, 1))

        self.predictor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass for prediction.

        Args:
            x: Input tensor (batch, input_dim, sequence_length)
            deterministic: Use deterministic encoding

        Returns:
            Predictions (batch, prediction_dim, sequence_length)
        """
        # Get latent representation
        z = self.tcn_vae.get_latent(x, deterministic=deterministic)

        # Predict from latent
        predictions = self.predictor(z)

        return predictions


if __name__ == "__main__":
    # Test TCN-VAE
    batch_size = 32
    input_dim = 10
    sequence_length = 50
    latent_dim = 32

    # Create model
    model = TCNVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_channels=[64, 128, 256],
        decoder_channels=[256, 128, 64]
    )

    # Test input
    x = torch.randn(batch_size, input_dim, sequence_length)

    # Forward pass
    recon, mu, logvar = model(x)

    print("TCN-VAE Architecture Test:")
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")

    # Test loss
    loss, recon_loss, kl_div = vae_loss_function(recon, x, mu, logvar)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kl_div.item():.4f}")