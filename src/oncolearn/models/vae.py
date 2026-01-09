"""
Variational AutoEncoder (VAE) for learning feature distributions and dimension reduction.
Now uses pretrained VAE from Hugging Face (Stable Diffusion) for better performance.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import AutoencoderKL
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: diffusers not installed. Install with: pip install diffusers")
    print("Falling back to custom VAE implementation.")


class PretrainedVAE(nn.Module):
    """
    Wrapper for Hugging Face's pretrained VAE (from Stable Diffusion).
    Adapts feature vectors to work with image-based VAE.
    
    This uses the VAE from stabilityai/sd-vae-ft-mse which is pretrained on
    millions of images and provides excellent feature encoding.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        model_name: str = "stabilityai/sd-vae-ft-mse",
        freeze_vae: bool = True,
        spatial_size: int = 16,  # Spatial dimensions for VAE input
    ):
        """
        Initialize pretrained VAE from Hugging Face.

        Args:
            input_dim: Input feature dimension (from attention layers)
            latent_dim: Target latent dimension for final features
            model_name: Hugging Face model name for VAE
                       Options:
                       - "stabilityai/sd-vae-ft-mse" (best for general images)
                       - "stabilityai/sd-vae-ft-ema" (alternative)
                       - "CompVis/stable-diffusion-v1-4" (original SD VAE)
            freeze_vae: Whether to freeze pretrained weights
            spatial_size: Spatial size to reshape features into (e.g., 16x16)
        """
        super().__init__()

        if not HF_AVAILABLE:
            raise ImportError(
                "diffusers library required for PretrainedVAE. "
                "Install with: pip install diffusers transformers"
            )

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size

        # Load pretrained VAE
        print(f"Loading pretrained VAE from {model_name}...")
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        # Freeze VAE weights if requested
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            print("Pretrained VAE weights frozen")

        # Get VAE's expected input channels (usually 3 for RGB or 4 for latent)
        vae_channels = self.vae.config.in_channels

        # Project input features to VAE's expected format
        # Features -> Spatial tensor (channels x spatial_size x spatial_size)
        target_elements = vae_channels * spatial_size * spatial_size
        self.feature_to_spatial = nn.Sequential(
            nn.Linear(input_dim, target_elements),
            nn.ReLU()
        )

        # Get VAE's latent channels
        vae_latent_channels = self.vae.config.latent_channels
        vae_latent_size = spatial_size // 8  # VAE downsamples by 8x
        vae_latent_elements = vae_latent_channels * vae_latent_size * vae_latent_size

        # Project VAE latent to target latent_dim
        self.latent_projection = nn.Sequential(
            nn.Linear(vae_latent_elements, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # For reconstruction, project back from latent to VAE latent space
        self.latent_to_vae = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, vae_latent_elements)
        )

        # Project reconstructed spatial back to input_dim
        self.spatial_to_feature = nn.Linear(target_elements, input_dim)

        self.vae_channels = vae_channels
        self.vae_latent_channels = vae_latent_channels
        self.vae_latent_size = vae_latent_size

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input features using pretrained VAE.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]
        """
        batch_size = x.size(0)

        # Project to spatial format
        spatial = self.feature_to_spatial(x)
        spatial = spatial.view(batch_size, self.vae_channels, 
                              self.spatial_size, self.spatial_size)

        # Encode with pretrained VAE
        with torch.set_grad_enabled(self.training and not all(not p.requires_grad for p in self.vae.parameters())):
            latent_dist = self.vae.encode(spatial).latent_dist
            vae_latent = latent_dist.sample()  # [B, latent_channels, H/8, W/8]

        # Flatten and project to target latent_dim
        vae_latent_flat = vae_latent.view(batch_size, -1)
        projected = self.latent_projection(vae_latent_flat)

        # Split into mu and logvar
        mu = projected
        # Use VAE's inherent uncertainty as logvar (simplified)
        logvar = torch.zeros_like(mu) - 1.0  # Small fixed variance

        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector using pretrained VAE.

        Args:
            z: Latent vector [batch_size, latent_dim]

        Returns:
            Reconstruction [batch_size, input_dim]
        """
        batch_size = z.size(0)

        # Project to VAE latent space
        vae_latent_flat = self.latent_to_vae(z)
        vae_latent = vae_latent_flat.view(
            batch_size, self.vae_latent_channels,
            self.vae_latent_size, self.vae_latent_size
        )

        # Decode with pretrained VAE
        with torch.set_grad_enabled(self.training and not all(not p.requires_grad for p in self.vae.parameters())):
            spatial = self.vae.decode(vae_latent).sample

        # Flatten and project back to input_dim
        spatial_flat = spatial.view(batch_size, -1)
        reconstruction = self.spatial_to_feature(spatial_flat)

        return reconstruction

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> dict:
        """
        Forward pass through pretrained VAE.

        Args:
            x: Input features [batch_size, input_dim]
            return_latent: Whether to return sampled latent

        Returns:
            Dictionary with reconstruction, mu, logvar, and optionally z
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        result = {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar
        }

        if return_latent:
            result['z'] = z

        return result

    def get_latent_dim(self) -> int:
        """Return latent dimension."""
        return self.latent_dim


class VariationalAutoEncoder(nn.Module):
    """
    Variational AutoEncoder for learning latent distributions of features.
    Provides dimension reduction while learning the distribution of features.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize VAE.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension (reduced dimension)
            hidden_dims: List of hidden layer dimensions for encoder/decoder
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.hidden_dims = hidden_dims

        # Build Encoder
        encoder_layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        decoder_layers = []
        in_dim = latent_dim

        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean [batch_size, latent_dim]
            logvar: Log variance [batch_size, latent_dim]

        Returns:
            Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.

        Args:
            z: Latent vector [batch_size, latent_dim]

        Returns:
            Reconstructed input [batch_size, input_dim]
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> dict:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor [batch_size, input_dim]
            return_latent: Whether to return latent parameters

        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed input
                - mu: Mean of latent distribution
                - logvar: Log variance of latent distribution
                - z: Sampled latent vector (if return_latent=True)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        result = {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar
        }

        if return_latent:
            result['z'] = z

        return result

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from latent space and decode.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def get_latent_dim(self) -> int:
        """Return latent dimension."""
        return self.latent_dim


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kld_weight: float = 1.0,
    reconstruction_loss: str = 'mse'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss (reconstruction + KLD).

    Args:
        reconstruction: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kld_weight: Weight for KLD term (beta in beta-VAE)
        reconstruction_loss: Type of reconstruction loss ('mse' or 'bce')

    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss
        kld_loss: KL divergence loss
    """
    # Reconstruction loss
    if reconstruction_loss == 'mse':
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
    elif reconstruction_loss == 'bce':
        recon_loss = F.binary_cross_entropy(
            torch.sigmoid(reconstruction), x, reduction='mean'
        )
    else:
        raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = kld_loss / x.size(0)  # Average over batch

    # Total loss
    total_loss = recon_loss + kld_weight * kld_loss

    return total_loss, recon_loss, kld_loss


class ConditionalVAE(nn.Module):
    """
    Conditional VAE that can condition on additional information (e.g., class labels).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize Conditional VAE.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            condition_dim: Conditioning variable dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Encoder (input + condition)
        encoder_layers = []
        in_dim = input_dim + condition_dim

        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder (latent + condition)
        decoder_layers = []
        in_dim = latent_dim + condition_dim

        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with conditioning."""
        x_cond = torch.cat([x, condition], dim=-1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode with conditioning."""
        z_cond = torch.cat([z, condition], dim=-1)
        return self.decoder(z_cond)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> dict:
        """Forward pass through conditional VAE."""
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, condition)

        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
