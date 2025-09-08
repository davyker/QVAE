import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add qvim-baseline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../qvim-baseline/src'))

from qvim_mn_baseline.mn.model import get_model
from qvim_mn_baseline.utils import NAME_TO_WIDTH


class QVAEEncoder(nn.Module):
    def __init__(self, pretrained_name="mn10_as", checkpoint_path=None, fine_tuning_config=None, latent_dim=512, feature_space="final"):
        super().__init__()
        
        # Store fine-tuning configuration and feature extraction settings
        self.fine_tuning_config = fine_tuning_config or {"enabled": False}
        self.feature_space = feature_space
        
        # Load pretrained MobileNetV3 - this includes imitation and reference encoder,
        # but we will only use the imitation encoder part
        self.backbone = get_model(
            width_mult=NAME_TO_WIDTH(pretrained_name),
            pretrained_name=pretrained_name
        )
        
        # Determine VAE head input dimension based on feature_space
        vae_head_input_dim = self._get_feature_dim()
        
        # Create VAE heads with dynamic input dimension
        self.mu_head = nn.Linear(vae_head_input_dim, latent_dim)
        self.logvar_head = nn.Linear(vae_head_input_dim, latent_dim)
        
        # Initialise VAE head weights
        nn.init.xavier_normal_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_normal_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        
        # Load checkpoint if provided, otherwise train from scratch
        self.from_scratch = checkpoint_path is None
        
        if checkpoint_path:
            # Initially freeze ALL backbone parameters for pretrained mode
            self.backbone.requires_grad_(False)
            self.load_checkpoint(checkpoint_path)
            print(f"✅ Loaded pretrained checkpoint: {checkpoint_path}")
        else:
            # From scratch mode - all parameters trainable
            self.backbone.requires_grad_(True)
            print("🎲 Training from scratch - random initialisation")
            
        # Apply fine-tuning configuration after loading checkpoint (only for pretrained mode)
        if not self.from_scratch:
            self._apply_fine_tuning_config()
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # Extract encoder weights (use imitation_encoder)
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith('imitation_encoder.'):
                new_key = key.replace('imitation_encoder.', 'backbone.')
                encoder_state[new_key] = value
        
        # Load pretrained weights (excluding classifier.2 and classifier.5)
        self.load_state_dict(encoder_state, strict=False)
    
    def forward(self, x):
        # Extract normalised 960-dim features
        features = self.get_features(x)
        
        # VAE heads: 960 -> latent_dim each
        mu = self.mu_head(features)  # [B, latent_dim]
        logvar = self.logvar_head(features)  # [B, latent_dim]
        
        return mu, logvar
    
    def get_features(self, x):
        """Extract features at specified layer."""
        if self.feature_space == "final":
            # Current logic: features + classifier[0:2]
            features = self.backbone.features(x)
            features = self.backbone.classifier[0](features)  # AdaptiveAvgPool2d -> [B, 960, 1, 1]
            features = self.backbone.classifier[1](features)  # Flatten -> [B, 960]
            return F.normalize(features, dim=1)
        else:
            # Extract from specific features.X layer
            layer_idx = int(self.feature_space)
            x = self.backbone.features[:layer_idx+1](x)  # Through features.X
            # Process spatial features [B, C, H, W] -> [B, C*H*W] and normalise
            features = x.flatten(1)  # [B, C, H, W] -> [B, C*H*W]
            return F.normalize(features, dim=1)
    
    def forward_precomputed(self, features):
        """Forward pass with pre-computed normalised features."""
        # features should be [B, vae_head_input_dim] and already normalised
        mu = self.mu_head(features)  # [B, latent_dim]
        logvar = self.logvar_head(features)  # [B, latent_dim]
        return mu, logvar
    
    def _get_feature_dim(self):
        """Calculate feature dimension based on feature_space setting."""
        if self.feature_space == "final":
            return 960
        else:
            # Use dummy input to determine spatial feature dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 128, 1000)
                dummy_features = self.get_features(dummy_input)
                return dummy_features.shape[1]
    
    def _apply_fine_tuning_config(self):
        """Apply fine-tuning configuration to unfreeze specified layers."""
        if not self.fine_tuning_config.get("enabled", False):
            return
            
        unfreeze_layers = self.fine_tuning_config.get("unfreeze_layers", [])
        
        print(f"🔓 Enabling encoder fine-tuning for layers: {unfreeze_layers}")
        
        for layer_idx in unfreeze_layers:
            if hasattr(self.backbone.features, str(layer_idx)):
                layer = getattr(self.backbone.features, str(layer_idx))
                layer.requires_grad_(True)
                print(f"  ✅ Unfroze features.{layer_idx}")
            else:
                print(f"  ⚠️  Warning: features.{layer_idx} not found")
                
    def get_trainable_encoder_params(self):
        """Get trainable encoder parameters for optimiser configuration."""
        if not self.fine_tuning_config.get("enabled", False):
            return []
            
        trainable_params = []
        unfreeze_layers = self.fine_tuning_config.get("unfreeze_layers", [])
        
        for layer_idx in unfreeze_layers:
            if hasattr(self.backbone.features, str(layer_idx)):
                layer = getattr(self.backbone.features, str(layer_idx))
                trainable_params.extend(layer.parameters())
                
        return trainable_params
    
    def reparameterise(self, mu, logvar, deterministic=False):
        if deterministic:
            if not hasattr(self, '_deterministic_warned'):
                print("!!! WARNING: YOU ARE USING DETERMINISTIC VAE SAMPLING !!!")
                print("!!! REMEMBER TO DISABLE THIS FOR ACTUAL TRAINING !!!")
                self._deterministic_warned = True
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std