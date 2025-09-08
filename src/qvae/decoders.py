"""
Decoder architecture registry and factory.
Supports multiple decoder architectures with config-based selection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math


class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow."""
    
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        
        return F.leaky_relu(x + residual, 0.1)


class UpsampleBlock(nn.Module):
    """Learnable upsampling block with feature mixing."""
    
    def __init__(self, in_channels, out_channels, scale_factor=3, dropout=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.upsample_conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=3,
            stride=1, 
            padding=1
        )
        
        # Feature refinement
        self.refine_conv = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Residual blocks for this scale
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(out_channels, dilation=1, dropout=dropout),
            ResidualBlock(out_channels, dilation=2, dropout=dropout),
        ])
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='linear')
        x = F.leaky_relu(self.bn(self.upsample_conv(x)), 0.1)
        x = F.leaky_relu(self.bn(self.refine_conv(x)), 0.1)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
            
        return x


class OriginalQVAEDecoder(nn.Module):
    """
    As original transposed SampleCNN decoder, except with stride=1 in most layers
    Still suffered resonance issues due to final stride=3 layer.
    """
    
    def __init__(self, latent_dim=512, weight_init_std=0.01, **kwargs):
        super().__init__()
        self.weight_init_std = weight_init_std
        
        # Layer 11^T: latent_dim -> 512
        self.conv11 = nn.ConvTranspose1d(latent_dim, 512, kernel_size=3, stride=1, padding=1)
        
        # Layer 10^T: 512 -> 256  
        self.conv10 = nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1)
        
        # Layers 9^T - 5^T: 256 -> 256
        self.conv9 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Layer 4^T: 256 -> 128
        self.conv4 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1)
        
        # Layers 3^T, 2^T: 128 -> 128
        self.conv3 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Layer 1^T: 128 -> 1 (final output)
        self.conv1 = nn.ConvTranspose1d(128, 1, kernel_size=3, stride=3, padding=0)
        
        # Batch normalisation and activation
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(128), nn.BatchNorm1d(128),
            nn.BatchNorm1d(128)
        ])

        self._initialise_weights()

    def forward(self, z):
        # Input: [B, 512] -> Reshape to [B, 512, 1]
        x = z.unsqueeze(-1)
        
        # Layer 11^T
        x = F.relu(self.bn_layers[0](self.conv11(x)))
        
        # Layer 10^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')  # Upsample 3x
        x = F.relu(self.bn_layers[1](self.conv10(x)))
        
        # Layer 9^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[2](self.conv9(x)))
        
        # Layer 8^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[3](self.conv8(x)))
        
        # Layer 7^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[4](self.conv7(x)))
        
        # Layer 6^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[5](self.conv6(x)))
        
        # Layer 5^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[6](self.conv5(x)))
        
        # Layer 4^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[7](self.conv4(x)))
        
        # Layer 3^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[8](self.conv3(x)))
        
        # Layer 2^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[9](self.conv2(x)))
        
        # Layer 1^T (final layer, no BN/activation)
        x = self.conv1(x)
        
        # Apply tanh to clamp output to [-1, 1] range
        x = torch.tanh(x)
        
        return x.squeeze(1)  # [B, 59049] - remove channel dimension

    def _initialise_weights(self):
        """Initialise convolutional layer weights using Kaiming normal initialisation."""
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class OriginalPlusQVAEDecoder(nn.Module):
    """
    Modified decoder with Conv1d + interpolation - keeping for backwards compatibility.
    """
    
    def __init__(self, latent_dim=512, weight_init_std=0.01, weight_init_method="fixed", kaiming_scale_factor=1.0, 
                 use_he_uniform_init=False, he_uniform_init_mode="fan_in", he_uniform_init_nonlinearity="leaky_relu", **kwargs):
        super().__init__()
        self.weight_init_std = weight_init_std
        self.weight_init_method = weight_init_method
        self.kaiming_scale_factor = kaiming_scale_factor
        self.use_he_uniform_init = use_he_uniform_init
        self.he_uniform_init_mode = he_uniform_init_mode
        self.he_uniform_init_nonlinearity = he_uniform_init_nonlinearity
        
        # Layer 11: 
        self.conv11 = nn.Conv1d(latent_dim, 512, kernel_size=3, stride=1, padding=1)
        
        # Layer 10:  
        self.conv10 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        
        # Layers 9 - 5:
        self.conv9 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Layer 4:
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        
        # Layers 3, 2:
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Multi-stage final layers
        self.final_conv1 = nn.Conv1d(128, 32, kernel_size=5, stride=1, padding=2) 
        self.final_conv2 = nn.Conv1d(32, 8, kernel_size=5, stride=1, padding=2)  
        self.final_conv3 = nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1)  
        
        # Batch normalisation and activation
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(128), nn.BatchNorm1d(128),
            nn.BatchNorm1d(128)
        ])
        
        # Batch normalisation for final layers
        self.final_bn1 = nn.BatchNorm1d(32)
        self.final_bn2 = nn.BatchNorm1d(8)
        
        self._initialise_weights()
    
    def forward(self, z):
        # Input: [B, 512] -> Reshape to [B, 512, 1]
        x = z.unsqueeze(-1)
        
        # Layer 11^T
        x = F.relu(self.bn_layers[0](self.conv11(x)))
        
        # Layer 10^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[1](self.conv10(x)))
        
        # Layer 9^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[2](self.conv9(x)))
        
        # Layer 8^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[3](self.conv8(x)))
        
        # Layer 7^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[4](self.conv7(x)))
        
        # Layer 6^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[5](self.conv6(x)))
        
        # Layer 5^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[6](self.conv5(x)))
        
        # Layer 4^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[7](self.conv4(x)))
        
        # Layer 3^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[8](self.conv3(x)))
        
        # Layer 2^T
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[9](self.conv2(x)))
        
        # Final upsampling and multi-stage output layers
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        
        # Multi-stage final convolution (original successful 4x tapering)
        x = F.leaky_relu(self.final_bn1(self.final_conv1(x)), 0.1) 
        x = F.leaky_relu(self.final_bn2(self.final_conv2(x)), 0.1) 
        x = self.final_conv3(x)  
        
        x = torch.tanh(x)
        
        return x.squeeze(1)  # [B, 59049] - remove channel dimension
    
    def _initialise_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                if self.use_he_uniform_init:
                    nn.init.kaiming_uniform_(
                        module.weight,
                        mode=self.he_uniform_init_mode,
                        nonlinearity=self.he_uniform_init_nonlinearity
                    )
                elif self.weight_init_method == "kaiming_scaled":
                    # Kaiming normal scaled by factor
                    fan_in = module.in_channels * module.kernel_size[0]
                    std = math.sqrt(2.0 / fan_in) / self.kaiming_scale_factor
                    nn.init.normal_(module.weight, mean=0, std=std)
                else:
                    # Use configurable fixed initialisation std
                    nn.init.normal_(module.weight, mean=0, std=self.weight_init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class OriginalResQVAEDecoder(nn.Module):
    """
    Original transposed SampleCNN decoder with residual connections.
    Uses skip connections between layers with matching channel dimensions.
    """
    
    def __init__(self, latent_dim=512, weight_init_std=0.01, weight_init_method="fixed", kaiming_scale_factor=1.0,
                 use_he_uniform_init=False, he_uniform_init_mode="fan_in", he_uniform_init_nonlinearity="leaky_relu", **kwargs):
        super().__init__()
        self.weight_init_std = weight_init_std
        self.weight_init_method = weight_init_method
        self.kaiming_scale_factor = kaiming_scale_factor
        self.use_he_uniform_init = use_he_uniform_init
        self.he_uniform_init_mode = he_uniform_init_mode
        self.he_uniform_init_nonlinearity = he_uniform_init_nonlinearity
        
        # Layer 11: latent_dim -> 512
        self.conv11 = nn.Conv1d(latent_dim, 512, kernel_size=3, stride=1, padding=1)
        
        # Layer 10: 512 -> 256  
        self.conv10 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        
        # Layers 9 - 5: 256 -> 256 
        self.conv9 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Layer 4: 256 -> 128
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        
        # Layers 3, 2: 128 -> 128 
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Multi-stage final layers for smoother audio generation
        self.final_conv1 = nn.Conv1d(128, 32, kernel_size=5, stride=1, padding=2)
        self.final_conv2 = nn.Conv1d(32, 8, kernel_size=5, stride=1, padding=2)
        self.final_conv3 = nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1)
        
        # Batch normalisation and activation
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(128), nn.BatchNorm1d(128),
            nn.BatchNorm1d(128)
        ])
        
        # Batch normalisation for final layers
        self.final_bn1 = nn.BatchNorm1d(32)
        self.final_bn2 = nn.BatchNorm1d(8)
        
        self._initialise_weights()
    
    def forward(self, z):
        # Input: [B, 512] -> Reshape to [B, 512, 1]
        x = z.unsqueeze(-1)
        
        # Layer 11^T (512 -> 512)
        x = F.relu(self.bn_layers[0](self.conv11(x)))
        
        # Layer 10^T (512 -> 256)
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[1](self.conv10(x)))
        
        # === Residual block for 256-channel layers ===
        
        # Layer 9^T (256 -> 256) - first in residual chain
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x9 = F.relu(self.bn_layers[2](self.conv9(x)))
        
        # Layer 8^T (256 -> 256) - with residual connection from layer 9
        x = F.interpolate(x9, scale_factor=3, mode='nearest')
        x8_out = self.conv8(x)
        x8_out = self.bn_layers[3](x8_out)
        # Residual connection from layer 9
        x9_resized = F.interpolate(x9, size=x8_out.shape[-1], mode='nearest')
        x8 = F.relu(x8_out + x9_resized)
        
        # Layer 7^T (256 -> 256) - with residual connection from layer 8
        x = F.interpolate(x8, scale_factor=3, mode='nearest')
        x7_out = self.conv7(x)
        x7_out = self.bn_layers[4](x7_out)
        # Residual connection from layer 8
        x8_resized = F.interpolate(x8, size=x7_out.shape[-1], mode='nearest')
        x7 = F.relu(x7_out + x8_resized)
        
        # Layer 6^T (256 -> 256) - with residual connection from layer 7
        x = F.interpolate(x7, scale_factor=3, mode='nearest')
        x6_out = self.conv6(x)
        x6_out = self.bn_layers[5](x6_out)
        # Residual connection from layer 7
        x7_resized = F.interpolate(x7, size=x6_out.shape[-1], mode='nearest')
        x6 = F.relu(x6_out + x7_resized)
        
        # Layer 5^T (256 -> 256) - with residual connection from layer 6
        x = F.interpolate(x6, scale_factor=3, mode='nearest')
        x5_out = self.conv5(x)
        x5_out = self.bn_layers[6](x5_out)
        # Residual connection from layer 6
        x6_resized = F.interpolate(x6, size=x5_out.shape[-1], mode='nearest')
        x5 = F.relu(x5_out + x6_resized)
        
        # Layer 4^T (256 -> 128) - channel dimension change, no residual
        x = F.interpolate(x5, scale_factor=3, mode='nearest')
        x = F.relu(self.bn_layers[7](self.conv4(x)))
        
        # === Residual block for 128-channel layers ===
        
        # Layer 3^T (128 -> 128) - first in 128-channel residual chain
        x = F.interpolate(x, scale_factor=3, mode='nearest')
        x3 = F.relu(self.bn_layers[8](self.conv3(x)))
        
        # Layer 2^T (128 -> 128) - with residual connection from layer 3
        x = F.interpolate(x3, scale_factor=3, mode='nearest')
        x2_out = self.conv2(x)
        x2_out = self.bn_layers[9](x2_out)
        # Residual connection from layer 3
        x3_resized = F.interpolate(x3, size=x2_out.shape[-1], mode='nearest')
        x2 = F.relu(x2_out + x3_resized)
        
        # Final upsampling and multi-stage output layers
        x = F.interpolate(x2, scale_factor=3, mode='nearest')
        
        # Multi-stage final convolution
        x = F.leaky_relu(self.final_bn1(self.final_conv1(x)), 0.1)
        x = F.leaky_relu(self.final_bn2(self.final_conv2(x)), 0.1)
        x = self.final_conv3(x)
        
        # Apply tanh to clamp output to [-1, 1] range
        x = torch.tanh(x)
        
        return x.squeeze(1)  # [B, 59049] - remove channel dimension
    
    def _initialise_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                if self.use_he_uniform_init:
                    # Use Kaiming uniform initialisation
                    nn.init.kaiming_uniform_(
                        module.weight,
                        mode=self.he_uniform_init_mode,
                        nonlinearity=self.he_uniform_init_nonlinearity
                    )
                elif self.weight_init_method == "kaiming_scaled":
                    # Kaiming normal scaled by factor
                    fan_in = module.in_channels * module.kernel_size[0]
                    std = math.sqrt(2.0 / fan_in) / self.kaiming_scale_factor
                    nn.init.normal_(module.weight, mean=0, std=std)
                else:
                    # Use configurable fixed initialisation std
                    nn.init.normal_(module.weight, mean=0, std=self.weight_init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class OriginalResPlusQVAEDecoder(nn.Module):
    """
    Original transposed SampleCNN decoder with residual connections and learnable upsampling.
    Uses transposed convolutions instead of fixed interpolation for upsampling.
    """
    
    def __init__(self, latent_dim=512, weight_init_std=0.01, weight_init_method="fixed", kaiming_scale_factor=1.0,
                 use_he_uniform_init=False, he_uniform_init_mode="fan_in", he_uniform_init_nonlinearity="leaky_relu", **kwargs):
        super().__init__()
        self.weight_init_std = weight_init_std
        self.weight_init_method = weight_init_method
        self.kaiming_scale_factor = kaiming_scale_factor
        self.use_he_uniform_init = use_he_uniform_init
        self.he_uniform_init_mode = he_uniform_init_mode
        self.he_uniform_init_nonlinearity = he_uniform_init_nonlinearity
        
        # Layer 11: latent_dim -> 512
        self.conv11 = nn.Conv1d(latent_dim, 512, kernel_size=3, stride=1, padding=1)
        
        # Layer 10: 512 -> 256  
        self.conv10 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        
        # Layers 9 - 5: 256 -> 256
        self.conv9 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Layer 4: 256 -> 128
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        
        # Layers 3, 2: 128 -> 128
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Multi-stage final layers for smoother audio generation
        self.final_conv1 = nn.Conv1d(128, 32, kernel_size=5, stride=1, padding=2)
        self.final_conv2 = nn.Conv1d(32, 8, kernel_size=5, stride=1, padding=2)
        self.final_conv3 = nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1)
        
        # Learnable upsampling layers
        self.upsample10 = nn.ConvTranspose1d(512, 512, kernel_size=3, stride=3, padding=0)
        self.upsample9 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.upsample8 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.upsample7 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.upsample6 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.upsample5 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.upsample4 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.upsample3 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=3, padding=0)
        self.upsample2 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=3, padding=0)
        self.upsample_final = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=3, padding=0)
        
        # Batch normalisation and activation
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(128), nn.BatchNorm1d(128),
            nn.BatchNorm1d(128)
        ])
        
        # Batch normalisation for final layers
        self.final_bn1 = nn.BatchNorm1d(32)
        self.final_bn2 = nn.BatchNorm1d(8)
        
        self._initialise_weights()
    
    def forward(self, z):
        # Input: [B, latent_dim] -> Reshape to [B, latent_dim, 1]
        x = z.unsqueeze(-1)
        
        # Layer 11^T (latent_dim -> 512)
        x = F.relu(self.bn_layers[0](self.conv11(x)))
        
        # Layer 10^T (512 -> 256) with learnable upsampling
        x = self.upsample10(x)
        x = F.relu(self.bn_layers[1](self.conv10(x)))
        
        # === Residual block for 256-channel layers ===
        
        # Layer 9^T (256 -> 256) - first in residual chain
        x = self.upsample9(x)
        x9 = F.relu(self.bn_layers[2](self.conv9(x)))
        
        # Layer 8^T (256 -> 256)
        x = self.upsample8(x9)
        x8_out = self.conv8(x)
        x8_out = self.bn_layers[3](x8_out)
        # Residual connection
        x9_resized = F.interpolate(x9, size=x8_out.shape[-1], mode='nearest')
        x8 = F.relu(x8_out + x9_resized)
        
        # Layer 7^T (256 -> 256)
        x = self.upsample7(x8)
        x7_out = self.conv7(x)
        x7_out = self.bn_layers[4](x7_out)
        # Residual connection
        x8_resized = F.interpolate(x8, size=x7_out.shape[-1], mode='nearest')
        x7 = F.relu(x7_out + x8_resized)
        
        # Layer 6^T (256 -> 256)
        x = self.upsample6(x7)
        x6_out = self.conv6(x)
        x6_out = self.bn_layers[5](x6_out)
        # Residual connection
        x7_resized = F.interpolate(x7, size=x6_out.shape[-1], mode='nearest')
        x6 = F.relu(x6_out + x7_resized)
        
        # Layer 5^T (256 -> 256)
        x = self.upsample5(x6)
        x5_out = self.conv5(x)
        x5_out = self.bn_layers[6](x5_out)
        # Residual connection
        x6_resized = F.interpolate(x6, size=x5_out.shape[-1], mode='nearest')
        x5 = F.relu(x5_out + x6_resized)
        
        # Layer 4^T (256 -> 128) - channel dimension change, no residual
        x = self.upsample4(x5)
        x = F.relu(self.bn_layers[7](self.conv4(x)))
        
        # Layer 3^T (128 -> 128) - first in 128-channel residual chain
        x = self.upsample3(x)
        x3 = F.relu(self.bn_layers[8](self.conv3(x)))
        
        # Layer 2^T (128 -> 128) - with residual connection from layer 3
        x = self.upsample2(x3)
        x2_out = self.conv2(x)
        x2_out = self.bn_layers[9](x2_out)
        # Residual connection
        x3_resized = F.interpolate(x3, size=x2_out.shape[-1], mode='nearest')
        x2 = F.relu(x2_out + x3_resized)
        
        # Final upsampling and multi-stage output layers
        x = self.upsample_final(x2)
        
        # Multi-stage final convolution
        x = F.leaky_relu(self.final_bn1(self.final_conv1(x)), 0.1)  # 128 -> 32
        x = F.leaky_relu(self.final_bn2(self.final_conv2(x)), 0.1)  # 32 -> 8
        x = self.final_conv3(x)  # 8 -> 1
        
        # Apply tanh to clamp output to [-1, 1] range
        x = torch.tanh(x)
        
        return x.squeeze(1)  # [B, 59049] - remove channel dimension
    
    def _initialise_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                if self.use_he_uniform_init:
                    # Use Kaiming uniform initialization
                    nn.init.kaiming_uniform_(
                        module.weight,
                        mode=self.he_uniform_init_mode,
                        nonlinearity=self.he_uniform_init_nonlinearity
                    )
                elif self.weight_init_method == "kaiming_scaled":
                    # Kaiming normal scaled by factor
                    fan_in = module.in_channels * module.kernel_size[0]
                    std = math.sqrt(2.0 / fan_in) / self.kaiming_scale_factor
                    nn.init.normal_(module.weight, mean=0, std=std)
                else:
                    # Use configurable fixed initialisation std
                    nn.init.normal_(module.weight, mean=0, std=self.weight_init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class ImprovedQVAEDecoder(nn.Module):
    """
    Improved decoder with skip connections and residual processing.
    """
    
    def __init__(self, latent_dim=512, dropout=0.1, channels=None, weight_init_std=0.01, weight_init_method="fixed", kaiming_scale_factor=1.0, **kwargs):
        super().__init__()
        self.weight_init_std = weight_init_std
        self.weight_init_method = weight_init_method
        self.kaiming_scale_factor = kaiming_scale_factor
        
        # Allow channel customisation
        if channels is None:
            channels = [512, 256, 256, 256, 256, 256, 128, 128, 64, 32]
        
        # Initial projection and expansion
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, channels[0]),
        )
        
        # Progressive upsampling with skip connections
        self.stage1 = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], 3, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.LeakyReLU(0.1),
            ResidualBlock(channels[0], dilation=1, dropout=dropout),
            ResidualBlock(channels[0], dilation=2, dropout=dropout),
        )
        
        # Upsampling stages
        self.upsample_layers = nn.ModuleList([
            UpsampleBlock(channels[i], channels[i+1], scale_factor=3, dropout=dropout)
            for i in range(len(channels)-1)
        ])
        
        # Final convolution after upsampling - use Conv1d to avoid resonance artifacts
        self.final_conv = nn.Conv1d(channels[-1], 16, kernel_size=9, stride=1, padding=4)
        
        # Multi-receptive field output
        self.output_layers = nn.ModuleList([
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.Conv1d(16, 8, kernel_size=7, padding=3),
            nn.Conv1d(16, 8, kernel_size=15, padding=7),
        ])
        
        self.output_combine = nn.Conv1d(24, 1, kernel_size=1)  # 8*3 = 24 channels
        
        self._initialise_weights()
    
    def forward(self, z):
        x = self.input_proj(z).unsqueeze(-1)
        x = self.stage1(x)
        
        # Progressive upsampling
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        
        # Final upsampling with interpolation (same as other fixed decoders)
        x = F.interpolate(x, scale_factor=3, mode='linear')
        x = F.leaky_relu(self.final_conv(x), 0.1)
        
        # Multi-receptive field processing
        outputs = [layer(x) for layer in self.output_layers]
        x = torch.cat(outputs, dim=1)
        x = self.output_combine(x)
        
        return torch.tanh(x).squeeze(1)
    
    def _initialise_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                if self.weight_init_method == "kaiming_scaled":
                    # Kaiming normal scaled by factor
                    fan_in = module.in_channels * module.kernel_size[0]
                    std = math.sqrt(2.0 / fan_in) / self.kaiming_scale_factor
                    nn.init.normal_(module.weight, mean=0, std=std)
                else:
                    # Use smaller initialisation to prevent tanh saturation (same as OriginalPlus)
                    nn.init.normal_(module.weight, mean=0, std=self.weight_init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class LightImprovedQVAEDecoder(nn.Module):
    """
    Lighter version of improved decoder - easier to train and debug.
    """
    
    def __init__(self, latent_dim=512, dropout=0.1, weight_init_std=0.01, weight_init_method="fixed", kaiming_scale_factor=1.0, **kwargs):
        super().__init__()
        self.weight_init_std = weight_init_std
        self.weight_init_method = weight_init_method
        self.kaiming_scale_factor = kaiming_scale_factor
        
        self.layers = nn.ModuleList([
            # Initial processing
            nn.Sequential(
                nn.Conv1d(latent_dim, 512, 3, padding=1),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                ResidualBlock(512, dropout=dropout),
            ),
            
            # Progressive upsampling layers (need 9 blocks + final layer for 3^10)
            UpsampleBlock(512, 256, scale_factor=3, dropout=dropout),
            UpsampleBlock(256, 256, scale_factor=3, dropout=dropout),
            UpsampleBlock(256, 256, scale_factor=3, dropout=dropout),
            UpsampleBlock(256, 256, scale_factor=3, dropout=dropout),
            UpsampleBlock(256, 256, scale_factor=3, dropout=dropout),
            UpsampleBlock(256, 256, scale_factor=3, dropout=dropout),
            UpsampleBlock(256, 128, scale_factor=3, dropout=dropout),
            UpsampleBlock(128, 128, scale_factor=3, dropout=dropout),
            UpsampleBlock(128, 64, scale_factor=3, dropout=dropout),
            
            # Final convolution after upsampling - use Conv1d to avoid resonance artifacts
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        ])
        
        self._initialise_weights()
    
    def forward(self, z):
        x = z.unsqueeze(-1)
        
        for layer in self.layers[:-1]:
            x = layer(x)
        
        # Final upsampling and convolution
        x = F.interpolate(x, scale_factor=3, mode='linear')
        x = torch.tanh(self.layers[-1](x))
        output = x.squeeze(1)
        
        # Debug: Check output shape
        if output.shape[-1] != 59049:
            print(f"WARNING: LightImprovedQVAEDecoder output shape is {output.shape}, expected [..., 59049]")
        
        return output
    
    def _initialise_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                if self.weight_init_method == "kaiming_scaled":
                    # Kaiming normal scaled by factor
                    fan_in = module.in_channels * module.kernel_size[0]
                    std = math.sqrt(2.0 / fan_in) / self.kaiming_scale_factor
                    nn.init.normal_(module.weight, mean=0, std=std)
                else:
                    # Smaller initialisation to prevent tanh saturation
                    nn.init.normal_(module.weight, mean=0, std=self.weight_init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


# Decoder registry
DECODER_REGISTRY = {
    'original': OriginalQVAEDecoder,
    'original-plus': OriginalPlusQVAEDecoder,
    'original-res': OriginalResQVAEDecoder,
    'original-res-plus': OriginalResPlusQVAEDecoder,
    'improved': ImprovedQVAEDecoder,
    'light_improved': LightImprovedQVAEDecoder,
}


def create_decoder(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create decoder based on config.
    
    Args:
        config: Dictionary with decoder configuration
                Should contain 'type' key and optional decoder-specific parameters
    
    Returns:
        Configured decoder instance
    """
    decoder_config = config.get('decoder', {})
    decoder_type = decoder_config.get('type', 'original')  # Default to original
    
    if decoder_type not in DECODER_REGISTRY:
        available = ', '.join(DECODER_REGISTRY.keys())
        raise ValueError(f"Unknown decoder type '{decoder_type}'. Available: {available}")
    
    decoder_class = DECODER_REGISTRY[decoder_type]
    
    # Extract decoder-specific parameters
    decoder_params = {k: v for k, v in decoder_config.items() if k != 'type'}
    
    # Add latent_dim from model config
    decoder_params['latent_dim'] = config['model']['latent_dim']
    
    return decoder_class(**decoder_params)