"""
QVAEDecoder: Transposed SampleCNN for audio generation.

Exact reversal of the 3^9-SampleCNN architecture from Lee et al. (2017).
Takes 512-dim latent vectors and generates 59,049 raw audio samples (~3 seconds at 22050 Hz).
Uses ConvTranspose1d layers with kernel=3, stride=3 for direct 3x upsampling operations.

--- NOTE: This decoder has been retired and training now uses decoders.py ---
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QVAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer 11^T: 512 -> 512
        self.conv11 = nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # Layer 10^T: 512 -> 256  
        self.conv10 = nn.ConvTranspose1d(512, 256, kernel_size=3, stride=3, padding=0)
        
        # Layers 9^T - 5^T: 256 -> 256
        self.conv9 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.conv8 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.conv7 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.conv6 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        self.conv5 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=3, padding=0)
        
        # Layer 4^T: 256 -> 128
        self.conv4 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=3, padding=0)
        
        # Layers 3^T, 2^T: 128 -> 128
        self.conv3 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=3, padding=0)
        self.conv2 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=3, padding=0)
        
        # Layer 1^T: 128 -> 1 (final output)
        self.conv1 = nn.ConvTranspose1d(128, 1, kernel_size=3, stride=3, padding=0)
        
        # Batch normalisation and activation
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(256), nn.BatchNorm1d(256),
            nn.BatchNorm1d(256), nn.BatchNorm1d(128), nn.BatchNorm1d(128),
            nn.BatchNorm1d(128)
        ])
        
        # Initialise weights
        self._initialise_weights()
    
    def forward(self, z):
        # Input: [B, 512] -> Reshape to [B, 512, 1]
        x = z.unsqueeze(-1)
        
        # Layer 11^T
        x = F.relu(self.bn_layers[0](self.conv11(x)))
        
        # Layer 10^T
        x = F.relu(self.bn_layers[1](self.conv10(x)))
        
        # Layer 9^T
        x = F.relu(self.bn_layers[2](self.conv9(x)))
        
        # Layer 8^T
        x = F.relu(self.bn_layers[3](self.conv8(x)))
        
        # Layer 7^T
        x = F.relu(self.bn_layers[4](self.conv7(x)))
        
        # Layer 6^T
        x = F.relu(self.bn_layers[5](self.conv6(x)))
        
        # Layer 5^T
        x = F.relu(self.bn_layers[6](self.conv5(x)))
        
        # Layer 4^T
        x = F.relu(self.bn_layers[7](self.conv4(x)))
        
        # Layer 3^T
        x = F.relu(self.bn_layers[8](self.conv3(x)))
        
        # Layer 2^T
        x = F.relu(self.bn_layers[9](self.conv2(x)))
        
        # Layer 1^T (final layer, no BN/activation)
        x = self.conv1(x)
        
        # Apply tanh to clamp output to [-1, 1] range
        x = torch.tanh(x)
        
        return x  # [B, 1, 59049] - keep channel dimension
    
    def _initialise_weights(self):
        """Initialise convolutional layer weights using Kaiming normal initialisation."""
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)