"""
QVAE Loss Functions

Multi-component perceptual loss for high-quality audio generation:
- Multi-Scale Spectral Loss: Primary reconstruction using multiple STFT window sizes
- Mel-Spectrogram Loss: Perceptual alignment using mel-scale frequency analysis  
- KL Divergence Loss: VAE regularisation with low weight for frozen encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def compute_trailing_silence_mask(spectrogram, silence_threshold):
    """
    Compute mask for trailing silence (padding) only.
    Assumes spectrogram is [B, freq, time] based on losses.py STFT output.
    """
    if torch.is_complex(spectrogram):
        spectrogram = torch.abs(spectrogram)
    
    energy = spectrogram.mean(dim=1, keepdim=True)  # [B, 1, time]
    above_threshold = (energy > silence_threshold).float()
    
    # Reverse cumsum to find trailing silence
    reverse_cumsum = torch.flip(
        torch.cumsum(torch.flip(above_threshold, dims=[-1]), dim=-1), 
        dims=[-1]
    )
    
    return (reverse_cumsum > 0).float()  # [B, 1, time]


class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale STFT loss using multiple window sizes for comprehensive frequency analysis."""
    
    def __init__(self, stft_config, use_silence_masking=False, silence_threshold=1e-4):
        super().__init__()
        self.use_silence_masking = use_silence_masking
        self.silence_threshold = silence_threshold
        
        # STFT configurations from config
        self.stft_configs = [
            {'n_fft': n_fft, 'hop_length': hop_length, 'win_length': win_length}
            for n_fft, hop_length, win_length in zip(
                stft_config['window_sizes'],
                stft_config['hop_lengths'], 
                stft_config['win_lengths']
            )
        ]
        
        # Pre-compute Hann windows for each configuration
        self.windows = nn.ParameterDict({
            f'window_{config["n_fft"]}': nn.Parameter(
                torch.hann_window(config['win_length']), requires_grad=False
            ) for config in self.stft_configs
        })
    
    def _compute_stft_loss(self, pred_stft, target_stft):
        """Compute spectral convergence + log magnitude loss for single STFT."""
        if self.use_silence_masking:
            mask = compute_trailing_silence_mask(target_stft, self.silence_threshold)
            pred_stft = pred_stft * mask
            target_stft = target_stft * mask
        
        # Spectral convergence loss: ||target - pred||_F / ||target||_F
        spec_conv = torch.norm(target_stft - pred_stft, p='fro') / (torch.norm(target_stft, p='fro') + 1e-7)
        
        # Log magnitude loss: ||log|target| - log|pred|||_1
        pred_mag = torch.abs(pred_stft) + 1e-7
        target_mag = torch.abs(target_stft) + 1e-7
        log_mag = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
        
        # Combined loss with equal weighting
        return spec_conv + log_mag
    
    def forward(self, pred_audio, target_audio):
        """
        Args:
            pred_audio: [B, 1, T] predicted audio
            target_audio: [B, 1, T] target audio
        Returns:
            Multi-scale spectral loss
        """
        total_loss = 0.0
        
        for config in self.stft_configs:
            window = self.windows[f'window_{config["n_fft"]}']
            
            # Compute STFT for predicted and target audio (squeeze channel dim for torch.stft)
            pred_stft = torch.stft(
                pred_audio.squeeze(1).float(),  # [B, 1, T] -> [B, T] and to fp32
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length'],
                window=window,
                return_complex=True
            )
            
            target_stft = torch.stft(
                target_audio.squeeze(1).float(),  # [B, 1, T] -> [B, T] and to fp32
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length'],
                window=window,
                return_complex=True
            )
            
            # Accumulate loss for this scale
            total_loss += self._compute_stft_loss(pred_stft, target_stft)
        
        return total_loss / len(self.stft_configs)
    
    def forward_precomputed(self, pred_audio, target_stfts):
        """
        Args:
            pred_audio: [B, 1, T] predicted audio
            target_stfts: List of pre-computed target STFT tensors [B, 1, freq, time]
        Returns:
            Multi-scale spectral loss
        """
        total_loss = 0.0
        
        for i, config in enumerate(self.stft_configs):
            window = self.windows[f'window_{config["n_fft"]}']
            
            # Compute STFT for predicted audio only (squeeze channel dim for torch.stft)
            pred_stft = torch.stft(
                pred_audio.squeeze(1).float(),  # [B, 1, T] -> [B, T] and to fp32
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length'],
                window=window,
                return_complex=True
            )
            
            # Use pre-computed target STFT
            target_stft = target_stfts[i]
            
            # Accumulate loss for this scale
            total_loss += self._compute_stft_loss(pred_stft, target_stft)
        
        return total_loss / len(self.stft_configs)


class MelSpectrogramLoss(nn.Module):
    """Mel-spectrogram loss for perceptual alignment with human auditory processing."""
    
    def __init__(self, mel_config, sample_rate=22050, use_silence_masking=False, silence_threshold=1e-4):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_silence_masking = use_silence_masking
        self.silence_threshold = silence_threshold
        
        # Mel-spectrogram configurations from config
        self.mel_configs = [
            {'n_fft': n_fft, 'hop_length': hop_length, 'f_min': mel_config['f_min'], 'f_max': mel_config['f_max']}
            for n_fft, hop_length in zip(mel_config['n_fft'], mel_config['hop_lengths'])
        ]
        
        
        # Pre-compute mel-spectrogram transforms
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                n_mels=mel_config['n_mels'],
                f_min=config['f_min'],
                f_max=config['f_max']
            ) for config in self.mel_configs
        ])
    
    def _compute_mel_loss(self, pred_mel, target_mel):
        """Core mel loss computation."""
        if self.use_silence_masking:
            mask = compute_trailing_silence_mask(target_mel, self.silence_threshold)
            pred_mel = pred_mel * mask
            target_mel = target_mel * mask
        
        # Convert to log scale
        pred_log_mel = torch.log(pred_mel + 1e-7)
        target_log_mel = torch.log(target_mel + 1e-7)
        
        # L1 loss in log-mel space
        loss = F.l1_loss(pred_log_mel, target_log_mel)
        return loss
    
    def forward(self, pred_audio, target_audio):
        """
        Args:
            pred_audio: [B, 1, T] predicted audio
            target_audio: [B, 1, T] target audio
        Returns:
            Mel-spectrogram loss
        """
        total_loss = 0.0
        
        for mel_transform in self.mel_transforms:
            # Compute mel spectrograms (squeeze channel dim for mel transform)
            # Convert to float32 to avoid fp16 overflow
            pred_mel = mel_transform(pred_audio.squeeze(1).float())  # [B, 1, T] -> [B, T] and to fp32
            target_mel = mel_transform(target_audio.squeeze(1).float())  # [B, 1, T] -> [B, T] and to fp32
            
            # Accumulate loss
            total_loss += self._compute_mel_loss(pred_mel, target_mel)
        
        return total_loss / len(self.mel_transforms)
    
    def forward_precomputed(self, pred_audio, target_mels):
        """
        Args:
            pred_audio: [B, 1, T] predicted audio
            target_mels: List of pre-computed target mel spectrograms [B, 1, mels, time]
        Returns:
            Mel-spectrogram loss
        """
        total_loss = 0.0
        
        # Compute mel spectrogram for predicted audio only (squeeze channel dim)
        # Convert to float32 to avoid fp16 overflow in mel transform
        pred_audio_fp32 = pred_audio.squeeze(1).float()  # [B, 1, T] -> [B, T] and to fp32
        
        for i, mel_transform in enumerate(self.mel_transforms):
            pred_mel = mel_transform(pred_audio_fp32)
            
            # Use pre-computed target mel
            target_mel = target_mels[i]
            
            # Accumulate loss
            mel_loss_i = self._compute_mel_loss(pred_mel, target_mel)
            total_loss += mel_loss_i
        
        return total_loss / len(self.mel_transforms)


class WaveformLoss(nn.Module):
    """
    L1 time-domain waveform loss for direct temporal alignment.
    Experimental.
    """
    
    def __init__(self, use_silence_masking=False, silence_threshold=1e-4):
        super().__init__()
        self.use_silence_masking = use_silence_masking
        self.silence_threshold = silence_threshold
    
    def forward(self, pred_audio, target_audio):
        """
        Args:
            pred_audio: [B, T] predicted audio
            target_audio: [B, T] target audio
        Returns:
            L1 waveform loss
        """
        if self.use_silence_masking:
            # Create mask based on target audio energy
            # Use simple energy threshold for time-domain masking
            target_energy = target_audio.abs()
            mask = (target_energy > self.silence_threshold).float()
            pred_audio = pred_audio * mask
            target_audio = target_audio * mask
        
        # L1 loss in time domain
        return F.l1_loss(pred_audio, target_audio)


class KLDivergenceLoss(nn.Module):
    """KL divergence loss for VAE regularisation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mu, logvar):
        """
        Args:
            mu: [B, latent_dim] mean of latent distribution
            logvar: [B, latent_dim] log variance of latent distribution
        Returns:
            KL divergence loss
        """
        # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0,I)
        # = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kl_loss)


class QVAELoss(nn.Module):
    """Combined QVAE loss with multi-component perceptual losses."""
    
    def __init__(self, config, use_silence_masking=None):
        super().__init__()
        
        # Use config setting if not explicitly overridden
        if use_silence_masking is None:
            use_silence_masking = config['loss'].get('use_silence_masking', False)
        
        silence_threshold = config['loss'].get('silence_threshold', 1e-4)
        
        # Loss components
        self.spectral_loss = MultiScaleSpectralLoss(
            config['target_spectrograms']['stft'], 
            use_silence_masking=use_silence_masking,
            silence_threshold=silence_threshold
        )
        self.mel_loss = MelSpectrogramLoss(
            config['target_spectrograms']['mel'], 
            config['data']['target_sample_rate'],
            use_silence_masking=use_silence_masking,
            silence_threshold=silence_threshold
        )
        self.waveform_loss = WaveformLoss(
            use_silence_masking=use_silence_masking,
            silence_threshold=silence_threshold
        )
        self.kl_loss = KLDivergenceLoss()
        
        # Loss weights
        self.alpha_spectral = config['loss']['alpha_spectral']
        self.alpha_mel = config['loss']['alpha_mel']
        self.alpha_waveform = config['loss'].get('alpha_waveform', 0.0)  # Default to 0 if not specified
        self.beta_kl = config['loss']['beta_kl']
    
    def forward(self, pred_audio, target_audio, mu, logvar):
        """
        Args:
            pred_audio: [B, T] predicted audio from decoder
            target_audio: [B, T] target reference audio
            mu: [B, latent_dim] mean of latent distribution
            logvar: [B, latent_dim] log variance of latent distribution
        Returns:
            dict with total loss and individual components
        """
        # Compute individual loss components
        spectral = self.spectral_loss(pred_audio, target_audio)
        mel = self.mel_loss(pred_audio, target_audio)
        waveform = self.waveform_loss(pred_audio, target_audio)
        kl = self.kl_loss(mu, logvar)
        
        # Weighted combination
        total_loss = (
            self.alpha_spectral * spectral +
            self.alpha_mel * mel +
            self.alpha_waveform * waveform +
            self.beta_kl * kl
        )
        
        return {
            'total_loss': total_loss,
            'spectral_loss': spectral,
            'mel_loss': mel,
            'waveform_loss': waveform,
            'kl_loss': kl
        }
    
    def forward_precomputed(self, pred_audio, target_spectrograms, mu, logvar, target_audio=None):
        """
        Args:
            pred_audio: [B, T] predicted audio from decoder
            target_spectrograms: dict with 'stft' and 'mel' keys containing pre-computed spectrograms
            mu: [B, latent_dim] mean of latent distribution
            logvar: [B, latent_dim] log variance of latent distribution
            target_audio: [B, T] target audio for waveform loss (optional)
        Returns:
            dict with total loss and individual components
        """
        # Compute individual loss components using pre-computed targets
        spectral = self.spectral_loss.forward_precomputed(pred_audio, target_spectrograms['stft'])
        mel = self.mel_loss.forward_precomputed(pred_audio, target_spectrograms['mel'])
        kl = self.kl_loss(mu, logvar)
        
        # Compute waveform loss if target audio is provided and weight > 0
        waveform = torch.tensor(0.0, device=pred_audio.device)
        if target_audio is not None and self.alpha_waveform > 0:
            waveform = self.waveform_loss(pred_audio, target_audio)
        
        # Weighted combination
        total_loss = (
            self.alpha_spectral * spectral +
            self.alpha_mel * mel +
            self.alpha_waveform * waveform +
            self.beta_kl * kl
        )
        
        return {
            'total_loss': total_loss,
            'spectral_loss': spectral,
            'mel_loss': mel,
            'waveform_loss': waveform,  # Always 0 in precomputed mode
            'kl_loss': kl
        }