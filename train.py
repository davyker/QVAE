"""
QVAE Training Script

PyTorch Lightning training pipeline for QVAE model with two-tier validation.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.qvae.callbacks import AudioGenerationCallback, DiagnosticCallback
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml
from pathlib import Path
import argparse
import numpy as np
import pickle

# QVAE components
from src.qvae.encoder import QVAEEncoder
from src.qvae.decoders import create_decoder
from src.qvae.losses import QVAELoss
from src.qvae.data import create_dataloader, precompute_features
from src.qvae.utils import load_config


class QVAEModule(pl.LightningModule):
    """PyTorch Lightning module for QVAE training."""
    
    def __init__(self, config, mode='train', overfit_mode=False, overfit_samples=5, reset_epoch_on_load=False):
        super().__init__()
        self.config = config
        self.mode = mode
        self.overfit_mode = overfit_mode
        self.overfit_samples = overfit_samples
        self.reset_epoch_on_load = reset_epoch_on_load
        
        # Determine training mode
        self.fine_tuning_enabled = config['model']['encoder_fine_tuning']['enabled']
        self.from_scratch = config['model'].get('from_scratch', False)
        
        if mode == 'train':
            # === TRAINING MODE ===
            # Load precomputed data once for all dataloaders
            self.encoder_features, self.target_spectrograms = precompute_features(config)
            
            # Load input mel-spectrograms if fine-tuning OR from_scratch
            if self.fine_tuning_enabled or self.from_scratch:
                cache_dir = Path(config['data']['cache_dir'])
                mel_spectrograms_cache_path = cache_dir / "input_mel_spectrograms.pkl"
                if not mel_spectrograms_cache_path.exists():
                    raise FileNotFoundError(f"Input mel-spectrograms cache not found: {mel_spectrograms_cache_path}")
                print("Loading input mel-spectrograms cache...")
                with open(mel_spectrograms_cache_path, 'rb') as f:
                    self.input_mel_spectrograms = pickle.load(f)
                print(f"Mel-spectrograms cache loaded successfully ({len(self.input_mel_spectrograms)} sounds)")
            else:
                self.input_mel_spectrograms = None
            
            # Load target audio if waveform loss is enabled
            if config['loss'].get('alpha_waveform', 0.0) > 0:
                cache_dir = Path(config['data']['cache_dir'])
                target_audio_cache_path = cache_dir / "target_audio.pkl"
                if not target_audio_cache_path.exists():
                    raise FileNotFoundError(f"Target audio cache not found: {target_audio_cache_path}")
                print("Loading target audio cache for waveform loss...")
                with open(target_audio_cache_path, 'rb') as f:
                    self.target_audio = pickle.load(f)
                print(f"Target audio cache loaded successfully ({len(self.target_audio)} sounds)")
            else:
                self.target_audio = None
            
            # Track training start time
            import time
            self.training_start_time = time.time()
            
            # Initialise loss function
            self.loss_fn = QVAELoss(config)
            
            if config['loss'].get('use_silence_masking', False):
                print("Using silence masking with threshold:", config['loss'].get('silence_threshold', 1e-4))
            
            # Save hyperparameters for Lightning
            self.save_hyperparameters(config)
            
        elif mode == 'inference':
            # === INFERENCE MODE ===
            # No precomputed data needed
            self.encoder_features = None
            self.target_spectrograms = None
            self.input_mel_spectrograms = None
            self.target_audio = None
            
            # Force deterministic sampling for inference
            self.config['model']['deterministic_sampling'] = True
            
            # Import preprocessing components for on-the-fly mel-spectrogram computation
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '../qvim-baseline/src'))
            from qvim_mn_baseline.mn.preprocess import AugmentMelSTFT
            
            # Initialise preprocessor for mel-spectrogram computation
            self.preprocessor = AugmentMelSTFT()
            self.preprocessor.eval()  # No augmentation in inference
            
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'")
        
        # Initialise models (common to both modes)
        self.encoder = QVAEEncoder(
            pretrained_name=config['model']['encoder_name'],
            checkpoint_path=None if config['model'].get('from_scratch', False) else config['model']['encoder_checkpoint'],
            fine_tuning_config=config['model']['encoder_fine_tuning'],
            latent_dim=config['model']['latent_dim'],
            feature_space=config['model'].get('feature_space', 'final')
        )
        self.decoder = create_decoder(config)
        
        # Note: Encoder freezing/unfreezing is handled by the encoder itself based on fine_tuning_config
        
    def forward(self, input_data):
        """VAE forward pass: input → encoder → VAE head → sample → decode
        Args:
            input_data: encoder_features [B, 960] if frozen mode, or mel_spectrograms [B, 1, 128, T] if fine-tuning
        Returns:
            generated_audio: [B, 1, T] generated audio with channel dimension
            mu, logvar: VAE latent distribution parameters
        """
        if self.fine_tuning_enabled or self.from_scratch:
            # Fine-tuning or from-scratch mode: encode mel-spectrograms on-the-fly
            mu, logvar = self.encoder.forward(input_data)
        else:
            # Frozen pretrained mode: use precomputed features
            mu, logvar = self.encoder.forward_precomputed(input_data)
            
        z = self.encoder.reparameterise(mu, logvar, deterministic=self.config['model']['deterministic_sampling'])
        
        # For single-sample batches during validation callbacks, temporarily set decoder to eval mode
        if z.size(0) == 1 and self.training:
            decoder_was_training = self.decoder.training
            self.decoder.eval()
            generated_audio = self.decoder(z)
            if decoder_was_training:
                self.decoder.train()
        else:
            generated_audio = self.decoder(z)
            
        return generated_audio, mu, logvar
    
    def inference_forward(self, audio_paths):
        """Inference forward pass: audio files → mel-spectrogram → encoder → VAE → decoder
        Args:
            audio_paths: List of paths to audio files or single path
        Returns:
            generated_audio: [B, 1, T] generated audio
            mu, logvar: VAE latent distribution parameters
        """
        import torch
        import torchaudio
        import numpy as np
        from pathlib import Path
        
        # Import load_audio_qvim_style for consistent preprocessing
        from src.qvae.data import load_audio_qvim_style
        
        # Handle single path or list of paths
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [audio_paths]
        
        # Load and preprocess audio files
        audio_batch = []
        for path in audio_paths:
            # Load audio using QVIM style (ensures 32kHz, proper length, etc.)
            audio = load_audio_qvim_style(path)
            audio_batch.append(torch.from_numpy(audio))
        
        # Stack into batch tensor
        audio_tensor = torch.stack(audio_batch)  # [B, samples]
        
        # Move to appropriate device
        device = next(self.parameters()).device
        audio_tensor = audio_tensor.to(device)
        
        # Convert to mel-spectrograms using preprocessor
        with torch.no_grad():
            mel_spectrograms = self.preprocessor(audio_tensor)  # [B, 128, T]
            
            # Add channel dimension for encoder
            mel_spectrograms = mel_spectrograms.unsqueeze(1)  # [B, 1, 128, T]
            
            # Pass through the standard forward method (deterministic sampling already set in __init__)
            generated_audio, mu, logvar = self.forward(mel_spectrograms)
        
        return generated_audio, mu, logvar
        
    def get_current_beta_kl(self):
        """Get current beta_kl value based on schedule if enabled."""
        schedule_config = self.config['training'].get('beta_kl_schedule', {})
        
        schedule_type = schedule_config.get('type', 'fixed')
        
        if schedule_type == 'fixed':
            # Use static beta_kl from loss config
            return self.config['loss']['beta_kl']
        
        elif schedule_type == 'cyclical':
            # Cyclical annealing with hold phase
            beta_min = schedule_config['min']
            beta_max = schedule_config['max']
            cycle_length = schedule_config['cycle_length']
            proportion = schedule_config.get('proportion', 0.5)  # Proportion of cycle for annealing
            
            current_step = self.global_step
            cycle_position = current_step % cycle_length
            annealing_steps = int(cycle_length * proportion)
            
            if cycle_position < annealing_steps:
                # Annealing phase: increase from min to max
                progress = cycle_position / annealing_steps
                beta_kl = beta_min + (beta_max - beta_min) * progress
            else:
                # Hold phase: stay at max
                beta_kl = beta_max
                
            return beta_kl
            
        elif schedule_type == 'linear':
            # Linear rampup schedule (original implementation)
            initial = schedule_config['initial']
            target = schedule_config['target']
            rampup_fraction = schedule_config['rampup_fraction']
            warmup_epochs = self.config['training']['scheduler']['warmup_epochs']
            total_epochs = self.config['training']['num_epochs']
            
            # Calculate rampup boundaries
            rampup_start = warmup_epochs
            rampup_end = warmup_epochs + int(rampup_fraction * total_epochs)
            
            current_epoch = self.current_epoch
            
            # Apply schedule
            if current_epoch < rampup_start:
                # Warmup phase
                beta_kl = initial
            elif current_epoch < rampup_end:
                # Rampup phase
                progress = (current_epoch - rampup_start) / (rampup_end - rampup_start)
                beta_kl = initial + progress * (target - initial)
            else:
                # Target phase
                beta_kl = target
                
            return beta_kl
        
        else:
            raise ValueError(f"Unknown beta_kl schedule type: {schedule_type}")
    
    def training_step(self, batch, batch_idx):
        """Training step with conditional input based on fine-tuning mode"""
        target_spectrograms = batch['target_spectrograms']
        
        # Get input based on mode
        if self.fine_tuning_enabled or self.from_scratch:
            input_data = batch['input_mel_spectrograms']
        else:
            input_data = batch['encoder_features']
        
        # Forward pass
        generated_audio, mu, logvar = self.forward(input_data)
        
        # Update beta_kl in loss function if using schedule
        current_beta_kl = self.get_current_beta_kl()
        self.loss_fn.beta_kl = current_beta_kl
        
        # Compute loss using precomputed targets
        target_audio = batch.get('target_audio', None)  # Get target audio if available
        loss_components = self.loss_fn.forward_precomputed(
            generated_audio, target_spectrograms, mu, logvar, target_audio
        )
        
        # Debug: Print loss components for problematic batches
        if torch.isnan(loss_components['total_loss']):
            print(f"\n!!! NaN detected in batch {batch_idx} !!!")
            print(f"Spectral loss: {loss_components['spectral_loss'].item()}")
            print(f"Mel loss: {loss_components['mel_loss'].item()}")
            print(f"KL loss: {loss_components['kl_loss'].item()}")
            if 'vocal_file' in batch:
                print("Files in this batch:")
                for vocal, ref in zip(batch['vocal_file'], batch['ref_file']):
                    print(f"  vocal={vocal}, ref={ref}")
        
        # Log training metrics with explicit batch size
        batch_size = input_data.shape[0]
        self.log('train_loss', loss_components['total_loss'], batch_size=batch_size, prog_bar=True)
        self.log('train_spectral_loss', loss_components['spectral_loss'], batch_size=batch_size)
        self.log('train_mel_loss', loss_components['mel_loss'], batch_size=batch_size)
        self.log('train_waveform_loss', loss_components['waveform_loss'], batch_size=batch_size)
        self.log('train_kl_loss', loss_components['kl_loss'], batch_size=batch_size)
        
        # Log current beta_kl value
        self.log('beta_kl', current_beta_kl, batch_size=batch_size)
        
        # Log cycle position if using cyclical schedule
        if self.config['training'].get('beta_kl_schedule', {}).get('type') == 'cyclical':
            cycle_length = self.config['training']['beta_kl_schedule']['cycle_length']
            cycle_position = self.global_step % cycle_length
            self.log('beta_kl_cycle_position', cycle_position / cycle_length, batch_size=batch_size)
        
        # Log current learning rate to progress bar
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True)
        
        # Log both learning rates separately for WandB charts
        if self.config['model'].get('from_scratch', False):
            # From scratch mode - all have same LR
            self.log('lr/all_components', self.optimizers().param_groups[0]['lr'])
        else:
            # Different LRs for encoder vs vae_head/decoder
            self.log('lr/encoder', self.optimizers().param_groups[0]['lr'])
            self.log('lr/vae_head_and_decoder', self.optimizers().param_groups[1]['lr'])
        
        return loss_components['total_loss']
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Validation step for interpolation/extrapolation datasets"""
        # Keep model in training mode for overfitting tests
        if self.overfit_mode:
            self.train()
            
        target_spectrograms = batch['target_spectrograms']
        
        # Get input based on mode
        if self.fine_tuning_enabled or self.from_scratch:
            input_data = batch['input_mel_spectrograms']
        else:
            input_data = batch['encoder_features']
        
        # Forward pass
        generated_audio, mu, logvar = self.forward(input_data)
        
        # Compute loss
        target_audio = batch.get('target_audio', None)  # Get target audio if available
        loss_components = self.loss_fn.forward_precomputed(
            generated_audio, target_spectrograms, mu, logvar, target_audio
        )
        
        # Log validation metrics per dataset with explicit batch size
        batch_size = input_data.shape[0]
        prefix = 'val_interpolation' if dataloader_idx == 0 else 'val_extrapolation'
        self.log(f'{prefix}_loss', loss_components['total_loss'], batch_size=batch_size, prog_bar=True)
        self.log(f'{prefix}_spectral_loss', loss_components['spectral_loss'], batch_size=batch_size)
        self.log(f'{prefix}_mel_loss', loss_components['mel_loss'], batch_size=batch_size)
        self.log(f'{prefix}_waveform_loss', loss_components['waveform_loss'], batch_size=batch_size)
        self.log(f'{prefix}_kl_loss', loss_components['kl_loss'], batch_size=batch_size)
        
        return loss_components['total_loss']
    
    def on_validation_epoch_end(self):
        """Print total training time after each epoch"""
        import time
        elapsed = time.time() - self.training_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
    def configure_optimizers(self):
        """Configure optimiser and scheduler with differential learning rates"""
        
        from_scratch = self.config['model'].get('from_scratch', False)
        
        if from_scratch:
            # From scratch mode: single learning rate for all parameters
            lr = self.config['training']['learning_rate']
            param_groups = [
                {'params': self.encoder.backbone.parameters(), 'lr': lr, 'name': 'encoder'},
                {'params': list(self.encoder.mu_head.parameters()) + list(self.encoder.logvar_head.parameters()), 'lr': lr, 'name': 'vae_head'},
                {'params': self.decoder.parameters(), 'lr': lr, 'name': 'decoder'}
            ]
            
            print(f"\n🎲 From Scratch Mode: LR = {lr} for all components")
            
        elif self.fine_tuning_enabled:
            # Fine-tuning mode: different learning rates for encoder vs others
            encoder_lr = self.config['model']['encoder_fine_tuning']['learning_rate']
            other_lr = self.config['training']['learning_rate']
            
            # Get trainable encoder parameters
            encoder_params = self.encoder.get_trainable_encoder_params()
            
            # Create parameter groups with different learning rates
            param_groups = [
                {'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'},
                {'params': list(self.encoder.mu_head.parameters()) + list(self.encoder.logvar_head.parameters()), 'lr': other_lr, 'name': 'vae_head'},
                {'params': self.decoder.parameters(), 'lr': other_lr, 'name': 'decoder'}
            ]
            
            print(f"\n🎯 Differential Learning Rates:")
            print(f"  Encoder layers: {encoder_lr}")
            print(f"  VAE head + Decoder: {other_lr}")
            
        else:
            # Frozen mode: only VAE head + decoder
            param_groups = [
                {'params': list(self.encoder.mu_head.parameters()) + list(self.encoder.logvar_head.parameters()), 'lr': self.config['training']['learning_rate'], 'name': 'vae_head'},
                {'params': self.decoder.parameters(), 'lr': self.config['training']['learning_rate'], 'name': 'decoder'}
            ]
            
            print(f"\n🔒 Frozen Encoder Mode: LR = {self.config['training']['learning_rate']}")
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.config['training']['adam_beta1'], 
                   self.config['training']['adam_beta2']),
            weight_decay=1e-4
        )
        
        # Cosine annealing scheduler with warmup
        scheduler_config = self.config['training']['scheduler']
        total_epochs = self.config['training']['num_epochs']
        warmup_epochs = scheduler_config['warmup_epochs']
        
        if warmup_epochs > 0:
            # Main scheduler (after warmup)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=scheduler_config['eta_min']
            )
            
            # Warmup scheduler (first few epochs)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # Start at 10% of base LR
                end_factor=1.0,    # Reach 100% of base LR
                total_iters=warmup_epochs
            )
            
            # Sequential scheduler: warmup → cosine annealing
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            # No warmup - just use cosine annealing from the start
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=scheduler_config['eta_min']
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
        
    def train_dataloader(self):
        """Return training dataloader"""
        if self.overfit_mode:
            # Create and store overfit dataloader if not already created
            if not hasattr(self, '_overfit_dataloader'):
                self._overfit_dataloader = create_dataloader(self.config, split='train', shuffle=False, 
                                           encoder_features=self.encoder_features, 
                                           target_spectrograms=self.target_spectrograms,
                                           input_mel_spectrograms=self.input_mel_spectrograms,
                                           target_audio=self.target_audio,
                                           overfit_subset=self.overfit_samples)
            return self._overfit_dataloader
        else:
            return create_dataloader(self.config, split='train', shuffle=True, 
                                   encoder_features=self.encoder_features, 
                                   target_spectrograms=self.target_spectrograms,
                                   input_mel_spectrograms=self.input_mel_spectrograms,
                                   target_audio=self.target_audio)
        
    def val_dataloader(self):
        """Return validation dataloaders [interpolation, extrapolation]"""
        if self.overfit_mode:
            # Use small validation dataloader with original samples (no 500 copies)
            if not hasattr(self, '_overfit_val_dataloader'):
                self._overfit_val_dataloader = create_dataloader(self.config, split='train', shuffle=False,
                                           encoder_features=self.encoder_features, 
                                           target_spectrograms=self.target_spectrograms,
                                           input_mel_spectrograms=self.input_mel_spectrograms,
                                           target_audio=self.target_audio,
                                           overfit_subset=self.overfit_samples,
                                           overfit_training_mode=False)
            return [self._overfit_val_dataloader, self._overfit_val_dataloader]
        else:
            return [
                create_dataloader(self.config, split='interpolation_val', shuffle=False,
                                encoder_features=self.encoder_features, 
                                target_spectrograms=self.target_spectrograms,
                                input_mel_spectrograms=self.input_mel_spectrograms,
                                target_audio=self.target_audio),
                create_dataloader(self.config, split='extrapolation_val', shuffle=False,
                                encoder_features=self.encoder_features, 
                                target_spectrograms=self.target_spectrograms,
                                input_mel_spectrograms=self.input_mel_spectrograms,
                                target_audio=self.target_audio)
            ]


def load_training_config(config_path="config.yaml"):
    """Load and validate training configuration"""
    config = load_config(config_path)
    
    # Validate required training parameters
    required_keys = [
        'training.batch_size', 'training.learning_rate', 'training.num_epochs',
        'training.grad_clip_threshold', 'training.early_stopping.monitor'
    ]
    
    for key in required_keys:
        if not _get_nested_value(config, key):
            raise ValueError(f"Missing required config key: {key}")
    
    return config


def _get_nested_value(config, key_path):
    """Helper to get nested config values (e.g., 'training.batch_size')"""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if key not in value:
            return None
        value = value[key]
    return value


def setup_callbacks(config, logger=None):
    """Configure training callbacks"""
    callbacks = []
    
    # Model checkpointing
    # Use run name if available, otherwise timestamp
    from datetime import datetime
    if logger and hasattr(logger, 'experiment'):
        run_name = logger.experiment.name
    else:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_dir = Path(config['logging']['run_dir']) / run_name
    
    # Save a copy of the config to the run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = run_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to: {config_save_path}")
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='epoch_{epoch:04d}',
        monitor='val_interpolation_loss/dataloader_idx_0',  # Monitor validation loss
        save_top_k=3,  # Keep only 3 best checkpoints
        every_n_epochs=1,  # Check every epoch
        mode='min',  # Lower loss is better
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_config = config['training']['early_stopping']
    early_stopping = EarlyStopping(
        monitor=early_stop_config['monitor'],
        patience=early_stop_config['patience'],
        min_delta=early_stop_config['min_delta'],
        mode='min',
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Audio generation callback
    audio_dir = run_dir / "generated_samples"
    audio_callback = AudioGenerationCallback(
        dirpath=str(audio_dir),
        num_samples=config['logging'].get('audio_samples_per_val', 3),
        log_to_wandb=True,
        generation_frequency=config['logging'].get('audio_generation_frequency', 1)
    )
    callbacks.append(audio_callback)
    
    # Diagnostic callback (if enabled)
    if config.get('diagnostics', {}).get('enabled', False):
        diagnostic_dir = run_dir / "diagnostics"
        diagnostic_callback = DiagnosticCallback(
            dirpath=str(diagnostic_dir),
            config=config,
            diagnostic_frequency=config['diagnostics'].get('frequency', 5)
        )
        callbacks.append(diagnostic_callback)
    
    return callbacks


def setup_logger(config, args):
    """Configure WandB logger"""
    wandb_project = args.wandb_project or config['logging']['wandb_project']
    
    logger = WandbLogger(
        project=wandb_project,
        entity=config['logging'].get('wandb_entity'),
        log_model=False,
        save_dir=config['logging']['run_dir']
    )
    
    return logger


def setup_trainer(config, callbacks, logger, args):
    """Configure PyTorch Lightning trainer"""
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        precision=32,  # 32-bit precision (was 16-mixed)
        gradient_clip_val=config['training']['grad_clip_threshold'],
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=config['logging']['log_frequency'],
        val_check_interval=1.0,  # Validate every epoch
        fast_dev_run=args.debug,  # Quick debug mode
        deterministic=False  # Disabled due to reflection_pad1d incompatibility
    )
    
    return trainer


def set_random_seeds(seed=422):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)


def log_model_info(model, config):
    """Log model architecture and parameter counts"""
    # Overall model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"=== Model Parameter Summary ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.1f}%")
    print()
    
    # Component breakdown
    components = {
        'encoder_backbone': model.encoder.backbone,
        'encoder_mu_head': model.encoder.mu_head,
        'encoder_logvar_head': model.encoder.logvar_head,
        'decoder': model.decoder
    }
    
    wandb_log = {
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/trainable_percentage": 100 * trainable_params / total_params
    }
    
    print("Component breakdown:")
    for component_name, component in components.items():
        comp_total = sum(p.numel() for p in component.parameters())
        comp_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
        comp_percentage = 100 * comp_trainable / comp_total if comp_total > 0 else 0
        
        print(f"  {component_name}:")
        print(f"    Total params: {comp_total:,}")
        print(f"    Trainable params: {comp_trainable:,}")
        print(f"    Trainable percentage: {comp_percentage:.1f}%")
        
        # Log to WandB
        wandb_log[f"model/{component_name}_total_params"] = comp_total
        wandb_log[f"model/{component_name}_trainable_params"] = comp_trainable
        wandb_log[f"model/{component_name}_trainable_percentage"] = comp_percentage
    
    print("=" * 35)


def train_qvae(config, args):
    """Main training function"""
    # Set random seeds from config
    seed = config['training'].get('random_seed', 422)  # Default to 422 if not specified
    set_random_seeds(seed)
    print(f"Random seed: {seed}")
    
    # Modify config for overfit test
    if args.overfit:
        print(f"=== OVERFIT TEST MODE ({args.overfit_samples} samples) ===")
        config['training']['num_epochs'] = 100  # Many epochs
        config['training']['early_stopping']['patience'] = 20  # Allow for some patience
        config['logging']['log_frequency'] = 50
        config['training']['batch_size'] = args.overfit_samples  # Set batch size = num samples
        print(f"Modified config: {config['training']['num_epochs']} epochs, patience={config['training']['early_stopping']['patience']}, log_frequency={config['logging']['log_frequency']}, batch_size={config['training']['batch_size']}")
    
    # Create model
    model = QVAEModule(config, overfit_mode=args.overfit, overfit_samples=args.overfit_samples, 
                      reset_epoch_on_load=args.reset_epoch)
    
    # Setup logger first so we can pass it to callbacks
    logger = setup_logger(config, args)
    
    # Setup callbacks with logger for proper run naming
    callbacks = setup_callbacks(config, logger)
    
    # Setup trainer
    trainer = setup_trainer(config, callbacks, logger, args)
    
    # Log model info
    log_model_info(model, config)
    
    # Resume from checkpoint if specified
    if args.resume and args.reset_epoch:
        # Load only model weights, ignore training state
        print("🔄 Loading model weights only (resetting training state)")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✅ Loaded model weights from {args.resume}")
        ckpt_path = None  # Don't pass checkpoint to trainer.fit()
    else:
        ckpt_path = args.resume if args.resume else None
    
    # Run training
    trainer.fit(model, ckpt_path=ckpt_path)
    
    print("Training completed!")
    return trainer, model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train QVAE model")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--reset-epoch", action="store_true", help="Reset epoch counter to 0 when resuming")
    parser.add_argument("--wandb-project", default=None, help="Override WandB project name")
    parser.add_argument("--debug", action="store_true", help="Debug mode (fast_dev_run)")
    parser.add_argument("--overfit", action="store_true", help="Overfit test on tiny dataset")
    parser.add_argument("--overfit-samples", type=int, default=5, help="Number of samples for overfitting test (default: 5)")
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Initialise WandB (will be handled by Lightning)
    print(f"Starting QVAE training with config: {args.config}")
    print(f"Debug mode: {args.debug}")
    
    # Run training
    trainer, model = train_qvae(config, args)
    
    # Training complete
    print("Training session finished.")
    wandb.finish()


if __name__ == "__main__":
    main()