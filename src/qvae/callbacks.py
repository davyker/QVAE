"""
Custom callbacks for QVAE training

Handles audio generation and diagnostic logging during validation.
"""

import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pathlib import Path
import wandb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle


class AudioGenerationCallback(Callback):
    """Callback to generate and log audio samples during validation."""
    
    def __init__(self, dirpath="generated_audio", num_samples=3, log_to_wandb=True, generation_frequency=1):
        """
        Args:
            dirpath: Directory path for saving generated audio (like ModelCheckpoint)
            num_samples: Number of samples to generate per validation set
            log_to_wandb: Whether to log audio to WandB
            generation_frequency: Generate audio every N epochs
        """
        self.dirpath = Path(dirpath)
        self.num_samples = num_samples
        self.log_to_wandb = log_to_wandb
        self.generation_frequency = generation_frequency
        self.sample_indices = {}  # Store fixed indices for consistent samples
        
    def on_validation_epoch_start(self, trainer, pl_module):
        """Initialise sample indices if not already set."""
        if not self.sample_indices:
            # Get validation dataloader sizes
            val_dataloaders = trainer.val_dataloaders
            if not isinstance(val_dataloaders, list):
                val_dataloaders = [val_dataloaders]
                
            for idx, dataloader in enumerate(val_dataloaders):
                dataset_size = len(dataloader.dataset)
                # Select evenly spaced indices
                indices = np.linspace(0, dataset_size - 1, self.num_samples, dtype=int)
                dataset_name = 'interpolation' if idx == 0 else 'extrapolation'
                self.sample_indices[dataset_name] = indices.tolist()
                
        # Reset generation tracking for this epoch
        self.generated_samples = {'interpolation': [], 'extrapolation': []}
        self.sample_metadata = {'interpolation': [], 'extrapolation': []}
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Collect samples during validation."""
        # Skip if not the right epoch for generation
        if trainer.current_epoch % self.generation_frequency != 0:
            return
            
        dataset_name = 'interpolation' if dataloader_idx == 0 else 'extrapolation'
        
        # Get input data based on training mode
        if pl_module.fine_tuning_enabled or pl_module.from_scratch:
            input_data = batch['input_mel_spectrograms']
        else:
            input_data = batch['encoder_features']
        
        # Calculate global indices for this batch
        batch_size = input_data.shape[0]
        global_start_idx = batch_idx * trainer.val_dataloaders[dataloader_idx].batch_size
        
        # Check if any of our target samples are in this batch
        for local_idx in range(batch_size):
            global_idx = global_start_idx + local_idx
            
            if global_idx in self.sample_indices[dataset_name] and len(self.generated_samples[dataset_name]) < self.num_samples:
                # Generate audio for this sample
                sample_input = input_data[local_idx:local_idx+1]
                
                with torch.no_grad():
                    generated_audio, mu, logvar = pl_module.forward(sample_input)
                    
                # Store the generated audio (convert to CPU)
                self.generated_samples[dataset_name].append(generated_audio.cpu())
                
                # Store metadata if available
                metadata = {
                    'global_idx': global_idx,
                    'batch_idx': batch_idx,
                    'local_idx': local_idx
                }
                if 'vocal_file' in batch:
                    metadata['vocal_file'] = batch['vocal_file'][local_idx]
                if 'ref_file' in batch:
                    metadata['ref_file'] = batch['ref_file'][local_idx]
                    
                self.sample_metadata[dataset_name].append(metadata)
                
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save and log generated audio at the end of validation."""
        if trainer.sanity_checking:
            return  # Skip during sanity check
            
        # Skip if not the right epoch for generation
        if trainer.current_epoch % self.generation_frequency != 0:
            return
            
        # Create epoch-specific directory using dirpath
        epoch = trainer.current_epoch
        epoch_dir = self.dirpath / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each validation set
        for dataset_name in ['interpolation', 'extrapolation']:
            dataset_dir = epoch_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            samples = self.generated_samples[dataset_name]
            metadata_list = self.sample_metadata[dataset_name]
            
            for idx, (audio, metadata) in enumerate(zip(samples, metadata_list)):
                # Prepare filename
                if 'vocal_file' in metadata:
                    # Extract meaningful name from vocal file path
                    vocal_path = Path(metadata['vocal_file'])
                    filename = f"sample_{idx:02d}_{vocal_path.stem}.wav"
                else:
                    filename = f"sample_{idx:02d}_idx{metadata['global_idx']}.wav"
                    
                filepath = dataset_dir / filename
                
                # Save audio (ensure it's 2D with shape [channels, samples])
                audio_to_save = audio.squeeze(0)  # Remove batch dimension
                if audio_to_save.dim() == 1:
                    audio_to_save = audio_to_save.unsqueeze(0)  # Add channel dimension if needed
                    
                torchaudio.save(
                    filepath,
                    audio_to_save,
                    sample_rate=22050  # Decoder output sample rate
                )
                
                # Log to WandB if enabled
                if self.log_to_wandb and trainer.logger:
                    # Create WandB audio object
                    # Ensure audio is 1D for wandb (it expects [samples] not [1, samples])
                    audio_numpy = audio_to_save.squeeze().numpy()
                    wandb_audio = wandb.Audio(
                        audio_numpy,
                        sample_rate=22050,
                        caption=f"{dataset_name} - {filename}"
                    )
                    
                    # Log with descriptive key
                    log_key = f"audio/{dataset_name}/sample_{idx:02d}"
                    trainer.logger.experiment.log({
                        log_key: wandb_audio,
                        "epoch": epoch
                    })
                    
        # Log summary of saved files
        print(f"\n=== Generated Audio Saved ===")
        print(f"Epoch {epoch} audio saved to: {epoch_dir}")
        print(f"Interpolation samples: {len(self.generated_samples['interpolation'])}")
        print(f"Extrapolation samples: {len(self.generated_samples['extrapolation'])}")
        print("=" * 30 + "\n")
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Generate and save audio samples from training set."""
        # Skip if not the right epoch for generation
        if trainer.current_epoch % self.generation_frequency != 0:
            return
            
        # Get a batch from training dataloader
        train_dataloader = trainer.train_dataloader
        batch = next(iter(train_dataloader))
        
        # Move batch to same device as model
        device = next(pl_module.parameters()).device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        
        # Get input data based on training mode
        if pl_module.fine_tuning_enabled or pl_module.from_scratch:
            input_data = batch['input_mel_spectrograms']
        else:
            input_data = batch['encoder_features']
            
        # Generate samples (up to num_samples or batch size)
        num_to_generate = min(self.num_samples, input_data.shape[0])
        
        # Create epoch-specific directory for training samples
        epoch = trainer.current_epoch
        epoch_dir = self.dirpath / f"epoch_{epoch:04d}" / "train"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save samples
        for idx in range(num_to_generate):
            sample_input = input_data[idx:idx+1]
            
            with torch.no_grad():
                generated_audio, mu, logvar = pl_module.forward(sample_input)
                
            # Convert to CPU and numpy
            audio_to_save = generated_audio.cpu()
            
            # Generate filename with sound ID if available (same format as validation samples)
            if 'vocal_file' in batch and idx < len(batch['vocal_file']):
                # Extract meaningful name from vocal file path (same logic as validation)
                vocal_path = Path(batch['vocal_file'][idx])
                filename = f"sample_{idx:02d}_{vocal_path.stem}.wav"
            else:
                filename = f"sample_{idx:02d}.wav"
            filepath = epoch_dir / filename
            
            # Save audio file
            torchaudio.save(
                filepath,
                audio_to_save,
                sample_rate=22050
            )
            
            # Log to WandB if enabled
            if self.log_to_wandb and trainer.logger:
                audio_numpy = audio_to_save.squeeze().numpy()
                wandb_audio = wandb.Audio(
                    audio_numpy,
                    sample_rate=22050,
                    caption=f"train - {filename}"
                )
                
                log_key = f"audio/train/sample_{idx:02d}"
                trainer.logger.experiment.log({
                    log_key: wandb_audio,
                    "epoch": epoch
                })
        
        print(f"Training samples: {num_to_generate} saved to {epoch_dir}")


class DiagnosticCallback(Callback):
    """Callback to track VAE internal states and save diagnostic information."""
    
    def __init__(self, dirpath, config, diagnostic_frequency=5):
        """
        Args:
            dirpath: Directory path for saving diagnostics (like ModelCheckpoint)
            config: Full training configuration
            diagnostic_frequency: Run diagnostics every N epochs
        """
        self.dirpath = Path(dirpath)
        self.config = config
        self.diagnostic_frequency = diagnostic_frequency
        self.is_overfit_mode = False
        self.sample_tracking_limit = 10
        self.tracked_samples = {}  # Store sample identities and indices
        self.multi_sample_generations = 5  # Number of generations for stochasticity analysis
        
        # Storage for batch data
        self.batch_data = {
            'mu': [],
            'logvar': [],
            'kl_losses': [],
            'sample_names': [],
            'gradients': {}
        }
        
    def on_train_start(self, trainer, pl_module):
        """Detect mode and setup tracking parameters"""
        self.is_overfit_mode = pl_module.overfit_mode
        
        if self.is_overfit_mode:
            self.sample_tracking_limit = pl_module.overfit_samples
            self.diagnostic_frequency = 1  # Every epoch for overfitting
            print(f"DiagnosticCallback: Overfit mode - tracking {self.sample_tracking_limit} samples every epoch")
        else:
            print(f"DiagnosticCallback: Full training mode - tracking {self.sample_tracking_limit} samples every {self.diagnostic_frequency} epochs")
            
        # Create base directory
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track diagnostics based on training mode"""
        # Only collect data on diagnostic epochs
        should_collect = (trainer.current_epoch % self.diagnostic_frequency == 0)
            
        if not should_collect:
            return
            
        # Capture gradient norms before they're cleared (must be done on first batch)
        if batch_idx == 0 and not hasattr(self, 'current_epoch_gradients'):
            self.current_epoch_gradients = {}
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    self.current_epoch_gradients[name] = param.grad.data.norm().item()
            
        if self.is_overfit_mode:
            # In overfit mode: track first batch (contains all unique samples)
            if batch_idx == 0:
                print(f"  - Collecting overfit mode data (batch {batch_idx})")
                self._track_overfit_batch(batch, pl_module, trainer)
        else:
            # In full training: track first few batches consistently
            if batch_idx < 3:
                print(f"  - Collecting full training data (batch {batch_idx})")
                self._track_training_batch(batch, pl_module, trainer, batch_idx)
    
    def _track_overfit_batch(self, batch, pl_module, trainer):
        """Track all samples in overfitting mode"""
        # Get input data based on training mode
        if pl_module.fine_tuning_enabled or pl_module.from_scratch:
            input_data = batch['input_mel_spectrograms']
        else:
            input_data = batch['encoder_features']
            
        with torch.no_grad():
            # Forward pass to get latent variables
            generated_audio, mu, logvar = pl_module.forward(input_data)
            
            # Compute KL loss per sample
            kl_per_sample = self._compute_kl_per_sample(mu, logvar)
            
            # Store data
            self.batch_data['mu'].append(mu.cpu())
            self.batch_data['logvar'].append(logvar.cpu())
            self.batch_data['kl_losses'].append(kl_per_sample.cpu())
            
            # Get sample names from batch if available
            if 'vocal_file' in batch:
                sample_names = [Path(f).stem for f in batch['vocal_file']]
                self.batch_data['sample_names'].extend(sample_names)
            else:
                sample_names = [f"sample_{i}" for i in range(input_data.shape[0])]
                self.batch_data['sample_names'].extend(sample_names)
                
            # Generate multiple samples for stochasticity analysis
            multi_generations = []
            for _ in range(self.multi_sample_generations):
                gen_audio, _, _ = pl_module.forward(input_data)
                multi_generations.append(gen_audio.cpu())
            
            # Store for later analysis
            if not hasattr(self, 'multi_generations'):
                self.multi_generations = []
            self.multi_generations.append(torch.stack(multi_generations))
    
    def _track_training_batch(self, batch, pl_module, trainer, batch_idx):
        """Track subset of samples in full training mode"""
        # First time: establish which samples we'll track
        if not self.tracked_samples and 'vocal_file' in batch:
            for i, vocal_file in enumerate(batch['vocal_file'][:self.sample_tracking_limit]):
                sample_id = Path(vocal_file).stem
                self.tracked_samples[sample_id] = len(self.tracked_samples)
                
        # Get input data
        if pl_module.fine_tuning_enabled or pl_module.from_scratch:
            input_data = batch['input_mel_spectrograms']
        else:
            input_data = batch['encoder_features']
            
        # Track only our established subset
        if 'vocal_file' in batch:
            indices_to_track = []
            names_to_track = []
            
            for i, vocal_file in enumerate(batch['vocal_file']):
                sample_id = Path(vocal_file).stem
                if sample_id in self.tracked_samples:
                    indices_to_track.append(i)
                    names_to_track.append(sample_id)
            
            if indices_to_track:
                # Extract subset
                subset_input = input_data[indices_to_track]
                
                with torch.no_grad():
                    generated_audio, mu, logvar = pl_module.forward(subset_input)
                    kl_per_sample = self._compute_kl_per_sample(mu, logvar)
                    
                    # Store data
                    self.batch_data['mu'].append(mu.cpu())
                    self.batch_data['logvar'].append(logvar.cpu())
                    self.batch_data['kl_losses'].append(kl_per_sample.cpu())
                    self.batch_data['sample_names'].extend(names_to_track)
    
    def _compute_kl_per_sample(self, mu, logvar):
        """Compute KL divergence per sample"""
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_per_sample
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save diagnostics and create visualisations"""
        print(f"[DEBUG] on_validation_epoch_end - Epoch {trainer.current_epoch}:")
        
        # Only run on diagnostic epochs
        should_save = (trainer.current_epoch % self.diagnostic_frequency == 0)
        print(f"  - Should save diagnostics: {should_save} (epoch % freq = {trainer.current_epoch % self.diagnostic_frequency})")
        
        if not should_save:
            print(f"  - Skipping diagnostic save")
            return
            
        print(f"  - Checking batch data: mu entries = {len(self.batch_data['mu'])}")
        
        if not self.batch_data['mu']:  # No data collected
            print(f"  - WARNING: No data collected to save!")
            return
            
        epoch = trainer.current_epoch
        epoch_dir = self.dirpath / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Concatenate all collected data
        all_mu = torch.cat(self.batch_data['mu'], dim=0)
        all_logvar = torch.cat(self.batch_data['logvar'], dim=0)
        all_kl = torch.cat(self.batch_data['kl_losses'], dim=0)
        
        # Save latent statistics
        self._save_latent_stats(epoch_dir, all_mu, all_logvar, all_kl)
        
        # Save stochasticity analysis if available
        if hasattr(self, 'multi_generations'):
            self._save_stochasticity_analysis(epoch_dir)
            
        # Create latent space visualisation
        self._create_latent_visualisation(epoch_dir, all_mu)
        
        # Track gradient norms
        self._save_gradient_analysis(epoch_dir, pl_module)
        
        # Terminal output
        self._print_diagnostic_summary(epoch, all_mu, all_logvar, all_kl)
        
        # WandB logging
        self._log_to_wandb(trainer, all_mu, all_logvar, all_kl)
        
        # Clear batch data for next epoch
        print(f"  - Successfully saved diagnostics to {epoch_dir}")
        print(f"  - Clearing batch data for next epoch")
        self._clear_batch_data()
    
    def _save_latent_stats(self, epoch_dir, mu, logvar, kl_losses):
        """Save latent space statistics"""
        # Compute statistics
        latent_norms = torch.norm(mu, dim=1)
        latent_variances = torch.exp(logvar).mean(dim=1)
        
        # Save to file
        np.savez(
            epoch_dir / "latent_stats.npz",
            mu_per_sample=mu.numpy(),
            logvar_per_sample=logvar.numpy(),
            sample_names=self.batch_data['sample_names'],
            latent_norms=latent_norms.numpy(),
            latent_variances=latent_variances.numpy(),
            kl_per_sample=kl_losses.numpy()
        )
    
    def _save_stochasticity_analysis(self, epoch_dir):
        """Save multiple generation analysis"""
        if not hasattr(self, 'multi_generations'):
            return
            
        # Stack all generations: [num_samples, num_generations, ...]
        all_generations = torch.cat(self.multi_generations, dim=0)
        
        # Compute variance across generations
        generation_variance = torch.var(all_generations, dim=1)  # Variance across generations
        generation_std = torch.std(all_generations, dim=1)
        
        np.savez(
            epoch_dir / "stochasticity.npz",
            multi_generations=all_generations.numpy(),
            generation_variance=generation_variance.numpy(),
            generation_std=generation_std.numpy(),
            num_generations=self.multi_sample_generations
        )
    
    def _create_latent_visualisation(self, epoch_dir, mu):
        """Create 2D visualisation of latent space"""
        if mu.shape[0] < 2:
            return  # Should have at least 2 samples for visualisation
            
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(mu.numpy())
        
        # Create plot
        plt.figure(figsize=(8, 6))
        
        # Color by sample type if we have names
        if self.batch_data['sample_names']:
            unique_names = list(set(self.batch_data['sample_names']))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_names)))
            
            for i, name in enumerate(unique_names):
                indices = [j for j, s in enumerate(self.batch_data['sample_names']) if s == name]
                if indices:
                    plt.scatter(mu_2d[indices, 0], mu_2d[indices, 1], 
                              c=[colors[i]], label=name, alpha=0.7, s=50)
            
            plt.legend()
        else:
            plt.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f} variance)')
        plt.title(f'Latent Space Visualisation (Epoch {epoch_dir.name})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(epoch_dir / "latent_viz.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_gradient_analysis(self, epoch_dir, pl_module):
        """Save gradient norm analysis"""
        # Use the gradients captured during training
        grad_norms = getattr(self, 'current_epoch_gradients', {})
        
        if grad_norms:
            # Save gradient information
            np.savez(
                epoch_dir / "gradients.npz",
                grad_norms=grad_norms
            )
        
        # Clear for next epoch
        if hasattr(self, 'current_epoch_gradients'):
            delattr(self, 'current_epoch_gradients')
    
    def _print_diagnostic_summary(self, epoch, mu, logvar, kl_losses):
        """Print concise diagnostic summary to terminal"""
        # Compute separation metric (if multiple samples)
        separation = 0.0
        if mu.shape[0] > 1:
            # Average pairwise distance
            mu_np = mu.numpy()
            distances = []
            for i in range(len(mu_np)):
                for j in range(i+1, len(mu_np)):
                    dist = np.linalg.norm(mu_np[i] - mu_np[j])
                    distances.append(dist)
            separation = np.mean(distances)
        
        # Per-sample KL breakdown
        kl_summary = ""
        if self.batch_data['sample_names'] and len(set(self.batch_data['sample_names'])) > 1:
            unique_names = list(set(self.batch_data['sample_names']))
            kl_by_name = {}
            for name in unique_names:
                indices = [i for i, s in enumerate(self.batch_data['sample_names']) if s == name]
                if indices:
                    avg_kl = kl_losses[indices].mean().item()
                    kl_by_name[name] = avg_kl
            
            kl_parts = [f"{name}={kl:.2f}" for name, kl in kl_by_name.items()]
            kl_summary = ", KL: " + ", ".join(kl_parts)
        
        print(f"Epoch {epoch:04d}: Latent sep={separation:.2f}{kl_summary}")
    
    def _log_to_wandb(self, trainer, mu, logvar, kl_losses):
        """Log metrics to WandB"""
        if not trainer.logger:
            return
            
        # Compute metrics
        avg_kl = kl_losses.mean().item()
        avg_latent_norm = torch.norm(mu, dim=1).mean().item()
        avg_variance = torch.exp(logvar).mean().item()
        
        # Latent separation
        separation = 0.0
        if mu.shape[0] > 1:
            mu_np = mu.numpy()
            distances = []
            for i in range(len(mu_np)):
                for j in range(i+1, len(mu_np)):
                    dist = np.linalg.norm(mu_np[i] - mu_np[j])
                    distances.append(dist)
            separation = np.mean(distances)
        
        # Log to WandB
        metrics = {
            "diagnostics/latent_separation": separation,
            "diagnostics/avg_kl": avg_kl,
            "diagnostics/avg_latent_norm": avg_latent_norm,
            "diagnostics/avg_variance": avg_variance
        }
        
        trainer.logger.experiment.log(metrics)
    
    def _clear_batch_data(self):
        """Clear accumulated batch data"""
        self.batch_data = {
            'mu': [],
            'logvar': [],
            'kl_losses': [],
            'sample_names': [],
            'gradients': {}
        }
        if hasattr(self, 'multi_generations'):
            del self.multi_generations