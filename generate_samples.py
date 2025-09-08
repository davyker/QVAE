"""
Generate audio samples from trained QVAE model

Loads a checkpoint and generates samples from validation sets.
"""

import torch
import torchaudio
import pytorch_lightning as pl
from pathlib import Path
import argparse
import yaml
import numpy as np

# Import QVAE components
from train import QVAEModule
from src.qvae.utils import load_config


def generate_samples_from_splits(checkpoint_path, config, output_dir, samples_per_split=5):
    """Generate samples from train/validation splits.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dict
        output_dir: Output directory for generated samples
        samples_per_split: Number of samples to generate per split
    """
    import pickle
    from pathlib import Path
    
    # Load splits
    splits_file = Path(config['data']['cache_dir']) / "splits.pkl"
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    
    print(f"\nLoaded splits from: {splits_file}")
    print(f"  Train sounds: {len(splits['train_sounds'])}")
    print(f"  Extrapolation sounds: {len(splits['extrapolation_sounds'])}")
    
    # Create output directory structure
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    interp_dir = output_dir / "interpolation"
    extrap_dir = output_dir / "extrapolation"
    
    for dir_path in [train_dir, interp_dir, extrap_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Create model in inference mode
    model = QVAEModule(config=config, mode='inference')
    
    # Load checkpoint manually to handle state dict mismatches
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Detect latent_dim from checkpoint
    if 'encoder.vae_head.weight' in state_dict:
        vae_head_shape = state_dict['encoder.vae_head.weight'].shape
        checkpoint_latent_dim = vae_head_shape[0] // 2
    elif 'encoder.mu_head.weight' in state_dict:
        checkpoint_latent_dim = state_dict['encoder.mu_head.weight'].shape[0]
    else:
        checkpoint_latent_dim = config['model']['latent_dim']
    
    config['model']['latent_dim'] = checkpoint_latent_dim
    print(f"Detected latent_dim={checkpoint_latent_dim} from checkpoint")
    
    # Recreate model with correct latent_dim
    model = QVAEModule(config=config, mode='inference')
    
    # Fix state dict key mismatches
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('loss_fn.'):
            continue
        if key == 'encoder.vae_head.weight':
            new_state_dict['encoder.mu_head.weight'] = value[:checkpoint_latent_dim, :]
            new_state_dict['encoder.logvar_head.weight'] = value[checkpoint_latent_dim:2*checkpoint_latent_dim, :]
        elif key == 'encoder.vae_head.bias':
            new_state_dict['encoder.mu_head.bias'] = value[:checkpoint_latent_dim]
            new_state_dict['encoder.logvar_head.bias'] = value[checkpoint_latent_dim:2*checkpoint_latent_dim]
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Loaded checkpoint with compatibility fixes")
    
    model.eval()
    model.cuda() if torch.cuda.is_available() else model.cpu()
    
    # Get dataset paths
    dataset_path = Path(config['data']['dataset_path']) / "Vim_Sketch_Dataset"
    vocal_dir = dataset_path / "vocal_imitations"
    ref_dir = dataset_path / "references"
    
    # Generate samples for each split
    splits_config = [
        ("train", train_dir, splits['train_sounds'][:samples_per_split], splits['train_imitations']),
        ("interpolation", interp_dir, splits['train_sounds'][:samples_per_split], splits['interpolation_imitations']),
        ("extrapolation", extrap_dir, splits['extrapolation_sounds'][:samples_per_split], None)
    ]
    
    with torch.no_grad():
        for split_name, output_split_dir, sound_ids, imitations_dict in splits_config:
            print(f"\n{'='*60}")
            print(f"Generating {split_name} samples")
            print('='*60)
            
            samples_generated = 0
            
            for sound_id in sound_ids:
                # Get vocal file for this sound
                if imitations_dict and sound_id in imitations_dict:
                    # Use specific imitation from split
                    vocal_files = imitations_dict[sound_id]
                    if not vocal_files:
                        continue
                    vocal_filename = vocal_files[0]  # Take first imitation
                else:
                    # For extrapolation, find any imitation for this sound
                    vocal_files = list(vocal_dir.glob(f"{sound_id}_*.wav"))
                    if not vocal_files:
                        continue
                    vocal_filename = vocal_files[0].name
                
                # Full paths
                vocal_path = vocal_dir / vocal_filename
                ref_path = ref_dir / f"{sound_id}.wav"
                
                if not vocal_path.exists():
                    print(f"  Skipping - vocal file not found: {vocal_filename}")
                    continue
                
                # Generate audio
                print(f"\n  Processing: {vocal_filename}")
                generated_audio, mu, logvar = model.inference_forward(vocal_path)
                
                # Save generated audio
                gen_audio = generated_audio[0].cpu()
                if gen_audio.dim() == 1:
                    gen_audio = gen_audio.unsqueeze(0)
                
                gen_filename = f"{samples_generated:02d}_{sound_id}_generated.wav"
                gen_path = output_split_dir / gen_filename
                
                torchaudio.save(gen_path, gen_audio, sample_rate=22050)
                
                # Copy input
                try:
                    input_audio, input_sr = torchaudio.load(vocal_path)
                    input_filename = f"{samples_generated:02d}_{sound_id}_input.wav"
                    input_path = output_split_dir / input_filename
                    torchaudio.save(input_path, input_audio, sample_rate=input_sr)
                    print(f"    Saved: {gen_filename}")
                    print(f"    Input: {input_filename}")
                except Exception as e:
                    print(f"    Saved: {gen_filename}")
                    print(f"    (Could not copy input: {e})")
                
                # Copy reference
                if ref_path.exists():
                    try:
                        ref_audio, ref_sr = torchaudio.load(ref_path)
                        ref_filename = f"{samples_generated:02d}_{sound_id}_reference.wav"
                        ref_path_out = output_split_dir / ref_filename
                        torchaudio.save(ref_path_out, ref_audio, sample_rate=ref_sr)
                        print(f"    Reference: {ref_filename}")
                    except Exception as e:
                        print(f"    (Could not copy reference: {e})")
                
                samples_generated += 1
            
            print(f"\nGenerated {samples_generated} samples for {split_name}")
    
    print(f"\n{'='*60}")
    print(f"All samples saved to: {output_dir}")
    print('='*60)


def generate_samples(checkpoint_path, config, output_dir, input_files=None, num_samples=5):
    """Generate audio samples using trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dict
        output_dir: Output directory for generated samples
        input_files: List of filenames from vocal_imitations directory, or None for random sampling
        num_samples: Number of samples to generate if input_files is None
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model from checkpoint in inference mode
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Load checkpoint to detect latent_dim
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Detect latent_dim from checkpoint
    if 'encoder.vae_head.weight' in state_dict:
        # Old model structure with single vae_head
        vae_head_shape = state_dict['encoder.vae_head.weight'].shape
        checkpoint_latent_dim = vae_head_shape[0] // 2  # vae_head outputs 2*latent_dim
    elif 'encoder.mu_head.weight' in state_dict:
        # New model structure with separate mu/logvar heads
        checkpoint_latent_dim = state_dict['encoder.mu_head.weight'].shape[0]
    else:
        # Fallback to config
        checkpoint_latent_dim = config['model']['latent_dim']
    
    # Override config latent_dim with checkpoint's value
    config['model']['latent_dim'] = checkpoint_latent_dim
    print(f"Detected latent_dim={checkpoint_latent_dim} from checkpoint")
    
    # Create model in inference mode with correct latent_dim
    model = QVAEModule(config=config, mode='inference')
    
    # Fix state dict key mismatches
    new_state_dict = {}
    for key, value in state_dict.items():
        # Skip loss function components in inference mode
        if key.startswith('loss_fn.'):
            continue
            
        # Handle encoder.vae_head -> encoder.mu_head/logvar_head mapping
        if key == 'encoder.vae_head.weight':
            # Old model had single vae_head, split it for mu and logvar
            new_state_dict['encoder.mu_head.weight'] = value[:checkpoint_latent_dim, :]
            new_state_dict['encoder.logvar_head.weight'] = value[checkpoint_latent_dim:2*checkpoint_latent_dim, :]
        elif key == 'encoder.vae_head.bias':
            new_state_dict['encoder.mu_head.bias'] = value[:checkpoint_latent_dim]
            new_state_dict['encoder.logvar_head.bias'] = value[checkpoint_latent_dim:2*checkpoint_latent_dim]
        else:
            new_state_dict[key] = value
    
    # Load the fixed state dict
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Loaded checkpoint with compatibility fixes")
    
    model.eval()
    model.cuda() if torch.cuda.is_available() else model.cpu()
    
    # Get dataset path
    dataset_path = Path(config['data']['dataset_path']) / "Vim_Sketch_Dataset"
    vocal_dir = dataset_path / "vocal_imitations"
    ref_dir = dataset_path / "references"
    
    # Determine which files to process
    if input_files is None:
        # Random sampling from all available files
        print(f"Randomly sampling {num_samples} files from vocal_imitations...")
        all_vocal_files = list(vocal_dir.glob("*.wav"))
        import random
        random.shuffle(all_vocal_files)
        vocal_files = all_vocal_files[:num_samples]
        vocal_files = [f.name for f in vocal_files]
    else:
        # Use provided list
        vocal_files = input_files
        print(f"Processing {len(vocal_files)} specified files...")
    
    print(f"\n{'='*60}")
    print(f"Generating {len(vocal_files)} samples")
    print('='*60)
    
    # Generate samples
    samples_generated = 0
    with torch.no_grad():
        for vocal_filename in vocal_files:
            # Full path to vocal file
            vocal_path = vocal_dir / vocal_filename
            
            if not vocal_path.exists():
                print(f"  Skipping - file not found: {vocal_filename}")
                continue
            
            # Extract sound ID to find reference
            from src.qvae.utils import extract_sound_id
            sound_id = extract_sound_id(vocal_filename)
            ref_filename = f"{sound_id}.wav"
            ref_path = ref_dir / ref_filename
            
            # Generate audio using inference_forward
            print(f"\n  Processing: {vocal_filename}")
            generated_audio, mu, logvar = model.inference_forward(vocal_path)
            
            # Extract meaningful name
            vocal_name = Path(vocal_filename).stem
            
            # Save generated audio
            gen_audio = generated_audio[0].cpu()  # Take first sample from batch
            if gen_audio.dim() == 1:
                gen_audio = gen_audio.unsqueeze(0)  # Add channel dimension
            
            gen_filename = f"{samples_generated:02d}_{vocal_name}_generated.wav"
            gen_path = output_dir / gen_filename
            
            torchaudio.save(
                gen_path,
                gen_audio,
                sample_rate=22050
            )
            
            # Copy original vocal imitation
            try:
                input_audio, input_sr = torchaudio.load(vocal_path)
                input_filename = f"{samples_generated:02d}_{vocal_name}_input.wav"
                input_path = output_dir / input_filename
                
                torchaudio.save(
                    input_path,
                    input_audio,
                    sample_rate=input_sr
                )
                
                print(f"    Saved: {gen_filename}")
                print(f"    Input: {input_filename}")
            except Exception as e:
                print(f"    Saved: {gen_filename}")
                print(f"    (Could not copy input: {e})")
            
            # Copy reference audio if available
            if ref_path.exists():
                try:
                    ref_audio, ref_sr = torchaudio.load(ref_path)
                    ref_filename = f"{samples_generated:02d}_{vocal_name}_reference.wav"
                    ref_path_out = output_dir / ref_filename
                    
                    torchaudio.save(
                        ref_path_out,
                        ref_audio,
                        sample_rate=ref_sr
                    )
                    print(f"    Reference: {ref_filename}")
                except Exception as e:
                    print(f"    (Could not copy reference: {e})")
            
            samples_generated += 1
    
    print(f"\n{'='*60}")
    print(f"Generated {samples_generated} samples")
    print(f"All samples saved to: {output_dir}")
    print('='*60)


def find_best_checkpoint(run_dir):
    """Find the best checkpoint in the checkpoints subdirectory."""
    checkpoint_dir = Path(run_dir) / "checkpoints"
    
    # First, try to find best.ckpt (most reliable)
    best_ckpt = checkpoint_dir / "best.ckpt"
    if best_ckpt.exists():
        print("Found best.ckpt")
        return best_ckpt
    
    # Try last.ckpt as second option
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        print("Found last.ckpt")
        return last_ckpt
    
    # Look for checkpoint files with numbers at the end
    checkpoints = []
    for path in checkpoint_dir.glob("*.ckpt"):
        # Skip best.ckpt and last.ckpt if they exist
        if path.name in ["best.ckpt", "last.ckpt"]:
            continue
        
        # Extract trailing numbers from filename
        import re
        match = re.search(r'(\d+)\.ckpt$', path.name)
        if match:
            number = int(match.group(1))
            checkpoints.append((number, path))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by number (descending) and take the largest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    largest_num, best_path = checkpoints[0]
    
    print(f"Found checkpoint with largest number: {best_path.name}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Generate audio samples from QVAE checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint file")
    parser.add_argument("--run-dir", default="runs", help="Directory containing runs with checkpoints")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output-dir", default=None, help="Output directory for samples")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate (if --input-files not specified)")
    parser.add_argument("--input-files", nargs="+", default=None, 
                        help="List of filenames from vocal_imitations directory to process")
    parser.add_argument("--use-splits", action="store_true",
                        help="Generate samples using train/val splits from splits.pkl")
    parser.add_argument("--samples-per-split", type=int, default=5,
                        help="Number of samples per split when using --use-splits")
    
    args = parser.parse_args()
    
    # Determine run_dir and checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        # Try to infer run_dir from checkpoint path
        if checkpoint_path.parent.name == "checkpoints":
            run_dir = checkpoint_path.parent.parent
        else:
            # Otherwise use the provided run_dir argument
            run_dir = Path(args.run_dir)
    else:
        # Use run_dir to find best checkpoint
        run_dir = Path(args.run_dir)
        print(f"Searching for best checkpoint in: {run_dir / 'checkpoints'}")
        checkpoint_path = find_best_checkpoint(run_dir)
    
    # Load config - prefer config from run_dir if it exists and no explicit config specified
    run_config_path = run_dir / "config.yaml"
    if run_config_path.exists() and args.config == "config.yaml":
        # Use config from run directory (unless user explicitly specified a different one)
        print(f"Loading config from run directory: {run_config_path}")
        config = load_config(str(run_config_path))
    else:
        # Use specified config or default
        print(f"Loading config: {args.config}")
        config = load_config(args.config)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.use_splits:
            output_dir = run_dir / "generated_samples_by_splits"
        else:
            output_dir = run_dir / "generated_samples"
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Generate samples
    if args.use_splits:
        generate_samples_from_splits(
            checkpoint_path=checkpoint_path,
            config=config,
            output_dir=output_dir,
            samples_per_split=args.samples_per_split
        )
    else:
        generate_samples(
            checkpoint_path=checkpoint_path,
            config=config,
            output_dir=output_dir,
            input_files=args.input_files,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()