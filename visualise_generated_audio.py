"""
Visualise and compare generated vs target audio spectrograms.

This script loads generated audio samples and their corresponding targets,
then creates side-by-side spectrogram comparisons to diagnose quality issues.
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from pathlib import Path
import argparse
import yaml
from glob import glob
import sys
sys.path.append('.')
from src.qvae.utils import load_config

def plot_audio_comparison(generated_path, reference_path, output_path, config=None, sample_dir=None, fixed_comparison=False):
    """Create detailed spectrogram comparison between generated and reference audio."""
    
    # Load config if not provided
    if config is None:
        config = load_config('config.yaml')
    
    # Load audio files
    gen_audio, gen_sr = torchaudio.load(generated_path)
    ref_audio, ref_sr = torchaudio.load(reference_path)
    
    # Convert to numpy and ensure mono
    gen_audio = gen_audio.squeeze().numpy()
    ref_audio = ref_audio.squeeze().numpy()
    
    # Ensure same sample rate (resample if needed)
    if gen_sr != ref_sr:
        gen_audio = librosa.resample(gen_audio, orig_sr=gen_sr, target_sr=ref_sr)
        gen_sr = ref_sr
    
    # Fixed comparison mode: standardise to decoder output length
    if fixed_comparison:
        target_samples = 59049  # Decoder output length
        target_duration = target_samples / 22050  # ~2.68 seconds
        
        # Ensure both are at 22050 Hz for fixed comparison
        if gen_sr != 22050:
            gen_audio = librosa.resample(gen_audio, orig_sr=gen_sr, target_sr=22050)
            gen_sr = 22050
        if ref_sr != 22050:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=22050)
            ref_sr = 22050
        
        # Pad or truncate reference audio to match target length
        if len(ref_audio) > target_samples:
            ref_audio = ref_audio[:target_samples]
        elif len(ref_audio) < target_samples:
            ref_audio = np.pad(ref_audio, (0, target_samples - len(ref_audio)), mode='constant')
        
        # Generated audio should already be correct length, but ensure it
        if len(gen_audio) > target_samples:
            gen_audio = gen_audio[:target_samples]
        elif len(gen_audio) < target_samples:
            gen_audio = np.pad(gen_audio, (0, target_samples - len(gen_audio)), mode='constant')
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Create title with sample directory path
    if sample_dir:
        title = f'Audio Comparison: {Path(generated_path).stem}\nFrom: {sample_dir}'
    else:
        title = f'Audio Comparison: {Path(generated_path).stem}'
    fig.suptitle(title, fontsize=14)
    
    # 1. Waveforms
    axes[0, 0].plot(np.linspace(0, len(gen_audio)/gen_sr, len(gen_audio)), gen_audio)
    axes[0, 0].set_title('Generated Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_ylim(-1, 1)
    
    axes[0, 1].plot(np.linspace(0, len(ref_audio)/ref_sr, len(ref_audio)), ref_audio)
    axes[0, 1].set_title('Reference Waveform')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_ylim(-1, 1)
    
    # 2. STFT Spectrograms
    # Use config settings - pick the second STFT window size (1024)
    stft_idx = 1  # Use window size 1024
    n_fft = config['target_spectrograms']['stft']['window_sizes'][stft_idx]
    hop_length = config['target_spectrograms']['stft']['hop_lengths'][stft_idx]
    
    # Compute STFT
    gen_stft = librosa.stft(gen_audio, n_fft=n_fft, hop_length=hop_length)
    ref_stft = librosa.stft(ref_audio, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to dB
    gen_stft_db = librosa.amplitude_to_db(np.abs(gen_stft), ref=np.max)
    ref_stft_db = librosa.amplitude_to_db(np.abs(ref_stft), ref=np.max)
    
    # Find frequency with highest median value in generated STFT
    gen_stft_median = np.median(np.abs(gen_stft), axis=1)  # Median across time
    max_freq_bin = np.argmax(gen_stft_median)
    max_freq_hz = max_freq_bin * gen_sr / n_fft  # Convert bin to Hz
    max_freq_value = gen_stft_median[max_freq_bin]
    
    print(f"\n  STFT Analysis - {Path(generated_path).stem}:")
    print(f"    Peak frequency: {max_freq_hz:.1f} Hz (bin {max_freq_bin})")
    print(f"    Median magnitude at peak: {max_freq_value:.4f}")
    
    # Plot STFT
    librosa.display.specshow(gen_stft_db, sr=gen_sr, hop_length=hop_length, 
                             x_axis='time', y_axis='hz', ax=axes[1, 0])
    axes[1, 0].set_title(f'Generated STFT Spectrogram (n_fft={n_fft})')
    axes[1, 0].set_colorbar = False
    if fixed_comparison:
        axes[1, 0].set_ylim(0, 11025)  # Limit to Nyquist frequency of 22050 Hz
    
    librosa.display.specshow(ref_stft_db, sr=ref_sr, hop_length=hop_length,
                             x_axis='time', y_axis='hz', ax=axes[1, 1])
    axes[1, 1].set_title(f'Reference STFT Spectrogram (n_fft={n_fft})')
    if fixed_comparison:
        axes[1, 1].set_ylim(0, 11025)  # Limit to Nyquist frequency of 22050 Hz
    
    # 3. Mel Spectrograms
    # Use config settings - pick the first mel configuration
    mel_idx = 0
    mel_n_fft = config['target_spectrograms']['mel']['n_fft'][mel_idx]
    mel_hop_length = config['target_spectrograms']['mel']['hop_lengths'][mel_idx]
    n_mels = config['target_spectrograms']['mel']['n_mels']
    
    # Compute mel spectrograms
    gen_mel = librosa.feature.melspectrogram(y=gen_audio, sr=gen_sr, n_mels=n_mels, 
                                              n_fft=mel_n_fft, hop_length=mel_hop_length)
    ref_mel = librosa.feature.melspectrogram(y=ref_audio, sr=ref_sr, n_mels=n_mels,
                                              n_fft=mel_n_fft, hop_length=mel_hop_length)
    
    # Convert to dB
    gen_mel_db = librosa.power_to_db(gen_mel, ref=np.max)
    ref_mel_db = librosa.power_to_db(ref_mel, ref=np.max)
    
    # Plot mel spectrograms
    librosa.display.specshow(gen_mel_db, sr=gen_sr, hop_length=mel_hop_length,
                             x_axis='time', y_axis='mel', ax=axes[2, 0])
    axes[2, 0].set_title(f'Generated Mel Spectrogram (n_fft={mel_n_fft}, n_mels={n_mels})')
    
    librosa.display.specshow(ref_mel_db, sr=ref_sr, hop_length=mel_hop_length,
                             x_axis='time', y_axis='mel', ax=axes[2, 1])
    axes[2, 1].set_title(f'Reference Mel Spectrogram (n_fft={mel_n_fft}, n_mels={n_mels})')
    
    for i in range(1, 3):
        for j in range(2):
            if len(axes[i, j].images) > 0:
                im = axes[i, j].images[0]
                cbar = plt.colorbar(im, ax=axes[i, j])
                cbar.set_label('dB')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print some statistics
    print(f"\nAudio Statistics for {Path(generated_path).stem}:")
    print(f"  Generated - Duration: {len(gen_audio)/gen_sr:.2f}s, RMS: {np.sqrt(np.mean(gen_audio**2)):.4f}")
    print(f"  Reference - Duration: {len(ref_audio)/ref_sr:.2f}s, RMS: {np.sqrt(np.mean(ref_audio**2)):.4f}")
    
    # Compute spectral statistics
    gen_spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=gen_audio, sr=gen_sr))
    ref_spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=ref_audio, sr=ref_sr))
    
    print(f"  Spectral Centroid - Generated: {gen_spectral_centroid:.1f} Hz, Reference: {ref_spectral_centroid:.1f} Hz")
    
    # Check if mostly noise (high spectral flatness)
    gen_flatness = np.mean(librosa.feature.spectral_flatness(y=gen_audio))
    ref_flatness = np.mean(librosa.feature.spectral_flatness(y=ref_audio))
    
    print(f"  Spectral Flatness - Generated: {gen_flatness:.4f}, Reference: {ref_flatness:.4f}")
    print(f"  (Flatness near 1.0 indicates white noise, near 0.0 indicates tonal)")


def find_reference_audio(gen_filename, dataset_path="data/Vim_Sketch_Dataset/references"):
    """Extract sound_id from generated filename and find reference audio."""
    # Generated filename format: sample_{num}_{imitation_file_stem}.wav
    # Example: sample_00_02818_148Sounds of things_Alarm_Air horn_truck horn.wav
    
    stem = gen_filename.stem
    
    # Split by underscore and remove first 2 parts to get imitation filename
    parts = stem.split('_', 2)  # Split into at most 3 parts
    if len(parts) >= 3:
        imitation_stem = parts[2]  # This is the original imitation filename stem
        # Extract sound_id using the same function used in data pipeline
        from src.qvae.utils import extract_sound_id
        sound_id = extract_sound_id(imitation_stem)
    else:
        return None
    
    # Import extract_sound_id to match reference files
    import sys
    sys.path.append('.')
    from src.qvae.utils import extract_sound_id
    
    # Look for reference file with matching sound_id
    dataset_path = Path(dataset_path)
    for ref_file in dataset_path.glob("*.wav"):
        if extract_sound_id(ref_file.stem) == sound_id:
            return ref_file
    
    # Try .mp3 if no .wav found
    for ref_file in dataset_path.glob("*.mp3"):
        if extract_sound_id(ref_file.stem) == sound_id:
            return ref_file
            
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualise generated audio quality")
    parser.add_argument("--samples-dir", default="runs/helpful-silence-50/generated_samples", 
                        help="Directory containing generated samples")
    parser.add_argument("--output-dir", default="audio_visualisations", 
                        help="Output directory for visualisations")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to visualise per validation set")
    parser.add_argument("--dataset-path", default="data/Vim_Sketch_Dataset/references",
                        help="Path to reference audio files")
    parser.add_argument("--train", type=lambda x: x.lower() == 'true', default=True, help="Generate spectrograms for training set (default: True)")
    parser.add_argument("--val-interpolation", type=lambda x: x.lower() == 'true', default=True, help="Generate spectrograms for interpolation validation set (default: True)")
    parser.add_argument("--val-extrapolation", type=lambda x: x.lower() == 'true', default=False, help="Generate spectrograms for extrapolation validation set (default: False)")
    parser.add_argument("--fixed-comparison", type=bool, default=True, help="Standardise comparison to 59049 samples at 22050Hz, limit frequency display to 11025Hz")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Also create audio_visualisations folder based on samples-dir location
    samples_path = Path(args.samples_dir)
    run_output_dir = None
    
    if str(samples_path).startswith('runs/'):
        # Special case for runs: Extract "runs/run-name" from path like "runs/run-name/generated_samples/epoch_0282"
        parts = str(samples_path).split('/')
        run_dir = Path('/'.join(parts[:2]))  # Get "runs/run-name"
        run_output_dir = run_dir / "audio_visualisations"
    else:
        # Default: Create audio_visualisations one level up from samples-dir
        # e.g., "generated_audio/qvae/v_01w4jqvw/epoch_0183" -> "generated_audio/qvae/v_01w4jqvw/audio_visualisations"
        run_output_dir = samples_path.parent / "audio_visualisations"
    
    run_output_dir.mkdir(exist_ok=True)
    print(f"Also saving visualisations to: {run_output_dir}")
    
    # Process sets
    sets_to_process = []
    if args.train:
        sets_to_process.append('train')
    if args.val_interpolation:
        sets_to_process.append('interpolation')
    if args.val_extrapolation:
        sets_to_process.append('extrapolation')
    
    if not sets_to_process:
        print("No sets selected for processing. Use --train, --val-interpolation, or --val-extrapolation")
        return
    
    for dataset in sets_to_process:
        dataset_dir = Path(args.samples_dir) / dataset
        if not dataset_dir.exists():
            print(f"Skipping {dataset} - directory not found")
            continue
            
        print(f"\nProcessing {dataset}...")
        
        # Create output subdirectory
        dataset_output_dir = output_dir / dataset
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Also create subdirectory in run folder if available
        run_dataset_output_dir = None
        if run_output_dir:
            run_dataset_output_dir = run_output_dir / dataset
            run_dataset_output_dir.mkdir(exist_ok=True)
        
        # Find generated files - look for .wav files (not necessarily *_generated.wav)
        generated_files = sorted(glob(str(dataset_dir / "*.wav")))[:args.num_samples]
        
        for gen_path in generated_files:
            gen_path = Path(gen_path)
            
            # Find corresponding reference file from dataset
            ref_path = find_reference_audio(gen_path, args.dataset_path)
            
            if not ref_path:
                print(f"  Warning: No reference found for {gen_path.name}")
                # Still visualise just the generated audio
                output_path = dataset_output_dir / f"{gen_path.stem}_generated_only.png"
                print(f"  Creating visualisation for generated audio only: {gen_path.stem}...")
                
                # Also save to run directory if available
                run_output_path = None
                if run_dataset_output_dir:
                    run_output_path = run_dataset_output_dir / f"{gen_path.stem}_generated_only.png"
                
                # Create a simplified visualisation for just generated audio
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                # Add title with sample directory
                title = f'{gen_path.stem} (no reference)\nFrom: {args.samples_dir}'
                fig.suptitle(title, fontsize=14)
                
                # Load generated audio
                gen_audio, gen_sr = torchaudio.load(gen_path)
                gen_audio = gen_audio.squeeze().numpy()
                
                # Waveform
                axes[0].plot(np.linspace(0, len(gen_audio)/gen_sr, len(gen_audio)), gen_audio)
                axes[0].set_title('Generated Waveform')
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Amplitude')
                
                # Load config for spectrogram settings
                config = load_config('config.yaml')
                
                # STFT (use same settings as comparison)
                n_fft = config['target_spectrograms']['stft']['window_sizes'][1]
                hop_length = config['target_spectrograms']['stft']['hop_lengths'][1]
                gen_stft = librosa.stft(gen_audio, n_fft=n_fft, hop_length=hop_length)
                gen_stft_db = librosa.amplitude_to_db(np.abs(gen_stft), ref=np.max)
                
                # Find frequency with highest median value
                gen_stft_median = np.median(np.abs(gen_stft), axis=1)  # Median across time
                max_freq_bin = np.argmax(gen_stft_median)
                max_freq_hz = max_freq_bin * gen_sr / n_fft  # Convert bin to Hz
                max_freq_value = gen_stft_median[max_freq_bin]
                
                print(f"\n  STFT Analysis - {gen_path.stem} (no reference):")
                print(f"    Peak frequency: {max_freq_hz:.1f} Hz (bin {max_freq_bin})")
                print(f"    Median magnitude at peak: {max_freq_value:.4f}")
                librosa.display.specshow(gen_stft_db, sr=gen_sr, hop_length=hop_length,
                                         x_axis='time', y_axis='hz', ax=axes[1])
                axes[1].set_title(f'Generated STFT Spectrogram (n_fft={n_fft})')
                
                # Mel
                mel_n_fft = config['target_spectrograms']['mel']['n_fft'][0]
                mel_hop_length = config['target_spectrograms']['mel']['hop_lengths'][0]
                n_mels = config['target_spectrograms']['mel']['n_mels']
                gen_mel = librosa.feature.melspectrogram(y=gen_audio, sr=gen_sr, n_mels=n_mels,
                                                          n_fft=mel_n_fft, hop_length=mel_hop_length)
                gen_mel_db = librosa.power_to_db(gen_mel, ref=np.max)
                librosa.display.specshow(gen_mel_db, sr=gen_sr, hop_length=mel_hop_length,
                                         x_axis='time', y_axis='mel', ax=axes[2])
                axes[2].set_title(f'Generated Mel Spectrogram (n_fft={mel_n_fft}, n_mels={n_mels})')
                
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                
                # Also save to run directory if available
                if run_output_path:
                    plt.savefig(run_output_path, dpi=150, bbox_inches='tight')
                
                plt.close()
                continue
                
            # Create visualisation
            output_path = dataset_output_dir / f"{gen_path.stem}_comparison.png"
            print(f"  Creating comparison for {gen_path.stem} with {ref_path.name}...")
            
            # Also save to run directory if available
            run_output_path = None
            if run_dataset_output_dir:
                run_output_path = run_dataset_output_dir / f"{gen_path.stem}_comparison.png"
            
            # Load config
            config = load_config('config.yaml')
            plot_audio_comparison(gen_path, ref_path, output_path, config, sample_dir=args.samples_dir, 
                                fixed_comparison=args.fixed_comparison)
            
            # Also save to run directory if available
            if run_output_path:
                plot_audio_comparison(gen_path, ref_path, run_output_path, config, sample_dir=args.samples_dir,
                                    fixed_comparison=args.fixed_comparison)
                
    print(f"\nVisualisations saved to: {output_dir}")


if __name__ == "__main__":
    main()