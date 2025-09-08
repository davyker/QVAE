"""
QVAE Data Pipeline

VimSketch dataset downloading and preprocessing for vocal imitation to audio generation.
"""

import os
import requests
import subprocess
import sys
import tarfile
from pathlib import Path
from tqdm import tqdm
import yaml
import pandas as pd
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import librosa

# Add QVIM baseline to path for preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '../../qvim-baseline/src'))
from qvim_mn_baseline.mn.preprocess import AugmentMelSTFT

# Import utilities
try:
    from .utils import extract_sound_id, load_config
except ImportError:
    # Handle case when running as script
    sys.path.append('.')
    from src.qvae.utils import extract_sound_id, load_config


def download_zip(url, zip_file):
    """Download a zip file with progress bar."""
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}. Status code: {response.status_code}")
    
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192
    
    with open(zip_file, "wb") as file, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"Downloading {zip_file.name}"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))
    
    print(f"Download completed: {zip_file}")


def extract_zip(zip_file, extract_to_dir):
    """Extract a ZIP file using system tools."""
    extract_to_dir = Path(extract_to_dir)
    extract_to_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if sys.platform == "darwin":
            # macOS
            subprocess.run(["/usr/bin/unzip", str(zip_file), "-d", str(extract_to_dir)], 
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
        elif sys.platform.startswith("win"):
            # Windows
            subprocess.run([
                "powershell", "-Command", 
                f"Expand-Archive -Path '{zip_file}' -DestinationPath '{extract_to_dir}' -Force"
            ], check=True)
        else:
            # Linux (or other Unix-like)
            try:
                subprocess.run(["unzip", str(zip_file), "-d", str(extract_to_dir)], 
                             stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                subprocess.run(["7z", "x", str(zip_file), f"-o{extract_to_dir}"], check=True)
                
        print(f"Extraction completed: {extract_to_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {zip_file}: {e}")
        print("Please extract manually or install unzip/7zip")
        raise

def download_and_extract_vimsketch_dataset(data_dir="data", source="gdrive"):
    """Download and extract VimSketch dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = data_dir / "Vim_Sketch_Dataset"
    
    # Check if already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"VimSketch dataset already exists at {extract_dir}")
        return extract_dir
    
    if source == "zenodo":
        # Original Zenodo source (some known unzipping issues)
        url = "https://zenodo.org/records/2596911/files/Vim_Sketch_Dataset.zip?download=1"
        zip_file = data_dir / "VimSketch.zip"
        
        # Download if needed
        if not zip_file.exists():
            print("Downloading VimSketch dataset from Zenodo...")
            download_zip(url, zip_file)
        else:
            print(f"VimSketch zip already exists: {zip_file}")
        
        # Extract
        print("Extracting VimSketch dataset...")
        extract_zip(zip_file, data_dir)
        
    elif source == "gdrive":
        # Google Drive source (more reliable)
        import tarfile
        
        GDRIVE_ID = "1qc8khcH0ipm2YBaUXXAk0qJR8TWN8k1k"
        tar_file = data_dir / "VimSketch.tar"
        
        # Download tar file if it doesn't exist
        if not tar_file.exists():
            print(f"Downloading VimSketch.tar from Google Drive...")
            subprocess.run([
                "gdown", 
                "--id", GDRIVE_ID, 
                "--output", str(tar_file)
            ], check=True)
        else:
            print(f"VimSketch tar already exists: {tar_file}")
        
        # Extract tar file
        print(f"Extracting {tar_file}...")
        with tarfile.open(tar_file) as tar:
            tar.extractall(path=data_dir)
    
    # Verify extraction
    if not extract_dir.exists():
        raise RuntimeError(f"Extraction failed - {extract_dir} not found")
    
    print(f"✅ VimSketch dataset ready at: {extract_dir}")
    return extract_dir


def load_audio_qvim_style(path, sample_rate=32000, duration=10.0):
    """Load audio exactly as QVIM does."""
    audio, sr = librosa.load(path, sr=sample_rate, mono=True, duration=duration)
    return pad_or_truncate(audio, sample_rate, duration)


def pad_or_truncate(audio, sample_rate=32000, duration=10.0):
    """Pad or truncate audio exactly as QVIM does."""
    fixed_length = int(sample_rate * duration)
    if len(audio) < fixed_length:
        array = np.zeros(fixed_length, dtype="float32")
        array[:len(audio)] = audio
    if len(audio) >= fixed_length:
        array = audio[:fixed_length]
    return array


def load_encoder_model(config):
    """Load frozen MobileNetV3 encoder from checkpoint."""
    from src.qvae.encoder import QVAEEncoder
    
    encoder = QVAEEncoder(
        pretrained_name=config['model']['encoder_name'],
        checkpoint_path=config['model']['encoder_checkpoint']
    )
    encoder.eval()
    return encoder


def compute_spectrograms(audio, config):
    """Compute multi-scale STFT and mel spectrograms for target audio."""
    spectrograms = {}
    
    # Multi-scale STFT spectrograms
    stft_configs = list(zip(
        config['target_spectrograms']['stft']['window_sizes'],
        config['target_spectrograms']['stft']['hop_lengths'],
        config['target_spectrograms']['stft']['win_lengths']
    ))
    
    spectrograms['stft'] = []
    for n_fft, hop_length, win_length in stft_configs:
        stft = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True
        )
        spectrograms['stft'].append(stft)
    
    # Mel spectrograms
    mel_configs = list(zip(
        config['target_spectrograms']['mel']['n_fft'],
        config['target_spectrograms']['mel']['hop_lengths']
    ))
    
    spectrograms['mel'] = []
    for n_fft, hop_length in mel_configs:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['data']['target_sample_rate'],
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=config['target_spectrograms']['mel']['n_mels'],
            f_min=config['target_spectrograms']['mel']['f_min'],
            f_max=config['target_spectrograms']['mel']['f_max']
        )
        mel_spec = mel_transform(audio)
        spectrograms['mel'].append(mel_spec)
    
    return spectrograms


def create_splits(encoder_features, config):
    """Create and save train/validation splits."""
    cache_dir = Path(config['data']['cache_dir'])
    splits_file = cache_dir / "splits.pkl"
    
    # Check if splits already exist
    if splits_file.exists():
        print("Splits already exist, skipping...")
        return
    
    print("Creating train/validation splits...")
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(config['data']['split_random_seed'])
    
    # Get all sound_ids
    all_sounds = list(encoder_features.keys())
    rng.shuffle(all_sounds)
    
    # Split sounds: 90% train, 10% extrapolation
    n_train_sounds = int(len(all_sounds) * config['data']['train_sound_ratio'])
    train_sounds = all_sounds[:n_train_sounds]
    extrapolation_sounds = all_sounds[n_train_sounds:]
    
    # Split imitations within train sounds
    train_imitations = {}
    interpolation_imitations = {}
    
    for sound_id in train_sounds:
        all_imitations = list(encoder_features[sound_id].keys())
        rng.shuffle(all_imitations)
        
        n_train_imitations = int(len(all_imitations) * config['data']['train_imitation_ratio'])
        train_imitations[sound_id] = all_imitations[:n_train_imitations]
        interpolation_imitations[sound_id] = all_imitations[n_train_imitations:]
    
    # Save splits
    splits = {
        'train_sounds': train_sounds,
        'extrapolation_sounds': extrapolation_sounds,
        'train_imitations': train_imitations,
        'interpolation_imitations': interpolation_imitations,
        'metadata': {
            'random_seed': config['data']['split_random_seed'],
            'train_sound_ratio': config['data']['train_sound_ratio'],
            'train_imitation_ratio': config['data']['train_imitation_ratio'],
            'total_sounds': len(all_sounds),
            'n_train_sounds': len(train_sounds),
            'n_extrapolation_sounds': len(extrapolation_sounds)
        }
    }
    
    with open(splits_file, 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"✅ Created splits: {len(train_sounds)} train sounds, {len(extrapolation_sounds)} extrapolation sounds")
    print(f"✅ Saved splits to {splits_file}")


def load_splits(config):
    """Load pre-computed train/validation splits."""
    splits_file = Path(config['data']['cache_dir']) / "splits.pkl"
    with open(splits_file, 'rb') as f:
        return pickle.load(f)


def precompute_features(config, force_recompute=False, batch_size=None):
    """Pre-compute all encoder features and target spectrograms."""
    if batch_size is None:
        batch_size = config['data']['precompute_batch_size']
    
    dataset_path = Path(config['data']['dataset_path']) / "Vim_Sketch_Dataset"
    cache_dir = Path(config['data']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    features_cache = cache_dir / "encoder_features.pkl"
    spectrograms_cache = cache_dir / "target_spectrograms.pkl"
    audio_cache = cache_dir / "target_audio.pkl"
    mel_spectrograms_cache = cache_dir / "input_mel_spectrograms.pkl"
    
    # Check if already computed
    if not force_recompute and features_cache.exists() and spectrograms_cache.exists() and audio_cache.exists():
        print("Pre-computed features found, loading from cache...")
        with open(features_cache, 'rb') as f:
            encoder_features = pickle.load(f)
        with open(spectrograms_cache, 'rb') as f:
            target_spectrograms = pickle.load(f)
        return encoder_features, target_spectrograms
    
    print("Computing encoder features and target spectrograms...")
    
    # Load filename lists
    with open(dataset_path / "vocal_imitation_file_names.csv") as f:
        vocal_files = [line.strip() for line in f]
    with open(dataset_path / "reference_file_names.csv") as f:
        ref_files = [line.strip() for line in f]

    # Load encoder and preprocessor
    encoder = load_encoder_model(config)
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    
    # QVIM preprocessor (no augmentation for pre-computation)
    preprocessor = AugmentMelSTFT().to(device)
    preprocessor.eval()
    
    # Verify no augmentation with dummy audio
    dummy_audio = torch.randn(1, 32000, device=device)
    mel1 = preprocessor(dummy_audio)
    mel2 = preprocessor(dummy_audio)
    print(f"Augmentation disabled: {torch.equal(mel1, mel2)}")
    
    encoder_features = {}  # Structure: {sound_id: {vocal_file: features}}
    target_spectrograms = {}  # Structure: {sound_id: {filename: ref_file, stft: [...], mel: [...]}}
    target_audio = {}  # Structure: {sound_id: audio_tensor}
    input_mel_spectrograms = {}  # Structure: {sound_id: {vocal_file: mel_spec}}
    
    print(f"Processing {len(vocal_files)} vocal imitations...")
    
    # Process vocal imitations -> encoder features in batches
    for i in tqdm(range(0, len(vocal_files), batch_size), desc="Computing encoder features"):
        batch_files = vocal_files[i:i+batch_size]
        batch_audio = []
        valid_files = []
        
        for vocal_file in batch_files:
            vocal_path = dataset_path / "vocal_imitations" / vocal_file
            if not vocal_path.exists():
                continue
            try:
                audio = load_audio_qvim_style(vocal_path)
                batch_audio.append(torch.from_numpy(audio))
                valid_files.append(vocal_file)
            except Exception as e:
                print(f"Error processing {vocal_path}: {e}")
        
        if batch_audio:
            with torch.no_grad():
                batch_tensor = torch.stack(batch_audio).to(device)
                mel_specs = preprocessor(batch_tensor)
                features = encoder.get_features(mel_specs.unsqueeze(1))
                for j, vocal_file in enumerate(valid_files):
                    sound_id = extract_sound_id(vocal_file)
                    if sound_id not in encoder_features:
                        encoder_features[sound_id] = {}
                    if sound_id not in input_mel_spectrograms:
                        input_mel_spectrograms[sound_id] = {}
                    encoder_features[sound_id][vocal_file] = features[j].cpu()
                    # Store mel spectrogram (add channel dim for encoder compatibility)
                    input_mel_spectrograms[sound_id][vocal_file] = mel_specs[j].cpu().unsqueeze(0)  # [1, 128, T]
    
    print(f"Processing {len(ref_files)} reference audio files...")
    
    # Process reference audio -> target spectrograms
    for ref_file in tqdm(ref_files, desc="Computing target spectrograms"):
        ref_path = dataset_path / "references" / ref_file
        
        if not ref_path.exists():
            continue
            
        try:
            audio, sr = torchaudio.load(ref_path)
            audio = audio.squeeze(0)  # [1, samples] -> [samples] for mono audio
            if sr != config['data']['target_sample_rate']:
                audio = torchaudio.functional.resample(audio, sr, config['data']['target_sample_rate'])
            
            # Ensure correct length for decoder target
            target_length = config['data']['target_audio_length']
            if audio.shape[-1] > target_length:
                audio = audio[..., :target_length]
            elif audio.shape[-1] < target_length:
                audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
            
            # Compute spectrograms
            spectrograms = compute_spectrograms(audio, config)
            sound_id = extract_sound_id(ref_file)
            target_spectrograms[sound_id] = {
                'filename': ref_file,
                'stft': spectrograms['stft'],
                'mel': spectrograms['mel']
            }
            
            # Store target audio for waveform loss
            target_audio[sound_id] = audio
            
        except Exception as e:
            print(f"Error processing {ref_path}: {e}")
            continue
    
    # Save to cache
    print("Saving pre-computed features to cache...")
    with open(features_cache, 'wb') as f:
        pickle.dump(encoder_features, f)
    with open(spectrograms_cache, 'wb') as f:
        pickle.dump(target_spectrograms, f)
    with open(audio_cache, 'wb') as f:
        pickle.dump(target_audio, f)
    with open(mel_spectrograms_cache, 'wb') as f:
        pickle.dump(input_mel_spectrograms, f)
    
    total_vocal_features = sum(len(vocals) for vocals in encoder_features.values())
    print(f"✅ Pre-computed {total_vocal_features} encoder features across {len(encoder_features)} sounds")
    print(f"✅ Pre-computed {len(target_spectrograms)} target spectrograms")
    print(f"✅ Pre-computed {len(target_audio)} target audio files")
    print(f"✅ Pre-computed {total_vocal_features} input mel spectrograms")
    return encoder_features, target_spectrograms


class VimSketchDataset(Dataset):
    """PyTorch Dataset for VimSketch vocal imitation -> reference audio pairs."""
    
    def __init__(self, config, split='train', encoder_features=None, target_spectrograms=None, 
                 input_mel_spectrograms=None, target_audio=None, overfit_subset=None, overfit_training_mode=True):
        self.config = config
        self.split = split
        
        # Determine mode based on fine-tuning configuration and from_scratch
        self.fine_tuning_enabled = config['model']['encoder_fine_tuning']['enabled']
        self.from_scratch = config['model'].get('from_scratch', False)
        
        # Load pre-computed features if not provided
        if encoder_features is None or target_spectrograms is None:
            self.encoder_features, self.target_spectrograms = precompute_features(config)
        else:
            self.encoder_features = encoder_features
            self.target_spectrograms = target_spectrograms
            
        # Store mel-spectrograms cache if provided
        self.input_mel_spectrograms = input_mel_spectrograms
        
        # Store target audio cache if provided
        self.target_audio = target_audio
            
        splits = load_splits(config)
        
        # Create pairs based on split
        self.pairs = []
        
        if split == 'train':
            # Training set: train sounds with train imitations
            for sound_id in splits['train_sounds']:
                if sound_id in self.target_spectrograms:
                    for vocal_file in splits['train_imitations'].get(sound_id, []):
                        self.pairs.append((vocal_file, sound_id))
        
        elif split == 'interpolation_val':
            # Interpolation val: train sounds with held-out imitations
            for sound_id in splits['train_sounds']:
                if sound_id in self.target_spectrograms:
                    for vocal_file in splits['interpolation_imitations'].get(sound_id, []):
                        self.pairs.append((vocal_file, sound_id))
        
        elif split == 'extrapolation_val':
            # Extrapolation val: held-out sounds with all their imitations
            for sound_id in splits['extrapolation_sounds']:
                if sound_id in self.target_spectrograms and sound_id in self.encoder_features:
                    for vocal_file in self.encoder_features[sound_id]:
                        self.pairs.append((vocal_file, sound_id))
        
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Apply overfit subset AFTER split logic
        if overfit_subset is not None:
            import random
            original_pairs = random.sample(self.pairs, overfit_subset)
            
            if overfit_training_mode:
                # Create 500 interleaved copies for fast epochs
                self.pairs = []
                for copy_num in range(500):
                    self.pairs.extend(original_pairs)
                
                print(f"=== Overfit TRAINING mode: {overfit_subset} unique samples, 500 copies each ({len(self.pairs)} total) ===")
                for i, (vocal_file, sound_id) in enumerate(original_pairs):
                    print(f"  {i+1}: {vocal_file}")
                print(f"Total pairs for training: {len(self.pairs)} (interleaved)")
                print("=" * 50)
            else:
                # Just use original samples (no copies)
                self.pairs = original_pairs
                print(f"=== Overfit VALIDATION mode: {overfit_subset} samples (no copies) ===")
                for i, (vocal_file, sound_id) in enumerate(self.pairs):
                    print(f"  {i+1}: {vocal_file}")
                print("=" * 30)
        
        print(f"Created {len(self.pairs)} vocal->reference pairs for {split} split")
    
    def __len__(self):
        return len(self.pairs)
    
    
    def __getitem__(self, idx):
        vocal_file, sound_id = self.pairs[idx]
        
        # Get pre-computed target spectrograms
        target_spectrograms = self.target_spectrograms[sound_id]
        
        batch_data = {
            'target_spectrograms': target_spectrograms,
            'vocal_file': vocal_file,
            'ref_file': target_spectrograms['filename'],
            'sound_id': sound_id
        }
        
        # Include target audio if available
        if self.target_audio is not None:
            batch_data['target_audio'] = self.target_audio[sound_id]
        
        if self.fine_tuning_enabled or self.from_scratch:
            # Fine-tuning or from-scratch mode: return input mel-spectrograms for on-the-fly encoding
            mel_spectrograms = self.input_mel_spectrograms[sound_id][vocal_file]
            batch_data['input_mel_spectrograms'] = mel_spectrograms
        else:
            # Frozen pretrained mode: return pre-computed encoder features
            encoder_features = self.encoder_features[sound_id][vocal_file]
            batch_data['encoder_features'] = encoder_features
            
        return batch_data


def create_dataloader(config, split='train', shuffle=True, encoder_features=None, target_spectrograms=None, 
                     input_mel_spectrograms=None, target_audio=None, overfit_subset=None, overfit_training_mode=True):
    """Create DataLoader for training/validation."""
    dataset = VimSketchDataset(config, split=split, encoder_features=encoder_features, 
                              target_spectrograms=target_spectrograms, 
                              input_mel_spectrograms=input_mel_spectrograms, 
                              target_audio=target_audio,
                              overfit_subset=overfit_subset, overfit_training_mode=overfit_training_mode)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=config['training']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def main():
    """Test dataset downloading and pre-computation."""
    config = load_config()
    data_dir = config['data']['dataset_path']
    cache_dir = Path(config['data']['cache_dir'])
    
    print("Testing VimSketch dataset download...")
    dataset_path = download_and_extract_vimsketch_dataset(data_dir)
    
    # Basic verification
    print(f"\nDataset contents:")
    for item in sorted(dataset_path.iterdir())[:10]:  # Show first 10 items
        print(f"  {item.name}")
    
    total_items = len(list(dataset_path.rglob("*")))
    print(f"\nTotal items in dataset: {total_items}")
    
    # Test pre-computation
    print("\nTesting pre-computation pipeline...")
    encoder_features, target_spectrograms = precompute_features(config)
    
    # Create train/val splits
    print("\nCreating train/validation splits...")
    create_splits(encoder_features, config)
    
    # Test dataset creation
    print("\nTesting dataset creation...")
    
    # Load mel-spectrograms if needed for fine-tuning or from-scratch mode
    input_mel_spectrograms = None
    if config['model']['encoder_fine_tuning']['enabled'] or config['model'].get('from_scratch', False):
        mel_cache_path = cache_dir / "input_mel_spectrograms.pkl"
        if mel_cache_path.exists():
            print("Loading input mel-spectrograms for fine-tuning/from-scratch mode...")
            with open(mel_cache_path, 'rb') as f:
                input_mel_spectrograms = pickle.load(f)
        else:
            print("Warning: Input mel-spectrograms cache not found!")
    
    dataset = VimSketchDataset(config, input_mel_spectrograms=input_mel_spectrograms)
    print(f"Dataset size: {len(dataset)}")
    
    # Test first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        if 'encoder_features' in sample:
            print(f"Encoder features shape: {sample['encoder_features'].shape}")
        if 'input_mel_spectrograms' in sample:
            print(f"Input mel-spectrograms shape: {sample['input_mel_spectrograms'].shape}")
        print(f"Target spectrograms - STFT scales: {len(sample['target_spectrograms']['stft'])}")
        print(f"Target spectrograms - Mel scales: {len(sample['target_spectrograms']['mel'])}")


if __name__ == "__main__":
    main()