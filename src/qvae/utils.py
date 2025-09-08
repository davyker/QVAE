"""
QVAE Utility Functions

Shared utilities for data loading and common operations.
"""

import pickle
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_precomputed_features(cache_dir):
    """Load pre-computed encoder features from cache directory.
    
    Args:
        cache_dir: Path to cache directory containing encoder_features.pkl
        
    Returns:
        dict: Encoder features organised by sound_id
    """
    cache_dir = Path(cache_dir)
    features_path = cache_dir / "encoder_features.pkl"
    
    if not features_path.exists():
        raise FileNotFoundError(f"Pre-computed features not found at {features_path}")
    
    with open(features_path, 'rb') as f:
        encoder_features = pickle.load(f)
    
    print(f"Loaded {len(encoder_features)} encoder features")
    return encoder_features


def load_precomputed_spectrograms(cache_dir):
    """Load pre-computed target spectrograms from cache directory.
    
    Args:
        cache_dir: Path to cache directory containing target_spectrograms.pkl
        
    Returns:
        dict: Target spectrograms organised by sound_id
    """
    cache_dir = Path(cache_dir)
    spectrograms_path = cache_dir / "target_spectrograms.pkl"
    
    if not spectrograms_path.exists():
        raise FileNotFoundError(f"Pre-computed spectrograms not found at {spectrograms_path}")
    
    with open(spectrograms_path, 'rb') as f:
        target_spectrograms = pickle.load(f)
    
    print(f"Loaded {len(target_spectrograms)} target spectrograms")
    return target_spectrograms


def extract_sound_id(filename):
    """Extract sound_id from filename (part after first underscore)."""
    return filename.split('_', 1)[1] if '_' in filename else filename


def extract_reference_id(filename):
    """Extract reference ID from filename (part before first underscore).
    
    Args:
        filename: Filename like "001_000Animal_Domestic..."
        
    Returns:
        str: Reference ID like "001"
    """
    return filename.split('_')[0] if '_' in filename else filename