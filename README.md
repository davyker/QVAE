# QVAE: Variational Autoencoder for Audio Generation from Vocal Imitations

A variational autoencoder that generates realistic audio from vocal imitations, using MobileNetV3 encoder features pretrained for query by vocal imitation.

## Overview

QVAE transforms human vocal imitations (e.g., "whoosh" sounds, beatboxing) into synthesised audio of a target sound. The model repurposes discriminative features from a pretrained MobileNetV3 encoder for the generative task of audio synthesis.

- Generates 2.68s audio clips (59,049 samples at 22,050 Hz) from 10s vocal imitations
- Uses frozen or fine-tuned MobileNetV3 encoder pretrained on AudioSet and VimSketch
- Uses multi-scale spectral and mel-spectrogram perceptual losses

## Requirements

- Python 3.12 recommended
- 8GB+ GPU memory recommended for training
- ~10GB disk space for dataset and checkpoints

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QVAE.git
cd QVAE

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Setup

The VimSketch dataset (~5.6GB) will be automatically downloaded by running:

```bash
python src/qvae/data.py
```

Alternatively, you can manually download it from https://zenodo.org/records/2596911 and extract to: data/Vim_Sketch_Dataset/


### Training
```bash
# Train on full dataset with default configuration in config.yaml
python train.py

# Overfitting test on small dataset
python train.py --overfit --overfit-samples 3

# Resume from checkpoint
python train.py --resume path/to/checkpoint.pt

# Resume from checkpoint and reset epoch counter
python train.py --resume path/to/checkpoint.pt --reset-epoch
```

### Generate Audio

The script `generate_samples.py` allows you to generate audio samples from a trained model checkpoint. If audio files are not specified via `--input-files`, the script will select `--num-samples` random audio files from the dataset, with default 5 unless specified. The generated audio will be saved alongside a copy of the vocal imitation used as input.

```bash
# Generate samples from specific checkpoint
python generate_samples.py --checkpoint path/to/checkpoint.pt --num-samples 10

# Generate samples from best checkpoint in run directory
python generate_samples.py --run-dir runs/your-run --num-samples 5
```

To demonstrate the audio generation capabilities of the decoder, you can generate samples from the following checkpoints, where the audio files specified are from the training set during the overfitting test:

```bash
python generate_samples.py --run-dir run-archive/overfitting_test_seed_421/ --input-files 10046_201-tambourine_shake_roll-acoustic_instruments.wav 01891_100Music_Musical\ instrument_Orchestra.wav 04742_254Sounds\ of\ things_Vehicle_Boat_Water\ vehicle_Ship.wav
```

```bash
python generate_samples.py --run-dir run-archive/overfitting_test_seed_422/ --input-files 04425_237Sounds\ of\ things_Mechanisms_Sewing\ machine.wav 02067_109Music_Musical\ instrument_Percussion_Drum_Timpani.wav 04194_223Sounds\ of\ things_Liquid_Splash_splatter_Slosh.wav
```

# Visualise generated audio quality
```bash
python visualise_generated_audio.py --samples-dir generated_audio/your-run/epoch_0100 --fixed-comparison
```

## Model Architecture

### Encoder
- **Base**: MobileNetV3-Large pretrained on AudioSet
- **Input**: 128-channel mel-spectrograms (10s at 32kHz)
- **Features**: Final pooled features (960-dim) or spatial features from earlier layers
- **VAE Head**: Projects to mean and log-variance of configurable dimension

### Decoder
- **Architecture**: Transposed SampleCNN with 10 upsampling stages
- **Upsampling**: 3× interpolation + Conv1d (avoids ConvTranspose1d resonance artifacts)
- **Output**: 59,049 audio samples at 22,050 Hz
- **Final Projection**: Multi-stage taper (128→32→8→1) for cleaner harmonics

### Loss Functions
```python
L_total = α_spec * L_spectral + α_mel * L_mel + β_kl * L_kl
```
- **Spectral Loss**: Multi-scale STFT with 4 window sizes [512, 1024, 2048, 4096]
- **Mel Loss**: 80-channel mel-spectrograms at 2 scales
- **KL Divergence**: Standard VAE regularisation

## Configuration

See `config.yaml` for all training configuration options.

## Project Structure

```
QVAE/
├── src/qvae/
│   ├── encoder.py         # MobileNetV3 encoder with VAE head
│   ├── decoder.py         # Original decoder implementation
│   ├── decoders.py        # Multiple decoder variants
│   ├── losses.py          # Multi-scale perceptual losses
│   ├── data.py            # VimSketch dataset loader
│   ├── callbacks.py       # Training callbacks for audio generation
│   └── utils.py           # Helper functions
├── train.py              # Main training script (PyTorch Lightning)
├── generate_samples.py   # Audio generation from checkpoints
├── visualise_generated_audio.py  # Spectrogram analysis and comparison
└── config.yaml           # Training configuration
```

## Results

- **Small subset (3-10 samples)**: High-quality reconstruction
- **Full Dataset (12,453 samples)**: Generates semantically appropriate audio on unseen sounds

## Known Issues

- Audio quality gap between small subset and full-scale training

## References

- [Greif et al. (2024)](https://github.com/qvim-aes/qvim-baseline): Query-by-Vocal Imitation baseline
- [Lee et al. (2018)](https://github.com/kyungyunlee/sampleCNN-pytorch): SampleCNN implementation
- [VimSketch Dataset](https://doi.org/10.5281/zenodo.1340763): Vocal imitations of everyday sounds

## Acknowledgements

- Queen Mary University of London, School of Electronic Engineering and Computer Science
- QVIM baseline implementation from AES AIMLA Challenge (https://github.com/qvim-aes/qvim-baseline/)