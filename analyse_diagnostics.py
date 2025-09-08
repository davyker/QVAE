"""
Diagnostic Analysis Script for QVAE Training

Analyses saved diagnostic data (.npz files) from DiagnosticCallback to provide
deep insights into VAE training dynamics, latent space evolution, and stochasticity.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from collections import defaultdict


def load_epoch_data(diagnostics_dir, epoch):
    """Load all diagnostic data for a specific epoch."""
    epoch_dir = Path(diagnostics_dir) / f"epoch_{epoch:04d}"
    
    if not epoch_dir.exists():
        return None
        
    data = {}
    
    # Load latent statistics
    latent_file = epoch_dir / "latent_stats.npz"
    if latent_file.exists():
        data['latent'] = dict(np.load(latent_file, allow_pickle=True))
        
    # Load stochasticity analysis
    stoch_file = epoch_dir / "stochasticity.npz"
    if stoch_file.exists():
        data['stochasticity'] = dict(np.load(stoch_file, allow_pickle=True))
        
    # Load gradients
    grad_file = epoch_dir / "gradients.npz"
    if grad_file.exists():
        data['gradients'] = dict(np.load(grad_file, allow_pickle=True))
        
    return data if data else None


def load_epoch_range(diagnostics_dir, start_epoch, end_epoch):
    """Load diagnostic data for a range of epochs."""
    epoch_data = {}
    
    for epoch in range(start_epoch, end_epoch + 1):
        data = load_epoch_data(diagnostics_dir, epoch)
        if data:
            epoch_data[epoch] = data
            
    return epoch_data


def analyse_latent_evolution(epoch_data, sample_names=None):
    """Analyse how latent representations evolve over epochs."""
    print("=== Latent Space Evolution Analysis ===")
    
    epochs = sorted(epoch_data.keys())
    
    # Track latent separation over epochs
    separations = []
    kl_by_sample = defaultdict(list)
    latent_norms = []
    
    for epoch in epochs:
        if 'latent' not in epoch_data[epoch]:
            continue
            
        latent_data = epoch_data[epoch]['latent']
        mu = latent_data['mu_per_sample']
        kl = latent_data['kl_per_sample']
        norms = latent_data['latent_norms']
        names = latent_data['sample_names']
        
        # Compute separation (average pairwise distance)
        if len(mu) > 1:
            distances = []
            for i in range(len(mu)):
                for j in range(i+1, len(mu)):
                    dist = np.linalg.norm(mu[i] - mu[j])
                    distances.append(dist)
            separations.append(np.mean(distances))
        else:
            separations.append(0.0)
            
        # Track KL per sample
        for name, kl_val in zip(names, kl):
            kl_by_sample[name].append(kl_val)
            
        latent_norms.append(np.mean(norms))
    
    # Plot latent separation evolution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Latent separation over time
    axes[0, 0].plot(epochs, separations, 'b-', marker='o')
    axes[0, 0].set_title('Latent Separation Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Average Pairwise Distance')
    axes[0, 0].grid(True)
    
    # Latent norm evolution
    axes[0, 1].plot(epochs, latent_norms, 'g-', marker='o')
    axes[0, 1].set_title('Average Latent Norm Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('L2 Norm')
    axes[0, 1].grid(True)
    
    # KL divergence per sample
    for sample_name, kl_values in kl_by_sample.items():
        if len(kl_values) == len(epochs):
            axes[1, 0].plot(epochs, kl_values, marker='o', label=sample_name[:20] + "...")
    axes[1, 0].set_title('KL Divergence Per Sample')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True)
    
    # Separation vs KL correlation
    if len(separations) == len(epochs) and kl_by_sample:
        avg_kl_per_epoch = []
        for epoch in epochs:
            if 'latent' in epoch_data[epoch]:
                kl = epoch_data[epoch]['latent']['kl_per_sample']
                avg_kl_per_epoch.append(np.mean(kl))
        
        if len(avg_kl_per_epoch) == len(separations):
            axes[1, 1].scatter(avg_kl_per_epoch, separations)
            axes[1, 1].set_xlabel('Average KL Divergence')
            axes[1, 1].set_ylabel('Latent Separation')
            axes[1, 1].set_title('KL vs Separation Correlation')
            axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def analyse_stochasticity(epoch_data, target_epochs=None):
    """Analyse VAE stochasticity across epochs."""
    print("=== Stochasticity Analysis ===")
    
    if target_epochs is None:
        target_epochs = sorted(epoch_data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epoch_variances = []
    epoch_stds = []
    epochs_with_data = []
    
    for epoch in target_epochs:
        if epoch not in epoch_data or 'stochasticity' not in epoch_data[epoch]:
            continue
            
        stoch_data = epoch_data[epoch]['stochasticity']
        gen_var = stoch_data['generation_variance']
        gen_std = stoch_data['generation_std']
        
        # Average across samples and audio length
        avg_var = np.mean(gen_var)
        avg_std = np.mean(gen_std)
        
        epoch_variances.append(avg_var)
        epoch_stds.append(avg_std)
        epochs_with_data.append(epoch)
    
    # Plot variance evolution
    if epoch_variances:
        axes[0, 0].plot(epochs_with_data, epoch_variances, 'r-', marker='o')
        axes[0, 0].set_title('Generation Variance Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Average Variance')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs_with_data, epoch_stds, 'purple', marker='o')
        axes[0, 1].set_title('Generation Std Dev Over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Average Std Dev')
        axes[0, 1].grid(True)
    
    # Show specific epoch stochasticity if available
    if target_epochs and target_epochs[0] in epoch_data:
        first_epoch = target_epochs[0]
        if 'stochasticity' in epoch_data[first_epoch]:
            stoch_data = epoch_data[first_epoch]['stochasticity']
            multi_gen = stoch_data['multi_generations']  # [samples, generations, audio_length]
            
            # Plot variance across audio length for first sample
            if len(multi_gen) > 0:
                sample_0_var = np.var(multi_gen[0], axis=0)  # Variance across generations
                axes[1, 0].plot(sample_0_var[:1000])  # First 1000 samples
                axes[1, 0].set_title(f'Sample 0 Variance Across Audio (Epoch {first_epoch})')
                axes[1, 0].set_xlabel('Audio Sample Index')
                axes[1, 0].set_ylabel('Variance Across Generations')
                axes[1, 0].grid(True)
    
    plt.tight_layout()
    return fig


def analyse_gradient_flow(epoch_data):
    """Analyse gradient norms across epochs."""
    print("=== Gradient Flow Analysis ===")
    
    epochs = sorted(epoch_data.keys())
    gradient_data = {}
    
    # Collect gradient norms
    for epoch in epochs:
        if 'gradients' not in epoch_data[epoch]:
            continue
            
        grad_norms = epoch_data[epoch]['gradients']['grad_norms'].item()
        
        for param_name, norm in grad_norms.items():
            if param_name not in gradient_data:
                gradient_data[param_name] = []
            gradient_data[param_name].append((epoch, norm))
    
    if not gradient_data:
        print("No gradient data found.")
        return None
    
    # Plot gradient evolution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot key component gradients
    key_components = ['decoder', 'encoder.vae_head', 'encoder.backbone']
    colours = ['blue', 'red', 'green']
    
    for i, (component, colour) in enumerate(zip(key_components, colours)):
        component_epochs = []
        component_norms = []
        
        for param_name, data in gradient_data.items():
            if component in param_name:
                for epoch, norm in data:
                    component_epochs.append(epoch)
                    component_norms.append(norm)
        
        if component_epochs:
            axes[0, 0].scatter(component_epochs, component_norms, 
                             c=colour, alpha=0.6, label=component, s=20)
    
    axes[0, 0].set_title('Gradient Norms by Component')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Gradient Norm')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # Total gradient norm per epoch
    total_norms = []
    epochs_with_grads = []
    
    for epoch in epochs:
        if 'gradients' in epoch_data[epoch]:
            grad_norms = epoch_data[epoch]['gradients']['grad_norms'].item()
            total_norm = np.sqrt(sum(norm**2 for norm in grad_norms.values()))
            total_norms.append(total_norm)
            epochs_with_grads.append(epoch)
    
    if total_norms:
        axes[0, 1].plot(epochs_with_grads, total_norms, 'orange', marker='o')
        axes[0, 1].set_title('Total Gradient Norm Over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Total Gradient Norm')
        axes[0, 1].grid(True)
    
    plt.tight_layout()
    return fig


def compare_epochs(epoch_data, good_epoch, bad_epoch):
    """Compare specific good vs bad epochs."""
    print(f"=== Comparing Epoch {good_epoch} vs Epoch {bad_epoch} ===")
    
    if good_epoch not in epoch_data or bad_epoch not in epoch_data:
        print("One or both epochs not found in data.")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Compare latent spaces
    for i, (epoch, title) in enumerate([(good_epoch, f'Good Epoch {good_epoch}'), 
                                       (bad_epoch, f'Bad Epoch {bad_epoch}')]):
        if 'latent' in epoch_data[epoch]:
            latent_data = epoch_data[epoch]['latent']
            mu = latent_data['mu_per_sample']
            names = latent_data['sample_names']
            
            if len(mu) >= 2:
                # PCA visualisation
                pca = PCA(n_components=2)
                mu_2d = pca.fit_transform(mu)
                
                # Color by sample type
                unique_names = list(set(names))
                colours = plt.cm.tab10(np.linspace(0, 1, len(unique_names)))
                
                for j, name in enumerate(unique_names):
                    indices = [k for k, s in enumerate(names) if s == name]
                    if indices:
                        axes[0, i].scatter(mu_2d[indices, 0], mu_2d[indices, 1], 
                                         c=[colours[j]], label=name[:15] + "...", alpha=0.7, s=50)
                
                axes[0, i].set_title(title + ' - Latent PCA')
                axes[0, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0, i].grid(True)
    
    # Compare stochasticity
    if 'stochasticity' in epoch_data[good_epoch] and 'stochasticity' in epoch_data[bad_epoch]:
        good_var = np.mean(epoch_data[good_epoch]['stochasticity']['generation_variance'])
        bad_var = np.mean(epoch_data[bad_epoch]['stochasticity']['generation_variance'])
        
        axes[0, 2].bar(['Good Epoch', 'Bad Epoch'], [good_var, bad_var], color=['green', 'red'])
        axes[0, 2].set_title('Average Generation Variance')
        axes[0, 2].set_ylabel('Variance')
    
    # Compare KL divergences
    if 'latent' in epoch_data[good_epoch] and 'latent' in epoch_data[bad_epoch]:
        good_kl = epoch_data[good_epoch]['latent']['kl_per_sample']
        bad_kl = epoch_data[bad_epoch]['latent']['kl_per_sample']
        
        axes[1, 0].hist(good_kl, alpha=0.7, label=f'Epoch {good_epoch}', color='green')
        axes[1, 0].hist(bad_kl, alpha=0.7, label=f'Epoch {bad_epoch}', color='red')
        axes[1, 0].set_title('KL Divergence Distribution')
        axes[1, 0].set_xlabel('KL Divergence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
    
    # Compare latent norms
    if 'latent' in epoch_data[good_epoch] and 'latent' in epoch_data[bad_epoch]:
        good_norms = epoch_data[good_epoch]['latent']['latent_norms']
        bad_norms = epoch_data[bad_epoch]['latent']['latent_norms']
        
        axes[1, 1].hist(good_norms, alpha=0.7, label=f'Epoch {good_epoch}', color='green')
        axes[1, 1].hist(bad_norms, alpha=0.7, label=f'Epoch {bad_epoch}', color='red')
        axes[1, 1].set_title('Latent Norm Distribution')
        axes[1, 1].set_xlabel('L2 Norm')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
    
    plt.tight_layout()
    return fig


def print_summary(epoch_data):
    """Print summary statistics."""
    print("=== Summary Statistics ===")
    
    epochs = sorted(epoch_data.keys())
    print(f"Epochs analysed: {len(epochs)} ({min(epochs)}-{max(epochs)})")
    
    # Find epochs with highest/lowest separation
    separations = {}
    kl_averages = {}
    
    for epoch in epochs:
        if 'latent' in epoch_data[epoch]:
            latent_data = epoch_data[epoch]['latent']
            mu = latent_data['mu_per_sample']
            kl = latent_data['kl_per_sample']
            
            if len(mu) > 1:
                distances = []
                for i in range(len(mu)):
                    for j in range(i+1, len(mu)):
                        dist = np.linalg.norm(mu[i] - mu[j])
                        distances.append(dist)
                separations[epoch] = np.mean(distances)
                
            kl_averages[epoch] = np.mean(kl)
    
    if separations:
        best_sep_epoch = max(separations, key=separations.get)
        worst_sep_epoch = min(separations, key=separations.get)
        
        print(f"Best latent separation: Epoch {best_sep_epoch} ({separations[best_sep_epoch]:.3f})")
        print(f"Worst latent separation: Epoch {worst_sep_epoch} ({separations[worst_sep_epoch]:.3f})")
    
    if kl_averages:
        highest_kl_epoch = max(kl_averages, key=kl_averages.get)
        lowest_kl_epoch = min(kl_averages, key=kl_averages.get)
        
        print(f"Highest KL: Epoch {highest_kl_epoch} ({kl_averages[highest_kl_epoch]:.3f})")
        print(f"Lowest KL: Epoch {lowest_kl_epoch} ({kl_averages[lowest_kl_epoch]:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Analyse QVAE diagnostic data")
    parser.add_argument("--run-dir", required=True, help="Path to run directory (e.g., runs/helpful-silence-50)")
    parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch for analysis")
    parser.add_argument("--end-epoch", type=int, help="End epoch for analysis (default: latest)")
    parser.add_argument("--compare-epochs", nargs=2, type=int, metavar=('GOOD', 'BAD'),
                        help="Compare two specific epochs")
    parser.add_argument("--focus-epochs", nargs='+', type=int, help="Focus analysis on specific epochs")
    parser.add_argument("--output-dir", default=None, help="Override output directory (default: run-dir/diagnostic_analysis)")
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    # Find diagnostics directory
    run_path = Path(args.run_dir)
    diagnostics_dir = run_path / "diagnostics"
    
    if not diagnostics_dir.exists():
        print(f"Diagnostics directory not found: {diagnostics_dir}")
        return
    
    # Find available epochs
    epoch_dirs = [d for d in diagnostics_dir.iterdir() if d.is_dir() and d.name.startswith('epoch_')]
    available_epochs = sorted([int(d.name.split('_')[1]) for d in epoch_dirs])
    
    if not available_epochs:
        print("No epoch data found in diagnostics directory")
        return
    
    print(f"Found diagnostic data for epochs: {available_epochs}")
    
    # Determine epoch range
    start_epoch = max(args.start_epoch, min(available_epochs))
    end_epoch = args.end_epoch if args.end_epoch else max(available_epochs)
    end_epoch = min(end_epoch, max(available_epochs))
    
    print(f"Analysing epochs {start_epoch}-{end_epoch}")
    
    # Load data
    epoch_data = load_epoch_range(diagnostics_dir, start_epoch, end_epoch)
    
    if not epoch_data:
        print("No data loaded. Check epoch range and diagnostic files.")
        return
    
    # Create output directory (default: run-dir/diagnostic_analysis)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_path / "diagnostic_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Print summary
    print_summary(epoch_data)
    print()
    
    # Generate analyses
    figures = []
    
    # Latent evolution analysis
    fig1 = analyse_latent_evolution(epoch_data)
    if fig1:
        fig1.savefig(output_dir / "latent_evolution.png", dpi=150, bbox_inches='tight')
        figures.append(fig1)
        print("✓ Latent evolution analysis saved")
    
    # Stochasticity analysis
    focus_epochs = args.focus_epochs if args.focus_epochs else list(epoch_data.keys())
    fig2 = analyse_stochasticity(epoch_data, focus_epochs)
    if fig2:
        fig2.savefig(output_dir / "stochasticity_analysis.png", dpi=150, bbox_inches='tight')
        figures.append(fig2)
        print("✓ Stochasticity analysis saved")
    
    # Gradient flow analysis
    fig3 = analyse_gradient_flow(epoch_data)
    if fig3:
        fig3.savefig(output_dir / "gradient_flow.png", dpi=150, bbox_inches='tight')
        figures.append(fig3)
        print("✓ Gradient flow analysis saved")
    
    # Epoch comparison
    if args.compare_epochs:
        good_epoch, bad_epoch = args.compare_epochs
        fig4 = compare_epochs(epoch_data, good_epoch, bad_epoch)
        if fig4:
            fig4.savefig(output_dir / f"epoch_comparison_{good_epoch}_vs_{bad_epoch}.png", 
                        dpi=150, bbox_inches='tight')
            figures.append(fig4)
            print(f"✓ Epoch comparison ({good_epoch} vs {bad_epoch}) saved")
    
    print(f"\nAll plots saved to: {output_dir}")
    
    # Show plots if requested
    if args.show_plots:
        plt.show()
    else:
        # Close figures to free memory
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()