#!/usr/bin/env python3
"""
Visualization script for bilinear modular arithmetic results

This script creates visualizations from:
1. Training logs (training_log.json)
2. Saved model checkpoints
3. Analysis results

Usage:
    python visualize_results.py [--checkpoint best_model.pt] [--log training_log.json]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
import sys
from datetime import datetime

# Import the model and analysis functions
from bilinear_modular_arithmetic import (
    BilinearLayer, load_checkpoint, load_training_log,
    compute_interaction_matrices, compute_tensor_svd,
    plot_interaction_matrices, plot_tensor_svd_analysis,
    analyze_modular_structure
)

def plot_training_curves(log_file='training_log.json', output_file='training_curves.png'):
    """Plot training curves from log file"""
    log_data = load_training_log(log_file)
    
    if not log_data:
        print(f"No log data found in {log_file}")
        return
    
    epochs = [entry['epoch'] for entry in log_data]
    train_losses = [entry['train_loss'] for entry in log_data]
    val_accuracies = [entry['val_accuracy'] for entry in log_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training loss
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[1].plot(epochs, val_accuracies, 'g-', linewidth=2)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    # Combined view
    ax_twin = axes[2].twinx()
    
    line1 = axes[2].plot(epochs, train_losses, 'b-', linewidth=2, label='Loss')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12, color='b')
    axes[2].tick_params(axis='y', labelcolor='b')
    axes[2].grid(True, alpha=0.3)
    
    line2 = ax_twin.plot(epochs, val_accuracies, 'g-', linewidth=2, label='Accuracy')
    ax_twin.set_ylabel('Accuracy (%)', fontsize=12, color='g')
    ax_twin.tick_params(axis='y', labelcolor='g')
    ax_twin.set_ylim([0, 100])
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[2].legend(lines, labels, loc='center right')
    axes[2].set_title('Training Progress', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to {output_file}")
    
    plt.show()

def plot_model_weights(model, P=113):
    """Visualize the learned weight tensor"""
    
    W = model.W.detach().cpu().numpy()
    
    # Plot weight distribution for each output dimension
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall weight distribution
    axes[0, 0].hist(W.flatten(), bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Weight Distribution (All)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Per-output weight statistics
    output_means = []
    output_stds = []
    for k in range(model.output_dim):
        output_means.append(np.mean(W[:, :, k]))
        output_stds.append(np.std(W[:, :, k]))
    
    x = np.arange(len(output_means))
    axes[0, 1].bar(x, output_means, alpha=0.7, label='Mean')
    axes[0, 1].set_title('Mean Weight per Output', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Output Dimension')
    axes[0, 1].set_ylabel('Mean Weight')
    axes[0, 1].grid(True, alpha=0.3)
    
    ax_twin = axes[0, 1].twinx()
    ax_twin.plot(x, output_stds, 'r-', linewidth=2, label='Std')
    ax_twin.set_ylabel('Std Dev', color='r')
    ax_twin.tick_params(axis='y', labelcolor='r')
    
    # 3. Weight heatmap for first output
    im1 = axes[1, 0].imshow(W[:, :, 0], cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('Weight Matrix (First Output)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Input Index')
    axes[1, 0].set_ylabel('Input Index')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # 4. Weight structure for all outputs
    weight_norms = [np.linalg.norm(W[:, :, k]) for k in range(model.output_dim)]
    axes[1, 1].plot(weight_norms, 'b-', linewidth=2)
    axes[1, 1].set_title('Weight Norm per Output', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Output Dimension')
    axes[1, 1].set_ylabel('Frobenius Norm')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_weights_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Model weights analysis saved to model_weights_analysis.png")
    
    plt.show()

def plot_modular_pattern_analysis(model, P=113):
    """Detailed analysis of learned modular patterns"""
    
    W = model.W.detach().cpu().numpy()
    
    # Split into a-b, a-a, b-b interactions
    W_ab = W[:P, P:, :]  # (P, P, output_dim) - a-b interactions
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Analyze patterns for different outputs
    outputs_to_plot = [0, 10, 57, 100, 112]  # Different c values
    
    for idx, output_idx in enumerate(outputs_to_plot):
        if output_idx >= model.output_dim:
            continue
        
        row = idx // 3
        col = idx % 3
        
        if row >= 2:
            break
        
        W_ab_c = W_ab[:, :, output_idx]
        
        # Compute correlation with ideal pattern
        ideal = np.zeros((P, P))
        for a in range(P):
            for b in range(P):
                c = (a + b) % P
                ideal[a, b] = 1.0 if c == output_idx else 0.0
        
        # Plot learned pattern
        im = axes[row, col].imshow(W_ab_c, cmap='RdBu_r', aspect='auto')
        axes[row, col].set_title(f'Learned Pattern for c={output_idx}', fontsize=10)
        axes[row, col].set_xlabel('b')
        axes[row, col].set_ylabel('a')
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    plt.savefig('modular_patterns_detail.png', dpi=300, bbox_inches='tight')
    print("✓ Modular pattern details saved to modular_patterns_detail.png")
    
    plt.show()

def create_summary_report(log_file='training_log.json', output_file='summary_report.txt'):
    """Create a text summary of training results"""
    
    log_data = load_training_log(log_file)
    
    if not log_data:
        print(f"No log data found in {log_file}")
        return
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BILINEAR LAYER TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Training statistics
        f.write("TRAINING STATISTICS:\n")
        f.write(f"  Total epochs: {len(log_data)}\n")
        f.write(f"  Start loss: {log_data[0]['train_loss']:.4f}\n")
        f.write(f"  End loss: {log_data[-1]['train_loss']:.4f}\n")
        f.write(f"  Start accuracy: {log_data[0]['val_accuracy']:.2f}%\n")
        f.write(f"  End accuracy: {log_data[-1]['val_accuracy']:.2f}%\n")
        
        # Best performance
        best_acc = max(entry['val_accuracy'] for entry in log_data)
        best_epoch = next(i for i, entry in enumerate(log_data) if entry['val_accuracy'] == best_acc) + 1
        f.write(f"  Best accuracy: {best_acc:.2f}% (epoch {best_epoch})\n")
        
        # Timing
        total_time = log_data[-1]['total_time']
        f.write(f"  Total training time: {total_time/60:.2f} minutes\n")
        avg_epoch_time = np.mean([entry['epoch_time'] for entry in log_data])
        f.write(f"  Average epoch time: {avg_epoch_time:.2f} seconds\n")
        
        # Early stopping info
        if 'patience_counter' in log_data[-1]:
            f.write(f"  Patience counter: {log_data[-1]['patience_counter']}/10\n")
        
        f.write("\n" + "="*60 + "\n\n")
        
        # Recent performance (last 5 epochs)
        f.write("RECENT PERFORMANCE (Last 5 Epochs):\n")
        for entry in log_data[-5:]:
            f.write(f"  Epoch {entry['epoch']}: Loss={entry['train_loss']:.4f}, "
                   f"Accuracy={entry['val_accuracy']:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
    
    print(f"✓ Summary report saved to {output_file}")

def main():
    """Main visualization function"""
    
    parser = argparse.ArgumentParser(description='Visualize bilinear layer training results')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--log', type=str, default='training_log.json',
                       help='Path to training log file')
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualizations')
    
    args = parser.parse_args()
    
    print("Bilinear Layer Results Visualization")
    print("="*60)
    
    # 1. Plot training curves if log exists
    if os.path.exists(args.log):
        print("\n1. Plotting training curves...")
        plot_training_curves(args.log)
    
    # 2. Load model and create visualizations
    if os.path.exists(args.checkpoint):
        print("\n2. Loading model...")
        
        # Create model with same dimensions as training
        P = 113
        input_dim = 2 * P
        output_dim = P
        
        model = BilinearLayer(input_dim, output_dim)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"   Model loaded from {args.checkpoint}")
        print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        
        # 3. Model weight analysis
        print("\n3. Analyzing model weights...")
        plot_model_weights(model, P)
        
        # 4. Interaction matrices
        print("\n4. Computing interaction matrices...")
        interaction_matrices = compute_interaction_matrices(model, P)
        plot_interaction_matrices(interaction_matrices, P)
        
        # 5. Tensor SVD analysis
        print("\n5. Computing tensor SVD...")
        svd_results = compute_tensor_svd(model, P)
        plot_tensor_svd_analysis(svd_results, P)
        
        # 6. Modular structure analysis
        print("\n6. Analyzing modular structure...")
        analyze_modular_structure(model, P)
        
        # 7. Detailed modular patterns
        print("\n7. Analyzing modular patterns in detail...")
        plot_modular_pattern_analysis(model, P)
    
    # 8. Generate summary report
    if os.path.exists(args.log):
        print("\n8. Generating summary report...")
        create_summary_report(args.log)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - model_weights_analysis.png")
    print("  - interaction_matrices.png")
    print("  - tensor_svd_analysis.png")
    print("  - modular_structure_analysis.png")
    print("  - modular_patterns_detail.png")
    print("  - summary_report.txt")
    print()

if __name__ == "__main__":
    main()

