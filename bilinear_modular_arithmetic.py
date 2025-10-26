"""
Bilinear Layer for Modular Arithmetic with Tensor Network Analysis

This implementation trains a bilinear layer on modular arithmetic (mod 113)
and analyzes the resulting tensor network properties including:
1. Interaction matrices
2. 3rd-order tensor SVD and top eigenvectors

Based on:
- LessWrong post on interpreting modular addition in MLPs
- Bilinear layer paper: https://arxiv.org/pdf/2410.08417
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import warnings
import time
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ModularArithmeticDataset(Dataset):
    """Dataset for modular arithmetic a + b = c (mod P)"""
    
    def __init__(self, P=113, num_samples=10000):
        self.P = P
        self.num_samples = num_samples
        
        # Generate random pairs (a, b) and compute c = (a + b) mod P
        a = torch.randint(0, P, (num_samples,))
        b = torch.randint(0, P, (num_samples,))
        c = (a + b) % P
        
        # Convert to one-hot vectors (optimized version)
        self.inputs = torch.zeros(num_samples, 2 * P)  # [a_onehot, b_onehot]
        self.targets = torch.zeros(num_samples, P)     # c_onehot
        
        # Use advanced indexing for speed
        indices_a = torch.arange(num_samples)
        indices_b = torch.arange(num_samples)
        indices_c = torch.arange(num_samples)
        
        self.inputs[indices_a, a] = 1.0
        self.inputs[indices_b, P + b] = 1.0
        self.targets[indices_c, c] = 1.0
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class BilinearLayer(nn.Module):
    """
    Bilinear layer: y = x^T W x + b
    where W is a 3rd-order tensor and x is the input vector
    """
    
    def __init__(self, input_dim, output_dim):
        super(BilinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize the 3rd-order tensor W with proper scaling
        self.W = nn.Parameter(torch.randn(input_dim, input_dim, output_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        """
        Forward pass: y = x^T W x + b
        x: (batch_size, input_dim)
        W: (input_dim, input_dim, output_dim)
        
        Vectorized computation using einsum for efficiency:
        x^T W[:,:,k] x = sum_i sum_j x_i * W_{i,j,k} * x_j
        """
        # Vectorized computation: bi=batch_input, ijk=input_input_output, bj=batch_input
        # Result: bk = batch_output
        output = torch.einsum('bi,ijk,bj->bk', x, self.W, x)
        
        return output + self.bias

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=1e-4, patience=10, checkpoint_dir='.', save_every=5, start_epoch=0, 
                prev_train_losses=None, prev_val_accuracies=None, prev_best_val_acc=None, prev_patience_counter=None,
                resume_from_checkpoint_path=None):
    """Train the bilinear model with weight decay and early stopping
    
    Args:
        start_epoch: Starting epoch number (for resuming training)
        prev_train_losses: Previous training losses to append to (for resuming)
        prev_val_accuracies: Previous validation accuracies to append to (for resuming)
        prev_best_val_acc: Previous best validation accuracy (for resuming)
        prev_patience_counter: Previous patience counter value (for resuming)
        resume_from_checkpoint_path: Path to checkpoint to resume optimizer state from
    """
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(checkpoint_dir, 'training_log.json')
    # Try to load existing log data
    if start_epoch > 0 and os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []
    
    # Fix: CrossEntropyLoss expects class indices, but we have one-hot vectors
    # Convert one-hot to class indices
    def one_hot_to_indices(batch_y):
        return torch.argmax(batch_y, dim=1)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Restore optimizer state if resuming
    if resume_from_checkpoint_path and os.path.exists(resume_from_checkpoint_path):
        checkpoint = torch.load(resume_from_checkpoint_path)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Restored optimizer state from {resume_from_checkpoint_path}")
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
    )
    
    train_losses = prev_train_losses if prev_train_losses is not None else []
    val_accuracies = prev_val_accuracies if prev_val_accuracies is not None else []
    
    # Early stopping variables
    best_val_acc = prev_best_val_acc if prev_best_val_acc is not None else 0.0
    patience_counter = prev_patience_counter if prev_patience_counter is not None else 0
    best_model_state = None
    
    print(f"Starting training with {len(train_loader)} batches per epoch...")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Saving checkpoint every {save_every} epochs")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch + 1}")
    start_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.time()
        # Training
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{epochs} - Training...")
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_x)
            targets = one_hot_to_indices(batch_y)  # Convert one-hot to class indices
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        print(f"Epoch {epoch+1}/{epochs} - Validation...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                targets = one_hot_to_indices(batch_y)  # Convert one-hot to class indices
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Print validation progress every 5 batches
                if batch_idx % 5 == 0:
                    current_acc = 100 * correct / total if total > 0 else 0
                    print(f"  Val batch {batch_idx+1}/{len(val_loader)} - Acc: {current_acc:.2f}%")
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(accuracy)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  Learning rate reduced to {new_lr:.6f}")
        
        # Early stopping logic
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  New best validation accuracy: {accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs")
        
        # Print epoch summary every epoch (not just every 10)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        avg_epoch_time = total_time / (epoch + 1)
        remaining_time = avg_epoch_time * (epochs - epoch - 1)
        
        print(f'Epoch {epoch+1}/{epochs} COMPLETE - Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
        print(f'  Epoch time: {epoch_time:.1f}s, Total time: {total_time/60:.1f}m, Est. remaining: {remaining_time/60:.1f}m')
        
        # Log results
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_accuracy': accuracy,
            'best_val_accuracy': best_val_acc,
            'patience_counter': patience_counter,
            'epoch_time': epoch_time,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
        log_data.append(epoch_log)
        
        # Save log file after every epoch
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter,
        }, checkpoint_path)
        
        # Save best model separately
        if accuracy == best_val_acc and patience_counter == 0:
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'best_val_acc': best_val_acc,
            }, best_model_path)
        
        # Clean up old checkpoints (keep only last 3)
        if epoch >= save_every:
            old_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch-save_every+1}.pt')
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            print(f"Restoring best model weights...")
            model.load_state_dict(best_model_state)
            break
        
        # Also print every 5 epochs for less verbose output
        if epoch % 5 == 0 and epoch > 0:
            print(f"Progress: {epoch+1}/{epochs} epochs ({100*(epoch+1)/epochs:.1f}%)")
    
    return train_losses, val_accuracies

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load a checkpoint and optionally restore optimizer state"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def load_training_log(log_file):
    """Load training log from JSON file"""
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    return []

def resume_training(model, train_loader, val_loader, checkpoint_path, epochs=100, lr=0.001, weight_decay=1e-4, patience=10, checkpoint_dir='checkpoints', save_every=5):
    """Resume training from a checkpoint"""
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path, model)
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint.get('train_losses', [])
    val_accuracies = checkpoint.get('val_accuracies', [])
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous best val accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
    
    # Continue training from checkpoint
    return train_model(
        model, train_loader, val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every
    )

def compute_interaction_matrices(model, P=113):
    """
    Compute interaction matrices for the bilinear layer
    Based on the formula from the bilinear layer paper (top of page 3)
    """
    W = model.W.detach().cpu().numpy()  # (input_dim, input_dim, output_dim)
    
    # For modular arithmetic, we have two input parts: a and b
    # The interaction matrix shows how a and b interact
    interaction_matrices = {}
    
    for k in range(min(10, model.output_dim)):  # Analyze first 10 output dimensions
        W_k = W[:, :, k]  # (2*P, 2*P)
        
        # Split into blocks: W_aa, W_ab, W_ba, W_bb
        W_aa = W_k[:P, :P]      # a-a interactions
        W_ab = W_k[:P, P:]      # a-b interactions  
        W_ba = W_k[P:, :P]      # b-a interactions
        W_bb = W_k[P:, P:]      # b-b interactions
        
        interaction_matrices[k] = {
            'W_aa': W_aa,
            'W_ab': W_ab,
            'W_ba': W_ba,
            'W_bb': W_bb,
            'cross_interaction': W_ab + W_ba.T  # Symmetric cross-interaction
        }
    
    return interaction_matrices

def plot_interaction_matrices(interaction_matrices, P=113, save_path='interaction_matrices.png'):
    """Plot the interaction matrices"""
    
    n_matrices = len(interaction_matrices)
    fig, axes = plt.subplots(2, n_matrices, figsize=(4*n_matrices, 8))
    
    if n_matrices == 1:
        axes = axes.reshape(2, 1)
    
    for i, (k, matrices) in enumerate(interaction_matrices.items()):
        # Plot cross-interaction matrix (a-b interactions)
        im1 = axes[0, i].imshow(matrices['cross_interaction'], cmap='RdBu_r', aspect='auto')
        axes[0, i].set_title(f'Cross-Interaction Matrix (k={k})')
        axes[0, i].set_xlabel('Input b')
        axes[0, i].set_ylabel('Input a')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Plot diagonal interactions (a-a and b-b)
        diagonal_interaction = np.zeros((P, P))
        diagonal_interaction += np.diag(np.diag(matrices['W_aa']))  # a-a diagonal
        diagonal_interaction += np.diag(np.diag(matrices['W_bb']))  # b-b diagonal
        
        im2 = axes[1, i].imshow(diagonal_interaction, cmap='RdBu_r', aspect='auto')
        axes[1, i].set_title(f'Diagonal Interactions (k={k})')
        axes[1, i].set_xlabel('Index')
        axes[1, i].set_ylabel('Index')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_tensor_svd(model, P=113):
    """
    Compute SVD of the 3rd-order tensor and analyze top eigenvectors
    Based on Section 3.3 of the bilinear layer paper
    """
    W = model.W.detach().cpu().numpy()  # (2*P, 2*P, output_dim)
    
    # Reshape the tensor for SVD analysis
    # We'll analyze the tensor by unfolding it along different modes
    
    # Mode-1 unfolding: (2*P) x (2*P * output_dim)
    W_mode1 = W.reshape(2*P, -1)
    
    # Mode-2 unfolding: (2*P) x (2*P * output_dim) 
    W_mode2 = W.transpose(1, 0, 2).reshape(2*P, -1)
    
    # Mode-3 unfolding: (output_dim) x (2*P * 2*P)
    W_mode3 = W.transpose(2, 0, 1).reshape(model.output_dim, -1)
    
    # Compute SVD for each mode
    svd_results = {}
    
    for mode, W_unfolded in enumerate([W_mode1, W_mode2, W_mode3], 1):
        U, S, Vt = np.linalg.svd(W_unfolded, full_matrices=False)
        svd_results[f'mode_{mode}'] = {
            'U': U,
            'S': S,
            'Vt': Vt,
            'W_unfolded': W_unfolded
        }
    
    return svd_results

def plot_tensor_svd_analysis(svd_results, P=113, save_path='tensor_svd_analysis.png'):
    """Plot the SVD analysis of the 3rd-order tensor"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (mode_name, svd_data) in enumerate(svd_results.items()):
        U, S, Vt = svd_data['U'], svd_data['S'], svd_data['Vt']
        
        # Plot singular values
        axes[0, i].semilogy(S[:20], 'o-')
        axes[0, i].set_title(f'Singular Values - {mode_name}')
        axes[0, i].set_xlabel('Index')
        axes[0, i].set_ylabel('Singular Value')
        axes[0, i].grid(True)
        
        # Plot top eigenvectors
        if mode_name in ['mode_1', 'mode_2']:
            # For input modes, show the structure
            top_eigenvec = U[:, 0]
            if len(top_eigenvec) == 2*P:
                # Split into a and b components
                a_component = top_eigenvec[:P]
                b_component = top_eigenvec[P:]
                
                axes[1, i].plot(a_component, 'b-', label='a component', alpha=0.7)
                axes[1, i].plot(b_component, 'r-', label='b component', alpha=0.7)
                axes[1, i].set_title(f'Top Eigenvector - {mode_name}')
                axes[1, i].set_xlabel('Index')
                axes[1, i].set_ylabel('Value')
                axes[1, i].legend()
                axes[1, i].grid(True)
        else:
            # For output mode, just plot the vector
            axes[1, i].plot(U[:, 0], 'g-')
            axes[1, i].set_title(f'Top Eigenvector - {mode_name}')
            axes[1, i].set_xlabel('Output Index')
            axes[1, i].set_ylabel('Value')
            axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_modular_structure(model, P=113):
    """Analyze the learned modular arithmetic structure"""
    
    W = model.W.detach().cpu().numpy()
    
    # Look for periodic patterns in the learned weights
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Analyze the cross-interaction matrix for the first output
    W_ab = W[:P, P:, 0]  # a-b interactions for first output
    
    # Plot the interaction matrix
    im1 = axes[0, 0].imshow(W_ab, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Learned a-b Interaction Pattern')
    axes[0, 0].set_xlabel('Input b')
    axes[0, 0].set_ylabel('Input a')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Check for modular structure: (a + b) mod P
    # The ideal pattern should be strong on the diagonal where a + b = c
    ideal_pattern = np.zeros((P, P))
    for a in range(P):
        for b in range(P):
            c = (a + b) % P
            ideal_pattern[a, b] = 1.0 if c == 0 else 0.0  # Focus on c=0 case
    
    im2 = axes[0, 1].imshow(ideal_pattern, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Ideal Pattern (c=0)')
    axes[0, 1].set_xlabel('Input b')
    axes[0, 1].set_ylabel('Input a')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Analyze learned patterns for different output classes
    learned_patterns = []
    for c in range(min(5, P)):  # Analyze first 5 output classes
        W_ab_c = W[:P, P:, c]
        learned_patterns.append(W_ab_c)
    
    # Plot learned patterns
    for i, pattern in enumerate(learned_patterns):
        row = 1
        col = i % 2
        if i >= 2:
            row = 1
            col = i - 2
        
        if row < 2 and col < 2:
            im = axes[row, col].imshow(pattern, cmap='RdBu_r', aspect='auto')
            axes[row, col].set_title(f'Learned Pattern (c={i})')
            axes[row, col].set_xlabel('Input b')
            axes[row, col].set_ylabel('Input a')
            plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    plt.savefig('modular_structure_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_from_log(log_file='checkpoints/training_log.json', output_file='training_curves_from_log.png'):
    """Plot training curves from saved log file"""
    log_data = load_training_log(log_file)
    
    if not log_data:
        print(f"No log data found in {log_file}")
        return
    
    epochs = [entry['epoch'] for entry in log_data]
    train_losses = [entry['train_loss'] for entry in log_data]
    val_accuracies = [entry['val_accuracy'] for entry in log_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(epochs, val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {output_file}")
    print(f"Training progress: {len(log_data)} epochs completed")

def main():
    """Main training and analysis pipeline"""
    
    print("Starting Bilinear Layer Training on Modular Arithmetic")
    print("=" * 60)
    
    # Parameters
    P = 113  # Modulus
    input_dim = 2 * P  # One-hot for a and b
    output_dim = P     # One-hot for c
    batch_size = 128  # Larger batch for speed
    epochs = 100  # Allow more training - early stopping will prevent overfitting
    lr = 0.001
    weight_decay = 1e-4
    
    # Create datasets - with faster training, we can afford more data
    # With 113^2 = 12,769 possible inputs, more samples = better coverage
    print("Creating datasets...")
    import time
    start = time.time()
    train_dataset = ModularArithmeticDataset(P=P, num_samples=10000)  # Better coverage for learning
    print(f"Train dataset created in {time.time()-start:.2f}s")
    
    start = time.time()
    val_dataset = ModularArithmeticDataset(P=P, num_samples=2000)  # Larger validation set
    print(f"Val dataset created in {time.time()-start:.2f}s")
    
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    print("Initializing bilinear model...")
    model = BilinearLayer(input_dim, output_dim)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    
    # Train model
    print(f"\nTraining for {epochs} epochs (will stop early if converged)...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        epochs=epochs, lr=lr, weight_decay=weight_decay, patience=7,
        checkpoint_dir='.', save_every=5
    )
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal validation accuracy: {val_accuracies[-1]:.2f}%")
    
    # Analysis
    print("\nAnalyzing learned representations...")
    
    # 1. Interaction matrices
    print("Computing interaction matrices...")
    interaction_matrices = compute_interaction_matrices(model, P)
    plot_interaction_matrices(interaction_matrices, P)
    
    # 2. Tensor SVD analysis
    print("Computing tensor SVD...")
    svd_results = compute_tensor_svd(model, P)
    plot_tensor_svd_analysis(svd_results, P)
    
    # 3. Modular structure analysis
    print("Analyzing modular structure...")
    analyze_modular_structure(model, P)
    
    print("\nAnalysis complete! Check the generated plots:")
    print("- interaction_matrices.png: Shows learned interaction patterns")
    print("- tensor_svd_analysis.png: Shows SVD analysis of the 3rd-order tensor")
    print("- modular_structure_analysis.png: Shows learned modular arithmetic patterns")
    print("- training_curves.png: Shows training progress")

if __name__ == "__main__":
    main()
