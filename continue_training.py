"""Continue training from a checkpoint"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bilinear_modular_arithmetic import BilinearLayer, ModularArithmeticDataset, train_model

def main():
    # Parameters (same as original)
    P = 113
    input_dim = 2 * P
    output_dim = P
    batch_size = 128
    
    print("Loading checkpoint and resuming training...")
    print("=" * 60)
    
    # Load the checkpoint
    checkpoint_path = 'checkpoint_epoch_100.pt'
    checkpoint = torch.load(checkpoint_path)
    
    start_epoch = checkpoint['epoch']  # Should be 100
    print(f"Loaded checkpoint from epoch {start_epoch}")
    print(f"Best validation accuracy so far: {checkpoint.get('best_val_acc', 0):.2f}%")
    
    # Create the model
    model = BilinearLayer(input_dim, output_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Creating datasets...")
    train_dataset = ModularArithmeticDataset(P=P, num_samples=10000)
    val_dataset = ModularArithmeticDataset(P=P, num_samples=2000)
    
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Continue training for another 100 epochs
    # Start from epoch 101 (start_epoch + 1), train until 200
    print(f"\nContinuing training from epoch {start_epoch + 1} for another 100 epochs...")
    
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader,
        epochs=100,  # Train for 100 more epochs
        start_epoch=start_epoch,  # Start from epoch 100, will train epochs 100-199
        lr=0.001,
        weight_decay=1e-4,
        patience=15,  # Increase patience for longer training
        checkpoint_dir='.',
        save_every=5,
        prev_train_losses=checkpoint.get('train_losses', []),  # Continue from previous history
        prev_val_accuracies=checkpoint.get('val_accuracies', []),
        prev_best_val_acc=checkpoint.get('best_val_acc', 0.0),
        prev_patience_counter=checkpoint.get('patience_counter', 0),
        resume_from_checkpoint_path=checkpoint_path  # Restore optimizer state
    )
    
    print("\nTraining complete!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()

