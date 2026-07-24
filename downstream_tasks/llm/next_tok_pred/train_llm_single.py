import os
import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from core.single.pikv_moe import PiKVMoE
from core.single.normal_moe import StandardMoE
from core.distributed.distributed_pikv import DistributedPiKVManager
from llm_config import llm_config as config
import math

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize text
        tokens = tokenizer.encode(text)
        
        # Create sequences with fixed length
        self.sequences = []
        for i in range(0, len(tokens) - max_length, max_length):
            sequence = tokens[i:i + max_length]
            if len(sequence) == max_length:
                # Split into input and target
                input_sequence = sequence[:-1]  # All tokens except last
                target_sequence = sequence[1:]  # All tokens except first
                self.sequences.append((input_sequence, target_sequence))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_sequence, target_sequence = self.sequences[idx]
        return torch.tensor(input_sequence), torch.tensor(target_sequence)

def get_lr(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

def train_single_model(model_type='pikv'):
    # Set device
    device = config['device']
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create dataset and dataloader with smaller batch size
    from data.dataloader import resolve_data_path
    try:
        text_path = resolve_data_path("test.txt", "train.txt")
    except FileNotFoundError:
        text_path = "data/test.txt"
    dataset = TextDataset(text_path, tokenizer, max_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Smaller batch size
    
    # Initialize model
    if model_type == 'pikv':
        model = PiKVMoE(rank=4, alpha=1.0).to(device)
    else:
        model = StandardMoE().to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Smaller learning rate
        weight_decay=0.01
    )
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(5):  # Fewer epochs for testing
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/5')
        for step, (input_ids, target_ids) in enumerate(progress_bar):
            # Move data to device
            input_ids = input_ids.to(device)  # [batch_size, seq_len-1]
            target_ids = target_ids.to(device)  # [batch_size, seq_len-1]
            
            # Forward pass
            logits = model(input_ids)  # [batch_size, seq_len-1, vocab_size]
            
            # Calculate loss
            # Reshape logits to [batch_size * seq_len-1, vocab_size]
            # and targets to [batch_size * seq_len-1]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),  # [batch_size * seq_len-1, vocab_size]
                target_ids.reshape(-1)  # [batch_size * seq_len-1]
            )
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})
    
    # Save model checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 5,
        'loss': total_loss / len(dataloader)
    }
    
    # Save checkpoint using the model's save method if available
    if hasattr(model, 'save_checkpoint'):
        model.save_checkpoint(f'checkpoints/{model_type}_final.pt')
    else:
        torch.save(checkpoint, f'checkpoints/{model_type}_final.pt')
    
    print(f"Model saved to checkpoints/{model_type}_final.pt")
    
    return model

def train_distributed_model():
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create dataset and dataloader
    dataset = TextDataset('data/train.txt', tokenizer, config['max_seq_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize distributed manager
    pikv_manager = DistributedPiKVManager()
    
    # Training loop
    for epoch in range(config['epochs']):
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for step, (input_ids, target_ids) in enumerate(progress_bar):
            # Move data to device
            input_ids = input_ids.to(config['device'])
            target_ids = target_ids.to(config['device'])
            
            # Train step
            loss = pikv_manager.train_step(input_ids, target_ids)
            
            # Update progress
            total_loss += loss
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})
            
            # Save checkpoint
            if (step + epoch * len(dataloader)) % config['save_steps'] == 0:
                save_path = f'checkpoints/distributed_epoch{epoch}_step{step}.pt'
                os.makedirs('checkpoints', exist_ok=True)
                pikv_manager.save_checkpoint(save_path)
    
    # Save final model
    save_path = 'checkpoints/distributed_final.pt'
    pikv_manager.save_checkpoint(save_path)

def evaluate_model(model_type='pikv', distributed=False):
    # Set device
    device = config['device']
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create dataset and dataloader
    dataset = TextDataset('data/test.txt', tokenizer, max_length=32)  # Use same max_length as training
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)  # Use same batch_size as training
    
    if distributed:
        # Initialize distributed manager
        pikv_manager = DistributedPiKVManager()
        pikv_manager.load_checkpoint(f'checkpoints/distributed_final.pt')
        model = pikv_manager.model
    else:
        # Initialize model
        if model_type == 'pikv':
            model = PiKVMoE(rank=4, alpha=1.0).to(device)
        else:
            model = StandardMoE().to(device)
        
        # Load checkpoint
        checkpoint_path = f'checkpoints/{model_type}_final.pt'
        if os.path.exists(checkpoint_path):
            try:
                # Try loading with model's load method first
                if hasattr(model, 'load_checkpoint'):
                    model.load_checkpoint(checkpoint_path)
                else:
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # If it's a direct state dict
                        model.load_state_dict(checkpoint)
                print(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
                print("Using randomly initialized model for evaluation")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using randomly initialized model.")
        
        model.eval()
    
    # Initialize metrics
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    
    # Evaluation loop
    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc='Evaluating'):
            # Move data to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == target_ids).sum().item()
            
            # Update metrics
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += target_ids.numel()
            correct_tokens += correct
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataset)
    accuracy = correct_tokens / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

def main():
    parser = argparse.ArgumentParser(description='Train PiKV models for next token prediction')
    parser.add_argument('--model_type', type=str, choices=['single', 'distributed'], required=True,
                      help='Type of model to train')
    parser.add_argument('--use_lora', action='store_true',
                      help='Use LoRA for fine-tuning')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    args = parser.parse_args()
    
    # Load and preprocess data
    # TODO: Implement data loading
    train_data = []
    test_data = []
    
    if args.model_type == 'single':
        # Train both PiKV and Standard MoE
        pikv_model = PiKVMoE(rank=4, alpha=1.0)
        standard_model = StandardMoE()
        
        print("Training PiKV model...")
        pikv_model = train_single_model('pikv')
        
        print("\nTraining Standard MoE model...")
        standard_model = train_single_model('standard')
        
        # Evaluate both models
        print("\nEvaluating models...")
        pikv_metrics = evaluate_model('pikv')
        standard_metrics = evaluate_model('standard')
        
        print(f"\nPiKV Model - Loss: {pikv_metrics['loss']:.4f}, Accuracy: {pikv_metrics['accuracy']:.2f}%")
        print(f"Standard MoE Model - Loss: {standard_metrics['loss']:.4f}, Accuracy: {standard_metrics['accuracy']:.2f}%")
    
    else:  # distributed
        print("Training distributed PiKV model...")
        manager = train_distributed_model()
        
        # Evaluate distributed model
        print("\nEvaluating distributed model...")
        distributed_metrics = evaluate_model(distributed=True)
        
        print(f"\nDistributed PiKV Model - Loss: {distributed_metrics['loss']:.4f}, Accuracy: {distributed_metrics['accuracy']:.2f}%")

if __name__ == '__main__':
    main() 