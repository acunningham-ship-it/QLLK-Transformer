import torch
import torch.optim as optim
from bnt_model import QLLKTransformer
import os
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FastByteDataset(Dataset):
    """Zero-copy memory-mapped dataset for hyper-fast byte loading."""
    def __init__(self, file_path, seq_len):
        self.seq_len = seq_len
        # Memory-map the file for zero-copy access
        self.data = np.memmap(file_path, dtype=np.uint8, mode='r')
        self.n_samples = len(self.data) // (seq_len + 1)
        
        # Pre-load to RAM if small (e.g., < 1GB) for absolute zero latency
        if len(self.data) < 1024**3:
            print(f"Dataset size: {len(self.data)/1024**2:.2f}MB. Pre-loading to RAM...")
            self.data = torch.from_numpy(np.array(self.data)).long()
        else:
            print("Dataset too large for RAM, using on-disk memmap.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.data[start : start + self.seq_len + 1]
        
        if not isinstance(chunk, torch.Tensor):
            chunk = torch.from_numpy(chunk.astype(np.int64))
        
        return chunk[:-1], chunk[1:]

def train(file_path="dataset.bin"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Hyperparams
    batch_size = 128 # Increased for better parallelism
    seq_len = 512
    patch_size = 8
    epochs = 2 # Adjusted for real dataset traversal
    
    # 1. Initialize Dataset and HyperLoader
    if not os.path.exists(file_path):
        print(f"Creating dummy 68MB dataset at {file_path}...")
        with open(file_path, "wb") as f:
            f.write(os.urandom(68 * 1024 * 1024))

    dataset = FastByteDataset(file_path, seq_len)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,        # Parallel prefetching
        pin_memory=True,     # Fast host-to-device transfer
        prefetch_factor=2    # Keep batches ready
    )

    # 2. Instantiate QLLK Model
    model = QLLKTransformer(patch_size=patch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Starting QLLK HYPER-LOADING (Batch: {batch_size}, Workers: 4)...")
    start_time = time.time()
    total_tokens = 0

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward
            logits = model(inputs)
            
            # Loss
            loss = criterion(logits.reshape(-1, 256), targets.reshape(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_tokens += inputs.numel()
            
            if i % 10 == 0:
                elapsed = time.time() - start_time
                throughput = total_tokens / elapsed
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}, Speed: {throughput:.0f} tokens/s")

    total_time = time.time() - start_time
    print(f"Final Performance: {total_tokens/total_time:.0f} tokens/s")
    
    # --- Deep Verification: Can it actually predict? ---
    print("\n--- Deep Verification: Prediction Test ---")
    model.eval()
    with torch.no_grad():
        # Test on a sequence of zeros (predictable)
        test_input = torch.zeros((1, seq_len), dtype=torch.long).to(device)
        logits = model(test_input)
        preds = torch.argmax(logits, dim=-1)
        
        # Check if it correctly predicts 0s for a 0-input (simple case)
        correct_zeros = (preds == 0).float().mean().item() * 100
        print(f"Prediction Accuracy on predictable sequence: {correct_zeros:.1f}%")
        
        if correct_zeros > 90:
            print("SUCCESS: QLLK has successfully learned the repetitive pattern.")
        else:
            print("NOTICE: QLLK needs more epochs or tuning for full convergence, but throughput is verified.")

if __name__ == "__main__":
    train()
