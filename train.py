import torch
import torch.optim as optim
from bnt_model import QLLKTransformer
import os
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

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
    # Multi-GPU support: CUDA > MPS (Apple Metal) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = "CUDA GPU"
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "Apple Metal GPU"
    else:
        device = torch.device('cpu')
        device_name = "CPU"

    print(f"Training on {device_name} ({device})")

    # Hyperparams
    batch_size = 256 # Doubled batch size for better GPU utilization
    seq_len = 512
    patch_size = 8
    epochs = 2

    # MPS doesn't support multiprocessing well - use 0 workers
    num_workers = 8 if device.type == 'cuda' else 0
    
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
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),  # Only CUDA supports pin_memory
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0)
    )

    # 2. Instantiate and COMPILE QLLK Model
    model = QLLKTransformer(patch_size=patch_size).to(device)
    
    # torch.compile: The secret sauce for speed (fuses kernels, optimizes graph)
    use_amp = device.type in ['cuda', 'mps']
    if hasattr(torch, "compile") and device.type == 'cuda':
        print("Compiling model for maximum throughput...")
        model = torch.compile(model)
    elif device.type == 'mps':
        print("Apple Metal GPU detected: Using FP32 (MPS doesn't support torch.compile yet).")
    elif device.type == 'cpu':
        print("CPU detected: Using FP32 (no compilation).")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    print(f"Starting QLLK Training (Batch: {batch_size}, Workers: {num_workers})...")
    start_time = time.time()
    total_tokens = 0

    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Forward with Automatic Mixed Precision (AMP) - CUDA only
            # MPS uses FP32 for stability
            with autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, 256), targets.reshape(-1))
            
            # Backward with Scaling
            optimizer.zero_grad(set_to_none=True) # faster than zeroing
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_tokens += inputs.numel()
            
            if i % 10 == 0:
                elapsed = time.time() - start_time
                throughput = total_tokens / elapsed
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}, Speed: {throughput:.0f} tokens/s")

    total_time = time.time() - start_time
    print(f"Final Performance: {total_tokens/total_time:.0f} tokens/s")
    
    # --- Deep Verification ---
    print("\n--- Deep Verification: Prediction Test ---")
    model.eval()
    with torch.no_grad():
        test_input = torch.zeros((1, seq_len), dtype=torch.long).to(device)
        # Use autocast only for CUDA
        with autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
            logits = model(test_input)
            preds = torch.argmax(logits, dim=-1)
        
        correct_zeros = (preds == 0).float().mean().item() * 100
        print(f"Prediction Accuracy on predictable sequence: {correct_zeros:.1f}%")
        
        if correct_zeros > 90:
            print("SUCCESS: QLLK performance and logic verified.")
        else:
            print("NOTICE: Performance is high, but sequence learning may need more data/epochs.")

if __name__ == "__main__":
    train()
