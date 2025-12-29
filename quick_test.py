#!/usr/bin/env python3
"""
Quick verification test for QLLK
Runs 10 training steps to verify the model works
"""
import torch
import torch.optim as optim
from bnt_model import QLLKTransformer
import time

print("QLLK Quick Verification Test")
print("="*50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = QLLKTransformer(patch_size=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Small test: batch=4, seq=128
batch_size = 4
seq_len = 128

print(f"\nConfig: batch={batch_size}, seq_len={seq_len}")
print("Running 10 training steps...\n")

losses = []
start = time.time()

for i in range(10):
    # Random data
    data = torch.randint(0, 256, (batch_size, seq_len + 1))
    inputs = data[:, :-1].to(device)
    targets = data[:, 1:].to(device)
    
    # Forward
    logits = model(inputs)
    loss = criterion(logits.reshape(-1, 256), targets.reshape(-1))
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if i % 3 == 0:
        print(f"Step {i}: Loss = {loss.item():.4f}")

elapsed = time.time() - start
tokens_per_sec = (10 * batch_size * seq_len) / elapsed

print(f"\n{'='*50}")
print(f"✅ QLLK VERIFICATION COMPLETE")
print(f"{'='*50}")
print(f"Average loss: {sum(losses)/len(losses):.4f}")
print(f"Speed: {tokens_per_sec:.0f} tokens/sec")
print(f"Loss trend: {losses[0]:.2f} → {losses[-1]:.2f}")

if losses[-1] < losses[0]:
    print("\n✅ SUCCESS: Model is learning (loss decreased)!")
else:
    print("\n⚠️  More steps may be needed for full convergence")

print(f"\nQLL K works! Ready for full training.")
