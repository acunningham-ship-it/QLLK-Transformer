import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearLatentKernel(nn.Module):
    """
    O(N) Linear Attention Kernel using associative state updates.
    Optimized for torch.compile and cross-device performance.
    Uses Gated Linear Unit (GLU) logic for enhanced feature selection.
    """
    def __init__(self, dim):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.gate_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: (B, L, D)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Associative state update (linearized attention)
        # S_t = S_{t-1} + k_t * v_t
        # Using cumsum for efficient O(N) parallel execution
        kv_state = torch.cumsum(k * v, dim=1)
        
        # Gating for noise suppression and focus
        g = torch.sigmoid(self.gate_proj(x))
        
        return q * kv_state * g

class QLLKTransformer(nn.Module):
    """
    Quantum-Leap Latent Kernel (QLLK) Transformer.
    Optimized for low-end and high-end hardware.
    """
    def __init__(self, dim=256, n_layers=4, patch_size=8):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        self.byte_embedding = nn.Embedding(256, dim)
        
        # Hardware-Aware Patching: Conv1d is faster and more expressive than mean-pooling
        # on both mobile CPUs (SIMD) and modern GPUs.
        self.patch_encoder = nn.Conv1d(dim, dim, kernel_size=patch_size, stride=patch_size)
        
        # Feature Hashing Shortcut (Sparse Pattern Recognition)
        self.hash_proj = nn.Parameter(torch.randn(dim, dim) * 0.02)
        
        # Optimized Kernel Layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'kernel': LinearLatentKernel(dim),
                'norm1': nn.LayerNorm(dim),
                'mlp': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
                'norm2': nn.LayerNorm(dim)
            }) for _ in range(n_layers)
        ])
        
        # Faster Unfolding with Transposed Conv
        self.patch_decoder = nn.ConvTranspose1d(dim, dim, kernel_size=patch_size, stride=patch_size)
        self.head = nn.Linear(dim, 256)

    def forward(self, bytes_in):
        # bytes_in: (B, S)
        x = self.byte_embedding(bytes_in) # (B, S, D)
        
        # 1. Hardware-Aware Patching
        # (B, S, D) -> (B, D, S)
        x = x.transpose(1, 2)
        x = self.patch_encoder(x) # (B, D, P)
        # (B, D, P) -> (B, P, D)
        x = x.transpose(1, 2)
        
        # 2. Sub-Byte Hashing (Pattern Lookup Shortcut)
        hashed = torch.matmul(x, self.hash_proj)
        x = x + torch.tanh(hashed)
        
        # 3. Linear Latent Kernel Processing
        for layer in self.layers:
            # Residual Connection + Kernel
            x = x + layer['kernel'](layer['norm1'](x))
            # Residual Connection + MLP
            x = x + layer['mlp'](layer['norm2'](x))
            
        # 4. Local Decoding (Unfolding)
        # (B, P, D) -> (B, D, P)
        x = x.transpose(1, 2)
        x = self.patch_decoder(x) # (B, D, S)
        # (B, D, S) -> (B, S, D)
        x = x.transpose(1, 2)
        
        logits = self.head(x) # (B, S, 256)
        
        return logits
