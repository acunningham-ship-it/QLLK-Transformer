import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyModel(nn.Module):
    """Small transformer to predict next-byte entropy."""
    def __init__(self, vocab_size=256, dim=128, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def get_entropy(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy

class MELTransformer(nn.Module):
    """Memory-Efficient Latent Folding (MELF) Transformer."""
    def __init__(self, dim=256, n_layers=6, n_heads=8, patch_size=8, fold_factor=4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.fold_factor = fold_factor
        
        self.byte_embedding = nn.Embedding(256, dim)
        
        # Folding Layer: Projects concatenated folded latents back to model dim
        self.fold_proj = nn.Linear(dim * fold_factor, dim)
        
        # Core Transformer (runs on folded sequence)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True),
            num_layers=n_layers
        )
        
        # Unfolding Layer: Expands folded latents back to patch sequence
        self.unfold_proj = nn.Linear(dim, dim * fold_factor)
        
        # Local Decoder
        self.decoder_head = nn.Linear(dim, 256 * patch_size)

    def forward(self, bytes_in):
        batch_size, seq_len = bytes_in.shape
        embeddings = self.byte_embedding(bytes_in) # (B, S, D)
        
        # 1. Patching
        num_patches = seq_len // self.patch_size
        patches = embeddings.view(batch_size, num_patches, self.patch_size, self.dim)
        patch_latents = patches.mean(dim=2) # (B, P, D)
        
        # 2. Sequence Folding
        # Fold sequence dimension into channel dimension
        # (B, P, D) -> (B, P/K, K, D) -> (B, P/K, K*D)
        num_folded = num_patches // self.fold_factor
        folded = patch_latents.view(batch_size, num_folded, self.fold_factor, self.dim)
        folded = folded.reshape(batch_size, num_folded, self.fold_factor * self.dim)
        
        # Project to model dimension
        folded_latents = self.fold_proj(folded)
        
        # 3. Process Folded Sequence (Much faster!)
        latent_out = self.transformer(folded_latents) # (B, P/K, D)
        
        # 4. Sequence Unfolding
        # (B, P/K, D) -> (B, P/K, K*D) -> (B, P, D)
        unfolded = self.unfold_proj(latent_out)
        unfolded_latents = unfolded.view(batch_size, num_patches, self.dim)
        
        # 5. Local Decoding
        logits = self.decoder_head(unfolded_latents) # (B, P, 256 * patch_size)
        logits = logits.view(batch_size, num_patches * self.patch_size, 256)
        
        return logits

class LinearLatentKernel(nn.Module):
    """O(N) Linear Attention Kernel using associative state updates."""
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, L, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = torch.sigmoid(self.gate(x))

        # Exponential moving average / Linear attention approximation
        # S_t = S_{t-1} + k_t^T v_t
        # y_t = q_t S_t
        # For prototype, we use a vectorized version of a simple linear recurrent unit
        # This simulates O(N) complexity by avoiding the L x L matrix.
        
        # Simplified associative scan: 
        # (B, L, D) -> cumulative sum over sequence
        k_v = k * v # (B, L, D)
        kv_state = torch.cumsum(k_v, dim=1) # (B, L, D)
        
        out = q * kv_state * g
        return out

class QLLKTransformer(nn.Module):
    """Quantum-Leap Latent Kernel (QLLK) Transformer."""
    def __init__(self, dim=256, n_layers=4, patch_size=8):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        self.byte_embedding = nn.Embedding(256, dim)
        
        # Feature Hashing layer (Sparse Pattern Recognition)
        self.hash_proj = nn.Parameter(torch.randn(dim, dim) * 0.02)
        
        # Linear Kernel Layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                LinearLatentKernel(dim),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim)
            ) for _ in range(n_layers)
        ])
        
        # Local Decoder
        self.decoder_head = nn.Linear(dim, 256 * patch_size)

    def forward(self, bytes_in):
        batch_size, seq_len = bytes_in.shape
        embeddings = self.byte_embedding(bytes_in) # (B, S, D)
        
        # 1. Patching
        num_patches = seq_len // self.patch_size
        patches = embeddings.view(batch_size, num_patches, self.patch_size, self.dim)
        patch_latents = patches.mean(dim=2) # (B, P, D)
        
        # 2. Sub-Byte Hashing (Pattern Lookup Shortcut)
        # Multiply by a static projection to "hash" the latent space
        hashed = torch.matmul(patch_latents, self.hash_proj)
        x = patch_latents + torch.tanh(hashed)
        
        # 3. Linear Latent Kernel Processing (O(N) Speed!)
        for layer in self.layers:
            # Residual-like application
            x = x + layer[0](x) # Kernel
            x = layer[1:](x)    # Norm + MLP
            
        # 4. Local Decoding
        logits = self.decoder_head(x) # (B, P, 256 * patch_size)
        logits = logits.view(batch_size, num_patches * self.patch_size, 256)
        
        return logits
