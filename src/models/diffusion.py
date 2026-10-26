"""
Denoising Diffusion Probabilistic Model (DDPM) & DDIM for Construction Site Safety Synthesis.

Includes:
1. SinusoidalPositionalEmbeddings: Timestep embeddings for diffusion step t.
2. Time & Class Conditioned U-Net Denoiser (UNetDiffusion):
   - Multi-scale residual convolution blocks.
   - Multi-Head Spatial Self-Attention at bottleneck.
   - Skip connections across resolution levels (128x128 -> 64x64 -> 32x32 -> 16x16).
   - Conditioned on 4 OSHA safety hazard categories.
3. GaussianDiffusion:
   - Forward diffusion process q(x_t | x_0) with cosine / linear variance schedule.
   - Reverse denoising Markov process p_theta(x_{t-1} | x_t) (DDPM).
   - Fast non-Markovian deterministic sampling (DDIM, 20-50 steps).
   - MSE noise residual loss computation: L_simple = E[||epsilon - epsilon_theta(x_t, t, c)||^2].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Sinusoidal Positional Embeddings for Timestep t
# ==============================================================================

class SinusoidalPositionalEmbeddings(nn.Module):
    """
    Computes sinusoidal positional embeddings for diffusion timesteps t in [0, T-1].
    Maps scalar timestep t to a continuous vector of dimension dim.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ==============================================================================
# 2. Residual Block with Time & Class Conditioning
# ==============================================================================

class ResBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU non-linearity, and time/class projection injection.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        # Inject timestep embedding
        t_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + t_proj
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


# ==============================================================================
# 3. Spatial Self-Attention Block for Feature Map Bottleneck
# ==============================================================================

class SpatialSelfAttention(nn.Module):
    """
    Multi-Head Spatial Self-Attention for diffusion feature maps.
    Captures long-range structural dependencies across construction site elements.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape to [B, num_heads, H*W, head_dim]
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        return x + self.proj(out)


# ==============================================================================
# 4. Time & Class Conditioned Diffusion U-Net
# ==============================================================================

class UNetDiffusion(nn.Module):
    """
    U-Net Denoiser Network for 128x128 construction site image synthesis.
    Inputs:
        x: [B, 3, 128, 128] (noisy image at timestep t)
        t: [B] (integer timesteps in [0, T-1])
        c: [B] (optional integer OSHA hazard category in [0, 3])
    Outputs:
        noise_pred: [B, 3, 128, 128] (predicted noise residual epsilon_theta)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_dim: int = 256,
        num_classes: int = 4,
        class_embed_dim: int = 32,
        base_channels: int = 64
    ):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes

        # Time & Class Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_dim // 2),
            nn.Linear(time_dim // 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.class_proj = nn.Linear(class_embed_dim, time_dim)

        # 1. Initial Convolution
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4 # 64, 128, 256
        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # 2. Downsampling Blocks (Encoder)
        # Level 1: 128x128 -> 64x64
        self.down1_block = ResBlock(c1, c1, time_dim)
        self.down1_pool = nn.Conv2d(c1, c1, kernel_size=4, stride=2, padding=1) # 64x64

        # Level 2: 64x64 -> 32x32
        self.down2_block = ResBlock(c1, c2, time_dim)
        self.down2_pool = nn.Conv2d(c2, c2, kernel_size=4, stride=2, padding=1) # 32x32

        # Level 3: 32x32 -> 16x16
        self.down3_block = ResBlock(c2, c3, time_dim)
        self.down3_pool = nn.Conv2d(c3, c3, kernel_size=4, stride=2, padding=1) # 16x16

        # 3. Bottleneck Block (16x16 with Self-Attention)
        self.mid_block1 = ResBlock(c3, c3, time_dim)
        self.mid_attn = SpatialSelfAttention(c3, num_heads=4)
        self.mid_block2 = ResBlock(c3, c3, time_dim)

        # 4. Upsampling Blocks (Decoder with Skip Connections)
        # Level 3 Up: 16x16 -> 32x32
        self.up3_conv = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.up3_block = ResBlock(c2 + c3, c2, time_dim)

        # Level 2 Up: 32x32 -> 64x64
        self.up2_conv = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up2_block = ResBlock(c1 + c2, c1, time_dim)

        # Level 1 Up: 64x64 -> 128x128
        self.up1_conv = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        self.up1_block = ResBlock(c1 + c1, c1, time_dim)

        # 5. Output Projection
        self.final_norm = nn.GroupNorm(8, c1)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        # Combined Time & Class Condition Embedding
        t_emb = self.time_embed(t)
        if c is not None:
            c_emb = self.class_proj(self.class_embed(c))
            time_emb = t_emb + c_emb
        else:
            time_emb = t_emb

        # Initial Conv
        h = self.init_conv(x)

        # Encoder Downsampling
        s1 = self.down1_block(h, time_emb)
        d1 = self.down1_pool(s1)

        s2 = self.down2_block(d1, time_emb)
        d2 = self.down2_pool(s2)

        s3 = self.down3_block(d2, time_emb)
        d3 = self.down3_pool(s3)

        # Bottleneck with Spatial Self-Attention
        m = self.mid_block1(d3, time_emb)
        m = self.mid_attn(m)
        m = self.mid_block2(m, time_emb)

        # Decoder Upsampling with Skips
        u3 = self.up3_conv(m)
        u3 = torch.cat([u3, s3], dim=1)
        u3 = self.up3_block(u3, time_emb)

        u2 = self.up2_conv(u3)
        u2 = torch.cat([u2, s2], dim=1)
        u2 = self.up2_block(u2, time_emb)

        u1 = self.up1_conv(u2)
        u1 = torch.cat([u1, s1], dim=1)
        u1 = self.up1_block(u1, time_emb)

        # Final projection to noise prediction
        out = self.final_conv(self.final_act(self.final_norm(u1)))
        return out


# ==============================================================================
# 5. Gaussian Diffusion Pipeline (DDPM & DDIM Sampling)
# ==============================================================================

class GaussianDiffusion:
    """
    Gaussian Diffusion Framework for Construction Site Image Generation.
    Implements:
    - Cosine / Linear beta schedules.
    - Closed-form forward diffusion q(x_t | x_0).
    - DDPM iterative reverse Markov sampling (Ho et al., 2020).
    - Fast non-Markovian DDIM sampling (Song et al., 2020) with 20-50 steps.
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: torch.device = None
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Beta schedule computation
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=self.device)
        elif beta_schedule == "cosine":
            # Improved cosine schedule (Nichol & Dhariwal, 2021)
            steps = timesteps + 1
            s = 0.008
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=self.device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.999)

        # Precompute diffusion variables
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for reverse process posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extracts appropriate coefficients at timesteps t and reshapes for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion process: Directly samples x_t from x_0 at arbitrary timestep t in closed form:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the simplified variational diffusion training loss:
        L_simple = E_{t, x_0, epsilon} [ ||epsilon - epsilon_theta(x_t, t, c)||^2 ]
        """
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)

        # Predict noise residual
        predicted_noise = self.model(x_noisy, t, c)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, c: torch.Tensor = None) -> torch.Tensor:
        """
        Single reverse Markov denoising step: Computes x_{t-1} from x_t.
        """
        t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        noise_pred = self.model(x, t_batch, c)

        betas_t = self._extract(self.betas, t_batch, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t_batch, x.shape)

        # Model mean mu_theta(x_t, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t_batch, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_ddpm(self, num_samples: int = 4, c: torch.Tensor = None, img_size: int = 128) -> torch.Tensor:
        """
        Full DDPM iterative reverse sampling across all timesteps (T -> 0).
        """
        self.model.eval()
        img = torch.randn(num_samples, 3, img_size, img_size, device=self.device)
        for t in reversed(range(self.timesteps)):
            img = self.p_sample(img, t, c)
        return torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)

    @torch.no_grad()
    def sample_ddim(
        self,
        num_samples: int = 4,
        ddim_steps: int = 25,
        eta: float = 0.0,
        c: torch.Tensor = None,
        img_size: int = 128
    ) -> torch.Tensor:
        """
        Fast non-Markovian DDIM sampling (Song et al., 2020).
        Accelerates reverse sampling by 40x (25 steps instead of 1000 steps).
        """
        self.model.eval()
        step_stride = self.timesteps // ddim_steps
        time_steps = list(range(0, self.timesteps, step_stride))
        time_steps.reverse()

        img = torch.randn(num_samples, 3, img_size, img_size, device=self.device)

        for i, t in enumerate(time_steps):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(img, t_batch, c)

            alpha_bar = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[time_steps[i + 1]] if i < len(time_steps) - 1 else torch.tensor(1.0, device=self.device)

            # Predict x_0
            pred_x0 = (img - torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - eta ** 2) * noise_pred
            random_noise = torch.randn_like(img) if eta > 0.0 else 0.0

            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + eta * random_noise

        return torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)

    @torch.no_grad()
    def sample_ddim_trajectory(
        self,
        num_samples: int = 1,
        ddim_steps: int = 25,
        eta: float = 0.0,
        c: torch.Tensor = None,
        img_size: int = 128
    ):
        """
        Fast DDIM sampling returning both final generated image and intermediate denoising trajectory frames.
        """
        self.model.eval()
        step_stride = self.timesteps // ddim_steps
        time_steps = list(range(0, self.timesteps, step_stride))
        time_steps.reverse()

        img = torch.randn(num_samples, 3, img_size, img_size, device=self.device)
        trajectory = []
        trajectory.append({
            "step": 0,
            "timestep": self.timesteps,
            "tensor": torch.clamp((img.clone() + 1.0) / 2.0, 0.0, 1.0)
        })

        for i, t in enumerate(time_steps):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(img, t_batch, c)

            alpha_bar = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[time_steps[i + 1]] if i < len(time_steps) - 1 else torch.tensor(1.0, device=self.device)

            pred_x0 = (img - torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - eta ** 2) * noise_pred
            random_noise = torch.randn_like(img) if eta > 0.0 else 0.0

            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + eta * random_noise

            trajectory.append({
                "step": i + 1,
                "timestep": t,
                "tensor": torch.clamp((img.clone() + 1.0) / 2.0, 0.0, 1.0)
            })

        final_img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
        return final_img, trajectory
