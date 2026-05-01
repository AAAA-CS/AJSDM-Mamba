import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, SelectiveScanFn
except:
    pass
import numpy as np

def compute_spatial_complexity_sobel(x: torch.Tensor, eps: float = 1e-6):
    """原始 Sobel 梯度幅值方法（数值稳定版）"""
    B, C, H, W = x.shape
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    x_gray = x.mean(1, keepdim=True)
    gx = F.conv2d(x_gray, sobel_x, padding=1)
    gy = F.conv2d(x_gray, sobel_y, padding=1)
    mag_sq = gx ** 2 + gy ** 2
    mag_sq = torch.clamp(mag_sq, min=0)
    mag = torch.sqrt(mag_sq + eps)
    mag_min = mag.amin(dim=[2, 3], keepdim=True)
    mag_max = mag.amax(dim=[2, 3], keepdim=True)
    range_mag = mag_max - mag_min
    complexity = torch.where(range_mag > 0, (mag - mag_min) / (range_mag + eps), torch.zeros_like(mag))
    complexity = torch.nan_to_num(complexity, nan=0.0)
    return complexity.squeeze(1)


def adaptive_step_from_complexity(m: torch.Tensor, s_min: int = 1, s_max: int = 5, alpha: float = 5.0):
    """根据空间复杂度计算每个方向的自适应步长（论文公式 9），使用 numpy 确保稳定性"""
    B, H, W = m.shape
    m_np = m.detach().cpu().numpy()
    m_np = np.nan_to_num(m_np, nan=0.5, posinf=1.0, neginf=0.0)

    from scipy.signal import convolve2d
    kernel = np.ones((3, 3)) / 9.0
    local_complexity_np = np.zeros_like(m_np)
    for b in range(B):
        local_complexity_np[b] = convolve2d(m_np[b], kernel, mode='same')

    h_mid, w_mid = H // 2, W // 2
    region_complexities_np = np.zeros((B, 4))
    for b in range(B):
        region_complexities_np[b, 0] = m_np[b, :h_mid, :w_mid].mean()
        region_complexities_np[b, 1] = m_np[b, :h_mid, w_mid:].mean()
        region_complexities_np[b, 2] = m_np[b, h_mid:, :w_mid].mean()
        region_complexities_np[b, 3] = m_np[b, h_mid:, w_mid:].mean()
    region_complexities_np = np.nan_to_num(region_complexities_np, nan=0.5)

    step_sizes = torch.zeros(B, 4, H, W, dtype=torch.long, device=m.device)
    for b in range(B):
        for d in range(4):
            region_comp = region_complexities_np[b, d]
            region_comp = np.clip(region_comp, 0.0, 1.0)
            base_step = s_max - (1.0 / (1.0 + np.exp(-alpha * (region_comp - 0.5)))) * (s_max - s_min)
            base_step = int(round(np.clip(base_step, s_min, s_max)))

            for i in range(H):
                for j in range(W):
                    local_comp = local_complexity_np[b, i, j]
                    local_comp = np.clip(local_comp, 0.0, 1.0)
                    local_adjustment = (1.0 / (1.0 + np.exp(-alpha * (local_comp - 0.5)))) * 2 - 1
                    adjusted_step = base_step + int(round(local_adjustment))
                    adjusted_step = max(s_min, min(s_max, adjusted_step))
                    step_sizes[b, d, i, j] = adjusted_step
    return step_sizes


def compute_spatial_complexity_learnable(x: torch.Tensor,
                                         edge_net: nn.Module, eps: float = 1e-6):
    """可学习边缘提取网络"""
    B, C, H, W = x.shape
    x_gray = x.mean(1, keepdim=True)
    edge_map = edge_net(x_gray)
    e_min = edge_map.amin(dim=[2,3], keepdim=True)
    e_max = edge_map.amax(dim=[2,3], keepdim=True)
    complexity = (edge_map - e_min) / (e_max - e_min + eps)
    return complexity.squeeze(1)

# ========== 位置编码函数 ==========
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([pos_embed, np.zeros([1, embed_dim])], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_l = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ========== 辅助函数：空间复杂度计算、自适应步长、扫描与合并 ==========
def compute_spatial_complexity(x: torch.Tensor, eps: float = 1e-6):
    """使用 Sobel 算子计算空间复杂度，用于 AJSS-Mamba 的步长控制"""
    B, C, H, W = x.shape
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    x_gray = x.mean(1, keepdim=True)
    gx = F.conv2d(x_gray, sobel_x, padding=1)
    gy = F.conv2d(x_gray, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + eps)
    mag_min = mag.amin(dim=[2, 3], keepdim=True)
    mag_max = mag.amax(dim=[2, 3], keepdim=True)
    complexity = (mag - mag_min) / (mag_max - mag_min + eps)
    return complexity.squeeze(1)

def adaptive_step_from_complexity(m: torch.Tensor, s_min: int = 1, s_max: int = 5, alpha: float = 5.0):
    """根据空间复杂度计算每个方向的自适应步长（论文公式 9）"""
    B, H, W = m.shape
    avg_kernel = torch.ones(1, 1, 3, 3, device=m.device) / 9.0
    local_complexity = F.conv2d(m.unsqueeze(1), avg_kernel, padding=1).squeeze(1)
    h_mid, w_mid = H // 2, W // 2
    region_complexities = torch.zeros(B, 4, device=m.device)
    region_complexities[:, 0] = m[:, :h_mid, :w_mid].mean(dim=(1, 2))
    region_complexities[:, 1] = m[:, :h_mid, w_mid:].mean(dim=(1, 2))
    region_complexities[:, 2] = m[:, h_mid:, :w_mid].mean(dim=(1, 2))
    region_complexities[:, 3] = m[:, h_mid:, w_mid:].mean(dim=(1, 2))
    step_sizes = torch.zeros(B, 4, H, W, device=m.device, dtype=torch.long)
    for b in range(B):
        for d in range(4):
            region_comp = region_complexities[b, d].item()
            base_step = s_max - torch.sigmoid(torch.tensor(alpha * (region_comp - 0.5))) * (s_max - s_min)
            base_step = torch.clamp(base_step, min=s_min, max=s_max).round().to(torch.int64).item()
            for i in range(H):
                for j in range(W):
                    local_comp = local_complexity[b, i, j].item()
                    local_adjustment = torch.sigmoid(torch.tensor(alpha * (local_comp - 0.5))) * 2 - 1
                    adjusted_step = base_step + round(local_adjustment.item())
                    step_sizes[b, d, i, j] = max(s_min, min(s_max, adjusted_step))
    return step_sizes

class DirectionalScan(torch.autograd.Function):
    """四个方向的扫描操作，支持变步长"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, step_sizes: torch.Tensor, H: int, W: int):
        B, C, H, W = x.shape
        ctx.orig_shape = (B, C, H, W)
        ctx.H, ctx.W = H, W
        ctx.step_sizes = step_sizes
        seq_lens = torch.zeros(B, 4, dtype=torch.long, device=x.device)
        results = []
        max_seq_len = 0
        for b in range(B):
            sample_results = []
            for d in range(4):
                step_map = step_sizes[b, d]
                seq = []
                if d == 0:  # 左→右
                    for i in range(H):
                        j = 0
                        while j < W:
                            step = max(1, step_map[i, j].item())
                            seq.append(x[b, :, i, j])
                            j += step
                elif d == 1:  # 右→左
                    for i in range(H):
                        j = W - 1
                        while j >= 0:
                            step = max(1, step_map[i, j].item())
                            seq.append(x[b, :, i, j])
                            j -= step
                elif d == 2:  # 上→下
                    for j in range(W):
                        i = 0
                        while i < H:
                            step = max(1, step_map[i, j].item())
                            seq.append(x[b, :, i, j])
                            i += step
                else:  # 下→上
                    for j in range(W):
                        i = H - 1
                        while i >= 0:
                            step = max(1, step_map[i, j].item())
                            seq.append(x[b, :, i, j])
                            i -= step
                seq = torch.stack(seq, dim=1) if seq else torch.empty(C, 0, device=x.device)
                seq_len = seq.size(1)
                seq_lens[b, d] = seq_len
                max_seq_len = max(max_seq_len, seq_len)
                sample_results.append(seq)
            results.append(sample_results)
        padded_results = torch.zeros(B, 4, C, max_seq_len, device=x.device, dtype=x.dtype)
        for b in range(B):
            for d in range(4):
                seq_len = seq_lens[b, d]
                if seq_len > 0:
                    padded_results[b, d, :, :seq_len] = results[b][d]
        ctx.seq_lens = seq_lens
        return padded_results, seq_lens

    @staticmethod
    def backward(ctx, grad_output, grad_seq_lens):
        B, C, H, W = ctx.orig_shape
        step_sizes = ctx.step_sizes
        seq_lens = ctx.seq_lens
        grad_input = torch.zeros(B, C, H, W, device=grad_output.device, dtype=grad_output.dtype)
        for b in range(B):
            for d in range(4):
                step_map = step_sizes[b, d]
                seq_len = seq_lens[b, d]
                grad_seq = grad_output[b, d, :, :seq_len]
                if d == 0:
                    pos = 0
                    for i in range(H):
                        j = 0
                        while j < W and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            grad_input[b, :, i, j] += grad_seq[:, pos]
                            j += step
                            pos += 1
                elif d == 1:
                    pos = 0
                    for i in range(H):
                        j = W - 1
                        while j >= 0 and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            grad_input[b, :, i, j] += grad_seq[:, pos]
                            j -= step
                            pos += 1
                elif d == 2:
                    pos = 0
                    for j in range(W):
                        i = 0
                        while i < H and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            grad_input[b, :, i, j] += grad_seq[:, pos]
                            i += step
                            pos += 1
                else:
                    pos = 0
                    for j in range(W):
                        i = H - 1
                        while i >= 0 and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            grad_input[b, :, i, j] += grad_seq[:, pos]
                            i -= step
                            pos += 1
        return grad_input, None, None, None

class DirectionalMerge(torch.autograd.Function):
    """将四个方向的扫描结果合并回二维图像"""
    @staticmethod
    def forward(ctx, ys: torch.Tensor, seq_lens: torch.Tensor, H: int, W: int, step_sizes: torch.Tensor):
        print(f"ys.dim()={ys.dim()}, ys.shape={ys.shape}")
        if ys.dim() == 5:
            B, num_directions, G, C, max_L = ys.shape
            ys = ys.reshape(B, num_directions, G * C, max_L)
        else:
            B, num_directions, C, max_L = ys.shape
            G = 1
        ctx.H, ctx.W = H, W
        ctx.step_sizes = step_sizes
        ctx.seq_lens = seq_lens
        ctx.G = G
        output = torch.zeros(B, G * C, H, W, device=ys.device, dtype=ys.dtype)
        weights = torch.zeros(B, G * C, H, W, device=ys.device, dtype=ys.dtype)
        for b in range(B):
            for d in range(4):
                step_map = step_sizes[b, d]
                seq_len = seq_lens[b, d]
                y = ys[b, d, :, :seq_len]
                pos = 0
                if d == 0:
                    for i in range(H):
                        j = 0
                        while j < W and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            output[b, :, i, j] += y[:, pos]
                            weights[b, :, i, j] += 1
                            j += step
                            pos += 1
                elif d == 1:
                    for i in range(H):
                        j = W - 1
                        while j >= 0 and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            output[b, :, i, j] += y[:, pos]
                            weights[b, :, i, j] += 1
                            j -= step
                            pos += 1
                elif d == 2:
                    for j in range(W):
                        i = 0
                        while i < H and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            output[b, :, i, j] += y[:, pos]
                            weights[b, :, i, j] += 1
                            i += step
                            pos += 1
                else:
                    for j in range(W):
                        i = H - 1
                        while i >= 0 and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            output[b, :, i, j] += y[:, pos]
                            weights[b, :, i, j] += 1
                            i -= step
                            pos += 1
        output = output / (weights + 1e-6)
        if output.dim() == 5:
            B, G, C, H, W = output.shape
            output = output.reshape(B, G * C, H, W)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        H, W = ctx.H, ctx.W
        step_sizes = ctx.step_sizes
        seq_lens = ctx.seq_lens
        G = ctx.G
        B, total_C, H, W = grad_output.shape
        C = total_C // G
        max_L = seq_lens.max().item()

        grad_ys = torch.zeros(B, 4, total_C, max_L, device=grad_output.device, dtype=grad_output.dtype)

        for b in range(B):
            for d in range(4):
                step_map = step_sizes[b, d]
                seq_len = seq_lens[b, d]
                pos = 0
                if d == 0:
                    for i in range(H):
                        j = 0
                        while j < W and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            grad_ys[b, d, :, pos] = grad_output[b, :, i, j]
                            j += step
                            pos += 1
                elif d == 1:
                    for i in range(H):
                        j = W - 1
                        while j >= 0 and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            grad_ys[b, d, :, pos] = grad_output[b, :, i, j]
                            j -= step
                            pos += 1
                elif d == 2:
                    for j in range(W):
                        i = 0
                        while i < H and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            grad_ys[b, d, :, pos] = grad_output[b, :, i, j]
                            i += step
                            pos += 1
                else:
                    for j in range(W):
                        i = H - 1
                        while i >= 0 and pos < seq_len:
                            step = max(1, step_map[i, j].item())
                            grad_ys[b, d, :, pos] = grad_output[b, :, i, j]
                            i -= step
                            pos += 1

        if G > 1:
            grad_ys = grad_ys.reshape(B, 4, G, C, max_L)

        return grad_ys, None, None, None, None

# ========== 核心模块：AJSS-Mamba（自适应跳跃空间扫描 Mamba） ==========
class AJSSMamba(nn.Module):
    def __init__(self, d_model=64, d_state=16, ssm_ratio=2.0, ssm_rank_ratio=2.0,
                 dt_rank="auto", act_layer=nn.SiLU, d_conv=3, conv_bias=True,
                 dropout=0.0, bias=False, dt_min=0.001, dt_max=0.1, dt_init="random",
                 dt_scale=1.0, dt_init_floor=1e-4, simple_init=True, forward_type="v2",
                 mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                 use_checkpoint=False, max_spatial_stride=5, complexity_method='sobel', **kwargs):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.max_spatial_stride = max_spatial_stride
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        self.forward_core = self.forward_corev2
        self.K = 9 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act = act_layer()

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand, out_channels=d_expand, groups=d_expand,
                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs
            )

        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        self.x_proj_weight = nn.Parameter(torch.randn(self.K, d_inner, 128))
        self.dt_projs_weight = nn.Parameter(torch.randn(self.K, d_inner, 64))
        self.dt_projs_bias = nn.Parameter(torch.randn(self.K, d_inner))
        self.A_logs = nn.Parameter(torch.randn(self.K2 * d_inner, 32))
        self.Ds = nn.Parameter(torch.ones(self.K2 * d_inner))

        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, 32)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, 64)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.complexity_method = complexity_method
        if self.complexity_method == 'learnable':
            self.edge_net = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=3, padding=1)
            )

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)

        if self.complexity_method == 'sobel':
            m = compute_spatial_complexity_sobel(x)
        elif self.complexity_method == 'local_var':
            m = compute_spatial_complexity_local_var(x)
        elif self.complexity_method == 'learnable':
            m = compute_spatial_complexity_learnable(x, self.edge_net)
        else:
            raise ValueError(f"Unknown complexity_method: {self.complexity_method}")

        s_max = max(1, self.max_spatial_stride)
        step_sizes = adaptive_step_from_complexity(m, s_min=1, s_max=s_max)
        step_sizes = torch.clamp(step_sizes, min=1)

        B, C, H, W = x.shape
        xs, seq_lens = DirectionalScan.apply(x, step_sizes, H, W)

        ys = []
        for d in range(4):
            xd = xs[:, d]
            seq_lens_d = seq_lens[:, d]
            xd = xd.transpose(1, 2)
            xd_proj = F.linear(xd, self.x_proj_weight[0], None)
            total_feat = xd_proj.shape[-1]
            R = self.dt_projs_weight.shape[-1]
            third = (total_feat - R) // 2
            dts, Bs, Cs = torch.split(xd_proj, [R, third, third], dim=-1)
            dts = torch.einsum("b l r, d r -> b d l", dts, self.dt_projs_weight[0])

            mask = torch.arange(xs.shape[-1], device=xs.device).expand(B, -1) < seq_lens_d.unsqueeze(1)
            mask = mask.unsqueeze(1)

            A_ = -torch.exp(self.A_logs[:C].float())
            D_ = self.Ds[:C].float()
            dt_bias_ = self.dt_projs_bias[0].float()

            yd = self._selective_scan_with_mask(xd.transpose(1,2), dts, A_, Bs.unsqueeze(2), Cs.unsqueeze(2),
                                                D_, dt_bias_, mask)
            ys.append(yd)

        ys = torch.stack(ys, dim=1)
        y = DirectionalMerge.apply(ys, seq_lens, H, W, step_sizes)

        C_target = x.shape[1]
        if y.shape[1] != C_target:
            total_C = y.shape[1]
            if total_C % C_target == 0:
                G = total_C // C_target
                y = y.view(B, G, C_target, H, W).mean(dim=1)
            else:
                if total_C > C_target:
                    y = y[:, :C_target, :, :]
                else:
                    repeat = (C_target + total_C - 1) // total_C
                    y = y.repeat(1, repeat, 1, 1)[:, :C_target, :, :]

        if self.ssm_low_rank:
            y = self.out_rank(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return y.permute(0, 2, 3, 1)

    def _selective_scan_with_mask(self, u, delta, A, B, C, D, delta_bias, mask):
        batch, dim, seq_len = u.shape
        d_state = A.shape[1]
        u = u * mask
        delta = delta * mask

        if A.shape[0] != dim:
            if dim > A.shape[0]:
                repeat_factor = dim // A.shape[0]
                A = A.repeat(repeat_factor, 1)
                if dim % A.shape[0] != 0:
                    extra_rows = dim % A.shape[0]
                    A = torch.cat([A, A[:extra_rows]], dim=0)
            else:
                A = A[:dim]

        if B.dim() == 2:
            B = B.unsqueeze(1).expand(-1, seq_len, -1)
        if C.dim() == 2:
            C = C.unsqueeze(1).expand(-1, seq_len, -1)

        if B.shape[-1] != d_state:
            if B.shape[-1] > d_state:
                B = B[:, :, :d_state]
            else:
                pad_size = d_state - B.shape[-1]
                B = F.pad(B, (0, pad_size))
        if C.shape[-1] != d_state:
            if C.shape[-1] > d_state:
                C = C[:, :, :d_state]
            else:
                pad_size = d_state - C.shape[-1]
                C = F.pad(C, (0, pad_size))

        x = torch.zeros(batch, dim, d_state, device=u.device)
        outputs = []
        for i in range(seq_len):
            valid_mask = mask[:, :, i].unsqueeze(-1)
            delta_A = delta[:, :, i].unsqueeze(-1) * A.unsqueeze(0)
            exp_delta_A = torch.exp(delta_A)
            x_exp = x * exp_delta_A
            u_B = u[:, :, i].unsqueeze(-1) * B[:, i, :].unsqueeze(1)
            x = x_exp * valid_mask + u_B * valid_mask
            dot_product = (x * C[:, i, :].unsqueeze(1)).sum(dim=-1)
            y = dot_product + u[:, :, i] * D
            outputs.append(y)
        return torch.stack(outputs, dim=-1)

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1)

        y = self.forward_core(x, channel_first=(self.d_conv > 1))

        if z.shape[-1] != y.shape[-1]:
            if not hasattr(self, 'z_channel_adjust'):
                self.z_channel_adjust = nn.Linear(z.shape[-1], y.shape[-1], device=z.device)
            z = self.z_channel_adjust(z)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# ========== 光谱分支：AJBS-Mamba（自适应跳跃波段扫描 Mamba） ==========
def compute_spectral_complexity(x: torch.Tensor, window: int = 2):
    """计算光谱复杂度（相邻波段差平方的局部均值）"""
    B, C, L = x.shape
    complexity = torch.zeros(B, L, device=x.device)
    for i in range(L):
        spec = x[:, :, i]
        diffs = (spec[:, 1:] - spec[:, :-1]) ** 2
        if window > 0:
            kernel = torch.ones(1, 1, window, device=x.device) / window
            diffs = F.conv1d(diffs.unsqueeze(1), kernel, padding=window//2).squeeze(1)
        complexity[:, i] = diffs.mean(dim=1)
    min_val = complexity.min(dim=1, keepdim=True)[0]
    max_val = complexity.max(dim=1, keepdim=True)[0]
    complexity = (complexity - min_val) / (max_val - min_val + 1e-6)
    return complexity

class AJBSMamba(nn.Module):
    def __init__(self, d_model=96, d_state=4, ssm_ratio=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, bias=False, max_spectral_stride=3, **kwargs):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.max_spectral_stride = max_spectral_stride

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_proj = self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                     **factory_kwargs)
        self.A_log = self._A_log_init(self.d_state, self.d_inner)
        self.D = self._D_init(self.d_inner)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _adaptive_spectral_scan(self, u, delta, A, B, C, D, delta_bias, delta_softplus, z):
        B_batch, dim, L = u.shape
        spectral_signal = u.mean(dim=1)
        diffs = (spectral_signal[:, 1:] - spectral_signal[:, :-1]) ** 2
        window = 3
        kernel = torch.ones(1, 1, window, device=u.device) / window
        complexity = F.conv1d(diffs.unsqueeze(1), kernel, padding=window // 2).squeeze(1)
        if complexity.shape[1] < L:
            complexity = F.pad(complexity, (0, L - complexity.shape[1]), value=0)

        complexity = torch.nan_to_num(complexity, nan=0.0, posinf=1.0, neginf=0.0)
        min_val = complexity.min(dim=1, keepdim=True)[0]
        max_val = complexity.max(dim=1, keepdim=True)[0]
        range_val = max_val - min_val
        complexity = torch.where(range_val > 0, (complexity - min_val) / (range_val + 1e-6),
                                 torch.zeros_like(complexity))
        complexity = torch.clamp(complexity, 0.0, 1.0)

        s_max = self.max_spectral_stride
        step_sizes = 1 + (s_max - 1) * (1 - torch.sigmoid(5 * (complexity - 0.5)))
        step_sizes = torch.clamp(step_sizes, min=1, max=s_max).round().long()
        step_sizes = torch.where(step_sizes < 1, torch.ones_like(step_sizes), step_sizes)

        sampled_indices = []
        for b in range(B_batch):
            idx = 0
            indices = []
            while idx < L:
                indices.append(idx)
                step = step_sizes[b, idx].item()
                if step <= 0:
                    step = 1
                idx += step
            sampled_indices.append(indices)

        max_len = max(len(indices) for indices in sampled_indices) if sampled_indices else 0
        if max_len == 0:
            if delta_softplus:
                delta = F.softplus(delta)
            y = selective_scan_fn(u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=False)
            return y, z

        u_sampled = torch.zeros(B_batch, dim, max_len, device=u.device, dtype=u.dtype)
        delta_sampled = torch.zeros(B_batch, dim, max_len, device=u.device, dtype=delta.dtype)
        mask = torch.zeros(B_batch, 1, max_len, device=u.device, dtype=torch.bool)
        for b in range(B_batch):
            indices = sampled_indices[b]
            Lb = len(indices)
            u_sampled[b, :, :Lb] = u[b, :, indices]
            delta_sampled[b, :, :Lb] = delta[b, :, indices]
            mask[b, 0, :Lb] = True

        B_sampled = torch.zeros(B_batch, self.d_state, max_len, device=u.device, dtype=B.dtype)
        C_sampled = torch.zeros(B_batch, self.d_state, max_len, device=u.device, dtype=C.dtype)
        z_sampled = torch.zeros(B_batch, dim, max_len, device=u.device, dtype=u.dtype)
        for b in range(B_batch):
            indices = sampled_indices[b]
            Lb = len(indices)
            B_sampled[b, :, :Lb] = B[b, :, indices]
            C_sampled[b, :, :Lb] = C[b, :, indices]
            z_sampled[b, :, :Lb] = z[b, :, indices]

        if delta_softplus:
            delta_sampled = F.softplus(delta_sampled)
        y = selective_scan_fn(u_sampled, delta_sampled, A, B_sampled, C_sampled, D,
                              delta_bias=delta_bias, delta_softplus=False)
        return y, z_sampled

    def forward_core(self, x: torch.Tensor, z: torch.Tensor):
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        z = z.permute(0, 2, 1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B_val, C_val = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())
        B_val = rearrange(B_val, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_val = rearrange(C_val, "(b l) dstate -> b dstate l", l=L).contiguous()

        y, z_sampled = self._adaptive_spectral_scan(
            x, dt, A, B_val, C_val, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            z=z
        )
        y = rearrange(y, "b d l -> b l d")
        z_sampled = rearrange(z_sampled, "b d l -> b l d")
        y = self.out_norm(y)
        return y, z_sampled

    def forward(self, x: torch.Tensor):
        B, L, d = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        y, z_sampled = self.forward_core(x, z)
        y = y * F.silu(z_sampled)
        out = self.out_proj(y)
        return out


# ========== 局部对比度增强 ==========
class LocalContrastEnhancement(nn.Module):
    def __init__(self, enhance_coeff=0.3):
        super().__init__()
        self.enhance_coeff = enhance_coeff
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        background = self.avg_pool(x)
        details = x - background
        enhanced = background + self.enhance_coeff * details
        return enhanced

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 ssm_d_state: int = 16, ssm_ratio=2.0, ssm_rank_ratio=2.0, ssm_dt_rank="auto",
                 ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0,
                 ssm_simple_init=False, forward_type="v2", mlp_ratio=4.0, mlp_act_layer=nn.GELU,
                 mlp_drop_rate=0.0, use_checkpoint=False, max_spatial_stride=5, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = AJSSMamba(
            d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio, dt_rank=ssm_dt_rank, act_layer=ssm_act_layer,
            d_conv=ssm_conv, conv_bias=ssm_conv_bias, dropout=ssm_drop_rate,
            simple_init=ssm_simple_init, forward_type=forward_type, max_spatial_stride=max_spatial_stride)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            LocalContrastEnhancement(enhance_coeff=0.3)
        )
        self.drop_path = DropPath(drop_path)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,
                           act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)
    def _forward(self, input: torch.Tensor):
        x = self.norm(input)
        x_ssm = self.op(x)
        x_conv = self.conv_branch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x_ssm + x_conv + input
        if self.mlp_branch:
            B, H, W, C = x.shape
            x_mlp = x.reshape(B, H * W, C)
            x_mlp = self.mlp(self.norm2(x_mlp))
            x_mlp = x_mlp.reshape(B, H, W, C)
            x = x + self.drop_path(x_mlp)
        return x
    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class block_2D(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, d_state: int = 16, bi: bool = False, cls: bool = True, **kwargs):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.vss_block = VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path, **kwargs)
        self.drop_path = DropPath(drop_path)
    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm(x)
        x = self.vss_block(x)
        return self.drop_path(x) + shortcut

class block_1D_spe(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, d_state: int = 16, bi: bool = True, cls: bool = True, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = AJBSMamba(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x1 = self.self_attention(x)
        return self.drop_path(x1)

class spectral_spatial_block(nn.Module):
    def __init__(self, embed_dim, bi=False, N=8, drop_path=0.0,
                 norm_layer=nn.LayerNorm, cls=True, fu=True, **kwargs):
        super().__init__()
        self.spa_block = block_2D(hidden_dim=embed_dim, drop_path=drop_path, bi=bi, cls=cls, **kwargs)
        self.spe_block = block_1D_spe(hidden_dim=embed_dim, drop_path=drop_path, bi=bi, cls=cls, **kwargs)
    def forward(self, x_spa, x_spe):
        x_spa = self.spa_block(x_spa)
        x_spe = self.spe_block(x_spe)
        return x_spa, x_spe

# ========== 噪声扰动集成融合模块 NPF ==========
class NPF(nn.Module):
    def __init__(self, perturbation_rate=0.2, num_perturbations=100):
        super().__init__()
        self.perturbation_rate = perturbation_rate
        self.num_perturbations = num_perturbations

    def _perturb_features(self, features):
        """对特征添加随机掩码噪声（公式17）"""
        mask = (torch.rand_like(features) < self.perturbation_rate).float()
        noise = torch.randn_like(features) * features.std(dim=1, keepdim=True)
        perturbed = features + mask * noise
        return perturbed

    def _dynamic_weighted_fusion(self, candidates):
        """动态加权融合（公式18-19）"""
        var = candidates.var(dim=0, unbiased=False) + 1e-6
        weight = 1.0 / var
        weight = weight / weight.sum(dim=0, keepdim=True)
        Z = (candidates * weight.unsqueeze(0)).sum(dim=0)
        return Z

    def forward(self, spatial_features, spectral_features):
        """
        spatial_features: [batch, dim]
        spectral_features: [batch, dim]
        return: fused_features [batch, dim]
        """
        candidates = []
        for _ in range(self.num_perturbations):
            perturbed_spatial = self._perturb_features(spatial_features)
            perturbed_spectral = self._perturb_features(spectral_features)
            fused = torch.cat([perturbed_spatial, perturbed_spectral], dim=1)
            candidates.append(fused)
        candidates = torch.stack(candidates, dim=0)
        Z = self._dynamic_weighted_fusion(candidates)
        return Z

# ========== Patch Embed 模块 ==========
class PatchEmbed_2D(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=15, embed_dim=64,
                 norm_layer=None, flatten=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

class PatchEmbed_Spe(nn.Module):
    def __init__(self, img_size=(9,9), patch_size=2, embed_dim=64, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels=img_size[0] * img_size[1],
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        x = x.transpose(2, 1)
        x = self.proj(x)
        x = x.transpose(2, 1)
        x = self.norm(x)
        return x

# ========== 整体框架 AJSDP-Mamba ==========
class AJSDPMMamba(nn.Module):
    def __init__(self, spa_img_size=(224,224), spe_img_size=(5,5),
                 spa_patch_size=16, spe_patch_size=2, in_chans=3,
                 hid_chans=32, embed_dim=128, nclass=10, drop_path=0.0,
                 depth=4, bi=False, norm_layer=nn.LayerNorm,
                 global_pool=True, cls=True, fu=True,
                 perturbation_rate=0.2, num_perturbations=100,
                 max_spatial_stride=5, max_spectral_stride=3,
                 **kwargs):
        super().__init__()
        self.name = 'AJSDP-Mamba'
        # 降维
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
        )
        self.half_spa_patch_size = spa_img_size[0] // 2
        self.half_spe_patch_size = spe_img_size[0] // 2
        self.spe_patch_embed = PatchEmbed_Spe(img_size=spe_img_size, patch_size=spe_patch_size, embed_dim=embed_dim)
        self.spa_patch_embed = PatchEmbed_2D(spa_img_size, spa_patch_size, hid_chans, embed_dim, flatten=False)
        spa_num_patches = self.spa_patch_embed.num_patches
        if in_chans % spe_patch_size == 0:
            spe_num_patches = in_chans // spe_patch_size
        else:
            spe_num_patches = in_chans // spe_patch_size

        self.cls = cls
        if self.cls:
            self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            N = spa_num_patches + spe_num_patches + 2
            self.cs = -1
        else:
            N = spa_num_patches + spe_num_patches
            self.cs = N

        block_kwargs = {
            'max_spatial_stride': max_spatial_stride,
            'max_spectral_stride': max_spectral_stride,
            **kwargs
        }
        self.blocks = nn.ModuleList([
            spectral_spatial_block(embed_dim, bi, N=N, drop_path=drop_path, cls=self.cls, fu=fu, **block_kwargs)
            for _ in range(depth)
        ])
        self.npf = NPF(perturbation_rate=perturbation_rate, num_perturbations=num_perturbations)
        self.head = nn.Linear(embed_dim * 2, nclass)
        self.spa_pos_embed = nn.Parameter(torch.zeros(1, spa_num_patches + 1, embed_dim), requires_grad=False)
        self.spe_pos_embed = nn.Parameter(
            torch.from_numpy(get_1d_sincos_pos_embed(embed_dim, spe_num_patches + 1, cls_token=False)).unsqueeze(0),
            requires_grad=False)
        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x):
        x_spa = self.spa_patch_embed(x)
        x_center = x[:, :,
                   self.half_spa_patch_size - self.half_spe_patch_size: self.half_spa_patch_size + self.half_spe_patch_size + 1,
                   self.half_spa_patch_size - self.half_spe_patch_size: self.half_spa_patch_size + self.half_spe_patch_size + 1]
        x_spe = self.spe_patch_embed(x_center)
        if self.cls:
            spe_cls_token = self.spe_cls_token + self.spe_pos_embed[:, -1:, :]
            spe_cls_tokens = spe_cls_token.expand(x_spe.shape[0], -1, -1)
            x_spe = torch.cat((x_spe, spe_cls_tokens), dim=1)
        for blk in self.blocks:
            x_spa, x_spe = blk(x_spa, x_spe)
        if self.global_pool:
            x_spa = x_spa.mean(dim=[1, 2])
            if self.cls:
                x_spe = x_spe[:, 0:-1, :].mean(dim=1)
            else:
                x_spe = x_spe.mean(dim=1)
            fused = self.npf(x_spa, x_spe)
        else:
            raise NotImplementedError("Only global_pool=True is supported for now")
        return fused

    def forward(self, x):
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[4], x.shape[2], x.shape[3])
        x = self.dimen_redu(x)
        features = self.forward_features(x)
        out = self.head(features)
        return out

# ========== 模型实例化函数 ==========
def AJSDP_Mamba(pretrained=False, **kwargs):
    model = AJSDPMMamba(
        spa_img_size=(21,21),
        spe_img_size=(3,3),
        spa_patch_size=3,
        spe_patch_size=2,
        in_chans=30,
        hid_chans=64,
        embed_dim=64,
        drop_path=0.1,
        nclass=9,
        depth=1,
        bi=False,
        norm_layer=nn.LayerNorm,
        global_pool=True,
        cls=True,
        fu=True,
        perturbation_rate=0.2,
        num_perturbations=100,
        max_spatial_stride=5,
        max_spectral_stride=3,
        complexity_method='sobel',
        **kwargs)
    return model

if __name__ == '__main__':
    model = AJSDP_Mamba()
    dummy_input = torch.randn(64, 30, 21, 21)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
