import torch
import torch.nn as nn
import math
from einops import rearrange


class Diffusion(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(Diffusion, self).__init__()
        self.n_spks = preprocess_config["preprocessing"]["spk"]["n_spks"]
        self.spk_emb_dim = preprocess_config["preprocessing"]["spk"]["spk_emb_dim"]
        self.n_feats = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.dim = model_config["decoder"]["dec_dim"]
        self.beta_min = model_config["decoder"]["beta_min"]
        self.beta_max = model_config["decoder"]["beta_max"]
        self.pe_scale = model_config["decoder"]["pe_scale"]

        self.estimator = GradLogPEstimator2d(self.dim, n_spks=self.n_spks,
                                             spk_emb_dim=self.spk_emb_dim,
                                             pe_scale=self.pe_scale)

    def forward(self, y_mask, mu_y, n_timesteps, temperature=None, mels=None, stoc=False, spk=None, offset=1e-5):
        if mels is not None:
            t = torch.rand(mels.shape[0], dtype=mels.dtype, device=mels.device,
                           requires_grad=False)
            t = torch.clamp(t, offset, 1.0 - offset)
            # 任意时间步t

            time = t.unsqueeze(-1).unsqueeze(-1)
            cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
            # noise = beta_init + (beta_term - beta_init)*t
            # 在给定无限时间范围 T 的情况下，将任何数据转换为高斯噪声
            # xt, z = self.forward_diffusion(mels, y_mask, mu_y, cum_noise)
            mean = mels * torch.exp(-0.5 * cum_noise) + mu_y * (1.0 - torch.exp(-0.5 * cum_noise))
            variance = 1.0 - torch.exp(-cum_noise)
            z = torch.randn(mels.shape, dtype=mels.dtype, device=mels.device,
                            requires_grad=False)
            xt = mean + z * torch.sqrt(variance)

            xt = xt * y_mask
            z = z * y_mask

            # 通过神经网络来估计t时刻噪声数据X_t的对数密度的梯度
            # 对数密度的梯度为\epsilon_t/sqrt(\lambda_t)，即为λt = 1 − e^(R_0~t β_s ds。)
            # 也就是z标准正态分布随机值
            noise_estimation = self.estimator(xt, y_mask, mu_y, t, spk)
            noise_estimation *= torch.sqrt(variance)

            return noise_estimation, z
        else:
            with torch.no_grad():
                # Sample latent representation from terminal distribution N(mu_y, I)
                noize = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
                # Generate sample by performing reverse dynamics
                h = 1.0 / n_timesteps
                xt = noize * y_mask

                for i in range(n_timesteps):
                    t = (1.0 - (i + 0.5) * h) * torch.ones(noize.shape[0], dtype=noize.dtype,
                                                           device=noize.device)
                    time = t.unsqueeze(-1).unsqueeze(-1)
                    noise_t = get_noise(time, self.beta_min, self.beta_max,
                                        cumulative=False)

                    if stoc:  # adds stochastic term
                        dxt_det = 0.5 * (mu_y - xt) - self.estimator(xt, y_mask, mu_y, t, spk)
                        dxt_det = dxt_det * noise_t * h
                        dxt_stoc = torch.randn(noize.shape, dtype=noize.dtype, device=noize.device,
                                               requires_grad=False)
                        dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                        dxt = dxt_det + dxt_stoc
                    else:
                        dxt = 0.5 * (mu_y - xt - self.estimator(xt, y_mask, mu_y, t, spk))
                        dxt = dxt * noise_t * h
                    xt = (xt - dxt) * y_mask
            return xt


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class GradLogPEstimator2d(nn.Module):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(
                torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4),
                Mish(),
                torch.nn.Linear(spk_emb_dim * 4, n_feats)
            )
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            Mish(),
            torch.nn.Linear(dim * 4, dim)
        )

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


class Mish(nn.Module):
    def __init__(self, ):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim,
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3,
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class Rezero(nn.Module):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)',
                            heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w',
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)