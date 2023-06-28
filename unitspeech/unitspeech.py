""" from https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS """

import math
import torch
from einops import rearrange

from unitspeech.base import BaseModule


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3,
                                                         padding=1), torch.nn.GroupNorm(
            groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8, spk_emb_dim=0):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim + spk_emb_dim, dim_out))

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


class LinearAttention(BaseModule):
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
                            heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w',
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
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


class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8, pe_scale=1000, spk_emb_dim=0):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.pe_scale = pe_scale

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, spk_emb_dim=spk_emb_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, spk_emb_dim=spk_emb_dim),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, spk_emb_dim=spk_emb_dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, spk_emb_dim=spk_emb_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, spk_emb_dim=spk_emb_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, spk_emb_dim=spk_emb_dim),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk_emb=None):
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        t = torch.cat((t, spk_emb.squeeze(1)), dim=-1)

        x = torch.stack([mu, x], 1)
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


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t ** 2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


class UnitSpeech(BaseModule):
    def __init__(self, n_feats, dim, dim_mults, beta_min=0.05, beta_max=20, pe_scale=1000, spk_emb_dim=0):
        super(UnitSpeech, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.dim_mults = dim_mults
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.text_uncon = torch.nn.Parameter(torch.zeros(1, n_feats, 1))
        self.spk_uncon = torch.nn.Parameter(torch.zeros(1, 1, spk_emb_dim))

        self.estimator = GradLogPEstimator2d(dim, dim_mults=dim_mults, pe_scale=pe_scale, spk_emb_dim=spk_emb_dim)

    def register_beta(self, betas):
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64).to(betas.device), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.register("sqrt_recip_one_minus_alphas_cumprod", torch.rsqrt(1 - alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod_prev", torch.sqrt(1 - alphas_cumprod_prev))

        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)),
        )
        self.register(
            "posterior_mean_coef2",
            ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)),
        )

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32), persistent=False)

    def predict_start_from_score(self, x_t, t, score):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                + extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) *
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * score
        )

    def q_posterior(self, x_0, score, t, eta=1.0):
        sigma = eta * torch.sqrt(extract(self.posterior_variance, t, score.shape))

        mean = (
                torch.sqrt(extract(self.alphas_cumprod_prev, t, score.shape)) * x_0
                - torch.sqrt(1 - extract(self.alphas_cumprod_prev, t, score.shape) - torch.pow(sigma, 2))
                * score * extract(self.sqrt_one_minus_alphas_cumprod, t, score.shape)
        )

        var = (eta ** 2) * extract(self.posterior_variance, t, score.shape)

        return mean, var

    def p_mean_variance(self, x, i, eta=1.0, score=None):
        x_recon = self.predict_start_from_score(x, i, score=score)
        mean, var = self.q_posterior(x_recon, score, i, eta=eta)
        return mean, var

    def classifier_free_guidance(self, xt, mask, cond, t, spk_emb,
                                 text_uncon, spk_uncon, text_gradient_scale, spk_gradient_scale):
        if text_gradient_scale > 0.0 and spk_gradient_scale > 0.0:
            xt = torch.cat([xt, xt, xt], dim=0)
            mask = torch.cat([mask, mask, mask], dim=0)
            cond = torch.cat([text_uncon, cond, cond], dim=0)
            t = torch.cat([t, t, t], dim=0)
            spk_emb = torch.cat([spk_emb, spk_uncon, spk_emb], dim=0)
        elif text_gradient_scale > 0.0:
            xt = torch.cat([xt, xt], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            cond = torch.cat([text_uncon, cond], dim=0)
            t = torch.cat([t, t], dim=0)
            spk_emb = torch.cat([spk_emb, spk_emb], dim=0)
        elif spk_gradient_scale > 0.0:
            xt = torch.cat([xt, xt], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            cond = torch.cat([cond, cond], dim=0)
            t = torch.cat([t, t], dim=0)
            spk_emb = torch.cat([spk_uncon, spk_emb], dim=0)

        score = self.estimator(xt, mask, cond, t, spk_emb)

        if text_gradient_scale > 0.0 and spk_gradient_scale > 0.0:
            score_text_uncon, score_spk_uncon, score = torch.chunk(score, 3, dim=0)
            score = score + text_gradient_scale * (score - score_text_uncon) + spk_gradient_scale * (
                        score - score_spk_uncon)
        elif text_gradient_scale > 0.0:
            score_text_uncon, score = torch.chunk(score, 2, dim=0)
            score = score + text_gradient_scale * (score - score_text_uncon)
        elif spk_gradient_scale > 0.0:
            score_spk_uncon, score = torch.chunk(score, 2, dim=0)
            score = score + spk_gradient_scale * (score - score_spk_uncon)
        return score

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, cond, spk_emb, n_timesteps, text_gradient_scale=0.0, spk_gradient_scale=0.0):
        h = 1.0 / n_timesteps
        alpha_cumprods = []

        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            alpha_cumprods.append(torch.exp(-get_noise(time, self.beta_min, self.beta_max, cumulative=True)))

        alpha_cumprods = torch.cat(
            [torch.cat(alpha_cumprods).squeeze(), torch.ones_like(torch.cat(alpha_cumprods).squeeze())[0:1]])
        betas = 1 - alpha_cumprods[:-1] / alpha_cumprods[1:]
        self.register_beta(betas.flip(0))

        xt = z * mask

        text_uncon = None
        spk_uncon = None

        if text_gradient_scale > 0.0:
            text_uncon = self.text_uncon.repeat(1, 1, cond.shape[-1])

        if spk_gradient_scale > 0.0:
            spk_uncon = self.spk_uncon / self.spk_uncon.norm()

        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            idx = torch.ones(z.shape[0], device=z.device).long() * (n_timesteps - 1 - i)

            score = self.classifier_free_guidance(xt, mask, cond, t, spk_emb, text_uncon, spk_uncon, text_gradient_scale, spk_gradient_scale)

            mean, var = self.p_mean_variance(xt, idx, eta=1.0, score=score)
            noise = torch.randn(xt.shape, dtype=xt.dtype, device=xt.device, requires_grad=False)
            shape = [xt.shape[0]] + [1] * (xt.ndim - 1)
            nonzero_mask = (1 - (idx == 0).type(torch.float32)).view(*shape)
            xt = (mean + nonzero_mask * torch.sqrt(var) * noise) * mask
        xt = xt * mask
        return xt

    def forward_diffusion(self, x0, mask, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise)
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def forward(self, z, mask, cond, spk_emb, n_timesteps, text_gradient_scale=0.0, spk_gradient_scale=0.0):
        return self.reverse_diffusion(
            z, mask, cond, spk_emb, n_timesteps,
            text_gradient_scale=text_gradient_scale, spk_gradient_scale=spk_gradient_scale
        )

    def loss_t(self, x0, mask, cond, t, spk_emb):
        xt, z = self.forward_diffusion(x0, mask, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        cond = cond * mask

        noise_estimation = self.estimator(xt, mask, cond, t, spk_emb)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, cond, spk_emb=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, cond, t, spk_emb)
