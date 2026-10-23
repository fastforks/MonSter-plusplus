import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)




    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr

class FastGeoEncodingVolume:
    """Inference version of Combined_Geo_Encoding_Volume (same math as grid_sample bilinear, align_corners=True,
    zero padding), without per-lookup host syncs / H2D copies:
    - all-pairs correlation from an fp16 GEMM stored fp16 ([P, W2]; upstream: fp32 einsum + contiguous copy),
    - use_kernel: one fused lookup kernel, only level 0 stored (fp16; level 1 pooled on the fly): Midd-F
      geometry + correlation pyramids 2.9 GB fp32 -> 0.86 GB; else torch gathers on fp32 geo / fp16 corr pyramids,
    - 1-D linear interpolation by two gathers instead of grid_sample on [P, C, 1, D] (+ upstream's torch.unique sync),
    - output written as NHWC [B, H, W, 162] (channels_last view), lookup coordinates fp32."""

    def __init__(self, fmap1, fmap2, geo_volume, num_levels=2, radius=4, out_dtype=torch.float16, use_kernel=False):
        self.num_levels, self.radius, self.out_dtype = num_levels, radius, out_dtype
        self.use_kernel = use_kernel and out_dtype == torch.float16
        b, c, d, h, w = geo_volume.shape
        self.shape = (b, h, w)
        f1 = fmap1.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        f2 = fmap2.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        if self.use_kernel:  # fp16 level 0 only; the kernel pools level 1 on the fly
            assert num_levels == 2
            self.geo = [geo_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, d).half().contiguous()]
            self.corr = [torch.bmm(f1.half(), f2.half()).view(b * h * w, w)]
            return
        g = geo_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, d).float().contiguous()
        # correlation: [B*H, W1, D] @ [B*H, D, W2] (fp16 in, fp32 accumulate, fp16 out)
        corr = torch.bmm(f1.half(), f2.half()).view(b * h * w, w).contiguous()
        self.geo, self.corr = [g], [corr]
        for _ in range(num_levels - 1):
            g = F.avg_pool1d(g, 2, 2)
            self.geo.append(g.contiguous())
            corr = F.avg_pool1d(corr.unsqueeze(1), 2, 2).squeeze(1)
            self.corr.append(corr.contiguous())
        self.dx = torch.arange(-radius, radius + 1, device=geo_volume.device, dtype=torch.float32).view(1, -1)
        self.coords = torch.arange(w, device=geo_volume.device, dtype=torch.float32).repeat(b * h).view(-1, 1)

    @staticmethod
    def _sample(vol, x):
        """vol [P, C, D] or [P, D]; x [P, K] fp32 positions -> [P, C*K] or [P, K] (fp32 math)."""
        D = vol.shape[-1]
        x0 = torch.floor(x)
        w1 = x - x0
        i0 = x0.long()
        i1 = i0 + 1
        v0 = ((i0 >= 0) & (i0 < D)).float() * (1 - w1)
        v1 = ((i1 >= 0) & (i1 < D)).float() * w1
        idx = torch.cat([i0, i1], 1).clamp_(0, D - 1)
        wgt = torch.cat([v0, v1], 1)
        K = x.shape[1]
        if vol.dim() == 3:
            P, C, _ = vol.shape
            s = vol.gather(2, idx.unsqueeze(1).expand(P, C, 2 * K)).float() * wgt.unsqueeze(1)
            return (s[..., :K] + s[..., K:]).reshape(P, C * K)
        s = vol.gather(1, idx).float() * wgt
        return s[:, :K] + s[:, K:]

    def __call__(self, disp):
        b, h, w = self.shape
        if self.use_kernel:
            from core.inference import kern
            k = kern()
            P = b * h * w
            C = self.geo[0].shape[1]
            K = 2 * self.radius + 1
            out = torch.empty((P, 2 * (C + 1) * K), device=disp.device, dtype=torch.float16)
            dsp = disp.reshape(-1).float().contiguous()
            k.launch("geo_lookup", k.linear(P * 2 * (C + 1)), self.geo[0], C, self.geo[0].shape[2], self.corr[0],
                     self.corr[0].shape[1], w, dsp, out, P, self.radius)
            return out.view(b, h, w, -1).permute(0, 3, 1, 2)
        dsp = disp.reshape(-1, 1).float()
        out = []
        for i in range(self.num_levels):
            sc = 1.0 / 2 ** i
            out.append(self._sample(self.geo[i], dsp * sc + self.dx))
            out.append(self._sample(self.corr[i], self.coords * sc - dsp * sc + self.dx))
        out = torch.cat(out, 1).to(self.out_dtype)
        return out.view(b, h, w, -1).permute(0, 3, 1, 2)
