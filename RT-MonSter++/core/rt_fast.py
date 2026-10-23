"""RT-MonSter++ general inference path (any size, no per-size compile / tune, no custom kernels needed).

prepare_rt(model, half=True): fp16 weights (no per-call autocast casts), channels-last (cuDNN NHWC convs), DAv2-S attention via SDPA (dinov2_layers/attention.py).
forward_fast(model, image1, image2): upstream test-mode forward (coarse-to-fine 1/16 -> 1/8 -> 1/4, iterations 1/1/2)
  under fp16 autocast, including the mono branch (upstream forces fp32 there); disparity states, the geometry volume
  lookups (Combined_Geo_Encoding_Volume, fp32 as upstream), the mono scale/shift fit, warps and the final upsampling
  softmax stay fp32. Only the outputs used in test mode are computed (no intermediate full-res upsampling).
"""
import torch
import torch.nn.functional as F

from core.monster import compute_scale_shift, get_cur_disp_range_samples
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import build_gwc_volume, build_gwc_volume_selective, disparity_regression, context_upsample, \
    context_upsample_2x
from core.warp import disp_warp


def prepare_rt(model, half=True, channels_last=True):
    """half: fp16 weights once (autocast would re-cast every fp32 weight in every call: ~20 % of the time)."""
    model.eval()
    if half:
        model.half()
    model._cl = channels_last
    for m in (model.modules() if channels_last else ()):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        elif isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    return model


def _f32():
    return torch.autocast("cuda", enabled=False)


def _up4(self, disp, mask_feat, stem_2x):
    xspx = self.spx_2_gru(mask_feat, stem_2x)
    spx = F.softmax(self.spx_gru(xspx).float(), 1)
    with _f32():
        return context_upsample(disp.float() * 4., spx).unsqueeze(1)


def _up2(self, disp, mask_feat, stem, i):
    xspx = self.spx_gru_2x[i](mask_feat, stem)
    spx = F.softmax(self.spx_gru_list[i](xspx).float(), 1)
    with _f32():
        return context_upsample_2x(disp.float() * 2., spx).unsqueeze(1)


@torch.no_grad()
def forward_fast(self, image1, image2, iters=4, amp=True):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        mf = torch.channels_last if getattr(self, "_cl", True) else torch.contiguous_format
        return _forward(self, image1.contiguous(memory_format=mf), image2.contiguous(memory_format=mf))


def _forward(self, image1, image2):
    depth_mono, features_mono_left, features_mono_right = self.infer_mono(image1, image2)
    depth_mono = depth_mono.float()
    image1 = (2 * (image1 / 255.0) - 1.0)
    image2 = (2 * (image2 / 255.0) - 1.0)
    size = (image1.shape[-2], image1.shape[-1])
    disp_mono_list = [F.interpolate(depth_mono, size=tuple(x // s for x in size), mode='bilinear', align_corners=False)
                      for s in (16, 8, 4)]

    features_left = self.feat_transfer(features_mono_left)
    features_right = self.feat_transfer(features_mono_right)
    image = torch.cat([image1, image2], 0)
    stem_2x = self.stem_2(image)
    stem_4x = self.stem_4(stem_2x)
    stem_8x = self.stem_8(stem_4x)
    stem_16x = self.stem_16(stem_8x)
    stem_2x, stem_2y = torch.chunk(stem_2x, 2, dim=0)
    stem_4x, stem_4y = torch.chunk(stem_4x, 2, dim=0)
    stem_8x, stem_8y = torch.chunk(stem_8x, 2, dim=0)
    stem_16x, stem_16y = torch.chunk(stem_16x, 2, dim=0)
    stem_x_list = [stem_16x, stem_8x, stem_4x, stem_2x]
    stem_y_list = [stem_16y, stem_8y, stem_4y, stem_2y]
    iters_list = [1, 1, 2]

    cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
    net_list_mono = [x.clone() for x in net_list]

    disp_next = None
    for i in range(3):
        feat_left = torch.cat((features_left[-1 - i], stem_x_list[i]), 1)
        feat_right = torch.cat((features_right[-1 - i], stem_y_list[i]), 1)
        match_left = self.desc(self.conv_list[i](feat_left))
        match_right = self.desc(self.conv_list[i](feat_right))
        if i == 0:
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp // 16, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att_list[i](gwc_volume, feat_left)
            geo_encoding_volume = self.cost_agg[i](gwc_volume)
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1).float(), dim=1)
            cur_disp = disparity_regression(prob, self.args.max_disp // 16)
            del prob
        else:
            with _f32():
                drs = get_cur_disp_range_samples(cur_disp=disp_next.squeeze(1), ndisp=self.ndisps[i - 1],
                                                 disp_interval_pixel=self.disp_interval_pixel[i - 1],
                                                 shape=[match_left.shape[0], match_left.shape[2], match_left.shape[3]])
                gwc_volume = build_gwc_volume_selective(match_left.float(), match_right.float(), drs, self.ndisps[i - 1], 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att_list[i](gwc_volume, feat_left)
            geo_encoding_volume = self.cost_agg[i](gwc_volume)
            cur_disp = disp_next
        del gwc_volume
        with _f32():
            geo_fn = Combined_Geo_Encoding_Volume(match_left.float(), match_right.float(), geo_encoding_volume.float(),
                                                  radius=self.args.corr_radius[i])
        b, c, h, w = match_left.shape
        coords = torch.arange(w, device=match_left.device, dtype=torch.float32).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = cur_disp.float()
        for itr in range(iters_list[i]):
            with _f32():
                geo_feat = geo_fn(disp, coords) if i == 0 else geo_fn(disp, coords, self.ndisps[i - 1])
                for bb in range(disp.shape[0]):
                    scale, shift = compute_scale_shift(disp_mono_list[i][bb].clone().squeeze(1).float(),
                                                       disp[bb].clone().squeeze(1).float())
                    disp_mono_list[i][bb] = scale * disp_mono_list[i][bb] + shift
            kw = dict(iter16=(self.args.n_gru_layers == 3), iter08=(self.args.n_gru_layers >= 2 and i >= 1),
                      iter04=(self.args.n_gru_layers >= 1 and i >= 2))
            if i == 2 and itr == iters_list[2] - 1:  # last 1/4 iteration: dual-branch update
                with _f32():
                    fr, fl = feat_right.float(), feat_left.float()
                    flaw_mono = disp_warp(fr, disp_mono_list[i].clone())[0] - fl
                    flaw_stereo = disp_warp(fr, disp.clone())[0] - fl
                    geo_feat_mono = geo_fn(disp_mono_list[i], coords, self.ndisps[i - 1])
                net_list, mask_feat, delta_disp = self.update_block_mix_stereo(
                    net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_list[i], geo_feat_mono,
                    iter16=False, iter08=False)
            else:
                net_list, mask_feat, delta_disp = self.update_block_list[i](net_list, inp_list, geo_feat, disp,
                                                                            disp_mono_list[i], **kw)
            disp = disp + delta_disp.float()
        disp_next = _up2(self, disp, mask_feat, stem_x_list[i + 1], i) if i < 2 else None
    return _up4(self, disp, mask_feat, stem_2x)
