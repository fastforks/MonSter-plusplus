"""MonSter general inference path (any size, prebuilt kernels, no per-size compile / tune). Port of IGEV-general 858fdb4:

prepare_monster(model, remp32=True): IGEV-general prepare_inference (pure fp16 NHWC, fused conv epilogues, CUTLASS
  fused ConvGRUs, fp16-acc cuDNN) for the whole model incl. the Depth Anything V2 branch (ViT attention via SDPA, see
  Depth-Anything-V2-list3 dinov2_layers/attention.py), merged z|r convs also for the two mix update blocks.
  remp32: the full-resolution REMP refinement stays an fp32 copy (as upstream).
forward_fast(model, image1, image2, iters=32): upstream test-mode schedule (iters-7 IGEV updates, then 7 mixed
  stereo/mono updates, mono scale/shift fit once, REMP once) with fp32 disparity states, warps and lookups coordinates;
  upsampling only where the result is used (upstream upsamples the mono disparity in every mixed iteration).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.inference import prepare_inference, prepare_context, update_block_fast, _conv_gru_fast, conv_ep, epilogue, ACT
from core.geometry import FastGeoEncodingVolume
from core.submodule import build_gwc_volume_fast, disparity_regression, context_upsample
from core.warp import disp_warp


def prepare_monster(model, remp32=None):
    """MONSTER_FAST=friendly: prepare_friendly (plain PyTorch, exportable ops) instead of the kernel path below.
    remp32: REMP as an fp32 copy (upstream precision). Default from MONSTER_REMP: fast (default) = fp16 NHWC convs with
    fused BN / activation epilogues (remp_fast: warps, flaws and the final disparity add in fp32), fp32 = previous path."""
    import os
    if os.environ.get("MONSTER_FAST", "max") == "friendly":
        return prepare_friendly(model)
    if remp32 is None:
        remp32 = os.environ.get("MONSTER_REMP", "fast") == "fp32"
    model._remp_fast = not remp32
    if remp32:
        model._remp32 = [copy.deepcopy(model.REMP).float().eval()]  # list: not a submodule (kept out of the fp16 cast)
    for blk in (model.update_block_mix_stereo, model.update_block_mix_mono):
        for g in (blk.gru04, blk.gru08, blk.gru16):
            zr = nn.Conv2d(g.convz.in_channels, 2 * g.convz.out_channels, 3, padding=1, bias=False)
            zr.weight.data = torch.cat([g.convz.weight.data, g.convr.weight.data], 0)
            g.convzr = zr.to(g.convz.weight.device)
    out = prepare_inference(model, half=True, half_acc=True, channels_last=True, kernels=True)
    _fastbb().install_vit(model.mono_encoder, __import__("os").environ.get("FASTBB_VIT", "hh-g1"))  # fastbb ViT path (FASTBB_VIT env: spec, "off" = previous path)
    return out


def prepare_friendly(model):
    """Upstream-friendly tier (plain PyTorch ops only, ONNX/TRT-exportable): fp16 weights, channels-last, DAv2 ViT
    through fastbb sdpa-g1 (FASTBB_VIT, plain SDPA), upstream module forwards; forward_friendly keeps the disparity
    states, lookups coordinates, warps and REMP's flaw / residual add in fp32."""
    import os
    model.eval()
    model.half()
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    vit = os.environ.get("FASTBB_VIT", "sdpa-g1")
    if vit != "off":
        _fastbb().install_vit(model.mono_encoder, vit)
    model._friendly = True
    return model


def _fastbb():
    """shared fast-backbone package (pip install git+https://github.com/fastforks/fastbb)."""
    import fastbb
    return fastbb


def _ctx_block(model, c_list, blk):
    """(czr, cq) per level for an update block from the shared context conv outputs c_list (+ that block's biases)."""
    out = []
    for c, g in zip(c_list, (blk.gru04, blk.gru08, blk.gru16)):
        Hd = c.shape[1] // 3
        bz, br, bq = g.convz.bias.float(), g.convr.bias.float(), g.convq.bias.float()
        czr = epilogue(c[:, :2 * Hd].contiguous(memory_format=torch.channels_last), None, torch.cat([bz, br]).contiguous(), 0)
        cq = epilogue(c[:, 2 * Hd:].contiguous(memory_format=torch.channels_last), None, bq.contiguous(), 0)
        out.append((czr, cq))
    return out


def _update_mix_fast(blk, net, ctx, flaw_stereo, disp, corr, flaw_mono, disp_mono, corr_mono, compute_mask):
    """BasicMultiUpdateBlock_mix2.forward (3 levels) with fused GRUs; the mix motion encoder runs on patched convs."""
    _conv_gru_fast(blk.gru16, net[2], ctx[2], [("pool2x", net[1])])
    _conv_gru_fast(blk.gru08, net[1], ctx[1], [("pool2x", net[0]), ("interp", net[2])])
    mf = blk.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono).contiguous(memory_format=torch.channels_last)
    _conv_gru_fast(blk.gru04, net[0], ctx[0], [mf, ("interp", net[1])])
    d = conv_ep(blk.disp_head.conv1, net[0], act=ACT["relu"])
    delta = conv_ep(blk.disp_head.conv2, d)
    mask = conv_ep(blk.mask_feat_4[0], net[0], act=ACT["relu"]) if compute_mask else None
    return mask, delta


def _pad_cin(conv, cpad):
    w = conv.weight.data
    if w.shape[1] < cpad:
        conv.weight.data = torch.cat([w, w.new_zeros(w.shape[0], cpad - w.shape[1], *w.shape[2:])], 1).contiguous(
            memory_format=torch.channels_last)


def _mix_fill(enc, d_own, c_own, d_oth, c_oth):
    """BasicMotionEncoder_mix2 on [corr | flaw | 0] buffers built once per iteration (no cats, no cuDNN channel padding);
    conv / conv_mono outputs and both disparities written straight into the GRU's [h | x] / [r*h | x] buffers."""
    R = ACT["relu"]
    _, _, hh, ww = d_own.shape
    for c in (enc.convc1, enc.convc1_mono):
        _pad_cin(c, c_own.shape[1])
    outs = []
    for c1, c2, dd1, dd2, cv, corr, d in ((enc.convc1, enc.convc2, enc.convd1, enc.convd2, enc.conv, c_own, d_own),
                                         (enc.convc1_mono, enc.convc2_mono, enc.convd1_mono, enc.convd2_mono, enc.conv_mono,
                                          c_oth, d_oth)):
        cd = torch.empty((1, 128, hh, ww), device=d.device, dtype=d.dtype, memory_format=torch.channels_last)
        cd[:, :64].copy_(conv_ep(c2, conv_ep(c1, corr, act=R), act=R))
        cd[:, 64:].copy_(conv_ep(dd2, conv_ep(dd1, d, act=R), act=R))
        outs.append((conv_ep(cv, cd, act=R, keep_pad=True), d))

    def fill(bufs, c0):
        for b in bufs:
            for j, (o, d) in enumerate(outs):
                b[:, c0 + 64 * j:c0 + 64 * j + 63].copy_(o[:, :63])
                b[:, c0 + 64 * j + 63:c0 + 64 * j + 64].copy_(d)
    return ("fill", fill, 128)


def _update_mix_fast2(blk, net, ctx, d_own, c_own, d_oth, c_oth, compute_mask):
    _conv_gru_fast(blk.gru16, net[2], ctx[2], [("pool2x", net[1])])
    _conv_gru_fast(blk.gru08, net[1], ctx[1], [("pool2x", net[0]), ("interp", net[2])])
    _conv_gru_fast(blk.gru04, net[0], ctx[0], [_mix_fill(blk.encoder, d_own, c_own, d_oth, c_oth), ("interp", net[1])])
    d = conv_ep(blk.disp_head.conv1, net[0], act=ACT["relu"])
    delta = conv_ep(blk.disp_head.conv2, d)
    mask = conv_ep(blk.mask_feat_4[0], net[0], act=ACT["relu"]) if compute_mask else None
    return mask, delta


def _corr_flaw(geo, fr0f, fl0f, disp, buf):
    """[geo lookup | warp(fr0, disp) - fl0 | 0] as one fp16 NHWC buffer (channels padded to a multiple of 8)."""
    g = geo.shape[1]
    if buf is None:
        c = (g + fl0f.shape[1] + 7) // 8 * 8
        buf = torch.empty((1, c, geo.shape[2], geo.shape[3]), device=geo.device, dtype=torch.float16,
                          memory_format=torch.channels_last).zero_()
    buf[:, :g].copy_(geo)
    buf[:, g:g + fl0f.shape[1]].copy_(_warp(fr0f, disp) - fl0f)
    return buf


def _gwc(ml, mr, D, G=8):
    """build_gwc_volume_fast as one kernel (NDHWC fp16 output); falls back to the PyTorch loop for other layouts."""
    B, C, H, W = ml.shape
    cl = torch.channels_last
    if B != 1 or C % (2 * G) or ml.dtype != torch.float16 or not (ml.is_contiguous(memory_format=cl) and mr.is_contiguous(memory_format=cl)):
        return build_gwc_volume_fast(ml, mr, D, G, True)
    from core.inference import kern
    k = kern()
    v = torch.empty((1, G, D, H, W), device=ml.device, dtype=ml.dtype, memory_format=torch.channels_last_3d)
    k.launch("gwc_volume", k.linear(H * W * D), ml, mr, H, W, C, G, D, v)
    return v


def _warp(img, disp):
    """core.warp.disp_warp(img, disp)[0] (border padding, same coordinate formula, fp32) without the unused valid mask."""
    b, _, h, w = img.shape
    xs = torch.arange(w, device=img.device, dtype=torch.float32).view(1, 1, w)
    gx = 2 * ((xs - disp[:, 0].float()) / (w - 1)) - 1
    gy = (2 * (torch.arange(h, device=img.device, dtype=torch.float32) / (h - 1)) - 1).view(1, h, 1).expand_as(gx)
    return F.grid_sample(img.float(), torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border")


def _cbr(seq_or_basic, x, act):
    """conv (no bias) + BN + activation as one conv_ep (BN as fp32 per-channel affine in the epilogue)."""
    if isinstance(seq_or_basic, nn.Sequential):   # refinement.conv2d: Conv2d, BN, LeakyReLU(0.2)
        c, bn = seq_or_basic[0], seq_or_basic[1]
    else:                                          # BasicConv_now: conv, bn, relu
        c, bn = seq_or_basic.conv, seq_or_basic.bn
    return conv_ep(c, x, s=bn._s, t=bn._t, act=act)


def _c2x(m, x, rem):
    """Conv2x_now (concat): conv1 (stride 2 / deconv 4x4 s2) + BN + ReLU, cat with rem, conv2 + BN + ReLU."""
    x = _cbr(m.conv1, x, ACT["relu"])
    return _cbr(m.conv2, torch.cat([x, rem], 1), ACT["relu"])


def remp_fast(R, disp_mono, disp_stereo, left, right):
    """REMP.forward with fp16 NHWC convs (fp16-acc cuDNN / CUTLASS, BN + activation fused into one epilogue pass);
    warps, flaws and the final residual disparity add stay fp32. left / right: fp32 images in [-1, 1]."""
    relu, lk = ACT["relu"], ACT["leaky02"]
    fm = _h16(torch.cat([_warp(right, disp_mono) - left, left], 1))
    fs = _h16(torch.cat([_warp(right, disp_stereo) - left, left], 1))
    x = torch.cat([_cbr(R.conv1_mono, fm, lk), _cbr(R.conv2_mono, _h16(disp_mono), lk),
                   _cbr(R.conv1_stereo, fs, lk), _cbr(R.conv2_stereo, _h16(disp_stereo), lk)], 1)
    del fm, fs
    x = _cbr(R.conv_start, x, relu)
    u = R.RefinementBlock
    rem0 = x
    x = _cbr(u.conv1a, x, relu); rem1 = x
    x = _cbr(u.conv2a, x, relu); rem2 = x
    x = _cbr(u.conv3a, x, relu); rem3 = x
    x = _cbr(u.conv4a, x, relu); rem4 = x
    x = _c2x(u.deconv4a, x, rem3); rem3 = x
    x = _c2x(u.deconv3a, x, rem2); rem2 = x
    x = _c2x(u.deconv2a, x, rem1); rem1 = x
    x = _c2x(u.deconv1a, x, rem0); rem0 = x
    x = _c2x(u.conv1b, x, rem1); rem1 = x
    x = _c2x(u.conv2b, x, rem2); rem2 = x
    x = _c2x(u.conv3b, x, rem3); rem3 = x
    x = _c2x(u.conv4b, x, rem4)
    x = _c2x(u.deconv4b, x, rem3)
    x = _c2x(u.deconv3b, x, rem2)
    x = _c2x(u.deconv2b, x, rem1)
    x = _c2x(u.deconv1b, x, rem0)
    del rem0, rem1, rem2, rem3, rem4
    low = R.LFE(x.float().mean((2, 3), keepdim=True).half())             # AP + LFE (tiny)
    l0, l1, l2 = R.LMC[0], R.LMC[1], R.LMC[3]
    motif = conv_ep(l2, conv_ep(l1, conv_ep(l0, x), act=relu), act=ACT["sigmoid"])
    x = torch.lerp(low.expand_as(x), x, motif)                              # (1 - motif) * low + motif * x
    x = conv_ep(R.final_conv, x.contiguous(memory_format=torch.channels_last))
    return F.leaky_relu(disp_stereo + x.float(), 0.01)


def remp_friendly(R, disp_mono, disp_stereo, left, right):
    """REMP.forward in fp16 through the module forwards (plain PyTorch); warps / flaws / residual add in fp32."""
    cl = torch.channels_last
    fm = torch.cat([_warp(right, disp_mono) - left, left], 1).half().contiguous(memory_format=cl)
    fs = torch.cat([_warp(right, disp_stereo) - left, left], 1).half().contiguous(memory_format=cl)
    x = torch.cat([R.conv1_mono(fm), R.conv2_mono(disp_mono.half()), R.conv1_stereo(fs), R.conv2_stereo(disp_stereo.half())], 1)
    x = R.RefinementBlock(R.conv_start(x))
    low = R.LFE(R.AP(x))
    motif = R.LMC(x)
    x = R.final_conv((1 - motif) * low + motif * x)
    return F.leaky_relu(disp_stereo + x.float(), 0.01)


@torch.no_grad()
def forward_friendly(self, image1, image2, iters=32):
    """forward_fast's schedule with upstream module forwards (fp16, channels-last), plain PyTorch only."""
    from core.monster import compute_scale_shift
    cl = torch.channels_last
    D = self.args.max_disp // 4
    im1 = 2 * (image1 / 255.0) - 1.0
    im2 = 2 * (image2 / 255.0) - 1.0
    i1, i2 = (x.half().contiguous(memory_format=cl) for x in (im1, im2))
    depth_mono, fml, fmr = self.infer_mono(i1, i2)
    size = (int(depth_mono.shape[-2] * 0.25), int(depth_mono.shape[-1] * 0.25))
    disp_mono_4x = F.interpolate(depth_mono.float(), size=size, mode='bilinear', align_corners=False)
    del depth_mono
    features_right = self.feat_transfer(fmr)
    del fmr
    fr0 = torch.cat((features_right[0], self.stem_4(self.stem_2(i2))), 1)
    del features_right
    match_right = self.desc(self.conv(fr0))
    features_left = self.feat_transfer(fml)
    stem_2x = self.stem_2(i1)
    stem_4x = self.stem_4(stem_2x)
    stem_8x = self.stem_8(stem_4x)
    stem_16x = self.stem_16(stem_8x)
    features_left[0] = torch.cat((features_left[0], stem_4x), 1)
    match_left = self.desc(self.conv(features_left[0]))
    gwc_volume = build_gwc_volume_fast(match_left, match_right, D, 8, True)
    gwc_volume = self.corr_stem(gwc_volume)
    gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
    geo_encoding_volume = self.cost_agg(gwc_volume, features_left)
    del gwc_volume
    prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1).float(), dim=1)
    init_disp = disparity_regression(prob, D)
    del prob
    fl0 = features_left[0]
    del features_left
    cnet_list = self.feat_transfer_cnet(fml, [stem_16x, stem_8x, stem_4x])
    del fml, stem_8x, stem_16x, stem_4x
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    del cnet_list
    inp_list = [list(conv(i).split(conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
    net_mono = [x.clone() for x in net_list]
    geo_fn = FastGeoEncodingVolume(match_left, match_right, geo_encoding_volume, radius=self.args.corr_radius,
                                   num_levels=self.args.corr_levels, out_dtype=torch.float16, use_kernel=False)
    del geo_encoding_volume, match_left, match_right
    fl0f, fr0f = fl0.float(), fr0.float()
    del fl0, fr0
    disp = init_disp.float()
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h = disp.half()
        if itr <= iters - 8:
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp_h)
        else:
            if itr == iters - 7:
                for i in range(disp.shape[0]):
                    scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze(1), disp[i].clone().squeeze(1))
                    disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
            flaw_mono = (_warp(fr0f, disp_mono_4x) - fl0f).half()
            flaw_stereo = (_warp(fr0f, disp) - fl0f).half()
            geo_feat_mono = geo_fn(disp_mono_4x)
            dm_h = disp_mono_4x.half()
            net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(net_list, inp_list, flaw_stereo, disp_h, geo_feat,
                                                                             flaw_mono, dm_h, geo_feat_mono)
            net_mono, mask_mono, delta_mono = self.update_block_mix_mono(net_mono, inp_list, flaw_mono, dm_h, geo_feat_mono,
                                                                         flaw_stereo, disp_h, geo_feat)
            disp_mono_4x = disp_mono_4x + delta_mono.float()
        disp = disp + delta_disp.float()

    def up(dl, mask):
        xspx = self.spx_2_gru(mask, stem_2x)
        return context_upsample(dl * 4., F.softmax(self.spx_gru(xspx).float(), 1)).unsqueeze(1)

    disp_up = up(disp, mask_feat_4)
    disp_mono_up = up(disp_mono_4x, mask_mono)
    return disp_up + remp_friendly(self.REMP, disp_mono_up, disp_up, im1, im2)


def _h16(x):
    return x.half().contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def forward_fast(self, image1, image2, iters=32):
    if getattr(self, "_friendly", False):
        return forward_friendly(self, image1, image2, iters)
    from core.monster import compute_scale_shift
    mf2 = torch.channels_last
    D = self.args.max_disp // 4
    im1 = 2 * (image1 / 255.0) - 1.0
    im2 = 2 * (image2 / 255.0) - 1.0
    i1, i2 = _h16(im1), _h16(im2)

    depth_mono, fml, fmr = self.infer_mono(i1, i2)
    depth_mono = depth_mono.float()
    size = (int(depth_mono.shape[-2] * 0.25), int(depth_mono.shape[-1] * 0.25))
    disp_mono_4x = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False)
    del depth_mono

    features_right = self.feat_transfer(fmr)
    del fmr
    stem_4y = self.stem_4(self.stem_2(i2))
    features_right[0] = torch.cat((features_right[0], stem_4y), 1)
    del stem_4y
    match_right = self.desc(self.conv(features_right[0]))
    fr0 = features_right[0]
    del features_right

    features_left = self.feat_transfer(fml)
    stem_2x = self.stem_2(i1)
    stem_4x = self.stem_4(stem_2x)
    stem_8x = self.stem_8(stem_4x)
    stem_16x = self.stem_16(stem_8x)
    features_left[0] = torch.cat((features_left[0], stem_4x), 1)
    match_left = self.desc(self.conv(features_left[0]))

    gwc_volume = _gwc(match_left, match_right, D)
    gwc_volume = self.corr_stem(gwc_volume)
    gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
    geo_encoding_volume = self.cost_agg(gwc_volume, features_left)
    del gwc_volume
    prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1).float(), dim=1)
    init_disp = disparity_regression(prob, D)
    del prob
    fl0 = features_left[0]
    del features_left

    cnet_list = self.feat_transfer_cnet(fml, [stem_16x, stem_8x, stem_4x])
    del fml, stem_8x, stem_16x, stem_4x
    net_list = [torch.tanh(x[0]).contiguous(memory_format=mf2) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]  # (upstream applies relu twice: idempotent)
    del cnet_list
    c_list = [conv_ep(conv, x) for x, conv in zip(inp_list, self.context_zqr_convs)]
    ctx = prepare_context(self, inp_list)
    ctx_ms = _ctx_block(self, c_list, self.update_block_mix_stereo)
    ctx_mm = _ctx_block(self, c_list, self.update_block_mix_mono)
    del inp_list, c_list
    net_mono = [x.clone() for x in net_list]

    geo_fn = FastGeoEncodingVolume(match_left, match_right, geo_encoding_volume, radius=self.args.corr_radius,
                                   num_levels=self.args.corr_levels, out_dtype=torch.float16, use_kernel=True)
    del geo_encoding_volume, match_left, match_right
    fl0f, fr0f = fl0.float(), fr0.float()  # warps in fp32 (upstream precision; fp16 sampling coordinates are too coarse)
    del fl0, fr0

    b, _, h, w = init_disp.shape
    disp = init_disp.float()
    disp_h = torch.empty((b, 1, h, w), device=disp.device, dtype=torch.float16, memory_format=mf2)
    dm_h = torch.empty_like(disp_h)
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h.copy_(disp)
        last = itr == iters - 1
        if itr <= iters - 8:
            net_list, mask_feat_4, delta_disp = update_block_fast(self.update_block, net_list, ctx, geo_feat, disp_h, last)
        else:
            if itr == iters - 7:
                for i in range(b):
                    scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze(1), disp[i].clone().squeeze(1))
                    disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
            dm_h.copy_(disp_mono_4x)
            if b == 1 and geo_feat.dtype == torch.float16:
                cs = _corr_flaw(geo_feat, fr0f, fl0f, disp, cs if itr > iters - 7 else None)
                cm = _corr_flaw(geo_fn(disp_mono_4x), fr0f, fl0f, disp_mono_4x, cm if itr > iters - 7 else None)
                mask_feat_4, delta_disp = _update_mix_fast2(self.update_block_mix_stereo, net_list, ctx_ms, disp_h, cs, dm_h, cm, last)
                mask_mono, delta_mono = _update_mix_fast2(self.update_block_mix_mono, net_mono, ctx_mm, dm_h, cm, disp_h, cs, last)
            else:
                flaw_mono = _h16(_warp(fr0f, disp_mono_4x) - fl0f)
                flaw_stereo = _h16(_warp(fr0f, disp) - fl0f)
                geo_feat_mono = geo_fn(disp_mono_4x)
                mask_feat_4, delta_disp = _update_mix_fast(self.update_block_mix_stereo, net_list, ctx_ms, flaw_stereo, disp_h,
                                                           geo_feat, flaw_mono, dm_h, geo_feat_mono, last)
                mask_mono, delta_mono = _update_mix_fast(self.update_block_mix_mono, net_mono, ctx_mm, flaw_mono, dm_h,
                                                         geo_feat_mono, flaw_stereo, disp_h, geo_feat, last)
            disp_mono_4x = disp_mono_4x + delta_mono.float()
        disp = disp + delta_disp.float()

    def up(dl, mask):
        xspx = self.spx_2_gru(mask, stem_2x)
        return context_upsample(dl * 4., F.softmax(self.spx_gru(xspx).float(), 1)).unsqueeze(1)

    disp_up = up(disp, mask_feat_4)
    disp_mono_up = up(disp_mono_4x, mask_mono)
    if getattr(self, "_remp_fast", False):
        refine = remp_fast(self.REMP, disp_mono_up, disp_up, im1, im2)
    elif getattr(self, "_remp32", None):
        refine = self._remp32[0](disp_mono_up, disp_up, im1, im2)
    else:
        refine = self.REMP(_h16(disp_mono_up), _h16(disp_up), i1, i2).float()
    return disp_up + refine
