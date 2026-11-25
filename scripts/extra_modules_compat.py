"""
Compatibility layer for the rice-grain model from
https://github.com/BearHuntingExpert/Rice-Grain-Phenotyping-System (best.pt, MIT).

That checkpoint was saved from a modified Ultralytics (8.3.114) whose custom layers live in
`ultralytics.nn.extra_modules.*` and `timm.layers.cond_conv2d`. Those modules are not part of
stock Ultralytics, so loading fails. This file re-implements the forward passes of those layers
(the trained weights come from the checkpoint itself) and registers them under the original
module paths. Import this module once before `YOLO(path_to_that_best_pt)`.

Re-implemented from the public reference designs:
- DynamicConv / CondConv2d: ParameterNet (Huawei Noah) + timm CondConv2d
- GhostModule: GhostNet
- LocalWindowAttention / CascadedGroupAttention / Conv2d_BN: EfficientViT (Microsoft)
- PST / PSAttnBlock / PSAttn: Pyramid Sparse Transformer (inlmouse/P2HCT)
"""
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- timm CondConv2d
class CondConv2d(nn.Module):
    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        weight = weight.view((B * self.out_channels, self.in_channels // self.groups) + tuple(self.kernel_size))
        bias = None
        if getattr(self, "bias", None) is not None:
            bias = torch.matmul(routing_weights, self.bias).view(B * self.out_channels)
        x = x.reshape(1, B * C, H, W)
        out = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding,
                       dilation=self.dilation, groups=self.groups * B)
        return out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])


# ---------------------------------------------------------------- block.py layers
class DynamicConv_Single(nn.Module):
    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing(pooled))
        return self.cond_conv(x, routing_weights)


class DynamicConv(nn.Module):
    def forward(self, x):
        return self.conv(x)


class GhostModule(nn.Module):
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class C2f_GhostDynamicConv(nn.Module):
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class PSAttn(nn.Module):
    def forward(self, x, upper_feat):
        B, C, H, W = x.shape
        N = H * W
        _, _, H_up, W_up = upper_feat.shape

        q = self.q(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        kv = self.kv(upper_feat).view(B, self.num_heads, 2 * self.head_dim, H_up * W_up).permute(0, 1, 3, 2)
        k, v = kv.split(self.head_dim, dim=3)

        sim = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = sim.softmax(dim=-1)
        out = attn @ v
        # This checkpoint has topk=0, so the fine (top-k) branch of the original is never used.
        if 0 < self.topk <= H_up * W_up:
            raise NotImplementedError("PSAttn fine branch (topk > 0) is not needed for this checkpoint")

        out = out.transpose(2, 3).reshape(B, self.all_head_dim, H, W)
        v_reshaped = v.transpose(2, 3).reshape(B, self.all_head_dim, H_up, W_up)
        v_pe = F.interpolate(self.pe(v_reshaped), size=(H, W), mode="bilinear", align_corners=False)
        return self.proj(out + v_pe)


class PSAttnBlock(nn.Module):
    def forward(self, x, upper_feat):
        x = x + self.attn(x, upper_feat)
        return x + self.mlp(x)


class PST(nn.Module):
    def forward(self, x):
        upper_feat = self.cvup(x[1])
        y = [self.cv1(x[0])]
        for i in range(self.num_layers):
            y.append(getattr(self, f"attnlayer_{i}")(y[-1], upper_feat))
        return self.cv2(torch.cat(y, 1))


# PSAttn* aliases that some forks also pickle
PSAttention = PSAttn


# ---------------------------------------------------------------- attention.py layers
class Conv2d_BN(nn.Sequential):
    pass  # children are `c` then `bn`; Sequential runs them in order


class CascadedGroupAttention(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        ab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = (q.transpose(-2, -1) @ k) * self.scale + ab[i]
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
            feats_out.append(feat)
        return self.proj(torch.cat(feats_out, 1))


class LocalWindowAttention(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        wr = self.window_resolution
        if H <= wr and W <= wr:
            return self.attn(x)
        x = x.permute(0, 2, 3, 1)
        pad_b = (wr - H % wr) % wr
        pad_r = (wr - W % wr) % wr
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        pH, pW = H + pad_b, W + pad_r
        nH, nW = pH // wr, pW // wr
        x = x.view(B, nH, wr, nW, wr, C).transpose(2, 3).reshape(B * nH * nW, wr, wr, C).permute(0, 3, 1, 2)
        x = self.attn(x)
        x = x.permute(0, 2, 3, 1).view(B, nH, nW, wr, wr, C).transpose(2, 3).reshape(B, pH, pW, C)
        if pad_b or pad_r:
            x = x[:, :H, :W].contiguous()
        return x.permute(0, 3, 1, 2)


# ---------------------------------------------------------------- register module paths
def _register(name, **attrs):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as package so submodules resolve
        sys.modules[name] = mod
    for k, v in attrs.items():
        if not hasattr(mod, k):
            v.__module__ = name
            setattr(mod, k, v)
    return mod


def install():
    try:
        import timm.layers.cond_conv2d  # noqa: F401  (use the real one if timm is installed)
    except ImportError:
        _register("timm")
        _register("timm.layers")
        _register("timm.layers.cond_conv2d", CondConv2d=CondConv2d)

    import ultralytics.nn  # noqa: F401  (make sure the parent package exists)
    _register("ultralytics.nn.extra_modules")
    _register("ultralytics.nn.extra_modules.block",
              DynamicConv_Single=DynamicConv_Single, DynamicConv=DynamicConv,
              GhostModule=GhostModule, C2f_GhostDynamicConv=C2f_GhostDynamicConv,
              PSAttn=PSAttn, PSAttnBlock=PSAttnBlock, PST=PST)
    _register("ultralytics.nn.extra_modules.attention",
              Conv2d_BN=Conv2d_BN, CascadedGroupAttention=CascadedGroupAttention,
              LocalWindowAttention=LocalWindowAttention)


install()
