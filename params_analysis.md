# `Scal3R` Model Parameter Analysis

This document analyzes the parameters owned by `Scal3R` and what happens to them during release inference. The key point is:

```text
Checkpoint/model parameters are loaded and used in eval mode.
They are not persistently optimized during inference.
Only temporary TTT fast-weight tensors are recomputed/updated inside an inference pass.
```

## 1. Model Construction Entry Point

Files:

- `scal3r/models/scal3r.py`
- `configs/models/scal3r.yaml`
- `configs/base.yaml`

`build_sampler_from_config(...)` constructs the model as:

```text
load_config(config_path)
  -> extract_sampler_cfg(cfg)
  -> Scal3R(sampler_cfg)
  -> load_checkpoint(...)
  -> sampler.to(device)
  -> sampler.eval()
```

The checkpoint is loaded strictly into the full model:

```python
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
model.load_state_dict(state_dict, strict=True)
```

So every checkpointed parameter must match the instantiated module structure.

## 2. Top-Level `Scal3R` Modules

File: `scal3r/models/scal3r.py`

`Scal3R.__init__` creates four learnable submodules:

```text
Scal3R
├── agg_regator: Aggregator
├── cam_decoder: CameraHead
├── xyz_decoder: DPTHead
└── dpt_decoder: DPTHead
```

It also stores `ttt_order`, which is not a parameter. It is a two-step control sequence:

```text
step 1: update=True,  apply=False
step 2: update=False, apply=True
```

The default release config sets:

```yaml
model_cfg:
  sampler_cfg:
    use_checkpoint: true
    agg_regator_cfg:
      img_size: 518
      patch_size: 14
      embed_dim: 1024
      frame_use_ttt: false
      global_use_ttt: true
      ttt_layer_idx: [14, 17, 20, 23]
      ttt_cfg:
        num_heads: 1
        inter_multi: 4
        base_lr: 0.01
        muon_update_steps: 5
        use_ddp_allreduce: true
        use_modulation: true
      num_global_tokens: 0
```

Because `embed_dim` is present, `Scal3R` forces all decoder input dimensions to `embed_dim * 2 = 2048`. The factor of two matches the concatenation of frame-attention and global-attention intermediate features.

## 3. Aggregator Parameters

Files:

- `scal3r/utils/vggt/models/aggregator.py`
- `scal3r/utils/vggt/layers/block.py`
- `scal3r/utils/ttt_utils.py`

The aggregator owns the backbone-style tokenization and transformer stack.

### 3.1 Patch Embed Parameters

`Aggregator.__build_patch_embed__(...)` creates `self.patch_embed`.

Default release value:

```text
patch_embed = "dinov2_vitl14_reg"
```

So the patch embed path uses a DINOv2-style ViT implementation from `scal3r/utils/vggt/layers/vision_transformer.py`, not the simple convolutional `PatchEmbed` path.

If the patch embed module has a `mask_token`, the code explicitly sets:

```python
self.patch_embed.mask_token.requires_grad_(False)
```

That is the only explicit `requires_grad_(False)` found in the `Scal3R` model construction path. During release inference, however, all parameters are effectively frozen because the model is in eval mode and the backend runs forward inference under `torch.no_grad()`.

### 3.2 Rotary Position Encoding

`self.rope = RotaryPositionEmbedding2D(...)` and `self.position_getter = PositionGetter()` are created when `rope_freq > 0`.

These are positional computation utilities. They are not the same as checkpointed transformer weights. In this code path, the ResNet normalization constants are also registered as non-persistent buffers:

```text
_resnet_mean
_resnet_std
```

Buffers are not trainable parameters.

### 3.3 Frame and Global Transformer Blocks

The aggregator creates two parallel `ModuleList`s:

```text
self.frame_blocks  = [Block(...), ...]  # depth entries
self.global_blocks = [Block(...), ...]  # depth entries
```

Default `depth` is `24`, so there are 24 frame blocks and 24 global blocks.

Each `Block` normally owns:

- `norm1`: layer norm before attention
- `attn`: attention module
- `ls1`: layer-scale parameter module when `init_values` is set
- `norm2`: layer norm before MLP
- `mlp`: feed-forward MLP
- `ls2`: second layer-scale module when `init_values` is set

If camera embedding is enabled, a block can also use `cam_mlp`, but the default release config does not enable camera embedding.

### 3.4 Special Token Parameters

The aggregator always creates:

```text
camera_token   shape: (1, 2, num_camera_tokens, embed_dim)
register_token shape: (1, 2, num_register_tokens, embed_dim)
```

With defaults:

```text
num_camera_tokens = 1
num_register_tokens = 4
embed_dim = 1024
```

So their default shapes are:

```text
camera_token   (1, 2, 1, 1024)
register_token (1, 2, 4, 1024)
```

The leading `2` encodes separate token values for the first frame and the remaining frames.

Optional parameters:

- `global_token`: created only when `num_global_tokens > 0`; default config sets `num_global_tokens: 0`, so this parameter is absent.
- `block_token`: created only when `num_block_tokens > 0`; default is `0`, so this parameter is absent.

### 3.5 TTT Module Parameters Inside Blocks

`Block` creates `self.ttt = FastWeightGluMLPMultihead(...)` only when `use_ttt` is true.

In the default release config:

```text
frame_use_ttt  = false
global_use_ttt = true
ttt_layer_idx  = [14, 17, 20, 23]
```

Therefore:

- no `frame_blocks[i].ttt` modules are created by default;
- `global_blocks[14].ttt`, `global_blocks[17].ttt`, `global_blocks[20].ttt`, and `global_blocks[23].ttt` are created.

Each TTT module owns checkpointed parameters:

```text
qkv      Linear(dim, dim * 3)
proj     Linear(dim, dim * 3) when use_modulation=True; otherwise Linear(dim, dim)
w0       Parameter(num_heads, head_dim, head_dim * inter_multi)
w1       Parameter(num_heads, head_dim * inter_multi, head_dim)
w2       Parameter(num_heads, head_dim, head_dim * inter_multi)
lrs      Linear(dim, num_heads * 3)
o_norm   RMSNorm(head_dim) with affine weight
```

With default TTT config:

```text
dim = 1024
num_heads = 1
head_dim = 1024
inter_multi = 4
hidden fast-weight dim = 4096
```

So the base fast-weight parameter shapes are:

```text
w0: (1, 1024, 4096)
w1: (1, 4096, 1024)
w2: (1, 1024, 4096)
```

These `w0`, `w1`, and `w2` are checkpointed model parameters. They serve as the initial fast weights. They are not overwritten in-place by the release backend.

## 4. Camera Head Parameters

File: `scal3r/utils/vggt/heads/camera_head.py`

`CameraHead` predicts camera parameters from the camera token in the final aggregated token set.

Default constructor settings include:

```text
dim_in = 2048
trunk_depth = 4
pose_encoding_type = "absT_quaR_FoV"
num_heads = 16
mlp_ratio = 4
init_values = 0.01
use_scale = false
```

For `pose_encoding_type == "absT_quaR_FoV"`, the target pose dimension is `9`.

Learnable parameter groups:

- `trunk`: four transformer `Block`s operating on camera tokens.
- `token_norm`: layer norm on camera tokens.
- `trunk_norm`: layer norm before pose prediction.
- `empty_pose_tokens`: learned initial pose token, shape `(1, 1, target_dim)`.
- `embed_pose`: linear embedding from pose encoding to `dim_in`.
- `poseLN_modulation`: SiLU plus linear projection to adaptive LN shift/scale/gate.
- `pose_branch`: MLP from `dim_in` to the target camera encoding.

During forward, the camera head iteratively refines pose predictions. The previous prediction is detached between refinement iterations, but this is only a graph-control detail; release inference is already under `torch.no_grad()`.

## 5. Dense DPT Head Parameters

File: `scal3r/utils/vggt/heads/dpt_head.py`

`Scal3R` creates two DPT heads:

```text
xyz_decoder: predicts dense 3D coordinates + confidence
dpt_decoder: predicts dense depth + confidence
```

Default `xyz_decoder` config:

```text
dim_in = 2048
output_dim = 4
activation = "inv_log"
conf_activation = "expp1"
```

Default `dpt_decoder` config:

```text
dim_in = 2048
output_dim = 2
activation = "exp"
conf_activation = "expp1"
```

Both heads use the same structural parameter groups:

- `norm`: layer norm over incoming token features.
- `projects`: one `1x1` convolution per selected intermediate layer.
- `resize_layers`: transposed/regular convolutions or identity modules to align feature scales.
- `scratch.layer*_rn`: convolutional projection layers for multi-scale fusion.
- `scratch.refinenet*`: feature fusion blocks with residual convolution units.
- `scratch.output_conv1`: final feature projection.
- `scratch.output_conv2`: final prediction head when `feature_only=False`.

Default intermediate layers are:

```text
[4, 11, 17, 23]
```

If `agg_regator_cfg.intermediate_layer_idx` is provided, `Scal3R` passes it into both DPT heads. Otherwise, each DPT head uses its own default.

## 6. What Is Frozen During Inference

Release inference freezes all persistent model parameters in the practical sense:

1. `build_sampler_from_config(...)` calls `sampler.eval()`.
2. `backend.main()` wraps the main model forward in:

   ```python
   with torch.no_grad():
       with torch.amp.autocast(...):
           output = forward(...)
   ```

3. The code does not create a PyTorch optimizer for `Scal3R`.
4. The code does not call `.backward()` on model losses.
5. The code does not assign into `.data` or replace checkpointed `nn.Parameter` objects during inference.

Therefore these persistent parameter groups are frozen during release inference:

- `agg_regator.patch_embed.*`
- `agg_regator.frame_blocks.*`
- ordinary parameters in `agg_regator.global_blocks.*`
- `agg_regator.camera_token`
- `agg_regator.register_token`
- optional `agg_regator.global_token`, if configured
- optional `agg_regator.block_token`, if configured
- checkpointed TTT module parameters:
  - `global_blocks[i].ttt.qkv.*`
  - `global_blocks[i].ttt.proj.*`
  - `global_blocks[i].ttt.w0`
  - `global_blocks[i].ttt.w1`
  - `global_blocks[i].ttt.w2`
  - `global_blocks[i].ttt.lrs.*`
  - `global_blocks[i].ttt.o_norm.*`
- `cam_decoder.*`
- `xyz_decoder.*`
- `dpt_decoder.*`

The only explicit `requires_grad=False` in this construction path is the optional patch-embed `mask_token`. Other parameters may still have `requires_grad=True`, but they are not updated because inference runs in eval/no-grad mode and no optimizer step exists.

## 7. What Is Updated During Inference

The release backend updates temporary TTT fast-weight tensors, not persistent model parameters.

The relevant call path is:

```text
backend.forward(...)
  -> agg_regator.forward_layer(..., enable_ttt=False)
  -> if layer is in ttt_layer_idx:
       backend.apply_ttt(...)
         -> agg_regator.ttt_gradient(...)
         -> agg_regator.ttt_update(...)
         -> agg_regator.ttt_apply(...)
```

### 7.1 Ordinary Aggregator Layers Run Without TTT

`Aggregator.forward_layer(...)` explicitly calls frame/global attention with:

```text
enable_ttt=False
```

So the normal layer pass first runs the checkpointed transformer blocks without invoking the TTT residual path.

### 7.2 TTT Gradient Phase

For each configured TTT layer and each block, `backend.apply_ttt(...)` calls:

```text
model.agg_regator.ttt_gradient(...)
```

This eventually calls `FastWeightGluMLPMultihead.gradient(...)`, which:

1. Projects patch tokens through checkpointed `qkv`.
2. Computes learned local learning rates with checkpointed `lrs`.
3. Starts from either cached fast weights or repeated checkpointed base fast weights:

   ```text
   w0 = self.w0.repeat(B, 1, 1)
   w1 = self.w1.repeat(B, 1, 1)
   w2 = self.w2.repeat(B, 1, 1)
   ```

4. Computes manual local gradients:

   ```text
   w0_grad
   w1_grad
   w2_grad
   ```

These are ordinary tensors, not optimizer gradients stored in `parameter.grad`.

`backend.apply_ttt(...)` sums these gradients across all chunks/blocks.

### 7.3 TTT Update Phase

After gradients are summed, `backend.apply_ttt(...)` calls:

```text
model.agg_regator.ttt_update(...)
```

This calls `FastWeightGluMLPMultihead.update(...)`, which:

1. Chooses starting fast weights from provided ready weights, cached fast weights, or repeated checkpoint parameters.
2. Records the pre-update norms of `w0`, `w1`, and `w2`.
3. Applies Muon-style orthogonalization with `zeropower_via_newtonschulz5(...)`.
4. Optionally applies an explicit scalar learning rate if one was passed.
5. Computes:

   ```text
   w0 = w0 + w0_grad
   w1 = w1 + w1_grad
   w2 = w2 + w2_grad
   ```

6. Renormalizes each fast-weight tensor back to its pre-update norm.
7. Returns updated tensors:

   ```text
   shared_w0, shared_w1, shared_w2
   ```

These returned tensors are the inference-time updated parameters in the mathematical sense, but they are not registered `nn.Parameter` replacements.

### 7.4 TTT Apply Phase

`backend.apply_ttt(...)` then applies the shared updated tensors to every block:

```text
model.agg_regator.ttt_apply(..., w0=shared_w0, w1=shared_w1, w2=shared_w2)
```

Inside `FastWeightGluMLPMultihead.forward(...)`, the provided `w0_cache`, `w1_cache`, and `w2_cache` take precedence over checkpointed `self.w0`, `self.w1`, and `self.w2`.

The TTT output is added residually to the token state:

```text
tokens += global_blocks[index].ttt(...)
```

The updated token state is then stored back into `agg_state_refs`, and the DPT intermediate state may be updated if this layer is also a decoder feature layer.

### 7.5 Cache Behavior in This Backend

`default_ttt_order()` uses `use_cached=True` and `cache_last=True`, but `backend.apply_ttt(...)` overrides both the gradient and apply orders:

```text
use_cached=False
cache_last=False
```

So in the backend path used by `run.py`, TTT does not persist `last_weights_test` across these explicit `apply_ttt(...)` calls. The updated fast weights are held in local variables and used immediately for the current layer's token update.

## 8. Parameter Status Table

| Parameter group | Persistent checkpoint parameter? | Updated during release inference? | Notes |
| --- | --- | --- | --- |
| `agg_regator.patch_embed.*` | Yes | No | DINOv2/ViT patch embed path by default. Optional `mask_token` is explicitly `requires_grad=False`. |
| `agg_regator.camera_token` | Yes | No | Learned first-frame/rest-frame camera tokens. |
| `agg_regator.register_token` | Yes | No | Learned first-frame/rest-frame register tokens. |
| `agg_regator.global_token` | Optional | No | Absent by default because `num_global_tokens: 0`. |
| `agg_regator.block_token` | Optional | No | Absent by default because `num_block_tokens: 0`. |
| `agg_regator.frame_blocks.*` | Yes | No | Default `frame_use_ttt: false`, so no frame TTT modules are created. |
| `agg_regator.global_blocks.*.attn/mlp/norm/layerscale` | Yes | No | Used to produce token states; no optimizer step. |
| `global_blocks[14/17/20/23].ttt.qkv.*` | Yes | No | Used to compute q/k/v for TTT. |
| `global_blocks[14/17/20/23].ttt.proj.*` | Yes | No | Used to project/modulate TTT output. |
| `global_blocks[14/17/20/23].ttt.lrs.*` | Yes | No | Predicts per-token learning rates for fast-weight update. |
| `global_blocks[14/17/20/23].ttt.o_norm.*` | Yes | No | Normalizes TTT output. |
| `global_blocks[14/17/20/23].ttt.w0/w1/w2` | Yes | No, not in-place | These are base fast weights; temporary repeated copies are updated. |
| temporary `shared_w0/shared_w1/shared_w2` | No | Yes | Created during `apply_ttt(...)`, used immediately, then discarded. |
| `cam_decoder.*` | Yes | No | Iterative camera prediction head. |
| `xyz_decoder.*` | Yes | No | Dense XYZ and confidence prediction head. |
| `dpt_decoder.*` | Yes | No | Dense depth and confidence prediction head. |

## 9. Exact Answer to "Frozen vs Updated"

Frozen during inference:

```text
All registered `Scal3R` model parameters loaded from checkpoint.
```

Updated during inference:

```text
Temporary TTT fast-weight tensors derived from selected global TTT layers:
global_blocks[14].ttt.{w0,w1,w2}
global_blocks[17].ttt.{w0,w1,w2}
global_blocks[20].ttt.{w0,w1,w2}
global_blocks[23].ttt.{w0,w1,w2}
```

More precisely, the registered checkpoint parameters above are not mutated. Repeated/cached tensor copies are updated and passed into the TTT forward call as `w0_cache`, `w1_cache`, and `w2_cache`.

Non-model state that changes during inference:

- aggregator token states in `agg_state_refs`
- DPT intermediate feature states in `dpt_state_refs`
- TTT local tensors `w0_grad`, `w1_grad`, `w2_grad`, `shared_w0`, `shared_w1`, `shared_w2`
- post-processing pose/submap state in the pose graph optimizer
- output artifacts on disk

These are runtime states, not persistent neural-network parameters.


## 10. Parameter Names in `data/checkpoints/scal3r.pt`

The checkpoint state dict contains 1451 entries. The names below are grouped by the `Scal3R` module hierarchy and shown with a multi-level indented collapsible layout.

Architecture outline:

```text
Scal3R checkpoint
├─ agg_regator (1258)
│  ├─ special sequence tokens (2)
│  ├─ patch_embed / DINOv2 image encoder (344)
│  │  ├─ root / stem (8)
│  │  └─ blocks.0-23 (336)
│  ├─ frame_blocks.0-23 (432)
│  └─ global_blocks.0-23 (480)
│     └─ TTT modules in global_blocks.14/17/20/23
├─ cam_decoder (69)
│  ├─ root / iterative pose layers (13)
│  └─ trunk.0-3 (56)
├─ xyz_decoder (62)
└─ dpt_decoder (62)
```

Top-level counts:

- `agg_regator.*`: 1258
- `cam_decoder.*`: 69
- `xyz_decoder.*`: 62
- `dpt_decoder.*`: 62

<details open>
<summary>├─ agg_regator (1258)</summary>

<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ special sequence tokens (2)</summary>

```text
agg_regator.camera_token
agg_regator.register_token
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed / DINOv2 image encoder (344)</summary>

<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed root / stem (8)</summary>

```text
agg_regator.patch_embed.cls_token
agg_regator.patch_embed.pos_embed
agg_regator.patch_embed.register_tokens
agg_regator.patch_embed.mask_token
agg_regator.patch_embed.patch_embed.proj.weight
agg_regator.patch_embed.patch_embed.proj.bias
agg_regator.patch_embed.norm.weight
agg_regator.patch_embed.norm.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.0 (14)</summary>

```text
agg_regator.patch_embed.blocks.0.norm1.weight
agg_regator.patch_embed.blocks.0.norm1.bias
agg_regator.patch_embed.blocks.0.attn.qkv.weight
agg_regator.patch_embed.blocks.0.attn.qkv.bias
agg_regator.patch_embed.blocks.0.attn.proj.weight
agg_regator.patch_embed.blocks.0.attn.proj.bias
agg_regator.patch_embed.blocks.0.ls1.gamma
agg_regator.patch_embed.blocks.0.norm2.weight
agg_regator.patch_embed.blocks.0.norm2.bias
agg_regator.patch_embed.blocks.0.mlp.fc1.weight
agg_regator.patch_embed.blocks.0.mlp.fc1.bias
agg_regator.patch_embed.blocks.0.mlp.fc2.weight
agg_regator.patch_embed.blocks.0.mlp.fc2.bias
agg_regator.patch_embed.blocks.0.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.1 (14)</summary>

```text
agg_regator.patch_embed.blocks.1.norm1.weight
agg_regator.patch_embed.blocks.1.norm1.bias
agg_regator.patch_embed.blocks.1.attn.qkv.weight
agg_regator.patch_embed.blocks.1.attn.qkv.bias
agg_regator.patch_embed.blocks.1.attn.proj.weight
agg_regator.patch_embed.blocks.1.attn.proj.bias
agg_regator.patch_embed.blocks.1.ls1.gamma
agg_regator.patch_embed.blocks.1.norm2.weight
agg_regator.patch_embed.blocks.1.norm2.bias
agg_regator.patch_embed.blocks.1.mlp.fc1.weight
agg_regator.patch_embed.blocks.1.mlp.fc1.bias
agg_regator.patch_embed.blocks.1.mlp.fc2.weight
agg_regator.patch_embed.blocks.1.mlp.fc2.bias
agg_regator.patch_embed.blocks.1.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.2 (14)</summary>

```text
agg_regator.patch_embed.blocks.2.norm1.weight
agg_regator.patch_embed.blocks.2.norm1.bias
agg_regator.patch_embed.blocks.2.attn.qkv.weight
agg_regator.patch_embed.blocks.2.attn.qkv.bias
agg_regator.patch_embed.blocks.2.attn.proj.weight
agg_regator.patch_embed.blocks.2.attn.proj.bias
agg_regator.patch_embed.blocks.2.ls1.gamma
agg_regator.patch_embed.blocks.2.norm2.weight
agg_regator.patch_embed.blocks.2.norm2.bias
agg_regator.patch_embed.blocks.2.mlp.fc1.weight
agg_regator.patch_embed.blocks.2.mlp.fc1.bias
agg_regator.patch_embed.blocks.2.mlp.fc2.weight
agg_regator.patch_embed.blocks.2.mlp.fc2.bias
agg_regator.patch_embed.blocks.2.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.3 (14)</summary>

```text
agg_regator.patch_embed.blocks.3.norm1.weight
agg_regator.patch_embed.blocks.3.norm1.bias
agg_regator.patch_embed.blocks.3.attn.qkv.weight
agg_regator.patch_embed.blocks.3.attn.qkv.bias
agg_regator.patch_embed.blocks.3.attn.proj.weight
agg_regator.patch_embed.blocks.3.attn.proj.bias
agg_regator.patch_embed.blocks.3.ls1.gamma
agg_regator.patch_embed.blocks.3.norm2.weight
agg_regator.patch_embed.blocks.3.norm2.bias
agg_regator.patch_embed.blocks.3.mlp.fc1.weight
agg_regator.patch_embed.blocks.3.mlp.fc1.bias
agg_regator.patch_embed.blocks.3.mlp.fc2.weight
agg_regator.patch_embed.blocks.3.mlp.fc2.bias
agg_regator.patch_embed.blocks.3.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.4 (14)</summary>

```text
agg_regator.patch_embed.blocks.4.norm1.weight
agg_regator.patch_embed.blocks.4.norm1.bias
agg_regator.patch_embed.blocks.4.attn.qkv.weight
agg_regator.patch_embed.blocks.4.attn.qkv.bias
agg_regator.patch_embed.blocks.4.attn.proj.weight
agg_regator.patch_embed.blocks.4.attn.proj.bias
agg_regator.patch_embed.blocks.4.ls1.gamma
agg_regator.patch_embed.blocks.4.norm2.weight
agg_regator.patch_embed.blocks.4.norm2.bias
agg_regator.patch_embed.blocks.4.mlp.fc1.weight
agg_regator.patch_embed.blocks.4.mlp.fc1.bias
agg_regator.patch_embed.blocks.4.mlp.fc2.weight
agg_regator.patch_embed.blocks.4.mlp.fc2.bias
agg_regator.patch_embed.blocks.4.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.5 (14)</summary>

```text
agg_regator.patch_embed.blocks.5.norm1.weight
agg_regator.patch_embed.blocks.5.norm1.bias
agg_regator.patch_embed.blocks.5.attn.qkv.weight
agg_regator.patch_embed.blocks.5.attn.qkv.bias
agg_regator.patch_embed.blocks.5.attn.proj.weight
agg_regator.patch_embed.blocks.5.attn.proj.bias
agg_regator.patch_embed.blocks.5.ls1.gamma
agg_regator.patch_embed.blocks.5.norm2.weight
agg_regator.patch_embed.blocks.5.norm2.bias
agg_regator.patch_embed.blocks.5.mlp.fc1.weight
agg_regator.patch_embed.blocks.5.mlp.fc1.bias
agg_regator.patch_embed.blocks.5.mlp.fc2.weight
agg_regator.patch_embed.blocks.5.mlp.fc2.bias
agg_regator.patch_embed.blocks.5.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.6 (14)</summary>

```text
agg_regator.patch_embed.blocks.6.norm1.weight
agg_regator.patch_embed.blocks.6.norm1.bias
agg_regator.patch_embed.blocks.6.attn.qkv.weight
agg_regator.patch_embed.blocks.6.attn.qkv.bias
agg_regator.patch_embed.blocks.6.attn.proj.weight
agg_regator.patch_embed.blocks.6.attn.proj.bias
agg_regator.patch_embed.blocks.6.ls1.gamma
agg_regator.patch_embed.blocks.6.norm2.weight
agg_regator.patch_embed.blocks.6.norm2.bias
agg_regator.patch_embed.blocks.6.mlp.fc1.weight
agg_regator.patch_embed.blocks.6.mlp.fc1.bias
agg_regator.patch_embed.blocks.6.mlp.fc2.weight
agg_regator.patch_embed.blocks.6.mlp.fc2.bias
agg_regator.patch_embed.blocks.6.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.7 (14)</summary>

```text
agg_regator.patch_embed.blocks.7.norm1.weight
agg_regator.patch_embed.blocks.7.norm1.bias
agg_regator.patch_embed.blocks.7.attn.qkv.weight
agg_regator.patch_embed.blocks.7.attn.qkv.bias
agg_regator.patch_embed.blocks.7.attn.proj.weight
agg_regator.patch_embed.blocks.7.attn.proj.bias
agg_regator.patch_embed.blocks.7.ls1.gamma
agg_regator.patch_embed.blocks.7.norm2.weight
agg_regator.patch_embed.blocks.7.norm2.bias
agg_regator.patch_embed.blocks.7.mlp.fc1.weight
agg_regator.patch_embed.blocks.7.mlp.fc1.bias
agg_regator.patch_embed.blocks.7.mlp.fc2.weight
agg_regator.patch_embed.blocks.7.mlp.fc2.bias
agg_regator.patch_embed.blocks.7.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.8 (14)</summary>

```text
agg_regator.patch_embed.blocks.8.norm1.weight
agg_regator.patch_embed.blocks.8.norm1.bias
agg_regator.patch_embed.blocks.8.attn.qkv.weight
agg_regator.patch_embed.blocks.8.attn.qkv.bias
agg_regator.patch_embed.blocks.8.attn.proj.weight
agg_regator.patch_embed.blocks.8.attn.proj.bias
agg_regator.patch_embed.blocks.8.ls1.gamma
agg_regator.patch_embed.blocks.8.norm2.weight
agg_regator.patch_embed.blocks.8.norm2.bias
agg_regator.patch_embed.blocks.8.mlp.fc1.weight
agg_regator.patch_embed.blocks.8.mlp.fc1.bias
agg_regator.patch_embed.blocks.8.mlp.fc2.weight
agg_regator.patch_embed.blocks.8.mlp.fc2.bias
agg_regator.patch_embed.blocks.8.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.9 (14)</summary>

```text
agg_regator.patch_embed.blocks.9.norm1.weight
agg_regator.patch_embed.blocks.9.norm1.bias
agg_regator.patch_embed.blocks.9.attn.qkv.weight
agg_regator.patch_embed.blocks.9.attn.qkv.bias
agg_regator.patch_embed.blocks.9.attn.proj.weight
agg_regator.patch_embed.blocks.9.attn.proj.bias
agg_regator.patch_embed.blocks.9.ls1.gamma
agg_regator.patch_embed.blocks.9.norm2.weight
agg_regator.patch_embed.blocks.9.norm2.bias
agg_regator.patch_embed.blocks.9.mlp.fc1.weight
agg_regator.patch_embed.blocks.9.mlp.fc1.bias
agg_regator.patch_embed.blocks.9.mlp.fc2.weight
agg_regator.patch_embed.blocks.9.mlp.fc2.bias
agg_regator.patch_embed.blocks.9.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.10 (14)</summary>

```text
agg_regator.patch_embed.blocks.10.norm1.weight
agg_regator.patch_embed.blocks.10.norm1.bias
agg_regator.patch_embed.blocks.10.attn.qkv.weight
agg_regator.patch_embed.blocks.10.attn.qkv.bias
agg_regator.patch_embed.blocks.10.attn.proj.weight
agg_regator.patch_embed.blocks.10.attn.proj.bias
agg_regator.patch_embed.blocks.10.ls1.gamma
agg_regator.patch_embed.blocks.10.norm2.weight
agg_regator.patch_embed.blocks.10.norm2.bias
agg_regator.patch_embed.blocks.10.mlp.fc1.weight
agg_regator.patch_embed.blocks.10.mlp.fc1.bias
agg_regator.patch_embed.blocks.10.mlp.fc2.weight
agg_regator.patch_embed.blocks.10.mlp.fc2.bias
agg_regator.patch_embed.blocks.10.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.11 (14)</summary>

```text
agg_regator.patch_embed.blocks.11.norm1.weight
agg_regator.patch_embed.blocks.11.norm1.bias
agg_regator.patch_embed.blocks.11.attn.qkv.weight
agg_regator.patch_embed.blocks.11.attn.qkv.bias
agg_regator.patch_embed.blocks.11.attn.proj.weight
agg_regator.patch_embed.blocks.11.attn.proj.bias
agg_regator.patch_embed.blocks.11.ls1.gamma
agg_regator.patch_embed.blocks.11.norm2.weight
agg_regator.patch_embed.blocks.11.norm2.bias
agg_regator.patch_embed.blocks.11.mlp.fc1.weight
agg_regator.patch_embed.blocks.11.mlp.fc1.bias
agg_regator.patch_embed.blocks.11.mlp.fc2.weight
agg_regator.patch_embed.blocks.11.mlp.fc2.bias
agg_regator.patch_embed.blocks.11.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.12 (14)</summary>

```text
agg_regator.patch_embed.blocks.12.norm1.weight
agg_regator.patch_embed.blocks.12.norm1.bias
agg_regator.patch_embed.blocks.12.attn.qkv.weight
agg_regator.patch_embed.blocks.12.attn.qkv.bias
agg_regator.patch_embed.blocks.12.attn.proj.weight
agg_regator.patch_embed.blocks.12.attn.proj.bias
agg_regator.patch_embed.blocks.12.ls1.gamma
agg_regator.patch_embed.blocks.12.norm2.weight
agg_regator.patch_embed.blocks.12.norm2.bias
agg_regator.patch_embed.blocks.12.mlp.fc1.weight
agg_regator.patch_embed.blocks.12.mlp.fc1.bias
agg_regator.patch_embed.blocks.12.mlp.fc2.weight
agg_regator.patch_embed.blocks.12.mlp.fc2.bias
agg_regator.patch_embed.blocks.12.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.13 (14)</summary>

```text
agg_regator.patch_embed.blocks.13.norm1.weight
agg_regator.patch_embed.blocks.13.norm1.bias
agg_regator.patch_embed.blocks.13.attn.qkv.weight
agg_regator.patch_embed.blocks.13.attn.qkv.bias
agg_regator.patch_embed.blocks.13.attn.proj.weight
agg_regator.patch_embed.blocks.13.attn.proj.bias
agg_regator.patch_embed.blocks.13.ls1.gamma
agg_regator.patch_embed.blocks.13.norm2.weight
agg_regator.patch_embed.blocks.13.norm2.bias
agg_regator.patch_embed.blocks.13.mlp.fc1.weight
agg_regator.patch_embed.blocks.13.mlp.fc1.bias
agg_regator.patch_embed.blocks.13.mlp.fc2.weight
agg_regator.patch_embed.blocks.13.mlp.fc2.bias
agg_regator.patch_embed.blocks.13.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.14 (14)</summary>

```text
agg_regator.patch_embed.blocks.14.norm1.weight
agg_regator.patch_embed.blocks.14.norm1.bias
agg_regator.patch_embed.blocks.14.attn.qkv.weight
agg_regator.patch_embed.blocks.14.attn.qkv.bias
agg_regator.patch_embed.blocks.14.attn.proj.weight
agg_regator.patch_embed.blocks.14.attn.proj.bias
agg_regator.patch_embed.blocks.14.ls1.gamma
agg_regator.patch_embed.blocks.14.norm2.weight
agg_regator.patch_embed.blocks.14.norm2.bias
agg_regator.patch_embed.blocks.14.mlp.fc1.weight
agg_regator.patch_embed.blocks.14.mlp.fc1.bias
agg_regator.patch_embed.blocks.14.mlp.fc2.weight
agg_regator.patch_embed.blocks.14.mlp.fc2.bias
agg_regator.patch_embed.blocks.14.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.15 (14)</summary>

```text
agg_regator.patch_embed.blocks.15.norm1.weight
agg_regator.patch_embed.blocks.15.norm1.bias
agg_regator.patch_embed.blocks.15.attn.qkv.weight
agg_regator.patch_embed.blocks.15.attn.qkv.bias
agg_regator.patch_embed.blocks.15.attn.proj.weight
agg_regator.patch_embed.blocks.15.attn.proj.bias
agg_regator.patch_embed.blocks.15.ls1.gamma
agg_regator.patch_embed.blocks.15.norm2.weight
agg_regator.patch_embed.blocks.15.norm2.bias
agg_regator.patch_embed.blocks.15.mlp.fc1.weight
agg_regator.patch_embed.blocks.15.mlp.fc1.bias
agg_regator.patch_embed.blocks.15.mlp.fc2.weight
agg_regator.patch_embed.blocks.15.mlp.fc2.bias
agg_regator.patch_embed.blocks.15.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.16 (14)</summary>

```text
agg_regator.patch_embed.blocks.16.norm1.weight
agg_regator.patch_embed.blocks.16.norm1.bias
agg_regator.patch_embed.blocks.16.attn.qkv.weight
agg_regator.patch_embed.blocks.16.attn.qkv.bias
agg_regator.patch_embed.blocks.16.attn.proj.weight
agg_regator.patch_embed.blocks.16.attn.proj.bias
agg_regator.patch_embed.blocks.16.ls1.gamma
agg_regator.patch_embed.blocks.16.norm2.weight
agg_regator.patch_embed.blocks.16.norm2.bias
agg_regator.patch_embed.blocks.16.mlp.fc1.weight
agg_regator.patch_embed.blocks.16.mlp.fc1.bias
agg_regator.patch_embed.blocks.16.mlp.fc2.weight
agg_regator.patch_embed.blocks.16.mlp.fc2.bias
agg_regator.patch_embed.blocks.16.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.17 (14)</summary>

```text
agg_regator.patch_embed.blocks.17.norm1.weight
agg_regator.patch_embed.blocks.17.norm1.bias
agg_regator.patch_embed.blocks.17.attn.qkv.weight
agg_regator.patch_embed.blocks.17.attn.qkv.bias
agg_regator.patch_embed.blocks.17.attn.proj.weight
agg_regator.patch_embed.blocks.17.attn.proj.bias
agg_regator.patch_embed.blocks.17.ls1.gamma
agg_regator.patch_embed.blocks.17.norm2.weight
agg_regator.patch_embed.blocks.17.norm2.bias
agg_regator.patch_embed.blocks.17.mlp.fc1.weight
agg_regator.patch_embed.blocks.17.mlp.fc1.bias
agg_regator.patch_embed.blocks.17.mlp.fc2.weight
agg_regator.patch_embed.blocks.17.mlp.fc2.bias
agg_regator.patch_embed.blocks.17.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.18 (14)</summary>

```text
agg_regator.patch_embed.blocks.18.norm1.weight
agg_regator.patch_embed.blocks.18.norm1.bias
agg_regator.patch_embed.blocks.18.attn.qkv.weight
agg_regator.patch_embed.blocks.18.attn.qkv.bias
agg_regator.patch_embed.blocks.18.attn.proj.weight
agg_regator.patch_embed.blocks.18.attn.proj.bias
agg_regator.patch_embed.blocks.18.ls1.gamma
agg_regator.patch_embed.blocks.18.norm2.weight
agg_regator.patch_embed.blocks.18.norm2.bias
agg_regator.patch_embed.blocks.18.mlp.fc1.weight
agg_regator.patch_embed.blocks.18.mlp.fc1.bias
agg_regator.patch_embed.blocks.18.mlp.fc2.weight
agg_regator.patch_embed.blocks.18.mlp.fc2.bias
agg_regator.patch_embed.blocks.18.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.19 (14)</summary>

```text
agg_regator.patch_embed.blocks.19.norm1.weight
agg_regator.patch_embed.blocks.19.norm1.bias
agg_regator.patch_embed.blocks.19.attn.qkv.weight
agg_regator.patch_embed.blocks.19.attn.qkv.bias
agg_regator.patch_embed.blocks.19.attn.proj.weight
agg_regator.patch_embed.blocks.19.attn.proj.bias
agg_regator.patch_embed.blocks.19.ls1.gamma
agg_regator.patch_embed.blocks.19.norm2.weight
agg_regator.patch_embed.blocks.19.norm2.bias
agg_regator.patch_embed.blocks.19.mlp.fc1.weight
agg_regator.patch_embed.blocks.19.mlp.fc1.bias
agg_regator.patch_embed.blocks.19.mlp.fc2.weight
agg_regator.patch_embed.blocks.19.mlp.fc2.bias
agg_regator.patch_embed.blocks.19.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.20 (14)</summary>

```text
agg_regator.patch_embed.blocks.20.norm1.weight
agg_regator.patch_embed.blocks.20.norm1.bias
agg_regator.patch_embed.blocks.20.attn.qkv.weight
agg_regator.patch_embed.blocks.20.attn.qkv.bias
agg_regator.patch_embed.blocks.20.attn.proj.weight
agg_regator.patch_embed.blocks.20.attn.proj.bias
agg_regator.patch_embed.blocks.20.ls1.gamma
agg_regator.patch_embed.blocks.20.norm2.weight
agg_regator.patch_embed.blocks.20.norm2.bias
agg_regator.patch_embed.blocks.20.mlp.fc1.weight
agg_regator.patch_embed.blocks.20.mlp.fc1.bias
agg_regator.patch_embed.blocks.20.mlp.fc2.weight
agg_regator.patch_embed.blocks.20.mlp.fc2.bias
agg_regator.patch_embed.blocks.20.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.21 (14)</summary>

```text
agg_regator.patch_embed.blocks.21.norm1.weight
agg_regator.patch_embed.blocks.21.norm1.bias
agg_regator.patch_embed.blocks.21.attn.qkv.weight
agg_regator.patch_embed.blocks.21.attn.qkv.bias
agg_regator.patch_embed.blocks.21.attn.proj.weight
agg_regator.patch_embed.blocks.21.attn.proj.bias
agg_regator.patch_embed.blocks.21.ls1.gamma
agg_regator.patch_embed.blocks.21.norm2.weight
agg_regator.patch_embed.blocks.21.norm2.bias
agg_regator.patch_embed.blocks.21.mlp.fc1.weight
agg_regator.patch_embed.blocks.21.mlp.fc1.bias
agg_regator.patch_embed.blocks.21.mlp.fc2.weight
agg_regator.patch_embed.blocks.21.mlp.fc2.bias
agg_regator.patch_embed.blocks.21.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ patch_embed.blocks.22 (14)</summary>

```text
agg_regator.patch_embed.blocks.22.norm1.weight
agg_regator.patch_embed.blocks.22.norm1.bias
agg_regator.patch_embed.blocks.22.attn.qkv.weight
agg_regator.patch_embed.blocks.22.attn.qkv.bias
agg_regator.patch_embed.blocks.22.attn.proj.weight
agg_regator.patch_embed.blocks.22.attn.proj.bias
agg_regator.patch_embed.blocks.22.ls1.gamma
agg_regator.patch_embed.blocks.22.norm2.weight
agg_regator.patch_embed.blocks.22.norm2.bias
agg_regator.patch_embed.blocks.22.mlp.fc1.weight
agg_regator.patch_embed.blocks.22.mlp.fc1.bias
agg_regator.patch_embed.blocks.22.mlp.fc2.weight
agg_regator.patch_embed.blocks.22.mlp.fc2.bias
agg_regator.patch_embed.blocks.22.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ patch_embed.blocks.23 (14)</summary>

```text
agg_regator.patch_embed.blocks.23.norm1.weight
agg_regator.patch_embed.blocks.23.norm1.bias
agg_regator.patch_embed.blocks.23.attn.qkv.weight
agg_regator.patch_embed.blocks.23.attn.qkv.bias
agg_regator.patch_embed.blocks.23.attn.proj.weight
agg_regator.patch_embed.blocks.23.attn.proj.bias
agg_regator.patch_embed.blocks.23.ls1.gamma
agg_regator.patch_embed.blocks.23.norm2.weight
agg_regator.patch_embed.blocks.23.norm2.bias
agg_regator.patch_embed.blocks.23.mlp.fc1.weight
agg_regator.patch_embed.blocks.23.mlp.fc1.bias
agg_regator.patch_embed.blocks.23.mlp.fc2.weight
agg_regator.patch_embed.blocks.23.mlp.fc2.bias
agg_regator.patch_embed.blocks.23.ls2.gamma
```
</details>
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks (432)</summary>

<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.0 (18)</summary>

```text
agg_regator.frame_blocks.0.norm1.weight
agg_regator.frame_blocks.0.norm1.bias
agg_regator.frame_blocks.0.attn.qkv.weight
agg_regator.frame_blocks.0.attn.qkv.bias
agg_regator.frame_blocks.0.attn.q_norm.weight
agg_regator.frame_blocks.0.attn.q_norm.bias
agg_regator.frame_blocks.0.attn.k_norm.weight
agg_regator.frame_blocks.0.attn.k_norm.bias
agg_regator.frame_blocks.0.attn.proj.weight
agg_regator.frame_blocks.0.attn.proj.bias
agg_regator.frame_blocks.0.ls1.gamma
agg_regator.frame_blocks.0.norm2.weight
agg_regator.frame_blocks.0.norm2.bias
agg_regator.frame_blocks.0.mlp.fc1.weight
agg_regator.frame_blocks.0.mlp.fc1.bias
agg_regator.frame_blocks.0.mlp.fc2.weight
agg_regator.frame_blocks.0.mlp.fc2.bias
agg_regator.frame_blocks.0.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.1 (18)</summary>

```text
agg_regator.frame_blocks.1.norm1.weight
agg_regator.frame_blocks.1.norm1.bias
agg_regator.frame_blocks.1.attn.qkv.weight
agg_regator.frame_blocks.1.attn.qkv.bias
agg_regator.frame_blocks.1.attn.q_norm.weight
agg_regator.frame_blocks.1.attn.q_norm.bias
agg_regator.frame_blocks.1.attn.k_norm.weight
agg_regator.frame_blocks.1.attn.k_norm.bias
agg_regator.frame_blocks.1.attn.proj.weight
agg_regator.frame_blocks.1.attn.proj.bias
agg_regator.frame_blocks.1.ls1.gamma
agg_regator.frame_blocks.1.norm2.weight
agg_regator.frame_blocks.1.norm2.bias
agg_regator.frame_blocks.1.mlp.fc1.weight
agg_regator.frame_blocks.1.mlp.fc1.bias
agg_regator.frame_blocks.1.mlp.fc2.weight
agg_regator.frame_blocks.1.mlp.fc2.bias
agg_regator.frame_blocks.1.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.2 (18)</summary>

```text
agg_regator.frame_blocks.2.norm1.weight
agg_regator.frame_blocks.2.norm1.bias
agg_regator.frame_blocks.2.attn.qkv.weight
agg_regator.frame_blocks.2.attn.qkv.bias
agg_regator.frame_blocks.2.attn.q_norm.weight
agg_regator.frame_blocks.2.attn.q_norm.bias
agg_regator.frame_blocks.2.attn.k_norm.weight
agg_regator.frame_blocks.2.attn.k_norm.bias
agg_regator.frame_blocks.2.attn.proj.weight
agg_regator.frame_blocks.2.attn.proj.bias
agg_regator.frame_blocks.2.ls1.gamma
agg_regator.frame_blocks.2.norm2.weight
agg_regator.frame_blocks.2.norm2.bias
agg_regator.frame_blocks.2.mlp.fc1.weight
agg_regator.frame_blocks.2.mlp.fc1.bias
agg_regator.frame_blocks.2.mlp.fc2.weight
agg_regator.frame_blocks.2.mlp.fc2.bias
agg_regator.frame_blocks.2.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.3 (18)</summary>

```text
agg_regator.frame_blocks.3.norm1.weight
agg_regator.frame_blocks.3.norm1.bias
agg_regator.frame_blocks.3.attn.qkv.weight
agg_regator.frame_blocks.3.attn.qkv.bias
agg_regator.frame_blocks.3.attn.q_norm.weight
agg_regator.frame_blocks.3.attn.q_norm.bias
agg_regator.frame_blocks.3.attn.k_norm.weight
agg_regator.frame_blocks.3.attn.k_norm.bias
agg_regator.frame_blocks.3.attn.proj.weight
agg_regator.frame_blocks.3.attn.proj.bias
agg_regator.frame_blocks.3.ls1.gamma
agg_regator.frame_blocks.3.norm2.weight
agg_regator.frame_blocks.3.norm2.bias
agg_regator.frame_blocks.3.mlp.fc1.weight
agg_regator.frame_blocks.3.mlp.fc1.bias
agg_regator.frame_blocks.3.mlp.fc2.weight
agg_regator.frame_blocks.3.mlp.fc2.bias
agg_regator.frame_blocks.3.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.4 (18)</summary>

```text
agg_regator.frame_blocks.4.norm1.weight
agg_regator.frame_blocks.4.norm1.bias
agg_regator.frame_blocks.4.attn.qkv.weight
agg_regator.frame_blocks.4.attn.qkv.bias
agg_regator.frame_blocks.4.attn.q_norm.weight
agg_regator.frame_blocks.4.attn.q_norm.bias
agg_regator.frame_blocks.4.attn.k_norm.weight
agg_regator.frame_blocks.4.attn.k_norm.bias
agg_regator.frame_blocks.4.attn.proj.weight
agg_regator.frame_blocks.4.attn.proj.bias
agg_regator.frame_blocks.4.ls1.gamma
agg_regator.frame_blocks.4.norm2.weight
agg_regator.frame_blocks.4.norm2.bias
agg_regator.frame_blocks.4.mlp.fc1.weight
agg_regator.frame_blocks.4.mlp.fc1.bias
agg_regator.frame_blocks.4.mlp.fc2.weight
agg_regator.frame_blocks.4.mlp.fc2.bias
agg_regator.frame_blocks.4.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.5 (18)</summary>

```text
agg_regator.frame_blocks.5.norm1.weight
agg_regator.frame_blocks.5.norm1.bias
agg_regator.frame_blocks.5.attn.qkv.weight
agg_regator.frame_blocks.5.attn.qkv.bias
agg_regator.frame_blocks.5.attn.q_norm.weight
agg_regator.frame_blocks.5.attn.q_norm.bias
agg_regator.frame_blocks.5.attn.k_norm.weight
agg_regator.frame_blocks.5.attn.k_norm.bias
agg_regator.frame_blocks.5.attn.proj.weight
agg_regator.frame_blocks.5.attn.proj.bias
agg_regator.frame_blocks.5.ls1.gamma
agg_regator.frame_blocks.5.norm2.weight
agg_regator.frame_blocks.5.norm2.bias
agg_regator.frame_blocks.5.mlp.fc1.weight
agg_regator.frame_blocks.5.mlp.fc1.bias
agg_regator.frame_blocks.5.mlp.fc2.weight
agg_regator.frame_blocks.5.mlp.fc2.bias
agg_regator.frame_blocks.5.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.6 (18)</summary>

```text
agg_regator.frame_blocks.6.norm1.weight
agg_regator.frame_blocks.6.norm1.bias
agg_regator.frame_blocks.6.attn.qkv.weight
agg_regator.frame_blocks.6.attn.qkv.bias
agg_regator.frame_blocks.6.attn.q_norm.weight
agg_regator.frame_blocks.6.attn.q_norm.bias
agg_regator.frame_blocks.6.attn.k_norm.weight
agg_regator.frame_blocks.6.attn.k_norm.bias
agg_regator.frame_blocks.6.attn.proj.weight
agg_regator.frame_blocks.6.attn.proj.bias
agg_regator.frame_blocks.6.ls1.gamma
agg_regator.frame_blocks.6.norm2.weight
agg_regator.frame_blocks.6.norm2.bias
agg_regator.frame_blocks.6.mlp.fc1.weight
agg_regator.frame_blocks.6.mlp.fc1.bias
agg_regator.frame_blocks.6.mlp.fc2.weight
agg_regator.frame_blocks.6.mlp.fc2.bias
agg_regator.frame_blocks.6.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.7 (18)</summary>

```text
agg_regator.frame_blocks.7.norm1.weight
agg_regator.frame_blocks.7.norm1.bias
agg_regator.frame_blocks.7.attn.qkv.weight
agg_regator.frame_blocks.7.attn.qkv.bias
agg_regator.frame_blocks.7.attn.q_norm.weight
agg_regator.frame_blocks.7.attn.q_norm.bias
agg_regator.frame_blocks.7.attn.k_norm.weight
agg_regator.frame_blocks.7.attn.k_norm.bias
agg_regator.frame_blocks.7.attn.proj.weight
agg_regator.frame_blocks.7.attn.proj.bias
agg_regator.frame_blocks.7.ls1.gamma
agg_regator.frame_blocks.7.norm2.weight
agg_regator.frame_blocks.7.norm2.bias
agg_regator.frame_blocks.7.mlp.fc1.weight
agg_regator.frame_blocks.7.mlp.fc1.bias
agg_regator.frame_blocks.7.mlp.fc2.weight
agg_regator.frame_blocks.7.mlp.fc2.bias
agg_regator.frame_blocks.7.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.8 (18)</summary>

```text
agg_regator.frame_blocks.8.norm1.weight
agg_regator.frame_blocks.8.norm1.bias
agg_regator.frame_blocks.8.attn.qkv.weight
agg_regator.frame_blocks.8.attn.qkv.bias
agg_regator.frame_blocks.8.attn.q_norm.weight
agg_regator.frame_blocks.8.attn.q_norm.bias
agg_regator.frame_blocks.8.attn.k_norm.weight
agg_regator.frame_blocks.8.attn.k_norm.bias
agg_regator.frame_blocks.8.attn.proj.weight
agg_regator.frame_blocks.8.attn.proj.bias
agg_regator.frame_blocks.8.ls1.gamma
agg_regator.frame_blocks.8.norm2.weight
agg_regator.frame_blocks.8.norm2.bias
agg_regator.frame_blocks.8.mlp.fc1.weight
agg_regator.frame_blocks.8.mlp.fc1.bias
agg_regator.frame_blocks.8.mlp.fc2.weight
agg_regator.frame_blocks.8.mlp.fc2.bias
agg_regator.frame_blocks.8.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.9 (18)</summary>

```text
agg_regator.frame_blocks.9.norm1.weight
agg_regator.frame_blocks.9.norm1.bias
agg_regator.frame_blocks.9.attn.qkv.weight
agg_regator.frame_blocks.9.attn.qkv.bias
agg_regator.frame_blocks.9.attn.q_norm.weight
agg_regator.frame_blocks.9.attn.q_norm.bias
agg_regator.frame_blocks.9.attn.k_norm.weight
agg_regator.frame_blocks.9.attn.k_norm.bias
agg_regator.frame_blocks.9.attn.proj.weight
agg_regator.frame_blocks.9.attn.proj.bias
agg_regator.frame_blocks.9.ls1.gamma
agg_regator.frame_blocks.9.norm2.weight
agg_regator.frame_blocks.9.norm2.bias
agg_regator.frame_blocks.9.mlp.fc1.weight
agg_regator.frame_blocks.9.mlp.fc1.bias
agg_regator.frame_blocks.9.mlp.fc2.weight
agg_regator.frame_blocks.9.mlp.fc2.bias
agg_regator.frame_blocks.9.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.10 (18)</summary>

```text
agg_regator.frame_blocks.10.norm1.weight
agg_regator.frame_blocks.10.norm1.bias
agg_regator.frame_blocks.10.attn.qkv.weight
agg_regator.frame_blocks.10.attn.qkv.bias
agg_regator.frame_blocks.10.attn.q_norm.weight
agg_regator.frame_blocks.10.attn.q_norm.bias
agg_regator.frame_blocks.10.attn.k_norm.weight
agg_regator.frame_blocks.10.attn.k_norm.bias
agg_regator.frame_blocks.10.attn.proj.weight
agg_regator.frame_blocks.10.attn.proj.bias
agg_regator.frame_blocks.10.ls1.gamma
agg_regator.frame_blocks.10.norm2.weight
agg_regator.frame_blocks.10.norm2.bias
agg_regator.frame_blocks.10.mlp.fc1.weight
agg_regator.frame_blocks.10.mlp.fc1.bias
agg_regator.frame_blocks.10.mlp.fc2.weight
agg_regator.frame_blocks.10.mlp.fc2.bias
agg_regator.frame_blocks.10.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.11 (18)</summary>

```text
agg_regator.frame_blocks.11.norm1.weight
agg_regator.frame_blocks.11.norm1.bias
agg_regator.frame_blocks.11.attn.qkv.weight
agg_regator.frame_blocks.11.attn.qkv.bias
agg_regator.frame_blocks.11.attn.q_norm.weight
agg_regator.frame_blocks.11.attn.q_norm.bias
agg_regator.frame_blocks.11.attn.k_norm.weight
agg_regator.frame_blocks.11.attn.k_norm.bias
agg_regator.frame_blocks.11.attn.proj.weight
agg_regator.frame_blocks.11.attn.proj.bias
agg_regator.frame_blocks.11.ls1.gamma
agg_regator.frame_blocks.11.norm2.weight
agg_regator.frame_blocks.11.norm2.bias
agg_regator.frame_blocks.11.mlp.fc1.weight
agg_regator.frame_blocks.11.mlp.fc1.bias
agg_regator.frame_blocks.11.mlp.fc2.weight
agg_regator.frame_blocks.11.mlp.fc2.bias
agg_regator.frame_blocks.11.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.12 (18)</summary>

```text
agg_regator.frame_blocks.12.norm1.weight
agg_regator.frame_blocks.12.norm1.bias
agg_regator.frame_blocks.12.attn.qkv.weight
agg_regator.frame_blocks.12.attn.qkv.bias
agg_regator.frame_blocks.12.attn.q_norm.weight
agg_regator.frame_blocks.12.attn.q_norm.bias
agg_regator.frame_blocks.12.attn.k_norm.weight
agg_regator.frame_blocks.12.attn.k_norm.bias
agg_regator.frame_blocks.12.attn.proj.weight
agg_regator.frame_blocks.12.attn.proj.bias
agg_regator.frame_blocks.12.ls1.gamma
agg_regator.frame_blocks.12.norm2.weight
agg_regator.frame_blocks.12.norm2.bias
agg_regator.frame_blocks.12.mlp.fc1.weight
agg_regator.frame_blocks.12.mlp.fc1.bias
agg_regator.frame_blocks.12.mlp.fc2.weight
agg_regator.frame_blocks.12.mlp.fc2.bias
agg_regator.frame_blocks.12.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.13 (18)</summary>

```text
agg_regator.frame_blocks.13.norm1.weight
agg_regator.frame_blocks.13.norm1.bias
agg_regator.frame_blocks.13.attn.qkv.weight
agg_regator.frame_blocks.13.attn.qkv.bias
agg_regator.frame_blocks.13.attn.q_norm.weight
agg_regator.frame_blocks.13.attn.q_norm.bias
agg_regator.frame_blocks.13.attn.k_norm.weight
agg_regator.frame_blocks.13.attn.k_norm.bias
agg_regator.frame_blocks.13.attn.proj.weight
agg_regator.frame_blocks.13.attn.proj.bias
agg_regator.frame_blocks.13.ls1.gamma
agg_regator.frame_blocks.13.norm2.weight
agg_regator.frame_blocks.13.norm2.bias
agg_regator.frame_blocks.13.mlp.fc1.weight
agg_regator.frame_blocks.13.mlp.fc1.bias
agg_regator.frame_blocks.13.mlp.fc2.weight
agg_regator.frame_blocks.13.mlp.fc2.bias
agg_regator.frame_blocks.13.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.14 (18)</summary>

```text
agg_regator.frame_blocks.14.norm1.weight
agg_regator.frame_blocks.14.norm1.bias
agg_regator.frame_blocks.14.attn.qkv.weight
agg_regator.frame_blocks.14.attn.qkv.bias
agg_regator.frame_blocks.14.attn.q_norm.weight
agg_regator.frame_blocks.14.attn.q_norm.bias
agg_regator.frame_blocks.14.attn.k_norm.weight
agg_regator.frame_blocks.14.attn.k_norm.bias
agg_regator.frame_blocks.14.attn.proj.weight
agg_regator.frame_blocks.14.attn.proj.bias
agg_regator.frame_blocks.14.ls1.gamma
agg_regator.frame_blocks.14.norm2.weight
agg_regator.frame_blocks.14.norm2.bias
agg_regator.frame_blocks.14.mlp.fc1.weight
agg_regator.frame_blocks.14.mlp.fc1.bias
agg_regator.frame_blocks.14.mlp.fc2.weight
agg_regator.frame_blocks.14.mlp.fc2.bias
agg_regator.frame_blocks.14.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.15 (18)</summary>

```text
agg_regator.frame_blocks.15.norm1.weight
agg_regator.frame_blocks.15.norm1.bias
agg_regator.frame_blocks.15.attn.qkv.weight
agg_regator.frame_blocks.15.attn.qkv.bias
agg_regator.frame_blocks.15.attn.q_norm.weight
agg_regator.frame_blocks.15.attn.q_norm.bias
agg_regator.frame_blocks.15.attn.k_norm.weight
agg_regator.frame_blocks.15.attn.k_norm.bias
agg_regator.frame_blocks.15.attn.proj.weight
agg_regator.frame_blocks.15.attn.proj.bias
agg_regator.frame_blocks.15.ls1.gamma
agg_regator.frame_blocks.15.norm2.weight
agg_regator.frame_blocks.15.norm2.bias
agg_regator.frame_blocks.15.mlp.fc1.weight
agg_regator.frame_blocks.15.mlp.fc1.bias
agg_regator.frame_blocks.15.mlp.fc2.weight
agg_regator.frame_blocks.15.mlp.fc2.bias
agg_regator.frame_blocks.15.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.16 (18)</summary>

```text
agg_regator.frame_blocks.16.norm1.weight
agg_regator.frame_blocks.16.norm1.bias
agg_regator.frame_blocks.16.attn.qkv.weight
agg_regator.frame_blocks.16.attn.qkv.bias
agg_regator.frame_blocks.16.attn.q_norm.weight
agg_regator.frame_blocks.16.attn.q_norm.bias
agg_regator.frame_blocks.16.attn.k_norm.weight
agg_regator.frame_blocks.16.attn.k_norm.bias
agg_regator.frame_blocks.16.attn.proj.weight
agg_regator.frame_blocks.16.attn.proj.bias
agg_regator.frame_blocks.16.ls1.gamma
agg_regator.frame_blocks.16.norm2.weight
agg_regator.frame_blocks.16.norm2.bias
agg_regator.frame_blocks.16.mlp.fc1.weight
agg_regator.frame_blocks.16.mlp.fc1.bias
agg_regator.frame_blocks.16.mlp.fc2.weight
agg_regator.frame_blocks.16.mlp.fc2.bias
agg_regator.frame_blocks.16.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.17 (18)</summary>

```text
agg_regator.frame_blocks.17.norm1.weight
agg_regator.frame_blocks.17.norm1.bias
agg_regator.frame_blocks.17.attn.qkv.weight
agg_regator.frame_blocks.17.attn.qkv.bias
agg_regator.frame_blocks.17.attn.q_norm.weight
agg_regator.frame_blocks.17.attn.q_norm.bias
agg_regator.frame_blocks.17.attn.k_norm.weight
agg_regator.frame_blocks.17.attn.k_norm.bias
agg_regator.frame_blocks.17.attn.proj.weight
agg_regator.frame_blocks.17.attn.proj.bias
agg_regator.frame_blocks.17.ls1.gamma
agg_regator.frame_blocks.17.norm2.weight
agg_regator.frame_blocks.17.norm2.bias
agg_regator.frame_blocks.17.mlp.fc1.weight
agg_regator.frame_blocks.17.mlp.fc1.bias
agg_regator.frame_blocks.17.mlp.fc2.weight
agg_regator.frame_blocks.17.mlp.fc2.bias
agg_regator.frame_blocks.17.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.18 (18)</summary>

```text
agg_regator.frame_blocks.18.norm1.weight
agg_regator.frame_blocks.18.norm1.bias
agg_regator.frame_blocks.18.attn.qkv.weight
agg_regator.frame_blocks.18.attn.qkv.bias
agg_regator.frame_blocks.18.attn.q_norm.weight
agg_regator.frame_blocks.18.attn.q_norm.bias
agg_regator.frame_blocks.18.attn.k_norm.weight
agg_regator.frame_blocks.18.attn.k_norm.bias
agg_regator.frame_blocks.18.attn.proj.weight
agg_regator.frame_blocks.18.attn.proj.bias
agg_regator.frame_blocks.18.ls1.gamma
agg_regator.frame_blocks.18.norm2.weight
agg_regator.frame_blocks.18.norm2.bias
agg_regator.frame_blocks.18.mlp.fc1.weight
agg_regator.frame_blocks.18.mlp.fc1.bias
agg_regator.frame_blocks.18.mlp.fc2.weight
agg_regator.frame_blocks.18.mlp.fc2.bias
agg_regator.frame_blocks.18.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.19 (18)</summary>

```text
agg_regator.frame_blocks.19.norm1.weight
agg_regator.frame_blocks.19.norm1.bias
agg_regator.frame_blocks.19.attn.qkv.weight
agg_regator.frame_blocks.19.attn.qkv.bias
agg_regator.frame_blocks.19.attn.q_norm.weight
agg_regator.frame_blocks.19.attn.q_norm.bias
agg_regator.frame_blocks.19.attn.k_norm.weight
agg_regator.frame_blocks.19.attn.k_norm.bias
agg_regator.frame_blocks.19.attn.proj.weight
agg_regator.frame_blocks.19.attn.proj.bias
agg_regator.frame_blocks.19.ls1.gamma
agg_regator.frame_blocks.19.norm2.weight
agg_regator.frame_blocks.19.norm2.bias
agg_regator.frame_blocks.19.mlp.fc1.weight
agg_regator.frame_blocks.19.mlp.fc1.bias
agg_regator.frame_blocks.19.mlp.fc2.weight
agg_regator.frame_blocks.19.mlp.fc2.bias
agg_regator.frame_blocks.19.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.20 (18)</summary>

```text
agg_regator.frame_blocks.20.norm1.weight
agg_regator.frame_blocks.20.norm1.bias
agg_regator.frame_blocks.20.attn.qkv.weight
agg_regator.frame_blocks.20.attn.qkv.bias
agg_regator.frame_blocks.20.attn.q_norm.weight
agg_regator.frame_blocks.20.attn.q_norm.bias
agg_regator.frame_blocks.20.attn.k_norm.weight
agg_regator.frame_blocks.20.attn.k_norm.bias
agg_regator.frame_blocks.20.attn.proj.weight
agg_regator.frame_blocks.20.attn.proj.bias
agg_regator.frame_blocks.20.ls1.gamma
agg_regator.frame_blocks.20.norm2.weight
agg_regator.frame_blocks.20.norm2.bias
agg_regator.frame_blocks.20.mlp.fc1.weight
agg_regator.frame_blocks.20.mlp.fc1.bias
agg_regator.frame_blocks.20.mlp.fc2.weight
agg_regator.frame_blocks.20.mlp.fc2.bias
agg_regator.frame_blocks.20.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.21 (18)</summary>

```text
agg_regator.frame_blocks.21.norm1.weight
agg_regator.frame_blocks.21.norm1.bias
agg_regator.frame_blocks.21.attn.qkv.weight
agg_regator.frame_blocks.21.attn.qkv.bias
agg_regator.frame_blocks.21.attn.q_norm.weight
agg_regator.frame_blocks.21.attn.q_norm.bias
agg_regator.frame_blocks.21.attn.k_norm.weight
agg_regator.frame_blocks.21.attn.k_norm.bias
agg_regator.frame_blocks.21.attn.proj.weight
agg_regator.frame_blocks.21.attn.proj.bias
agg_regator.frame_blocks.21.ls1.gamma
agg_regator.frame_blocks.21.norm2.weight
agg_regator.frame_blocks.21.norm2.bias
agg_regator.frame_blocks.21.mlp.fc1.weight
agg_regator.frame_blocks.21.mlp.fc1.bias
agg_regator.frame_blocks.21.mlp.fc2.weight
agg_regator.frame_blocks.21.mlp.fc2.bias
agg_regator.frame_blocks.21.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ frame_blocks.22 (18)</summary>

```text
agg_regator.frame_blocks.22.norm1.weight
agg_regator.frame_blocks.22.norm1.bias
agg_regator.frame_blocks.22.attn.qkv.weight
agg_regator.frame_blocks.22.attn.qkv.bias
agg_regator.frame_blocks.22.attn.q_norm.weight
agg_regator.frame_blocks.22.attn.q_norm.bias
agg_regator.frame_blocks.22.attn.k_norm.weight
agg_regator.frame_blocks.22.attn.k_norm.bias
agg_regator.frame_blocks.22.attn.proj.weight
agg_regator.frame_blocks.22.attn.proj.bias
agg_regator.frame_blocks.22.ls1.gamma
agg_regator.frame_blocks.22.norm2.weight
agg_regator.frame_blocks.22.norm2.bias
agg_regator.frame_blocks.22.mlp.fc1.weight
agg_regator.frame_blocks.22.mlp.fc1.bias
agg_regator.frame_blocks.22.mlp.fc2.weight
agg_regator.frame_blocks.22.mlp.fc2.bias
agg_regator.frame_blocks.22.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ frame_blocks.23 (18)</summary>

```text
agg_regator.frame_blocks.23.norm1.weight
agg_regator.frame_blocks.23.norm1.bias
agg_regator.frame_blocks.23.attn.qkv.weight
agg_regator.frame_blocks.23.attn.qkv.bias
agg_regator.frame_blocks.23.attn.q_norm.weight
agg_regator.frame_blocks.23.attn.q_norm.bias
agg_regator.frame_blocks.23.attn.k_norm.weight
agg_regator.frame_blocks.23.attn.k_norm.bias
agg_regator.frame_blocks.23.attn.proj.weight
agg_regator.frame_blocks.23.attn.proj.bias
agg_regator.frame_blocks.23.ls1.gamma
agg_regator.frame_blocks.23.norm2.weight
agg_regator.frame_blocks.23.norm2.bias
agg_regator.frame_blocks.23.mlp.fc1.weight
agg_regator.frame_blocks.23.mlp.fc1.bias
agg_regator.frame_blocks.23.mlp.fc2.weight
agg_regator.frame_blocks.23.mlp.fc2.bias
agg_regator.frame_blocks.23.ls2.gamma
```
</details>
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;└─ global_blocks (480)</summary>

<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.0 (18)</summary>

```text
agg_regator.global_blocks.0.norm1.weight
agg_regator.global_blocks.0.norm1.bias
agg_regator.global_blocks.0.attn.qkv.weight
agg_regator.global_blocks.0.attn.qkv.bias
agg_regator.global_blocks.0.attn.q_norm.weight
agg_regator.global_blocks.0.attn.q_norm.bias
agg_regator.global_blocks.0.attn.k_norm.weight
agg_regator.global_blocks.0.attn.k_norm.bias
agg_regator.global_blocks.0.attn.proj.weight
agg_regator.global_blocks.0.attn.proj.bias
agg_regator.global_blocks.0.ls1.gamma
agg_regator.global_blocks.0.norm2.weight
agg_regator.global_blocks.0.norm2.bias
agg_regator.global_blocks.0.mlp.fc1.weight
agg_regator.global_blocks.0.mlp.fc1.bias
agg_regator.global_blocks.0.mlp.fc2.weight
agg_regator.global_blocks.0.mlp.fc2.bias
agg_regator.global_blocks.0.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.1 (18)</summary>

```text
agg_regator.global_blocks.1.norm1.weight
agg_regator.global_blocks.1.norm1.bias
agg_regator.global_blocks.1.attn.qkv.weight
agg_regator.global_blocks.1.attn.qkv.bias
agg_regator.global_blocks.1.attn.q_norm.weight
agg_regator.global_blocks.1.attn.q_norm.bias
agg_regator.global_blocks.1.attn.k_norm.weight
agg_regator.global_blocks.1.attn.k_norm.bias
agg_regator.global_blocks.1.attn.proj.weight
agg_regator.global_blocks.1.attn.proj.bias
agg_regator.global_blocks.1.ls1.gamma
agg_regator.global_blocks.1.norm2.weight
agg_regator.global_blocks.1.norm2.bias
agg_regator.global_blocks.1.mlp.fc1.weight
agg_regator.global_blocks.1.mlp.fc1.bias
agg_regator.global_blocks.1.mlp.fc2.weight
agg_regator.global_blocks.1.mlp.fc2.bias
agg_regator.global_blocks.1.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.2 (18)</summary>

```text
agg_regator.global_blocks.2.norm1.weight
agg_regator.global_blocks.2.norm1.bias
agg_regator.global_blocks.2.attn.qkv.weight
agg_regator.global_blocks.2.attn.qkv.bias
agg_regator.global_blocks.2.attn.q_norm.weight
agg_regator.global_blocks.2.attn.q_norm.bias
agg_regator.global_blocks.2.attn.k_norm.weight
agg_regator.global_blocks.2.attn.k_norm.bias
agg_regator.global_blocks.2.attn.proj.weight
agg_regator.global_blocks.2.attn.proj.bias
agg_regator.global_blocks.2.ls1.gamma
agg_regator.global_blocks.2.norm2.weight
agg_regator.global_blocks.2.norm2.bias
agg_regator.global_blocks.2.mlp.fc1.weight
agg_regator.global_blocks.2.mlp.fc1.bias
agg_regator.global_blocks.2.mlp.fc2.weight
agg_regator.global_blocks.2.mlp.fc2.bias
agg_regator.global_blocks.2.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.3 (18)</summary>

```text
agg_regator.global_blocks.3.norm1.weight
agg_regator.global_blocks.3.norm1.bias
agg_regator.global_blocks.3.attn.qkv.weight
agg_regator.global_blocks.3.attn.qkv.bias
agg_regator.global_blocks.3.attn.q_norm.weight
agg_regator.global_blocks.3.attn.q_norm.bias
agg_regator.global_blocks.3.attn.k_norm.weight
agg_regator.global_blocks.3.attn.k_norm.bias
agg_regator.global_blocks.3.attn.proj.weight
agg_regator.global_blocks.3.attn.proj.bias
agg_regator.global_blocks.3.ls1.gamma
agg_regator.global_blocks.3.norm2.weight
agg_regator.global_blocks.3.norm2.bias
agg_regator.global_blocks.3.mlp.fc1.weight
agg_regator.global_blocks.3.mlp.fc1.bias
agg_regator.global_blocks.3.mlp.fc2.weight
agg_regator.global_blocks.3.mlp.fc2.bias
agg_regator.global_blocks.3.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.4 (18)</summary>

```text
agg_regator.global_blocks.4.norm1.weight
agg_regator.global_blocks.4.norm1.bias
agg_regator.global_blocks.4.attn.qkv.weight
agg_regator.global_blocks.4.attn.qkv.bias
agg_regator.global_blocks.4.attn.q_norm.weight
agg_regator.global_blocks.4.attn.q_norm.bias
agg_regator.global_blocks.4.attn.k_norm.weight
agg_regator.global_blocks.4.attn.k_norm.bias
agg_regator.global_blocks.4.attn.proj.weight
agg_regator.global_blocks.4.attn.proj.bias
agg_regator.global_blocks.4.ls1.gamma
agg_regator.global_blocks.4.norm2.weight
agg_regator.global_blocks.4.norm2.bias
agg_regator.global_blocks.4.mlp.fc1.weight
agg_regator.global_blocks.4.mlp.fc1.bias
agg_regator.global_blocks.4.mlp.fc2.weight
agg_regator.global_blocks.4.mlp.fc2.bias
agg_regator.global_blocks.4.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.5 (18)</summary>

```text
agg_regator.global_blocks.5.norm1.weight
agg_regator.global_blocks.5.norm1.bias
agg_regator.global_blocks.5.attn.qkv.weight
agg_regator.global_blocks.5.attn.qkv.bias
agg_regator.global_blocks.5.attn.q_norm.weight
agg_regator.global_blocks.5.attn.q_norm.bias
agg_regator.global_blocks.5.attn.k_norm.weight
agg_regator.global_blocks.5.attn.k_norm.bias
agg_regator.global_blocks.5.attn.proj.weight
agg_regator.global_blocks.5.attn.proj.bias
agg_regator.global_blocks.5.ls1.gamma
agg_regator.global_blocks.5.norm2.weight
agg_regator.global_blocks.5.norm2.bias
agg_regator.global_blocks.5.mlp.fc1.weight
agg_regator.global_blocks.5.mlp.fc1.bias
agg_regator.global_blocks.5.mlp.fc2.weight
agg_regator.global_blocks.5.mlp.fc2.bias
agg_regator.global_blocks.5.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.6 (18)</summary>

```text
agg_regator.global_blocks.6.norm1.weight
agg_regator.global_blocks.6.norm1.bias
agg_regator.global_blocks.6.attn.qkv.weight
agg_regator.global_blocks.6.attn.qkv.bias
agg_regator.global_blocks.6.attn.q_norm.weight
agg_regator.global_blocks.6.attn.q_norm.bias
agg_regator.global_blocks.6.attn.k_norm.weight
agg_regator.global_blocks.6.attn.k_norm.bias
agg_regator.global_blocks.6.attn.proj.weight
agg_regator.global_blocks.6.attn.proj.bias
agg_regator.global_blocks.6.ls1.gamma
agg_regator.global_blocks.6.norm2.weight
agg_regator.global_blocks.6.norm2.bias
agg_regator.global_blocks.6.mlp.fc1.weight
agg_regator.global_blocks.6.mlp.fc1.bias
agg_regator.global_blocks.6.mlp.fc2.weight
agg_regator.global_blocks.6.mlp.fc2.bias
agg_regator.global_blocks.6.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.7 (18)</summary>

```text
agg_regator.global_blocks.7.norm1.weight
agg_regator.global_blocks.7.norm1.bias
agg_regator.global_blocks.7.attn.qkv.weight
agg_regator.global_blocks.7.attn.qkv.bias
agg_regator.global_blocks.7.attn.q_norm.weight
agg_regator.global_blocks.7.attn.q_norm.bias
agg_regator.global_blocks.7.attn.k_norm.weight
agg_regator.global_blocks.7.attn.k_norm.bias
agg_regator.global_blocks.7.attn.proj.weight
agg_regator.global_blocks.7.attn.proj.bias
agg_regator.global_blocks.7.ls1.gamma
agg_regator.global_blocks.7.norm2.weight
agg_regator.global_blocks.7.norm2.bias
agg_regator.global_blocks.7.mlp.fc1.weight
agg_regator.global_blocks.7.mlp.fc1.bias
agg_regator.global_blocks.7.mlp.fc2.weight
agg_regator.global_blocks.7.mlp.fc2.bias
agg_regator.global_blocks.7.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.8 (18)</summary>

```text
agg_regator.global_blocks.8.norm1.weight
agg_regator.global_blocks.8.norm1.bias
agg_regator.global_blocks.8.attn.qkv.weight
agg_regator.global_blocks.8.attn.qkv.bias
agg_regator.global_blocks.8.attn.q_norm.weight
agg_regator.global_blocks.8.attn.q_norm.bias
agg_regator.global_blocks.8.attn.k_norm.weight
agg_regator.global_blocks.8.attn.k_norm.bias
agg_regator.global_blocks.8.attn.proj.weight
agg_regator.global_blocks.8.attn.proj.bias
agg_regator.global_blocks.8.ls1.gamma
agg_regator.global_blocks.8.norm2.weight
agg_regator.global_blocks.8.norm2.bias
agg_regator.global_blocks.8.mlp.fc1.weight
agg_regator.global_blocks.8.mlp.fc1.bias
agg_regator.global_blocks.8.mlp.fc2.weight
agg_regator.global_blocks.8.mlp.fc2.bias
agg_regator.global_blocks.8.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.9 (18)</summary>

```text
agg_regator.global_blocks.9.norm1.weight
agg_regator.global_blocks.9.norm1.bias
agg_regator.global_blocks.9.attn.qkv.weight
agg_regator.global_blocks.9.attn.qkv.bias
agg_regator.global_blocks.9.attn.q_norm.weight
agg_regator.global_blocks.9.attn.q_norm.bias
agg_regator.global_blocks.9.attn.k_norm.weight
agg_regator.global_blocks.9.attn.k_norm.bias
agg_regator.global_blocks.9.attn.proj.weight
agg_regator.global_blocks.9.attn.proj.bias
agg_regator.global_blocks.9.ls1.gamma
agg_regator.global_blocks.9.norm2.weight
agg_regator.global_blocks.9.norm2.bias
agg_regator.global_blocks.9.mlp.fc1.weight
agg_regator.global_blocks.9.mlp.fc1.bias
agg_regator.global_blocks.9.mlp.fc2.weight
agg_regator.global_blocks.9.mlp.fc2.bias
agg_regator.global_blocks.9.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.10 (18)</summary>

```text
agg_regator.global_blocks.10.norm1.weight
agg_regator.global_blocks.10.norm1.bias
agg_regator.global_blocks.10.attn.qkv.weight
agg_regator.global_blocks.10.attn.qkv.bias
agg_regator.global_blocks.10.attn.q_norm.weight
agg_regator.global_blocks.10.attn.q_norm.bias
agg_regator.global_blocks.10.attn.k_norm.weight
agg_regator.global_blocks.10.attn.k_norm.bias
agg_regator.global_blocks.10.attn.proj.weight
agg_regator.global_blocks.10.attn.proj.bias
agg_regator.global_blocks.10.ls1.gamma
agg_regator.global_blocks.10.norm2.weight
agg_regator.global_blocks.10.norm2.bias
agg_regator.global_blocks.10.mlp.fc1.weight
agg_regator.global_blocks.10.mlp.fc1.bias
agg_regator.global_blocks.10.mlp.fc2.weight
agg_regator.global_blocks.10.mlp.fc2.bias
agg_regator.global_blocks.10.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.11 (18)</summary>

```text
agg_regator.global_blocks.11.norm1.weight
agg_regator.global_blocks.11.norm1.bias
agg_regator.global_blocks.11.attn.qkv.weight
agg_regator.global_blocks.11.attn.qkv.bias
agg_regator.global_blocks.11.attn.q_norm.weight
agg_regator.global_blocks.11.attn.q_norm.bias
agg_regator.global_blocks.11.attn.k_norm.weight
agg_regator.global_blocks.11.attn.k_norm.bias
agg_regator.global_blocks.11.attn.proj.weight
agg_regator.global_blocks.11.attn.proj.bias
agg_regator.global_blocks.11.ls1.gamma
agg_regator.global_blocks.11.norm2.weight
agg_regator.global_blocks.11.norm2.bias
agg_regator.global_blocks.11.mlp.fc1.weight
agg_regator.global_blocks.11.mlp.fc1.bias
agg_regator.global_blocks.11.mlp.fc2.weight
agg_regator.global_blocks.11.mlp.fc2.bias
agg_regator.global_blocks.11.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.12 (18)</summary>

```text
agg_regator.global_blocks.12.norm1.weight
agg_regator.global_blocks.12.norm1.bias
agg_regator.global_blocks.12.attn.qkv.weight
agg_regator.global_blocks.12.attn.qkv.bias
agg_regator.global_blocks.12.attn.q_norm.weight
agg_regator.global_blocks.12.attn.q_norm.bias
agg_regator.global_blocks.12.attn.k_norm.weight
agg_regator.global_blocks.12.attn.k_norm.bias
agg_regator.global_blocks.12.attn.proj.weight
agg_regator.global_blocks.12.attn.proj.bias
agg_regator.global_blocks.12.ls1.gamma
agg_regator.global_blocks.12.norm2.weight
agg_regator.global_blocks.12.norm2.bias
agg_regator.global_blocks.12.mlp.fc1.weight
agg_regator.global_blocks.12.mlp.fc1.bias
agg_regator.global_blocks.12.mlp.fc2.weight
agg_regator.global_blocks.12.mlp.fc2.bias
agg_regator.global_blocks.12.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.13 (18)</summary>

```text
agg_regator.global_blocks.13.norm1.weight
agg_regator.global_blocks.13.norm1.bias
agg_regator.global_blocks.13.attn.qkv.weight
agg_regator.global_blocks.13.attn.qkv.bias
agg_regator.global_blocks.13.attn.q_norm.weight
agg_regator.global_blocks.13.attn.q_norm.bias
agg_regator.global_blocks.13.attn.k_norm.weight
agg_regator.global_blocks.13.attn.k_norm.bias
agg_regator.global_blocks.13.attn.proj.weight
agg_regator.global_blocks.13.attn.proj.bias
agg_regator.global_blocks.13.ls1.gamma
agg_regator.global_blocks.13.norm2.weight
agg_regator.global_blocks.13.norm2.bias
agg_regator.global_blocks.13.mlp.fc1.weight
agg_regator.global_blocks.13.mlp.fc1.bias
agg_regator.global_blocks.13.mlp.fc2.weight
agg_regator.global_blocks.13.mlp.fc2.bias
agg_regator.global_blocks.13.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.14 / includes TTT fast-weight module (30)</summary>

```text
agg_regator.global_blocks.14.norm1.weight
agg_regator.global_blocks.14.norm1.bias
agg_regator.global_blocks.14.attn.qkv.weight
agg_regator.global_blocks.14.attn.qkv.bias
agg_regator.global_blocks.14.attn.q_norm.weight
agg_regator.global_blocks.14.attn.q_norm.bias
agg_regator.global_blocks.14.attn.k_norm.weight
agg_regator.global_blocks.14.attn.k_norm.bias
agg_regator.global_blocks.14.attn.proj.weight
agg_regator.global_blocks.14.attn.proj.bias
agg_regator.global_blocks.14.ls1.gamma
agg_regator.global_blocks.14.norm2.weight
agg_regator.global_blocks.14.norm2.bias
agg_regator.global_blocks.14.mlp.fc1.weight
agg_regator.global_blocks.14.mlp.fc1.bias
agg_regator.global_blocks.14.mlp.fc2.weight
agg_regator.global_blocks.14.mlp.fc2.bias
agg_regator.global_blocks.14.ls2.gamma
agg_regator.global_blocks.14.ttt.w0
agg_regator.global_blocks.14.ttt.w1
agg_regator.global_blocks.14.ttt.w2
agg_regator.global_blocks.14.ttt.qkv.weight
agg_regator.global_blocks.14.ttt.qkv.bias
agg_regator.global_blocks.14.ttt.q_norm.weight
agg_regator.global_blocks.14.ttt.k_norm.weight
agg_regator.global_blocks.14.ttt.proj.weight
agg_regator.global_blocks.14.ttt.proj.bias
agg_regator.global_blocks.14.ttt.lrs.weight
agg_regator.global_blocks.14.ttt.lrs.bias
agg_regator.global_blocks.14.ttt.o_norm.weight
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.15 (18)</summary>

```text
agg_regator.global_blocks.15.norm1.weight
agg_regator.global_blocks.15.norm1.bias
agg_regator.global_blocks.15.attn.qkv.weight
agg_regator.global_blocks.15.attn.qkv.bias
agg_regator.global_blocks.15.attn.q_norm.weight
agg_regator.global_blocks.15.attn.q_norm.bias
agg_regator.global_blocks.15.attn.k_norm.weight
agg_regator.global_blocks.15.attn.k_norm.bias
agg_regator.global_blocks.15.attn.proj.weight
agg_regator.global_blocks.15.attn.proj.bias
agg_regator.global_blocks.15.ls1.gamma
agg_regator.global_blocks.15.norm2.weight
agg_regator.global_blocks.15.norm2.bias
agg_regator.global_blocks.15.mlp.fc1.weight
agg_regator.global_blocks.15.mlp.fc1.bias
agg_regator.global_blocks.15.mlp.fc2.weight
agg_regator.global_blocks.15.mlp.fc2.bias
agg_regator.global_blocks.15.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.16 (18)</summary>

```text
agg_regator.global_blocks.16.norm1.weight
agg_regator.global_blocks.16.norm1.bias
agg_regator.global_blocks.16.attn.qkv.weight
agg_regator.global_blocks.16.attn.qkv.bias
agg_regator.global_blocks.16.attn.q_norm.weight
agg_regator.global_blocks.16.attn.q_norm.bias
agg_regator.global_blocks.16.attn.k_norm.weight
agg_regator.global_blocks.16.attn.k_norm.bias
agg_regator.global_blocks.16.attn.proj.weight
agg_regator.global_blocks.16.attn.proj.bias
agg_regator.global_blocks.16.ls1.gamma
agg_regator.global_blocks.16.norm2.weight
agg_regator.global_blocks.16.norm2.bias
agg_regator.global_blocks.16.mlp.fc1.weight
agg_regator.global_blocks.16.mlp.fc1.bias
agg_regator.global_blocks.16.mlp.fc2.weight
agg_regator.global_blocks.16.mlp.fc2.bias
agg_regator.global_blocks.16.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.17 / includes TTT fast-weight module (30)</summary>

```text
agg_regator.global_blocks.17.norm1.weight
agg_regator.global_blocks.17.norm1.bias
agg_regator.global_blocks.17.attn.qkv.weight
agg_regator.global_blocks.17.attn.qkv.bias
agg_regator.global_blocks.17.attn.q_norm.weight
agg_regator.global_blocks.17.attn.q_norm.bias
agg_regator.global_blocks.17.attn.k_norm.weight
agg_regator.global_blocks.17.attn.k_norm.bias
agg_regator.global_blocks.17.attn.proj.weight
agg_regator.global_blocks.17.attn.proj.bias
agg_regator.global_blocks.17.ls1.gamma
agg_regator.global_blocks.17.norm2.weight
agg_regator.global_blocks.17.norm2.bias
agg_regator.global_blocks.17.mlp.fc1.weight
agg_regator.global_blocks.17.mlp.fc1.bias
agg_regator.global_blocks.17.mlp.fc2.weight
agg_regator.global_blocks.17.mlp.fc2.bias
agg_regator.global_blocks.17.ls2.gamma
agg_regator.global_blocks.17.ttt.w0
agg_regator.global_blocks.17.ttt.w1
agg_regator.global_blocks.17.ttt.w2
agg_regator.global_blocks.17.ttt.qkv.weight
agg_regator.global_blocks.17.ttt.qkv.bias
agg_regator.global_blocks.17.ttt.q_norm.weight
agg_regator.global_blocks.17.ttt.k_norm.weight
agg_regator.global_blocks.17.ttt.proj.weight
agg_regator.global_blocks.17.ttt.proj.bias
agg_regator.global_blocks.17.ttt.lrs.weight
agg_regator.global_blocks.17.ttt.lrs.bias
agg_regator.global_blocks.17.ttt.o_norm.weight
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.18 (18)</summary>

```text
agg_regator.global_blocks.18.norm1.weight
agg_regator.global_blocks.18.norm1.bias
agg_regator.global_blocks.18.attn.qkv.weight
agg_regator.global_blocks.18.attn.qkv.bias
agg_regator.global_blocks.18.attn.q_norm.weight
agg_regator.global_blocks.18.attn.q_norm.bias
agg_regator.global_blocks.18.attn.k_norm.weight
agg_regator.global_blocks.18.attn.k_norm.bias
agg_regator.global_blocks.18.attn.proj.weight
agg_regator.global_blocks.18.attn.proj.bias
agg_regator.global_blocks.18.ls1.gamma
agg_regator.global_blocks.18.norm2.weight
agg_regator.global_blocks.18.norm2.bias
agg_regator.global_blocks.18.mlp.fc1.weight
agg_regator.global_blocks.18.mlp.fc1.bias
agg_regator.global_blocks.18.mlp.fc2.weight
agg_regator.global_blocks.18.mlp.fc2.bias
agg_regator.global_blocks.18.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.19 (18)</summary>

```text
agg_regator.global_blocks.19.norm1.weight
agg_regator.global_blocks.19.norm1.bias
agg_regator.global_blocks.19.attn.qkv.weight
agg_regator.global_blocks.19.attn.qkv.bias
agg_regator.global_blocks.19.attn.q_norm.weight
agg_regator.global_blocks.19.attn.q_norm.bias
agg_regator.global_blocks.19.attn.k_norm.weight
agg_regator.global_blocks.19.attn.k_norm.bias
agg_regator.global_blocks.19.attn.proj.weight
agg_regator.global_blocks.19.attn.proj.bias
agg_regator.global_blocks.19.ls1.gamma
agg_regator.global_blocks.19.norm2.weight
agg_regator.global_blocks.19.norm2.bias
agg_regator.global_blocks.19.mlp.fc1.weight
agg_regator.global_blocks.19.mlp.fc1.bias
agg_regator.global_blocks.19.mlp.fc2.weight
agg_regator.global_blocks.19.mlp.fc2.bias
agg_regator.global_blocks.19.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.20 / includes TTT fast-weight module (30)</summary>

```text
agg_regator.global_blocks.20.norm1.weight
agg_regator.global_blocks.20.norm1.bias
agg_regator.global_blocks.20.attn.qkv.weight
agg_regator.global_blocks.20.attn.qkv.bias
agg_regator.global_blocks.20.attn.q_norm.weight
agg_regator.global_blocks.20.attn.q_norm.bias
agg_regator.global_blocks.20.attn.k_norm.weight
agg_regator.global_blocks.20.attn.k_norm.bias
agg_regator.global_blocks.20.attn.proj.weight
agg_regator.global_blocks.20.attn.proj.bias
agg_regator.global_blocks.20.ls1.gamma
agg_regator.global_blocks.20.norm2.weight
agg_regator.global_blocks.20.norm2.bias
agg_regator.global_blocks.20.mlp.fc1.weight
agg_regator.global_blocks.20.mlp.fc1.bias
agg_regator.global_blocks.20.mlp.fc2.weight
agg_regator.global_blocks.20.mlp.fc2.bias
agg_regator.global_blocks.20.ls2.gamma
agg_regator.global_blocks.20.ttt.w0
agg_regator.global_blocks.20.ttt.w1
agg_regator.global_blocks.20.ttt.w2
agg_regator.global_blocks.20.ttt.qkv.weight
agg_regator.global_blocks.20.ttt.qkv.bias
agg_regator.global_blocks.20.ttt.q_norm.weight
agg_regator.global_blocks.20.ttt.k_norm.weight
agg_regator.global_blocks.20.ttt.proj.weight
agg_regator.global_blocks.20.ttt.proj.bias
agg_regator.global_blocks.20.ttt.lrs.weight
agg_regator.global_blocks.20.ttt.lrs.bias
agg_regator.global_blocks.20.ttt.o_norm.weight
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.21 (18)</summary>

```text
agg_regator.global_blocks.21.norm1.weight
agg_regator.global_blocks.21.norm1.bias
agg_regator.global_blocks.21.attn.qkv.weight
agg_regator.global_blocks.21.attn.qkv.bias
agg_regator.global_blocks.21.attn.q_norm.weight
agg_regator.global_blocks.21.attn.q_norm.bias
agg_regator.global_blocks.21.attn.k_norm.weight
agg_regator.global_blocks.21.attn.k_norm.bias
agg_regator.global_blocks.21.attn.proj.weight
agg_regator.global_blocks.21.attn.proj.bias
agg_regator.global_blocks.21.ls1.gamma
agg_regator.global_blocks.21.norm2.weight
agg_regator.global_blocks.21.norm2.bias
agg_regator.global_blocks.21.mlp.fc1.weight
agg_regator.global_blocks.21.mlp.fc1.bias
agg_regator.global_blocks.21.mlp.fc2.weight
agg_regator.global_blocks.21.mlp.fc2.bias
agg_regator.global_blocks.21.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ global_blocks.22 (18)</summary>

```text
agg_regator.global_blocks.22.norm1.weight
agg_regator.global_blocks.22.norm1.bias
agg_regator.global_blocks.22.attn.qkv.weight
agg_regator.global_blocks.22.attn.qkv.bias
agg_regator.global_blocks.22.attn.q_norm.weight
agg_regator.global_blocks.22.attn.q_norm.bias
agg_regator.global_blocks.22.attn.k_norm.weight
agg_regator.global_blocks.22.attn.k_norm.bias
agg_regator.global_blocks.22.attn.proj.weight
agg_regator.global_blocks.22.attn.proj.bias
agg_regator.global_blocks.22.ls1.gamma
agg_regator.global_blocks.22.norm2.weight
agg_regator.global_blocks.22.norm2.bias
agg_regator.global_blocks.22.mlp.fc1.weight
agg_regator.global_blocks.22.mlp.fc1.bias
agg_regator.global_blocks.22.mlp.fc2.weight
agg_regator.global_blocks.22.mlp.fc2.bias
agg_regator.global_blocks.22.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ global_blocks.23 / includes TTT fast-weight module (30)</summary>

```text
agg_regator.global_blocks.23.norm1.weight
agg_regator.global_blocks.23.norm1.bias
agg_regator.global_blocks.23.attn.qkv.weight
agg_regator.global_blocks.23.attn.qkv.bias
agg_regator.global_blocks.23.attn.q_norm.weight
agg_regator.global_blocks.23.attn.q_norm.bias
agg_regator.global_blocks.23.attn.k_norm.weight
agg_regator.global_blocks.23.attn.k_norm.bias
agg_regator.global_blocks.23.attn.proj.weight
agg_regator.global_blocks.23.attn.proj.bias
agg_regator.global_blocks.23.ls1.gamma
agg_regator.global_blocks.23.norm2.weight
agg_regator.global_blocks.23.norm2.bias
agg_regator.global_blocks.23.mlp.fc1.weight
agg_regator.global_blocks.23.mlp.fc1.bias
agg_regator.global_blocks.23.mlp.fc2.weight
agg_regator.global_blocks.23.mlp.fc2.bias
agg_regator.global_blocks.23.ls2.gamma
agg_regator.global_blocks.23.ttt.w0
agg_regator.global_blocks.23.ttt.w1
agg_regator.global_blocks.23.ttt.w2
agg_regator.global_blocks.23.ttt.qkv.weight
agg_regator.global_blocks.23.ttt.qkv.bias
agg_regator.global_blocks.23.ttt.q_norm.weight
agg_regator.global_blocks.23.ttt.k_norm.weight
agg_regator.global_blocks.23.ttt.proj.weight
agg_regator.global_blocks.23.ttt.proj.bias
agg_regator.global_blocks.23.ttt.lrs.weight
agg_regator.global_blocks.23.ttt.lrs.bias
agg_regator.global_blocks.23.ttt.o_norm.weight
```
</details>
</details>
</details>

<details>
<summary>├─ cam_decoder (69)</summary>

<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ cam_decoder root / iterative pose layers (13)</summary>

```text
cam_decoder.empty_pose_tokens
cam_decoder.token_norm.weight
cam_decoder.token_norm.bias
cam_decoder.trunk_norm.weight
cam_decoder.trunk_norm.bias
cam_decoder.embed_pose.weight
cam_decoder.embed_pose.bias
cam_decoder.poseLN_modulation.1.weight
cam_decoder.poseLN_modulation.1.bias
cam_decoder.pose_branch.fc1.weight
cam_decoder.pose_branch.fc1.bias
cam_decoder.pose_branch.fc2.weight
cam_decoder.pose_branch.fc2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;└─ cam_decoder transformer trunk (56)</summary>

<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ trunk.0 (14)</summary>

```text
cam_decoder.trunk.0.norm1.weight
cam_decoder.trunk.0.norm1.bias
cam_decoder.trunk.0.attn.qkv.weight
cam_decoder.trunk.0.attn.qkv.bias
cam_decoder.trunk.0.attn.proj.weight
cam_decoder.trunk.0.attn.proj.bias
cam_decoder.trunk.0.ls1.gamma
cam_decoder.trunk.0.norm2.weight
cam_decoder.trunk.0.norm2.bias
cam_decoder.trunk.0.mlp.fc1.weight
cam_decoder.trunk.0.mlp.fc1.bias
cam_decoder.trunk.0.mlp.fc2.weight
cam_decoder.trunk.0.mlp.fc2.bias
cam_decoder.trunk.0.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ trunk.1 (14)</summary>

```text
cam_decoder.trunk.1.norm1.weight
cam_decoder.trunk.1.norm1.bias
cam_decoder.trunk.1.attn.qkv.weight
cam_decoder.trunk.1.attn.qkv.bias
cam_decoder.trunk.1.attn.proj.weight
cam_decoder.trunk.1.attn.proj.bias
cam_decoder.trunk.1.ls1.gamma
cam_decoder.trunk.1.norm2.weight
cam_decoder.trunk.1.norm2.bias
cam_decoder.trunk.1.mlp.fc1.weight
cam_decoder.trunk.1.mlp.fc1.bias
cam_decoder.trunk.1.mlp.fc2.weight
cam_decoder.trunk.1.mlp.fc2.bias
cam_decoder.trunk.1.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ trunk.2 (14)</summary>

```text
cam_decoder.trunk.2.norm1.weight
cam_decoder.trunk.2.norm1.bias
cam_decoder.trunk.2.attn.qkv.weight
cam_decoder.trunk.2.attn.qkv.bias
cam_decoder.trunk.2.attn.proj.weight
cam_decoder.trunk.2.attn.proj.bias
cam_decoder.trunk.2.ls1.gamma
cam_decoder.trunk.2.norm2.weight
cam_decoder.trunk.2.norm2.bias
cam_decoder.trunk.2.mlp.fc1.weight
cam_decoder.trunk.2.mlp.fc1.bias
cam_decoder.trunk.2.mlp.fc2.weight
cam_decoder.trunk.2.mlp.fc2.bias
cam_decoder.trunk.2.ls2.gamma
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ trunk.3 (14)</summary>

```text
cam_decoder.trunk.3.norm1.weight
cam_decoder.trunk.3.norm1.bias
cam_decoder.trunk.3.attn.qkv.weight
cam_decoder.trunk.3.attn.qkv.bias
cam_decoder.trunk.3.attn.proj.weight
cam_decoder.trunk.3.attn.proj.bias
cam_decoder.trunk.3.ls1.gamma
cam_decoder.trunk.3.norm2.weight
cam_decoder.trunk.3.norm2.bias
cam_decoder.trunk.3.mlp.fc1.weight
cam_decoder.trunk.3.mlp.fc1.bias
cam_decoder.trunk.3.mlp.fc2.weight
cam_decoder.trunk.3.mlp.fc2.bias
cam_decoder.trunk.3.ls2.gamma
```
</details>
</details>
</details>

<details>
<summary>├─ xyz_decoder (62)</summary>

<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ xyz_decoder.norm (2)</summary>

```text
xyz_decoder.norm.weight
xyz_decoder.norm.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ xyz_decoder.projects (8)</summary>

```text
xyz_decoder.projects.0.weight
xyz_decoder.projects.0.bias
xyz_decoder.projects.1.weight
xyz_decoder.projects.1.bias
xyz_decoder.projects.2.weight
xyz_decoder.projects.2.bias
xyz_decoder.projects.3.weight
xyz_decoder.projects.3.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ xyz_decoder.resize_layers (6)</summary>

```text
xyz_decoder.resize_layers.0.weight
xyz_decoder.resize_layers.0.bias
xyz_decoder.resize_layers.1.weight
xyz_decoder.resize_layers.1.bias
xyz_decoder.resize_layers.3.weight
xyz_decoder.resize_layers.3.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ xyz_decoder.scratch.layer*_rn (4)</summary>

```text
xyz_decoder.scratch.layer1_rn.weight
xyz_decoder.scratch.layer2_rn.weight
xyz_decoder.scratch.layer3_rn.weight
xyz_decoder.scratch.layer4_rn.weight
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ xyz_decoder.scratch.refinenet1 (10)</summary>

```text
xyz_decoder.scratch.refinenet1.out_conv.weight
xyz_decoder.scratch.refinenet1.out_conv.bias
xyz_decoder.scratch.refinenet1.resConfUnit1.conv1.weight
xyz_decoder.scratch.refinenet1.resConfUnit1.conv1.bias
xyz_decoder.scratch.refinenet1.resConfUnit1.conv2.weight
xyz_decoder.scratch.refinenet1.resConfUnit1.conv2.bias
xyz_decoder.scratch.refinenet1.resConfUnit2.conv1.weight
xyz_decoder.scratch.refinenet1.resConfUnit2.conv1.bias
xyz_decoder.scratch.refinenet1.resConfUnit2.conv2.weight
xyz_decoder.scratch.refinenet1.resConfUnit2.conv2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ xyz_decoder.scratch.refinenet2 (10)</summary>

```text
xyz_decoder.scratch.refinenet2.out_conv.weight
xyz_decoder.scratch.refinenet2.out_conv.bias
xyz_decoder.scratch.refinenet2.resConfUnit1.conv1.weight
xyz_decoder.scratch.refinenet2.resConfUnit1.conv1.bias
xyz_decoder.scratch.refinenet2.resConfUnit1.conv2.weight
xyz_decoder.scratch.refinenet2.resConfUnit1.conv2.bias
xyz_decoder.scratch.refinenet2.resConfUnit2.conv1.weight
xyz_decoder.scratch.refinenet2.resConfUnit2.conv1.bias
xyz_decoder.scratch.refinenet2.resConfUnit2.conv2.weight
xyz_decoder.scratch.refinenet2.resConfUnit2.conv2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ xyz_decoder.scratch.refinenet3 (10)</summary>

```text
xyz_decoder.scratch.refinenet3.out_conv.weight
xyz_decoder.scratch.refinenet3.out_conv.bias
xyz_decoder.scratch.refinenet3.resConfUnit1.conv1.weight
xyz_decoder.scratch.refinenet3.resConfUnit1.conv1.bias
xyz_decoder.scratch.refinenet3.resConfUnit1.conv2.weight
xyz_decoder.scratch.refinenet3.resConfUnit1.conv2.bias
xyz_decoder.scratch.refinenet3.resConfUnit2.conv1.weight
xyz_decoder.scratch.refinenet3.resConfUnit2.conv1.bias
xyz_decoder.scratch.refinenet3.resConfUnit2.conv2.weight
xyz_decoder.scratch.refinenet3.resConfUnit2.conv2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ xyz_decoder.scratch.refinenet4 (6)</summary>

```text
xyz_decoder.scratch.refinenet4.out_conv.weight
xyz_decoder.scratch.refinenet4.out_conv.bias
xyz_decoder.scratch.refinenet4.resConfUnit2.conv1.weight
xyz_decoder.scratch.refinenet4.resConfUnit2.conv1.bias
xyz_decoder.scratch.refinenet4.resConfUnit2.conv2.weight
xyz_decoder.scratch.refinenet4.resConfUnit2.conv2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;└─ xyz_decoder.scratch.output_conv* (6)</summary>

```text
xyz_decoder.scratch.output_conv1.weight
xyz_decoder.scratch.output_conv1.bias
xyz_decoder.scratch.output_conv2.0.weight
xyz_decoder.scratch.output_conv2.0.bias
xyz_decoder.scratch.output_conv2.2.weight
xyz_decoder.scratch.output_conv2.2.bias
```
</details>
</details>

<details>
<summary>└─ dpt_decoder (62)</summary>

<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ dpt_decoder.norm (2)</summary>

```text
dpt_decoder.norm.weight
dpt_decoder.norm.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ dpt_decoder.projects (8)</summary>

```text
dpt_decoder.projects.0.weight
dpt_decoder.projects.0.bias
dpt_decoder.projects.1.weight
dpt_decoder.projects.1.bias
dpt_decoder.projects.2.weight
dpt_decoder.projects.2.bias
dpt_decoder.projects.3.weight
dpt_decoder.projects.3.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ dpt_decoder.resize_layers (6)</summary>

```text
dpt_decoder.resize_layers.0.weight
dpt_decoder.resize_layers.0.bias
dpt_decoder.resize_layers.1.weight
dpt_decoder.resize_layers.1.bias
dpt_decoder.resize_layers.3.weight
dpt_decoder.resize_layers.3.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ dpt_decoder.scratch.layer*_rn (4)</summary>

```text
dpt_decoder.scratch.layer1_rn.weight
dpt_decoder.scratch.layer2_rn.weight
dpt_decoder.scratch.layer3_rn.weight
dpt_decoder.scratch.layer4_rn.weight
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ dpt_decoder.scratch.refinenet1 (10)</summary>

```text
dpt_decoder.scratch.refinenet1.out_conv.weight
dpt_decoder.scratch.refinenet1.out_conv.bias
dpt_decoder.scratch.refinenet1.resConfUnit1.conv1.weight
dpt_decoder.scratch.refinenet1.resConfUnit1.conv1.bias
dpt_decoder.scratch.refinenet1.resConfUnit1.conv2.weight
dpt_decoder.scratch.refinenet1.resConfUnit1.conv2.bias
dpt_decoder.scratch.refinenet1.resConfUnit2.conv1.weight
dpt_decoder.scratch.refinenet1.resConfUnit2.conv1.bias
dpt_decoder.scratch.refinenet1.resConfUnit2.conv2.weight
dpt_decoder.scratch.refinenet1.resConfUnit2.conv2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ dpt_decoder.scratch.refinenet2 (10)</summary>

```text
dpt_decoder.scratch.refinenet2.out_conv.weight
dpt_decoder.scratch.refinenet2.out_conv.bias
dpt_decoder.scratch.refinenet2.resConfUnit1.conv1.weight
dpt_decoder.scratch.refinenet2.resConfUnit1.conv1.bias
dpt_decoder.scratch.refinenet2.resConfUnit1.conv2.weight
dpt_decoder.scratch.refinenet2.resConfUnit1.conv2.bias
dpt_decoder.scratch.refinenet2.resConfUnit2.conv1.weight
dpt_decoder.scratch.refinenet2.resConfUnit2.conv1.bias
dpt_decoder.scratch.refinenet2.resConfUnit2.conv2.weight
dpt_decoder.scratch.refinenet2.resConfUnit2.conv2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ dpt_decoder.scratch.refinenet3 (10)</summary>

```text
dpt_decoder.scratch.refinenet3.out_conv.weight
dpt_decoder.scratch.refinenet3.out_conv.bias
dpt_decoder.scratch.refinenet3.resConfUnit1.conv1.weight
dpt_decoder.scratch.refinenet3.resConfUnit1.conv1.bias
dpt_decoder.scratch.refinenet3.resConfUnit1.conv2.weight
dpt_decoder.scratch.refinenet3.resConfUnit1.conv2.bias
dpt_decoder.scratch.refinenet3.resConfUnit2.conv1.weight
dpt_decoder.scratch.refinenet3.resConfUnit2.conv1.bias
dpt_decoder.scratch.refinenet3.resConfUnit2.conv2.weight
dpt_decoder.scratch.refinenet3.resConfUnit2.conv2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;├─ dpt_decoder.scratch.refinenet4 (6)</summary>

```text
dpt_decoder.scratch.refinenet4.out_conv.weight
dpt_decoder.scratch.refinenet4.out_conv.bias
dpt_decoder.scratch.refinenet4.resConfUnit2.conv1.weight
dpt_decoder.scratch.refinenet4.resConfUnit2.conv1.bias
dpt_decoder.scratch.refinenet4.resConfUnit2.conv2.weight
dpt_decoder.scratch.refinenet4.resConfUnit2.conv2.bias
```
</details>
<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;└─ dpt_decoder.scratch.output_conv* (6)</summary>

```text
dpt_decoder.scratch.output_conv1.weight
dpt_decoder.scratch.output_conv1.bias
dpt_decoder.scratch.output_conv2.0.weight
dpt_decoder.scratch.output_conv2.0.bias
dpt_decoder.scratch.output_conv2.2.weight
dpt_decoder.scratch.output_conv2.2.bias
```
</details>
</details>
