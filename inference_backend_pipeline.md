# `scal3r/pipelines/backend.py` Inference Backend Pipeline

This document explains the backend-only inference process that consumes an image directory, runs the `Scal3R` model over overlapping image blocks, aligns block predictions into a global reconstruction, and writes the final camera/depth/point-cloud artifacts.

The backend is normally launched by `scal3r/run.py` through `scal3r.pipelines.inference.run_inference(...)`, but it can also be run directly:

```bash
python -m scal3r.pipelines.backend \
  --config configs/models/scal3r.yaml \
  --input_dir /path/to/images \
  --result_dir data/result/custom/run \
  --runtime_dir data/result/custom/run/runtime
```

At a high level:

```text
backend.main()
  -> parse and resolve runtime args
  -> build_sampler_from_config(...)
  -> load_data(...)
  -> forward(...)
  -> post_process(...)
  -> save_results(...)
  -> cleanup_offload_root(...)
```

## 1. Backend Responsibilities

File: `scal3r/pipelines/backend.py`

The backend is the process that actually performs model inference. It owns:

- command-line parsing for the standalone backend process,
- path resolution for config/result/runtime/offload/probe locations,
- checkpointed model construction,
- image discovery and preprocessing,
- chunk/block construction for long image sequences,
- optional loop-closure block construction,
- `Scal3R` forward inference,
- test-time-training fast-weight application,
- block alignment and pose-graph optimization,
- final frame selection from overlapping blocks,
- result serialization.

It does not create the wrapper-level `run_plan.json` or `image_manifest.txt`; those are written by `scal3r/pipelines/inference.py` before the backend subprocess is started.

## 2. Runtime Arguments

`parse_args()` defines the standalone backend interface. The most important groups are:

Input and paths:

- `--config`: model/config YAML. Default: `configs/models/scal3r.yaml`.
- `--checkpoint`: optional model checkpoint override.
- `--input_dir`: required image directory.
- `--result_dir`: final output directory.
- `--runtime_dir`: runtime metadata and temporary artifact directory.
- `--image_patterns`: comma-separated glob patterns.

Image selection:

- `--max_images`: cap after frame slicing when positive.
- `--start_frame`: inclusive start index in sorted image paths.
- `--end_frame`: exclusive stop index; `-1` means through the end.
- `--interval`: stride.

Block construction:

- `--block_size`: frames per sequential block.
- `--overlap_size`: shared frames between adjacent blocks.
- `--loop_size`: frame window size for loop-closure blocks.

Alignment and loop closure:

- `--use_loop`: enable loop detection and loop blocks.
- `--loop_ckpt`: SALAD loop detector checkpoint.
- `--use_xyz_align`: choose direct predicted XYZ alignment instead of depth/ray-derived XYZ.
- `--max_align_points_per_frame`: optional point cap for PGO alignment.
- `--pgo_workers`: worker count for adjacent block alignment; `0` means auto.

Inference and output:

- `--device`: defaults to `cuda` if available, otherwise `cpu`.
- `--test_use_amp`: enables AMP in the main `forward(...)` call.
- `--save_dpt`: save depth maps.
- `--save_xyz`: save point clouds and masks.
- `--downsample_xyz_ratio`: point sampling ratio for saved point clouds.
- `--confidence_xyz_threshold`: confidence multiplier for point mask selection.

Memory/runtime debugging:

- `--streaming_state`: offload aggregator and decoder states to disk.
- `--offload_batches`: offload prepared image blocks.
- `--offload_outputs`: offload decoded block outputs.
- `--cleanup_offload`: remove temporary offload directory in `finally`.
- `--offload_dir`: optional explicit offload root.
- `--probe_dir`: write stage-by-stage JSON probes.
- `--stop_after_stage`: stop after an exact recorded stage name.

## 3. Startup in `main()`

`main()` begins by parsing args, resolving release-relative paths, and creating the device and recorder:

```text
args = parse_args()
args.config = resolve_release_path(args.config)
args.result_dir = resolve_release_path(args.result_dir or get_default_output_dir("run"))
args.runtime_dir = resolve_release_path(args.runtime_dir) or <result_dir>/runtime
device = torch.device(args.device)
recorder = StageRecorder(args.probe_dir or "", args.device)
```

Path resolution is also applied to:

- `args.checkpoint`
- `args.loop_ckpt`
- `args.offload_dir`
- `args.probe_dir`

The backend logs a compact runtime config via `format_runtime_config(args)`.

If probes are enabled, `StageRecorder` starts with:

```text
process.begin
load_model.begin
```

Any later `maybe_stop_after(stage, args, recorder)` call can raise `StopAfterStage` when `--stop_after_stage` exactly matches the stage name.

## 4. Model Loading

Call site:

```text
sampler, dataset_cfg = build_sampler_from_config(args.config, device, args.checkpoint)
```

Implementation: `scal3r/models/scal3r.py`

This function:

1. Loads the merged config.
2. Extracts `model_cfg.sampler_cfg`.
3. Instantiates `Scal3R`.
4. Resolves the checkpoint from `--checkpoint` or `model.checkpoint`.
5. Loads the state dict strictly.
6. Moves the model to `device`.
7. Calls `sampler.eval()`.
8. Returns the model and dataset preprocessing config.

The backend records:

```text
load_model.done
```

The returned model is called `sampler` in `backend.py`.

## 5. Input Image Loading and Block Construction

Call site:

```text
batches, indices = load_data(dataset_cfg, args, recorder=recorder)
```

`load_data(...)` converts the input image folder into a list of block payloads and block index ranges.

### 5.1 Image Discovery

The backend collects image paths with:

```text
collect_image_paths(args.input_dir, args.image_patterns)
```

`collect_image_paths(...)`:

1. Splits comma-separated patterns such as `*.png,*.jpg,*.jpeg,*.bmp`.
2. Globs each pattern inside `input_dir`.
3. Deduplicates and sorts paths.
4. Raises `FileNotFoundError` if no images match.

The sorted paths are then sliced with:

```text
apply_frame_range_to_sorted_paths(
  image_paths,
  start_frame=args.start_frame,
  end_frame=args.end_frame,
  interval=args.interval,
)
```

The semantics are Python slicing:

```text
sorted_paths[start_frame:end_frame:interval]
```

where `end_frame` is exclusive and a negative `end_frame` means the sequence end.

If the slice removes every frame, `load_data(...)` raises an error explaining that the frame range should be widened. If `max_images > 0`, the sliced list is capped afterward.

Probe stages:

```text
collect_images.begin
collect_images.done
```

### 5.2 Image Preprocessing

The backend calls:

```text
sequence, height, width = load_and_preprocess_images(
  image_paths,
  dataset_cfg,
  preprocess_workers=args.preprocess_workers,
)
```

For the first image, preprocessing:

1. Loads RGB bytes and normalizes to float tensor.
2. Creates an all-ones mask.
3. Builds a dummy intrinsic matrix from image size and `dataset_cfg.focal_ratio`.
4. Uses identity `w2c` for preprocessing transforms.
5. Applies optional `render_ratio`.
6. Applies optional `rot90`.
7. Computes a common target size with `proc_max_size` and `proc_align_size`.
8. Resizes to cover the target size.
9. Crops to the final target size and adjusts intrinsics.
10. Stores a flattened frame record.

Remaining images reuse the first image's target height/width. They are processed sequentially when `preprocess_workers == 1`; otherwise `parallel_execution(...)` processes them in parallel.

Each sequence entry contains:

```text
rgb: flattened RGB, shape (H * W, 3)
msk: flattened mask, shape (H * W, 1)
ixt: intrinsic matrix
path: source path
```

Probe stages:

```text
preprocess_images.begin
preprocess_images.done
```

### 5.3 Sequential Block Partitioning

After preprocessing, `load_data(...)` computes overlapping sequential blocks:

```text
n_blocks = ceil((n_samples - overlap_size) / (block_size - overlap_size))
```

In code:

```text
n_blocks = (n_samples - overlap_size + (block_size - overlap_size) - 1) // (block_size - overlap_size)
```

If all samples fit inside one block, it forces:

```text
block_size = n_samples
n_blocks = 1
```

The backend requires:

```text
block_size > overlap_size
```

Each block uses:

```text
block_indices = range(i * (block_size - overlap_size), min(...))
```

`build_image_only_block(...)` then creates a block `DotDict`.

Important block fields:

```text
H, W
src_inds
orig_src_inds
msk
meta.rgb
meta.H, meta.W
meta.overlap_size
meta.block_size
meta.aspect_ratio
meta.cam_param_type
meta.use_world_coord
```

`src_inds` are reordered by `reorder_middle_reference(...)`: the middle frame of the block is placed first, then the frames before and after it. `orig_src_inds` preserves the original chronological block indices for overlap handling during final assembly.

If `--offload_batches 1`, the prepared block is serialized to:

```text
<offload_root>/batches/<block_index>.pt
```

Probe stages:

```text
build_blocks.begin
build_blocks.done
```

### 5.4 Optional Loop-Closure Blocks

If `args.use_loop` is true:

1. The backend creates the runtime root.
2. `detect_loops(...)` searches loop pairs using `args.loop_ckpt`.
3. `build_loop_batches(...)` constructs extra loop-closure blocks from the preprocessed sequence.
4. These loop blocks are appended after normal sequential blocks.
5. `args.n_blocks_loop` records the number of appended loop blocks.

The returned `batches` can therefore contain:

```text
[normal sequential blocks..., loop closure blocks...]
```

Probe stages:

```text
loop_detection.begin
loop_detection.done
load_data.done
```

## 6. Model Forward Inference

Call site:

```text
with torch.no_grad():
  with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
    output = forward(sampler, batches, args, recorder=recorder)
```

`amp_enabled` is true only when `--test_use_amp` is passed and the device is CUDA. Otherwise the main forward path runs without AMP. Some inner TTT and decoder sections explicitly disable CUDA autocast.

`forward(...)` has three major phases:

```text
embedder -> aggregator/TTT -> decoders
```

It returns a list named `output`, with one decoded raw output per block.

### 6.1 Initial Shape Contract

The first batch determines:

```text
B, S = batch0.meta.rgb.shape[:2]
H, W = batch0.meta.H[0], batch0.meta.W[0]
N = len(batches)
```

The backend asserts:

```text
B == 1
```

So this backend implementation is for one image sequence split into blocks, not a batch of independent sequences.

Runtime state containers:

```text
agg_state_refs = [None for each block]
dpt_state_refs = [dict() for each block]
dpt_layer_set = set(model.agg_regator.intermediate_layer_idx)
```

Depending on `--streaming_state`, these references hold either live tensors or offload-pointer payloads.

### 6.2 Embedder: Image Tokens per Block

For each block:

1. Materialize the block payload.
2. Move flattened RGB to device.
3. Reshape:

   ```text
   (B, S, H * W, C) -> (B, S, C, H, W)
   ```

4. Call:

   ```text
   model.agg_regator.prepare(rgb)
   ```

5. Store the returned aggregator state with `store_agg_state(...)`.

The aggregator `prepare(...)` step normalizes images, runs the patch embedder, adds camera/register tokens, and builds position metadata.

Stored aggregator state fields include:

```text
tokens
pos
B, S, P, C
```

Probe stages:

```text
embedder.begin
embedder_block_00.done
embedder_block_01.done
...
embedder.done
```

### 6.3 Aggregator Layers

The backend loops over:

```text
range(model.agg_regator.aa_block_num)
```

For each aggregator index `j`:

1. Determine whether this layer should produce DPT intermediate features:

   ```text
   collect_intermediate_layers(model, j)
   ```

2. For every block:
   - materialize `agg_state_refs[b]`,
   - create `temp_output = {}` only if DPT intermediate output is needed,
   - call:

     ```text
     model.agg_regator.forward_layer(index=j, output=temp_output, ...)
     ```

   - store the updated aggregator state,
   - persist any intermediate DPT state into `dpt_state_refs[b]`.

`Aggregator.forward_layer(...)` runs the normal frame/global alternating attention layer with `enable_ttt=False`; the backend triggers TTT separately afterward.

Probe stages:

```text
aggregator.begin
aggregator_layer_00.begin
aggregator_layer_00.done
...
aggregator.done
```

The `aggregator_layer_XX.done` probe records whether that layer used TTT.

### 6.4 TTT / GCM Application

After the normal forward layer for index `j`, the backend checks:

```text
(model.agg_regator.frame_use_ttt or model.agg_regator.global_use_ttt)
and j in model.agg_regator.ttt_layer_idx
```

If true, it calls:

```text
apply_ttt(model, agg_state_refs, dpt_state_refs, args, j, dpt_layer_set)
```

The default config enables global TTT on layers:

```text
14, 17, 20, 23
```

`apply_ttt(...)` performs a cross-block fast-weight update:

1. Build a gradient-only TTT order from `model.ttt_order[0:1]`.
2. Force `use_cached=False` and `cache_last=False`.
3. For each block:
   - materialize its aggregator state,
   - call `model.agg_regator.ttt_gradient(...)`,
   - accumulate `w0_grad`, `w1_grad`, `w2_grad` across blocks.
4. Call `model.agg_regator.ttt_update(...)` once to produce shared updated fast weights:

   ```text
   shared_w0, shared_w1, shared_w2
   ```

5. Build an apply-only TTT order from `model.ttt_order[-1:]`.
6. For each block:
   - materialize its aggregator state,
   - materialize any DPT intermediate state for this layer,
   - call `model.agg_regator.ttt_apply(..., w0=shared_w0, w1=shared_w1, w2=shared_w2)`,
   - store the updated aggregator state,
   - persist updated DPT intermediate state if needed.

The persistent checkpoint parameters are not updated. The updated fast weights are temporary tensors used to modify the current inference token states.

### 6.5 Decoder Setup

After all aggregator layers:

1. For each block, set:

   ```text
   dpt_state[-1] = dpt_state[model.agg_regator.depth - 1]
   ```

2. Remove the final aggregator state payload.
3. Keep only DPT intermediate features for decoder heads.

This frees aggregator states before decoding.

### 6.6 Camera, XYZ, and Depth Decoders

For each block:

1. Materialize the block.
2. Materialize `dpt_state_refs[b]`.
3. Move feature tensors to device.
4. Reshape block RGB back to:

   ```text
   (B, S, C, H, W)
   ```

5. Run:

   ```text
   cam_maps = model.cam_decoder(rgb_feats)
   xyz_map, xyz_cnf = model.xyz_decoder(rgb_feats, images=rgb, patch_start_idx=...)
   dpt_map, dpt_cnf = model.dpt_decoder(rgb_feats, images=rgb, patch_start_idx=...)
   ```

6. Crop predictions to `H, W`.
7. Store CPU output:

   ```text
   cam_map: first 9 camera parameters
   xyz_map: flattened dense XYZ map
   dpt_map: flattened dense depth map
   xyz_cnf: flattened XYZ confidence
   dpt_cnf: flattened depth confidence
   scale: optional, if camera output has more than 9 channels
   ```

If `--offload_outputs 1`, each decoded block output is serialized to:

```text
<offload_root>/outputs/<block_index>.pt
```

Probe stages:

```text
decoder.begin
decoder_block_00.begin
decoder_block_00.done
...
decoder.done
```

## 7. Post-Processing and Alignment

Call site:

```text
processed, output, batches, indices, visualize = post_process(
  output,
  batches,
  indices,
  args,
  n_blocks_loop=args.get("n_blocks_loop", 0),
  alignment="sim3_wet",
  use_xyz_align=args.use_xyz_align,
  recorder=recorder,
)
```

`post_process(...)` converts raw per-block predictions into ordered, globally aligned outputs.

Returned values:

```text
processed: final c2w, intrinsics, optional depths
raw: normal-block raw outputs after loop blocks are removed
batches: normal blocks after loop blocks are removed
indices: normal block index ranges after loop blocks are removed
visualize: point-cloud/mask buffers for saving
```

### 7.1 Prediction Preparation

The inner `prepare(batch, output)` function converts one decoded raw block into NumPy arrays used for alignment.

It always starts with:

```text
dpt = output.dpt_map[0].cpu()
cnf = output.dpt_cnf[0].cpu()
```

If `use_xyz_align == 0`:

1. Decode camera parameters into `w2c` and `ixt`.
2. Generate rays with `get_rays(...)`.
3. Compute points with:

   ```text
   xyz = ray_o + ray_d * dpt
   ```

If `use_xyz_align != 0`:

1. Use `output.xyz_map[0]`.
2. Use `output.xyz_cnf[0]`.

If `output.scale` exists, both XYZ and depth are scaled.

### 7.2 Submap Construction

The backend builds a `map_processor`:

```text
build_map_processor("sim3_wet", max_align_points_per_frame=args.max_align_points_per_frame)
```

For each normal sequential block, it:

1. Materializes the block and raw output.
2. Calls `prepare(...)`.
3. Adds the predicted points/depth/confidence/mask/source indices as a submap.

Loop blocks are excluded from this normal-submap pass by:

```text
n_blocks = len(batches) - n_blocks_loop
```

when loop blocks exist.

Probe stage:

```text
post_process.begin
```

### 7.3 Adjacent Block Alignment

The backend aligns adjacent normal block pairs:

```text
norm_track = map_processor.align_submaps_parallel(max_workers=n_workers)
```

Worker selection:

```text
if pgo_workers > 0:
  n_workers = min(n_pairs, pgo_workers)
else:
  n_workers = min(n_pairs, os.cpu_count() or 4)
```

`norm_track` stores the sequential pairwise transforms used for global pose assembly.

### 7.4 Optional Loop Closure Optimization

If loop blocks were appended by `load_data(...)`, `post_process(...)` processes them after adjacent alignment.

For each loop block:

1. Read the loop block index tuple:

   ```text
   block0, _, block2, _ = indices[block1]
   ```

2. Build two temporary processors:
   - one for block0 and loop block,
   - one for block2 and loop block.
3. Compute two relative Sim3 transforms.
4. Combine those transforms into a loop constraint.
5. Append:

   ```text
   (block0, block2, (scale, rotation, translation))
   ```

Then:

```text
optimizer = build_sim3_loop_optimizer()
ini_track = optimizer.sequential_to_absolute_poses(norm_track)
opt_track = optimizer.optimize(norm_track, loop_track)
res_track = accumulate_transform(opt_track)
vis_track = optimizer.sequential_to_absolute_poses(opt_track)
visualize_loop(ini_track, vis_track, loop_track, get_runtime_root(args))
```

Finally, loop-only blocks are removed:

```text
raw = raw[:n_blocks]
batches = batches[:n_blocks]
indices = indices[:n_blocks]
```

If no loop blocks exist, `res_track` is taken directly from the map processor's global submap poses.

### 7.5 Overlap-Aware Final Frame Selection

Adjacent blocks contain duplicated overlap frames. The backend first computes overlap index maps between each pair:

```text
overlap_prev_inds
overlap_curr_inds
```

Then, for each block, it chooses a subset `midx` of frame positions to keep.

The selection alternates by block parity:

- even non-final blocks remove the second half of their forward overlap and, when applicable, the first half of their backward overlap;
- odd blocks remove the first half of their backward overlap and, when applicable, the second half of their forward overlap;
- the last even block removes only the backward overlap first half;
- a single block keeps every frame.

The kept source indices are appended into `i_src`, then the final output is sorted by `i_src` at the end.

### 7.6 Camera and Depth Assembly

For every kept frame:

1. Decode raw camera parameters:

   ```text
   w2c, ixt = decode_camera_params(...)
   ```

2. Apply optional predicted scale to translation.
3. Convert to global camera-to-world:

   ```text
   c2w = res_track[j][None] @ affine_inverse(affine_padding(w2c[midx]))
   ```

4. Append:

   ```text
   processed.output.c2w
   processed.output.ixt
   processed.output.dpt_map  # only when save_dpt is true
   ```

After all blocks:

```text
order = np.argsort(i_src, kind="stable")
```

Every `processed.output` array is concatenated and reordered by original source-frame order.

### 7.7 Point-Cloud Visualization Buffers

If `save_xyz` is true, the backend also builds `visualize` buffers.

For each frame in a block:

1. Convert XYZ to homogeneous coordinates.
2. Transform block-local points with `res_track[j]`.
3. Build a confidence mask:

   ```text
   cnf > mean(cnf) * confidence_xyz_threshold
   ```

4. Keep masked XYZ and RGB.
5. Add kept-frame data to world buffers.
6. Add all block-frame data to block buffers.

The buffers are later consumed by `save_results(...)` to write masks and PLY point clouds.

Probe stage:

```text
post_process.done
```

## 8. Runtime Metrics and Result Saving

After post-processing, `main()` builds a runtime summary:

```text
time: wall-clock inference/post-process duration
memory: max CUDA memory allocated, in GB
n_frames: final output frame count
fps: n_frames / time
offload: enabled offload modes and offload root
runtime_dir
runtime_json
```

Then it calls:

```text
save_results(processed, batches, visualize, runtime, args, recorder=recorder)
```

`save_results(...)` writes:

Always:

```text
<runtime_dir>/runtime.json
<result_dir>/intri.yml
<result_dir>/extri.yml
<result_dir>/mat.txt
```

When `save_dpt` is true:

```text
<result_dir>/depths/<frame>.exr
```

When `save_xyz` is true:

```text
<result_dir>/points/blocks/<block>.ply
<result_dir>/points/whole.ply
<result_dir>/points/whole_indices.npy
<result_dir>/masks/<frame>.png
```

`mat.txt` contains one flattened `4x4` camera-to-world matrix per final output frame.

Probe stages:

```text
save_results.begin
save_results.done
```

## 9. Offload and Memory Behavior

Offload helpers live in `scal3r/utils/offload_utils.py`.

Runtime roots:

```text
runtime_root = args.runtime_dir
offload_root = args.offload_dir or <runtime_dir>/offload
```

Offload categories:

```text
batches/
outputs/
agg_state/
dpt_state/
```

Behavior by flag:

- `offload_batches`: serialized prepared image blocks.
- `offload_outputs`: serialized decoded block outputs.
- `streaming_state`: serialized aggregator and DPT intermediate states.
- `cleanup_offload`: removes the offload root in `finally`.

`materialize_payload(...)` is the key read path. It accepts either a live object or a small offload pointer and returns the actual payload. The backend uses this pattern throughout `forward(...)` and `post_process(...)`, so most stages are written to work with either in-memory or disk-backed runtime state.

When any runtime offload mode is enabled, `should_release_runtime_state(args)` becomes true and the backend calls `release_memory(args.device)` after major per-block operations.

## 10. Probe and Early-Stop Stages

With `--probe_dir`, `StageRecorder` writes JSON files recording stage name, elapsed time, process memory, optional CUDA memory, and stage-specific metadata.

Useful stage names:

```text
process.begin
load_model.begin
load_model.done
collect_images.begin
collect_images.done
preprocess_images.begin
preprocess_images.done
build_blocks.begin
build_blocks.done
loop_detection.begin
loop_detection.done
load_data.done
embedder.begin
embedder_block_00.done
embedder.done
aggregator.begin
aggregator_layer_00.begin
aggregator_layer_00.done
aggregator.done
decoder.begin
decoder_block_00.begin
decoder_block_00.done
decoder.done
post_process.begin
post_process.done
save_results.begin
save_results.done
```

Exact per-block and per-layer stage names depend on the number of blocks and aggregator layers.

To stop after a stage:

```bash
python -m scal3r.pipelines.backend \
  --input_dir /path/to/images \
  --probe_dir data/result/custom/debug/runtime/probes \
  --stop_after_stage build_blocks.done
```

`maybe_stop_after(...)` records `stop_after_stage` and raises `StopAfterStage`, which is caught in `main()`. The `finally` block still runs `cleanup_offload_root(args)`.

## 11. Compact Execution Chain

```text
python -m scal3r.pipelines.backend
└── main()
    ├── parse_args()
    ├── resolve_release_path(...)
    ├── StageRecorder(...)
    ├── build_sampler_from_config(...)
    │   ├── load_config(...)
    │   ├── Scal3R(...)
    │   ├── torch.load(checkpoint)
    │   ├── load_state_dict(strict=True)
    │   └── sampler.eval()
    ├── load_data(...)
    │   ├── collect_image_paths(...)
    │   ├── apply_frame_range_to_sorted_paths(...)
    │   ├── load_and_preprocess_images(...)
    │   │   ├── load_rgb_from_path(...)
    │   │   ├── build_dummy_ixt(...)
    │   │   ├── apply_base_transforms(...)
    │   │   ├── determine_target_size(...)
    │   │   └── finalize_transforms(...)
    │   ├── build_image_only_block(...)
    │   └── optional detect_loops(...) + build_loop_batches(...)
    ├── forward(...)
    │   ├── agg_regator.prepare(...)
    │   ├── for each aggregator layer:
    │   │   ├── agg_regator.forward_layer(...)
    │   │   └── optional apply_ttt(...)
    │   │       ├── agg_regator.ttt_gradient(...)
    │   │       ├── agg_regator.ttt_update(...)
    │   │       └── agg_regator.ttt_apply(...)
    │   ├── cam_decoder(...)
    │   ├── xyz_decoder(...)
    │   └── dpt_decoder(...)
    ├── post_process(...)
    │   ├── prepare(...)
    │   ├── build_map_processor(...)
    │   ├── add_submap(...)
    │   ├── align_submaps_parallel(...)
    │   ├── optional loop optimizer
    │   ├── overlap-aware frame selection
    │   └── ordered processed outputs
    ├── save_results(...)
    │   ├── runtime.json
    │   ├── intri.yml / extri.yml
    │   ├── mat.txt
    │   ├── optional depths/*.exr
    │   └── optional points + masks
    └── cleanup_offload_root(...)
```

## 12. Data Objects Through the Pipeline

```text
input_dir images
  -> image_paths: list[str]
  -> sequence: list[DotDict(rgb, msk, ixt, path)]
  -> batches: list[DotDict or offload refs]
  -> agg_state_refs: per-block token states
  -> dpt_state_refs: per-block intermediate features
  -> output/raw: per-block cam/xyz/depth/confidence predictions
  -> processed.output:
       c2w
       ixt
       optional dpt_map
  -> visualize:
       block_xyz/block_rgb/block_msk
       world_xyz/world_rgb/world_msk
  -> files under result_dir and runtime_dir
```

