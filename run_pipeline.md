# `scal3r/run.py` Running Pipeline

This document traces what happens when running:

```bash
python -m scal3r.run --input_dir /path/to/images
```

The entrypoint is a thin orchestration wrapper. It resolves paths and command-line overrides, writes a runtime manifest and backend command plan, then launches the real inference backend as a subprocess:

```text
scal3r/run.py
  -> scal3r.pipelines.inference.run_inference(...)
    -> python -m scal3r.pipelines.backend ...
      -> load model
      -> load and chunk images
      -> forward model over chunks
      -> align chunks and post-process predictions
      -> save camera, depth, point-cloud, and runtime outputs
```

## 1. CLI Entrypoint

File: `scal3r/run.py`

`main()` parses arguments with `build_parser()`. The most important arguments are:

- `--config`: release config, defaulting to `configs/models/scal3r.yaml`.
- `--input_dir`: required image folder.
- `--output_dir`: optional result directory. If omitted, the default is `data/result/custom/<tag>`.
- `--runtime_dir`: optional runtime directory. If omitted, the default is `<output_dir>/runtime`.
- `--checkpoint`, `--device`: optional model and device overrides.
- `--start_frame`, `--end_frame`, `--interval`, `--max_images`: frame selection controls.
- `--block_size`, `--overlap_size`: long-sequence chunking controls.
- `--use_loop`, `--use_xyz_align`, `--pgo_workers`, `--max_align_points_per_frame`: alignment and pose-graph controls.
- `--save_dpt`, `--save_xyz`: final artifact toggles.
- `--streaming_state`, `--offload_batches`, `--offload_outputs`, `--offload_dir`, `--cleanup_offload`: memory/offload controls.
- `--probe_dir`, `--stop_after_stage`: stage-by-stage debug controls.
- `--dry_run`: writes the command plan but does not launch the backend.

The wrapper loads the config with `load_config(args.config)`, resolves release-relative paths with `resolve_release_path(...)`, and constructs an `InferenceRequest`.

Default path behavior comes from `scal3r.engine.path`:

- release root: repository root inferred from the installed `scal3r` package path.
- default output root: `data/result/custom`.
- default output directory: `data/result/custom/<tag>`.
- default runtime directory: `<output_dir>/runtime`.

## 2. Config Merge

Files:

- `configs/models/scal3r.yaml`
- `configs/base.yaml`
- `scal3r/engine/config.py`

`configs/models/scal3r.yaml` includes:

```yaml
configs:
    - configs/base.yaml
```

`load_config(...)` recursively loads listed base configs, merges dictionaries, and stores the resolved path in `__config_path__`.

The default merged runtime settings include:

- image patterns: `*.png,*.jpg,*.jpeg,*.bmp`
- preprocessing workers: `32`
- block size: `60`
- overlap size: `30`
- loop closure enabled: `use_loop: 1`
- loop checkpoint: `data/checkpoints/dino_salad.ckpt`
- depth and point export enabled: `save_dpt: 1`, `save_xyz: 1`
- model checkpoint: `data/checkpoints/scal3r.pt`
- preprocessing shape controls: `proc_max_size: 518`, `proc_align_size: 14`, `center_crop: true`

Command-line overrides in `run.py` replace these config values before the backend command is built.

## 3. Wrapper Planning Layer

Files:

- `scal3r/pipelines/inference.py`
- `scal3r/dataloaders/datasets/image_folder_dataset.py`

`run_inference(config, request)` does not run the neural network directly. It prepares a reproducible backend invocation.

The steps are:

1. Read `data` and `model` sections from the merged config.
2. Build an `ImageFolderDataset`.
3. List input images by glob pattern.
4. Sort and deduplicate resolved absolute paths.
5. Apply frame slicing with Python slice semantics:

   ```text
   sorted_paths[start_frame:end_frame:interval]
   ```

   Here `end_frame` is exclusive, and `end_frame < 0` means through the final image.

6. Apply `max_images` after frame slicing, if provided.
7. Create `output_dir` and `runtime_dir`.
8. Write:

   - `<runtime_dir>/image_manifest.txt`
   - `<runtime_dir>/run_plan.json`

9. Build the subprocess command:

   ```text
   <current-python> -m scal3r.pipelines.backend ...
   ```

10. If `dry_run` is false, launch the backend with `subprocess.run(..., check=True, cwd=<release_root>)`.

`run_plan.json` is the best first artifact to inspect when debugging a run. It records the resolved config, input/output paths, image count, frame range, chunking settings, loop/alignment toggles, offload settings, and the exact backend command.

## 4. Backend Startup

File: `scal3r/pipelines/backend.py`

The backend has its own CLI parser. The wrapper passes almost all runtime choices explicitly, so the backend receives a self-contained command.

At startup, `backend.main()`:

1. Parses backend args.
2. Resolves config, result, runtime, checkpoint, loop checkpoint, offload, and probe paths.
3. Creates a `torch.device`.
4. Creates a `StageRecorder`.
5. Logs the runtime config.
6. Enters the main pipeline:

   ```text
   build_sampler_from_config(...)
   load_data(...)
   forward(...)
   post_process(...)
   save_results(...)
   cleanup_offload_root(...)
   ```

If `--probe_dir` is set, `StageRecorder` writes one JSON file per recorded stage plus `latest.json`. If `--stop_after_stage` equals a recorded stage name, `maybe_stop_after(...)` raises `StopAfterStage` and the backend exits after writing the matching probe.

## 5. Model Construction

Files:

- `scal3r/models/scal3r.py`
- `scal3r/utils/vggt/models/aggregator.py`
- `scal3r/utils/vggt/heads/camera_head.py`
- `scal3r/utils/vggt/heads/dpt_head.py`

`build_sampler_from_config(config_path, device, checkpoint_path)`:

1. Loads the merged config again inside the backend process.
2. Extracts `model_cfg.sampler_cfg`.
3. Instantiates `Scal3R`.
4. Resolves the checkpoint path from `--checkpoint` or `model.checkpoint`.
5. Loads the checkpoint with `torch.load(..., map_location="cpu", weights_only=False, mmap=True)`.
6. Moves the model to the selected device.
7. Sets eval mode.
8. Returns the model and dataset preprocessing config.

`Scal3R` contains:

- `agg_regator`: VGGT-style aggregator that prepares image tokens and runs alternating attention layers.
- `cam_decoder`: camera parameter head.
- `xyz_decoder`: dense 3D point head.
- `dpt_decoder`: dense depth head.
- `ttt_order`: test-time-training update/apply order.

The default release config enables global test-time training on aggregator layers `14`, `17`, `20`, and `23`.

## 6. Image Loading and Chunking

Files:

- `scal3r/pipelines/backend.py`
- `scal3r/utils/image_utils.py`

`load_data(dataset_cfg, args, recorder)` performs the backend-side data preparation.

### 6.1 Collect Images

The backend independently collects image paths with `collect_image_paths(args.input_dir, args.image_patterns)`, then applies the same frame slicing:

```text
sorted_paths[start_frame:end_frame:interval]
```

If no images remain, it raises:

```text
No images remain after start_frame/end_frame/interval; widen the range or check input_dir/patterns.
```

Then `max_images` is applied if positive.

### 6.2 Preprocess Images

`load_and_preprocess_images(...)`:

1. Loads the first image.
2. Builds a dummy intrinsic matrix from image size and `focal_ratio`.
3. Applies base transforms:
   - optional `render_ratio`
   - optional `rot90`
4. Determines a common target size from the first processed image:
   - `proc_max_size`
   - `proc_align_size`
5. Resizes and crops images to the target size.
6. Applies the same target size to remaining images, either sequentially or with `parallel_execution(...)`.

Each preprocessed frame stores:

- flattened RGB: `(H * W, 3)`
- flattened mask: `(H * W, 1)`
- intrinsic matrix
- source path

### 6.3 Build Blocks

The sequence is split into overlapping blocks:

```text
n_blocks = ceil((n_samples - overlap_size) / (block_size - overlap_size))
```

If the sequence fits in one block, the backend uses a single block containing all frames.

For each block, `build_image_only_block(...)` creates a `DotDict` with:

- `H`, `W`
- `src_inds`
- `orig_src_inds`
- `msk`
- `meta.rgb`
- `meta.H`, `meta.W`
- `meta.overlap_size`
- `meta.block_size`
- `meta.aspect_ratio`
- `meta.cam_param_type`
- `meta.use_world_coord`

`src_inds` are reordered so the middle frame comes first. `orig_src_inds` preserves the original block order for overlap handling later.

If `--offload_batches 1`, each block is serialized through the offload utilities instead of being kept fully in memory.

### 6.4 Optional Loop Blocks

If `use_loop` is enabled:

1. `detect_loops(...)` runs loop detection using the configured loop checkpoint when available.
2. `build_loop_batches(...)` creates additional loop-closure blocks.
3. These loop blocks are appended after normal sequential blocks.
4. `args.n_blocks_loop` records how many appended loop blocks exist.

## 7. Forward Pass

File: `scal3r/pipelines/backend.py`

`forward(model, batches, args, recorder)` runs inference block by block, but synchronizes some aggregator/TTT work across all blocks.

### 7.1 Shape Assumption

The backend materializes the first block and reads:

- `B`: batch size
- `S`: frames per block
- `H`, `W`: processed image size
- `N`: number of blocks

It asserts `B == 1`. This release path is sequential inference over one image sequence, not batched independent sequences.

### 7.2 Embedder Stage

For each block:

1. Reshape flattened RGB from:

   ```text
   B, S, H*W, C
   ```

   to:

   ```text
   B, S, C, H, W
   ```

2. Move RGB to the target device.
3. Call `model.agg_regator.prepare(rgb)`.
4. Store the aggregator state in memory or disk, depending on streaming/offload settings.

The aggregator prepare step normalizes images, patch-embeds them, adds special tokens, and prepares positional information for later attention.

### 7.3 Aggregator Layers

The backend iterates over `model.agg_regator.aa_block_num`.

For each aggregator layer `j`:

1. Check whether any intermediate outputs are needed for the DPT/camera heads.
2. For every block:
   - materialize the stored aggregator state
   - call `model.agg_regator.forward_layer(index=j, ...)`
   - persist any intermediate decoder features
   - store the updated aggregator state
3. If the layer is a configured TTT layer, run `apply_ttt(...)`.

### 7.4 Test-Time Training

`apply_ttt(...)` is called only when:

```text
(frame_use_ttt or global_use_ttt) and layer_index in ttt_layer_idx
```

The default release config has `global_use_ttt: true` and `ttt_layer_idx: [14, 17, 20, 23]`.

The TTT flow is:

1. For each block, materialize aggregator state and compute TTT gradients.
2. Sum gradients across blocks.
3. Compute shared updated weights with `ttt_update(...)`.
4. For each block, apply the shared update with `ttt_apply(...)`.
5. Persist updated aggregator state and decoder intermediate state.

This is the main cross-block adaptation mechanism during inference.

### 7.5 Decoder Stage

After all aggregator layers:

1. The backend maps the final DPT state key `-1` to the last aggregator depth state.
2. Aggregator states are removed.
3. For each block:
   - materialize block data and decoder features
   - call `model.cam_decoder(...)`
   - call `model.xyz_decoder(...)`
   - call `model.dpt_decoder(...)`
   - crop predictions to `H, W`
   - flatten dense maps back to `(B, S, H*W, C)`
   - move outputs to CPU
   - optionally store block outputs through offload utilities

Each block output may contain:

- `cam_map`
- `xyz_map`
- `dpt_map`
- `xyz_cnf`
- `dpt_cnf`
- optional `scale`

## 8. Post-Processing and Alignment

File: `scal3r/pipelines/backend.py`

`post_process(...)` converts raw per-block predictions into globally aligned camera poses, depth maps, and optional point-cloud data.

The default alignment string is:

```text
sim3_wet
```

### 8.1 Per-Block Prediction Preparation

The inner `prepare(batch, output)` function chooses how to produce 3D points:

- If `use_xyz_align == 0`, it decodes camera parameters, creates rays from predicted depth, and forms XYZ as `ray_o + ray_d * depth`.
- If `use_xyz_align != 0`, it uses the predicted `xyz_map` directly and uses `xyz_cnf`.
- If the model output includes `scale`, it scales XYZ and depth.

### 8.2 Submap Construction

For each normal sequential block:

1. Materialize the block and its raw output.
2. Convert predictions with `prepare(...)`.
3. Add a submap to `map_processor`.

The map processor is built with `build_map_processor(alignment, max_align_points_per_frame=...)`.

### 8.3 Adjacent Block Alignment

Adjacent block pairs are aligned in parallel:

```text
map_processor.align_submaps_parallel(max_workers=n_workers)
```

`pgo_workers > 0` caps the worker count. `pgo_workers == 0` uses an automatic count based on CPU availability.

### 8.4 Optional Loop Closure Optimization

If loop blocks were appended:

1. Each loop block is compared against the two normal blocks it connects.
2. Relative transforms are composed into loop constraints.
3. `build_sim3_loop_optimizer()` builds the optimizer.
4. The optimizer converts sequential transforms to absolute poses.
5. It optimizes with normal and loop constraints.
6. `visualize_loop(...)` writes loop visualization artifacts under the runtime root.
7. Loop-only blocks are removed from `raw`, `batches`, and `indices` before final result assembly.

If there are no loop blocks, the backend uses each submap's global pose directly.

### 8.5 Final Frame Assembly

The backend then iterates over normal blocks and:

1. Decodes each block's camera parameters.
2. Applies optional predicted scale to camera translations.
3. Uses overlap logic to choose which frames from each block survive into final outputs.
4. Appends camera-to-world matrices and intrinsics.
5. Appends depth maps if `save_dpt` is enabled.
6. Builds per-block and global point-cloud visualization buffers if `save_xyz` is enabled.
7. Sorts final outputs by original source frame index.

The returned `processed.output` contains globally ordered:

- `c2w`
- `ixt`
- optional `dpt_map`

## 9. Result Saving

File: `scal3r/utils/result_utils.py`

`save_results(processed, batches, visualize, runtime, args, recorder)` writes final artifacts.

Always written:

- `<runtime_dir>/runtime.json`
- `<result_dir>/intri.yml`
- `<result_dir>/extri.yml`
- `<result_dir>/mat.txt`

When `save_dpt` is enabled:

- `<result_dir>/depths/<frame>.exr`

When `save_xyz` is enabled:

- `<result_dir>/points/blocks/<block>.ply`
- `<result_dir>/points/whole.ply`
- `<result_dir>/points/whole_indices.npy`
- `<result_dir>/masks/<frame>.png`

`mat.txt` stores one raveled camera-to-world `4x4` matrix per output frame.

Before writing camera files, `_collect_camera_results(...)` normalizes pose scale and converts each camera-to-world matrix to world-to-camera form for the EasyVolcap-style camera outputs.

## 10. Runtime and Debug Artifacts

Typical runtime files:

```text
<output_dir>/
  intri.yml
  extri.yml
  mat.txt
  depths/
  masks/
  points/
  runtime/
    image_manifest.txt
    run_plan.json
    runtime.json
```

If `--probe_dir` is used, it also contains stage JSON files such as:

```text
001_process_begin.json
002_load_model_begin.json
...
latest.json
```

Useful stage names for `--stop_after_stage` include:

- `load_model.done`
- `collect_images.done`
- `preprocess_images.done`
- `build_blocks.done`
- `loop_detection.done`
- `load_data.done`
- `embedder.done`
- `aggregator_layer_00.done`, `aggregator_layer_01.done`, ...
- `aggregator.done`
- `decoder_block_00.done`, `decoder_block_01.done`, ...
- `decoder.done`
- `post_process.done`
- `save_results.done`

Exact availability depends on which branch of the run is reached and whether optional features such as loop detection are enabled.

## 11. Compact Call Graph

```text
python -m scal3r.run
└── scal3r.run.main
    ├── build_parser().parse_args()
    ├── load_config(args.config)
    ├── resolve output/runtime/checkpoint paths
    ├── InferenceRequest(...)
    └── run_inference(config, request)
        ├── ImageFolderDataset(...).list_images()
        │   └── apply_frame_range_to_sorted_paths(...)
        ├── ensure_dir(output_dir), ensure_dir(runtime_dir)
        ├── write image_manifest.txt
        ├── write run_plan.json
        └── subprocess.run([python, -m, scal3r.pipelines.backend, ...])
            └── scal3r.pipelines.backend.main
                ├── parse_args()
                ├── StageRecorder(...)
                ├── build_sampler_from_config(...)
                │   ├── load_config(...)
                │   ├── Scal3R(...)
                │   │   ├── Aggregator(...)
                │   │   ├── CameraHead(...)
                │   │   ├── DPTHead(...) for xyz
                │   │   └── DPTHead(...) for depth
                │   ├── load_checkpoint(...)
                │   └── sampler.eval()
                ├── load_data(...)
                │   ├── collect_image_paths(...)
                │   ├── apply_frame_range_to_sorted_paths(...)
                │   ├── load_and_preprocess_images(...)
                │   ├── build_image_only_block(...)
                │   └── optional detect_loops(...) + build_loop_batches(...)
                ├── forward(...)
                │   ├── agg_regator.prepare(...)
                │   ├── agg_regator.forward_layer(...) for each layer/block
                │   ├── optional apply_ttt(...)
                │   ├── cam_decoder(...)
                │   ├── xyz_decoder(...)
                │   └── dpt_decoder(...)
                ├── post_process(...)
                │   ├── build_map_processor(...)
                │   ├── add_submap(...)
                │   ├── align_submaps_parallel(...)
                │   ├── optional loop optimizer
                │   └── assemble ordered c2w/ixt/depth/point outputs
                ├── save_results(...)
                │   ├── write runtime.json
                │   ├── write camera files and mat.txt
                │   ├── optional depth EXR files
                │   └── optional PLY points and masks
                └── cleanup_offload_root(...)
```

## 12. Practical Debug Flow

For a new sequence, start with:

```bash
python -m scal3r.run \
  --input_dir /path/to/images \
  --output_dir data/result/custom/debug_run \
  --dry_run
```

Inspect:

```text
data/result/custom/debug_run/runtime/image_manifest.txt
data/result/custom/debug_run/runtime/run_plan.json
```

Then run with probes:

```bash
python -m scal3r.run \
  --input_dir /path/to/images \
  --output_dir data/result/custom/debug_run \
  --probe_dir data/result/custom/debug_run/runtime/probes
```

To stop after a specific stage:

```bash
python -m scal3r.run \
  --input_dir /path/to/images \
  --output_dir data/result/custom/debug_run \
  --probe_dir data/result/custom/debug_run/runtime/probes \
  --stop_after_stage build_blocks.done
```

This lets you verify input discovery, frame slicing, preprocessing, and block construction before running the full model.

