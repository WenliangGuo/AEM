# Data Preparation

Self-contained scripts that regenerate the two kinds of effect-model supervision
shipped in `data/<dataset>/`:

- **Object detections** — `detection/<...>.json` (VLM object descriptions + GroundingDINO boxes)
- **Symbolic scene graphs** — `scene_graph_json/<...>.json` and the CLIP-feature `scene_graph_npy/<...>.npy`

These are only needed if you want to **re-create** the supervision. For training
and evaluation the pre-computed files already in the repo are enough — you can
ignore this folder entirely.

Outputs are written straight into the dataset layout described in the top-level
`README.md`, so a full run reproduces the provided `detection/`,
`scene_graph_json/`, `scene_graph_npy/` (and, for EgoPER, `effect_frames/*.npz`).

---

## Setup

```bash
# from the AEM/ directory
export OPENAI_API_KEY=sk-...        # required by every GPT step (or pass --api_key)
```

Every script is run from the `AEM/` directory and takes `--data_root`
(default `data/egoper` or `data/captaincook4d`). No paths or keys are hardcoded.

Dependencies by step (imported lazily, so a step only needs its own deps):

| Step                                                   | Needs                                                                                                     |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| effect descriptions, object descriptions, scene graphs | `openai`                                                                                                |
| effect-frame selection / encoding, feature extraction  | `torch`, `open_clip` (EVA02-L-14-336), `opencv-python`, `Pillow`, GPU                             |
| object detection                                       | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) on the `PYTHONPATH` + SwinB weights, GPU |

`common.py` holds the shared helpers (OpenAI client, EVA-CLIP loader,
effect-frame scoring, GroundingDINO wrapper, scene-graph → feature packing).

---

## EgoPER pipeline (`egoper/`)

Required inputs in `data/egoper/`: `annotation.json`, `gpt4o_action_objects.json`,
and `<task>/{training,validation}.txt`. Raw 1 fps frames
(`<task>/frames_1fps/<vid>_frames/*.jpg`) live wherever you extracted EgoPER —
pass that root as `--frames_root`. `<task>` ∈ {coffee, oatmeal, pinwheels, quesadilla, tea}.

```bash
# 1. effect descriptions  ->  effect_desc_egoper.json
python data_preparation/egoper/gen_effect_desc.py --data_root data/egoper

# 2. select effect frames ->  effect_frames/<task>/<vid>/<seg>/*.jpg      (GPU)
python data_preparation/egoper/select_effect_frames.py --task coffee \
    --data_root data/egoper --frames_root /path/to/EgoPER

# 3. pack effect frames   ->  effect_frames/effect_frames_<task>.npz
python data_preparation/egoper/encode_effect_frames.py --task coffee --data_root data/egoper

# --- detection branch ---
# 4. object descriptions  ->  detection/<task>_descriptions.json
python data_preparation/egoper/describe_objects.py --task coffee --data_root data/egoper
# 5. GroundingDINO boxes  ->  detection/<task>_detection.json               (GPU + DINO)
python data_preparation/egoper/detect_objects.py --task coffee --data_root data/egoper \
    --config_path GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --weights_path GroundingDINO/weights/groundingdino_swinb_cogcoor.pth

# --- scene-graph branch ---
# 6. GPT-4o scene graphs  ->  scene_graph_json/<task>_gpt4o.json
python data_preparation/egoper/generate_scene_graph.py --task coffee --data_root data/egoper
# 7. CLIP features        ->  scene_graph_npy/<task>_gpt4o.npy              (GPU)
python data_preparation/egoper/extract_graph_features.py --task coffee --data_root data/egoper
```

Steps 4–5 and 6–7 are independent branches; both only need the effect frames
from step 2 (and, for the loader, the `.npz` from step 3). Use `--task all`
for the frame-selection/encoding steps to cover every task at once.

---

## CaptainCook4D pipeline (`captaincook4d/`)

Required inputs in `data/captaincook4d/`: `non_error_samples_processed.json`,
`gpt4o_action_objects.json`, `activity_step_collection.json`,
`id_activity_mapping.json`, `step_id_mapping.json`. Raw 1 fps 360p frames
(`<video_id>_360p/*.png`) live wherever you extracted CaptainCook4D — pass that
root as `--frames_root`. There is no `.npz` step: the loader reads the effect
jpgs directly.

```bash
# 1. effect descriptions  ->  effect_desc_ccp4d.json
python data_preparation/captaincook4d/gen_effect_desc.py --data_root data/captaincook4d

# 2. select effect frames ->  effect_frames/<vid>/<seg>/*.jpg              (GPU)
python data_preparation/captaincook4d/select_effect_frames.py \
    --data_root data/captaincook4d --frames_root /path/to/frames_1fps

# --- detection branch ---
# 3. object descriptions  ->  detection/descriptions_ccp4d.json
python data_preparation/captaincook4d/describe_objects.py --data_root data/captaincook4d
# 4. GroundingDINO boxes  ->  detection/openset_detection_ccp4d.json       (GPU + DINO)
python data_preparation/captaincook4d/detect_objects.py --data_root data/captaincook4d \
    --config_path GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --weights_path GroundingDINO/weights/groundingdino_swinb_cogcoor.pth

# --- scene-graph branch ---
# 5. GPT scene graphs     ->  scene_graph_json/scene_graph.json
python data_preparation/captaincook4d/generate_scene_graph.py --data_root data/captaincook4d
# 6. CLIP features        ->  scene_graph_npy/scene_graph.npy              (GPU)
python data_preparation/captaincook4d/extract_graph_features.py --data_root data/captaincook4d
```

---

## Notes

- **Models.** EgoPER scene graphs use `gpt-4o` (the canonical `_gpt4o` files);
  all object/effect descriptions and CaptainCook4D scene graphs use
  `gpt-4o-mini`. Override with `--model` on any GPT step.
- **Resumable.** The GPT steps skip videos already present in their output file,
  so a rerun continues where it stopped.
- **`img_path` convention.** EgoPER stores `img_path` as the `.npz` key
  (`<task>/<vid>/<seg>/<frame>.jpg`); CaptainCook4D stores it relative to
  `data_root` (`effect_frames/<vid>/<seg>/<frame>.jpg`). Both match what the
  dataset loaders expect.
