"""CaptainCook4D step 2 -- select the effect frame(s) for every action segment.

Same F = alpha * CLIP + (1 - alpha) * sharpness scoring as EgoPER, adapted to
CaptainCook4D's recipe/step-id remapping and 360p png frames.

Inputs :
    <data_root>/non_error_samples_processed.json
    <data_root>/effect_desc_ccp4d.json                     (from step 1)
    <data_root>/{activity_step_collection,id_activity_mapping,step_id_mapping}.json
    <frames_root>/<video_id>_360p/*.png                     (raw 1 fps frames)
Output :
    <data_root>/effect_frames/<video_id>/<start>_<end>/<frame>_<F>_<C>_<Q>.jpg

Run from the AEM/ directory (needs a GPU + open_clip):
    python data_preparation/captaincook4d/select_effect_frames.py \
        --data_root data/captaincook4d --frames_root /path/to/frames_1fps
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    ccp4d_remapped_idx2action,
    clarity_score,
    clip_content_score,
    effect_prompt,
    laplacian_var,
    load_eva_clip,
    read_json,
)

MAX_FILENAME = 255


def process_video(clip, recipe, vid, action_list, action_bounds, effect_desc,
                  frame_folder, save_root, alpha, topk):
    import cv2
    import numpy as np

    model, preprocess, tokenizer, device = clip

    frames = sorted(Path(frame_folder).glob("*.png"))
    if not frames:
        print(f"  no frames in {frame_folder}, skipping")
        return
    vols = [laplacian_var(cv2.imread(str(p), 0)) for p in frames]
    p5, p95 = np.percentile(vols, [5, 95])

    for action, bounds in zip(action_list, action_bounds):
        start_time = int(math.ceil(bounds[0]))
        end_time = int(math.floor(bounds[1]))
        rows = []
        for frame_idx in range(max(start_time, 1), min(end_time + 1, len(frames))):
            frame_path = os.path.join(frame_folder, f"{frame_idx:06d}.png")
            img = cv2.imread(frame_path)
            if img is None:
                continue
            prompts = effect_prompt(effect_desc[recipe][action])
            c = clip_content_score(model, preprocess, tokenizer, device, img, prompts)
            q = clarity_score(laplacian_var(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), p5, p95)
            rows.append((c, q, frame_path, img))
        if not rows:
            continue

        c_arr = np.array([r[0] for r in rows])
        q_arr = np.array([r[1] for r in rows])
        c_arr = (c_arr - c_arr.min()) / (c_arr.max() - c_arr.min() + 1e-9)
        q_arr = (q_arr - q_arr.min()) / (q_arr.max() - q_arr.min() + 1e-9)
        f_arr = alpha * c_arr + (1 - alpha) * q_arr
        scored = [(f_arr[i], c_arr[i], q_arr[i], rows[i][2], rows[i][3]) for i in range(len(rows))]
        scored.sort(key=lambda x: x[0], reverse=True)

        seg_folder = os.path.join(save_root, vid, f"{start_time}_{end_time}")
        os.makedirs(seg_folder, exist_ok=True)
        for f, c, q, frame_path, img in scored[:topk]:
            suffix = f"_{f:.2f}_{c:.2f}_{q:.2f}.jpg"
            base = os.path.basename(frame_path)[:-4]
            if len(base + suffix) > MAX_FILENAME:
                base = base[:MAX_FILENAME - len(suffix)]
            cv2.imwrite(os.path.join(seg_folder, base + suffix), img)
            print(f"  [{vid} {start_time}-{end_time}] F={f:.2f} C={c:.2f} Q={q:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Select CaptainCook4D effect frames")
    parser.add_argument("--data_root", default="data/captaincook4d")
    parser.add_argument("--frames_root", required=True,
                        help="folder holding <video_id>_360p/*.png raw 1 fps frames")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()

    annot = read_json(os.path.join(args.data_root, "non_error_samples_processed.json"))
    effect_desc = read_json(os.path.join(args.data_root, "effect_desc_ccp4d.json"))
    id_activity = read_json(os.path.join(args.data_root, "id_activity_mapping.json"))
    activity_steps = read_json(os.path.join(args.data_root, "activity_step_collection.json"))
    step_id_map = read_json(os.path.join(args.data_root, "step_id_mapping.json"))
    idx2action = ccp4d_remapped_idx2action(id_activity, activity_steps, step_id_map)

    clip = load_eva_clip()
    save_root = os.path.join(args.data_root, "effect_frames")

    for activity_id, videos in annot.items():
        if activity_id not in id_activity:
            continue
        recipe = id_activity[activity_id]
        for vid, vid_info in videos.items():
            save_dir = os.path.join(save_root, vid)
            if os.path.exists(save_dir):
                print(f"effect frames for {vid} exist, skipping")
                continue
            print(f"Processing {recipe}/{vid}")

            action_list = [idx2action[activity_id][a] if a != 0 else "BG" for a in vid_info["labels"]]
            action_bounds = [[s / 10.0, e / 10.0] for s, e in vid_info["segments"]]
            for i in range(len(action_list) - 1, -1, -1):
                if action_list[i] == "BG":
                    if i > 0:
                        action_bounds[i - 1][1] = action_bounds[i][1]
                    del action_list[i]
                    del action_bounds[i]

            frame_folder = os.path.join(args.frames_root, vid + "_360p")
            if not os.path.isdir(frame_folder):
                print(f"  missing frame folder {frame_folder}, skipping")
                continue
            process_video(clip, recipe, vid, action_list, action_bounds, effect_desc,
                          frame_folder, save_root, args.alpha, args.topk)


if __name__ == "__main__":
    main()
