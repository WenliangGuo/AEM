"""EgoPER step 2 -- select the effect frame(s) for every action segment.

For each 1 fps frame inside an action segment we score:
    F = alpha * CLIP(frame, effect_description) + (1 - alpha) * sharpness
and keep the top-k frames. These are the images fed to the VLM in later steps.

Inputs :
    <data_root>/annotation.json
    <data_root>/effect_desc_egoper.json                    (from step 1)
    <frames_root>/<task>/frames_1fps/<video_id>_frames/*.jpg   (raw 1 fps frames)
Output :
    <data_root>/effect_frames/<task>/<video_id>/<action>_<start>_<end>/<frame>_<F>_<C>_<Q>.jpg

Run from the AEM/ directory (needs a GPU + open_clip):
    python data_preparation/egoper/select_effect_frames.py \
        --task coffee --data_root data/egoper --frames_root /path/to/EgoPER
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    clarity_score,
    clip_content_score,
    effect_prompt,
    laplacian_var,
    load_eva_clip,
    read_json,
)

EGOPER_TASKS = ["coffee", "oatmeal", "pinwheels", "quesadilla", "tea"]


def process_video(clip, task, vid_id, action_list, action_bounds, effect_desc,
                  frame_folder, save_root, alpha, topk):
    import cv2
    import numpy as np

    model, preprocess, tokenizer, device = clip

    frames = sorted(Path(frame_folder).glob("*.jpg"))
    if not frames:
        print(f"  no frames in {frame_folder}, skipping")
        return
    vols = [laplacian_var(cv2.imread(str(p), 0)) for p in frames]
    p5, p95 = np.percentile(vols, [5, 95])

    for action, bounds in zip(action_list, action_bounds):
        start_time = int(math.ceil(bounds[0]))
        end_time = int(math.floor(bounds[1]))
        rows = []
        for frame_idx in range(start_time, min(end_time + 1, len(frames))):
            frame_path = os.path.join(frame_folder, f"{frame_idx:06d}.jpg")
            img = cv2.imread(frame_path)
            if img is None:
                continue
            prompts = effect_prompt(effect_desc[task][action])
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

        seg_folder = os.path.join(save_root, vid_id, f"{action}_{start_time}_{end_time}")
        os.makedirs(seg_folder, exist_ok=True)
        for f, c, q, frame_path, img in scored[:topk]:
            name = os.path.basename(frame_path)[:-4]
            save_path = os.path.join(seg_folder, f"{name}_{f:.2f}_{c:.2f}_{q:.2f}.jpg")
            cv2.imwrite(save_path, img)
            print(f"  [{action} {start_time}-{end_time}] F={f:.2f} C={c:.2f} Q={q:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Select EgoPER effect frames (CLIP + sharpness)")
    parser.add_argument("--task", required=True, choices=EGOPER_TASKS + ["all"])
    parser.add_argument("--data_root", default="data/egoper")
    parser.add_argument("--frames_root", required=True,
                        help="EgoPER root holding <task>/frames_1fps/<vid>_frames/*.jpg")
    parser.add_argument("--alpha", type=float, default=0.7, help="weight on CLIP content score")
    parser.add_argument("--topk", type=int, default=5, help="frames kept per segment")
    args = parser.parse_args()

    annot = read_json(os.path.join(args.data_root, "annotation.json"))
    effect_desc = read_json(os.path.join(args.data_root, "effect_desc_egoper.json"))
    clip = load_eva_clip()

    tasks = EGOPER_TASKS if args.task == "all" else [args.task]
    for task in tasks:
        idx2action = {v: k for k, v in annot[task]["action2idx"].items()}
        save_root = os.path.join(args.data_root, "effect_frames", task)
        for vid_info in annot[task]["segments"]:
            vid_id = vid_info["video_id"]
            if "error" in vid_id:  # effect frames are only used for the (error-free) training videos
                continue
            print(f"Processing {task}/{vid_id}")

            action_list = [idx2action[a] for a in vid_info["labels"]["action"]]
            action_bounds = [list(b) for b in vid_info["labels"]["time_stamp"]]
            # merge each BG segment into the preceding action
            for i in range(len(action_list) - 1, -1, -1):
                if action_list[i] == "BG":
                    if i > 0:
                        action_bounds[i - 1][1] = action_bounds[i][1]
                    del action_list[i]
                    del action_bounds[i]

            frame_folder = os.path.join(args.frames_root, task, "frames_1fps", vid_id + "_frames")
            if not os.path.isdir(frame_folder):
                print(f"  missing frame folder {frame_folder}, skipping")
                continue
            process_video(clip, task, vid_id, action_list, action_bounds, effect_desc,
                          frame_folder, save_root, args.alpha, args.topk)


if __name__ == "__main__":
    main()
