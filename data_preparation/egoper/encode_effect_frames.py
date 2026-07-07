"""EgoPER step 3 -- pack the selected effect frames into a compressed .npz.

The EgoPER dataset loader reads effect frames from this archive (keyed by the
``img_path`` stored in the detection json) instead of touching the many small
jpg files, which is much faster. CaptainCook4D does NOT need this step -- its
loader reads the jpgs directly.

Input : <data_root>/effect_frames/<task>/<video_id>/<seg>/<frame>.jpg   (from step 2)
Output: <data_root>/effect_frames/effect_frames_<task>.npz
        (keys are <task>/<video_id>/<seg>/<frame>.jpg -> RGB uint8 array)

Run from the AEM/ directory:
    python data_preparation/egoper/encode_effect_frames.py --task coffee --data_root data/egoper
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EGOPER_TASKS = ["coffee", "oatmeal", "pinwheels", "quesadilla", "tea"]


def collect_frames(task_dir, task):
    """Walk <task_dir> and yield (npz_key, absolute_path) for every frame.

    Handles both the normal ``<seg>/<frame>.jpg`` layout and the nested
    ``Measure 1/<seg>/<frame>.jpg`` layout produced by the 'Measure 1/2 cup
    water' action.
    """
    for vid in sorted(os.listdir(task_dir)):
        vid_dir = os.path.join(task_dir, vid)
        if not (vid.startswith(task) and os.path.isdir(vid_dir)):
            continue
        for seg in sorted(os.listdir(vid_dir)):
            seg_dir = os.path.join(vid_dir, seg)
            entries = sorted(os.listdir(seg_dir))
            first = os.path.join(seg_dir, entries[0])
            if os.path.isdir(first):  # nested 'Measure 1/<sub>/<frame>' case
                sub = entries[0]
                for frame in sorted(os.listdir(first)):
                    key = f"{task}/{vid}/{seg}/{sub}/{frame}"
                    yield key, os.path.join(first, frame)
            else:
                for frame in entries:
                    key = f"{task}/{vid}/{seg}/{frame}"
                    yield key, os.path.join(seg_dir, frame)


def encode_task(data_root, task):
    import cv2 as cv
    import numpy as np

    task_dir = os.path.join(data_root, "effect_frames", task)
    if not os.path.isdir(task_dir):
        print(f"  missing {task_dir}, skipping {task}")
        return
    frame_dict = {}
    for key, path in collect_frames(task_dir, task):
        img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
        frame_dict[key] = img
    out_path = os.path.join(data_root, "effect_frames", f"effect_frames_{task}.npz")
    np.savez(out_path, **frame_dict)
    print(f"[{task}] wrote {len(frame_dict)} frames -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Encode EgoPER effect frames into .npz")
    parser.add_argument("--task", required=True, choices=EGOPER_TASKS + ["all"])
    parser.add_argument("--data_root", default="data/egoper")
    args = parser.parse_args()

    tasks = EGOPER_TASKS if args.task == "all" else [args.task]
    for task in tasks:
        encode_task(args.data_root, task)


if __name__ == "__main__":
    main()
