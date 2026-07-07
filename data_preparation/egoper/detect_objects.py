"""EgoPER step 5 -- ground the two objects with GroundingDINO (detection, part 2).

Reads the per-segment object descriptions (step 4), runs open-set detection for
OBJECT A / OBJECT B on the segment's effect frame, and writes the final
detection json used by the effect model.

Inputs :
    <data_root>/detection/<task>_descriptions.json           (from step 4)
    <data_root>/effect_frames/<task>/<vid>/<seg>/<frame>.jpg  (from step 2)
    GroundingDINO config + weights
Output :
    <data_root>/detection/<task>_detection.json
        each segment gains "BBox of OBJECT A/B" (cxcywh, normalised) and
        "img_path" (<task>/<vid>/<seg>/<frame>.jpg -- the key into the .npz).

Run from the AEM/ directory (needs a GPU + GroundingDINO):
    python data_preparation/egoper/detect_objects.py --task coffee --data_root data/egoper \
        --config_path GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py \
        --weights_path GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import GroundingDINO, augmented_prompt, find_egoper_effect_frame, read_json, write_json

EGOPER_TASKS = ["coffee", "oatmeal", "pinwheels", "quesadilla", "tea"]


def main():
    parser = argparse.ArgumentParser(description="Ground EgoPER objects with GroundingDINO")
    parser.add_argument("--task", required=True, choices=EGOPER_TASKS)
    parser.add_argument("--data_root", default="data/egoper")
    parser.add_argument("--config_path",
                        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--weights_path",
                        default="GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    args = parser.parse_args()

    task = args.task
    desc_path = os.path.join(args.data_root, "detection", f"{task}_descriptions.json")
    descriptions = read_json(desc_path)
    effect_root = os.path.join(args.data_root, "effect_frames", task)
    out_path = os.path.join(args.data_root, "detection", f"{task}_detection.json")

    detector = GroundingDINO(args.config_path, args.weights_path)

    for vid, seg_list in descriptions[task].items():
        print(f"Processing {task}/{vid}")
        for seg in seg_list:
            has_desc = "Description of OBJECT A" in seg and "Description of OBJECT B" in seg
            if has_desc:
                seg_folder, frame = find_egoper_effect_frame(
                    os.path.join(effect_root, vid), seg["action"], seg["time_stamp"][0])
            else:
                seg_folder = frame = None

            if seg_folder is not None:
                frame_path = os.path.join(effect_root, vid, seg_folder, frame)
                prompt_a = augmented_prompt(seg["OBJECT A"], seg["Description of OBJECT A"])
                prompt_b = augmented_prompt(seg["OBJECT B"], seg["Description of OBJECT B"])
                bbox_a, prob_a = detector.detect(frame_path, prompt_a)
                bbox_b, prob_b = detector.detect(frame_path, prompt_b)
                seg["img_path"] = f"{task}/{vid}/{seg_folder}/{frame}"
            else:
                bbox_a = bbox_b = prob_a = prob_b = None
                seg["img_path"] = None

            seg["BBox of OBJECT A"] = {"bbox": bbox_a, "prob": prob_a}
            seg["BBox of OBJECT B"] = {"bbox": bbox_b, "prob": prob_b}
        write_json(out_path, descriptions, indent=4)

    print(f"Saved detection -> {out_path}")


if __name__ == "__main__":
    main()
