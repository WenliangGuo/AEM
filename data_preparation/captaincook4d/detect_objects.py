"""CaptainCook4D step 4 -- ground the two objects with GroundingDINO (detection, part 2).

Inputs :
    <data_root>/detection/descriptions_ccp4d.json           (from step 3)
    <data_root>/id_activity_mapping.json
    <data_root>/effect_frames/<vid>/<seg>/<frame>.jpg        (from step 2)
    GroundingDINO config + weights
Output :
    <data_root>/detection/openset_detection_ccp4d.json
        each segment gains "BBox of OBJECT A/B" and "img_path" (relative to
        <data_root>, e.g. effect_frames/<vid>/<seg>/<frame>.jpg).

Run from the AEM/ directory (needs a GPU + GroundingDINO):
    python data_preparation/captaincook4d/detect_objects.py --data_root data/captaincook4d \
        --config_path GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py \
        --weights_path GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import GroundingDINO, augmented_prompt, find_ccp4d_effect_frame, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Ground CaptainCook4D objects with GroundingDINO")
    parser.add_argument("--data_root", default="data/captaincook4d")
    parser.add_argument("--config_path",
                        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--weights_path",
                        default="GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    args = parser.parse_args()

    descriptions = read_json(os.path.join(args.data_root, "detection", "descriptions_ccp4d.json"))
    id_activity = read_json(os.path.join(args.data_root, "id_activity_mapping.json"))
    out_path = os.path.join(args.data_root, "detection", "openset_detection_ccp4d.json")

    detector = GroundingDINO(args.config_path, args.weights_path)

    for activity_id, recipe in id_activity.items():
        if activity_id not in descriptions:
            continue
        for vid, seg_list in descriptions[activity_id].items():
            print(f"Processing {recipe}/{vid}")
            effect_vid_dir = os.path.join(args.data_root, "effect_frames", vid)
            for seg in seg_list:
                has_desc = "Description of OBJECT A" in seg and "Description of OBJECT B" in seg
                seg_folder = frame = None
                if has_desc and os.path.isdir(effect_vid_dir):
                    seg_folder, frame = find_ccp4d_effect_frame(effect_vid_dir, seg["time_stamp"][0])

                if seg_folder is not None:
                    frame_path = os.path.join(effect_vid_dir, seg_folder, frame)
                    prompt_a = augmented_prompt(seg["OBJECT A"], seg["Description of OBJECT A"])
                    prompt_b = augmented_prompt(seg["OBJECT B"], seg["Description of OBJECT B"])
                    bbox_a, prob_a = detector.detect(frame_path, prompt_a)
                    bbox_b, prob_b = detector.detect(frame_path, prompt_b)
                    seg["img_path"] = os.path.join("effect_frames", vid, seg_folder, frame)
                else:
                    bbox_a = bbox_b = prob_a = prob_b = None
                    seg["img_path"] = None

                seg["BBox of OBJECT A"] = {"bbox": bbox_a, "prob": prob_a}
                seg["BBox of OBJECT B"] = {"bbox": bbox_b, "prob": prob_b}
            write_json(out_path, descriptions, indent=4)

    print(f"Saved detection -> {out_path}")


if __name__ == "__main__":
    main()
