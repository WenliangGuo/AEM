"""EgoPER step 7 -- turn scene-graph json into CLIP-feature .npy (the effect-model input).

Each node name / relation sentence / attribute sentence is encoded with the
EVA02-CLIP text encoder and packed into the tensor dict the effect model
expects (see common.build_segment_graph_features for the exact schema).

Input : <data_root>/scene_graph_json/<task>_gpt4o.json   (from step 6)
        <data_root>/gpt4o_action_objects.json
Output: <data_root>/scene_graph_npy/<task>_gpt4o.npy      ({vid: [segment_feature_dict, ...]})

Run from the AEM/ directory (needs a GPU + open_clip):
    python data_preparation/egoper/extract_graph_features.py --task coffee --data_root data/egoper
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import build_segment_graph_features, load_eva_clip, read_json

EGOPER_TASKS = ["coffee", "oatmeal", "pinwheels", "quesadilla", "tea"]


def segment_features(seg, task, action_objects, clip):
    if seg["action"] == "BG":
        return build_segment_graph_features(seg, None, None, clip)
    objs = action_objects[task][seg["action"]]
    return build_segment_graph_features(seg, objs["objectA"], objs["objectB"], clip)


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="Extract EgoPER scene-graph CLIP features")
    parser.add_argument("--task", required=True, choices=EGOPER_TASKS)
    parser.add_argument("--data_root", default="data/egoper")
    args = parser.parse_args()

    task = args.task
    scene_graphs = read_json(os.path.join(args.data_root, "scene_graph_json", f"{task}_gpt4o.json"))
    action_objects = read_json(os.path.join(args.data_root, "gpt4o_action_objects.json"))
    clip = load_eva_clip()

    result = {}
    for vid, seg_list in scene_graphs.items():
        print(f"Processing {task}/{vid}")
        result[vid] = [segment_features(seg, task, action_objects, clip) for seg in seg_list]

    out_path = os.path.join(args.data_root, "scene_graph_npy", f"{task}_gpt4o.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, result)
    print(f"Saved scene-graph features -> {out_path}")


if __name__ == "__main__":
    main()
