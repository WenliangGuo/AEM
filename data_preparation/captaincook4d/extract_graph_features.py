"""CaptainCook4D step 6 -- turn scene-graph json into CLIP-feature .npy.

Input : <data_root>/scene_graph_json/scene_graph.json     (from step 5)
        <data_root>/gpt4o_action_objects.json
        <data_root>/id_activity_mapping.json
Output: <data_root>/scene_graph_npy/scene_graph.npy        ({activity_id: {vid: [feature, ...]}})

Run from the AEM/ directory (needs a GPU + open_clip):
    python data_preparation/captaincook4d/extract_graph_features.py --data_root data/captaincook4d
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import build_segment_graph_features, load_eva_clip, read_json


def segment_features(seg, recipe, action_objects, clip):
    if seg["action"] == "BG":
        return build_segment_graph_features(seg, None, None, clip)
    objs = action_objects[recipe][seg["action"]]
    return build_segment_graph_features(seg, objs["objectA"], objs["objectB"], clip)


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="Extract CaptainCook4D scene-graph CLIP features")
    parser.add_argument("--data_root", default="data/captaincook4d")
    args = parser.parse_args()

    scene_graphs = read_json(os.path.join(args.data_root, "scene_graph_json", "scene_graph.json"))
    action_objects = read_json(os.path.join(args.data_root, "gpt4o_action_objects.json"))
    id_activity = read_json(os.path.join(args.data_root, "id_activity_mapping.json"))
    clip = load_eva_clip()

    result = {}
    for activity_id, videos in scene_graphs.items():
        recipe = id_activity[activity_id]
        result[activity_id] = {}
        for vid, seg_list in videos.items():
            print(f"Processing {recipe}/{vid}")
            result[activity_id][vid] = [
                segment_features(seg, recipe, action_objects, clip) for seg in seg_list
            ]

    out_path = os.path.join(args.data_root, "scene_graph_npy", "scene_graph.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, result)
    print(f"Saved scene-graph features -> {out_path}")


if __name__ == "__main__":
    main()
