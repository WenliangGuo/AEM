"""CaptainCook4D step 5 -- generate symbolic scene graphs with GPT.

Inputs :
    <data_root>/non_error_samples_processed.json
    <data_root>/gpt4o_action_objects.json
    <data_root>/{activity_step_collection,id_activity_mapping,step_id_mapping}.json
    <data_root>/effect_frames/<vid>/<seg>/<frame>.jpg        (from step 2)
Output :
    <data_root>/scene_graph_json/scene_graph.json           ({activity_id: {vid: [graph, ...]}})

Run from the AEM/ directory:
    python data_preparation/captaincook4d/generate_scene_graph.py --data_root data/captaincook4d
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    ccp4d_remapped_idx2action,
    encode_image_b64,
    find_ccp4d_effect_frame,
    get_openai_client,
    parse_json_response,
    read_json,
    write_json,
)

SYSTEM_PROMPT = """
You are an expert in spatial reasoning and visual scene understanding. Given an image and two specified objects, your task is to:
1. Identify the spatial relationship between the two input objects and any other directly related objects in the image.
2. Capture key attributes for both specified objects and any relevant related objects.
3. Generate a structured scene graph that accurately represents the real-world state in the image.
4. Generate natural language sentences to describe each relationship and attribute.

Important Guidelines:
1. Focus on the specified input objects and their direct spatial relationships.
2. Include related objects only if they are directly interacting with the input objects.
3. Describe spatial relations using clear spatial terms: ("above", "below", "on", "under", "to the left of", "to the right of", "next to", "in front of", "behind").
4. Ensure each object's attributes (e.g., shape, color, material) are captured accurately.
5. Generate concise, clear natural language sentences for each relation and attribute.
6. Only describe what you see in the image; do not infer beyond the visual evidence.

Input Format (Image and Text):
Image: (Provided image)
Objects: "object_1", "object_2"

Output Format (JSON - Relations + Attributes + Sentences):
{
  "relation": [
    {"subject": "<object_1>", "relation": "<relation>", "object": "<object_2>"},
    {"subject": "<object_1>", "relation": "<relation>", "object": "<related_object>"},
    {"subject": "<object_2>", "relation": "<relation>", "object": "<related_object>"}
  ],
  "attribute": [
    {"subject": "<object_1>", "attribute": ["<attribute_1>", "<attribute_2>"]},
    {"subject": "<object_2>", "attribute": ["<attribute_1>", "<attribute_2>"]},
    {"subject": "<related_object>", "attribute": ["<attribute_1>", "<attribute_2>"]}
  ],
  "relation_sentence": [
    "<Natural language sentence for object relationships>"
  ],
  "attribute_sentence": [
    "<Natural language sentence for object attributes>"
  ]
}
"""


def empty_scene_graph(action, action_idx, time_stamp):
    return {"action": action, "action_idx": action_idx, "time_stamp": time_stamp,
            "relation": [], "attribute": [], "relation_sentence": [], "attribute_sentence": []}


def query_scene_graph(client, model, frame_path, objectA, objectB):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"{objectA}, {objectB}"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(frame_path)}"}},
            ]},
        ],
    )
    return parse_json_response(resp.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description="Generate CaptainCook4D scene graphs")
    parser.add_argument("--data_root", default="data/captaincook4d")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    client = get_openai_client(args.api_key)
    annot = read_json(os.path.join(args.data_root, "non_error_samples_processed.json"))
    action_objects = read_json(os.path.join(args.data_root, "gpt4o_action_objects.json"))
    id_activity = read_json(os.path.join(args.data_root, "id_activity_mapping.json"))
    activity_steps = read_json(os.path.join(args.data_root, "activity_step_collection.json"))
    step_id_map = read_json(os.path.join(args.data_root, "step_id_mapping.json"))
    idx2action = ccp4d_remapped_idx2action(id_activity, activity_steps, step_id_map)

    out_path = os.path.join(args.data_root, "scene_graph_json", "scene_graph.json")
    result = read_json(out_path) if os.path.exists(out_path) else {}

    for activity_id, recipe in id_activity.items():
        result.setdefault(activity_id, {})
        for vid, vid_info in annot[activity_id].items():
            if vid in result[activity_id]:
                print(f"scene graph for {vid} exists, skipping")
                continue
            print(f"Processing {recipe}/{vid}")
            effect_vid_dir = os.path.join(args.data_root, "effect_frames", vid)

            graphs = []
            for label, (s, e) in zip(vid_info["labels"], vid_info["segments"]):
                time_stamp = [s / 10.0, e / 10.0]
                if label == 0:
                    graphs.append(empty_scene_graph("BG", 0, time_stamp))
                    continue

                action = idx2action[activity_id][label]
                objectA = action_objects[recipe][action]["objectA"]
                objectB = action_objects[recipe][action]["objectB"]
                seg_folder, frame = (None, None)
                if os.path.isdir(effect_vid_dir):
                    seg_folder, frame = find_ccp4d_effect_frame(effect_vid_dir, time_stamp[0])
                if seg_folder is None:
                    graphs.append(empty_scene_graph(action, label, time_stamp))
                    continue

                frame_path = os.path.join(effect_vid_dir, seg_folder, frame)
                sg = query_scene_graph(client, args.model, frame_path, objectA, objectB)
                if sg is None:
                    graphs.append(empty_scene_graph(action, label, time_stamp))
                else:
                    sg.update({"action": action, "action_idx": label, "time_stamp": time_stamp})
                    graphs.append(sg)

            result[activity_id][vid] = graphs
            write_json(out_path, result, indent=4)
            print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
