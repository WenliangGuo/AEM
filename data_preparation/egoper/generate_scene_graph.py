"""EgoPER step 6 -- generate symbolic scene graphs with GPT-4o.

For each action segment we prompt GPT-4o with the effect frame and the two
interaction objects, and get back a structured scene graph (relation triplets,
per-object attributes, and natural-language sentences).

Inputs :
    <data_root>/annotation.json
    <data_root>/gpt4o_action_objects.json
    <data_root>/<task>/training.txt, <task>/validation.txt
    <data_root>/effect_frames/<task>/<vid>/<seg>/<frame>.jpg   (from step 2)
Output :
    <data_root>/scene_graph_json/<task>_gpt4o.json   ({vid: [segment_scene_graph, ...]})

Run from the AEM/ directory:
    python data_preparation/egoper/generate_scene_graph.py --task coffee --data_root data/egoper
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    encode_image_b64,
    find_egoper_effect_frame,
    get_openai_client,
    parse_json_response,
    read_json,
    write_json,
)

EGOPER_TASKS = ["coffee", "oatmeal", "pinwheels", "quesadilla", "tea"]

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


def load_video_list(data_root, task):
    videos = []
    for split in ("training.txt", "validation.txt"):
        with open(os.path.join(data_root, task, split), "r") as f:
            videos += [line.strip() for line in f if line.strip()]
    return set(videos)


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
    parser = argparse.ArgumentParser(description="Generate EgoPER scene graphs with GPT-4o")
    parser.add_argument("--task", required=True, choices=EGOPER_TASKS)
    parser.add_argument("--data_root", default="data/egoper")
    parser.add_argument("--model", default="gpt-4o",
                        help="canonical release uses gpt-4o")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    task = args.task
    client = get_openai_client(args.api_key)
    annot = read_json(os.path.join(args.data_root, "annotation.json"))
    action_objects = read_json(os.path.join(args.data_root, "gpt4o_action_objects.json"))
    idx2action = {v: k for k, v in annot[task]["action2idx"].items()}
    target_videos = load_video_list(args.data_root, task)
    effect_root = os.path.join(args.data_root, "effect_frames", task)

    out_path = os.path.join(args.data_root, "scene_graph_json", f"{task}_gpt4o.json")
    result = read_json(out_path) if os.path.exists(out_path) else {}

    for vid_info in annot[task]["segments"]:
        vid = vid_info["video_id"]
        if vid not in target_videos or vid in result:
            continue
        print(f"Processing {task}/{vid}")

        graphs = []
        for action_idx, time_stamp in zip(vid_info["labels"]["action"], vid_info["labels"]["time_stamp"]):
            action = idx2action[action_idx]
            if action == "BG":
                graphs.append(empty_scene_graph(action, action_idx, time_stamp))
                continue

            objectA = action_objects[task][action]["objectA"]
            objectB = action_objects[task][action]["objectB"]
            seg_folder, frame = find_egoper_effect_frame(
                os.path.join(effect_root, vid), action, time_stamp[0])
            if seg_folder is None:
                graphs.append(empty_scene_graph(action, action_idx, time_stamp))
                continue

            frame_path = os.path.join(effect_root, vid, seg_folder, frame)
            sg = query_scene_graph(client, args.model, frame_path, objectA, objectB)
            if sg is None:
                graphs.append(empty_scene_graph(action, action_idx, time_stamp))
            else:
                sg.update({"action": action, "action_idx": action_idx, "time_stamp": time_stamp})
                graphs.append(sg)

        result[vid] = graphs
        write_json(out_path, result, indent=2)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
