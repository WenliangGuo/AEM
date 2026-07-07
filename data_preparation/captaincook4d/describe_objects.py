"""CaptainCook4D step 3 -- describe the two interaction objects per segment (detection, part 1).

Inputs :
    <data_root>/non_error_samples_processed.json
    <data_root>/gpt4o_action_objects.json                  ({recipe: {step_text: {objectA, objectB}}})
    <data_root>/{activity_step_collection,id_activity_mapping,step_id_mapping}.json
    <data_root>/effect_frames/<vid>/<seg>/<frame>.jpg       (from step 2)
Output :
    <data_root>/detection/descriptions_ccp4d.json          ({activity_id: {vid: [segment, ...]}})

Run from the AEM/ directory:
    python data_preparation/captaincook4d/describe_objects.py --data_root data/captaincook4d
"""

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    ccp4d_remapped_idx2action,
    encode_image_b64,
    find_ccp4d_effect_frame,
    get_openai_client,
    read_json,
    write_json,
)

BASE_PROMPT = """
Firstly, use one sentence to describe [OBJECT A] and [OBJECT B] in scenario of making {task}. Secondly, use one sentence to describe their locations. Thirdly, use one sentence to describe their location relationship.

Example (only for showing response format):
[OBJECT A]: coffee beans
[OBJECT B]: bowl
[Description of OBJECT A]: coffee beans are dark brown, uniformly small, and have a matte surface.
[Description of OBJECT B]: bowl is white, round, and smooth, with a moderate depth.
[Location of OBJECT A]: coffee beans are scattered on the table, inside a white bowl, and partially inside a grinder.
[Location of OBJECT B]: bowl is placed on the table, near the center-left, and the grinder is on the right side of the table.
[Location Relationship]: coffee beans are both inside the bowl and the grinder, with some beans scattered between them on the table.
"""


def parse_description(text):
    cleaned = re.sub(r"\n\s*\n", "\n", text)
    matches = re.findall(r"\[(.*?)\]:\s*(.+?)(?=\n\[|$)", cleaned)
    return {title: desc.strip() for title, desc in matches}


def describe_segment(client, model, recipe, objectA, objectB, frame_path):
    prompt = BASE_PROMPT.format(task=recipe) + f"[OBJECT A]: {objectA}\n[OBJECT B]: {objectB}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(frame_path)}"}},
            ],
        }],
    )
    return parse_description(resp.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description="Generate CaptainCook4D object descriptions")
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

    out_path = os.path.join(args.data_root, "detection", "descriptions_ccp4d.json")
    result = read_json(out_path) if os.path.exists(out_path) else {}

    for activity_id, recipe in id_activity.items():
        result.setdefault(activity_id, {})
        for vid, vid_info in annot[activity_id].items():
            if vid in result[activity_id]:
                print(f"descriptions for {vid} exist, skipping")
                continue
            print(f"Processing {recipe}/{vid}")
            effect_vid_dir = os.path.join(args.data_root, "effect_frames", vid)

            segments = []
            for label, (s, e) in zip(vid_info["labels"], vid_info["segments"]):
                time_stamp = [s / 10.0, e / 10.0]
                if label == 0:
                    segments.append({"action": "BG", "action_idx": 0,
                                     "time_stamp": time_stamp, "description": None})
                    continue

                action = idx2action[activity_id][label]
                objectA = action_objects[recipe][action]["objectA"]
                objectB = action_objects[recipe][action]["objectB"]
                seg_folder, frame = (None, None)
                if os.path.isdir(effect_vid_dir):
                    seg_folder, frame = find_ccp4d_effect_frame(effect_vid_dir, time_stamp[0])
                if seg_folder is None:
                    print(f"  no effect frame for '{action}', skipping segment")
                    segments.append({"action": action, "action_idx": label,
                                     "time_stamp": time_stamp, "description": None})
                    continue

                frame_path = os.path.join(effect_vid_dir, seg_folder, frame)
                desc = describe_segment(client, args.model, recipe, objectA, objectB, frame_path)
                desc.update({"action": action, "action_idx": label, "time_stamp": time_stamp})
                segments.append(desc)

            result[activity_id][vid] = segments
            write_json(out_path, result, indent=4)
            print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
