"""EgoPER step 4 -- describe the two interaction objects per segment (detection, part 1).

For each action segment we ask GPT to describe OBJECT A / OBJECT B, their
locations, and their spatial relationship, given the segment's effect frame.
These descriptions drive the open-set detector (step 5) and end up in the
detection json consumed by the effect model.

Inputs :
    <data_root>/annotation.json
    <data_root>/gpt4o_action_objects.json                 ({task: {action: {objectA, objectB}}})
    <data_root>/<task>/training.txt, <task>/validation.txt
    <data_root>/effect_frames/<task>/<vid>/<seg>/<frame>.jpg   (from step 2)
Output :
    <data_root>/detection/<task>_descriptions.json   ({task: {vid: [segment, ...]}})

Run from the AEM/ directory:
    python data_preparation/egoper/describe_objects.py --task coffee --data_root data/egoper
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    encode_image_b64,
    find_egoper_effect_frame,
    get_openai_client,
    read_json,
    write_json,
)

EGOPER_TASKS = ["coffee", "oatmeal", "pinwheels", "quesadilla", "tea"]

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


def load_video_list(data_root, task):
    videos = []
    for split in ("training.txt", "validation.txt"):
        path = os.path.join(data_root, task, split)
        with open(path, "r") as f:
            videos += [line.strip() for line in f if line.strip()]
    return set(videos)


def describe_segment(client, model, task, objectA, objectB, frame_path):
    prompt = BASE_PROMPT.format(task=task) + f"[OBJECT A]: {objectA}\n[OBJECT B]: {objectB}"
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
    parser = argparse.ArgumentParser(description="Generate EgoPER object descriptions")
    parser.add_argument("--task", required=True, choices=EGOPER_TASKS)
    parser.add_argument("--data_root", default="data/egoper")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    task = args.task
    client = get_openai_client(args.api_key)
    annot = read_json(os.path.join(args.data_root, "annotation.json"))
    action_objects = read_json(os.path.join(args.data_root, "gpt4o_action_objects.json"))
    idx2action = {v: k for k, v in annot[task]["action2idx"].items()}
    target_videos = load_video_list(args.data_root, task)
    effect_root = os.path.join(args.data_root, "effect_frames", task)

    out_path = os.path.join(args.data_root, "detection", f"{task}_descriptions.json")
    result = read_json(out_path) if os.path.exists(out_path) else {task: {}}

    for vid_info in annot[task]["segments"]:
        vid = vid_info["video_id"]
        if vid not in target_videos or vid in result[task]:
            continue
        print(f"Processing {task}/{vid}")

        segments = []
        for action_idx, time_stamp in zip(vid_info["labels"]["action"], vid_info["labels"]["time_stamp"]):
            action = idx2action[action_idx]
            if action == "BG":
                segments.append({"action": "BG", "action_idx": action_idx,
                                 "time_stamp": time_stamp, "description": None})
                continue

            objectA = action_objects[task][action]["objectA"]
            objectB = action_objects[task][action]["objectB"]
            seg_folder, frame = find_egoper_effect_frame(
                os.path.join(effect_root, vid), action, time_stamp[0])
            if seg_folder is None:
                print(f"  no effect frame for {action}, skipping segment")
                segments.append({"action": action, "action_idx": action_idx,
                                 "time_stamp": time_stamp, "description": None})
                continue

            frame_path = os.path.join(effect_root, vid, seg_folder, frame)
            desc = describe_segment(client, args.model, task, objectA, objectB, frame_path)
            desc.update({"action": action, "action_idx": action_idx, "time_stamp": time_stamp})
            segments.append(desc)

        result[task][vid] = segments
        write_json(out_path, result, indent=4)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
