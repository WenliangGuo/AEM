"""EgoPER step 1 -- generate effect descriptions for every action.

For each task/action we ask GPT for three short sentences describing the object
states *after* the action. These sentences are later used as CLIP text prompts
to pick the most informative effect frame (step 2).

Input : <data_root>/annotation.json          (per-task action2idx)
Output: <data_root>/effect_desc_egoper.json   ({task: {action: [sentence, ...]}})

Run from the AEM/ directory:
    python data_preparation/egoper/gen_effect_desc.py --data_root data/egoper
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_openai_client, read_json, write_json


def build_prompt(goal, step):
    return f"""
For the following step in the process of [goal], describe the resulting states of relevant objects after the action is complete. Use 3 concise sentences. Do not use the verb in [step].
[goal]: Make Kimchi Fried Rice
[step]: add ham
Description:
- The diced ham is mixed with the fried rice.
- The ham is on the pan.
- The pan contains ham.

[goal]: Make Pancakes
[step]: pour egg
Description:
- The egg is mixed with the pancake batter.
- The egg is in the mixing bowl.
- The pancake batter contains egg.

[goal]: {goal}
[step]: {step}
"""


def parse_sentences(text):
    cleaned = re.sub(r"\n\s*\n", "\n", text)
    return re.findall(r"- (.+)", cleaned)


def main():
    parser = argparse.ArgumentParser(description="Generate EgoPER effect descriptions")
    parser.add_argument("--data_root", default="data/egoper")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api_key", default=None, help="OpenAI key (else $OPENAI_API_KEY)")
    args = parser.parse_args()

    client = get_openai_client(args.api_key)
    annot = read_json(os.path.join(args.data_root, "annotation.json"))
    out_path = os.path.join(args.data_root, "effect_desc_egoper.json")

    effect_desc = {}
    for task in annot:
        effect_desc[task] = {}
        for step in annot[task]["action2idx"]:
            if step == "BG":
                continue
            print(f"[{task}] {step}")
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": build_prompt(task, step)}],
            )
            effect_desc[task][step] = parse_sentences(resp.choices[0].message.content)
        write_json(out_path, effect_desc, indent=4)

    print(f"Saved effect descriptions to {out_path}")


if __name__ == "__main__":
    main()
