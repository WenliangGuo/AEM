"""CaptainCook4D step 1 -- generate effect descriptions for every recipe step.

Same idea as the EgoPER version: three short sentences describing object states
after each step, later used as CLIP prompts for effect-frame selection.

Input : <data_root>/activity_step_collection.json   ({recipe: {step_id: step_text}})
Output: <data_root>/effect_desc_ccp4d.json           ({recipe: {step_text: [sentence, ...]}})

Run from the AEM/ directory:
    python data_preparation/captaincook4d/gen_effect_desc.py --data_root data/captaincook4d
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
    parser = argparse.ArgumentParser(description="Generate CaptainCook4D effect descriptions")
    parser.add_argument("--data_root", default="data/captaincook4d")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    client = get_openai_client(args.api_key)
    activity_steps = read_json(os.path.join(args.data_root, "activity_step_collection.json"))
    out_path = os.path.join(args.data_root, "effect_desc_ccp4d.json")

    effect_desc = {}
    for recipe, steps in activity_steps.items():
        effect_desc[recipe] = {}
        for step in steps.values():
            print(f"[{recipe}] {step}")
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": build_prompt(recipe, step)}],
            )
            effect_desc[recipe][step] = parse_sentences(resp.choices[0].message.content)
        write_json(out_path, effect_desc, indent=4)

    print(f"Saved effect descriptions to {out_path}")


if __name__ == "__main__":
    main()
