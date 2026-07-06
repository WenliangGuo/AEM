"""CLIP encoding helpers for CaptainCook4D.

Self-contained so it does not touch the EgoPER pipeline. Unlike EgoPER (which reads
pre-extracted effect frames from an .npz), CaptainCook4D reads the object/global crops
directly from the frame images referenced by the open-set detection json (``img_path``).
"""
import os

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert


def encode_image(clip, image, device):
    clip.eval().to(device)
    with torch.no_grad():
        return clip.encode_image(image.to(device)).cpu()


def convert_bbox(bbox, size=(720, 1280), in_fmt="cxcywh", out_fmt="xyxy"):
    h, w = size
    cxcy = bbox * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=cxcy, in_fmt=in_fmt, out_fmt=out_fmt).tolist()
    return list(map(int, xyxy))


def crop_image(image, bbox):
    h, w, _ = image.shape
    x1, y1, x2, y2 = convert_bbox(bbox, size=(h, w))
    return image[y1:y2, x1:x2]


def _read_rgb(path):
    image = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    return image


def get_global_vis_embedding(clip_model, info, device, img_path):
    clip, _, preprocess = clip_model
    image = _read_rgb(img_path)
    img_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    return encode_image(clip, img_tensor, device)


def get_obj_vis_embedding(clip_model, info, device, img_path):
    clip, _, preprocess = clip_model
    image = _read_rgb(img_path)
    a = crop_image(image, torch.tensor(info["BBox of OBJECT A"]["bbox"]).float())
    b = crop_image(image, torch.tensor(info["BBox of OBJECT B"]["bbox"]).float())
    a = preprocess(Image.fromarray(a)).unsqueeze(0).to(device)
    b = preprocess(Image.fromarray(b)).unsqueeze(0).to(device)
    return encode_image(clip, torch.cat([a, b], dim=0), device)


def encode_relative_pos(bbox_a, bbox_b):
    a = torch.tensor(list(map(float, convert_bbox(torch.tensor(bbox_a)))))
    b = torch.tensor(list(map(float, convert_bbox(torch.tensor(bbox_b)))))
    return torch.stack((a, b), dim=0)  # (2, 4)


def process_visual_info(clip_model, visual_info, device, root_dir=''):
    """Return (bbox_relative_pos, object_vis_feature, global_vis_feature, frame_contrast_mask),
    one entry per segment. Background / incompletely-annotated segments get zero features.

    ``img_path`` in the detection json is relative to ``root_dir`` (e.g.
    ``effect_frames/<recipe_video>/<seg>/<frame>.jpg``); it is resolved here."""
    bbox_relative_pos, object_vis_feature, global_vis_feature, frame_contrast_mask = [], [], [], []
    for info in visual_info:
        is_bg = info["action"] == "BG"
        ok_a = info["BBox of OBJECT A"]['bbox'] is not None
        ok_b = info["BBox of OBJECT B"]['bbox'] is not None
        ok_loc = "Location Relationship" in info
        if is_bg or not ok_a or not ok_b or not ok_loc:
            bbox_relative_pos.append(torch.zeros(2, 4))
            object_vis_feature.append(torch.zeros(2, 768))
            global_vis_feature.append(torch.zeros(1, 768))
            frame_contrast_mask.append(False)
            continue
        try:
            img_path = os.path.join(root_dir, info["img_path"])
            bbox_relative_pos.append(encode_relative_pos(info["BBox of OBJECT A"]["bbox"],
                                                         info["BBox of OBJECT B"]["bbox"]))
            object_vis_feature.append(get_obj_vis_embedding(clip_model, info, device, img_path))
            global_vis_feature.append(get_global_vis_embedding(clip_model, info, device, img_path))
            frame_contrast_mask.append(True)
        except Exception as e:
            print(f"Warning: failed to process visual info for {info.get('img_path')}: {e}")
            bbox_relative_pos.append(torch.zeros(2, 4))
            object_vis_feature.append(torch.zeros(2, 768))
            global_vis_feature.append(torch.zeros(1, 768))
            frame_contrast_mask.append(False)
    return bbox_relative_pos, object_vis_feature, global_vis_feature, frame_contrast_mask


def action_tokenize_ccp4d(clip_model, task, activity_step_collection, step_id_mapping, device):
    """Build CLIP text tokens for a recipe's action steps (index 0 = background frame).

    Returns (action_tokens, num_classes). Labels in the annotations are the mapped step
    ids in ``step_id_mapping['step_id_mapping'][task]`` (background = 0).
    """
    _, tokenizer, _ = clip_model
    id2action = {0: {"action": "background frame", "activity": None}}
    for activity, steps in activity_step_collection.items():
        for org_step_id, step_name in steps.items():
            if org_step_id not in step_id_mapping["step_id_mapping"][task]:
                continue
            mapped_id = step_id_mapping["step_id_mapping"][task][org_step_id]
            id2action[mapped_id] = {"action": step_name.lower(), "activity": activity}

    text_input = []
    for i in range(len(id2action)):
        action, activity = id2action[i]["action"], id2action[i]["activity"]
        text_input.append("background frame" if activity is None
                          else f"An image showing {action} for making {activity}")
    action_tokens = tokenizer(text_input).long().to(device)
    return action_tokens, len(id2action)
