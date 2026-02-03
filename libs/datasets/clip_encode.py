import torch
import cv2 as cv
import numpy as np
from torchvision.ops import box_convert
from PIL import Image

def encode_image(clip, image, device):
    clip.eval()
    clip.to(device)

    image = image.to(device)
    with torch.no_grad():
        image_features = clip.encode_image(image)
    return image_features.cpu()

def get_global_vis_embedding(clip_model, vlm_info, effect_frame_dict, device):
    clip, _, preprocess = clip_model
    frame_path = vlm_info["img_path"]
    # image = cv.imread(frame_path)
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = effect_frame_dict[frame_path]
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    return encode_image(clip, img_tensor, device)

def get_obj_vis_embedding(clip_model, vlm_info, effect_frame_dict, device):
    clip, _, preprocess = clip_model
    frame_path = vlm_info["img_path"]
    object_a_bbox = torch.tensor(vlm_info["BBox of OBJECT A"]["bbox"]).float()
    object_b_bbox = torch.tensor(vlm_info["BBox of OBJECT B"]["bbox"]).float()
    # image = cv.imread(frame_path)
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = effect_frame_dict[frame_path]
    object_a = crop_image(image, object_a_bbox)
    object_b = crop_image(image, object_b_bbox)
    if isinstance(object_a, np.ndarray):
        object_a = Image.fromarray(object_a)
    if isinstance(object_b, np.ndarray):
        object_b = Image.fromarray(object_b)
    object_a = preprocess(object_a).unsqueeze(0).to(device)
    object_b = preprocess(object_b).unsqueeze(0).to(device)
    object_tensor = torch.cat([object_a, object_b], dim=0)
    return encode_image(clip, object_tensor, device)

def convert_bbox(bbox, size=(720, 1280), in_fmt="cxcywh", out_fmt="xyxy"):
    h, w = size
    cxcy_bbox = bbox * torch.Tensor([w, h, w, h])
    xyxy_bbox = box_convert(boxes=cxcy_bbox, in_fmt=in_fmt, out_fmt=out_fmt).tolist()
    xyxy_bbox = list(map(int, xyxy_bbox))
    return xyxy_bbox

def crop_resize_image(image, bbox, tar_size):
    h, w, _ = image.shape
    xyxy_bbox = convert_bbox(bbox, size=(h,w))
    object = image[xyxy_bbox[1]:xyxy_bbox[3], xyxy_bbox[0]:xyxy_bbox[2]]
    object = cv.resize(object, (tar_size, tar_size))
    return object

def crop_image(image, bbox):
    h, w, _ = image.shape
    xyxy_bbox = convert_bbox(bbox, size=(h,w))
    object = image[xyxy_bbox[1]:xyxy_bbox[3], xyxy_bbox[0]:xyxy_bbox[2]]
    return object

def encode_relative_pos(bbox_a, bbox_b):
    bbox_a = list(map(float,convert_bbox(torch.tensor(bbox_a))))
    bbox_b = list(map(float,convert_bbox(torch.tensor(bbox_b))))
    bbox_a = torch.tensor(bbox_a)
    bbox_b = torch.tensor(bbox_b)
    bbox = torch.stack((bbox_a, bbox_b), dim=0) # (2, 4)
    return bbox

def process_visual_info(clip_model, visual_info, effect_frame_dict, device):
    bbox_relative_pos = []
    object_vis_feature = []
    global_vis_feature = []
    frame_contrast_mask = []

    for i in range(len(visual_info)):
        is_bg = visual_info[i]["action"] == "BG"
        correct_annot_a = (visual_info[i]["BBox of OBJECT A"]['bbox'] != None)
        correct_annot_b = (visual_info[i]["BBox of OBJECT B"]['bbox'] != None)
        correct_annot_loc = "Location Relationship" in visual_info[i]
        if is_bg or not correct_annot_a or not correct_annot_b or not correct_annot_loc:
            bbox_relative_pos.append(torch.zeros(2, 4))
            object_vis_feature.append(torch.zeros(2, 768))
            global_vis_feature.append(torch.zeros(1, 768))
            frame_contrast_mask.append(False)
        else:
            obj_bbox_a = visual_info[i]["BBox of OBJECT A"]["bbox"]
            obj_bbox_b = visual_info[i]["BBox of OBJECT B"]["bbox"]
            try:
                bbox_relative_pos.append(
                    encode_relative_pos(obj_bbox_a, obj_bbox_b)
                )
                object_vis_feature.append(
                    get_obj_vis_embedding(clip_model, visual_info[i], effect_frame_dict, device)
                )
                global_vis_feature.append(
                    get_global_vis_embedding(clip_model, visual_info[i],effect_frame_dict, device)
                )
                frame_contrast_mask.append(True)
            except Exception as e:
                print(f"Warning: Failed to process VLM info for frame {visual_info[i]['img_path']}: {e}")
                bbox_relative_pos.append(torch.zeros(2, 4))
                object_vis_feature.append(torch.zeros(2, 768))
                global_vis_feature.append(torch.zeros(1, 768))
                frame_contrast_mask.append(False)

    return bbox_relative_pos, object_vis_feature, global_vis_feature, frame_contrast_mask


def action_tokenize_egoper(clip_model, task, annot, device):
    _, tokenizer, _ = clip_model
    action2idx = annot['action2idx']
    idx2action = {v: k for k, v in action2idx.items()}
    text_input = []
    for i in range(len(idx2action)):
        action = idx2action[i].lower() if i != 0 else "background frame"
        text_str = f"An image showing the action of {action} for making {task}"
        text_input.append(text_str)
    action_tokens = tokenizer(text_input).long().to(device)
    return action_tokens