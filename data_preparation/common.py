"""Shared helpers for the AEM data-preparation scripts.

All heavy dependencies (torch, open_clip, openai, cv2, PIL) are imported lazily
inside the functions that need them, so a script only requires the packages for
the step it actually runs (e.g. running ``--help`` never imports torch).
"""

import base64
import json
import math
import os


# --------------------------------------------------------------------------- #
#  OpenAI                                                                      #
# --------------------------------------------------------------------------- #
def get_openai_client(api_key=None):
    """Return an OpenAI client. The key comes from ``--api_key`` or the
    ``OPENAI_API_KEY`` environment variable. Fails loudly if neither is set."""
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment "
            "variable or pass --api_key."
        )
    return OpenAI(api_key=key)


def encode_image_b64(image_path):
    """Base64-encode an image for the OpenAI vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_response(text):
    """Parse a JSON object out of a chat completion, tolerating ```json fences.
    Returns a dict, or None if nothing parseable is found."""
    t = (text or "").strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0]
    try:
        return json.loads(t.strip())
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
#  EVA02-CLIP (used for effect-frame scoring and scene-graph text features)    #
# --------------------------------------------------------------------------- #
def load_eva_clip(device=None):
    """Load the EVA02-L-14-336 CLIP model used throughout the paper.

    Returns ``(model, preprocess, tokenizer, device)``.
    """
    import torch
    import open_clip

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="EVA02-L-14-336",
        pretrained="merged2b_s6b_b61k",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("EVA02-L-14-336")
    model.eval()
    return model, preprocess, tokenizer, device


# --------------------------------------------------------------------------- #
#  Effect-frame selection helpers                                             #
# --------------------------------------------------------------------------- #
def laplacian_var(gray):
    """Variance of the Laplacian -- a simple image-sharpness proxy."""
    import cv2

    return cv2.Laplacian(gray, cv2.CV_64F).var()


def clarity_score(v, p5, p95):
    """Normalise a sharpness value into [0, 1] using per-video percentiles."""
    import numpy as np

    return np.clip((v - p5) / (p95 - p5 + 1e-9), 0, 1)


def effect_prompt(sentences):
    """Wrap effect-description sentences into CLIP text prompts."""
    prompts = []
    for p in sentences:
        p = p.replace(",", "").replace(".", "").strip()
        prompts.append(f"An image showing that {p}")
    return prompts


def clip_content_score(clip_model, preprocess, tokenizer, device, img_bgr, texts):
    """Max cosine similarity between a BGR image and a list of text prompts."""
    import cv2
    import torch
    from PIL import Image

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    text_tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor)
        txt_feat = clip_model.encode_text(text_tokens)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    score = (img_feat @ txt_feat.T).squeeze().cpu().numpy()
    return float(score.max())


# --------------------------------------------------------------------------- #
#  Effect-frame lookup (shared by describe / detect / scene-graph steps)      #
# --------------------------------------------------------------------------- #
def find_egoper_effect_frame(effect_vid_dir, action, start_time):
    """Locate the effect frame for one EgoPER segment.

    ``effect_vid_dir`` = ``<data_root>/effect_frames/<task>/<video_id>``.
    Segment folders are named ``<action>_<ceil(start)>_<floor(end)>``.
    Returns ``(seg_folder, frame_file)`` (both relative to ``effect_vid_dir``),
    or ``(None, None)`` if nothing matches.

    The ``Measure 1/2 cup water`` action contains a ``/`` and therefore lands in
    a nested ``Measure 1/...`` folder -- handled as a special case, matching how
    the frames were written.
    """
    if action == "Measure 1/2 cup water":
        folder_1 = os.path.join(effect_vid_dir, "Measure 1")
        sub = sorted(os.listdir(folder_1))[0]
        frame = sorted(os.listdir(os.path.join(folder_1, sub)))[0]
        return os.path.join("Measure 1", sub), frame

    prefix = f"{action}_{int(math.ceil(start_time))}"
    for name in sorted(os.listdir(effect_vid_dir)):
        if name.startswith(prefix):
            frame = sorted(os.listdir(os.path.join(effect_vid_dir, name)))[0]
            return name, frame
    return None, None


def find_ccp4d_effect_frame(effect_vid_dir, start_time):
    """Locate the effect frame for one CaptainCook4D segment.

    ``effect_vid_dir`` = ``<data_root>/effect_frames/<video_id>``.
    Segment folders are named ``<ceil(start)>_<floor(end)>``.
    Returns ``(seg_folder, frame_file)`` or ``(None, None)``.
    """
    prefix = str(int(math.ceil(start_time)))
    for name in sorted(os.listdir(effect_vid_dir)):
        if name.startswith(prefix):
            frame = sorted(os.listdir(os.path.join(effect_vid_dir, name)))[0]
            return name, frame
    return None, None


def ccp4d_remapped_idx2action(id_activity_mapping, activity_step_collection, step_id_mapping):
    """Build ``{activity_id: {remapped_step_id(int): step_text}}`` for CaptainCook4D.

    CaptainCook4D labels use per-recipe *remapped* step ids; this inverts the
    remapping so a segment label can be turned back into its step text.
    """
    remapped = {}
    for activity_id, activity_name in id_activity_mapping.items():
        remapped[activity_id] = {}
        idx_remap = step_id_mapping["step_id_mapping"][activity_id]
        for original_id, step_text in activity_step_collection[activity_name].items():
            remapped[activity_id][idx_remap[original_id]] = step_text
    return remapped


# --------------------------------------------------------------------------- #
#  Open-set detection (GroundingDINO)                                         #
# --------------------------------------------------------------------------- #
class GroundingDINO:
    """Thin GroundingDINO wrapper returning the single highest-confidence box.

    ``GroundingDINO`` must be importable (its repo on the PYTHONPATH) and the
    SwinB config/weights available. Imports are lazy so the other steps do not
    need GroundingDINO installed.
    """

    def __init__(self, config_path, weights_path):
        from GroundingDINO.groundingdino.util.inference import load_model
        self.model = load_model(config_path, weights_path)

    def detect(self, image_path, prompt, box_threshold=0.1, text_threshold=0.25):
        """Return ``(bbox_cxcywh_norm, prob)`` or ``(None, None)``."""
        import torch
        from GroundingDINO.groundingdino.util.inference import load_image, predict
        try:
            _, image = load_image(image_path)
            boxes, logits, _ = predict(model=self.model, image=image, caption=prompt,
                                       box_threshold=box_threshold, text_threshold=text_threshold)
            if boxes.shape[0] == 0 or logits.shape[0] == 0:
                return None, None
            best = torch.argmax(logits, dim=0).item()
            return boxes[best].tolist(), logits[best].item()
        except Exception as e:
            print(f"  detection failed on {image_path}: {e}")
            return None, None


def augmented_prompt(obj, text):
    """Insert ' which' right after the object name so DINO grounds the object,
    not the trailing descriptive clause."""
    pos = text.find(obj)
    if pos == -1:
        return text
    end = pos + len(obj)
    return text[:end] + " which" + text[end:]


# --------------------------------------------------------------------------- #
#  Scene-graph -> CLIP features (shared by both extract_graph_features scripts) #
# --------------------------------------------------------------------------- #
def clip_encode_text(clip, texts):
    """Encode a list of strings with the EVA02-CLIP text encoder -> numpy array."""
    import torch

    model, _, tokenizer, device = clip
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        return model.encode_text(tokens).cpu().numpy()


def empty_graph_features(action):
    """Placeholder feature dict for background / invalid segments."""
    import torch

    return {
        "action": action,
        "graph_node_name": [],
        "graph_x": torch.zeros(1, 768).to(torch.float32),
        "graph_x_type": torch.zeros(1).to(torch.long),
        "graph_edge": torch.zeros(2, 1).to(torch.long),
        "spatial_graph_node": torch.zeros(1).to(torch.long),
        "state_graph_node": torch.zeros(1).to(torch.long),
        "relation_feature": [],
        "attribute_feature": [],
        "valid": False,
    }


def build_segment_graph_features(seg, obj1, obj2, clip):
    """Convert one scene-graph segment into the effect-model feature dict.

    ``obj1``/``obj2`` are the two interaction-object names (ignored for BG).
    Node types: 0=object, 1=relation, 2=attribute. ``spatial_graph_node`` /
    ``state_graph_node`` collect the nodes touching an interaction object.
    Returns an invalid/empty dict when the graph is missing the pieces the
    effect model needs.
    """
    import numpy as np
    import torch

    action = seg["action"]
    if action == "BG":
        return empty_graph_features(action)

    rel_triplets = seg["relation"]
    attribute_list = seg["attribute"]
    if len(rel_triplets) == 0 or len(attribute_list) == 0:
        return empty_graph_features(action)

    graph_node_name, graph_x_type, graph_edge = [], [], []
    spatial_graph_node, state_graph_node = [], []

    # relations: subject -> relation -> object
    for triplet in rel_triplets:
        subject, relation, obj = triplet["subject"], triplet["relation"], triplet["object"]
        for node, node_type in [(subject, 0), (obj, 0), (relation, 1)]:
            if node not in graph_node_name:
                graph_node_name.append(node)
                graph_x_type.append(node_type)
        s_idx = graph_node_name.index(subject)
        r_idx = graph_node_name.index(relation)
        o_idx = graph_node_name.index(obj)
        graph_edge.extend([[s_idx, r_idx], [r_idx, o_idx]])
        if subject in (obj1, obj2):
            for idx in (s_idx, o_idx, r_idx):
                if idx not in spatial_graph_node:
                    spatial_graph_node.append(idx)

    # attributes: subject -> attribute
    for attr_info in attribute_list:
        subject = attr_info["subject"]
        if subject not in graph_node_name:
            graph_node_name.append(subject)
            graph_x_type.append(0)
        s_idx = graph_node_name.index(subject)
        for attr in attr_info["attribute"]:
            if attr not in graph_node_name:
                graph_node_name.append(attr)
                graph_x_type.append(2)
            a_idx = graph_node_name.index(attr)
            graph_edge.append([s_idx, a_idx])
            if subject in (obj1, obj2):
                for idx in (s_idx, a_idx):
                    if idx not in state_graph_node:
                        state_graph_node.append(idx)

    if len(state_graph_node) == 0 or len(spatial_graph_node) == 0:
        return empty_graph_features(action)

    try:
        graph_x = [clip_encode_text(clip, [node]) for node in graph_node_name]
        rel_features = clip_encode_text(clip, seg["relation_sentence"])
        attr_features = clip_encode_text(clip, seg["attribute_sentence"])
        graph_x = torch.from_numpy(np.array(graph_x, dtype=np.float32).squeeze(1))
        graph_x_type = torch.from_numpy(np.array(graph_x_type, dtype=np.int32))
        graph_edge = torch.from_numpy(np.transpose(np.array(graph_edge, dtype=np.int32)))
        spatial_graph_node = torch.from_numpy(np.array(spatial_graph_node, dtype=np.int32))
        state_graph_node = torch.from_numpy(np.array(state_graph_node, dtype=np.int32))
    except Exception as e:
        print(f"  feature extraction failed for '{action}': {e}")
        return empty_graph_features(action)

    return {
        "action": action,
        "graph_node_name": graph_node_name,
        "graph_x": graph_x,                      # (num_nodes, 768)
        "graph_x_type": graph_x_type,            # (num_nodes,)
        "graph_edge": graph_edge,                # (2, num_edges)
        "spatial_graph_node": spatial_graph_node,
        "state_graph_node": state_graph_node,
        "relation_feature": rel_features,
        "attribute_feature": attr_features,
        "valid": True,
    }


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, obj, indent=2):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
