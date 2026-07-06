import os
import json
import copy

import torch
import numpy as np
from torch.utils.data import Dataset

from .datasets import register_dataset
from .data_utils import truncate_feats
from .clip_encode_ccp4d import process_visual_info


@register_dataset("CaptainCook4D")
class CaptainCook4DDataset(Dataset):
    """CaptainCook4D dataset for procedural-error (mistake) detection.

    Following the one-class setting, each recipe (``task``) is trained on its
    error-free recordings (``split='train'`` -> non_error_samples) and tested on
    its erroneous recordings (``split='test'`` -> error_samples). ``task='0'`` is
    a special pseudo-task that pools all recipes together.

    Expected layout (everything under root_dir, default 'data/captaincook4d')::

        <root_dir>/
            non_error_samples_processed.json
            error_samples_processed.json
            activity_step_collection.json / step_id_mapping.json / id_activity_mapping.json
            detection/openset_detection_ccp4d.json   (VLM object detections; effect model)
            scene_graph_npy/scene_graph.npy          (scene graphs; effect model)
            scene_graph_json/scene_graph.json        (human-readable scene graphs)
            effect_frames/<recipe_video>/<seg>/<frame>.jpg   (effect frames; effect model, training only)
            <features_subdir>_features_10fps/<video_id>_360p.npy   (I3D features)

    ``img_path`` in the detection json is relative to ``root_dir`` and points into
    ``effect_frames/``; it is resolved against ``root_dir`` at load time.
    """

    def __init__(
        self,
        is_training,          # training mode flag
        split,                # 'train' (non-error) or 'test' (error)
        clip_model,           # (clip, tokenizer, preprocess) for VLM feature extraction
        use_gcn,              # kept for a uniform dataset interface (unused here)
        max_seq_len,          # max sequence length during training
        trunc_thresh,         # overlap threshold for truncating an action segment
        crop_ratio,           # e.g. [0.9, 1.0] for random temporal cropping
        task,                 # recipe id (string), '0' = all recipes pooled
        root_dir,             # folder holding all CaptainCook4D data (default data/captaincook4d)
        default_fps=10,
        features_subdir='I3D',
        background_ratio=1.0,  # <1.0 randomly drops that fraction of background segments
        **kwargs,
    ):
        self.feat_root_dir = os.path.join(root_dir, f"{features_subdir}_features_10fps")
        self.data_root = root_dir
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.task = task
        self.background_ratio = background_ratio
        self.bg_idx = 0
        # VLM / scene-graph supervision is only needed for the (training) effect model.
        self.load_pre_data = (split == 'train')
        # For backbone-only pretraining (segmentation loss only) the VLM / scene-graph
        # data is unused; setting CCP4D_SKIP_VLM=1 skips it and speeds up init a lot.
        if os.environ.get('CCP4D_SKIP_VLM') == '1':
            self.load_pre_data = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if split == 'train':
            annotation_path = os.path.join(self.data_root, 'non_error_samples_processed.json')
        elif split == 'test':
            annotation_path = os.path.join(self.data_root, 'error_samples_processed.json')
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(annotation_path, 'r') as f:
            self.annotation_data = json.load(f)[self.task]

        vlm_annot = None
        scene_graph = None
        if self.load_pre_data:
            with open(os.path.join(self.data_root, 'detection', 'openset_detection_ccp4d.json'), 'r') as fp:
                vlm_annot = json.load(fp)
            scene_graph = np.load(
                os.path.join(self.data_root, 'scene_graph_npy', 'scene_graph.npy'), allow_pickle=True
            ).item()

        self.annots = []
        for video_id, video_data in self.annotation_data.items():
            # For the pooled pseudo-task '0', ids are '<recipe>_<video>'; strip the prefix.
            temp_video_id = video_id if self.task != '0' else video_id.split('_', 1)[1]
            feat_path = os.path.join(self.feat_root_dir, f'{temp_video_id}_360p.npy')
            if not os.path.exists(feat_path):
                continue

            activity_id = temp_video_id.split("_")[0]
            visual_info = None
            if self.load_pre_data and vlm_annot and temp_video_id in vlm_annot.get(activity_id, {}):
                visual_info = vlm_annot[activity_id][temp_video_id]
                bbox_relative_pos, object_vis_feature, global_vis_feature, frame_contrast_masks = \
                    process_visual_info(clip_model, visual_info, self.device, root_dir=self.data_root)
                for j, info in enumerate(visual_info):
                    info.update({
                        'object_vis_feature': object_vis_feature[j],
                        'bbox_relative_pos': bbox_relative_pos[j],
                        'frame_contrast_mask': frame_contrast_masks[j],
                        'global_vis_feature': global_vis_feature[j],
                    })

            has_sg = (self.load_pre_data and scene_graph
                      and temp_video_id in scene_graph.get(activity_id, {}))
            self.annots.append({
                'video_id': temp_video_id,
                'feat_path': feat_path,
                'segments': video_data['segments'],
                'labels': video_data['labels'],
                'labels_error': video_data['labels_error'],
                'descriptions': video_data['descriptions'],
                'visual_info': visual_info,
                'scene_graph': scene_graph[activity_id][temp_video_id] if has_sg else None,
            })

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        annot = self.annots[idx]
        feats = np.load(annot['feat_path'])  # (T, 2048)

        data_dict = {
            'video_id': annot['video_id'],
            'feats': torch.from_numpy(feats).transpose(0, 1).float(),
            'segments': torch.tensor(annot['segments']).float(),
            'labels': torch.tensor(annot['labels']).long(),
            'labels_error': torch.tensor(annot['labels_error']).long(),
            'fps': self.default_fps,
            'duration': len(feats) / self.default_fps,
            'visual_info': copy.deepcopy(annot['visual_info']) if annot['visual_info'] else None,
            'scene_graph': copy.deepcopy(annot['scene_graph']) if annot['scene_graph'] else None,
        }

        if self.is_training:
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)
        return data_dict
