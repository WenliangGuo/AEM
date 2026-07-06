import os
import json
import torch
import random
import numpy as np
import copy
from torch.utils.data import Dataset
from .datasets import register_dataset
from .data_utils import truncate_feats, generate_node_connected
from .clip_encode import process_visual_info

@register_dataset("EgoPER")
class EgoPERdataset(Dataset):
    """EgoPER dataset for egocentric procedural error recognition."""
    
    def __init__(self, is_training, split, clip_model, use_gcn, default_fps, max_seq_len,
                 trunc_thresh, crop_ratio, height, width, num_classes, background_ratio,
                 num_node, task, root_dir):
        """Initialize EgoPER dataset."""
        # Basic configuration
        self.split = split
        self.is_training = is_training
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
        self.background_ratio = background_ratio
        self.bg_idx = 0
        self.use_gcn = use_gcn
        self.cluster_mode = False
        
        # Setup device and paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # I3D feature path
        self.feat_path = os.path.join(root_dir, task, 'features_10fps')
        # DINO feature path
        # self.feat_path = os.path.join(root_dir, task, 'dinov2_features_10fps')
        self.image_path = os.path.join(root_dir, task, 'frames_1fps')
        self.load_pre_data = split in ['training', 'validation']
        
        # Load basic data
        self.data_list = self._load_video_list(root_dir, task)
        self.annotations = self._load_annotations(root_dir, task, clip_model)
        
        # Load GCN data if needed
        if self.use_gcn:
            self._load_gcn_data(root_dir, task, num_node, height, width)
    
    def _load_video_list(self, root_dir: str, task: str) -> list:
        """Load video list from split file."""
        split_file = os.path.join(root_dir, task, f'{self.split}.txt')
        with open(split_file, 'r') as fp:
            lines = fp.readlines()
        return [line.strip('\n') for line in lines]
    
    def _load_annotations(self, root_dir: str, task: str, clip_model) -> dict:
        """Load and process all annotations."""
        # Load base annotations
        with open(os.path.join(root_dir, 'annotation.json'), 'r') as fp:
            all_annot = json.load(fp)
        
        # Load visual and symbolic supervision data
        visual_annot = None
        scene_graph = None
        if self.load_pre_data:
            with open(os.path.join(root_dir, 'detection', f"{task}_detection.json"), 'r') as fp:
                vlm_data = json.load(fp)
                visual_annot = vlm_data[task]
            scene_graph = np.load(
                os.path.join(root_dir, 'scene_graph_npy', f"{task}_gpt4o.npy"),
                allow_pickle=True
            ).item()
            effect_frame_dict = np.load(
                os.path.join(root_dir, 'effect_frames', f"effect_frames_{task}.npz"),
                allow_pickle=False
            )
        
        # Process annotations
        annotations = {}
        annot = all_annot[task]
        
        for i in range(len(annot['segments'])):
            video_id = annot['segments'][i]['video_id']
            
            if video_id not in self.data_list:
                continue
            
            # Extract basic labels
            actions = [int(action) for action in annot['segments'][i]['labels']['action']]
            action_types = [int(action_type) for action_type in annot['segments'][i]['labels']['action_type']]
            
            # Process VLM info if available
            visual_info = None
            if self.load_pre_data and visual_annot and video_id in visual_annot:
                visual_info = visual_annot[video_id]
                
                # Extract CLIP features
                vis_embeddings = process_visual_info(
                    clip_model, visual_info, 
                    effect_frame_dict, self.device
                )
                bbox_relative_pos, object_vis_feature, global_vis_feature, frame_contrast_masks = vis_embeddings
                
                # Update VLM info with features
                for j, info in enumerate(visual_info):
                    info.update({
                        'object_vis_feature': object_vis_feature[j],
                        'bbox_relative_pos': bbox_relative_pos[j],
                        'frame_contrast_mask': frame_contrast_masks[j],
                        'global_vis_feature': global_vis_feature[j]
                    })
            
            # Load I3D features
            i3d_feature = np.load(os.path.join(self.feat_path, f'{video_id}.npy'))
            
            # Store annotation
            annotations[video_id] = {
                "frame_time_stamps": np.array(annot['segments'][i]['labels']['time_stamp']) * self.default_fps,
                "segment_action_labels": np.array(actions),
                "segment_error_labels": np.array(action_types),
                "error_description": annot['segments'][i]['labels']['error_description'],
                "i3d_feature": i3d_feature,
                "visual_info": visual_info,
                "scene_graph": scene_graph[video_id] if (self.load_pre_data and scene_graph and video_id in scene_graph) else None
            }
        
        return annotations
    
    def _load_gcn_data(self, root_dir: str, task: str, num_node: int, height: int, width: int):
        """Load GCN-related data (bboxes, classes, edge maps)."""
        with open(os.path.join(root_dir, 'active_object.json'), 'r') as fp:
            all_active_obj = json.load(fp)
        
        active_obj = all_active_obj[task]
        self.bboxes = {}
        self.bbox_classes = {}
        self.edge_maps = {}
        
        for obj_info in active_obj:
            video_id = obj_info['video_id']
            
            if video_id in self.data_list:
                object_info = obj_info['active_obj']
                bbox_class, bbox, edge_map = generate_node_connected(
                    object_info, num_node, height, width
                )
                self.bboxes[video_id] = bbox
                self.bbox_classes[video_id] = bbox_class
                self.edge_maps[video_id] = edge_map
    
    def __len__(self):
        """Return dataset size."""
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """Get data item by index."""
        video_id = self.data_list[idx]
        annots = self.annotations[video_id]
        
        # Extract basic data
        time_stamps = annots["frame_time_stamps"]
        action_labels = annots["segment_action_labels"]
        action_labels_error = annots["segment_error_labels"]
        feats = annots["i3d_feature"]
        visual_info = copy.deepcopy(annots["visual_info"]) if annots["visual_info"] else None
        scene_graph = copy.deepcopy(annots["scene_graph"]) if annots["scene_graph"] else None
        
        # Filter background segments during training
        if self.is_training and self.background_ratio < 1.0:
            time_stamps, action_labels, action_labels_error, visual_info, scene_graph = \
                self._filter_background_segments(
                    time_stamps, action_labels, action_labels_error, visual_info, scene_graph
                )
        
        # Create data dictionary
        data_dict = {
            'feats': torch.from_numpy(feats).permute(1, 0).float(),
            'segments': torch.from_numpy(time_stamps).float(),
            'labels': torch.from_numpy(action_labels).long(),
            'labels_error': torch.from_numpy(action_labels_error).long(),
            'error_bool': torch.from_numpy(action_labels_error > 0).long(),
            'video_id': str(video_id),
            'fps': self.default_fps,
            'duration': len(feats) / self.default_fps,
            'visual_info': visual_info,
            'scene_graph': scene_graph
        }
        
        # Add GCN data if needed
        if self.use_gcn and video_id in self.bboxes:
            data_dict.update({
                'bbox_class': torch.tensor(self.bbox_classes[video_id]).long(),
                'bbox': torch.tensor(self.bboxes[video_id]).float(),
                'edge_map': torch.tensor(self.edge_maps[video_id]).float()
            })
        
        # Apply truncation during training
        if self.is_training and not self.cluster_mode:
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )
        
        return data_dict
    
    def _filter_background_segments(self, time_stamps, action_labels, action_labels_error, 
                                   visual_info, scene_graph):
        """Filter out some background segments during training."""
        delete_idx = []
        
        for i in range(len(action_labels)):
            if action_labels[i] == self.bg_idx and random.random() > self.background_ratio:
                delete_idx.append(i)
        
        if delete_idx:
            # Filter numpy arrays
            time_stamps = np.delete(time_stamps, delete_idx, 0)
            action_labels = np.delete(action_labels, delete_idx, 0)
            action_labels_error = np.delete(action_labels_error, delete_idx, 0)
            
            # Filter lists (visual info and scene graph)
            if visual_info:
                for k in sorted(delete_idx, reverse=True):
                    del visual_info[k]
            
            if scene_graph:
                for k in sorted(delete_idx, reverse=True):
                    del scene_graph[k]
        
        return time_stamps, action_labels, action_labels_error, visual_info, scene_graph