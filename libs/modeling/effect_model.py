import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.data import Data, Batch
from .blocks import Exp_Scale, MLP
from .graph_att_network import GAT


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            return x  # If input length exceeds max_len, skip positional encoding
        return x + self.pe[:, :seq_len, :].to(x.device)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor with learnable positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


class VLMFeatureProcessor:
    """Handles VLM feature extraction and processing."""
    
    @staticmethod
    def extract_vlm_features(video_list: List[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
        """Extract and concatenate VLM features from video list.
        
        Args:
            video_list: List of dictionaries containing video data with 'visual_info' field.
            device: Target device for tensor operations.
            
        Returns:
            Dictionary containing 'object_vis_feature', 'relative_pos_embed', 'global_vis_feature', and 'frame_contrast_mask' tensors.
        """
        all_object_vis = []
        all_relative_pos = []
        all_global_vis = []
        all_contrast_mask = []

        for vid_feat_dict in video_list:
            for visual_info in vid_feat_dict['visual_info']:
                all_object_vis.append(visual_info['object_vis_feature'])
                all_relative_pos.append(visual_info['bbox_relative_pos'])
                all_global_vis.append(visual_info['global_vis_feature'])
                all_contrast_mask.append(visual_info['frame_contrast_mask'])

        return {
            'object_vis_feature': torch.stack(all_object_vis).to(device).float(),
            'relative_pos_embed': torch.stack(all_relative_pos).to(device).float(),
            'global_vis_feature': torch.stack(all_global_vis).to(device).float(),
            'frame_contrast_mask': torch.tensor(all_contrast_mask).to(device)
        }


class SceneGraphProcessor:
    """Processes scene graph data for GNN operations."""
    
    def __init__(self, graph_node_type_embed: nn.Embedding):
        self.graph_node_type_embed = graph_node_type_embed
    
    def process_scene_graphs(self, video_list: List[Dict], device: torch.device) -> Tuple[Batch, torch.Tensor, Dict]:
        """Process scene graphs from video list into batched format.
        
        Args:
            video_list: List of dictionaries containing video data with 'scene_graph' field.
            device: Target device for tensor operations.
            
        Returns:
            Tuple containing batched graph data, graph validity mask, and dictionary of subgraph nodes with 'state_graph_node' and 'spatial_graph_node'.
        """
        data_list = []
        graph_mask = []
        sub_graph_node = {'state_graph_node': [], 'spatial_graph_node': []}
        
        for vid_feat_dict in video_list:
            for graph in vid_feat_dict['scene_graph']:
                # Prepare node features
                x = graph["graph_x"].flatten(1).to(device)
                node_type_embed = self.graph_node_type_embed(graph["graph_x_type"].to(device))
                x = torch.cat([x, node_type_embed], dim=-1)
                
                # Create graph data
                data = Data(x=x, edge_index=graph["graph_edge"].to(device))
                data_list.append(data)
                graph_mask.append(graph["valid"])
                
                # Store subgraph information
                sub_graph_node['state_graph_node'].append(graph["state_graph_node"].to(device).long())
                sub_graph_node['spatial_graph_node'].append(graph["spatial_graph_node"].to(device).long())

        batched_data = Batch.from_data_list(data_list)
        graph_mask = torch.as_tensor(graph_mask).to(device)
        
        return batched_data, graph_mask, sub_graph_node


class ContrastiveLearningModule:
    """Handles contrastive learning operations."""
    
    def __init__(self, criterion: nn.Module, feat_dim: int):
        self.criterion = criterion
        self.feat_dim = feat_dim
    
    def compute_image_text_contrast(self, vis_feat: torch.Tensor, txt_feat: torch.Tensor,
                                   logits_scale: nn.Module, action_label: Optional[torch.Tensor] = None,
                                   mode: str = 'loss') -> torch.Tensor:
        """Compute image-text contrastive similarity and optionally return loss.
        
        Args:
            vis_feat: Visual feature tensor.
            txt_feat: Text feature tensor.
            logits_scale: Module to scale the similarity logits.
            action_label: Optional action labels for supervised contrastive learning (default: None).
            mode: Mode of operation - 'loss' returns contrastive loss, 'logits' returns similarity logits (default: 'loss').
            
        Returns:
            Contrastive loss tensor if mode='loss', or similarity logits tensor if mode='logits'.
        """
        vis_feat = vis_feat.reshape(-1, self.feat_dim)
        txt_feat = txt_feat.reshape(-1, self.feat_dim)
        
        # Normalize features
        norm_vis_feat = vis_feat / torch.norm(vis_feat, dim=-1, keepdim=True)
        norm_txt_feat = txt_feat / torch.norm(txt_feat, dim=-1, keepdim=True)
        img_txt_logits = logits_scale(norm_vis_feat @ norm_txt_feat.t())
        
        if mode == 'logits':
            return img_txt_logits
        
        # Compute contrastive loss
        txt_img_logits = img_txt_logits.t()
        L = img_txt_logits.shape[0]
        device = img_txt_logits.device
        
        if action_label is None:
            gt_labels = torch.arange(L).to(device)
            img_txt_loss = self.criterion(img_txt_logits, gt_labels)
            txt_img_loss = self.criterion(txt_img_logits, gt_labels)
            return (img_txt_loss + txt_img_loss) / 2
        else:
            gt_labels = action_label.to(device)
            return self.criterion(img_txt_logits, gt_labels)


class FeatureEnhancer(nn.Module):
    """Processes segment features using transformer and enhances them with effect tokens."""

    def __init__(self, num_actions: int, effect_transformer: TransformerEncoder, 
                 state_effect_proj: nn.Module, relation_effect_proj: nn.Module, 
                 enhanced_segment_proj: nn.Module, fpn_vis_input_dim: int = 512, 
                 embed_dim: int = 512, pos_encoding_type: str = 'sinusoidal', 
                 max_len: int = 5000):
        super().__init__()
        self.effect_transformer = effect_transformer
        self.state_effect_proj = state_effect_proj
        self.relation_effect_proj = relation_effect_proj
        self.enhanced_segment_proj = enhanced_segment_proj
        self.CLS_token = nn.Parameter(torch.zeros(1, 1, fpn_vis_input_dim))
        nn.init.trunc_normal_(self.CLS_token, std=0.02)
        self.action_tokens = nn.Parameter(torch.zeros(num_actions - 1, embed_dim))
        nn.init.trunc_normal_(self.action_tokens, std=0.02)
        
        # Initialize positional encoding
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(fpn_vis_input_dim, max_len)
        elif pos_encoding_type == 'learnable':
            self.pos_encoding = LearnablePositionalEncoding(fpn_vis_input_dim, max_len)
        else:
            self.pos_encoding = None
        if isinstance(self.pos_encoding, LearnablePositionalEncoding):
            nn.init.trunc_normal_(self.pos_encoding.pe, std=0.02)

    def enhance_feature(self, feature_segments: List[torch.Tensor], 
                        effect_model_mask: torch.Tensor, 
                        device: torch.device) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Process segment features through transformer and generate effect tokens.
        
        Args:
            feature_segments: List of feature tensors for each segment.
            effect_model_mask: Boolean mask indicating which segments to enhance.
            device: Target device for tensor operations.
            
        Returns:
            Tuple containing enhanced segment features list, state effect tokens, and relation effect tokens.
        """
        effect_segments = [feature_segments[i] for i in range(len(feature_segments)) 
                          if effect_model_mask[i]]
        if not effect_segments:
            return feature_segments, None, None
        
        # Get effect tokens through transformer
        padded_segments = pad_sequence(effect_segments, batch_first=True)
        b, _, _ = padded_segments.shape
        cls_tokens = self.CLS_token.expand(b, -1, -1).to(device)
        padded_segments = torch.cat([cls_tokens, padded_segments], dim=1)

        # Apply positional encoding and transformer
        if self.pos_encoding is not None:
            padded_segments = self.pos_encoding(padded_segments)
        
        lengths = torch.tensor([feat.shape[0] for feat in effect_segments], device=device)
        max_length = padded_segments.size(1)
        memory_mask = torch.arange(max_length, device=device).expand(
            len(lengths), max_length) >= (lengths + 1).unsqueeze(1)
        
        transformed = self.effect_transformer(padded_segments, src_key_padding_mask=memory_mask)
        effect_tokens = transformed[:, 0, :]
        
        # Generate state and relation effect tokens
        state_effect_token = self.state_effect_proj(effect_tokens)
        relation_effect_token = self.relation_effect_proj(effect_tokens)
        
        # Enhance segments with effect tokens
        segment_features = transformed[:, 1:, :]  # Remove CLS token
        seq_len = segment_features.size(1)
        
        # Concatenate effect tokens to each segment feature
        state_expanded = state_effect_token.unsqueeze(1).expand(-1, seq_len, -1)
        relation_expanded = relation_effect_token.unsqueeze(1).expand(-1, seq_len, -1)
        enhanced = torch.cat([segment_features, state_expanded, relation_expanded], dim=-1)
        enhanced = self.enhanced_segment_proj(enhanced)
        
        # Extract segments back to original lengths and reconstruct full list
        enhanced_effect_segments = [enhanced[i, :lengths[i], :] for i in range(len(lengths))]
        enhanced_iter = iter(enhanced_effect_segments)
        enhanced_segments = [
            next(enhanced_iter) if effect_model_mask[i] else feature_segments[i]
            for i in range(len(feature_segments))
        ]
        
        return enhanced_segments, state_effect_token, relation_effect_token


class Effect_Model(nn.Module):
    """
    Effect Model for learning object state and spatial relation representations.
    
    This model processes visual segments and scene graphs to learn representations of
    object states and spatial relations through contrastive learning objectives.
    """
    
    def __init__(self, clip_vis_input_dim: int, fpn_vis_input_dim: int, txt_input_dim: int,
                 node_type_dim: int, feat_dim: int, vis_proj_layers: int,
                 pos_embed_proj_layers: int, effect_proj_layers: int,
                 segment_proj_layers: int, effect_transformer_layers: int,
                 state_graph_encoder_layers: int, spatial_graph_encoder_layers: int,
                 gnn_layers: int, gnn_heads: int, gnn_dropout: float, num_classes: int,
                 pos_encoding_type: str = 'sinusoidal', max_pos_len: int = 5000):
        super().__init__()
        
        # Store configuration
        self.clip_vis_dim = clip_vis_input_dim
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # Initialize all components
        self._init_all_components(
            clip_vis_input_dim=clip_vis_input_dim, fpn_vis_input_dim=fpn_vis_input_dim,
            txt_input_dim=txt_input_dim, node_type_dim=node_type_dim, embed_dim=feat_dim,
            vis_proj_layers=vis_proj_layers, pos_embed_layers=pos_embed_proj_layers,
            effect_proj_layers=effect_proj_layers, segment_proj_layers=segment_proj_layers,
            effect_transformer_layers=effect_transformer_layers,
            state_graph_encoder_layers=state_graph_encoder_layers,
            spatial_graph_encoder_layers=spatial_graph_encoder_layers,
            gnn_layers=gnn_layers, gnn_heads=gnn_heads, gnn_dropout=gnn_dropout,
            pos_encoding_type=pos_encoding_type, max_pos_len=max_pos_len
        )
        
        # Initialize parameters
        self._init_params()
    
    def _init_all_components(self, clip_vis_input_dim: int, fpn_vis_input_dim: int, 
                            txt_input_dim: int, node_type_dim: int, embed_dim: int,
                            vis_proj_layers: int, pos_embed_layers: int,
                            effect_proj_layers: int, segment_proj_layers: int,
                            effect_transformer_layers: int,
                            state_graph_encoder_layers: int, spatial_graph_encoder_layers: int,
                            gnn_layers: int, gnn_heads: int, gnn_dropout: float,
                            pos_encoding_type: str, max_pos_len: int):
        """Initialize all model components."""
        
        # Projection layers
        self.state_vis_proj = MLP(3 * clip_vis_input_dim, embed_dim, vis_proj_layers)
        self.relation_vis_proj = MLP(3 * clip_vis_input_dim, embed_dim, vis_proj_layers)
        self.pos_embed_proj = MLP(4, clip_vis_input_dim, pos_embed_layers)
        self.state_effect_proj = MLP(fpn_vis_input_dim, embed_dim, effect_proj_layers)
        self.relation_effect_proj = MLP(fpn_vis_input_dim, embed_dim, effect_proj_layers)
        self.enhanced_segment_proj = MLP(fpn_vis_input_dim + 2 * embed_dim, fpn_vis_input_dim, segment_proj_layers)

        # Graph components
        self.graph_node_type_embed = nn.Embedding(3, node_type_dim)
        self.gnn = GAT(
            in_channels=txt_input_dim + node_type_dim,
            hidden_channels=embed_dim,
            out_channels=embed_dim,
            num_layers=gnn_layers,
            heads=gnn_heads,
            dropout=gnn_dropout
        )
        
        # Encoders
        self.effect_transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=fpn_vis_input_dim,
                nhead=8,
                dim_feedforward=fpn_vis_input_dim * 4,
                batch_first=True
            ),
            num_layers= effect_transformer_layers
        )
        
        self.state_graph_encoder = MLP(embed_dim, embed_dim, state_graph_encoder_layers)
        self.spatial_graph_encoder = MLP(embed_dim, embed_dim, spatial_graph_encoder_layers)
        
        # Loss components
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.logits_scale_1 = Exp_Scale(np.log(1 / 0.07))
        self.logits_scale_2 = Exp_Scale(np.log(1 / 0.07))
        self.l2_loss = nn.MSELoss(reduction='sum')
        
        # Helper modules
        self.vlm_processor = VLMFeatureProcessor()
        self.graph_processor = SceneGraphProcessor(self.graph_node_type_embed)
        self.contrastive_module = ContrastiveLearningModule(self.criterion, embed_dim)
        self.feature_enhancer = FeatureEnhancer(
            self.num_classes,
            self.effect_transformer, self.state_effect_proj,
            self.relation_effect_proj, self.enhanced_segment_proj,
            fpn_vis_input_dim, embed_dim, pos_encoding_type, max_pos_len
        )
    
    def _init_params(self):
        """Initialize model parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        feature_segments, 
        video_list, 
        segment_action_labels, 
        contrast_mode, device
    ):
        """Forward pass of the Effect Model.
        
        Args:
            feature_segments: List of feature tensors for each segment.
            video_list: List of dictionaries containing video data with 'visual_info' and 'scene_graph' fields.
            segment_action_labels: List of action label tensors for each segment.
            contrast_mode: Mode for contrastive learning ('loss' or 'logits').
            device: Target device for tensor operations.
            
        Returns:
            Tuple containing loss dictionary with keys 'rel_contrast_loss', 'stt_contrast_loss', 'effect_rel_contrast_loss', 'effect_stt_contrast_loss', and list of enhanced segment features.
        """
        
        # Process inputs and get masks
        segment_action_labels = torch.cat(segment_action_labels, dim=0).to(device)
        vlm_features = self.vlm_processor.extract_vlm_features(video_list, device)
        graph_data, graph_mask, sub_graph_node = self.graph_processor.process_scene_graphs(video_list, device)
        effect_model_mask = torch.logical_and(vlm_features['frame_contrast_mask'], graph_mask).bool()
        
        # Early return if no valid data
        valid_indices = torch.where(effect_model_mask)[0]
        if valid_indices.shape[0] == 0:
            zero_loss = torch.tensor(0.0).to(device)
            return {
                'rel_contrast_loss': zero_loss,
                'stt_contrast_loss': zero_loss,
                'effect_rel_contrast_loss': zero_loss,
                'effect_stt_contrast_loss': zero_loss,
            }, feature_segments
        
        # Process graphs through GNN
        node_features, graph_batch = self.gnn(graph_data)
        graph_outputs = []
        for graph_id in graph_batch.unique():
            graph_node_features = node_features[graph_batch == graph_id]
            graph_outputs.append(graph_node_features)
        
        # Extract subgraph features
        state_features = []
        spatial_features = []
        for i in range(valid_indices.shape[0]):
            idx = valid_indices[i].item()
            output_node_features = graph_outputs[idx]
            
            state_nodes = sub_graph_node['state_graph_node'][idx]
            spatial_nodes = sub_graph_node['spatial_graph_node'][idx]
            
            state_features.append(output_node_features[state_nodes].mean(dim=0))
            spatial_features.append(output_node_features[spatial_nodes].mean(dim=0))
        
        graph_features = {
            'state': self.state_graph_encoder(torch.stack(state_features, dim=0)),
            'relation': self.spatial_graph_encoder(torch.stack(spatial_features, dim=0))
        }
        
        # Extract and project visual features
        rel_pos_feat = self.pos_embed_proj(vlm_features['relative_pos_embed'][effect_model_mask])
        rel_pos_feat = rel_pos_feat.reshape(-1, 2 * self.clip_vis_dim)
        vis_obj_feat = vlm_features['object_vis_feature'][effect_model_mask]
        vis_obj_feat = vis_obj_feat.reshape(-1, 2 * self.clip_vis_dim)
        vis_glb_feat = vlm_features['global_vis_feature'][effect_model_mask]
        vis_glb_feat = vis_glb_feat.reshape(-1, self.clip_vis_dim)

        visual_features = {
            'state': self.state_vis_proj(torch.cat([vis_obj_feat, vis_glb_feat], dim=-1)),
            'relation': self.relation_vis_proj(torch.cat([rel_pos_feat, vis_glb_feat], dim=-1))
        }
        
        # Enhance segments and get effect tokens
        enhanced_segments, state_effect_token, relation_effect_token = \
            self.feature_enhancer.enhance_feature(feature_segments, effect_model_mask, device)
        
        # Compute all losses
        losses = self._compute_all_losses(
            visual_features, graph_features, 
            state_effect_token, relation_effect_token,
            contrast_mode
        )
        
        return losses, enhanced_segments
    
    def _compute_all_losses(self, visual_features: Dict, graph_features: Dict,
                           state_effect_token: torch.Tensor, relation_effect_token: torch.Tensor,
                           contrast_mode: str) -> Dict[str, torch.Tensor]:
        """Compute all losses.
        
        Args:
            visual_features: Dictionary containing 'state' and 'relation' visual features.
            graph_features: Dictionary containing 'state' and 'relation' graph features.
            state_effect_token: State effect token tensor from the transformer.
            relation_effect_token: Relation effect token tensor from the transformer.
            contrast_mode: Mode for contrastive learning ('loss' or 'logits').
            
        Returns:
            Dictionary containing 'rel_contrast_loss', 'stt_contrast_loss', 'effect_stt_contrast_loss', and 'effect_rel_contrast_loss'.
        """

        stt_loss = self.contrastive_module.compute_image_text_contrast(
            visual_features['state'], graph_features['state'],
            self.logits_scale_1, None, contrast_mode
        )

        rel_loss = self.contrastive_module.compute_image_text_contrast(
            visual_features['relation'], graph_features['relation'],
            self.logits_scale_2, None, contrast_mode
        )

        features_to_normalize = [
            state_effect_token, relation_effect_token,
            visual_features['state'], visual_features['relation'],
            graph_features['state'], graph_features['relation']
        ]
        
        normalized_features = []
        for feat in features_to_normalize:
            normalized_feat = feat / torch.norm(feat, dim=-1, keepdim=True)
            normalized_features.append(normalized_feat)
        state_effect_token, relation_effect_token, state_vis, \
            relation_vis, graph_state, graph_spatial = normalized_features

        state_vis_loss = self.l2_loss(state_effect_token, state_vis)
        rel_vis_loss = self.l2_loss(relation_effect_token, relation_vis)
        state_graph_loss = self.l2_loss(state_effect_token, graph_state)
        rel_graph_loss = self.l2_loss(relation_effect_token, graph_spatial)
        effect_stt_loss = (state_vis_loss + state_graph_loss) / 2           
        effect_rel_loss = (rel_vis_loss + rel_graph_loss) / 2

        return {
            'rel_contrast_loss': rel_loss,
            'stt_contrast_loss': stt_loss,
            'effect_stt_contrast_loss': effect_stt_loss,
            'effect_rel_contrast_loss': effect_rel_loss,
        }
    
    @torch.no_grad()
    def inference(self, feature_segments: List[torch.Tensor], action_labels: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
        """Inference mode for segment enhancement without computing losses.
        
        Args:
            feature_segments: List of feature tensors for each segment.
            action_labels: List of action label tensors for each segment.
            device: Target device for tensor operations.
            
        Returns:
            List of enhanced segment feature tensors.
        """
        concat_action_labels = torch.cat(action_labels, dim=0).to(device)
        effect_model_mask = (concat_action_labels > 0).to(device).bool()
        if not effect_model_mask.any():
            return feature_segments
        else:
            enhanced_segments, _, _ = self.feature_enhancer.enhance_feature(
                feature_segments, effect_model_mask, device
            )
            return enhanced_segments