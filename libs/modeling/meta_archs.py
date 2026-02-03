import torch
from torch import nn
from torch.nn import functional as F
from .action_model import Action_Model
from .effect_model import Effect_Model
from libs.datasets import to_segments, to_frame_wise
from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from ..utils import batched_nms
from .heads import DynClsHead, DynRegHead


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """Transformer-based model for single stage action localization with action and effect modeling."""
    
    def __init__(
        self, backbone_type, fpn_type, backbone_arch, scale_factor, input_dim,
        max_seq_len, max_buffer_len_factor, n_head, n_mha_win_size, embd_kernel_size,
        embd_dim, embd_with_ln, fpn_dim, fpn_with_ln, fpn_start_level, head_dim,
        regression_range, head_num_layers, head_kernel_size, head_with_ln,
        use_abs_pe, use_rel_pe, num_classes, num_node, train_cfg, test_cfg,
        action_model_cfg, effect_model_cfg
    ):
        super().__init__()
        
        # Initialize all configuration parameters
        self._init_all_params(backbone_type, backbone_arch, scale_factor, fpn_start_level,
                             regression_range, num_classes, max_seq_len, n_mha_win_size,
                             train_cfg, test_cfg)
        
        # Build all network components
        self._build_all_components(backbone_type, input_dim, embd_dim, embd_kernel_size,
                                  backbone_arch, n_head, max_seq_len, scale_factor,
                                  embd_with_ln, use_abs_pe, use_rel_pe, num_node,
                                  fpn_type, fpn_dim, fpn_with_ln, fpn_start_level,
                                  head_dim, head_kernel_size, head_num_layers, head_with_ln,
                                  max_buffer_len_factor, action_model_cfg, effect_model_cfg)
        
        # Initialize training state
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.cur_epoch = -1
        self.tot_epoch = -1

    def _init_all_params(self, backbone_type, backbone_arch, scale_factor, fpn_start_level,
                        regression_range, num_classes, max_seq_len, n_mha_win_size,
                        train_cfg, test_cfg):
        """Initialize all model parameters."""
        # Basic parameters
        self.backbone_type = backbone_type
        self.fpn_strides = [scale_factor**i for i in range(fpn_start_level, backbone_arch[-1]+1)]
        self.reg_range = regression_range
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        
        # Multi-head attention window sizes
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        
        # Calculate max divisible factor
        max_div_factor = 1
        for s, w in zip(self.fpn_strides, self.mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert self.max_seq_len % stride == 0, \
                "max_seq_len must be divisible by fpn stride and window size"
            max_div_factor = max(max_div_factor, stride)
        self.max_div_factor = max_div_factor
        
        # Training configuration
        self.train_center_sample = train_cfg['center_sample']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']
        
        # Test configuration
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

    def _build_all_components(self, backbone_type, input_dim, embd_dim, embd_kernel_size,
                             backbone_arch, n_head, max_seq_len, scale_factor, embd_with_ln,
                             use_abs_pe, use_rel_pe, num_node, fpn_type, fpn_dim, fpn_with_ln,
                             fpn_start_level, head_dim, head_kernel_size, head_num_layers,
                             head_with_ln, max_buffer_len_factor, action_model_cfg, effect_model_cfg):
        """Build all network components."""
        
        # Backbone
        backbone_params = {
            'n_in': input_dim, 'n_embd': embd_dim, 'n_head': n_head, 'n_embd_ks': embd_kernel_size,
            'max_len': max_seq_len, 'arch': backbone_arch, 'mha_win_size': self.mha_win_size,
            'scale_factor': scale_factor, 'with_ln': embd_with_ln, 'attn_pdrop': 0.0,
            'proj_pdrop': self.train_dropout, 'path_pdrop': self.train_droppath,
            'use_abs_pe': use_abs_pe, 'use_rel_pe': use_rel_pe,
        }
        if backbone_type == 'convGCNTransformer':
            backbone_params['num_node'] = num_node
        self.backbone = make_backbone(backbone_type, **backbone_params)
        
        # Neck
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)
        self.neck = make_neck(fpn_type, **{
            'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
            'out_channel': fpn_dim, 'scale_factor': scale_factor,
            'start_level': fpn_start_level, 'with_ln': fpn_with_ln
        })
        
        # Heads
        self.point_generator = make_generator('point', **{
            'max_seq_len': self.max_seq_len * max_buffer_len_factor,
            'fpn_strides': self.fpn_strides, 'regression_range': self.reg_range
        })

        self.cls_head = DynClsHead(
            input_dim=fpn_dim, feat_dim=head_dim, 
            num_classes=self.num_classes
        )

        self.reg_head = DynRegHead(
            input_dim=fpn_dim, feat_dim=head_dim, 
            fpn_levels=len(self.fpn_strides)
        )
        
        self.action_model = Action_Model(
            vis_input_dim=fpn_dim, 
            txt_input_dim=action_model_cfg['txt_input_dim'],
            feat_dim=action_model_cfg['feat_dim'], 
            num_actions=self.num_classes, length_prompt=4
        )

        self.effect_model = Effect_Model(
            clip_vis_input_dim=effect_model_cfg['vis_input_dim'], 
            fpn_vis_input_dim=fpn_dim,
            txt_input_dim=effect_model_cfg['txt_input_dim'], 
            node_type_dim=effect_model_cfg['node_type_dim'],
            feat_dim=effect_model_cfg['feat_dim'],
            num_classes=self.num_classes, 
            vis_proj_layers=effect_model_cfg['vis_proj_num_layers'],
            pos_embed_proj_layers=effect_model_cfg['pos_embed_proj_num_layers'],
            effect_proj_layers=effect_model_cfg['effect_proj_num_layers'],
            segment_proj_layers=effect_model_cfg['segment_proj_num_layers'],
            effect_transformer_layers=effect_model_cfg['effect_transformer_num_layers'],
            state_graph_encoder_layers=effect_model_cfg['state_graph_encoder_num_layers'],
            spatial_graph_encoder_layers=effect_model_cfg['spatial_graph_encoder_num_layers'],
            gnn_layers=effect_model_cfg['gnn_num_layers'], 
            gnn_heads=effect_model_cfg['gnn_num_heads'],
            gnn_dropout=effect_model_cfg['gnn_dropout'],
            pos_encoding_type = 'sinusoidal',
            max_pos_len = 5000
        )

    @property
    def device(self):
        """Get device of model parameters."""
        params = list(self.parameters())
        if len(params) == 0:
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return params[0].device

    def extract_feature(self, video_list):
        """Extract features from video inputs through backbone and neck.
        
        Args:
            video_list: List of video data containing features and masks.
            
        Returns:
            tuple: Contains:
                - fpn_feats: Features from the feature pyramid network
                - fpn_masks: Masks corresponding to the features
                - points: Generated points for action localization
                - cls_prob_logits: Classification logits for action classes
                - out_offsets: Regression offsets for segment boundaries
        """
        # Preprocess inputs
        batched_bboxes, batched_bbox_classes, batched_edge_maps, \
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # Forward through backbone and neck
        if self.backbone_type == 'convTransformer':
            feats, masks = self.backbone(batched_inputs, batched_masks)
        else:
            feats, masks = self.backbone(batched_bboxes, batched_bbox_classes,
                                       batched_edge_maps, batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # Generate predictions
        points = self.point_generator(fpn_feats)
        cls_prob_logits = self.cls_head(fpn_feats, fpn_masks)
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # Reshape outputs
        cls_prob_logits = [x.permute(0, 2, 1) for x in cls_prob_logits]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        
        return fpn_feats, fpn_masks, points, cls_prob_logits, out_offsets

    def generate_segment_features(self, fpn_feats, fpn_masks, action_segments):
        """Generate segment-wise visual features from frame-wise features.
        
        Args:
            fpn_feats: Features from the feature pyramid network.
            fpn_masks: Masks corresponding to the features.
            action_segments: Segments of actions to extract features from.
            
        Returns:
            list: List of segment-wise visual features extracted from the frame-wise features.
        """
        valid_mask = fpn_masks[0]
        frame_vis_feat = fpn_feats[0].permute(0, 2, 1)[valid_mask]
        
        # Merge and extract segments
        merged_gt_segments = [action_segments[0]]
        for i in range(1, len(action_segments)):
            offset = merged_gt_segments[-1][-1][-1] + 1
            merged_gt_segments.append(action_segments[i] + offset)
        merged_gt_segments = torch.concat(merged_gt_segments, dim=0).to(self.device)
        
        starts = torch.clamp(torch.tensor([seg[0] for seg in merged_gt_segments]), min=0).long()
        ends = torch.clamp(torch.tensor([seg[1] for seg in merged_gt_segments]), 
                          max=frame_vis_feat.shape[0]).long()
        
        return [frame_vis_feat[start:end+1] for start, end in zip(starts, ends)]

    def forward(self, clip_model, video_list, action_tokens, cur_epoch=-1, tot_epoch=-1):
        """Main forward pass for training.
        
        Args:
            clip_model: The model used for action recognition.
            video_list: List of video data containing features, segments, and labels.
            action_tokens: Tokens representing action classes.
            cur_epoch: Current epoch number (default: -1).
            tot_epoch: Total number of epochs (default: -1).
            
        Returns:
            tuple: Contains:
                - updated_losses: Dictionary of all computed losses including final_loss
                - action_align_acc: Dictionary containing action alignment accuracy metrics
        """
        if cur_epoch != -1 and tot_epoch != -1:
            self.cur_epoch = cur_epoch
            self.tot_epoch = tot_epoch

        # Extract features and generate predictions
        fpn_feats, fpn_masks, points, cls_prob_logits, out_offsets = self.extract_feature(video_list)
        gt_segments = [x['segments'].to(self.device) for x in video_list]
        gt_labels = [x['labels'].to(self.device) for x in video_list]
        gt_cls_labels, gt_offsets = self.label_points(points, gt_segments, gt_labels)
        
        # Generate segment features and compute losses
        segments = self.generate_segment_features(fpn_feats, fpn_masks, gt_segments)
        
        losses, enhanced_segments = self.effect_model(
            feature_segments=segments, video_list=video_list,
            segment_action_labels=gt_labels, contrast_mode='loss', 
            device=self.device
        )
        
        # Action model losses
        segment_vis_feature = torch.stack([seg.mean(dim=0) for seg in enhanced_segments], dim=0)
        action_loss, action_align_acc = self.action_model(
            clip_model=clip_model, segment_vis_feature=segment_vis_feature,
            action_tokens=action_tokens, action_labels=gt_labels, device=self.device
        )
        losses["action_contrast_loss"] = action_loss
        
        # Compute final losses
        updated_losses = self._compute_losses(
            fpn_masks, cls_prob_logits, out_offsets, gt_cls_labels, gt_offsets, losses
        )
        return updated_losses, {**action_align_acc}

    @torch.no_grad()
    def inference(self, video_list, clip_model, action_tokens):
        """Inference for video action detection.
        
        Args:
            video_list: List of video data for inference.
            clip_model: The model for action recognition.
            action_tokens: Tokens representing action classes.
            
        Returns:
            dict: Processed results of action detection for the video, including:
                - segments: Detected action segments
                - segment_action_pred: Predicted action labels
                - framewise_action_sim: Frame-wise action similarities
                - segment_action_sim: Segment-wise action similarities
        """
        fpn_feats, fpn_masks, points, cls_prob_logits, out_offsets = self.extract_feature(video_list)
        results = self.segment_video(video_list, points, fpn_masks, cls_prob_logits, out_offsets)
        video_id, vid_result = list(results.items())[0]
        
        # Generate enhanced segment features
        action_segments = [vid_result['segments']]
        segment_feature = self.generate_segment_features(fpn_feats, fpn_masks, action_segments)
        enhanced_segment_feature = \
            self.effect_model.inference(
                feature_segments=segment_feature, 
                action_labels=[vid_result['segment_action_pred']], 
                device=self.device
            )
        
        # Compute action similarities
        segment_vis_feature = torch.stack([seg.mean(dim=0) for seg in enhanced_segment_feature], dim=0)
        action_similarity = self.action_model.inference(
            clip_model=clip_model, 
            segment_vis_feature=segment_vis_feature,
            action_tokens=action_tokens, 
            action_labels=[vid_result['segment_action_pred']],
            device=self.device
        )

        # Update results with similarities
        for vid, result in results.items():
            repeat_num = result['segments'][:, 1] - result['segments'][:, 0] + 1
            result['framewise_action_sim'] = torch.repeat_interleave(
                action_similarity.cpu(), repeats=repeat_num
            )
            result['segment_action_sim'] = action_similarity.cpu()
        return results

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        """Label points for training.
        
        Args:
            points: Generated points from the point generator.
            gt_segments: Ground truth action segments for all videos.
            gt_labels: Ground truth action labels for all videos.
            
        Returns:
            tuple: Contains:
                - gt_cls: Classification targets for all points
                - gt_offset: Regression targets (offsets) for all points
        """
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        """Label points for a single video.
        
        Args:
            concat_points: Concatenated points from all pyramid levels.
            gt_segment: Ground truth action segments for a single video.
            gt_label: Ground truth action labels for a single video.
            
        Returns:
            tuple: Contains:
                - cls_targets: Classification targets for the points
                - reg_targets: Regression targets (normalized offsets) for the points
        """
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # Compute distances and regression targets
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        # Apply center sampling if enabled
        if self.train_center_sample == 'radius':
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            t_mins = center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # Apply regression range constraints
        max_regress_distance = reg_targets.max(-1)[0]
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # Select targets based on shortest duration
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        min_len, min_len_inds = lens.min(dim=1)

        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        gt_label_one_hot = F.one_hot(gt_label, self.num_classes).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        cls_targets.clamp_(min=0.0, max=1.0)
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def _compute_losses(self, fpn_masks, out_cls_logits, out_offsets,
                       gt_cls_labels, gt_offsets, losses):
        """Compute all losses including classification, regression, and auxiliary losses.
        
        Args:
            fpn_masks: Masks for the feature pyramid network features.
            out_cls_logits: Output classification logits from the model.
            out_offsets: Output regression offsets from the model.
            gt_cls_labels: Ground truth classification labels.
            gt_offsets: Ground truth regression offsets.
            losses: Dictionary of auxiliary losses (e.g., from effect model).
            
        Returns:
            dict: Updated losses dictionary including:
                - cls_loss: Classification loss
                - reg_loss: Regression loss
                - final_loss: Weighted sum of all losses
                - Other auxiliary losses (effect_loss, action_contrast_loss, etc.)
        """
        valid_mask = torch.cat(fpn_masks, dim=1)
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # Update loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + \
                              (1 - self.loss_normalizer_momentum) * max(num_pos, 1)

        # Classification loss
        gt_target = gt_cls[valid_mask]
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask], gt_target,
            reduction='sum', gamma=0.0
        ) / self.loss_normalizer

        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            reg_loss = ctr_diou_loss_1d(pred_offsets, gt_offsets, reduction='sum') / self.loss_normalizer

        losses.update({'cls_loss': cls_loss, 'reg_loss': reg_loss})

        # Normalize auxiliary losses
        loss_keys_to_normalize = [key for key in losses.keys() 
                                 if key not in ['cls_loss', 'reg_loss']]
        for key in loss_keys_to_normalize:
            losses[key] /= self.loss_normalizer

        # Calculate loss weights and final loss
        loss_weights = self._calculate_loss_weights(cls_loss, losses, loss_keys_to_normalize)
        
        final_loss = cls_loss + reg_loss * loss_weights['reg_loss']
        for key in loss_keys_to_normalize:
            final_loss += losses[key] * loss_weights[key]
                
        losses['final_loss'] = final_loss
        return losses

    def _calculate_loss_weights(self, cls_loss, losses, loss_keys_to_normalize):
        """Calculate weights for all loss components."""
        loss_weights = {}
        if self.train_loss_weight > 0:
            for key in ['reg_loss'] + loss_keys_to_normalize:
                loss_weights[key] = self.train_loss_weight
        else:
            loss_weights['reg_loss'] = cls_loss.detach() / max(losses['reg_loss'].item(), 0.01)
            for key in loss_keys_to_normalize:
                loss_weights[key] = cls_loss.detach() / max(losses[key].item(), 0.01)
        return loss_weights

    @torch.no_grad()
    def segment_video(self, video_list, points, fpn_masks, out_cls_logits, out_offsets):
        """Segment video into action proposals.
        
        Args:
            video_list: List of video data containing metadata (video_id, fps, duration).
            points: Generated points for action localization.
            fpn_masks: Masks for the feature pyramid network features.
            out_cls_logits: Output classification logits.
            out_offsets: Output regression offsets.
            
        Returns:
            dict: Processed results for each video containing segments, predictions, and labels.
        """
        results = []
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]

        for idx, (vidx, fps, vlen) in enumerate(zip(vid_idxs, vid_fps, vid_lens)):
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid, cls_logits_per_vid, offsets_per_vid
            )
            results_per_vid.update({'video_id': vidx, 'fps': fps, 'duration': vlen})
            results.append(results_per_vid)

        return self.postprocessing(results, video_list)

    @torch.no_grad()
    def inference_single_video(self, points, fpn_masks, out_cls_logits, out_offsets):
        """Inference on a single video.
        
        Args:
            points: Generated points for action localization.
            fpn_masks: Masks for the feature pyramid network features.
            out_cls_logits: Output classification logits for the video.
            out_offsets: Output regression offsets for the video.
            
        Returns:
            dict: Dictionary containing:
                - segments: Predicted action segments (N x 2 tensor)
                - scores: Confidence scores for each segment
                - labels: Predicted action class labels
                - scores_all: Original scores before filtering
        """
        segs_all, scores_all, cls_idxs_all, org_scores_all = [], [], [], []

        for cls_i, offsets_i, pts_i, mask_i in zip(out_cls_logits, out_offsets, points, fpn_masks):
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()
            org_scores_all.append(pred_prob)
            
            # Filter by confidence and top-k
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
            
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()
            
            pt_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)
            
            # Generate segments
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)
            
            # Filter by duration
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh
            
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        segs_all, scores_all, cls_idxs_all = [torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]]
        return {'segments': segs_all, 'scores': scores_all, 'labels': cls_idxs_all, 'scores_all': org_scores_all}

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """Preprocess video inputs for batching.
        
        Args:
            video_list: List of video data containing features and optional inputs.
            padding_val: Value used for padding inputs (default: 0.0).
            
        Returns:
            tuple: Contains:
                - batched_bboxes: Batched bounding boxes (or None)
                - batched_bbox_classes: Batched bounding box classes (or None)
                - batched_edge_maps: Batched edge maps (or None)
                - batched_inputs: Batched video features
                - batched_masks: Masks indicating valid frames
        """
        feats = [x['feats'] for x in video_list]
        bboxes = [x['bbox'].permute(1, 2, 0) for x in video_list] if 'bbox' in video_list[0] else None
        bbox_classes = [x['bbox_class'].permute(1, 0) for x in video_list] if 'bbox_class' in video_list[0] else None
        edge_maps = [x['edge_map'].permute(1, 2, 0) for x in video_list] if 'edge_map' in video_list[0] else None
        
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            max_len = self.max_seq_len
            
            # Batch features
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            
            # Batch other inputs if they exist
            batched_bboxes = self._batch_optional_input(bboxes, max_len, padding_val) if bboxes else None
            batched_bbox_classes = self._batch_optional_input(bbox_classes, max_len, padding_val) if bbox_classes else None
            batched_edge_maps = self._batch_optional_input(edge_maps, max_len, padding_val) if edge_maps else None
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(feats[0], padding_size, value=padding_val).unsqueeze(0)
            batched_bboxes = F.pad(bboxes[0], [0, max_len - bboxes[0].shape[-1]], value=padding_val).unsqueeze(0) if bboxes else None
            batched_bbox_classes = F.pad(bbox_classes[0], [0, max_len - bbox_classes[0].shape[-1]], value=padding_val).unsqueeze(0) if bbox_classes else None
            batched_edge_maps = F.pad(edge_maps[0], [0, max_len - edge_maps[0].shape[-1]], value=padding_val).unsqueeze(0) if edge_maps else None

        batched_masks = (torch.arange(max_len)[None, :] < feats_lens[:, None]).unsqueeze(1).to(self.device)
        
        # Move to device
        batched_inputs = batched_inputs.to(self.device)
        if batched_bboxes is not None:
            batched_bboxes = batched_bboxes.to(self.device)
        if batched_bbox_classes is not None:
            batched_bbox_classes = batched_bbox_classes.to(self.device)
        if batched_edge_maps is not None:
            batched_edge_maps = batched_edge_maps.to(self.device)

        return batched_bboxes, batched_bbox_classes, batched_edge_maps, batched_inputs, batched_masks

    def _batch_optional_input(self, inputs, max_len, padding_val):
        """Helper function to batch optional inputs like bboxes, bbox_classes, edge_maps."""
        if inputs[0].ndim == 3:  # For bboxes and edge_maps
            batch_shape = [len(inputs), inputs[0].shape[0], inputs[0].shape[1], max_len]
        else:  # For bbox_classes
            batch_shape = [len(inputs), inputs[0].shape[0], max_len]
        
        batched_data = inputs[0].new_full(batch_shape, padding_val)
        for data, pad_data in zip(inputs, batched_data):
            pad_data[..., :data.shape[-1]].copy_(data)
        return batched_data

    @torch.no_grad()
    def postprocessing(self, results, video_list):
        """Postprocess results including NMS and temporal conversion.
        
        Args:
            results: Raw inference results from the model.
            video_list: List of video data for reference (features, segments, labels).
            
        Returns:
            dict: Processed results for each video containing:
                - segments: Time stamps of detected segments
                - gt_segments: Ground truth segments
                - gt_segments_labels: Ground truth action labels
                - segment_action_pred: Predicted action labels for segments
                - frame_action_pred: Frame-wise action predictions
                - frame_action_label: Frame-wise ground truth action labels
                - frame_error_label: Frame-wise error labels
                - segment_error_label: Segment-wise error labels
                - segment_action_prob: Action probabilities for segments
        """
        new_video_list = {vid_feat_dict['video_id']: vid_feat_dict for vid_feat_dict in video_list}
        processed_results = {}
        
        for results_per_vid in results:
            video_id = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            processed_results[video_id] = {}
            
            # Move to CPU and apply NMS
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            
            if self.test_nms_method != 'none':
                segs, scores, labels = batched_nms(
                    segs, scores, labels, self.test_iou_threshold, self.test_min_score,
                    self.test_max_seg_num, use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms, sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
                
                # Convert to seconds
                if segs.shape[0] > 0:
                    segs = segs / fps
                    segs[segs<=0.0] *= 0.0
                    segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            else:
                num_frames = int(vlen * fps)
                segs = segs[:num_frames,:]
                scores = scores[:num_frames]
                labels = labels[:num_frames]
            
            # Convert to frame-wise and segment representations
            framewise_action_pred = to_frame_wise(
                segs, labels, scores, new_video_list[video_id]['feats'].size(1),
                fps=new_video_list[video_id]['fps']
            )
            framewise_action_label = to_frame_wise(
                new_video_list[video_id]['segments'], new_video_list[video_id]['labels'],
                None, int(new_video_list[video_id]['segments'][-1,1])
            )
            framewise_error_label = to_frame_wise(
                new_video_list[video_id]['segments'], new_video_list[video_id]['labels_error'],
                None, int(new_video_list[video_id]['segments'][-1,1])
            )
            segment_action_pred, time_stamp_labels = to_segments(framewise_action_pred)
            
            # Store results
            processed_results[video_id].update({
                'segments': torch.tensor(time_stamp_labels).cpu(),
                'gt_segments': new_video_list[video_id]['segments'].cpu(),
                'gt_segments_labels': new_video_list[video_id]['labels'].cpu(),
                'segment_action_pred': torch.tensor(segment_action_pred).cpu(),
                'frame_action_pred': framewise_action_pred.cpu(),
                'frame_action_label': framewise_action_label.cpu(),
                'frame_error_label': framewise_error_label.cpu(),
                'segment_error_label': new_video_list[video_id]['labels_error'].cpu().numpy(),
                'segment_action_prob': labels.numpy(),
            })

        return processed_results
