import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import DTFAM, MaskedConv1D, Scale, LayerNorm, DynamicScale_chk, DynamicFeatureAttentionLayer

class PtTransformerHead(nn.Module):
    """Base class for transformer heads with shared functionality."""
    
    def __init__(self, input_dim, feat_dim, num_layers=3, kernel_size=3, 
                 act_layer=nn.ReLU, with_ln=False):
        super().__init__()
        self.act = act_layer()
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        # Build head layers
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            self.head.append(MaskedConv1D(
                in_dim, feat_dim, kernel_size, stride=1,
                padding=kernel_size//2, bias=(not with_ln)
            ))
            self.norm.append(LayerNorm(feat_dim) if with_ln else nn.Identity())

    def forward_head(self, fpn_feats, fpn_masks):
        """Apply head layers to FPN features."""
        out_feats = []
        for cur_feat, cur_mask in zip(fpn_feats, fpn_masks):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            out_feats.append((cur_out, cur_mask))
        return out_feats


class PtTransformerClsHead(PtTransformerHead):
    """1D Conv heads for classification with configurable layers and prior probability."""
    
    def __init__(self, input_dim, feat_dim, num_classes, prior_prob=0.01, 
                 num_layers=3, kernel_size=3, act_layer=nn.ReLU, 
                 with_ln=False, empty_cls=[]):
        super().__init__(input_dim, feat_dim, num_layers, kernel_size, act_layer, with_ln)
        
        # Classification layer
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size//2
        )
        
        # Initialize weights
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # Handle empty categories
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        """Forward pass through classification head."""
        head_feats = self.forward_head(fpn_feats, fpn_masks)
        out_logits = tuple()
        for cur_out, cur_mask in head_feats:
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )
        return out_logits


class PtTransformerRegHead(PtTransformerHead):
    """Shared 1D Conv heads for regression with scale factors per FPN level."""
    
    def __init__(self, input_dim, feat_dim, fpn_levels, num_layers=3, 
                 kernel_size=3, act_layer=nn.ReLU, with_ln=False):
        super().__init__(input_dim, feat_dim, num_layers, kernel_size, act_layer, with_ln)
        
        self.fpn_levels = fpn_levels
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size, stride=1, padding=kernel_size//2
        )

    def forward(self, fpn_feats, fpn_masks):
        """Forward pass through regression head."""
        assert len(fpn_feats) == len(fpn_masks) == self.fpn_levels
        
        head_feats = self.forward_head(fpn_feats, fpn_masks)
        out_offsets = tuple()
        for l, (cur_out, cur_mask) in enumerate(head_feats):
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )
        return out_offsets


class TDynPtTransformerClsHead(nn.Module):
    """
    Shared 1D MSDy-head for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=5,
        act_layer=nn.ReLU,
        empty_cls = [],
        gate_activation_kargs: dict = None
    ):
        super().__init__()
        self.act = act_layer()
        
        assert num_layers-1 >0

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()

        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            cls_subnet_conv = DTFAM(
                dim=in_dim, o_dim= feat_dim, ka=kernel_size, 
                conv_type = 'others', 
                gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs
            )

            self.head.append(
                DynamicScale_chk(
                in_dim,
                out_dim,
                num_convs=1,
                kernel_size=kernel_size,
                padding=1,
                stride=kernel_size // 2,
                num_groups=1,
                num_adjacent_scales=2,
                depth_module=cls_subnet_conv,
                gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs))

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        feats = fpn_feats
        for i in range(len(self.head)):
            feats, fpn_masks = self.head[i](feats, fpn_masks)
            for j in range(len(feats)):
                feats[j] =  self.act(feats[j])

        for cur_out, cur_mask in zip(feats, fpn_masks):
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        return out_logits

class TDynPtTransformerRegHead(nn.Module):
    """
    Shared 1D MSDy-head for regression
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=5,
        act_layer=nn.ReLU,
        gate_activation_kargs: dict = None,
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        
        assert num_layers-1 > 0

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            reg_subnet_conv = DTFAM(dim=in_dim, o_dim= feat_dim, ka=kernel_size, conv_type = 'others', gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs)

            self.head.append(DynamicScale_chk(
                in_dim,
                out_dim,
                num_convs=1,
                kernel_size=kernel_size,
                padding=1,
                stride=kernel_size // 2,
                num_groups=1,
                num_adjacent_scales=2,
                depth_module=reg_subnet_conv,
                gate_activation=gate_activation_kargs['type'],
                gate_activation_kargs = gate_activation_kargs)
                )

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )


    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels
        
        # apply the classifier for each pyramid level
        out_offsets = tuple()
        feats = fpn_feats
        for i in range(len(self.head)):
            
            feats, fpn_masks = self.head[i](feats, fpn_masks)
            for j in range(len(feats)):
                feats[j] = self.act(feats[j])
        
        for l in range(self.fpn_levels):
            cur_offsets, _  = self.offset_head(feats[l], fpn_masks[l])            
            out_offsets +=  ( F.relu(self.scale[l](cur_offsets)) , )
               
        return out_offsets 
    
class CrossScaleLayer(nn.Module):
    """Simplified cross-scale information exchange layer with weighted fusion"""
    
    def __init__(self, in_dim, out_dim, kernel_size=3, with_ln=False, num_adjacent_scales=2, depth_module=None):
        super().__init__()
        self.num_adjacent_scales = num_adjacent_scales
        
        # Individual processing conv
        self.individual_conv = MaskedConv1D(
            in_dim, out_dim, kernel_size,
            stride=1, padding=kernel_size//2, bias=(not with_ln)
        )
        self.norm = LayerNorm(out_dim) if with_ln else nn.Identity()
        
        # Weighted fusion parameters
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        
        # Resize function using interpolation
        self.resize = lambda x, s: F.interpolate(x, size=s, mode="linear", align_corners=False)
        self.depth_module = depth_module
    
    def forward(self, fpn_feats, fpn_masks):
        # Step 1: Process each level individually
        processed_feats = []
        for feat, mask in zip(fpn_feats, fpn_masks):
            processed_feat, _ = self.individual_conv(feat, mask)
            processed_feat = self.norm(processed_feat)
            processed_feats.append(processed_feat)
        
        # Step 2: Weighted fusion across scales
        enhanced_feats = []
        
        for l, current_feat in enumerate(processed_feats):
            scale_features = []
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                
                if 0 <= l_source < len(processed_feats):
                    feature = self.resize(processed_feats[l_source], current_feat.shape[-1:])
                    scale_features.append(feature)
            
            if scale_features:
                fused_feature = sum(scale_features)
                enhanced_feat = fused_feature * self.scale_weight + current_feat * self.output_weight
            else:
                enhanced_feat = current_feat
            
            if self.depth_module is not None:
                enhanced_feat, masks = self.depth_module(enhanced_feat, fpn_masks[l])
            
            enhanced_feats.append(enhanced_feat)
        
        return enhanced_feats, fpn_masks


class DynClsHead(nn.Module):
    """Simplified dynamic multi-scale classification head with weighted fusion"""
    
    def __init__(self, input_dim, feat_dim, num_classes, prior_prob=0.01,
                 num_layers=3, kernel_size=3, act_layer=nn.ReLU, 
                 with_ln=False, empty_cls=[], num_adjacent_scales=2):
        super().__init__()
        self.act = act_layer()
        self.head = nn.ModuleList()
        
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            depth_module = DynamicFeatureAttentionLayer(dim=feat_dim)
            layer = CrossScaleLayer(
                in_dim, feat_dim, kernel_size, 
                with_ln=with_ln, num_adjacent_scales=num_adjacent_scales,
                depth_module=depth_module
            )
            self.head.append(layer)
        
        # Classification head
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size//2
        )
        
        # Initialize bias
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)
        
    def forward(self, fpn_feats, fpn_masks):
        """Forward pass with weighted multi-scale fusion"""
        feats = fpn_feats
        masks = fpn_masks
        
        # Apply cross-scale layers
        for layer in self.head:
            feats, masks = layer(feats, masks)
            feats = [self.act(feat) for feat in feats]
        
        # Classification output
        out_logits = tuple()
        for cur_feat, cur_mask in zip(feats, masks):
            cur_logits, _ = self.cls_head(cur_feat, cur_mask)
            out_logits += (cur_logits,)
        
        return out_logits


class DynRegHead(nn.Module):
    """Simplified dynamic multi-scale regression head with weighted fusion"""
    
    def __init__(self, input_dim, feat_dim, fpn_levels, num_layers=3,
                 kernel_size=3, act_layer=nn.ReLU, with_ln=False, num_adjacent_scales=2):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        self.head = nn.ModuleList()
        
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            depth_module = DynamicFeatureAttentionLayer(dim=feat_dim)
            layer = CrossScaleLayer(
                in_dim, feat_dim, kernel_size,
                with_ln=with_ln, num_adjacent_scales=num_adjacent_scales,
                depth_module=depth_module
            )
            self.head.append(layer)
        
        # Regression components
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size, stride=1, padding=kernel_size//2
        )
    
    def forward(self, fpn_feats, fpn_masks):
        """Forward pass with weighted multi-scale fusion"""
        assert len(fpn_feats) == len(fpn_masks) == self.fpn_levels
        
        feats = fpn_feats
        masks = fpn_masks
        
        # Apply cross-scale layers
        for layer in self.head:
            feats, masks = layer(feats, masks)
            feats = [self.act(feat) for feat in feats]
        
        # Regression output
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(feats, masks)):
            cur_offsets, _ = self.offset_head(cur_feat, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)
        
        return out_offsets