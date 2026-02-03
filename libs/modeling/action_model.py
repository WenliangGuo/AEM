import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List
from .blocks import Exp_Scale
from .losses import sigmoid_focal_loss
from metric import topk_accuracy

class CLIPTokenProcessor:
    """Handles CLIP token processing and embedding operations."""
    
    # CLIP special token IDs
    START_TOKEN_ID = 49406  # <|startoftext|>
    END_TOKEN_ID = 49407    # <|endoftext|>
    
    @staticmethod
    def find_token_positions(text: torch.Tensor, token_id: int) -> torch.Tensor:
        """Find positions of specific token in text sequence.
        
        Args:
            text: Input text tensor containing token IDs.
            token_id: Specific token ID to search for.
            
        Returns:
            Tensor containing indices where the token appears.
        """
        return (text == token_id).nonzero(as_tuple=True)[0]
    
    @staticmethod
    def extract_text_segments(text: torch.Tensor, base_embeddings: torch.Tensor, 
                             sample_idx: int) -> Dict[str, torch.Tensor]:
        """Extract start, middle, and end text segments for a single sample.
        
        Args:
            text: Input text tensor with token IDs.
            base_embeddings: Base token embeddings tensor of shape (batch_size, max_len, embed_dim).
            sample_idx: Index of the sample to extract segments from.
            
        Returns:
            Dictionary containing 'start' (1, embed_dim), 'middle' (var_len, embed_dim), and 'end' (1, embed_dim) embeddings.
        """
        max_len, embed_dim = base_embeddings.shape[1], base_embeddings.shape[2]
        device = text.device
        
        # Find token positions
        start_positions = CLIPTokenProcessor.find_token_positions(
            text[sample_idx], CLIPTokenProcessor.START_TOKEN_ID
        )
        end_positions = CLIPTokenProcessor.find_token_positions(
            text[sample_idx], CLIPTokenProcessor.END_TOKEN_ID
        )
        
        # Determine indices
        start_idx = start_positions[0].item() if len(start_positions) > 0 else 0
        end_idx = end_positions[0].item() if len(end_positions) > 0 else max_len - 1
        
        # Extract embeddings
        start_embed = base_embeddings[sample_idx, start_idx, :].unsqueeze(0)
        end_embed = base_embeddings[sample_idx, end_idx, :].unsqueeze(0)
        
        if end_idx > (start_idx + 1):
            middle_embed = base_embeddings[sample_idx, start_idx+1:end_idx, :]
        else:
            middle_embed = torch.empty(0, embed_dim, device=device)
        
        return {
            'start': start_embed,
            'middle': middle_embed,
            'end': end_embed
        }


class CooperativeTextEncoder:
    """Handles cooperative text encoding with learnable prompts."""
    
    def __init__(self, action_prompts: nn.Parameter):
        self.action_prompts = action_prompts
        self.token_processor = CLIPTokenProcessor()
    
    def forward(self, text_tokens: torch.Tensor, clip_model) -> torch.Tensor:
        """Forward pass for cooperative text encoding.
        
        Args:
            text_tokens: Input token IDs of shape (batch_size, max_len).
            clip_model: CLIP model instance for encoding.
            
        Returns:
            Output embeddings of shape (batch_size, embed_dim).
        """
        batch_size, max_len = text_tokens.shape
        device = text_tokens.device
        _, M, embed_dim = self.action_prompts.shape
        
        # Get base text embeddings
        base_embeddings = self._get_base_embeddings(text_tokens, clip_model) # [B, L, D]
        
        # Build cooperative embeddings
        coop_embeddings = self._build_cooperative_embeddings(
            base_embeddings, text_tokens, batch_size, max_len, embed_dim, device
        )
        
        # Add positional embeddings and forward through transformer
        return self._forward_through_transformer(coop_embeddings, text_tokens, clip_model, device)
    
    def _get_base_embeddings(self, text_tokens: torch.Tensor, clip_model) -> torch.Tensor:
        """Get base text embeddings from the CLIP token-embedding layer.
        
        Args:
            text_tokens: Input token IDs.
            clip_model: CLIP model instance.
            
        Returns:
            Base token embeddings of shape (batch_size, max_len, embed_dim).
        """
        clip, _, _ = clip_model
        token_embeds = clip.text.token_embedding(text_tokens)
        return token_embeds          
    
    def _build_cooperative_embeddings(self, base_embeddings: torch.Tensor, text: torch.Tensor,
                                     batch_size: int, max_len: int, embed_dim: int, 
                                     device: torch.device) -> torch.Tensor:
        """Build cooperative embeddings by inserting action prompts.
        
        Args:
            base_embeddings: Base token embeddings.
            text: Input text tensor with token IDs.
            batch_size: Number of samples in the batch.
            max_len: Maximum sequence length.
            embed_dim: Embedding dimension.
            device: Target device for tensors.
            
        Returns:
            Cooperative embeddings tensor of shape (batch_size, max_len, embed_dim).
        """
        coop_embeddings = torch.zeros(batch_size, max_len, embed_dim, device=device)
        
        for i in range(batch_size):
            # Get action-specific prompt
            prompt_i = self.action_prompts[i]  # [M, embed_dim]
            
            # Extract text segments
            segments = self.token_processor.extract_text_segments(text, base_embeddings, i)
            
            # Build new sequence: [start, prompt, middle, end]
            seq_new = torch.cat([
                segments['start'],
                prompt_i,
                segments['middle'],
                segments['end']
            ], dim=0)
            
            # Place in cooperative embeddings tensor
            actual_len = seq_new.size(0)
            coop_embeddings[i, :actual_len, :] = seq_new
        
        return coop_embeddings
    
    def _forward_through_transformer(self, coop_embeddings: torch.Tensor, text: torch.Tensor,
                                   clip_model, device: torch.device) -> torch.Tensor:
        """Forward cooperative embeddings through CLIP transformer.
        
        Args:
            coop_embeddings: Cooperative embeddings with inserted prompts.
            text: Input text tensor for extracting final features.
            clip_model: CLIP model instance.
            device: Target device for tensors.
            
        Returns:
            Final text features of shape (batch_size, embed_dim).
        """
        # Add positional embeddings
        clip, _, _ = clip_model
        pos_embed = clip.text.positional_embedding.to(device)
        coop_embeddings = coop_embeddings + pos_embed.unsqueeze(0)
        
        # Forward through transformer
        x = coop_embeddings.permute(1, 0, 2).type(next(clip.text.parameters()).dtype)
        x = clip.text.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip.text.ln_final(x)
        
        # Extract final features using text argmax
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        
        return x.type(self.action_prompts.dtype)


class ActionSimilarityComputer:
    """Computes similarity between visual and text features."""
    
    def __init__(self, logits_scale: nn.Module, cos_sim: nn.Module):
        self.logits_scale = logits_scale
        self.cos_sim = cos_sim
    
    def compute_training_logits(self, vis_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        """Compute logits for training with temperature scaling.
        
        Args:
            vis_feat: Visual feature tensor.
            txt_feat: Text feature tensor.
            
        Returns:
            Scaled logits tensor of shape (batch_size, num_classes).
        """
        vis_feat_norm = vis_feat / torch.norm(vis_feat, dim=-1, keepdim=True)
        txt_feat_norm = txt_feat / torch.norm(txt_feat, dim=-1, keepdim=True)
        
        logits = self.logits_scale(vis_feat_norm @ txt_feat_norm.t())
        return logits.reshape(-1, txt_feat.shape[0])
    
    def compute_inference_similarity(self, vis_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        """Compute similarity for inference using cosine similarity.
        
        Args:
            vis_feat: Visual feature tensor.
            txt_feat: Text feature tensor.
            
        Returns:
            Normalized similarity scores in range [0, 1].
        """
        score = self.cos_sim(vis_feat, txt_feat)
        return (score + 1) / 2  # Normalize to [0, 1]


class Action_Model(nn.Module):
    """
    Action Model for learning action representations through contrastive learning.
    
    This model learns to associate visual segment features with text representations
    using learnable prompts and CLIP-based encoding.
    """
    
    def __init__(self, vis_input_dim: int, txt_input_dim: int, feat_dim: int,
                 num_actions: int, length_prompt: int = 8):
        super().__init__()
        
        # Store dimensions
        self.vis_input_dim = vis_input_dim
        self.txt_input_dim = txt_input_dim
        self.num_actions = num_actions
        self.feat_dim = feat_dim
        
        # Initialize model components
        self._init_learnable_parameters(num_actions, length_prompt, txt_input_dim,
                                       vis_input_dim, feat_dim)
        self._init_similarity_components()
        
        # Initialize helper classes
        self.text_encoder = CooperativeTextEncoder(self.action_prompts)
        self.similarity_computer = ActionSimilarityComputer(self.logits_scale, self.cos_sim)
    
    def _init_learnable_parameters(self, num_actions: int, length_prompt: int, txt_input_dim: int,
                                  vis_input_dim: int, feat_dim: int):
        """Initialize learnable parameters for action prompts and projections."""
        # Action prompts
        self.action_prompts = nn.Parameter(
            torch.randn(num_actions, length_prompt, txt_input_dim)
        )
        
        # Projection matrices
        vis_scale = vis_input_dim ** -0.5
        txt_scale = txt_input_dim ** -0.5
        
        self.vis_proj = nn.Parameter(vis_scale * torch.randn(vis_input_dim, feat_dim))
        self.txt_proj = nn.Parameter(txt_scale * torch.randn(txt_input_dim, feat_dim))
    
    def _init_similarity_components(self):
        """Initialize components for similarity computation."""
        self.logits_scale = Exp_Scale(np.log(1 / 0.07))
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of model parameters.
        
        Returns:
            Data type of action_prompts parameter.
        """
        return self.action_prompts.dtype
    
    def forward(self, clip_model, segment_vis_feature: torch.Tensor, action_tokens: torch.Tensor,
                action_labels: List[torch.Tensor], device: str = "cuda") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass for training.
        
        Args:
            clip_model: CLIP model instance for text and visual encoding.
            segment_vis_feature: Visual features for action segments.
            action_tokens: Text tokens representing action classes.
            action_labels: List of ground truth action label tensors.
            device: Target device for computation (default: "cuda").
            
        Returns:
            Tuple containing focal loss tensor and dictionary of top-k accuracy metrics.
        """
        # Encode text and visual features
        encoded_features = self._encode_features(
            clip_model, segment_vis_feature, action_tokens
        )
        
        # Prepare labels
        labels = self._prepare_labels(action_labels, device)
        
        # Compute logits and loss
        logits = self.similarity_computer.compute_training_logits(
            encoded_features['visual'], encoded_features['text']
        )
        
        loss = sigmoid_focal_loss(logits, labels['onehot'], reduction="sum")
        
        # Compute accuracy metrics
        topk_acc = topk_accuracy(
            labels['class_ids'].clone(), logits.clone(),
            topk=(1, 3, 5), name="action"
        )
        
        return loss, topk_acc

    @torch.no_grad()
    def inference(self, clip_model, segment_vis_feature: torch.Tensor, action_tokens: torch.Tensor,
                  action_labels: List[torch.Tensor], device: str = "cuda") -> torch.Tensor:
        """Inference pass for computing action similarities.
        
        Args:
            clip_model: CLIP model instance for text and visual encoding.
            segment_vis_feature: Visual features for action segments.
            action_tokens: Text tokens representing action classes.
            action_labels: List of action label tensors for segments.
            device: Target device for computation (default: "cuda").
            
        Returns:
            Similarity scores tensor between visual features and corresponding text features.
        """
        # Encode features
        encoded_features = self._encode_features(
            clip_model, segment_vis_feature, action_tokens
        )
        
        # Get segment-specific text features
        clip_action_labels = torch.cat(action_labels, dim=0).to(device)
        segment_text_feat = encoded_features['text'][clip_action_labels]
        
        # Compute similarity
        return self.similarity_computer.compute_inference_similarity(
            encoded_features['visual'], segment_text_feat
        )

    def _encode_features(self, clip_model, segment_vis_feature: torch.Tensor,
                        action_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode visual and text features through projection layers.
        
        Args:
            clip_model: CLIP model instance for text encoding.
            segment_vis_feature: Visual features for segments.
            action_tokens: Text tokens for actions.
            
        Returns:
            Dictionary containing 'visual' and 'text' projected feature tensors.
        """
        text_features = clip_model[0].encode_text(action_tokens)
        text_features = text_features @ self.txt_proj
        visual_features = segment_vis_feature @ self.vis_proj
        
        return {
            'visual': visual_features,
            'text': text_features
        }
    
    def _prepare_labels(self, action_labels: List[torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        """Prepare labels for loss computation.
        
        Args:
            action_labels: List of action label tensors from different segments.
            device: Target device for tensors.
            
        Returns:
            Dictionary containing 'class_ids' (merged label tensor) and 'onehot' (one-hot encoded labels).
        """
        # Merge labels across batches
        clip_action_labels = torch.cat(action_labels, dim=0).to(device)
        
        # Create one-hot encoding
        onehot_labels = F.one_hot(clip_action_labels, num_classes=self.num_actions).float()
        
        return {
            'class_ids': clip_action_labels,
            'onehot': onehot_labels
        }