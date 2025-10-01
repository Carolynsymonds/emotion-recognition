"""
Model definitions for emotion recognition experiments.
This module contains CLIP-based models and custom architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math
from typing import List, Optional, Tuple


class CLIPEmotionClassifier(nn.Module):
    """
    CLIP-based emotion classifier with frozen backbone and trainable last layer.
    """
    
    def __init__(self, model_name: str = "ViT-B/16", num_classes: int = 7, 
                 freeze_backbone: bool = True, trainable_layers: int = 1):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.trainable_layers = trainable_layers
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Set fixed temperature
        self._set_fixed_temperature()
        
    def _freeze_backbone(self):
        """Freeze CLIP backbone parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the last transformer block(s)
        if hasattr(self.clip_model.visual, 'transformer'):
            for i in range(-self.trainable_layers, 0):
                for param in self.clip_model.visual.transformer.resblocks[i].parameters():
                    param.requires_grad = True
    
    def _set_fixed_temperature(self, tau: float = 0.07):
        """Set fixed temperature for CLIP."""
        with torch.no_grad():
            self.clip_model.logit_scale.fill_(math.log(1.0 / tau))
        self.clip_model.logit_scale.requires_grad = False
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP visual encoder."""
        return self.clip_model.encode_image(images)
    
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode text using CLIP text encoder."""
        text_tokens = clip.tokenize(text)
        return self.clip_model.encode_text(text_tokens)
    
    def forward(self, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for emotion classification.
        
        Args:
            images: Input images [B, C, H, W]
            text_features: Pre-computed text features [num_classes, D]
            
        Returns:
            Logits [B, num_classes]
        """
        # Encode images
        image_features = self.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        scale = self.clip_model.logit_scale.exp()
        logits = scale * torch.matmul(image_features, text_features.t())
        
        return logits


class MultiFeatureDecoder(nn.Module):
    """
    Multi-Feature Decoupling module for combining global and local features.
    """
    
    def __init__(self, feature_dim: int, num_classes: int, k: int = 16, gamma: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.k = k
        self.gamma = gamma
        
    def compute_affinities(self, v_cls: torch.Tensor, v_patches: torch.Tensor, 
                          text_features: torch.Tensor, temp: float = 0.07) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global and local affinities.
        
        Args:
            v_cls: Global CLS token features [B, D]
            v_patches: Local patch features [B, N, D]
            text_features: Text features [C, D]
            temp: Temperature parameter
            
        Returns:
            Tuple of (global_affinities, local_affinities)
        """
        # Normalize embeddings
        v_cls = F.normalize(v_cls, dim=-1)
        v_patches = F.normalize(v_patches, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Global affinities
        logits_global = torch.matmul(v_cls, text_features.t()) / temp
        affinity_global = F.softmax(logits_global, dim=-1)
        
        # Local affinities
        logits_local = torch.matmul(v_patches, text_features.t()) / temp
        affinity_local = F.softmax(logits_local, dim=-1)
        
        return affinity_global, affinity_local
    
    def topk_local_affinity(self, affinity_local: torch.Tensor) -> torch.Tensor:
        """
        Extract top-K local affinities.
        
        Args:
            affinity_local: Local affinities [B, N, C]
            
        Returns:
            Top-K local affinities [B, C]
        """
        topk_values, _ = torch.topk(affinity_local, k=self.k, dim=1)
        affinity_local_decoupled = topk_values.mean(dim=1)
        return affinity_local_decoupled
    
    def combine_affinities(self, affinity_global: torch.Tensor, 
                          affinity_local_decoupled: torch.Tensor) -> torch.Tensor:
        """
        Combine global and local affinities.
        
        Args:
            affinity_global: Global affinities [B, C]
            affinity_local_decoupled: Local affinities [B, C]
            
        Returns:
            Combined affinities [B, C]
        """
        combined_affinity = (self.gamma * affinity_global + 
                           (1 - self.gamma) * affinity_local_decoupled)
        return combined_affinity
    
    def forward(self, v_cls: torch.Tensor, v_patches: torch.Tensor, 
                text_features: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
        """
        Forward pass for multi-feature decoupling.
        
        Args:
            v_cls: Global CLS token features [B, D]
            v_patches: Local patch features [B, N, D]
            text_features: Text features [C, D]
            temp: Temperature parameter
            
        Returns:
            Combined logits [B, C]
        """
        # Compute affinities
        affinity_global, affinity_local = self.compute_affinities(
            v_cls, v_patches, text_features, temp
        )
        
        # Extract top-K local affinities
        affinity_local_decoupled = self.topk_local_affinity(affinity_local)
        
        # Combine affinities
        combined_affinity = self.combine_affinities(affinity_global, affinity_local_decoupled)
        
        return combined_affinity


class EvidenceExtractor(nn.Module):
    """
    Evidence extractor for uncertainty quantification using Evidential Deep Learning.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, 
                 max_evidence: float = 50.0):
        super().__init__()
        self.max_evidence = max_evidence
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(beta=1.0, threshold=20.0)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters for low initial evidence."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, -2.0)  # Negative bias for low initial evidence
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for evidence extraction.
        
        Args:
            x: Input features [B, D]
            
        Returns:
            Evidence values [B, C]
        """
        evidence = self.mlp(x)
        return torch.clamp(evidence, max=self.max_evidence)


class UncertaintyQuantifier(nn.Module):
    """
    Uncertainty quantification module using Evidential Deep Learning.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_extractor = EvidenceExtractor(input_dim, num_classes, hidden_dim)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for uncertainty quantification.
        
        Args:
            features: Input features [B, D]
            
        Returns:
            Tuple of (predictions, uncertainty, evidence)
        """
        evidence = self.evidence_extractor(features)
        
        # Compute Dirichlet parameters
        alpha = evidence + 1.0
        
        # Compute predictions (expected probability)
        predictions = alpha / alpha.sum(dim=-1, keepdim=True)
        
        # Compute uncertainty (inverse of total evidence)
        total_evidence = alpha.sum(dim=-1)
        uncertainty = self.num_classes / total_evidence
        
        return predictions, uncertainty, evidence


def build_text_features_simple(emotions: List[str], model: nn.Module, device: torch.device) -> torch.Tensor:
    """
    Build simple text features for emotion classes.
    
    Args:
        emotions: List of emotion class names
        model: CLIP model
        device: Device to place features on
        
    Returns:
        Text features tensor [num_classes, feature_dim]
    """
    with torch.no_grad():
        text_features = model.encode_text(emotions)
        text_features = F.normalize(text_features, dim=-1)
    return text_features.to(device)


def build_text_features_mean(prompts: List[str], model: nn.Module, device: torch.device, 
                           emotions: List[str]) -> torch.Tensor:
    """
    Build text features using mean of multiple prompts per emotion.
    
    Args:
        prompts: List of prompt templates
        model: CLIP model
        device: Device to place features on
        emotions: List of emotion class names
        
    Returns:
        Text features tensor [num_classes, feature_dim]
    """
    all_features = []
    
    with torch.no_grad():
        for emotion in emotions:
            emotion_prompts = [prompt.format(emotion) for prompt in prompts]
            features = model.encode_text(emotion_prompts)
            features = F.normalize(features, dim=-1)
            mean_features = features.mean(dim=0)
            all_features.append(mean_features)
    
    return torch.stack(all_features).to(device)


def by_class_prompt(emotions: List[str]) -> List[str]:
    """
    Generate class-specific prompts for emotions.
    
    Args:
        emotions: List of emotion class names
        
    Returns:
        List of prompt templates
    """
    prompts = [
        "a photo of a {} person",
        "a {} facial expression",
        "someone looking {}",
        "a {} face",
        "an image showing {} emotion"
    ]
    return prompts
