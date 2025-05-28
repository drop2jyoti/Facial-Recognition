import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
import torch.nn.functional as F

class FaceNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(FaceNet, self).__init__()
        # Use ResNet50 as backbone with VGGFace2 pre-trained weights
        self.backbone = models.resnet50(pretrained=False)  # We'll load VGGFace2 weights
        
        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add a new fully connected layer for face embeddings
        self.fc = nn.Linear(2048, 128)  # 128-dimensional face embedding
        
        if pretrained:
            # Load VGGFace2 pre-trained weights
            self._load_vggface2_weights()
    
    def _load_vggface2_weights(self):
        """Load pre-trained VGGFace2 weights"""
        try:
            # Try to load from torch hub
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/ox-vgg/vgg_face2/raw/master/models/resnet50_ft_weight.pth',
                map_location='cpu'
            )
            # Load weights into backbone
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Successfully loaded VGGFace2 pre-trained weights")
        except Exception as e:
            print(f"Could not load VGGFace2 weights: {e}")
            print("Using ImageNet pre-trained weights instead")
            # Fallback to ImageNet weights
            self.backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Get face embeddings
        embeddings = self.fc(features)
        
        # L2 normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Calculate distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Calculate triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        return loss.mean() 