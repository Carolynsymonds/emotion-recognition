import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

class ResNeXtEmotionClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNeXtEmotionClassifier, self).__init__()

        # Load model with recommended weight usage
        weights = ResNeXt50_32X4D_Weights.DEFAULT
        model = resnext50_32x4d(weights=weights)

        # Replace the final classification layer with a new one tailored to our number of classes
        self.backbone = model
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Define your classifier
class CLIPEmotionClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=8):
        super().__init__()
        self.clip_visual = clip_model.vision_model
        self.pool = clip_model.visual_projection  # Keeps alignment with original CLIP
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        vision_outputs = self.clip_visual(pixel_values=pixel_values)
        pooled_output = vision_outputs[1]  # CLS token embedding
        projected = self.pool(pooled_output)
        return self.classifier(projected)