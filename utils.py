import torch.nn as nn

class EvidenceExtractor(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softplus()  # ensures positive evidence
        )

    def forward(self, image_features):
        return self.mlp(image_features)