import torch
import torch.nn as nn
from torchvision import models
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

# 모델 설정
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.vgg_features = models.vgg16(pretrained=True)
        self.vgg_features.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.vgg_features.classifier[6] = nn.Linear(4096, 512)
        
        self.gender_layer = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc = nn.Linear(544, 1)  # 512 from VGG and 32 from gender_layer

    def forward(self, image, gender):
        image_features = self.vgg_features(image)
        gender_features = self.gender_layer(gender.float().unsqueeze(1))
        combined_features = torch.cat((image_features, gender_features), dim=1)
        output = self.fc(combined_features)
        return output