import torch
import torch.nn as nn
from torchvision import models
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
import timm

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


class MultiModalModel_swintrans(nn.Module):
    def __init__(self, model_name = 'swin_base_patch4_window12_384', pretrained=True):
        super(MultiModalModel_swintrans, self).__init__()
        self.backbone = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=0)
        self.in_features = self.backbone.num_features

        self._convert_input_layer_to_grayscale()

        self.combined_regressor = nn.Sequential(
            nn.Linear(self.in_features+32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.gender_layer = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

    def _convert_input_layer_to_grayscale(self):
        patch_embed = self.backbone.patch_embed
        old_conv = patch_embed.proj

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias
        patch_embed.proj = new_conv

    def forward(self, image, gender):
        img_feat = self.backbone(image)

        gender_features = self.gender_layer(gender.float().unsqueeze(1))
        combined_features = torch.cat((img_feat, gender_features), dim=1)
        x = self.combined_regressor(combined_features)
        return x*240