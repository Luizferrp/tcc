import torch
import torch.nn as nn

# Bloco básico: Inverted Residual Block (MobileNetV2)
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)

        layers = []
        # 1x1 expansão
        if expansion_factor != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # Projeção
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Modelo MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Primeira camada
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),  # (128 → 64)
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Blocos Inverted Residual (config do paper simplificada)
        self.blocks = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 320, 1, 6),
        )

        # Camada final
        self.final = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # (1280, 1, 1)
        )

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)