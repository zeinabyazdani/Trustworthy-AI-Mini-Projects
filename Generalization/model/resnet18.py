
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Implement a ResNet Block with optional Batch Normalization.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels or conv feature maps
        first_stride (int): stride of first convolution layer in resnet block (1: no change in input size, 2: downsampling)
        use_bn (bool): whether to use Batch Normalization or not.

    """

    def __init__(self, in_channels: int, out_channels: int, first_stride: int = 1, use_bn: bool = True):
        super().__init__()

        # In Batch Normalization, there is no need for a bias term in Conv2d layers because BatchNorm itself includes a bias parameter (β).
        # However, if BatchNorm is removed, bias=True should be used so that the model can learn the bias properly.
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=first_stride, padding=1, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=not use_bn))
        if use_bn: layers.append(nn.BatchNorm2d(out_channels))
        self.res_block = nn.Sequential(*layers)

        self.relu = nn.ReLU(inplace=True)
        
        # Skip connections: If downsampling occurs, the input and output sizes of the ResNet block are different. 
        # To match their dimensions and enable summation, we use a 1x1 convolution.
        # First, we check whether downsampling has occurred or not.
        self.skip_connection = nn.Sequential()
        if first_stride != 1 or in_channels != out_channels:
            skip_layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=first_stride, padding=0, bias=not use_bn)]
            if use_bn: skip_layers.append(nn.BatchNorm2d(out_channels))
            self.skip_connection = nn.Sequential(*skip_layers)


    def forward(self, x):
        return self.relu(self.res_block(x) + self.skip_connection(x))


class ResNet18(nn.Module):
    """Implement ResNet18.

    Args:
        in_channels (int): number of input channels
        n_classes (int): number of output classes or targets
        use_bn (bool): whether to use Batch Normalization or not.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10, use_bn: bool=True):
        super().__init__()


        layers = [nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block1 = nn.Sequential(*layers)

        self.block2 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64, first_stride=1, use_bn=use_bn),
            ResBlock(in_channels=64, out_channels=64, first_stride=1, use_bn=use_bn)
        )

        self.block3 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=128, first_stride=2, use_bn=use_bn),
            ResBlock(in_channels=128, out_channels=128, first_stride=1, use_bn=use_bn)
        )

        self.block4 = nn.Sequential(
            ResBlock(in_channels=128, out_channels=256, first_stride=2, use_bn=use_bn),
            ResBlock(in_channels=256, out_channels=256, first_stride=1, use_bn=use_bn)
        )

        self.block5 = nn.Sequential(
            ResBlock(in_channels=256, out_channels=512, first_stride=2, use_bn=use_bn),
            ResBlock(in_channels=512, out_channels=512, first_stride=1, use_bn=use_bn)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # x = self.out(x)
        
        return x


if __name__ == "__main__":
    
    from torchsummary import summary
    from torchvision.models import resnet18

    model_torchvision = resnet18(num_classes=10)
    model_mine = ResNet18()

    x = torch.randn(1, 3, 224, 224)

    output_mine = model_mine(x)
    output_torchvision = model_torchvision(x)

    print(output_mine.shape, output_torchvision.shape)
    print("Difference:", torch.norm(output_mine - output_torchvision))

    # summary(model_mine, (3, 224, 224))
    # summary(model_mine, (3, 512, 512), batch_size=1)
    # summary(model_torchvision, (3, 224, 224))