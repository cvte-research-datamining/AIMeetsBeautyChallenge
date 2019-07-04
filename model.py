import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152, resnet101
from efficientnet_pytorch import EfficientNet
import numpy as np

class MultiHeadResNet(nn.Module):
    def __init__(self, model='resnet152', pretrained=True, finetune=False):
        """Declare all needed layers."""
        super(MultiHeadResNet, self).__init__()
        self.finetune = finetune
        if model == 'resnet152':
            self.model = resnet152(pretrained=pretrained)
        elif model == 'resnet101':
            self.model = resnet101(pretrained=pretrained)
        else:
            raise ValueError
        print('Load {:}'.format(model))
        self.model_name = model
        self.last_channels = self.model.fc.weight.shape[1]
        self.out_channels = []
        for idx in range(1, 5):
            self.out_channels.append(getattr(self.model, 'layer{:}'.format(idx))[-1].conv3.weight.shape[0])

    def forward(self, x):
        outputs = []
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            l1 = self.model.layer1(x)
            l2 = self.model.layer2(l1)
            l3 = self.model.layer3(l2)
            l4 = self.model.layer4(l3)

        outputs.extend([l1, l2, l3, l4])
        return outputs


class MultiHeadEffNet(nn.Module):
    def __init__(self, model='efficientnet-b3', pretrained=True):
        """Declare all needed layers."""
        super(MultiHeadEffNet, self).__init__()
        self.NUM_BLOCK = 4
        if model.startswith('efficientnet'):
            if pretrained:
                self.model = EfficientNet.from_pretrained(model)
            else:
                self.model = EfficientNet.from_name(model)
        else:
            raise ValueError
        print('Load {:}'.format(model))
        self.model_name = model

        self.idx_blocks = np.linspace(len(self.model._blocks) - 1, 0, self.NUM_BLOCK, endpoint=False)[::-1]
        self.idx_blocks = list(map(int, self.idx_blocks))

        self.relu = nn.ReLU(inplace=True)
        self.out_channels = []
        for idx in range(self.NUM_BLOCK):
            self.out_channels.append(self.model._blocks[self.idx_blocks[idx]]._project_conv.weight.shape[0])
        self.last_channels = self.out_channels[-1]

    def forward(self, x):
        x = self.relu(self.model._bn0(self.model._conv_stem(x)))

        outputs = []
        for idx, block in enumerate(self.model._blocks):
            with torch.no_grad():
                x = block(x)
                if idx in self.idx_blocks:
                    outputs.append(x)
        assert len(outputs) == 4, 'The number of feature maps should be 4'
        return outputs


class MaskEncoder(nn.Module):
    def __init__(self, in_channels=2048):
        super(MaskEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        out = torch.sigmoid(self.conv2(x))
        return out


class MultiHeadMaskEncoder(nn.Module):
    def __init__(self, model=None, name='efficientnet'):
        super(MultiHeadMaskEncoder, self).__init__()
        self.NUM_BLOCK = 4

        if name.startswith('efficientnet'):
            # Uniformly select block of efficientnet
            idx_blocks = np.linspace(len(model._blocks)-1, 0, self.NUM_BLOCK, endpoint=False)
            idx_blocks = list(map(int, idx_blocks))

            # Initialize the mask encoder and upsample layer
            self.mask_encoder_blocks = nn.ModuleList([])
            self.att_head_blocks = nn.ModuleList([])
            for idx in range(len(idx_blocks)):
                # Mask Encoder
                in_channels = model._blocks[idx_blocks[idx]]._project_conv.weight.shape[0]
                self.mask_encoder_blocks.append(MaskEncoder(in_channels=in_channels))
                if idx < self.NUM_BLOCK - 1:  # Except the last block
                    # Use upsample layer to match the dimension of different block
                    out_channels = model._blocks[idx_blocks[idx+1]]._project_conv.weight.shape[0]
                    self.att_head_blocks.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1)))

        elif name.startswith('resnet'):
            # Initialize the mask encoder and upsample layer
            self.mask_encoder_blocks = nn.ModuleList([])
            self.att_head_blocks = nn.ModuleList([])
            for idx in range(self.NUM_BLOCK, 0, -1):
                in_channels = getattr(model, 'layer{:}'.format(idx))[-1].conv3.weight.shape[0]
                self.mask_encoder_blocks.append(MaskEncoder(in_channels=in_channels))
                if idx > 1: # Except the last block
                    out_channels = getattr(model, 'layer{:}'.format(idx - 1))[-1].conv3.weight.shape[0]
                    self.att_head_blocks.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                                               nn.Conv2d(out_channels, out_channels, 1)))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        masks = []
        att_vectors = []
        for idx in range(self.NUM_BLOCK):
            if idx == 0:    # The first block is no need to use the mask as input.
                input = inputs[::-1][idx]
                mask_encoder_block = self.mask_encoder_blocks[idx]
                masks.append(mask_encoder_block(input))

            else:
                input = inputs[::-1][idx]
                att_input = att_vectors[idx-1] * input
                mask_encoder_block = self.mask_encoder_blocks[idx]
                masks.append(mask_encoder_block(att_input))

            # Attention
            if idx < self.NUM_BLOCK - 1:    # Except the last block
                att_head_block = self.att_head_blocks[idx]
                att_vector = torch.sigmoid(att_head_block(self.attention_pooling(input, masks[idx])))
                att_vectors.append(att_vector)

        return masks[::-1]

    @staticmethod
    def attention_pooling(x, mask):
        z = (mask * x).view(x.size(0), x.size(1), -1).sum(2) / mask.view(mask.size(0), mask.size(1), -1).sum(2)
        return z.unsqueeze(2).unsqueeze(3)