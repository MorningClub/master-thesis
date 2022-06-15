import torch
from torch import nn
from common.resnet import resnet18, resnet34, ResNet34Dec
from common.normalize import Normalize
from common.segmentation import SegmentationHead

class Autoencoder(nn.Module):
    def __init__(self, config, num_cmds=6):
        super().__init__()
        
        # Configs
        self.num_cmds   = num_cmds
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']
        self.num_speeds = config['num_speeds']
        self.num_labels = len(config['seg_channels'])
        self.all_speeds = config['all_speeds']
        self.two_cam    = config['use_narr_cam']

        #rgb decoder
        self.decoder = ResNet34Dec()
        self.backbone_wide = resnet34(pretrained=config['imagenet_pretrained'])
        self.seg_head_wide = SegmentationHead(512, self.num_labels+1)

        self.wide_seg_head = SegmentationHead(512, self.num_labels+1)
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, wide_rgb, spd=None):
        
        assert (self.all_speeds and spd is None) or \
               (not self.all_speeds and spd is not None)

        wide_embed = self.backbone_wide(wide_rgb)

        wide_seg_output = self.seg_head_wide(wide_embed)

        decoded_rgb = self.decoder(wide_embed)

        return wide_seg_output, decoded_rgb


