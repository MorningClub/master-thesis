from matplotlib.cbook import flatten
import torch
from torch import nn
from common.resnet import resnet18, resnet34
from common.normalize import Normalize
from common.segmentation import SegmentationHead

class CameraModelA(nn.Module):
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

        self.backbone_wide = resnet34(pretrained=config['imagenet_pretrained'])
        self.backbone_wide.load_state_dict(torch.load("/encoder_models/encoder_model_12.th"))
        for param in self.backbone_wide.parameters():
           param.requires_grad = False

        self.latent_space_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 64, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(896, 512),
            nn.ReLU(True)
        )

        if self.two_cam:
            self.backbone_narr = resnet18(pretrained=config['imagenet_pretrained'])
            self.seg_head_narr = SegmentationHead(512, self.num_labels+1)
            self.bottleneck_narr = nn.Sequential(
                nn.Linear(512,64),
                nn.ReLU(True),
            )

        if self.all_speeds:
            self.num_acts = self.num_cmds*self.num_speeds*(self.num_steers+self.num_throts+1)
        else:
            self.num_acts = self.num_cmds*(self.num_steers+self.num_throts+1)
            self.spd_encoder = nn.Sequential(
                nn.Linear(1,64),
                nn.ReLU(True),
                nn.Linear(64,64),
                nn.ReLU(True),
            )

        self.act_head = nn.Sequential(
            nn.Linear(512 + (0 if self.all_speeds else 64) + (64 if self.two_cam else 0),256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,self.num_acts),
        )

    def forward(self, wide_rgb, narr_rgb, spd=None):
        
        assert (self.all_speeds and spd is None) or \
               (not self.all_speeds and spd is not None)

        wide_embed = self.backbone_wide(wide_rgb/255.)
        
        space_interpretation = self.latent_space_head(wide_embed)
        
        act_output = self.act_head(space_interpretation).view(-1,self.num_cmds,self.num_speeds,self.num_steers+self.num_throts+1)

        act_output = action_logits(act_output, self.num_steers, self.num_throts)

        return act_output


    @torch.no_grad()
    def policy(self, wide_rgb, narr_rgb, cmd, spd=None):
        
        assert (self.all_speeds and spd is None) or \
               (not self.all_speeds and spd is not None)
        
        wide_embed = self.backbone_wide(wide_rgb/255.)

        space_interpretation = self.latent_space_head(wide_embed)

        act_output = self.act_head(space_interpretation).view(-1,self.num_cmds,self.num_speeds,self.num_steers+self.num_throts+1)
        # Action logits
        steer_logits = act_output[0,cmd,:,:self.num_steers]
        throt_logits = act_output[0,cmd,:,self.num_steers:self.num_steers+self.num_throts]
        brake_logits = act_output[0,cmd,:,-1]

        return steer_logits, throt_logits, brake_logits


def action_logits(raw_logits, num_steers, num_throts):
    
    steer_logits = raw_logits[...,:num_steers]
    throt_logits = raw_logits[...,num_steers:num_steers+num_throts]
    brake_logits = raw_logits[...,-1:]
    
    steer_logits = steer_logits.repeat(1,1,1,num_throts)
    throt_logits = throt_logits.repeat_interleave(num_steers,-1)
    
    act_logits = torch.cat([steer_logits + throt_logits, brake_logits], dim=-1)
    
    return act_logits
