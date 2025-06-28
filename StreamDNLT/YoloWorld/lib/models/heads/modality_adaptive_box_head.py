import sys
sys.path.append('/ssd/myc/VL_project/MUTrack')

import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.models.heads.utils import conv, DistributionBasedCrossAttention
import numpy as np

class ModalityAdaptiveBoxHead(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False,
                 num_t=2, num_b=4, offset_sigmoid=True, cls_tokenize=True, joint_cls=False, 
                 drop_rate=0.0, softmax_one=False, grounding_dilation=1, contrastive_conv=False):
        super(ModalityAdaptiveBoxHead, self).__init__()
        self.num_t = num_t
        self.num_b = num_b
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.offset_sigmoid = offset_sigmoid
        self.cls_tokenize = cls_tokenize
        self.joint_cls = joint_cls
        self.softmax_one = softmax_one
        self.contrastive_conv = contrastive_conv

        self.conv_cls = nn.Sequential(conv(inplanes, channel), 
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 1, kernel_size=1))

        self.conv_offset = nn.Sequential(conv(inplanes, channel), 
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 2, kernel_size=1))

        self.conv_bbox = nn.Sequential(conv(inplanes, channel), 
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 2, kernel_size=1))

        self.conv_bbox_grounding = nn.Sequential(conv(inplanes, channel), 
                                        conv(channel, channel // 2),
                                        conv(channel // 2, channel // 4),
                                        conv(channel // 4, channel // 8),
                                        nn.Conv2d(channel // 8, 2, kernel_size=1))

        self.prompter = DistributionBasedCrossAttention(inplanes, drop_rate=drop_rate)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        x, y = torch.arange(0, self.feat_sz), torch.arange(0, self.feat_sz)
        x, y = torch.meshgrid(x, y)
        if offset_sigmoid:
            coodinate = torch.cat([y.reshape(-1)[None, :], x.reshape(-1)[None, :]])[None] # b, 2, s
        else:
            coodinate = torch.cat([y.reshape(-1)[None, :], x.reshape(-1)[None, :]])[None] + 0.5
        self.register_buffer("coodinate", coodinate)

    def forward(self, out_dict, test=False):
        # top-left branch 回归一个cls，offset，size，现在对cls进行一个修正了
        flag = out_dict['flag']# 形1数2
        vis_token = out_dict['vis_token']#1 1 768
        txt_token = out_dict['txt_token']#1 1 768
        token_group = torch.cat([vis_token, txt_token, (vis_token+txt_token)/2], dim=1)#1 3 768
        bid = torch.arange(flag.shape[0])#0 
        token = token_group[bid, flag, ..., None, None]#1 768 1 1
        
        cont_score, prompts = self.contractive_learning(out_dict)
        b = out_dict['search'].shape[0]
        x = out_dict['search'].transpose(-2, -1).reshape(b, -1, self.feat_sz, self.feat_sz).contiguous()#1 768 16 16
        cls_map = self.conv_cls(x*token).sigmoid().squeeze(1) if self.cls_tokenize else self.conv_cls(x).sigmoid().squeeze(1)#1 16 16 分类分数
        offset_map = self.conv_offset(x).sigmoid() if self.offset_sigmoid else self.conv_offset(x)#1 2 16 16 偏移量
        size_map_tr = self.conv_bbox(x).sigmoid().unsqueeze(1)#1 1 2 16 16 回归尺寸
        size_map_gr = self.conv_bbox_grounding(x).sigmoid().unsqueeze(1)#1 1 2 16 16
        
        flag = out_dict['flag']
        size_map_group = torch.cat([size_map_tr, size_map_gr, size_map_tr], dim=1)#1 3 2 16 16
        bid = torch.arange(flag.shape[0])
        size_map = size_map_group[bid, flag]#1 2 16 16
        
        bbox_map, bbox = self.convert2bbox(cls_map, offset_map, size_map, cont_score)#bbox_map：1 256 4 bbox：1 1 4
        cont_score_2d = cont_score.softmax(-1)[..., 0].reshape(-1, *cls_map.shape[1:])#1 16 16
        out_dict.update({
            "cls_score": (cls_map * cont_score_2d) if self.joint_cls else cls_map,
            "bbox_map": bbox_map,
            "pred_boxes": bbox,
            "cont_score": cont_score,
            "prompts": prompts,
            "cls_score_test": cls_map
        })
        return out_dict
    
    def forward_prompt(self, out_dict):
        flag = out_dict['flag']
        vis_token = out_dict['vis_token']
        txt_token = out_dict['txt_token']
        token_group = torch.cat([vis_token, txt_token, (vis_token+txt_token)/2], dim=1)
        bid = torch.arange(flag.shape[0])
        token = token_group[bid, flag]
        prompt = self.prompter(out_dict['template'], out_dict['template_mask'], 
                                out_dict['search'], out_dict['context_mask'], 
                                token, out_dict['flag'])#1 3 768
        return prompt
        
    def convert2bbox(self, cls_map, offset_map, size_map, cont_score):
        b = cls_map.shape[0]
        cls_map = cls_map.reshape(b, -1) * cont_score.softmax(-1)[:, :, 0]
        b_idx = torch.arange(0, b)
        s_idx = torch.argmax(cls_map, dim=-1)#单目标检测，取概率最大的值作为系数了 用一个patch来预测了
        offset_map = offset_map.reshape(b, 2, -1)#1 2 256
        size_map   = size_map.reshape(b, 2, -1)#1 2 256
        ctr_map = self.coodinate.repeat(b, 1, 1) # 1，2，256
        ctr_map = (ctr_map+offset_map) / self.feat_sz#1 2 256
        bbox_map = torch.cat([ctr_map, size_map], dim=1).transpose(-2, -1) # 1 256 4
        bbox = bbox_map[b_idx, s_idx]#1 4
        return bbox_map, bbox.unsqueeze(1)

    def contractive_learning(self, out_dict):
        # for training
        if out_dict.get('prompt', None) is None:
            flag = out_dict['flag']
            vis_token = out_dict['vis_token']
            txt_token = out_dict['txt_token']
            token_group = torch.cat([vis_token, txt_token, (vis_token+txt_token)/2], dim=1)
            bid = torch.arange(flag.shape[0])
            token = token_group[bid, flag]
            search = out_dict['search']
            context = torch.cat([search[bid.shape[0]//2:], search[:bid.shape[0]//2]], dim=0)
            prompt = self.prompter(out_dict['template'], out_dict['template_mask'], context, out_dict['context_mask'], token, out_dict['flag']) # 16 3 768
            cont_score = self.logit_scale.exp() * (F.normalize(out_dict['search'], dim=-1) @ F.normalize(prompt, dim=-1).transpose(-2,-1)) #16 256 3
            if self.softmax_one:
                ext_one = torch.zeros_like(cont_score[:, :, :1])
                cont_score = torch.cat([cont_score[:, :, :1], torch.cat([cont_score[:, :, 1:], ext_one], dim=-1).max(dim=-1, keepdim=True).values], dim=-1)#16 256 2
            else:
                cont_score = torch.cat([cont_score[:, :, :1], cont_score[:, :, 1:].max(dim=-1, keepdim=True).values], dim=-1)
        # for testing
        else:
            prompt = out_dict['prompt']#1 3 768
            cont_score = self.logit_scale.exp() * (F.normalize(out_dict['search'], dim=-1) @ F.normalize(prompt, dim=-1).transpose(-2,-1))#1 256 3 计算搜索图像每个patch 和 prompt：目标、背景、干扰物的相似度
            
            if self.softmax_one:
                ext_one = torch.zeros_like(cont_score[:, :, :1])
                cont_score = torch.cat([cont_score[:, :, :1], torch.cat([cont_score[:, :, 1:], ext_one], dim=-1).max(dim=-1, keepdim=True).values, ext_one], dim=-1)#1 256 3
            else:
                cont_score = torch.cat([cont_score[:, :, :1], cont_score[:, :, 1:].max(dim=-1, keepdim=True).values], dim=-1)

        return cont_score, prompt