import torch
import torch.nn.functional as F
from torch import nn

from .mae_vit import mae_vit_base_patch16, mae_vit_large_patch16

from .bert_backbone import BertModel
import numpy as np


class ModalityUnifiedFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        """ Initializes the model."""
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_layer = cfg.MODEL.BACKBONE.FUSION_LAYER
        self.cont_loss_layer = cfg.MODEL.BACKBONE.CONT_LOSS_LAYER
        self.txt_token_mode = cfg.MODEL.BACKBONE.TXT_TOKEN_MODE
        
        
        
        
        
        
        
        
        
        
        
        if 'base' in cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            self.vit = mae_vit_base_patch16(img_size=(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE), 
                                            learnable_pos=cfg.MODEL.LEARNABLE_POSITION,
                                            drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
            self.vit.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_PATH, map_location='cpu')['model'], strict=False)

            # Text feature encoder (BERT)
            self.bert = BertModel.from_pretrained(cfg.MODEL.BACKBONE.LANGUAGE.TYPE)
            self.bert.encoder.layer = self.bert.encoder.layer[:min(self.fusion_layer)]
            
        elif 'large' in cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            self.vit = mae_vit_large_patch16(img_size=(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE), 
                                            learnable_pos=cfg.MODEL.LEARNABLE_POSITION,
                                            drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
            self.vit.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_PATH, map_location='cpu')['model'], strict=False)

            # Text feature encoder (BERT)
            self.bert = BertModel.from_pretrained(cfg.MODEL.BACKBONE.LANGUAGE.TYPE)
            self.bert.encoder.layer = self.bert.encoder.layer[:min(self.fusion_layer)]
            
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

    def cat_mask(self, text, flag):
        x_mask = torch.ones([flag.shape[0], self.vit.num_patches_x]).to(flag.device)
        z_mask = torch.ones([flag.shape[0], self.vit.num_patches_z]).to(flag.device)*(flag!=1) # =1 mask
        c_mask = torch.ones([flag.shape[0], 1]).to(flag.device)*(flag!=1) # =1 mask
        t_mask = text.mask*(flag!=0) # =0 mask
        mask = ~torch.cat([c_mask, z_mask, x_mask, t_mask], dim=1).bool()
        visual_mask = ~torch.cat([c_mask, z_mask, x_mask], dim=1).bool()
        return mask, visual_mask

    def forward(self, template, search, text, flag): # one more token 
        img_feat = self.vit.patchify(template, search) #16 321 768 视觉embeding
        txt_feat, bert_mask = self.bert.embedding(text.tensors, token_type_ids=None, attention_mask=text.mask) # 文本embdeding txt_feat:16 40 768 bert_mask: 16 1 1 40
        mask, visual_mask = self.cat_mask(text, flag)#mask:16 361  visual_mask:16 321
        logits_list = []
        for i in range(len(self.vit.blocks)):#12层，前6层单独提取特征，后6层特征融合
            if i in self.fusion_layer:#6 7 8 9 11
                img_feat, txt_feat = self.vit.forward_joint(img_feat, txt_feat, mask, i, flag=flag)
            else:
                img_feat = self.vit.blocks[i](img_feat, visual_mask, flag=flag)#16 321 768 用VIT对视觉特征进行编码
                txt_feat = self.bert.encoder.layer[i](txt_feat, bert_mask)#16 40 768 用BERT 对文本特征进行编码
            if i in self.cont_loss_layer:#3 4 5 6 7 8 9 10 11
                logits = self.contractive_learning(img_feat, txt_feat, text, flag)#16 256 1 对比学习损失，应该是每个visual patch和对应向量的相似度 对于got-10k是 visual patch 和 总的视觉向量的相似度
                logits_list.append(logits)
        vis_token, z, x = img_feat.split([1, self.vit.num_patches_z, self.vit.num_patches_x], dim=1)
        b, s, c = x.shape#1.提取视觉（搜索、模板）和文本特征 2. 然后计算搜索图像和整体视觉特征和文本特征相似度
        out_dict = {
            "search": x,#1 256 768
            "template": z,#1 64 768
            "text": txt_feat,#1 40 768
            "vis_token": vis_token,#1 1 768
            "txt_token": self.generate_txt_token(txt_feat, text),#1 1 768
            "flag": flag.reshape(-1),#2
            "logits": torch.stack(logits_list, dim=1).reshape(b, -1, int(s**0.5), int(s**0.5))#1 9 16 16
        }
        return out_dict
    
    def generate_txt_token(self, txt_feat, text):
        if self.txt_token_mode == 'mean':
            return (txt_feat*text.mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / text.mask.unsqueeze(-1).sum(dim=1, keepdim=True)
        elif self.txt_token_mode == 'cls':
            return txt_feat[:, :1]
    
    def contractive_learning(self, img_feat, txt_feat, text, flag):
        vis_token, z, x = img_feat.split([1, self.vit.num_patches_z, self.vit.num_patches_x], dim=1)#计算可视化token 模板token 和 search token
        txt_token = self.generate_txt_token(txt_feat, text)# 计算文本token 
        vis_logits = self.logit_scale.exp()*(F.normalize(x, dim=-1) @ F.normalize(vis_token, dim=-1).transpose(-2,-1))#计算搜索图像和可视化token相似度
        txt_logits = self.logit_scale.exp()*(F.normalize(x, dim=-1) @ F.normalize(txt_token, dim=-1).transpose(-2,-1))#计算搜索图像和文本token相似度
        logits_group = torch.stack([vis_logits, txt_logits, (vis_logits+txt_logits)/2], dim=1)
        bid = torch.arange(flag.shape[0])
        logits = logits_group[bid, flag.reshape(-1)]
        return logits
        
        

def modality_unified_feature_extractor(cfg):
    model = ModalityUnifiedFeatureExtractor(cfg)
    return model
