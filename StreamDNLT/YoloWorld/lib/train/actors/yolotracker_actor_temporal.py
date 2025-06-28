from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh, box_xywh_to_cxcywh_scale
import torch
from lib import registry
from lib.utils.box_ops import giou_loss,giou_loss_cost,GaussWeightedLoss
from torch.nn.functional import l1_loss
from lib.utils.misc import NestedTensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt
from .loss.focal_loss import FocalLoss,FocalLossCost
from lib.groundingdino.models.GroundingDINO.constant.constant import MEMORY_SEQENCE

@registry.ACTORS.register("yolotracker_finetune_temporal")
class DINOTrackActor(BaseActor):
    """ Actor for training the TSP_online and TSP_cls_online"""
    def __init__(self, net, cfg):
        super().__init__(net)
        self.cfg = cfg
        self.build_loss(cfg)

    def build_loss(self, cfg):
        weight = torch.tensor([self.cfg.DATA.SEARCH.FACTOR**2, self.cfg.TRAIN.CTR_RATIO**2]).cuda()
        weight = weight / weight.sum()
        self.objective ={'giou': giou_loss, 'l1': l1_loss,'giou_cost':giou_loss_cost,
                          'cls': FocalLoss(alpha=0.25,gamma=2.0),
                          'cls_cost':FocalLossCost(alpha=0.25,gamma=2.0),
                          'class_score':l1_loss}
        self.loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 
                            'cls': 1, 'aux': cfg.TRAIN.AUX_WEIGHT, 
                            'cib': cfg.TRAIN.CIB_WEIGHT, 'cont': cfg.TRAIN.CONT_WEIGHT}

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        
    
    
    
    
    
    









        
        loss_dic = self.forward_pass(data)
        

        
        
        
        
        
        
        
        
        loss= loss_dic["loss_cls"]+self.loss_weight['giou'] * loss_dic["loss_bbox"] + self.loss_weight['l1'] * loss_dic["loss_dfl"]


        return loss, loss_dic



    def forward_pass(self, data):
        _, b, _, ht, wt = data['template_images'].shape 
        n, b, _, hs, ws = data['search_images'].shape 
        template_images = data['template_images'].repeat(n, 1, 1, 1, 1).reshape(n*b, 3, ht, wt)
        search_images = data['search_images'].reshape(n*b, 3, hs, ws)
        search_frames_data_samples=data["search_frames_data_samples"]
        
        search_frames_data_samples = [item for sublist in search_frames_data_samples for item in sublist]

        
        
        
        
        
        data_batch = dict(inputs=search_images,
                      data_samples=search_frames_data_samples,
                      mode = 'loss')
    
    


   


        data_batch = self.net.data_preprocessor(data_batch,False)
        
        
        data_batch_template = template_images
    


   
        
        
        
        
        
        
        data_batch = dict(inputs=search_images,
                      data_samples=data_batch["data_samples"],
                      mode = 'loss')
        
        n, b, _ = data['search_anno'].shape
        gt_bboxes = data['search_anno'].reshape(n*b, -1)  
        gt_bboxes = box_xywh_to_xyxy(gt_bboxes)
        
        
        gt_instances=[]
        image_metas=[]

        
        
        
        
        
        
        
        for index,data_sample in enumerate(data_batch["data_samples"]):
            
            gt_instance=data_sample.gt_instances
            
            
            
            
            
            data_sample.set_metainfo(dict(img_shape=(hs,ws,3)))
            data_sample.set_metainfo(dict(ori_shape=(hs,ws)))
            data_sample.set_metainfo(dict(pad_param=np.array([0.0,0.0,0.0,0.0])))
            
            data_sample.set_metainfo(dict(scale_factor=(1.0,1.0)))
            
            gt_instance.bboxes=gt_bboxes[index].reshape(1,-1)*torch.tensor([[ws,hs,ws,hs]],device=gt_bboxes.device)
            
            
            gt_instance.labels=torch.tensor([0],device=gt_bboxes.device)
            image_meta=data_sample.metainfo
            gt_instances.append(gt_instance)
            image_metas.append(image_meta)
            
            
            
            

        
        
        
        
        data_batch = dict(inputs=search_images,
            data_samples=(data_batch["data_samples"],gt_instances,image_metas,data_batch_template),
            mode = 'loss'
            )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        loss = self.net.forward(**data_batch)
        
        
        
        
        
        return loss
    
    def cont_gt(self, gt_bboxes, size):
        bboxes = box_cxcywh_to_xyxy(box_xywh_to_cxcywh_scale(gt_bboxes, self.cfg.TRAIN.CTR_RATIO))*size
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) 
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) 
        mask_c = (x_mask & y_mask)

        cx = torch.floor((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = torch.floor((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask_c[bid, cy, cx] = True
        
        bboxes = box_cxcywh_to_xyxy(box_xywh_to_cxcywh(gt_bboxes))*size
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) 
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) 
        mask_t = 1-2*(x_mask & y_mask).long()
        mask_t[mask_c] = 0
        return mask_t.flatten(1)
        
    def anno2mask(self, gt_bboxes, size, reverse=False):
        bboxes = box_xywh_to_xyxy(gt_bboxes)*size 
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda()+0.5 
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) 
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) 
        mask = (x_mask & y_mask)

        cx = torch.floor((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = torch.floor((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask[bid, cy, cx] = True
        
        if reverse:
            mask = torch.cat([mask[bid.shape[0]//2:], mask[:bid.shape[0]//2]], dim=0)
        return mask.flatten(1)
        
    def sample_negative(self, logits, gt_bboxes, size):
        bboxes = gt_bboxes 
        cood_1d = (torch.arange(size)+0.5) / size
        cood = cood_1d.unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda() 
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) 
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) 
        mask = (x_mask & y_mask) 
        mask = (mask.reshape(gt_bboxes.shape[0], -1))*(-1e9) 
        sample_logits = torch.sort(logits.reshape(gt_bboxes.shape[0], -1)+mask, descending=True, dim=-1).values[:, :9]
        return sample_logits
        
    def contractive_learning(self, logits, gt_bbox):  
        b, n, sz, sz = logits.shape
        logits = logits.reshape(-1, 1, sz, sz)
        gt_bbox = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, n, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        ctr = (gt_bbox[:, :2] + gt_bbox[:, 2:]).reshape(b*n, 1, 1, 2) / 2
        neg_logits = self.sample_negative(logits, gt_bbox, sz).to(logits)
        sample_points = ctr * 2 - 1
        pos_logits = F.grid_sample(logits, sample_points, padding_mode="border", align_corners=True).reshape(b*n, -1) 
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        target = torch.zeros(b*n).to(gt_bbox.device).long()
        return logits, target 
    def compute_losses(self, pred_dict, gt_bbox):
        
        pred_logits = pred_dict['pred_logits'] 
        pred_boxes = pred_dict['pred_boxes'] 
        
        
        
        
        
        
        
        
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
        
        pre_logits_com=pred_logits 

        
        pre_logits_gt = torch.ones_like(pre_logits_com,dtype=pre_logits_com.dtype)
        cls_loss=self.objective['cls'](pre_logits_com.reshape(-1,1),pre_logits_gt.reshape(-1,1))



    

        
        
        
        
        loss=torch.tensor(0.0).cuda()+cls_loss

        pred_boxes_vec = pred_boxes.view(-1, 4)
        num_queries=1
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4)  
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  
        
        loss = loss+self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        
        status = {"Loss/total": loss,
                    "Loss/giou": giou_loss,
                    "Loss/l1": l1_loss,
                    "Loss/cls": cls_loss
                    }
        return loss, status
    
    
    
    def compute_losses_match(self, pred_dict, gt_bbox, gt_cls, gt_cont):
        
        pred_logits = pred_dict['pred_logits'] 
        pred_boxes = pred_dict['pred_boxes'] 
        
        
        
        
        
        

        

        
        
        
        
        
        
        
        
        pred_logits_last=pred_logits
        pred_logits_last_index=pred_logits_last.sigmoid().argmax(-1,keepdim=True)
        pred_logits_last=torch.gather(pred_logits_last,dim=-1,index=pred_logits_last_index)
        
        
        
        
        pred_boxes_last=pred_boxes
        
        
        lvl,bs,num_query,num_channel=pred_logits_last.shape
        
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes_last).view(-1, 4)
        
        num_queries=num_query
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[None,:, None, :].repeat((lvl,1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  
        try:
            giou_cost, iou = self.objective['giou_cost'](pred_boxes_vec, gt_boxes_vec)  
        except:
            giou_cost, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            
        l1_cost = torch.abs(pred_boxes_vec-gt_boxes_vec).mean(-1)  
        
        cost = self.loss_weight['giou'] * giou_cost + self.loss_weight['l1'] * l1_cost
        
        cost = cost.reshape(lvl,bs,num_queries,1) 
        
        
        pred_logits_last_gt=torch.ones_like(pred_logits_last)
        cls_cost=self.objective['cls_cost'](pred_logits_last.reshape(-1,1),pred_logits_last_gt.reshape(-1,1))
        cls_cost=cls_cost.reshape(lvl,bs,num_query,num_channel)
        
        
        cost = cost + 2.0*cls_cost 
        
        cost = -cost
        
        
    
        
         
        
        
        pred_logits_index=pred_logits.sigmoid().argmax(-1,keepdim=True)
        pre_logits_com=torch.gather(pred_logits,dim=-1,index=pred_logits_index).squeeze(-1)

    
        
        indices = cost.argmax(dim=-2)  

        
        pre_logits_com=torch.gather(pre_logits_com,dim=-1,index=indices)
        
        pre_logits_gt=torch.ones_like(pre_logits_com)
        
        
        cls_loss=self.objective['cls'](pre_logits_com.reshape(-1,1),pre_logits_gt.reshape(-1,1).detach())
    
        
        
        pred_logits_index=indices.unsqueeze(-1)
        pred_logits_index_box=pred_logits_index.expand(-1, -1, -1, 4)
        pred_boxes=torch.gather(pred_boxes,dim=-2,index=pred_logits_index_box)
        pred_boxes=pred_boxes.squeeze(2)
            
        
        
        
        loss=torch.tensor(0.0).cuda()+2.0*cls_loss
        for lvl,pred_logit in enumerate(pred_logits):
            pred_box=pred_boxes[lvl]
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_box).view(-1, 4)
            num_queries=1
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  
            
            
            
            
            
            loss = loss+self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        
        status = {"Loss/total": loss,
                    "Loss/giou": giou_loss,
                    "Loss/l1": l1_loss,
                    "Loss/cls_pos": cls_loss
                    }
        return loss, status
    
    
    



    
    def compute_losses_confidence(self, pred_dict, gt_bbox, gt_cls, gt_cont):
        
        pred_logits = pred_dict['pred_logits'] 
        pred_boxes = pred_dict['pred_boxes'] 
        confidence_query=pred_dict['confidence_query']

        

        
        
        
        
        
        

        

        pred_logits_index=pred_logits.sigmoid().argmax(-1,keepdim=True)
        pred_logits=torch.gather(pred_logits,dim=-1,index=pred_logits_index)
        
        
         
        
        pre_logits_com=pred_logits.squeeze(-1)
        pre_logits_gt=torch.zeros_like(pre_logits_com)
        
        
        indices = pre_logits_com.sigmoid().argmax(dim=-1, keepdim=True)  

        
        values = torch.ones_like(indices,dtype=pre_logits_com.dtype)
        pre_logits_gt.scatter_(-1,indices,values).detach()
        cls_loss=self.objective['cls'](pre_logits_com.reshape(-1,1),pre_logits_gt.reshape(-1,1))



    

        
        
        

        pred_logits_index=pred_logits.sigmoid().argmax(-2,keepdim=True)
        pred_logits=torch.gather(pred_logits,dim=-2,index=pred_logits_index)
        pred_logits=pred_logits.squeeze(-1).squeeze(-1)
        

        pred_logits_index_box=pred_logits_index.expand(-1, -1, -1, 4)
        pred_boxes=torch.gather(pred_boxes,dim=-2,index=pred_logits_index_box)
        pred_boxes=pred_boxes.squeeze(2)
            
        
        
        
        loss=torch.tensor(0.0).cuda()+cls_loss
        for lvl,pred_logit in enumerate(pred_logits):
            pred_box=pred_boxes[lvl]
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_box).view(-1, 4)
            num_queries=1
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  
            
            
            
            
            
            loss = loss+self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss


        gt_confidence=self.compute_iou(pre_boxes=pred_boxes_vec,gt_boxes=gt_boxes_vec).detach()
        confidence_query=confidence_query.sigmoid()    
        confidence_loss = self.objective['l1'](confidence_query[:,0], gt_confidence)
        loss=loss+confidence_loss


        
        status = {"Loss/total": loss,
                    "Loss/giou": giou_loss,
                    "Loss/l1": l1_loss,
                    "Loss/cls": cls_loss,
                    "confidence_loss":confidence_loss
                    }
        return loss, status
    
    
    
    
    
    
    
    def compute_losses_memory(self, pred_dict, gt_bbox, gt_cls, gt_cont):
        
        pred_logits = pred_dict['pred_logits'] 
        pred_boxes = pred_dict['pred_boxes'] 
        
        
        
        
        
        

        

        pred_logits_index=pred_logits.sigmoid().argmax(-1,keepdim=True)
        pred_logits=torch.gather(pred_logits,dim=-1,index=pred_logits_index)
        
        
         
        
        pre_logits_com=pred_logits.squeeze(-1)
        pre_logits_gt=torch.zeros_like(pre_logits_com)
        
        
        indices = pre_logits_com.sigmoid().argmax(dim=-1, keepdim=True)  

        
        values = torch.ones_like(indices,dtype=pre_logits_com.dtype)
        pre_logits_gt.scatter_(-1,indices,values).detach()
        cls_loss=self.objective['cls'](pre_logits_com.reshape(-1,1),pre_logits_gt.reshape(-1,1))



    

        
        
        

        pred_logits_index=pred_logits.sigmoid().argmax(-2,keepdim=True)
        pred_logits=torch.gather(pred_logits,dim=-2,index=pred_logits_index)
        pred_logits=pred_logits.squeeze(-1).squeeze(-1)
        

        pred_logits_index_box=pred_logits_index.expand(-1, -1, -1, 4)
        pred_boxes=torch.gather(pred_boxes,dim=-2,index=pred_logits_index_box)
        pred_boxes=pred_boxes.squeeze(2)
            
        
        
        
        num_queries=1
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  
        ntimes=4
        BN,num_gt=gt_boxes_vec.shape
        num_pre=num_gt
        gt_boxes_vec1=gt_boxes_vec.reshape(ntimes,BN//ntimes,num_pre)
        gt_boxes_vec_dir=gt_boxes_vec1[:-1]-gt_boxes_vec1[1:]
        gt_boxes_vec_dir=gt_boxes_vec_dir.reshape(BN//ntimes*(ntimes-1),num_pre)
        
        loss=torch.tensor(0.0).cuda()+cls_loss
        for lvl,pred_logit in enumerate(pred_logits):
            pred_box=pred_boxes[lvl]
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_box).view(-1, 4)
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  
            
            pred_boxes_vec=pred_boxes_vec.reshape(ntimes,BN//ntimes,num_pre)
            pred_boxes_vec_dir=pred_boxes_vec[:-1]-pred_boxes_vec[1:]
            pred_boxes_vec_dir=pred_boxes_vec_dir.reshape(BN//ntimes*(ntimes-1),num_pre)
          
            
            dir_loss = self.objective['l1'](pred_boxes_vec_dir, gt_boxes_vec_dir)  
            
            
            
            
            
            
            
            loss = loss+self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss+0.1*dir_loss
        
        status = {"Loss/total": loss,
                    "Loss/giou": giou_loss,
                    "Loss/l1": l1_loss,
                    "Loss/cls": cls_loss,
                    "Loss/dir_loss":dir_loss
                    }
        return loss, status

    def compute_iou(self,pre_boxes, gt_boxes):
        """
        计算每个预测框和每个真实框之间的IOU.

        Args:
        - pre_boxes (torch.Tensor): 预测框，形状为 [Bs, 4]，每个框的格式为 (x1, y1, x2, y2)。
        - gt_boxes (torch.Tensor): 真实框，形状为 [Bs, 4]，每个框的格式为 (x1, y1, x2, y2)。

        Returns:
        - ious (torch.Tensor): IOU的张量，形状为 [Bs]。
        """
        
        inter_x1 = torch.max(pre_boxes[:, 0], gt_boxes[:, 0])
        inter_y1 = torch.max(pre_boxes[:, 1], gt_boxes[:, 1])
        inter_x2 = torch.min(pre_boxes[:, 2], gt_boxes[:, 2])
        inter_y2 = torch.min(pre_boxes[:, 3], gt_boxes[:, 3])

        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        
        
        inter_area = inter_w * inter_h

        
        pre_area = (pre_boxes[:, 2] - pre_boxes[:, 0]) * (pre_boxes[:, 3] - pre_boxes[:, 1])
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

        
        union_area = pre_area + gt_area - inter_area

        
        eps = 1e-7  
        ious = inter_area / (union_area + eps)

        return ious
