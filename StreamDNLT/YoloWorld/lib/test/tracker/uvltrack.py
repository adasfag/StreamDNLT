from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, grounding_resize
# for debug
from copy import deepcopy
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.uvltrack.uvltrack import build_model
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box,box_xywh_to_xyxy, box_cxcywh_to_xywh, box_cxcywh_to_xyxy,box_xyxy_to_xywh,box_xyxy_to_cxcywh,clip_box_tensor,box_xywh_to_cxcywh
import numpy as np
import matplotlib.pyplot as plt
from lib.test.utils.hann import hann2d

from pytorch_pretrained_bert import BertTokenizer
from lib.utils.misc import NestedTensor
from lib.groundingdino.util.inference import load_model, load_image, predict, annotate,load_model_from_fine_tune,predict_grounding
from lib.groundingdino.models.GroundingDINO.constant.constant import MEMORY_SEQENCE
from torchvision.ops import box_convert

from lib.utils.box_ops import box_iou,box_xywh_to_xyxy



try:
    from mmdet.utils import get_test_pipeline_cfg
except:
    pass
from mmengine.dataset import Compose

from mmengine.config import Config


import copy





import os.path as osp





from .Kalman import SimpleBoxKalman



import torch.nn.functional as F

import torchvision.transforms.functional as TF












def center_distance(box1, box2):
    cx1, cy1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    cx2, cy2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

def size_difference(box1, box2):
    return abs(box1[2] - box2[2]) + abs(box1[3] - box2[3])

def select_best_box_with_presorted_scores(pre_kf_bbox, boxes, pre_scores, top_k=5, alpha=1.0, beta=0.5):
    """
    从 top-k 分数最高的框中，选择一个中心距离 + 尺寸最小的（加权组合）框
    """
    best_score = float('inf')
    selected_index = None
    selected_dist = None
    selected_size_diff = None

    for i in range(min(top_k, len(pre_scores))):
        box = boxes[i]  # 注意：scores 已排序，boxes 没排
        pre_score=pre_scores[i]
        dist = center_distance(box, pre_kf_bbox)
        size_diff = size_difference(box, pre_kf_bbox)

        score = alpha * dist + beta * size_diff # 越小越好

        if score < best_score:
            best_score = score
            selected_index = i
            selected_box = box
            selected_dist = dist
            selected_size_diff = size_diff

    return selected_index, selected_box, pre_scores[selected_index], selected_dist, selected_size_diff





def save_search_with_plt(search, filename="my_search_image.png"):
    # 去掉 batch 维度: (1, 3, H, W) -> (3, H, W)
    image = search.squeeze(0)

    # 归一化到 [0, 1]
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min + 1e-5)
    image = torch.clamp(image, 0, 1)

    # 转换为 numpy 并调整为 (H, W, 3)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # 保存图像
    plt.imsave(filename, image_np)


def compute_attention_map(feature_map, norm=True, method='mean'):
    """
    从一个特征图生成 attention map。
    feature_map: Tensor[C, H, W]
    return: attention_map [H, W]
    """
    if method == 'mean':
        attn_map = feature_map.abs().mean(dim=0)  # [H, W]
    elif method == 'l2':
        attn_map = torch.norm(feature_map, p=2, dim=0)  # [H, W]
    else:
        raise ValueError("Unknown method")

    if norm:
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)

    return attn_map.detach().cpu()

def save_attention_maps(current_image_feature, save_dir="attention_maps", method='mean'):
    """
    保存多尺度特征的 attention maps 为 PNG 图像。
    """
    os.makedirs(save_dir, exist_ok=True)

    for index, image_feature in enumerate(current_image_feature):
        feature_map = image_feature.squeeze(0)  # [C, H, W]
        attn_map = compute_attention_map(feature_map, method=method)  # [H, W]

        # 转为 PIL Image
        attn_img = TF.to_pil_image(attn_map)  # 自动把 [H, W] 转为灰度图

        # 保存图像
        filename = os.path.join(save_dir, f"attention_{index}.png")
        attn_img.save(filename)

        print(f"Saved: {filename}")


def extract_selected_scores(norm_pre_bbox, saved_cls):
    """
    Args:
        norm_pre_bbox (Tensor): shape [10, 4], coordinates in xyxy normalized (0~1)
        saved_cls (Tensor): shape [1, 2, H, W], logits or scores map

    Returns:
        selected_score (Tensor): shape [10, 1], average score in first class region
    """
    device = saved_cls.device
    _, _, H, W = saved_cls.shape

    # Get the first class score map: shape [1, 1, H, W]
    class_0_score_map = saved_cls[:, 0:1, :, :]  # shape: [1, 1, H, W]

    selected_score = []

    for bbox in norm_pre_bbox:
        x1, y1, x2, y2 = bbox

        # Denormalize to image coordinates
        ix1 = int(x1 * W)
        iy1 = int(y1 * H)
        ix2 = int(x2 * W)
        iy2 = int(y2 * H)

        # Clamp the values to be within valid range
        ix1 = max(0, min(ix1, W - 1))
        ix2 = max(0, min(ix2, W - 1))
        iy1 = max(0, min(iy1, H - 1))
        iy2 = max(0, min(iy2, H - 1))

        # Avoid degenerate boxes
        if ix2 <= ix1 or iy2 <= iy1:
            selected_score.append(torch.tensor([0.0], device=device))
            continue

        region = class_0_score_map[0, 0, iy1:iy2, ix1:ix2]  # shape [h', w']
        mean_score = region.mean() if region.numel() > 0 else torch.tensor(0.0, device=device)
        selected_score.append(mean_score.unsqueeze(0))  # shape [1]

    selected_score = torch.stack(selected_score, dim=0)  # shape [10, 1]
    return selected_score

class UVLTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        
        
        
        
        
        
        
        
        
        
        super(UVLTrack, self).__init__(params)
        #network = build_model(params.cfg) just for debug
        #network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False) just for debug
        self.map_size = params.search_size // 16
        self.cfg = params.cfg
        #self.network = network.cuda() just for debug
        #self.network.eval() just for debug
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = self.params.debug
        self.frame_id = 0
        # if self.debug:
        #     self.save_dir = "/ssd/myc/VL_project/MUTrack/debug"
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)
        self.update_interval = self.cfg.TEST.UPDATE_INTERVAL
        self.feat_size = self.params.search_size // 16
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH, do_lower_case=True)
        self.threshold = self.params.cfg.TEST.THRESHOLD
        self.has_cont = self.params.cfg.TRAIN.CONT_WEIGHT > 0
        self.max_score = 0
        self.memory_query=None
        self.grounding_model=None
        
        


        # try :
        # load config
    
        config_path="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        
        # load config
        cfg = Config.fromfile(config_path)

        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(config_path))[0])
            
        # init test pipeline
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    
        test_pipeline = Compose(test_pipeline_cfg)
        self.test_pipeline=test_pipeline
        # except:
        #     pass
        
        self.pre_states=[]
        self.memory_feature=[]
        self.kf=SimpleBoxKalman()
       

    def grounding(self, image, info: dict):
        bbox = torch.tensor([0., 0., 0., 0.]).cuda()
        h, w = image.shape[:2]
        im_crop_padded, _, _, _, _ = grounding_resize(image, self.params.grounding_size, bbox, None) #256 256 3
        ground = self.preprocessor.process(im_crop_padded).cuda()
        template = torch.zeros([1, 3, self.params.template_size, self.params.template_size]).cuda()
        template_mask = torch.zeros([1, (self.params.template_size//16)**2]).bool().cuda()
        context_mask = torch.zeros([1, (self.params.search_size//16)**2]).bool().cuda()
        text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)
        self.text = NestedTensor(text, mask)
        flag = torch.tensor([[1]]).cuda()
        with torch.no_grad():
            out_dict = self.network.forward(template, ground, self.text, template_mask, context_mask, flag)
        out_dict['pred_boxes'] = box_cxcywh_to_xywh(out_dict['pred_boxes']*np.max(image.shape[:2]))[0, 0].cpu().tolist()
        dx, dy = min(0, (w-h)/2), min(0, (h-w)/2)
        out_dict['pred_boxes'][0] = out_dict['pred_boxes'][0] + dx
        out_dict['pred_boxes'][1] = out_dict['pred_boxes'][1] + dy
        return out_dict
    
    

    def grounding_dino(self, image,info: dict):
        

        model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_test/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar")

        
        
        
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_adapter_decode/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar")
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_adapter_query_decode_adapter_enhancement_addloss/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar") 
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_adapter_query_decode_adapter_enhancement_addloss_moreinstances/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar") 
        
        
        
        
        
        
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_adapter_query_decode_adapter_enhancement_addloss_moreinstances_gai/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar") 
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_adapter_query_decode_adapter_enhancement_addloss_moreinstances_gai/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar") 
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_adapter_class_score_moreinstances/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0005.pth.tar") 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_class_score_moreinstances/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar") 
        
        
        
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_v0/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar") 
        
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_v0_epoch/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0020.pth.tar") 
        
        
        
        
        
        
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_v0_epoch_tnl2k/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0020.pth.tar") 
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_v0_epoch_lasot/checkpoints/train/uvltrack/baseline_base_dino_memory_lasot_grounding/GroundingDINOTrackerAdapter_ep0010.pth.tar") 
        
        #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_grounding_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinogrounding_query_v0_epoch_tnl2k/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_grounding/GroundingDINOTrackerAdapter_ep0020.pth.tar") 
        
        
        
        
     
        TEXT_PROMPT=info['language']
        BOX_TRESHOLD=0.0
        TEXT_TRESHOLD=0.25
        
        
        
        bbox = torch.tensor([0., 0., 0., 0.]).cuda()
        h, w = image.shape[:2]
        im_crop_padded, _, _, _, _ = grounding_resize(image, 320, bbox, None) #256 256 3
        search = self.preprocessor.process(im_crop_padded)[0].cuda()#3 256 256
        template = None
        boxes, logits, phrases,class_score= predict_grounding(
                            model=model,
                            image=search,
                            template=template,
                            caption=TEXT_PROMPT,
                            box_threshold=BOX_TRESHOLD,
                            text_threshold=TEXT_TRESHOLD
                        )
                
                
                
             
                  
        max_index=logits.argmax(dim=0)

        boxes=boxes[0:1] #现在直接用第一个
        logits=logits[max_index:max_index+1]
        phrases=phrases[max_index:max_index+1]  #cx,cy,w,h
        boxes=boxes*np.max(image.shape[:2])
        boxes = box_cxcywh_to_xywh(boxes).cpu().tolist()[0]
        
        







        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        out_dict={'pred_boxes':boxes}
        dx, dy = min(0, (w-h)/2), min(0, (h-w)/2)
        out_dict['pred_boxes'][0] = out_dict['pred_boxes'][0] + dx
        out_dict['pred_boxes'][1] = out_dict['pred_boxes'][1] + dy
        
        gt_bbox=info['init_bbox']
        iou,union=box_iou(box_xywh_to_xyxy(torch.Tensor(out_dict['pred_boxes']).reshape(-1,4)),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
        print(iou)
        
        return out_dict
    
    
    def grounding_yolo(self, image,info: dict,imagepath:str):#对于NL使用grounding模型，剩下交给tracking模型
        

        model = info["model"]
        
        
     
        TEXT_PROMPT=info['language']
        frame_path=imagepath
        
        
        with torch.no_grad():

            
            
            
            
            if TEXT_PROMPT.endswith('.txt'):
                with open(TEXT_PROMPT) as f:
                    lines = f.readlines()
                texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
            else:
                texts = [t.strip() for t in TEXT_PROMPT.split(',')]#texts后面需要加一个' '
                texts=[["".join(texts)]]
                texts.append([' '])
            
            
            
            
            
            data_info = dict(img_id=0, img_path=frame_path, texts=texts)#图像路径:str 文本:两个list ['prompt']+['']
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


            data_info = self.test_pipeline(data_info)
            
            
         
            data_samples=copy.deepcopy(data_info['data_samples'])
            data_samples.set_metainfo(dict(img_shape=(self.params.search_size,self.params.search_size,3)))
            data_samples.set_metainfo(dict(ori_shape=(self.params.search_size,self.params.search_size)))
            data_samples.set_metainfo(dict(pad_param=np.array([0.0,0.0,0.0,0.0])))
            
            data_samples.set_metainfo(dict(scale_factor=(1.0,1.0)))
            
            
            
            
            
            
            


            
        
            
            
            
            
            
            
            
            
            
            data_batch = dict(inputs=(data_info["inputs"]).unsqueeze(0),
            data_samples=[data_info["data_samples"]],
            template=None
            )

        
            
    
      
      
      
      
      
      
      
      
      
           
            output = model.test_step_template(data_batch)[0]
            
            
            
            
            
            
            
            
            
            
            
            
           
           
            pred_instances = output.pred_instances
            

            pre_bbox=pred_instances.bboxes[0]
            pre_bbox_anno=pre_bbox
            pre_scores=pred_instances.scores[0]
        




        del model
            

        self.state= box_xyxy_to_xywh(pre_bbox).numpy() #363 101 409 116

   
        return {"target_bbox": self.state,"image":data_batch["inputs"]}
    
    
    
    
    
    
    
    
    
    
    def grounding_dino_traingfree(self, image,info: dict,imagepath:str):
        #这一块直接调用原始模型进行全图推理了
        
        
        
        
        
        if self.grounding_model is None:
            #model = load_model("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/qui_wzh/20240630/home/language-guided-tracking/GroundingDINO/weights/groundingdino_swint_ogc.pth")
            model = load_model("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinB_cfg.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/pre_models_weight/groundingdino_swinb_cogcoor.pth")
  
            self.grounding_model=model
        else:
            model=self.grounding_model
        TEXT_PROMPT=info['language']
        BOX_TRESHOLD=0.0
        TEXT_TRESHOLD=0.00
        
        image_source, image = load_image(imagepath)#image_source 360 640 3； image 3 750 1333
                        
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD #通常是排名靠前的单词更可靠
        )
        
        
        
        
        max_index=logits.argmax(dim=0)

        boxes=boxes[max_index:max_index+1]
        logits=logits[max_index:max_index+1]
        phrases=phrases[max_index:max_index+1]
                            
        
        
        
        
        
        print(TEXT_PROMPT)
        print(phrases)
        
        
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
        xyxy=xyxy[0].tolist()#[258.31088 ,  91.37656 ,  23.302063,  30.50621 ]
        
        
        
        gt_bbox=info['init_bbox']
        iou,union=box_iou(box_xywh_to_xyxy(torch.Tensor(xyxy).reshape(-1,4)),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))

        if iou==0.0:
            xyxy=gt_bbox
                     
        
        out_dict={'pred_boxes':xyxy}

        return out_dict
    
    

    def window_prior(self):
        hanning = np.hanning(self.map_size)#16
        window = np.outer(hanning, hanning)#16 16
        self.window = window.flatten()#256
        self.torch_window = hann2d(torch.tensor([self.map_size, self.map_size]).long(), centered=True).flatten()

    def initialize(self, image, info: dict,test:bool=False):
        if self.cfg.TEST.MODE == 'NL':
            grounding_state = self.grounding(image, info)
            init_bbox = grounding_state['pred_boxes']
            self.flag = torch.tensor([[2]]).cuda()
        elif self.cfg.TEST.MODE == 'NLBBOX':
            text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)#text:1 40 mask:1 40
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[2]]).cuda()
        else:
            text = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).long().cuda()
            mask = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).cuda()
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[0]]).cuda()
        self.window_prior()
        z_patch_arr, _, _, bbox = sample_target(image, init_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size, return_bbox=True)#z_patch_arr:128 128 3 bbox:1 1 4
        self.template_mask = self.anno2mask(bbox.reshape(1, 4), size=self.params.template_size//16)#1 64
        self.z_patch_arr = z_patch_arr
        self.template_bbox = (bbox*self.params.template_size)[0, 0].tolist()
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        # forward the context once
        y_patch_arr, _, _, y_bbox = sample_target(image, init_bbox, self.params.search_factor,
                                                    output_sz=self.params.search_size, return_bbox=True)#y_patch_arr:256 256  y_bbox:1 1 4
        self.y_patch_arr = y_patch_arr
        self.context_bbox = (y_bbox*self.params.search_size)[0, 0].tolist()
        context = self.preprocessor.process(y_patch_arr)
        context_mask = self.anno2mask(y_bbox.reshape(1, 4), self.params.search_size//16)
        self.prompt = self.network.forward_prompt_init(self.template, context, self.text, self.template_mask, context_mask, self.flag)#1 3 768
        # save states
        self.state = init_bbox
        self.frame_id = 0
        if test:
            return self.state
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def initialize_dino(self, image, info: dict,imagepath:str):
        if self.cfg.TEST.MODE == 'NL':
            grounding_state = self.grounding_dino(image, info) #info['init_bbox']=[262.0, 94.0, 16.0, 26.0]
            init_bbox = grounding_state['pred_boxes']
            self.flag = torch.tensor([[2]]).cuda()
        elif self.cfg.TEST.MODE == 'NLBBOX':
            text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)#text:1 40 mask:1 40
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[2]]).cuda()
        else:
            text = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).long().cuda()
            mask = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).cuda()
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[0]]).cuda()
        self.window_prior()
        z_patch_arr, _, _, bbox = sample_target(image, init_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size, return_bbox=True)#z_patch_arr:128 128 3 bbox:1 1 4
        self.template_mask = self.anno2mask(bbox.reshape(1, 4), size=self.params.template_size//16)#1 64
        self.z_patch_arr = z_patch_arr
        self.template_bbox = (bbox*self.params.template_size)[0, 0].tolist()
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        # forward the context once
        y_patch_arr, _, _, y_bbox = sample_target(image, init_bbox, self.params.search_factor,
                                                    output_sz=self.params.search_size, return_bbox=True)#y_patch_arr:256 256  y_bbox:1 1 4
        self.y_patch_arr = y_patch_arr
        self.context_bbox = (y_bbox*self.params.search_size)[0, 0].tolist()
        # save states
        self.state = init_bbox
        self.frame_id = 0
        self.memory_query=None
        
        self.kf.initialize(self.state)  # 初始检测框

    def initialize_yolo(self, image, info: dict,imagepath:str):
        if self.cfg.TEST.MODE == 'NL':
            grounding_state = self.grounding_yolo(image, info,imagepath) #info['init_bbox']=[262.0, 94.0, 16.0, 26.0]
            init_bbox = grounding_state['target_bbox']
            self.flag = torch.tensor([[2]]).cuda()
        elif self.cfg.TEST.MODE == 'NLBBOX':
            text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)#text:1 40 mask:1 40
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[2]]).cuda()
        else:
            text = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).long().cuda()
            mask = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).cuda()
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[0]]).cuda()
        self.window_prior()
        z_patch_arr, _, _, bbox = sample_target(image, init_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size, return_bbox=True)#z_patch_arr:128 128 3 bbox:1 1 4
        self.template_mask = self.anno2mask(bbox.reshape(1, 4), size=self.params.template_size//16)#1 64
        self.z_patch_arr = z_patch_arr
        self.template_bbox = (bbox*self.params.template_size)[0, 0].tolist()
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        # forward the context once
        y_patch_arr, _, _, y_bbox = sample_target(image, init_bbox, self.params.search_factor,
                                                    output_sz=self.params.search_size, return_bbox=True)#y_patch_arr:256 256  y_bbox:1 1 4
        self.y_patch_arr = y_patch_arr
        self.context_bbox = (y_bbox*self.params.search_size)[0, 0].tolist()
        # save states
        self.state = init_bbox
        self.pre_states.append(self.state)
        
        self.frame_id = 0
        self.memory_query=None

        self.kf.initialize(box_xywh_to_cxcywh(torch.Tensor(self.state)).numpy())  # 初始检测框

        self.pre_feat=None

        self.init_kf=False

        self.memory_feature=[]



    

    def track(self, image, info: dict = None):
        H, W, _ = image.shape#480 684 3
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)#1 3 256 256
        with torch.no_grad():
            template = self.template
            out_dict = self.network.forward_test(template, search, self.text, self.prompt, self.flag)

        pred_boxes = out_dict['bbox_map'].view(-1, 4).detach().cpu() # b, s, 4
        pred_cls = out_dict['cls_score_test'].view(-1).detach().cpu() # b, s
        pred_cont = out_dict['cont_score'].softmax(-1)[:, :, 0].view(-1).detach().cpu() if self.has_cont else 1 # b, s
        pred_cls_merge = pred_cls * self.window * pred_cont
        pred_box_net = pred_boxes[torch.argmax(pred_cls_merge)]
        score = (pred_cls * pred_cont)[torch.argmax(pred_cls_merge)]
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_box_net * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        if score > self.max_score and self.has_cont:
            self.pred_box_net = pred_box_net
            self.out_dict = out_dict
            self.max_score = score
        
        if self.frame_id % self.update_interval == 0 and self.has_cont and self.max_score > self.threshold:
            self.y_patch_arr = x_patch_arr
            context_bbox = box_cxcywh_to_xywh(self.pred_box_net.reshape(1, 4))
            context_mask = self.anno2mask(context_bbox, self.params.search_size//16)
            self.context_bbox = (context_bbox[0]*self.params.search_size).detach().cpu().tolist()
            self.prompt = self.network.forward_prompt(self.out_dict, self.template_mask, context_mask)
            self.max_score = 0
            
        return {"target_bbox": self.state}
    
    
    
    def dino_track(self, image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info: dict = None,frame_path=None):
        
        
        
        
        H, W, _ = image.shape#480 684 3
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)[0]#1 3 256 256
        _,h,w=search.shape
        with torch.no_grad():
            template = self.template
            if not MEMORY_SEQENCE:
                boxes, logits, phrases = predict(
                                model=model,
                                image=search,
                                template=template,
                                caption=TEXT_PROMPT,
                                box_threshold=BOX_TRESHOLD,
                                text_threshold=TEXT_TRESHOLD #boxes 1 4； logits 1 ；phrases white airplane landing 按照顺序排列
    
                                
                            )
            else:
                boxes, logits, phrases,confidence_query = predict(
                                model=model,
                                image=search,
                                template=template,
                                caption=TEXT_PROMPT,
                                box_threshold=BOX_TRESHOLD,
                                text_threshold=TEXT_TRESHOLD ,
                                memory_image_feature=self.memory_image_feature
                            )
            
            
            
            
            
            
            
            
            
            
            max_index=logits.argmax(dim=0)
            


            boxes=boxes[max_index:max_index+1]
            logits=logits[max_index:max_index+1]
            phrases=phrases[max_index:max_index+1]  #cx,cy,w,h
            boxes = (boxes * self.params.search_size / resize_factor)[0].tolist()  # (cx, cy, w, h)
            boxes=clip_box(self.map_box_back(boxes, resize_factor), H, W, margin=10) # x , y , w ,h
            vis_boxes=torch.Tensor(boxes)/torch.Tensor([W,H,W,H])
            vis_boxes[0] = vis_boxes[0] + vis_boxes[2]/2
            vis_boxes[1] = vis_boxes[1] + vis_boxes[3]/2
       
        
        gt_bbox=info['gt_bbox']
        iou,union=box_iou(box_xywh_to_xyxy(torch.Tensor(boxes).reshape(-1,4)),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
        print(f'iou:{iou}') 
        

        self.state=boxes
   
        return {"target_bbox": self.state}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def yolo_track(self, image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info: dict = None,frame_path=None):
        
        
        
        
    
    
    
    
        with torch.no_grad():

            
            
            
            
            if TEXT_PROMPT.endswith('.txt'):
                with open(TEXT_PROMPT) as f:
                    lines = f.readlines()
                texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
            else:
                texts = [t.strip() for t in TEXT_PROMPT.split(',')]#texts后面需要加一个' '
                texts=[["".join(texts)]]
                texts.append([' '])
            
            
            
            
            
            data_info = dict(img_id=0, img_path=frame_path, texts=texts)#图像路径:str 文本:两个list ['prompt']+['']
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


            data_info = self.test_pipeline(data_info)
            
            
         
            data_samples=copy.deepcopy(data_info['data_samples'])
            data_samples.set_metainfo(dict(img_shape=(self.params.search_size,self.params.search_size,3)))
            data_samples.set_metainfo(dict(ori_shape=(self.params.search_size,self.params.search_size)))
            data_samples.set_metainfo(dict(pad_param=np.array([0.0,0.0,0.0,0.0])))
            
            data_samples.set_metainfo(dict(scale_factor=(1.0,1.0)))
            
            
            
            
            
            
            


            
            template = self.template#没有归一化
            
            data_batch_template = dict(inputs=template,
                        data_samples=[data_samples],
                        mode = 'loss')
            
            
            
            
            data_batch_template = model.data_preprocessor(data_batch_template,False)
        
            
            
            
            
            
            
            
            
            
            
            data_batch = dict(inputs=(data_info["inputs"]).unsqueeze(0),
            data_samples=[data_info["data_samples"]],
            template=None
            )

        
            
    
      
      
      
      
      
      
      
      
      
           
            output = model.test_step_template(data_batch)[0]
            
            
            
            
            
            
            
            
            
            
            
            
           
           
            pred_instances = output.pred_instances
            

            pre_bbox=pred_instances.bboxes[0]
            pre_bbox_anno=pre_bbox
            pre_scores=pred_instances.scores[0]
            
            

            H,W=data_info["data_samples"].ori_shape

    
            pre_bbox=box_xyxy_to_xywh(torch.Tensor(pre_bbox))

            if not self.init_kf:
                self.kf.initialize(pre_bbox)
                self.init_kf=True
            else:
                pre_kf_boxes=self.kf.predict(W=W,H=H)

            
                pre_bbox=pred_instances.bboxes
                pre_scores=pred_instances.scores
                pre_bbox=box_xyxy_to_xywh(pre_bbox)

                num_bbox,num_channel=pre_bbox.shape
                pre_kf_boxes=torch.tensor(pre_kf_boxes)[None]

                gt_boxes=pre_kf_boxes.repeat(num_bbox,1)
                iou,union=box_iou(box_xywh_to_xyxy(pre_bbox),box_xywh_to_xyxy(gt_boxes))

                selected_index=0
                for index, pre_score in enumerate(pre_scores):
                    if iou[index]>0.7:
                        selected_index=index
                
                print(f'selected_index:{selected_index}')
                pre_bbox=pred_instances.bboxes
                pre_scores=pred_instances.scores
                pre_bbox=pre_bbox[selected_index]
                pre_scores=pre_scores[selected_index]
                pre_bbox=box_xyxy_to_xywh(pre_bbox)

                self.kf.update(pre_bbox.numpy())
                
            
            vis_boxes=pre_bbox

         

        
        
        
        
        
        annotated_frame = annotate(image_source=image, boxes=vis_boxes[None], logits=pre_scores.reshape(1), phrases=texts[0],gt_boxes=torch.Tensor(info['gt_bbox']).reshape(1,-1),vis_pre_kf_bbox=pre_kf_boxes)
        output_path=os.path.join('work_dir','test_otb_datasetfullpicture.jpg')#用两个模型
        #output_path=os.path.join('work_dir','test_tnl2k_tracking.jpg')
        #output_path=os.path.join('work_dir','test_lasot_tracking.jpg')
        
        # pic_path='test_lasot_tracking.jpg'
        # if frame_path is not None:
        #     file_name=frame_path.split('/')[-1]
        #     seq_name=info['seq_name']
        #     pic_path=os.path.join('work_dir',seq_name+'com')
        #     if os.path.exists(pic_path):
        #         pass
        #     else:
        #         os.makedirs(pic_path)
        #     file_name=file_name
        #     pic_path=os.path.join(seq_name+'com',file_name)
        # output_path=os.path.join('work_dir',pic_path)
        cv2.imwrite(output_path, annotated_frame)      
        
        
        
        
        
        

        
        
    
    
            





        
        
        gt_bbox=info['gt_bbox']
        
    
        # pre_bbox[1]=gt_bbox[1]
        # pre_bbox[3]=gt_bbox[1]+gt_bbox[3]
        
        boxes=torch.Tensor(pre_bbox)
        iou,union=box_iou(box_xywh_to_xyxy(boxes).reshape(-1,4),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
        print(f'iou:{iou}') 
        
        self.state= box_xyxy_to_xywh(pre_bbox).numpy()

   
        return {"target_bbox": self.state,"image":data_batch["inputs"]}
    
    
    
    
    def yolo_track_tracking(self, image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info: dict = None,frame_path=None):
        
        
    
        H, W, _ = image.shape#480 684 3
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process_yolo(x_patch_arr)[0]#不要归一化
        
        #这里有问题
        
        
        
        
        
    
    
    
    
        with torch.no_grad():

            
            
            
            
            if TEXT_PROMPT.endswith('.txt'):
                with open(TEXT_PROMPT) as f:
                    lines = f.readlines()
                texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
            else:
                texts = [t.strip() for t in TEXT_PROMPT.split(',')]#texts后面需要加一个' '
                texts=[["".join(texts)]]
                texts.append([' '])
            
            
            
            
            
            
            
            
            data_info = dict(img_id=0, img_path=frame_path, texts=texts)#图像路径:str 文本:两个list ['prompt']+['']
            
            
        
            data_info = self.test_pipeline(data_info)
            
            
            
            
    
            
            
            
            
            
            
            
            
            
            
            
            
                        
            
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_info['data_samples']])
            data_samples=data_info['data_samples']
            data_samples.set_metainfo(dict(img_shape=(self.params.search_size,self.params.search_size,3)))
            data_samples.set_metainfo(dict(ori_shape=(self.params.search_size,self.params.search_size)))
            data_samples.set_metainfo(dict(pad_param=np.array([0.0,0.0,0.0,0.0])))
            
            data_samples.set_metainfo(dict(scale_factor=(1.0,1.0)))
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_samples])
            
            
            
            
            
            
            
            template = self.template#没有归一化
            
            data_batch_template = dict(inputs=template,
                        data_samples=[data_samples],
                        mode = 'loss')
            
            
            
            data_batch_template = model.data_preprocessor(data_batch_template,False)
        
            
            
            
            
            
            
            
            
            
            
            
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_samples],
            template=data_batch_template["inputs"]
            )
            
    
      
           
            output = model.test_step_template(data_batch)[0]
            pred_instances = output.pred_instances
            

            pre_bbox=pred_instances.bboxes[0]
            pre_bbox_anno=pre_bbox
            pre_scores=pred_instances.scores[0]
            
            
            pre_bbox=pre_bbox
            pre_bbox=box_xyxy_to_cxcywh(pre_bbox)
            
            boxes = (pre_bbox / resize_factor).tolist()  # (cx, cy, w, h)
            boxes=clip_box(self.map_box_back(boxes, resize_factor), H, W, margin=10) # x , y , w ,h
            
            
            

            
            
                            
        
        
        
        
        
        
        
        gt_bbox=info['gt_bbox']
        boxes=torch.Tensor(boxes)
        iou,union=box_iou(box_xywh_to_xyxy(boxes).reshape(-1,4),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
        print(f'iou:{iou}') 

        self.state= boxes.numpy()
        

   
        return {"target_bbox": self.state}
    
    
    
    
    def yolo_track_tracking_templateT(self, image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info: dict = None,frame_path=None):
        
        
    
        H, W, _ = image.shape#480 684 3
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)[0]#1 3 256 256
        


        
    

        template = self.template#没有归一化
    
    
        with torch.no_grad():

            
            
            
            
            if TEXT_PROMPT.endswith('.txt'):
                with open(TEXT_PROMPT) as f:
                    lines = f.readlines()
                texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
            else:
                texts = [t.strip() for t in TEXT_PROMPT.split(',')]#texts后面需要加一个' '
                texts=[["".join(texts)]]
                texts.append([' '])
            
            
            
            
            
            
            
            
            data_info = dict(img_id=0, img_path=frame_path, texts=texts)#图像路径:str 文本:两个list ['prompt']+['']
            
            
        
            data_info = self.test_pipeline(data_info)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                        
            
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_info['data_samples']])
            data_samples=data_info['data_samples']
            data_samples.set_metainfo(dict(img_shape=(self.params.search_size,self.params.search_size,3)))
            data_samples.set_metainfo(dict(ori_shape=(self.params.search_size,self.params.search_size)))
            data_samples.set_metainfo(dict(pad_param=np.array([0.0,0.0,0.0,0.0])))
            
            data_samples.set_metainfo(dict(scale_factor=(1.0,1.0)))
            
            
            
            
            
               
            data_batch_template = dict(inputs=template,
                        data_samples=[data_samples],
                        mode = 'loss')
    
    


   


            data_batch_template = model.data_preprocessor(data_batch_template,False)
        
            
            
            
            
            
            
            
            
            
            
            
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_samples],
            template=template
            )
            
            
            
            
            
            
            
    
      
      







           
            output = model.test_step_template_tracking(data_batch)[0]



            pred_instances = output.pred_instances
            
            
            #这里和历史的bboxes进行匹配，取iou最大的作为结果
            
            pre_bbox=pred_instances.bboxes  #300 4
            
            norm_prebbox=pre_bbox/torch.tensor([[W,H,W,H]])
            

           
























            


            


            
            pre_bbox=box_xyxy_to_cxcywh(pre_bbox)
        
    

            
            boxes = pre_bbox / resize_factor  # (cx, cy, w, h)
            boxes=clip_box_tensor(self.map_box_back_tensor(boxes, resize_factor), H, W, margin=10) # x , y , w ,h
            
            
        
        
            

            
            gt_bbox=info['gt_bbox']




            boxes=torch.Tensor(boxes)
            
            
            gt_bbox=self.kf.predict(W=W,H=H)
            gt_bbox=torch.tensor(gt_bbox)
            pre_kf_bbox=gt_bbox

            num_boxes,num_channel=boxes.shape
            gt_bbox=gt_bbox[None].repeat(num_boxes,1)
            iou,union=box_iou(box_xywh_to_xyxy(boxes).reshape(-1,4),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))

            
        
     




    


            selected_index=0
            pre_scores=pred_instances.scores
            labels=pred_instances.labels






            for index,pre_score in enumerate(pre_scores):
                if iou[index]>0.7 or pre_score>0.8:
                    selected_index=index
                    selected_iou=[index]
                    break
    
            selected_index=0
            selected_label=labels[selected_index]
            selected_score=pre_scores[selected_index]
            #selected_index=torch.argmax(iou)
    
            print(f'selected_index:{selected_index}')
            print(f'selected_score:{selected_score}')
            print(f'selected_label:{selected_label}')
            print(f'pre_kf_box:{pre_kf_bbox}')







            # selected_similarity=0.0
            # if  similarity is not None:
            #     for index in range(0,30):
            #         selected_similarity = similarity[index]

            #         if selected_similarity > 0.8:#只有前三十个有一个大于0.8才更新
            #             selected_index=index
            #             break 
            
            # self.pre_feat=features[selected_index] 
            # print(f'selected_similarity:{selected_similarity}')
            
                                

        
  
        
            
            


            
            
            
            
   
            pre_bbox=pred_instances.bboxes[selected_index]
            
            
            pre_scores=pred_instances.scores[selected_index]
            
            
            pre_bbox=pre_bbox
            pre_bbox=box_xyxy_to_cxcywh(pre_bbox)
            
            boxes = (pre_bbox / resize_factor).tolist()  # (cx, cy, w, h)

            boxes=clip_box(self.map_box_back(boxes, resize_factor), H, W, margin=10) # x , y , w ,h

            
            vis_boxes=torch.Tensor(boxes)

            
            vis_pre_kf_bbox=torch.Tensor(pre_kf_bbox)

        
        
        
        
        
        annotated_frame = annotate(image_source=image, boxes=vis_boxes[None], logits=pre_scores.reshape(1), phrases=texts[0],gt_boxes=torch.Tensor(info['gt_bbox']).reshape(1,-1))
        output_path=os.path.join('work_dir','vis_result/test_otb_dataset.jpg')
        #output_path=os.path.join('work_dir','test_otb_datasetfullpicture.jpg')
        #output_path=os.path.join('work_dir','test_tnl2k_tracking.jpg')
        #output_path=os.path.join('work_dir','test_lasot_tracking.jpg')
        
        # pic_path='test_lasot_tracking.jpg'
        # if frame_path is not None:
        #     file_name=frame_path.split('/')[-1]
        #     seq_name=info['seq_name']
        #     pic_path=os.path.join('work_dir',seq_name+'com')
        #     if os.path.exists(pic_path):
        #         pass
        #     else:
        #         os.makedirs(pic_path)
        #     file_name=file_name
        #     pic_path=os.path.join(seq_name+'com',file_name)
        # output_path=os.path.join('work_dir',pic_path)
        cv2.imwrite(output_path, annotated_frame)
            

            
            
                            
        
        
        
        
        
        
        
        gt_bbox=info['gt_bbox']
        boxes=torch.Tensor(boxes)
        iou,union=box_iou(box_xywh_to_xyxy(boxes).reshape(-1,4),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
        print(f'iou:{iou}') 
        
    
        

        


        self.state= boxes.numpy()


        self.kf.update(self.state)
        


        # with open("work_dir/output.txt","a") as f:
        #     f.write(f'----------------------------\n')
        #     f.write(f'iou:{iou}\n')
        #     f.write(f'selected_index:{selected_index}\n')
        #     f.write(f'pre_kf_box:{pre_kf_bbox}\n')
        #     f.write(f'pre_box:{self.state}\n')
        #     f.write(f'gt_box:{gt_bbox}\n')
        #     f.write(f'pre_scores:{pre_scores}\n')
        #     f.write(f'prediction_gt_dist:{prediction_gt_dist}\n')
        #     f.write(f'prediction_gt_size_diff:{prediction_gt_size_diff}\n')
        #     f.write(f'predicion_pre_dist:{predicion_pre_dist}\n')
        #     f.write(f'predicion_pre_size_diff:{predicion_pre_size_diff}\n')
        #     f.write(f'unpredicion_pre_dist:{unpredicion_pre_dist}\n')
        #     f.write(f'unpredicion_pre_size_diff:{unpredicion_pre_size_diff}\n')
        #     f.write(f'----------------------------\n')
            




   
        return {"target_bbox": self.state}
    


    
    


    def yolo_track_tracking_template(self, image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info: dict = None,frame_path=None):
        
        
    
        H, W, _ = image.shape#480 684 3
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)[0]#1 3 256 256
        

        save_path="work_dir/vis_result/"
        #save_search_with_plt(search,save_path+"search_image.png")
        
        
        
    

        template = self.template#没有归一化 动态template？ 1 3 128 128
        # 自动归一化到 [0, 1]，避免图像太暗或无法保存
        template -= template.min()
        template /= template.max() + 1e-5
        #TF.to_pil_image(template[0]).save(save_path+"template.png")

            
        with torch.no_grad():

            
            
            
            
            if TEXT_PROMPT.endswith('.txt'):
                with open(TEXT_PROMPT) as f:
                    lines = f.readlines()
                texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
            else:
                texts = [t.strip() for t in TEXT_PROMPT.split(',')]#texts后面需要加一个' '
                texts=[["".join(texts)]]
                texts.append([' '])
            
            
            
            
            
            
            
            
            data_info = dict(img_id=0, img_path=frame_path, texts=texts)#图像路径:str 文本:两个list ['prompt']+['']
            
            
        
            data_info = self.test_pipeline(data_info)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                        
            
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_info['data_samples']])
            data_samples=data_info['data_samples']
            data_samples.set_metainfo(dict(img_shape=(self.params.search_size,self.params.search_size,3)))
            data_samples.set_metainfo(dict(ori_shape=(self.params.search_size,self.params.search_size)))
            data_samples.set_metainfo(dict(pad_param=np.array([0.0,0.0,0.0,0.0])))
            
            data_samples.set_metainfo(dict(scale_factor=(1.0,1.0)))
            
            
            
            
            
               
            data_batch_template = dict(inputs=template,
                        data_samples=[data_samples],
                        mode = 'loss')
    
    


   


            data_batch_template = model.data_preprocessor(data_batch_template,False)
        
            
            
            
            
            
            
            
            
            
            
            
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_samples],
            template=template
            )
            
            
            
            
            
            
            
    
      
      







           
            output = model.test_step_template_tracking(data_batch)[0]
            current_image_feature=output.current_img_feature #3 256 40 40
            #save_attention_maps(current_image_feature,save_dir="work_dir/vis_result/")

            current_txt_feats=output.current_txt_feats #1 2 768

            saved_cls_and_box=output.saved_cls_and_box #1 2 768
            saved_cls,saved_box=saved_cls_and_box  #saved_cls:3 [1 2 w h] 分别用框截取第二个特征map
                                                   # saved_box:3 [1 4 w h]

            low_image_feature=current_image_feature[0]#256 40 40






            pred_instances = output.pred_instances
            pre_query = output.pre_query
            


            memory_len=6
            len_selected=10

            
            #这里和历史的bboxes进行匹配，取iou最大的作为结果
            
            pre_bbox=pred_instances.bboxes  #300 4

            norm_pre_bbox=pre_bbox/torch.tensor([W,H,W,H])
            
            norm_pre_bbox=norm_pre_bbox[0:len_selected]

           
            # 预处理
            low_image_feature=low_image_feature[0]
            C, feat_height, feat_width = low_image_feature.shape

            # 将归一化 box 转换为像素坐标
            boxes = norm_pre_bbox.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * feat_width  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * feat_height  # y1, y2

            # 限制在边界内，防止越界
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, feat_width - 1)  # x1, x2
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, feat_height - 1)  # y1, y2

            # 开始提取区域特征
            features = []
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())

                # 防止 box 为 0 尺寸
                if x2 <= x1 or y2 <= y1:
                    patch = torch.zeros(C).cuda()  # 空区域用0向量代替
                else:
                    patch = low_image_feature[:, y1:y2+1, x1:x2+1]  # C, h, w
                    patch = patch.mean(dim=(1, 2))  # 对 h, w 维度平均 -> (C,)
                
                
                
                features.append(patch)

            features = torch.stack(features, dim=0)  # shape = (300, 256)

         












    
            
            


            


            
            pre_bbox=box_xyxy_to_cxcywh(pre_bbox)
        
    


            
            boxes = pre_bbox / resize_factor  # (cx, cy, w, h)
            boxes=clip_box_tensor(self.map_box_back_tensor(boxes, resize_factor), H, W, margin=10) # x , y , w ,h
            

            
        
        
            

            
            gt_bbox=info['gt_bbox']

            
            
         
            boxes=torch.Tensor(boxes)
            gt_bbox=torch.tensor(gt_bbox)
            #gt_bbox=self.kf.predict(W=W,H=H)
            #gt_bbox=box_cxcywh_to_xywh(gt_bbox)
            
    
            
            #关键在score
            #基于文本的动态特征选择
            
            pre_kf_bbox=gt_bbox

            num_boxes,num_channel=boxes.shape
            gt_bbox=gt_bbox[None].repeat(num_boxes,1)
            iou,union=box_iou(box_xywh_to_xyxy(boxes).reshape(-1,4),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))

            
        
     




    


            selected_index=0
            selected_iou=0
            pre_scores=pred_instances.scores
            labels=pred_instances.labels







            new_score=0.10*iou+0.90*pre_scores
    
                    
            
          
            selected_index=torch.argmax(iou[0:len_selected])




            cos_sim=None
            if len(self.memory_feature)<memory_len:
                self.memory_feature.append(features[0])
            else:
                vec_stack = torch.stack(self.memory_feature, dim=0)

                # 按行求平均，得到一个 shape 为 (256,) 的平均向量
                avg_vector = vec_stack.mean(dim=0)
                # 计算余弦相似度，dim=1 表示在特征维度上计算
                cos_sim = F.cosine_similarity(features, avg_vector, dim=1)  # shape: (300,)

             
              
                self.memory_feature.pop(1)
                self.memory_feature.append(features[selected_index])
        




            print(f'--------------------------')




            
            if cos_sim is not None:
                print(f'cos_sim:{cos_sim}')
                cos_sim=0.6*cos_sim+pre_scores[0:len_selected].cuda()*0.4

                selected_index=torch.argmax(cos_sim)

                selected_index=selected_index
                selected_cos_sim=cos_sim[selected_index]
                print(f'selected_cos_sim:{selected_cos_sim}')

            selected_index=torch.argmax(iou)
            selected_label=labels[selected_index]
            selected_score=pre_scores[selected_index]
            selected_iou=iou[selected_index]

            print(f'selected_index:{selected_index}')
            print(f'selected_score:{selected_score}')
            print(f'pre_kf_box:{pre_kf_bbox}')
            print(f'selected_label:{selected_label}')
            print(f'selected_iou:{selected_iou}')

            print(f'--------------------------')


                                

        
  
        
            
            


            
            
            
            
   
            pre_bbox=pred_instances.bboxes[selected_index]
            
            
            pre_scores=pred_instances.scores[selected_index]
            
            
            pre_bbox=pre_bbox
            pre_bbox=box_xyxy_to_cxcywh(pre_bbox)
            
            boxes = (pre_bbox / resize_factor).tolist()  # (cx, cy, w, h)

            boxes=clip_box(self.map_box_back(boxes, resize_factor), H, W, margin=10) # x , y , w ,h

            
            vis_boxes=torch.Tensor(boxes)

            
            vis_pre_kf_bbox=torch.Tensor(pre_kf_bbox)
          
        
        
        
        
        annotated_frame = annotate(image_source=image, boxes=vis_boxes[None], logits=pre_scores.reshape(1), phrases=texts[0],gt_boxes=torch.Tensor(info['gt_bbox']).reshape(1,-1))
        output_path=os.path.join('work_dir',"test_otb_dataset")
        pic_path=output_path
        iou_path="iou_result.txt"
        if frame_path is not None:
            file_name=frame_path.split('/')[-1]
            seq_name=info['seq_name']
            pic_path=os.path.join('work_dir',seq_name+'com')
            if os.path.exists(pic_path):
                pass
            else:
                os.makedirs(pic_path)
            file_name=file_name
            pic_path=os.path.join(seq_name+'com',file_name)
            iou_path=os.path.join(seq_name+'com',iou_path)
        output_path=os.path.join('work_dir',pic_path)
        iou_path=os.path.join('work_dir',iou_path)
        cv2.imwrite(output_path, annotated_frame)      
        

            

            
            
                            
        
        
        
        
        
        
        
        gt_bbox=info['gt_bbox']
        boxes=torch.Tensor(boxes)
        iou,union=box_iou(box_xywh_to_xyxy(boxes).reshape(-1,4),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
        print(f'iou:{iou}') 


        # with open(iou_path,"a") as f:
        #     f.write(str(iou.item())+'\n')
        
    
        



        



        self.state= boxes


        self.kf.update( box_xywh_to_cxcywh(self.state).numpy())

        self.state= boxes.numpy()


   
        return {"target_bbox": self.state}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def yolo_track_tracking_spacial(self, image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info: dict = None,frame_path=None):
        
        
    
        H, W, _ = image.shape#480 684 3
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process_yolo(x_patch_arr)[0]#1 3 256 256
        
        #这里有问题
        
        
        
    
    
    
    
        with torch.no_grad():

            
            
            
            
            if TEXT_PROMPT.endswith('.txt'):
                with open(TEXT_PROMPT) as f:
                    lines = f.readlines()
                texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
            else:
                texts = [t.strip() for t in TEXT_PROMPT.split(',')]#texts后面需要加一个' '
                texts=[["".join(texts)]]
                texts.append([' '])
            
            
            
            
            
            
            
            
            data_info = dict(img_id=0, img_path=frame_path, texts=texts)#图像路径:str 文本:两个list ['prompt']+['']
            
            
            
            template = self.template
        
            data_info = self.test_pipeline(data_info)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                        
            
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_info['data_samples']])
            data_samples=data_info['data_samples']
            data_samples.set_metainfo(dict(ori_shape=(self.params.search_size,self.params.search_size)))
            data_samples.set_metainfo(dict(pad_param=np.array([0.0,0.0,0.0,0.0])))
            
            data_samples.set_metainfo(dict(scale_factor=(1.0,1.0)))
            data_batch = dict(inputs=search.unsqueeze(0),
            data_samples=[data_samples])
      
           
            output = model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
            

            pre_bbox=pred_instances.bboxes[0]
            pre_bbox_anno=pre_bbox
            pre_scores=pred_instances.scores[0]
            
            
            pre_bbox=pre_bbox
            pre_bbox=box_xyxy_to_cxcywh(pre_bbox)
            
            boxes = (pre_bbox / resize_factor).tolist()  # (cx, cy, w, h)
            boxes=clip_box(self.map_box_back(boxes, resize_factor), H, W, margin=10) # x , y , w ,h
            
            
            

            
            
                            
        
        
        
        
        
        
        
        gt_bbox=info['gt_bbox']
        boxes=torch.Tensor(boxes)
        iou,union=box_iou(box_xywh_to_xyxy(boxes).reshape(-1,4),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
        print(f'iou:{iou}') 
        
        self.state= boxes.numpy()
        

   
        return {"target_bbox": self.state}
    
    
    
    
    
    
    
    
    
    def dino_track_memory(self, image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info: dict = None,frame_path:str=None):
        
        
        
        







        H, W, _ = image.shape#480 684 3
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)[0]#1 3 256 256
        _,h,w=search.shape
        with torch.no_grad():
            
            template = self.template
            # if self.cfg.TEST.MODE == 'NL':
            #     template=None
            boxes, logits, phrases,memory_query= predict(
                            model=model,
                            image=search,
                            template=template,
                            caption=TEXT_PROMPT,
                            box_threshold=BOX_TRESHOLD,
                            text_threshold=TEXT_TRESHOLD, #boxes 1 4； logits 1 ；phrases white airplane landing 按照顺序排列
                            memeory_query=self.memory_query,
                            MEMORY_SEQENCE=True
                            
                        )
        
            
            
            gt_bbox=info['gt_bbox']

            # if sum(v == 0 for v in gt_bbox) >= 3:#如果真值无效
            #     self.state=gt_bbox+0.1
            #     return {"target_bbox": self.state}


            boxes = (boxes * self.params.search_size / resize_factor) # (cx, cy, w, h)
            boxes=clip_box_tensor(self.map_box_back_tensor(boxes, resize_factor), H, W, margin=10) # x, y , w ,h

            boxes=torch.Tensor(boxes)
            #gt_bbox=self.kf.predict(W=W,H=H)
            gt_bbox=torch.tensor(gt_bbox)
            pre_kf_bbox=gt_bbox
            num_boxes,num_channel=boxes.shape
            gt_bbox=gt_bbox[None].repeat(num_boxes,1)
            iou,union=box_iou(box_xywh_to_xyxy(boxes).reshape(-1,4),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
    
            







            selected_index=torch.argmax(iou)
            

            boxes=boxes[selected_index:selected_index+1]
            logits=logits[selected_index:selected_index+1]
            selected_score=logits
            phrases=phrases[selected_index:selected_index+1]  #cx,cy,w,h
            vis_boxes=boxes
            



        
        annotated_frame = annotate(image_source=image, boxes=vis_boxes, logits=logits, phrases=phrases,gt_boxes=torch.Tensor(info['gt_bbox']).reshape(1,-1),vis_pre_kf_bbox=pre_kf_bbox[None])
    
        output_path=os.path.join('work_dir','test_dinotrackerotb99_tracking.jpg')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        # cv2.imwrite(output_path, annotated_frame)
        
        gt_bbox=info['gt_bbox']
        iou,union=box_iou(box_xywh_to_xyxy(torch.Tensor(boxes).reshape(-1,4)),box_xywh_to_xyxy(torch.Tensor(gt_bbox)).reshape(-1,4))
        print(f'iou:{iou}') 

        self.state=boxes[0].tolist()

        self.kf.update(np.array(self.state))


        
        self.memory_query=memory_query
                    
        return {"target_bbox": self.state}
    
    

    def save_visualization(self, image, vis_info):
        # save_name = os.path.join(self.save_dir, self.params.yaml_name+'_vis', vis_info['name'], '%.4d'%self.frame_id)
        save_name = self.save_dir
        if not os.path.exists(os.path.join(save_name)):
            os.makedirs(save_name)

        color = [(255, 0, 0), (0, 255, 0)]

        for img, name, bbox in zip(vis_info['patches'], vis_info['patches_name'], vis_info['patches_bbox']):
            x, y, w, h = bbox
            img = cv2.rectangle(img, (int(x), int(y)),(int(x+w), int(y+h)), color[0], 2)
            plt.imsave(os.path.join(save_name, f'{name}.png'), img)

        for i, img in enumerate(vis_info['cls_map']):
            img = cv2.resize(img.numpy(), (200, 200))
            plt.imsave(os.path.join(save_name, f'clsmap_{i}.png'), img)

        for i, vis_bbox in enumerate(vis_info['image_bbox']):
            x, y, w, h = vis_bbox
            image = cv2.rectangle(image, (int(x), int(y)),(int(x+w), int(y+h)), color[i], 2)
        scale = 400/max(image.shape[:2])
        dh, dw = image.shape[:2]
        image = cv2.resize(image, (int(dw*scale), int(dh*scale)))
        plt.imsave(os.path.join(save_name, 'image_bbox.jpg'), image)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
    
    
    def map_box_back_tensor(self, pred_boxes: torch.Tensor, resize_factor: float):
        """
        pred_boxes: Tensor of shape [N, 4] in cx, cy, w, h
        return: Tensor of shape [N, 4] in x, y, w, h
        """
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        half_side = 0.5 * self.params.search_size / resize_factor

        cx = pred_boxes[:, 0]
        cy = pred_boxes[:, 1]
        w = pred_boxes[:, 2]
        h = pred_boxes[:, 3]

        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)

        x = cx_real - 0.5 * w
        y = cy_real - 0.5 * h

        return torch.stack([x, y, w, h], dim=1)
    

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
        
    def anno2mask(self, gt_bboxes, size):
        bboxes = box_xywh_to_xyxy(gt_bboxes)*size # b, 4
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1)+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask)

        cx = ((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = ((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask[bid, cy, cx] = True
        return mask.flatten(1).cuda()
    
    def extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")#给每个token前面加一个cls后面加一个seq选项
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)#将token转化为id

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:#将长度进行补全
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return torch.tensor(input_ids).unsqueeze(0).cuda(), torch.tensor(input_mask).unsqueeze(0).cuda()


def get_tracker_class():
    return UVLTrack
