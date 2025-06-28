# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS,MODELS
from mmyolo.utils import is_metainfo_lower


import torch

from mmyolo.utils import register_all_modules

register_all_modules()





from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

import sys
sys.path.append("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master")
sys.path.append("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib")

from lib.groundingdino.util.inference import load_model, load_image_fromimagepath, predict, annotate


from mmdet.apis import init_detector





import cv2






from mmdet.utils import get_test_pipeline_cfg


from mmengine.dataset import Compose


import numpy as np


def main():

    # load config
    
    config_path="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    
    # load config
    cfg = Config.fromfile(config_path)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(config_path))[0])

    

    

    checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/pre_train/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain-8ff2e744.pth"    
    model = init_detector(cfg, checkpoint=checkpoint, device="cuda:0")
    
    
    IMAGE_PATH= "/home/share/hhd/dataset/lgt/uvltrack_work/data/lasot/airplane/airplane-1/img/00000001.jpg"
    
    
    texts="white airplane landing on the ground"
    
    output_path="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/tools/build_model/output/anno.jpg"
    
    
    
    
    
    if texts.endswith('.txt'):
        with open(texts) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in texts.split(',')] + [[' ']]#texts后面需要加一个' '
    
    
    
    
    
    data_info = dict(img_id=0, img_path=IMAGE_PATH, texts=texts,gt_bboxes=np.array([[1,3,6,60]]))#图像路径:str 文本:两个list ['prompt']+['']
    
    
    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)
    
    
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    
    

   
    model.eval()
    output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    

    pre_bbox=pred_instances.bboxes[0]
    pre_bbox_anno=pre_bbox
    pre_scores=pred_instances.scores[0]
    
    
    image_source = cv2.imread(IMAGE_PATH)
    
    annotated_frame = annotate(image_source=image_source, boxes=pre_bbox_anno[None].detach() , logits=[pre_scores], phrases=[texts[0]])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    cv2.imwrite(output_path, annotated_frame)
    print(pre_bbox)
    

if __name__ == '__main__':
    main()
