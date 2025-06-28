import os
# loss function related
from lib.utils.box_ops import giou_loss, GaussWeightedLoss
from torch.nn.functional import l1_loss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
import lib.models
import lib.train.actors
# for import modules
import importlib
from lib import registry




from lib.groundingdino.util.inference import load_model, load_image, predict, annotate
from mmengine.config import Config
from mmyolo.registry import RUNNERS,MODELS






from mmyolo.utils import register_all_modules

register_all_modules()

from mmdet.apis import init_detector

import os.path as osp





def run(settings):
    settings.description = 'Training script for Mixformer'



    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)#导入config文件
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)#用yarm的配置覆盖config

    
    # update settings based on cfg
    update_settings(settings, cfg)#用config属性覆盖settings

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')#logs文件路径
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))#log文件路径

    
    
    










    # Build dataloaders
    loader_list = build_dataloaders(cfg, settings)#建设5个数据加载器dataload
    loader_list=[loader_list[0]]#仅仅 训练 train 数据了

    # Create network
    
    
    
    
    
    
    
    
    
    # load config
    
    device="cuda:0"
    
    config_path="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    
    # load config
    cfg_model = Config.fromfile(config_path)

    cfg_model.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(config_path))[0])

    

    

    checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/pre_train/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain-8ff2e744.pth"    
    net = init_detector(cfg_model, checkpoint=checkpoint, device=device)
    
    
   
   
   
   

    # wrap networks to distributed one
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank],  output_device=settings.local_rank,find_unused_parameters=True)    
        net._set_static_graph()
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device(device)
        
    # settings.save_every_epoch = True
    actor = registry.ACTORS["yolotracker_finetune_temporal"](net, cfg)#将net cfg用一个actor封装

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)#优化器，优化安排
    trainer = LTRTrainer(actor, loader_list, optimizer, settings, lr_scheduler, use_amp=False)#将数据加载、模型执行、优化器、优化安排和参数封装为一个训练器
    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
