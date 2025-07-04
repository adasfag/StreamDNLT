import os
import sys
import random
import argparse
import importlib
import cv2 as cv
import _init_paths
import numpy as np
import torch.backends.cudnn
import torch.distributed as dist
torch.backends.cudnn.benchmark = False
import lib.train.admin.settings as ws_settings

import warnings
warnings.filterwarnings('ignore')
import torch.distributed as dist
torch.cuda.set_device(0)
def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None,
                 use_lmdb=False, script_name_prv=None, config_name_prv=None,
                 distill=None, script_teacher=None, config_teacher=None, stage1_model=None):
    """Run the train script.
    args:
        script_name: Name of emperiment in the "experiments/" folder.
        config_name: Name of the yaml file in the "experiments/<script_name>".
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """
    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")
    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    if int(os.environ["LOCAL_RANK"]) <= 0:
        print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))
    
    print(int(os.environ["LOCAL_RANK"]) )

    '''2021.1.5 set seed for different process'''
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)            
    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.stage1_model = stage1_model
    settings.project_path = 'train/{}/{}'.format(script_name, config_name)
    if script_name_prv is not None and config_name_prv is not None:
        settings.project_path_prv = 'train/{}/{}'.format(script_name_prv, config_name_prv)
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    settings.use_lmdb = use_lmdb
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))
    expr_module = importlib.import_module('lib.train.train_script_yolotrack_finetune_temporal')
    expr_func = getattr(expr_module, 'run')
    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, default="uvltrack", required=False, help='Name of the train script.')
    parser.add_argument('--config', type=str, default="streamdnlt_yoloworld_tracking", required=False, help="Name of the config file.")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    
    
    
    

    parser.add_argument('--save_dir', type=str, default="./work_dir/yolotacker", help='the directory to save checkpoints and logs')
    
    



    
    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, default=None, help='Name of the train script of previous model.')
    parser.add_argument('--config_prv', type=str, default='baseline', help="Name of the config file of previous model.")
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, default=None,help='teacher script name')
    parser.add_argument('--config_teacher', type=str, default=None, help='teacher yaml configure file name')
    parser.add_argument('--stage1_model', type=str, default=None, help='stage1 model used to train SPM.')
    parser.add_argument('--local_rank', type=int, default=0, help='which is used for muti gpus.')
    
    
    args = parser.parse_args()
    os.environ['LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '-1')
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://')
        
        
        
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)
        
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=local_rank, save_dir=args.save_dir, base_seed=args.seed,
                 use_lmdb=args.use_lmdb, script_name_prv=args.script_prv, config_name_prv=args.config_prv,
                 distill=args.distill, script_teacher=args.script_teacher, config_teacher=args.config_teacher,
                 stage1_model=args.stage1_model)


if __name__ == '__main__':
    main()
