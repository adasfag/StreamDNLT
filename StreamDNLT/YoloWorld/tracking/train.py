import os
import time
import argparse

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='uvltrack',help='training script name')
    parser.add_argument('--config', type=str, default='baseline_base', help='yaml configure file name')
    parser.add_argument('--stage1_model', type=str, default=None, help='stage1 model used to train SPM.')
    parser.add_argument('--save_dir', type=str,default='./work_dir/bs8gpus2lr0.0004_alldataset/',help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="single",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--master_port', type=int, help="master port", default=26500)
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')
    
    parser.add_argument('--port', type=int, default=-1,help='the port is used for muti gpu train')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = "python lib/train/run_training.py --script %s --config %s --save_dir %s --use_lmdb %d " \
                    "--script_prv %s --config_prv %s --distill %d --script_teacher %s --config_teacher %s --stage1_model %s" \
                    % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                       args.distill, args.script_teacher, args.config_teacher, args.stage1_model)
    elif args.mode == "multiple":
        if args.port==-1:
            port = int(time.time()*1000)%20000+10000
            args.port=port
        train_cmd = f"python -m torch.distributed.launch  --nnodes=1 --master_port %d --nproc_per_node %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s  " \
                    "--distill %d --script_teacher %s --config_teacher %s --stage1_model %s"  \
                    % (args.port, args.nproc_per_node, args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv,
                       args.config_prv, args.distill, args.script_teacher, args.config_teacher, args.stage1_model)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
