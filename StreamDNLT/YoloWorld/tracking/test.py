import os




import sys



import argparse
import importlib

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
    
    


    


from lib.test.evaluation import get_dataset









from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker_tracking_tnl2k import Tracker

import warnings
warnings.filterwarnings('ignore')













import torch


torch.cuda.set_device(0)


from mmyolo.utils import register_all_modules

register_all_modules()


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """
    
    
    
    
    
    
    

    dataset = get_dataset(dataset_name)
    if sequence is not None:
        dataset = [dataset[sequence]]
    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus,test_dino=False,test_yolo=True)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', default="uvltrack", type=str, help='Name of tracking method.')
    parser.add_argument('--tracker_param', default="streamdnlt_yoloworld_tracking", type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='tnl2k', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=1, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=2)

    parser.add_argument('--params__model', type=str, default=None, help="Tracking model path.")
    parser.add_argument('--params__update_interval', type=int, default=None, help="Update interval of online tracking.")
    parser.add_argument('--params__online_sizes', type=int, default=None)
    parser.add_argument('--params__search_area_scale', type=float, default=None)
    parser.add_argument('--params__max_score_decay', type=float, default=1.0)
    parser.add_argument('--params__vis_attn', type=int, choices=[0, 1], default=0, help="Whether visualize the attention maps.")

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()
