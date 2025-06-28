import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
from lib.groundingdino.util.inference import load_model, load_image, predict, annotate,load_model_from_fine_tune


import torch

from torchvision.ops import box_convert


from mmengine.config import Config
from mmyolo.registry import RUNNERS,MODELS



import os.path as osp


from mmdet.apis import init_detector


def Trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of Trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only, eval=True) for run_id in run_ids]


class Tracker:
    """Wraps the Tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, eval=False, epoch=None):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.params = self.get_parameters(epoch) if not eval else None
        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        Tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(Tracker_module_abspath):
            Tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.Tracker_class = Tracker_module.get_tracker_class()
        else:
            self.Tracker_class = None
        self.model=None

    def create_Tracker(self, params):
        Tracker = self.Tracker_class(params, self.dataset_name)
        return Tracker

    def run_sequence(self, seq, debug=None,test_dino=False,test_yolo=False):
        """Run Tracker on sequence.
        args:
            seq: Sequence to run the Tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """

        debug_ = debug
        if debug is None:
            debug_ = getattr(self.params, 'debug', 0)
        self.params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        Tracker = self.create_Tracker(self.params)

        output = self._track_sequence(Tracker, seq, init_info, debug,test_dino=test_dino,test_yolo=test_yolo)
        return output

    def _track_sequence(self, Tracker, seq, init_info, debug,test_dino=False,test_yolo=False):
        # Define outputs
        # Each field in output is a list containing Tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if Tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(Tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = Tracker_out.get(key, defaults.get(key, None))
                if key in Tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])









        start_time = time.time()
        init_info['seq_name'] = seq.name
        init_info['language'] = seq.language
        
        
        
        
        
        
        
        
        
        
        
        if test_dino:
            
            
            
            
            
            
            
            
            
            if self.model is None:  
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/checkpoints/train/uvltrack/baseline_base_dino/GroundingDINOTracker_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrack_full_finetune_classloss_add_template/checkpoints/train/uvltrack/baseline_base_dino/GroundingDINOTracker_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrack_full_finetune_classloss_add_template_and_adapter_epoch50/checkpoints/train/uvltrack/baseline_base_dino/GroundingDINOTrackerAdapter_ep0050.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_size32/checkpoints/train/uvltrack/baseline_base_dino/GroundingDINOTrackerAdapter_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_epoch10_add_memory/checkpoints/train/uvltrack/baseline_base_dino/GroundingDINOTrackerAdapter_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_size32_epoch60/checkpoints/train/uvltrack/baseline_base_dino/GroundingDINOTrackerAdapter_ep0060.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_query_6/checkpoints/train/uvltrack/baseline_base_dino/GroundingDINOTrackerAdapter_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_query_6_nomemory/checkpoints/train/uvltrack/baseline_base_dino/GroundingDINOTrackerAdapter_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_query_6_memory/checkpoints/train/uvltrack/baseline_base_dino_memory/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_memory_moreframe/checkpoints/train/uvltrack/baseline_base_dino_memory/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_memory_moreframe_tgt/checkpoints/train/uvltrack/baseline_base_dino_memory/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_memory_moreframe_tgt_add_loss/checkpoints/train/uvltrack/baseline_base_dino_memory/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_memory_moreframe_query12/checkpoints/train/uvltrack/baseline_base_dino_memory/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_memory_moreframe_query3_allntldataset/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinotrackaddtemplate_adapter_memory_moreframe_query3_samplevideo/checkpoints/train/uvltrack/baseline_base_dino_memory/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinB_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwb_allnltdataset_epoch10_samplevideo_en/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwt_otb_epoch10_new_match_query16_en/checkpoints/train/uvltrack/baseline_base_dino_memory_otb/GroundingDINOTrackerAdapterMemory_ep0005.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwt_otb_epoch10_query16_en/checkpoints/train/uvltrack/baseline_base_dino_memory_otb/GroundingDINOTrackerAdapterMemory_ep0005.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwt_otb_epoch10_query16_optimisele_en/checkpoints/train/uvltrack/baseline_base_dino_memory_otb/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwt_otb_epoch10_query16_adapter_backbone_en/checkpoints/train/uvltrack/baseline_base_dino_memory_otb/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwt_otb_epoch10_query3_en/checkpoints/train/uvltrack/baseline_base_dino_memory_size_320_otb/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwt_alldataset_epoch10_query6_en/checkpoints/train/uvltrack/baseline_base_dino_memory_alldataset/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                
                
                
                
                
                
                
                
                
                
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwt_allnltdataset_epoch10_query6_en/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwt_allnltdataset_addsize_epoch10_query6_en/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch10_query3_samplevideo_fixrefeccocoproblem/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                
                
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinB_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_memory_siwb_allnltdataset_epoch160_samplevideo_en/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0060.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch10_query3_samplevideo_reducerefcoco/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch10_query3_samplevideo_joint/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_joint/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch10_query3_samplevideo_addmatch/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch10_query3_samplevideo_original/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_joint/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch10_query3_samplevideo_consistant/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset/GroundingDINOTrackerAdapterMemory_ep0009.pth.tar")
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch160_query16_samplevideo/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_joint/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                
                
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch160_query16_samplevideo_v10/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_joint/GroundingDINOTrackerAdapterMemory_ep0060.pth.tar")
                
                
                 
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch160_query16_samplevideo_v10/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_joint/GroundingDINOTrackerAdapterMemory_ep0060.pth.tar")
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch160_query16_samplevideo_v10/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_joint/GroundingDINOTrackerAdapterMemory_ep0160.pth.tar")
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_lasot_epoch10_query6_samplevideo/checkpoints/train/uvltrack/baseline_base_dino_memory_lasot/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_lasot_epoch10_query6_samplevideo_template256/checkpoints/train/uvltrack/baseline_base_dino_memory_lasot/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                 
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_memory_siwt_allnltdataset_epoch10_query6_samplevideo_v0_match/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_joint/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_allnltdataset_epoch60/checkpoints/train/uvltrack/baseline_base_dino_memory_allnltdataset_joint/GroundingDINOTrackerAdapterMemory_ep0060.pth.tar")
                
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_lasot_epoch60/checkpoints/train/uvltrack/baseline_base_dino_memory_lasot/GroundingDINOTrackerAdapterMemory_ep0060.pth.tar")
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_lasot_epoch10_query3/checkpoints/train/uvltrack/baseline_base_dino_memory_lasot/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_lasot_epoch10_query12/checkpoints/train/uvltrack/baseline_base_dino_memory_lasot/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                
                
        
        
        
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_adapter_optimise.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/dinoTracker_lasot_epoch10_query16/checkpoints/train/uvltrack/baseline_base_dino_memory_lasot/GroundingDINOTrackerAdapterMemory_ep0010.pth.tar")
                 
        
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_finetune.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_finetune/checkpoints/train/uvltrack/baseline_base_dino_finetune/GroundingDINOTracker_ep0050.pth.tar")
                
                #model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_finetune.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_finetune/checkpoints/train/uvltrack/baseline_base_dino_finetune/GroundingDINOTracker_ep0060.pth.tar")
                
                model = load_model_from_fine_tune("/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/groundingdino/config/GroundingDINO_SwinT_OGC_track_finetune.py", "/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_dinoTracker_finetune_addspatialadapter/checkpoints/train/uvltrack/baseline_base_dino_finetune/GroundingDINOTracker_ep0060.pth.tar")
                
                
                

                

    
                
                
                print('-----------------------------------------------it is testing dino--------------------------------------------------------------------------')
                self.model=model
            else:
                model=self.model
        elif test_yolo:
            if self.model is None:
        
                config_path="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
                
                # load config
                cfg = Config.fromfile(config_path)

                cfg.work_dir = osp.join('./work_dirs',
                                        osp.splitext(osp.basename(config_path))[0])

                



                
                
                
                
                
                
                
                
                

                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/pre_train/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain-8ff2e744.pth"    
                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yoloTracker_finetunetrack/checkpoints/train/uvltrack/baseline_base_yolo_finetune/YOLOWorldDetector_ep0010.pth.tar"

                
            
                
                
                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yoloTracker_finetune_temporaltrying/checkpoints/train/uvltrack/baseline_base_yolo_finetune_temporal/YOLOWorldDetector_ep0010.pth.tar" #全picture方式训练，引入时序了，lasot数据集训练epoch 10

                
                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporallasotepoch30_dev_correct/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_dev/YOLOWorldDetector_ep0030.pth.tar" #全picture方式训练，引入时序了，lasot数据集训练epoch 30

                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporaltnl2kepoch30_dev_correct/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_dev/YOLOWorldDetector_ep0030.pth.tar" #全picture方式训练，引入时序了，tnl2k数据集训练epoch 30
                
                
                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporalotb99epoch30_dev_correct/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_dev/YOLOWorldDetector_ep0030.pth.tar" #全picture方式训练，引入时序了，otb99数据集训练epoch 30

                
                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporaljointepoch10_dev_correct_tracking/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_tracking/YOLOWorldDetector_ep0010.pth.tar" #tracking方式训练，引入空间和时序，lasot数据集训练epoch 10
                
                
                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporallasotepoch30_dev_correct_tracking/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_tracking/YOLOWorldDetector_ep0030.pth.tar" #tracking方式训练，引入空间和时序，lasot数据集训练epoch 30

                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporaljointepoch30_dev_correct_tracking/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_tracking/YOLOWorldDetector_ep0030.pth.tar" #tracking方式训练，引入空间和时序，joint数据集训练epoch 30

                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporalotb99epoch30_dev_correct_tracking/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_tracking_size320/YOLOWorldDetector_ep0030.pth.tar" #tracking方式训练，引入空间和时序，joint数据集训练epoch 30

                
                checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporaljointepoch30_dev_correct_tracking/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_tracking_size320/YOLOWorldDetector_ep0030.pth.tar" #tracking方式训练，引入空间和时序，joint数据集训练epoch 30

                
                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetuneepoch30/checkpoints/train/uvltrack/baseline_base_yolo_finetune/YOLOWorldDetector_ep0030.pth.tar" #tracking方式训练，全量微调，joint数据集训练epoch 30

                
                #checkpoint="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/work_dir/baseline_yolotracker_finetune_spatial_adn_temporaljointepoch30_dev_correct_trackingfromescratch/checkpoints/train/uvltrack/baseline_base_yolo_finetune_spatial_and_temporal_joint_tracking_size320/YOLOWorldDetector_ep0030.pth.tar" #tracking方式训练，引入空间和时序，joint数据集训练epoch 30

 
   
            
            
              
     
                    
             
            
            
            
            
            
          
                model = init_detector(cfg, checkpoint=checkpoint, device="cuda:0")
                
            
                
    
                model.eval()
                #self.model=model
                
            else:
                model=self.model
                
        
        else:
            print('-----------------------------------------------it is testing uvltracking---------------------------------------------------------------------------------')
        
        
        
        
        
        
        
        
        
        
        if test_dino:
            init_info['model']=model
            out = Tracker.initialize_dino(image, init_info,seq.frames[0])
        elif test_yolo:
            init_info['model']=model
            out = Tracker.initialize_yolo(image, init_info,seq.frames[0])        
        else:
            out = Tracker.initialize(image, init_info)
        
        
        if out is None:
            out = {}

        debug_ground = False
        if not debug_ground:
            prev_output = OrderedDict(out)
            init_default = {'target_bbox': init_info.get('init_bbox'),
                            'time': time.time() - start_time}
            if Tracker.params.save_all_boxes:
                init_default['all_boxes'] = out['all_boxes']
                init_default['all_scores'] = out['all_scores']

            _store_outputs(out, init_default)

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            for frame_num, frame_path in enumerate(seq.frames[1:], start=1):#第0帧默认不做Tracker
                if test_dino or test_yolo:
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    try: 
                        TEXT_PROMPT = seq.language #描述的是整个视频场景而不是单个图片
                        IMAGE_PATH=frame_path
                        BOX_TRESHOLD = 0.00
                        TEXT_TRESHOLD = 0.00
                        image = self._read_image(frame_path)
                        start_time = time.time()
                        info = seq.frame_info(frame_num)
                        info['previous_output'] = prev_output
                        info['gt_bbox'] = seq.ground_truth_rect[frame_num] if len(seq.ground_truth_rect) > frame_num else [0, 0, 0, 0]
                        info['seq_name'] = seq.name
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        if test_dino:
                            out = Tracker.dino_track(image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info,frame_path)#dict target_bbox [lx,ly,w,h]
                            #out = Tracker.dino_track_memory(image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info,frame_path)#dict target_bbox [lx,ly,w,h]
                        elif test_yolo:
                            #out = Tracker.yolo_track(image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info,frame_path)#全图像跟踪
                            #out = Tracker.yolo_track_tracking(image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info,frame_path)#单目标跟踪
                            out = Tracker.yolo_track_tracking_template(image,model, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD,info,frame_path)#引入模板的单目标跟踪
                            
                            
                        else:
                            raise Exception("Now test_dino or test_yolo is not specific,the finetune must find code")
                        print(f'finish the process of {frame_num} picture')
                    except Exception as e:
                        print(e)
                        self.state= info['gt_bbox']
                        out={"target_bbox": self.state}
                
                else: 
                    image = self._read_image(frame_path)

                    start_time = time.time()

                    info = seq.frame_info(frame_num)
                    info['previous_output'] = prev_output
                    info['gt_bbox'] = seq.ground_truth_rect[frame_num] if len(seq.ground_truth_rect) > frame_num else [0, 0, 0, 0]
                    info['seq_name'] = seq.name

                    out = Tracker.track(image, info)#dict target_bbox [lx,ly,w,h]
                    # print(f'finish the process of {frame_num} picture')
        
                            
                        
                prev_output = OrderedDict(out)
                _store_outputs(out, {'time': time.time() - start_time})

            for key in ['target_bbox', 'all_boxes', 'all_scores']:
                if key in output and len(output[key]) <= 1:
                    output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the Tracker with the vieofile.
        args:
            debug: Debug level.
        """

        # params = self.get_parameters()
        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.Tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.Tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            Tracker = self.create_Tracker(params)

        elif multiobj_mode == 'parallel':
            Tracker = MultiObjectWrapper(self.Tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []
        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + Tracker.params.Tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        # cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            Tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            raise NotImplementedError("We haven't support cv_show now.")

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = Tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                # cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                Tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self, epoch=None):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name, epoch=epoch)
        return params


    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")