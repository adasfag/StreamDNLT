import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
from .utils import SimpleTokenizer

from pytorch_pretrained_bert import BertTokenizer

def no_processing(data):
    return data

from mmdet.utils import get_test_pipeline_cfg



from mmengine.dataset import Compose
from mmengine.config import Config
import os.path as osp




import torchvision.transforms as transforms


class GroundingAndTrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    [base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, grounding_processing=None, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5, bert_path=None, mode='joint', grounding_ratio=None, vl_ratio=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.train_lang = False
        self.datasets = datasets
        self.train_cls = train_cls  
        self.pos_prob = pos_prob  
        self.mode = mode
        if mode == 'joint':
            assert grounding_ratio is not None
            assert vl_ratio is not None
            self.p_tracking = 1-grounding_ratio-vl_ratio
            self.p_vl = vl_ratio
            self.p_grounding = grounding_ratio
        elif mode == 'tracking':
            self.p_tracking = 1.0
            self.p_vl = 0.0
            self.p_grounding = 0.0
        elif mode == 'grounding':
            self.p_tracking = 0.0
            self.p_vl = 0.0
            self.p_grounding = 1.0

        
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]


        
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.num_grounding_frames = 1
        self.processing = processing
        self.grounding_processing = grounding_processing
        self.frame_sample_mode = frame_sample_mode
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
        
        
        self.tracking_dataset = [d for d in self.datasets if d.is_tracking_sequence()]
        
        self.p_tracking_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_tracking_sequence()]
        
        
        
        
        self.grounding_dataset = [d for d in self.datasets if d.is_grounding_sequence()]
        self.p_grounding_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_grounding_sequence()]
        
        self.vl_dataset = [d for d in self.datasets if d.is_vl_sequence()]
        self.p_vl_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_vl_sequence()]
        
        
        
        
        
        
        
        config_path="/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master/lib/YOLO-World-master/configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        
        
        cfg = Config.fromfile(config_path)

        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(config_path))[0])
         
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        
        test_pipeline = Compose(test_pipeline_cfg)
        self.test_pipeline=test_pipeline
        
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        
        self.inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        
        
        
        
        
        
    def __len__(self):
        if self.mode == "grounding_test":
            return self.datasets[0].get_num_sequences()
        else:
            return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        
        if len(valid_ids) == 0 or (len(valid_ids) - num_ids)<1:
            return None
        
        start_index = random.randint(0, len(valid_ids) - num_ids)

        return valid_ids[start_index:start_index+num_ids]
    
    
    
    def _sample_visible_ids_bak(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)
    
    

    def __getitem__(self, index):
        
        if self.mode == "grounding_test":
            return self.sample_grounding_test(index)
        elif self.mode == "tracking_test":
            return self.sample_track_test()
        elif self.mode == "vl_test":
            return self.sample_vl_test()
        
        elif self.mode == 'tracking':
            return self.sample_track()
        elif self.mode == 'grounding':
            return self.sample_grounding()
        elif self.mode == 'joint':
            
            
            seed = random.random()
            if seed < self.p_tracking:
                return self.sample_track()
            elif seed < self.p_tracking + self.p_grounding:
                return self.sample_grounding()
            else:
                return self.sample_vl()
            
            
            
            
            
            
        else:
            raise ValueError(f"No {self.mode} mode!")

    def sample_track(self):
        """
        returns:
            TensorDict - dict containing all the data blocks 取样track
        """
        valid = False
        while not valid:
            
            dataset = random.choices(self.tracking_dataset, self.p_tracking_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()

            
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames) 
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0]) 
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames 
                
            template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if not is_video_dataset:
                language=template_anno.get("nlp",None)
                if language is not None:
                    language=language[0]
            if language is None or language == '':
                language = 'object, thing or stuff'
            ori_language=language
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'template_images': template_frames,
                                'template_anno': template_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language*self.num_search_frames,
                                'text_mask': mask*self.num_search_frames,
                                'flag': torch.tensor([[2]]),
                                'ori_language':ori_language
                                })
            
       
            data = self.processing.track_process_memory(data)
      
            valid = data['valid']
        del data['valid']
        return data

    def sample_vl(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            
            dataset = random.choices(self.vl_dataset, self.p_vl_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()

            
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                
                template_frame_ids = [1] * self.num_template_frames 
                search_frame_ids = [1] * self.num_search_frames 
            
            
            
            template_frames_input, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            search_frames_input, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
            
            
            
            data = TensorDict({ 'template_images': template_frames_input,
                                'template_anno': template_anno['bbox'],
                                'search_images': search_frames_input,
                                'search_anno': search_anno['bbox']
                                })
            
            
            
            
       
            data = self.processing.track_process_memory(data)
            
            search_images=data["search_images"]
            template_images= data["template_images"]
            
            
            if isinstance(template_images,list):
                continue
            
            
            template_images_track=template_images
            
            
            
            search_images_track=search_images
            
            
            
            
             
            
            
            search_anno_bbox=data["search_anno"]
            
            template_anno_bbox=data["template_anno"]
            
            
            
            
            
            
            

            
            
            
            template_frames, template_anno, meta_obj_train = dataset.get_frames_path(seq_id, template_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames_path(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if not is_video_dataset:
                language=template_anno.get("nlp",None)
                if language is not None:
                    language=language[0]
            if language is None or language == '':
                language = 'object, thing or stuff'
            ori_language=language
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False

    
            template_frames_input=template_images_track
            
            
            search_frames_data_samples=[]
            
            for search_frames_path in search_frames:
                if ori_language.endswith('.txt'):
                    with open(ori_language) as f:
                        lines = f.readlines()
                    texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
                else:
                    texts = [t.strip() for t in ori_language.split(',')]
                    texts=[["".join(texts)]]
                    texts.append([' '])
    
                
                
                data_info = dict(img_id=0, img_path=search_frames_path, texts=texts)
    
                
                data_info = self.test_pipeline(data_info)
                search_frames_data_samples.append(data_info["data_samples"])
            search_frames_input=search_images_track
        
                
                
                
                
                
                
                
                
            data = TensorDict({ 'template_images': template_frames_input,
                                'search_images': search_frames_input,
                                'template_anno': template_anno_bbox,
                                'search_anno': search_anno_bbox,
                                'text': language*self.num_search_frames,
                                'ori_language':ori_language,
                                "search_frames_data_samples":search_frames_data_samples
                                })
            if language is None:
                valid = False
            else:
                valid = True
  
           
        return data

    def sample_grounding(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            
            dataset = random.choices(self.grounding_dataset, self.p_grounding_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()
            
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            if is_video_dataset:
                grounding_frame_ids = None
                search_frame_ids = None
                gap_increase = 0
                MAX_N = 30
                while search_frame_ids is None:
                    if len(visible) < MAX_N:
                        MAX_N = len(visible)
                    base_frame_id = self._sample_visible_ids_bak(visible, num_ids=1,
                                                             min_id=self.num_grounding_frames - 1, 
                                                             max_id=MAX_N - self.num_search_frames + 1)
                    prev_frame_ids = self._sample_visible_ids_bak(visible, num_ids=self.num_grounding_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    grounding_frame_ids = base_frame_id + prev_frame_ids
                    search_frame_ids = self._sample_visible_ids_bak(visible, min_id=grounding_frame_ids[0] + 1, max_id=
                                                                grounding_frame_ids[0] + self.max_gap + gap_increase, 
                                                                num_ids=(self.num_search_frames-1))
                    
                    gap_increase += 5
            else:
                
                grounding_frame_ids = [1] * self.num_grounding_frames
                search_frame_ids = [1] * (self.num_search_frames-1)
                
            grounding_frames, grounding_anno, meta_obj_train = dataset.get_frames(seq_id, grounding_frame_ids, seq_info_dict) 
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if not is_video_dataset:
                language=grounding_anno.get("nlp",None)
                if language is not None:
                    language=language[0]
            if language is None or language == '':
                language = 'object, thing or stuff'
            ori_language=language
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'grounding_images': grounding_frames,
                                'grounding_anno': grounding_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language, 
                                'text_mask': mask*self.num_search_frames ,
                                'flag': torch.tensor([[1]]),
                                'ori_language':ori_language
                                })
            data = self.processing.grounding_process(data)
            valid = data['valid']
        del data['valid']
        return data

    def sample_vl_test(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            
            track_dataset = [d for d in self.datasets if d.is_video_sequence()]
            p_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_video_sequence()]
            dataset = random.choices(track_dataset, p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames 
                
            template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            ori_language=language
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'template_images': template_frames,
                                'template_anno': template_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language*2,
                                'text_mask': mask*2,
                                'flag': torch.tensor([[2]]),
                                'ori_language':ori_language
                                }) 
            data = self.processing.track_process(data)
            valid = data['valid']
        del data['valid']
        return data

    def sample_track_test(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            
            track_dataset = [d for d in self.datasets if d.is_video_sequence()]
            p_datasets = [p for i, p in enumerate(self.p_datasets) if self.datasets[i].is_video_sequence()]
            dataset = random.choices(track_dataset, p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames 
                
            template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            ori_language=language
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'template_images': template_frames,
                                'template_anno': template_anno['bbox'],
                                'search_images': search_frames,
                                'search_anno': search_anno['bbox'],
                                'text': language*2,
                                'text_mask': mask*2,
                                'flag': torch.tensor([[0]]),
                                'ori_language':ori_language
                                }) 
            data = self.processing.track_process(data)
            valid = data['valid']
        del data['valid']
        return data

    def sample_grounding_test(self, i):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        
        valid = False
        while not valid:
            dataset = self.datasets[0]
            seq_id, visible, seq_info_dict = self.get_seq_from_dataset_by_id(dataset, i)
            grounding_frame_ids = [0]
            grounding_frames, grounding_anno, meta_obj_train = dataset.get_frames(seq_id, grounding_frame_ids, seq_info_dict)

            language = meta_obj_train.get('language', None)
            if language is None or language == '':
                language = 'object, thing or stuff'
            ori_language=language
            language, mask = self.extract_token_from_nlp(language, 40)
            if language is None:
                data['valid'] = False
            data = TensorDict({ 'grounding_images': grounding_frames,
                                'grounding_anno': grounding_anno['bbox'],
                                'text': language,
                                'text_mask': mask,
                                'flag': torch.tensor([[1]]),
                                'ori_language':ori_language
                                })
            data = self.processing.grounding_process(data)
            valid = data['valid']
        return data

    def get_center_box(self, H, W, ratio=1 / 8):
        cx, cy, w, h = W / 2, H / 2, W * ratio, H * ratio
        return torch.tensor([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        
        enough_visible_frames = False
        while not enough_visible_frames:
            
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_seq_from_dataset_by_id(self, dataset, seq_id):
        seq_id = random.randint(0, dataset.get_num_sequences() - 1)
        seq_info_dict = dataset.get_sequence_info(seq_id)
        visible = seq_info_dict['visible']
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  
            
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  
            
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids
    
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
        
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        
        
        input_mask = [1] * len(input_ids)

        
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return [torch.tensor(input_ids)], [torch.tensor(input_mask)]


_tokenizer = SimpleTokenizer()

def tokenize(texts, context_length: int = 64, truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    mask   = torch.ones(len(all_tokens), context_length+1, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            return None, None
        result[i, :len(tokens)] = torch.tensor(tokens)
        mask[i, :len(tokens)+1] = 0

    return result, mask.bool()
