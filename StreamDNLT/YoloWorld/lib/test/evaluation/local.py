from lib.test.evaluation.environment import EnvSettings
import os
data_dir='/home/share/hhd/dataset/lgt/uvltrack_work/'#TODO:just for debug
prj_dir=data_dir

def local_env_settings():
    settings = EnvSettings()
    settings.data_dir = data_dir
    settings.save_dir = data_dir
    settings.result_plot_path = os.path.join(data_dir, 'test/result_plots')
    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size256_part_picture_full_finetune')#just for debug
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size256_part_picture_init_gttruth_full_finetuneaddtemplate')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size256_part_picture_init_gttruth_adapter_tune_addtemplate_epoch50')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_epoch10')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_memory_norme_epoch10')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_epoch60')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_query_6')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_query_6_nomemory')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_query_6_memory')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_memory_more_frames')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_memory_more_frames_tgt')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_memory_more_frames_tgt_addloss')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_memory_more_frames_query3')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_memory_more_frames_query3_alldataset')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracking_baseline_size320_part_picture_init_gttruth_adapter_tune_addtemplate_memory_more_frames_query3_samplevideo')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/alonelything/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swinb_allnltdataset_samplevideo_epoch10/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_otb_new_match_samplevideo_epoch10/')#otb 46.79 效果急剧下降了
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_otb_samplevideo_epoch10/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_otb_optimise_samplevideo_epoch10/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_otb_adapter_backbone_samplevideo_epoch10/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_otb_samplevideo_query6_epoch10/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_alldataset_query6_epoch10/')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_allnltdataset_samplevideo_query6_epoch10/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_allnltdataset_query6_epoch10/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_allnltdataset_query3_epoch10_samplevideo_fixrefcocoproblem/')
    
    
    
    
    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/just_debug_something/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/just_debug_something1/')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_allnltdataset_reducerefcoco_query3_epoch10_samplevideo_grounding00vl10tracking00')
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_allnltdataset_reducerefcoco_query3_epoch10_samplevideo_grounding00vl10tracking00_joint')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_allnltdataset_reducerefcoco_query3_epoch10_samplevideo_grounding00vl10tracking00_addmatch')#使用new match 会让精度下降很多了
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_allnltdataset_reducerefcoco_query3_epoch10_samplevideo_grounding00vl10tracking00_original')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_swint_allnltdataset_reducerefcoco_query3_epoch10_samplevideo_grounding00vl10tracking00_consistant')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/justtrackingsomething2')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_trackingsomething')
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking')
    
    
    

    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_query')










    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_query_adapter')

    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_adapter_class_score')

    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_adapter_v1')

    
    
    
    
    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_adapter_kan')

    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_nlbbox_epoch60')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_nlbbox_epoch160')

    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_nlbbox_lasot')

    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_nlbbox_lasot_template256')

    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_groundingandtracking_nlbbox_allnltdataset_template256_match')

    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_tnl_epoch60')

    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_lasot_epoch60')

    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_lasot_epochtest')
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_lasot_tracking')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_lasotquery3')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_lasotquery12')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_lasotquery16')
    
    
        
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_lasot_tracking')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_tnl2k_tracking')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_tnl2k_trackingv1')
     
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/dinotracker_lasottrackingv1')
    
    
    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_finetune')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_finetuneepoch1')
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_dinotracker_addspatial')
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_zeroshot')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_zeroshot_tracking')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_finetune')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_finetunetemporal')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_finetunetemporalcurrentv')
    
    
    
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_finetunetracking')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_zeroshottracking')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporal')#使用之前的时序模型，采用全图片微调
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporaltracking')
    
    
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_spatialfinetune')
        
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_spatialfinetune1')
   
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_spatialandtemporalfinetune')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_spatialandtemporalfinetuneepoch6030')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_spatialandtemporalfinetuneepoch6040')
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_spatialandtemporaltnl2kepoch10')
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_spatialandtemporaljointepoch10')
    
    
        
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporallasotepoch30')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporaltnl2kepoch30')
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporalotb99epoch30')
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporaljointepoch30')
    
    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporaljointepoch10_tracking')
    
    settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporaljointepoch30_tracking')
    


    
    #settings.results_path = os.path.join('/home/qui_wzh/20240630/home/language-guided-tracking/UVLTrack-master', 'test/baseline_yolotracker_temporaljointepoch30_finetune')
    
 
 
 
 
    
    settings.lasot_path = os.path.join(data_dir, 'data/lasot')
    settings.nfs_path = os.path.join(data_dir, 'data/nfs')
    settings.otb_path = os.path.join(data_dir, 'data/otb99/OTB_videos')
    settings.trackingnet_path = os.path.join(data_dir, 'data/trackingnet')
    settings.uav_path = os.path.join(data_dir, 'data/uav')
    settings.tnl2k_path = os.path.join(data_dir, 'data/tnl2k/test')
    settings.otb99_path = os.path.join(data_dir, 'data/otb99')
    settings.lasot_ext_path = os.path.join(data_dir, 'data/lasotext')

    return settings