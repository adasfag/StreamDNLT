import os
prj_dir ='/home/share/hhd/dataset/lgt/uvltrack_work/'
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = prj_dir    # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(prj_dir, 'tensorboard')    # Directory for tensorboard files.
        self.pretrained_networks = os.path.join(prj_dir, 'pretrained_networks')
        self.lasot_dir = os.path.join(prj_dir, 'data/lasot')
        self.lasotext_dir = os.path.join(prj_dir, 'data/lasotext')
        self.got10k_dir = os.path.join(prj_dir, 'data/got10k')#just for debug
        self.got10k_dir_train = os.path.join(self.got10k_dir, 'train')
        self.trackingnet_dir = os.path.join(prj_dir, 'data/trackingnet')
        self.coco_dir = os.path.join(prj_dir, 'data/coco')
        self.tnl2k_dir = os.path.join(prj_dir, 'data/tnl2k/test')
        self.tnl2k_test_dir = os.path.join(prj_dir, 'data/tnl2k/train')
        self.otb99_dir = os.path.join(prj_dir, 'data/otb99')
        self.refcoco_dir = os.path.join(prj_dir, 'data/refcocog')