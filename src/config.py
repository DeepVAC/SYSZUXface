import os
from deepvac.syszux_config import *
################## face begin #################
config.face = AttrDict()
config.face.device = 'cuda'
config.face.disable_git = True

config.face.input_size = [112,112]
config.face.embedding_size = 512
config.face.drop_ratio = 0.4

config.face.db_img_path = '/ your db img path / '
config.face.model_path = '/ your face model path /'
config.face.db_path = '/ your face feature path /'
config.face.np_path = '/ your face np path /'

config.face.net_mode = 'ir'
config.face.net_depth = 50

config.face.threshold = 1.3

#### top_k test####
config.face.database = 'your database name'
config.face.top_k = 5
config.face.dataset_path = '/ your test dataset path /'


################## detect begin ##############
config.detect = AttrDict()
config.detect.device = 'cuda'
config.detect.disable_git = True

config.detect.confidence_threshold = 0.4
config.detect.top_k = 1
config.detect.nms_threshold = 0.4
config.detect.max_hw = 2000
config.detect.keep_top_k = 50
config.detect.resize = 1

config.detect.network = 'Resnet50' # 'mobilenet0.25' or 'Resnet50'
config.detect.model_path = '/ your detect model path /'

config.detect.cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

config.detect.cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

config.detect.net_structure = {
    'mobilenet0.25': config.detect.cfg_mnet,
    'Resnet50': config.detect.cfg_re50
}

################## dump begin ##############
config.dumpimg = AttrDict()
config.dumpimg.disable_git = True
config.dumpimg.input_path = '/ your input detect dataset path /'
config.dumpimg.output_path = '/ your output detect dataset path /'
config.dumpimg.file_var = None
