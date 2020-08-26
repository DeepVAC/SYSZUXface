import os
from deepvac.syszux_config import *
################# global begin ##############
config.kafka_server = '192.168.0.113'
config.record_topic = 'deepvac'
config.report_topic = 'deepvac_report'
config.file_base = '/opt/data'
################## face begin #################
#config.face.model_path = './model/model_2020-06-25-02-42_accuracy:0.972857142857143_step:218340_None.pth'
#config.face.model_path = './model/deepvac_50.pth'
config.face = AttrDict()
config.face.device = 'cuda'
config.face.disable_git = True
config.face.model_path = './model/deepvac100_v1.pth'
config.face.input_size = [112,112]
config.face.embedding_size = 512
config.face.use_mobilfacenet = False
config.face.net_depth = 100
config.face.drop_ratio = 0.4
config.face.net_mode = 'ir_se' # or 'ir'
config.face.data_mode = 'emore'
config.face.batch_size = 100 # irse net depth 50 
config.face.threshold = 1.3
config.face.yx_threshold = 0.35
config.face.face_limit = 10
#when inference, at maximum detect 10 faces in one image, my laptop is slow
config.face.min_face_size = 30
# the larger this value, the faster deduction, comes with tradeoff in small faces
config.face.db_path = "./db/db_asia2.npz"
# test topk face
config.face.database = "ipc"
config.face.dataset_path = "/gemfield/hostpv/gemfield/val_dataset_mob/ipc/ds"
config.face.top_k = 5
################## detect begin ##############
config.detect = AttrDict()
config.detect.device = 'cuda'
config.detect.disable_git = True
config.detect.network = "mobilenet0.25"
config.detect.confidence_threshold = 0.8
config.detect.top_k = 100
config.detect.nms_threshold = 0.4
config.detect.max_hw = 2000
config.detect.keep_top_k = 50
#for step1 audit
#config.detect.confidence_threshold = 0.9
#config.detect.top_k = 10
#config.detect.keep_top_k = 2
#config.detect.model_path = './model/Resnet50_Final.pth'
config.detect.model_path = './model/mobilenet0.25_Final.pth'
config.detect.resize = 1

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
################## dump begin ##############
config.dumpimg = AttrDict()
config.dumpimg.disable_git = True
config.dumpimg.output_path = "/gemfield/hostpv/gemfield/self/"
config.dumpimg.file_var = None
