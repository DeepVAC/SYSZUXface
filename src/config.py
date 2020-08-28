import os
from deepvac.syszux_config import *
################## face begin #################
config.face = AttrDict()
config.face.device = 'cuda'
config.face.disable_git = True

config.face.input_size = [112,112]
config.face.embedding_size = 512
config.face.drop_ratio = 0.4

config.face.db_img_path = '/gemfield/hostpv/gemfield/val_dataset/total_base/db'
config.face.model_path = '/gemfield/hostpv/gemfield/deepvac-service/src/model/branch3_best.pth'
config.face.db_path = '/gemfield/hostpv/gemfield/deepvac-service/src/db/deppvac_b3_best.feature'
config.face.np_path = '/gemfield/hostpv/gemfield/deepvac-service/src/db/deppvac_b3_best.feature.npz'

config.face.net_mode = 'ir'
config.face.net_depth = 50

config.face.threshold = 1.3

#### top_k test####
config.face.database = "ipc"
config.face.top_k = 5
config.face.dataset_path = "/gemfield/hostpv/gemfield/val_dataset/" + config.face.database + "/ds"
