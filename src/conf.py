import os
from lib.libconfig import *
################## face begin #################
config.face.model_path = './model/your_model.pth'
config.face.input_size = [112,112]
config.face.embedding_size = 512
config.face.threshold = 1.2
config.face.db_path = "your_feature_db_file"

################## detect begin ##############
config.detect.model_path = 'your_detect_model.pth'
config.detect.confidence_threshold = 0.5
config.detect.top_k = 100
config.detect.nms_threshold = 0.4
config.detect.keep_top_k = 50
