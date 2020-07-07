import cv2
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import time
from lib.deepvac import DeepVAC
from lib.syszux_log import LOG

class MyFace(DeepVAC):
    def __init__(self, deepvac_config):
        super(MyFace,self).__init__(deepvac_config.face)

    def initNetWithCode(self):
        #to initial self.net
        conf = self.getConf()
        self.net = YourModel()

    def initModelPath(self):
        self.model_path = self.getConf().model_path

    def loadDB(self):
        #load your DB

    def process(self):
        #single image
        if not isinstance(self.input_output['input'], list):
            LOG.log(LOG.S.E, "illegal input of MyFace: {}".format(type(self.input_output['input'])))
            return None

if __name__ == "__main__":
    from conf import config as deepvac_config
    face = MyFace(deepvac_config)
    face.loadDB()
    # detect = MyDetect(deepvac_config)
    LOG.logI,('MyFace Model load finished...')
        
    # detect.setInput('dataset/ds/soccer/c_luo/10.jpg')
    # detect.process()
    # ori_list = detect.getOutput()
    face.setInput(ori_list)
    face.process()
