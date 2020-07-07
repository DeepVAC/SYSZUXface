import os
import argparse
import torch
import numpy as np
import cv2
import time
from lib.pymtcnn import AlignFace
from lib.deepvac import DeepVAC
from lib.syszux_log import LOG

class MyDetect(DeepVAC):
    def __init__(self, deepvac_config):
        super(MyDetect,self).__init__(deepvac_config.detect)
        self.alignFace = AlignFace()

    def initNetWithCode(self):
        pass
        #self.net = YourModel()

    def initModelPath(self):
        self.model_path = self.getConf().model_path

    def process(self):
        imglist_to_detect = []
        #single image
        if not isinstance(self.input_output['input'], list):
            LOG.log(LOG.S.E, "illegal input of MyDetect...")
            return imglist_to_detect
        #currently only support single image
        input = self.input_output['input'][0]
        if not isinstance(input, str):
            LOG.log(LOG.S.E, "illegal input of MyDetect...")
            return imglist_to_detect

        img_raw = cv2.imread(input)

        if img_raw is None or img_raw.shape is None:
            LOG.log(LOG.S.E, "Read img:[{}] error!".format(input))
            return imglist_to_detect

        #add your detect logic here

        LOG.log(LOG.S.I, "detectd {} faces...".format(len(dets)))
        if len(dets) == 0:
            return imglist_to_detect
            
        if len(dets) != len(landms):
            LOG.logE("dets len mismatched landms len: {} vs {}".format(len(dets),len(landms)))
            return imglist_to_detect

        for i in range(len(dets)):
            bbox = dets[i]
            landmark = landms[i]
            align_img = self.alignFace(img_raw, landmark)
            self.addOutput(align_img)

if __name__ == "__main__":
    from conf import config as deepvac_config
    detect = MyDetect(deepvac_config)
    LOG.logI('MyDetect Model load finished...')
    detect.setInput('dataset/ds/soccer/c_luo/10.jpg')
    detect.process()
    print(len(detect.getOutput()) )
