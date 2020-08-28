import os
import sys
import cv2
import numpy as np
from deepvac.syszux_log import LOG
from deepvac.syszux_report import FaceReport
from deepvac.syszux_executor import DeepvacChain
from deepvac.syszux_loader import OsWalkerLoader
from dump_image import DumpImage
from retina_detect import RetinaDetect

class DetectAndDumpImg(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        self.walker_loader = OsWalkerLoader(deepvac_config)
        DeepvacChain.RetinaDetect = RetinaDetect
        DeepvacChain.DumpImage = DumpImage
        self.chain = DeepvacChain("RetinaDetect => DumpImage", deepvac_config)
        self.setMaterialsInput(self.conf.dumpimg.input_path)
        
    def setMaterialsInput(self, input_path):
        self.input_path = input_path
        self.file_loader = self.walker_loader(self.input_path)
        return self

    def process(self,f):
        LOG.logI("process {}".format(f))
        self.conf.dumpimg.file_var = f
        self.chain(self.conf.dumpimg.file_var)

    def __call__(self):
        for f in self.file_loader:
            self.process(f)

if __name__ == "__main__":
    from config import config as deepvac_config
    detect_dump = DetectAndDumpImg(deepvac_config)
    detect_dump()
