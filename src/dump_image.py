import os
import cv2
import numpy as np
from deepvac.syszux_log import LOG
from deepvac.syszux_deepvac import Deepvac
from config import config as deepvac_config

class DumpImage(Deepvac):
    def __init__(self,deepvac_config):
        super(DumpImage,self).__init__(deepvac_config.dumpimg)
        self.output_path = self.conf.output_path

    def initNet(self):
        pass

    def process(self):
        #single image
        if not isinstance(self.input_output['input'], list):
            LOG.logE( "illegal input: {}".format(type(self.input_output['input'])))
            return None
        
        ori_num = len(self.input_output['input'])
        if ori_num != 1:
            LOG.logE("{} faces detected in {}".format(ori_num, self.conf.file_var))
            return
            
        for ori in self.input_output['input']:
            if type(ori).__module__ != np.__name__:
                LOG.logE("illegal input of ISFace: {}".format(type(ori)))
                return None
            
            if ori.shape != (112, 112, 3):
                LOG.logE("illegal input shape: {}".format(ori.shape))
                return None

            fdir,fname = self.conf.file_var.split(os.sep)[-2:]

            #make sure output dir exist
            output_dir = os.path.join(self.conf.output_path, fdir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            #replace name
            fname = fname.replace(",", "__")
            output_file_name = os.path.join(output_dir, fname)
            
            cv2.imwrite(output_file_name, ori)
