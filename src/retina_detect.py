import os
import sys
import argparse
import torch
import numpy as np
import cv2
import time
from lib.layers.functions.prior_box import PriorBox
from lib.nms.py_cpu_nms import py_cpu_nms
from lib.box_utils import decode, decode_landm
from lib.retina.models.retinaface import RetinaFace
from deepvac.syszux_align import AlignFace
from deepvac.syszux_deepvac import Deepvac
from deepvac.syszux_log import LOG
#Gemfield begin
torch.set_grad_enabled(False)

class RetinaDetect(Deepvac):
    def __init__(self, deepvac_config):
        super(RetinaDetect,self).__init__(deepvac_config.detect)
        self.alignFace = AlignFace()
    
    def initNetWithCode(self):
        #to initial self.net
        self.net = RetinaFace(cfg=self.conf.net_structure[self.conf.network], phase = 'test')

    def initModelPath(self):
        self.model_path = self.conf.model_path

    def process(self):
        imglist_to_detect = []
        #single image
        if not isinstance(self.input_output['input'], list):
            LOG.logE( "illegal input of RetinaDetect...")
            return imglist_to_detect
        #currently only support single image
        input = self.input_output['input'][0]
        if not isinstance(input, str):
            LOG.logE( "illegal input of RetinaDetect...")
            return imglist_to_detect

        img_raw = cv2.imread(input)

        if img_raw is None or img_raw.shape is None:
            LOG.logE( "Read img:[{}] error!".format(input))
            return imglist_to_detect

        conf = self.conf
        #resize begin
        h,w,c = img_raw.shape
        max_edge = max(h,w)
        if(max_edge > conf.max_hw):
            img_raw = cv2.resize(img_raw,(int(w * 2000.0 / max_edge), int(h * 2000.0 / max_edge)))
        #resize end

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, forward_conf, landms = self.net(img)  # forward pass
        #LOG.logI( 'retina net detect forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(conf.cfg_re50, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, conf.cfg_re50['variance'])
        boxes = boxes * scale / conf.resize
        boxes = boxes.cpu().numpy()
        scores = forward_conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, conf.cfg_re50['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / conf.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > conf.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:conf.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, conf.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:conf.keep_top_k, :]
        landms = landms[:conf.keep_top_k, :]

        LOG.logI( "detected {} faces@{}".format(len(dets), input) )
        if len(dets) == 0:
            return imglist_to_detect
            
        if len(dets) != len(landms):
            LOG.logE( "dets len mismatched landms len: {} vs {}".format(len(dets),len(landms)))
            return imglist_to_detect

        for i in range(len(dets)):
            bbox = dets[i]
            landmark = landms[i]
            align_img = self.alignFace(img_raw, landmark)
            self.addOutput(align_img)

