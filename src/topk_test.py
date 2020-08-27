import sys
sys.path.append("/home/wangyuhang/wangyuhang/deepvac")
import cv2
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import time
from deepvac.syszux_deepvac import Deepvac
from deepvac.syszux_log import LOG
from deepvac.syszux_report import FaceReport
from lib.isface.model import Backbone

class ISFace(Deepvac):
    def __init__(self, deepvac_config):
        super(ISFace,self).__init__(deepvac_config)
        self.db = None
        self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.tensor_list = []
        self.torch_db_emb = torch.Tensor().to(self.device)
        self.idx_name_map = {}
        self.k = self.conf.top_k
        #LOG.logI("start load DEEPVAC-FACE DB...")
        #self.loadDB()
        #LOG.logI("Load DEEPVAC-FACE DB finished...")

    def initNetWithCode(self):
        #to initial self.net
        conf = self.conf
        self.net = Backbone(conf.net_depth, conf.drop_ratio, conf.embedding_size, conf.net_mode)

    def initModelPath(self):
        self.model_path = self.conf.model_path

    def loadDB(self):
        if not self.conf.db_path:
            LOG.logI('No db configured.')
            return
        if not self.conf.np_path:
            LOG.logI('No np configured.')
            return
        super(ISFace,self).loadDB(self.conf.db_path)
        face_db = np.load(self.conf.np_path)
        self.db_names = face_db['names']

    def addEmb2DB(self, emb):
        if not self.conf.db_path:
            LOG.logW('No db path configured.')
        super(ISFace,self).addEmb2DB(emb)
    
    def saveDB(self, names, paths):
        super(ISFace,self).saveDB(self.conf.db_path)
        np.savez(self.conf.np_path, names=names, paths=paths)


    def makeDB(self, root):
        features = []
        names = []
        paths = []
        img_path_lists = self.getImgPathLists_(root)
        LOG.logI("start make db, image count is {}".format(len(img_path_lists)) )
        for idx, img_path in enumerate(img_path_lists):
            emb = self.inference_(cv2.imread(os.path.join(root, img_path)))
            
            self.addEmb2DB(emb)
            names.append(img_path.strip().split('/')[-2])
            paths.append(img_path)
            
            if idx % 10000 == 0 and idx != 0:
                LOG.logI("gen db features: {}".format(idx))
        db_path = self.conf.db_path
        db_dir = os.path.split(db_path)[0]
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        self.saveDB(names, paths)
        LOG.logI("gen db successful, save in {}".format(self.conf.db_path))
    
    def getImgPathLists_(self, root):
        img_path_lists = []
        for root, dirs, files in os.walk(root):
            for name in files:
                img_path_lists.append(os.path.join(root, name))

        return img_path_lists

    def inference_(self, ori):
        with torch.no_grad():
            img = self.transformer(Image.fromarray(ori))
            img = torch.unsqueeze(img,0).to(self.device)
            emb = self.net(img)
            self.exportTorchViaTrace(img)
        return emb

    def addEmb2DB(self, emb):
        if not self.conf.db_path:
            LOG.logW('No db path configured.')
        super(ISFace,self).addEmb2DB(emb)

    def getPredName(self, D, I, label):
        name = self.db_names[I[D.index(min(D))]]

        for i in I[1:]:
            if self.db_names[i] == label:
                name = label
        
        return name

    def process(self, label):
        if not isinstance(self.input_output['input'], list):
            LOG.logE("illegal input of ISFace: {}".format(type(self.input_output['input'])))
            return None
        
        for ori in self.input_output['input']:
            if type(ori).__module__ != np.__name__:
                LOG.logE("illegal input of ISFace: {}".format(type(ori)))
                return None
            if ori.shape != (112, 112, 3):
                LOG.logE("illegal input shape: {}".format(ori.shape))
                return None
            
            emb = self.inference_(ori).to('cuda')

            D, I = self.search(emb, self.k)

            if min(D) > self.conf.threshold:
                LOG.logI("Detected a stranger...")
                continue
            
            name = self.getPredName(D, I, label)
            
            LOG.logI("Detected {}".format(name))
            self.addOutput(name)

    def __call__(self):
        database = self.conf.database
        dataset_path = self.conf.dataset_path
        img_path_lists = self.getImgPathLists_(dataset_path)
        
        report = FaceReport(database, len(img_path_lists))
        for i in range(len(img_path_lists)):
            label = img_path_lists[i].split('/')[-2]
            img = cv2.imread(img_path_lists[i])
            self.setInput(img)
            self.process(label)

            output = self.getOutput()
            if len(output) == 0:
                pred = None                                                                            
            else:           
                pred = output[0] 
            
            report.add(label, pred)
            
        report()


if __name__ == "__main__":
    from config import config as deepvac_config

    topK = ISFace(deepvac_config.face)
    #topK.makeDB(deepvac_config.face.db_img_path)
    topK.loadDB()
    topK()
