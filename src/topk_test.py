import sys
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
from deepvac.syszux_loader import OsWalkerLoader
from modules.model_isface import Backbone

class ISFace(Deepvac):
    def __init__(self, deepvac_config):
        super(ISFace,self).__init__(deepvac_config)
        self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.k = self.conf.top_k
        self.walker_loader = OsWalkerLoader(deepvac_config)
        self.names = []
        self.paths = []

    def initNetWithCode(self):
        #to initial self.net
        conf = self.conf
        self.net = Backbone(conf.net_depth, conf.drop_ratio, conf.embedding_size, conf.net_mode)

    def loadDB(self):
        if not self.conf.db_path:
            LOG.logE('No db configured.', exit=True)
        if not self.conf.np_path:
            LOG.logE('No np configured.', exit=True)
        super(ISFace,self).loadDB(self.conf.db_path)
        face_db = np.load(self.conf.np_path)
        self.db_names = face_db['names']

    def addEmb2DB(self, emb, name, img_path):
        if not self.conf.db_path:
            LOG.logW('No db path configured.')
        super(ISFace,self).addEmb2DB(emb)
        self.names.append(name)
        self.paths.append(img_path)
    
    def saveDB(self, names, paths):
        super(ISFace,self).saveDB(self.conf.db_path)
        np.savez(self.conf.np_path, names=names, paths=paths)

    def clearDB(self):
        self.xb = torch.Tensor().to(self.conf.device)
        self.names = []
        self.paths = []

    def makeDB(self, root):
        if not self.conf.db_path:
            LOG.logE('No db configured.', exit=True)
        if not self.conf.np_path:
            LOG.logE('No np configured.', exit=True)
        db_path = self.conf.db_path
        db_dir = os.path.split(db_path)[0]
        if not os.path.exists(db_dir):
             os.makedirs(db_dir)

        self.clearDB()
        img_path_lists = self.walker_loader(root)
        length = sum(1 for _ in img_path_lists)
        LOG.logI("start make db, image count is {}".format(length) )
        for idx, img_path in enumerate(self.walker_loader(root)):
            with torch.no_grad():
                emb = self.inference_(cv2.imread(img_path))
            name = img_path.strip().split('/')[-2]
            self.addEmb2DB(emb, name, img_path)
            
            if idx % 10000 == 0 and idx != 0:
                LOG.logI("gen db features: {}".format(idx))

        self.saveDB(names, paths)
        LOG.logI("gen db successful, save in {}".format(self.conf.db_path))

    def inference_(self, ori):
        img = self.transformer(Image.fromarray(ori))
        img = torch.unsqueeze(img,0).to(self.device)
        emb = self.net(img)
        self.exportTorchViaTrace(img)
        
        return emb

    def getPredName(self, D, I, label):
        names = self.db_names[I]
        if label in names:
            pred = label
        else:
            pred = names[0]
        
        return pred

    def process(self):
        database = self.conf.database
        dataset_path = self.conf.dataset_path
        img_path_lists = self.walker_loader(dataset_path)
        length = sum(1 for _ in img_path_lists)
        report = FaceReport(database, length)
        for img_path_list in self.walker_loader(dataset_path):
            label = img_path_list.split('/')[-2]
            img = cv2.imread(img_path_list)
            
            emb = self.inference_(img).to('cuda')

            D, I = self.search(emb, self.k)

            if min(D) > self.conf.threshold:
                pred = None
                LOG.logI("Detected a stranger...")
            else:
                pred = self.getPredName(D, I, label)
                LOG.logI("Detected {}".format(pred))
            report.add(label, pred)
        report()

if __name__ == "__main__":
    from config import config as deepvac_config

    topK = ISFace(deepvac_config.face)
    #topK.makeDB(deepvac_config.face.db_img_path)
    topK.loadDB()
    topK.process()
