import sys
sys.path.append("/your deepvac_path/")
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
        super(ISFace,self).__init__(deepvac_config.face)
        self.db = None
        self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.tensor_list = []
        self.torch_db_emb = torch.Tensor().to(self.device)
        self.idx_name_map = {}
        LOG.log(LOG.S.I, "start load DEEPVAC-FACE DB...")
#        self.loadDB()
        LOG.log(LOG.S.I, "Load DEEPVAC-FACE DB finished...")

    def initNetWithCode(self):
        #to initial self.net
        conf = self.getConf()
        self.net = Backbone(conf.net_depth, conf.drop_ratio, conf.embedding_size, conf.net_mode)

    def initModelPath(self):
        self.model_path = self.getConf().model_path

    def loadDB(self):
        face_db = np.load(self.getConf().np_path)
        self.torch_db_emb = torch.load(self.getConf().db_path).to('cuda')
        self.db_names = face_db['names']
        
    def makeDB(self, root):
        features = []
        names = []
        paths = []
        img_path_lists = self.getImgPathLists_(root)
        LOG.log(LOG.S.I, "start make db, image count is {}".format(len(img_path_lists)) )
        for idx, img_path in enumerate(img_path_lists):
            # time.sleep(1)
            emb = self.inference_(cv2.imread(os.path.join(root, img_path)))
            self.torch_db_emb = torch.cat((self.torch_db_emb, emb))
            print("feature shape: ", self.torch_db_emb.shape)
            #features.append(emb)
            names.append(img_path.strip().split('/')[-2])
            paths.append(img_path)
            if idx % 10000 == 0 and idx != 0:
                LOG.log(LOG.S.I, "gen db features: {}".format(idx))
        db_path = self.getConf().db_path
        db_dir = os.path.split(db_path)[0]
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        torch.save(self.torch_db_emb, self.getConf().db_path)
        np.savez(self.getConf().np_path, names=names, paths=paths)
        LOG.log(LOG.S.I, "gen db successful, save in {}".format(self.getConf().db_path))
    
    def getImgPathLists_(self, root):
        img_path_lists = []
        folder_dir_lists = os.listdir(root)
        folder_dir_lists.sort()
        for folder_dir in folder_dir_lists:
            sub_img_path_lists = os.listdir(os.path.join(root, folder_dir))
            sub_img_path_lists.sort()
            for img_path in sub_img_path_lists:
                img_path_lists.append(os.path.join(folder_dir, img_path))
        return img_path_lists

    def inference_(self, ori):
        with torch.no_grad():
            img = self.transformer(Image.fromarray(ori))
            img = torch.unsqueeze(img,0).to(self.device)
            emb = self.net(img)
            self.exportTorchViaTrace(img)
        return emb

    def process(self):
        #single image
        if not isinstance(self.input_output['input'], list):
            LOG.log(LOG.S.E, "illegal input of ISFace: {}".format(type(self.input_output['input'])))
            return None
        #currently only support single image
        ex_time = 0
        compare_total = 0
        for ori in self.input_output['input']:
            if type(ori).__module__ != np.__name__:
                LOG.log(LOG.S.E, "illegal input of ISFace: {}".format(type(ori)))
                return None
            
            if ori.shape != (112, 112, 3):
                LOG.log(LOG.S.E, "illegal input shape: {}".format(ori.shape))
                return None

            start_time = time.time()
            emb = self.inference_(ori).to('cuda')
            elapsed_time = time.time() - start_time
            ex_time += elapsed_time
            LOG.log(LOG.S.I, "elapsed time of extract feature of ori {}".format(elapsed_time) )
            
            compare_start = time.time()
            min_distance, min_index = self.compare_(emb)
            compare_time = time.time() - compare_start
            compare_total += compare_time
            
            if min_distance > self.getConf().threshold:
                LOG.log(LOG.S.I, "Detected a stranger...")
                continue
            
            name = self.db_names[min_index].split("_")[-1]
            LOG.log(LOG.S.I, "Detected {}".format(name))
            self.addOutput(name)

        LOG.log(LOG.S.I, "Ext time {}".format(ex_time))
        LOG.log(LOG.S.I, "Com time {}".format(compare_total))
        return ex_time, compare_total

    def compare_(self, emb):
        print("gemfield compare: ", emb.type(),emb.shape,self.torch_db_emb.type(),self.torch_db_emb.shape )
        distance = torch.norm(self.torch_db_emb - emb, dim=1)
        print("distance: ",distance.type(), distance.shape)
        min_index = torch.argmin(distance).item()
        print("min_index: ",min_index)
        return distance[min_index], min_index

class DeepvacTopKTest(ISFace):
    def __init__(self, deepvac_config):
        super(DeepvacTopKTest, self).__init__(deepvac_config)
        self.k = self.getConf().top_k

    def process(self, label):
        if not isinstance(self.input_output['input'], list):
            LOG.log(LOG.S.E, "illegal input of ISFace: {}".format(type(self.input_output['input'])))
            return None
        ex_time = 0
        comp_time = 0
        for ori in self.input_output['input']:
            if type(ori).__module__ != np.__name__:
                LOG.log(LOG.S.E, "illegal input of ISFace: {}".format(type(ori)))
                return None
            if ori.shape != (112, 112, 3):
                LOG.log(LOG.S.E, "illegal input shape: {}".format(ori.shape))
                return None
            start_time = time.time()
            emb = self.inference_(ori).to('cuda')
            elapsed_time = time.time() - start_time
            ex_time += elapsed_time
            LOG.log(LOG.S.I, "elapsed time of extract feature of ori {}".format(elapsed_time) )

            compare_start = time.time()
            topk_tups = self.compare_(emb)
            comp_time += time.time() - compare_start

            if topk_tups[0][1] > self.getConf().threshold:
                LOG.log(LOG.S.I, "Detected a stranger...")
                continue

            name = self.db_names[topk_tups[0][0]]
            for tup in topk_tups[1:]:
                if self.db_names[tup[0]] == label:
                    name = self.db_names[tup[0]]
            
            LOG.log(LOG.S.I, "Detected {}".format(name))
            self.addOutput(name)

        LOG.log(LOG.S.I, "Ext time {}".format(ex_time))
        LOG.log(LOG.S.I, "Com time {}".format(comp_time))
        
        return ex_time, comp_time

    def compare_(self, emb):
        topk_tups = []
        distance = torch.norm(self.torch_db_emb - emb, dim=1)
        for i in range(self.k):
            val, index = distance.kthvalue(i+1)
            topk_tups.append((index.item(), val.item()))
        return topk_tups

    def getImgPathsAndLists_(self, dataset_path):
        img_lists = []
        img_paths = []
        persons = os.listdir(dataset_path)
        for person in persons:
            pics = os.listdir(os.path.join(dataset_path, person))
            for pic in pics:
                img_paths.append(os.path.join(dataset_path, person, pic))
                img_lists.append(cv2.imread(os.path.join(dataset_path, person, pic)))

        return img_lists, img_paths


    def __call__(self):
        database = self.getConf().database
        dataset_path = self.getConf().dataset_path
        img_lists, img_paths = self.getImgPathsAndLists_(dataset_path)
        
        total_extime = 0
        total_comtime = 0
        report = FaceReport(database, len(img_lists))
        for i in range(len(img_lists)):
            label = img_paths[i].split('/')[-2]
            self.setInput(img_lists[i])
            ex_time, com_time = self.process(label)
            total_extime += ex_time
            total_comtime += com_time
            output = self.getOutput()
            if len(output) == 0:
                pred = None                                                                            
            else:           
                pred = output[0] 
            
            report.add(label, pred)
            
        print('total_extime:', total_extime)
        print('total_comtime:', total_comtime)
        report()


if __name__ == "__main__":
    from conf import config as deepvac_config
    deepvac_config.face.model_path = '/gemfield/hostpv/gemfield/deepvac-service/src/model/branch3_best.pth'
    deepvac_config.face.db_path = '/gemfield/hostpv/gemfield/deepvac-service/src/db/deppvac_model_0825_9731.feature'
    deepvac_config.face.np_path = '/gemfield/hostpv/gemfield/deepvac-service/src/db/deppvac_model_0825_9731.feature.npz'
    deepvac_config.face.net_mode = 'ir'
    deepvac_config.face.net_depth = 50
    deepvac_config.face.database = "ipc"
    deepvac_config.face.top_k = 5
    deepvac_config.face.dataset_path = "/gemfield/hostpv/gemfield/val_dataset/" + deepvac_config.face.database + "/ds"

    topK = DeepvacTopKTest(deepvac_config)
    topK.loadDB()
    topK()
