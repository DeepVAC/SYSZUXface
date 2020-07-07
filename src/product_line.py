import os
import time
from conf import config as deepvac_config
from lib.syszux_log import LOG
# from mydetect import MyDetect
# from myface import MyFace


class ProductLine(object):
    def __init__(self,worker_name_list,deepvac_config):
        self.workers = []
        for work_name in worker_name_list:
            worker = eval('{}'.format(work_name))(deepvac_config)
            self.workers.append(worker)
        assert len(self.workers) >0, 'Worker construct failed...'
        deepvac_config = deepvac_config

    def process(self, input):
        output = None
        input = input
        
        for w in self.workers:
            w.setInput(input)
            w.process()
            output = w.getOutput()
            input = output

        #print(output)
        return output

def processMaterials(product_line, your_iterator):
    for file,gt in your_iterator:
        result = product_line.process(file)
        #implement the score logic here.

def getFileList(target_dir):
    base = '../dataset/ds'
    for subdir, dirs, files in os.walk('{}/{}'.format(base, target_dir)):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            ground_truth = subdir.split(os.sep)[-1]
            yield filepath, '{}_{}'.format(target_dir,ground_truth)

if __name__ == "__main__":
    product_line = ProductLine(['MyDetect','MyFace'],deepvac_config)
    LOG.logI('All workers engaged...')
    target_dirs = ['famous','soccer']
    # try:
    for target_dir in target_dirs:
        processMaterials(product_line,your_iterator)
    # except Exception as e:
    #     LOG.logE("ERROR during processMaterials: {}".format(str(e)))