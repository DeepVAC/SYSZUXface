import sys
import time
import json
from collections import deque
from confluent_kafka import Consumer, KafkaError, Producer
from syszux_log import LOG

class KF(object):
    def __init__(self):
        self.dts_queue = deque(maxlen=100)
        self.rec_k = ['id','op','value']
        self.count = 0
        self.poll_timeout = 5.0

class KFP(KF):
    def __init__(self, kafka_ip, topic='deepvac_report'):
        super(KFP,self).__init__()
        p_config = {'bootstrap.servers': kafka_ip, 'queue.buffering.max.kbytes': 2000000,'queue.buffering.max.messages': 10000000}
        self.P = Producer(p_config)
        self.topic = topic

    def produce(self, key, msg):
        self.P.produce(self.topic, key=key, value=msg)
        self.P.poll(0)
    
    def __call__(self, id, op, value):
        rec_d = dict.fromkeys(self.rec_k)
        rec_d['id'] = str(id)
        rec_d['op'] = op
        rec_d['value'] = value
        rec_b = json.dumps(rec_d).encode('utf-8')
        self.produce(rec_d['id'], rec_b)

class KFC(KF):
    def __init__(self, kafka_ip, topics, gid):
        super(KFC,self).__init__()
        self.C = Consumer({
            'bootstrap.servers': kafka_ip,
            'group.id': gid,
        })
        self.C.subscribe([topics])

    def pollMsg(self):
        return self.C.poll(self.poll_timeout)

    def getMsg(self):
        if not self.step:
            raise Exception('You should set step first: e.g. gemfieldc.step=5...')

        while True:
            # valid check
            sys.stdout.flush()
            msg = self.pollMsg()
            LOG.log(LOG.S.I, "CONSUME NONE.")
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                LOG.log(LOG.S.E, 'KAFKA Poll error: {}'.format(msg.error()))
                continue
            yield msg

    def consume(self):
        for msg in self.getMsg():
            yield msg

class KFCImage(KFC):
    def __init__(self, kafka_ip, topics='deepvac', gid='face'):
        super(KFCImage, self).__init__(kafka_ip, topics, gid)
        self.step = 1

    def __call__(self):
        for msg in self.getMsg():
            self.count += 1
            id = msg.key()
            rec_b = msg.value()
            rec_d = json.loads(rec_b.decode('utf-8'))
            if id.decode('utf-8') != rec_d['id']:
                LOG.log(LOG.S.E, 'id mismatch {} vs {}...'.format(id,rec_d['id']))
                continue
                
            if rec_d['op'] != 'img':
                LOG.log(LOG.S.E, 'illegal op: {}'.format(rec_d['op']))
                continue
            yield rec_d

if __name__ == "__main__":
    kfp = KFP('192.168.0.113','deepvac')
    kfp(123, 'img', sys.argv[1])
    import time
    time.sleep(3)
    sys.exit(0)
    kfci = KFCImage('192.168.0.113')
    for c in kfci():
        print(c['id'])
        print(c['op'])
        print(c['value'])


    
