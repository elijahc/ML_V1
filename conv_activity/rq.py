from builtins import object
import redis


class Queue(object):
    def __init__(self):
        self.r = redis.StrictRedis(host='localhost', port=6379, db=0)
        local_id = self.r.incr('queue_space')
        id_name = "queue:%s" % (local_id)
        self.id_name = id_name

    def push(self, element):
        id_name = self.id_name
        push_element = self.r.lpush(id_name, element)

    def pop(self):
        id_name = self.id_name
        popped_element = self.r.rpop(id_name)
        return popped_element

