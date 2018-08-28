# helper abstract class
class Technique(object):
    def __init__(self, data, id, sess = None):
        self.tec_data = data
        self.tec_sess = sess
        self.id = id

    def execute(self):
        return 0