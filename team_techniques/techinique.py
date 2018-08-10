# helper abstract class
class Technique(object):
    def __init__(self, data, sess = None):
        self.tec_data = data
        self.tec_sess = sess

    def execute(self):
        return 0