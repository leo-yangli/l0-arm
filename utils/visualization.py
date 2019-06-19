from time import localtime
import time
import os
import shutil
from tensorboardX import SummaryWriter


class Visualizer(object):

    def __init__(self, dir, name, current_time, title_note='', **kwargs):
        directory = '{}/{}_{}'.format(dir, current_time, name + ' ' + title_note)
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
        self.vis = SummaryWriter(directory)

        self.index = {}
        self.log_text = ''

    def plot(self, tag, value, global_step=None):
        if global_step is None:
            x = self.index.get(tag, 0)
            self.index[tag] = x + 1
        else:
            x = global_step
        self.vis.add_scalar(tag=tag, scalar_value=value, global_step=x)

    def log(self, tag, info, global_step=None):
        if global_step is None:
            x = self.index.get(tag, 0)
            self.index[tag] = x + 1
        else:
            x = global_step
        log_text = ('[{time}] {info}'.format(time=time.strftime('%Y-%m-%d %H:%M:%S', localtime()), info=info))
        self.vis.add_text(tag, log_text, x)

    def hist(self, tag, array, global_step=None):
        if global_step is None:
            x = self.index.get(tag, 0)
            self.index[tag] = x + 1
        else:
            x = global_step
        self.vis.add_histogram(tag, array, x)

    def __getattr__(self, name):
        return getattr(self.vis, name)
