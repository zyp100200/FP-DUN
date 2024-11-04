import os

class Setting:

    def __init__(self, ratio):

        self.ratio = ratio

        self.lr = 8e-5
        self.epoch = 100
        self.step = [101]
        self.batch = 28

        self.train_dataset_name = '/home/zhouyu/xiewei/compressed sensing/train2.mat'
        # self.train_dataset_name = './data/train2.mat'

        self.val_dataset_name = '/home/zhouyu/xiewei/compressed sensing/val/'
        # self.val_dataset_name = './data/val/'

        self.save_dir = './results'
        # self.work_name = '/'.join(os.getcwd().split('/')[4:])
        self.result_dir = self.save_dir + '/{}_{}'.format(self.ratio,self.lr)
        self.model_dir = self.result_dir + '/model'
        self.pic_dir = self.result_dir + '/pic'
        self.analysis = self.result_dir + '/analysis'
        self.log_file = self.result_dir + 'log_{}.txt'.format(self.ratio)

        self.mkdirs()

    def mkdirs(self):

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.pic_dir):
            os.makedirs(self.pic_dir)
        if not os.path.exists(self.analysis):
            os.makedirs(self.analysis)