import os, pickle


class UCF101_splitter():

    def __init__(self, path, split):
        self.path = path
        self.split = split

    def get_action_index(self):
        self.action_label={}
        with open(self.path+'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        for line in content:
            label, action = line.split(' ')
            if action not in self.action_label.keys():
                self.action_label[action]=label

    def split_video(self):
        self.get_action_index()
        for path,subdir,files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    self.train_video = self.file2_dic(self.path+filename)
                if filename.split('.')[0] == 'testlist'+self.split:
                    self.test_video = self.file2_dic(self.path+filename)
        print '==> (Training video, Validation video):(', len(self.train_video),len(self.test_video),')'

        return self.train_video, self.test_video

    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\n') for x in content]
        dic={}
        for line in content:
            video = line.split('/',1)[1].split(' ',1)[0]
            key = video.split('_',1)[1].split('.',1)[0]
            label = self.action_label[line.split('/')[0]]

            dic[key] = int(label)
        return dic



if __name__ == '__main__':
    path = '/hdd/NLN/UCF_list/'
    split = '05'
    splitter = UCF101_splitter(path=path,split=split)
    train_video,test_video = splitter.split_video()
    print len(train_video),len(test_video)