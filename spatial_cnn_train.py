import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
# import torchvision.transforms as transforms
import opencv_transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import dataloader
from utils import *
from network import NLN_CNN
from dataloader.split_train_test_video import UCF101_splitter

parser = argparse.ArgumentParser(description='UCF101 Non-Local CNN')
parser.add_argument('--cwd', default=os.getcwd(), type=str, metavar='CWD', help='curent working directory')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=2, type=int, metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--demo', dest='demo', action='store_true', help='initialize inference on video source')
parser.add_argument('--resume', default='/hdd/NLN/record/checkpoint.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')


def main():
    global arg
    arg = parser.parse_args()
    print arg

    # Skip loading data if inference mode is on
    if arg.demo is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Prepare DataLoader
        data_loader = dataloader.spatial_dataloader(
                            BATCH_SIZE=arg.batch_size,
                            num_workers=8,
                            path='/hdd/UCF-101/Data/jpegs_256/',
                            ucf_list='/hdd/NLN/UCF_list/',
                            ucf_split='03',
                            )
        train_loader, test_loader, test_video = data_loader.run()

        # Model
        model = NLN_Trainer(
                            nb_epochs=arg.epochs,
                            lr=arg.lr,
                            batch_size=arg.batch_size,
                            resume=arg.resume,
                            start_epoch=arg.start_epoch,
                            evaluate=arg.evaluate,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            test_video=test_video
        )
        # Training
        model.run()

    else:
        # Start predicting action on video source (default: webcam)
        vs = cv2.VideoCapture(0)
        model = NLN_Demo(vs=vs, resume=arg.resume)
        model.inference(vs)


class NLN_Demo():

    def __init__(self, vs, resume):
        # Create a dictionary of the classes and their corresponding indices
        self.data_handler = UCF101_splitter(arg.cwd+'/UCF_list/', None)
        self.data_handler.get_action_index()
        self.class_to_idx = self.data_handler.action_label
        self.idx_to_class = {v: k for k, v in self.class_to_idx.iteritems()}

        # load the same transformation mechanism on images like in training
        self.transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # Load the saved model
        self.model = NLN_CNN().cpu()
        self.model.eval()
        self.checkpoint = torch.load(resume, map_location='cpu')
        self.model.load_state_dict(self.checkpoint['state_dict'])

    def inference(self, vs):
        # Start looping on frames received from video source
        while True:
            # read each frame and prepare it for feedforward in nn (resize and type)
            _, orig_frame = vs.read()
            frame = self.transform(orig_frame).view(1, 3, 224, 224)

            # frame = cv2.resize(orig_frame, (224, 224))
            # frame = np.rollaxis(frame, 2, 0).reshape((1, 3, 224, 224))
            # frame = torch.from_numpy(frame).type('torch.FloatTensor')

            # feed the frame to the neural network
            nn_output = self.model(frame)
            softmax = torch.nn.Softmax()
            nn_output = softmax(nn_output)
            nn_output = nn_output.data.cpu().numpy()

            # extract the highest ranked prediction
            preds = nn_output.argsort()[0][-5:][::-1]
            pred_classes = [(self.idx_to_class[str(pred)], nn_output[0, pred]) for pred in preds]

            # Display the resulting frame and the classified action
            font = cv2.FONT_HERSHEY_SIMPLEX
            y0, dy = 300, 40
            for i in xrange(5):
                y = y0 + i * dy
                cv2.putText(orig_frame, '{} - {:.2f}'.format(pred_classes[i][0], pred_classes[i][1]),
                            (5, y), font, 1, (0, 0, 255), 2)
            cv2.imshow('frame', orig_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        vs.release()
        cv2.destroyAllWindows()


class NLN_Trainer():

    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video):
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.start_epoch = start_epoch
        self.evaluate = evaluate
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.test_video = test_video

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')

        # build model
        self.model = NLN_CNN().cuda()

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))

        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            print prec1, val_loss
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1

            #lr_scheduler
            self.scheduler.step(val_loss)
            
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('/hdd/NLN/record/spatial_video_preds.pickle', 'wb') as f:
                    pickle.dump(self.dic_video_level_preds, f)
                f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()},
                is_best,
                '/hdd/NLN/record/checkpoint.pth.tar',
                '/hdd/NLN/record/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        #switch to train mode
        self.model.train()    
        end = time.time()

        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict, label) in enumerate(progress):

            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # compute output
            output = Variable(torch.zeros(len(data_dict['img1']), 101).float()).cuda()
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, '/hdd/NLN/record/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys, data, label) in enumerate(progress):
            
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j].split('/',1)[0]
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]

        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
            

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg, 3)],
                'Loss':[round(video_loss, 5)],
                'Prec@1':[round(video_top1, 3)],
                'Prec@5':[round(video_top5, 3)]}
        record_info(info, '/hdd/NLN/record/rgb_test.csv', 'test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):
        
            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1         
            if np.argmax(preds) == label:
                correct+=1

        # top1-top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
            
        top1, top5 = accuracy(video_level_preds, video_level_labels, topk=(1, 5))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())     
                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1, top5, loss.data.cpu().numpy()


if __name__=='__main__':
    main()