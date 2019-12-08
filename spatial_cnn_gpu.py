import argparse
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dataloader
from utils import *
from network import resnet101
# from dataloader import UCF101_splitter
from dataloader import my_splitter
from opt_flow import opt_flow_infer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=3, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--batch-size-my', default=5, type=int, metavar='N', help='mini-batch size for new dataset')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--demo', dest='demo', action='store_true', help='use model inference on video')

#  check both train/test validation data
# add drop out to network
# slower schedualer for LR
# more data  

def main():
    global arg
    arg = parser.parse_args()
    print arg

    if not arg.demo:
        #Prepare DataLoader
        data_loader = dataloader.spatial_dataloader(
                            BATCH_SIZE=arg.batch_size,
                            num_workers=8,
                            # path='/home/niloofar/Work/Data/Spatial/UCF-101/',
                            path='/home/niloofar/Work/Data/Spatial/UCF-101/',
                            ucf_list =os.getcwd()+'/UCF_list/',
                            # ucf_list =os.getcwd()+'/My_list/',
                            ucf_split ='01',
                            )

        data_loader_my = dataloader.spatial_dataloader(
                            BATCH_SIZE=arg.batch_size_my,
                            num_workers=8,
                            # path='/home/niloofar/Work/Data/Spatial/UCF-101/',
                            path='/home/niloofar/Work/Data/Spatial/UCF-101/',
                            # ucf_list =os.getcwd()+'/UCF_list/',
                            ucf_list =os.getcwd()+'/My_list/',
                            ucf_split ='01',
                            )

        train_loader, test_loader, test_video = data_loader.run()
        train_loader_my, test_loader_my, test_video_my = data_loader_my.run()

        #Model
        model = Spatial_CNN(
                            nb_epochs=arg.epochs,
                            lr=arg.lr,
                            batch_size=arg.batch_size,
                            batch_size_my=arg.batch_size_my,
                            resume=arg.resume,
                            start_epoch=arg.start_epoch,
                            evaluate=arg.evaluate,
                            demo=arg.demo,
                            train_loader=train_loader,
                            train_loader_my=train_loader_my,
                            test_loader=test_loader,
                            test_loader_my=test_loader_my,
                            test_video=test_video,
                            test_video_my=test_video_my
                            )
    else:
        #Model
        model = Spatial_CNN(
                            nb_epochs=arg.epochs,
                            lr=arg.lr,
                            batch_size=arg.batch_size,
                            resume=arg.resume,
                            start_epoch=arg.start_epoch,
                            evaluate=arg.evaluate,
                            demo=arg.demo)



    # Run
    model.run()




class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size,batch_size_my, resume, start_epoch, evaluate,  demo, train_loader=None, train_loader_my=None, test_loader=None, test_loader_my=None, test_video=None, test_video_my=None):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.batch_size_my=batch_size_my
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.train_loader_my=train_loader_my
        self.test_loader=test_loader
        self.test_loader_my=test_loader_my
        self.best_prec1=0
        self.test_video=test_video
        self.test_video_my=test_video_my
        self.demo = demo

    def webcam_inference(self):

        frame_count = 0

        # config the transform to match the network's format
        transform = transforms.Compose([
                transforms.Resize((342, 256)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # prepare the translation dictionary label-action
        # data_handler = UCF101_splitter(os.getcwd()+'/UCF_list/', None)
        data_handler = my_splitter(os.getcwd()+'/UCF_list/', None)
        data_handler.get_action_index()
        class_to_idx = data_handler.action_label
        idx_to_class = {v: k for k, v in class_to_idx.iteritems()}

        # Start looping on frames received from webcam
        vs = cv2.VideoCapture(-1)
        softmax = torch.nn.Softmax()
        nn_output = torch.tensor(np.zeros((1, 101)), dtype=torch.float32).cuda()

        while True:
            # read each frame and prepare it for feedforward in nn (resize and type)
            ret, orig_frame = vs.read()

            if ret is False:
                print "Camera disconnected or not recognized by computer"
                break

            if frame_count == 0:
                old_frame = orig_frame.copy()

            else:
                optical_flow = opt_flow_infer(old_frame, orig_frame)
                old_frame = orig_frame.copy()

            frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame).view(1, 3, 224, 224).cuda()

            # feed the frame to the neural network
            nn_output += self.model(frame)

            # vote for class with 25 consecutive frames
            if frame_count % 10 == 0:
                nn_output = softmax(nn_output)
                nn_output = nn_output.data.cpu().numpy()
                preds = nn_output.argsort()[0][-5:][::-1]
                pred_classes = [(idx_to_class[str(pred+1)], nn_output[0, pred]) for pred in preds]

                # reset the process
                nn_output = torch.tensor(np.zeros((1, 101)), dtype=torch.float32).cuda()

            # Display the resulting frame and the classified action
            font = cv2.FONT_HERSHEY_SIMPLEX
            y0, dy = 300, 40
            for i in xrange(5):
                y = y0 + i * dy
                cv2.putText(orig_frame, '{} - {:.2f}'.format(pred_classes[i][0], pred_classes[i][1]),
                            (5, y), font, 1, (0, 0, 255), 2)

            cv2.imshow('Webcam', orig_frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        vs.release()
        cv2.destroyAllWindows()


    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained=True, channel=3).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5,verbose=True)

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

        elif self.demo:
            self.model.eval()
            self.webcam_inference()

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True

        if self.evaluate or self.demo:
            return

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/test9/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.my_dic_video_level_preds,f)
                f.close()

            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/test9/checkpoint.pth.tar','record/test9/model_best.pth.tar')

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

        self.train_loader.dataset.offset = np.random.randint(len(self.train_loader.dataset))

        # for i, (data_dict,label) in enumerate(progress_my):
        for (my_batch,ucf_batch) in tqdm(zip(self.train_loader_my, self.train_loader)):

            data_time.update(time.time() - end)

            # ---------------  My Loss -------------
            data_dict = my_batch[0]
            label = my_batch[1]

            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            output = Variable(torch.zeros(len(data_dict['img1']),101).float()).cuda()
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)


            loss_my = self.criterion(output, target_var)
            prec1_my,prec5_my = accuracy(output.data, label, topk=(1, 5))
            
            # ---------------  UCF Loss -------------
            data_dict = ucf_batch[0]
            label = ucf_batch[1]

            label = label.cuda(async=True)
            target_var = Variable(label).cuda()
            
            output = Variable(torch.zeros(len(data_dict['img1']),101).float()).cuda()
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)
            
            loss_ucf = self.criterion(output, target_var)
            prec1,prec5  = accuracy(output.data, label, topk=(1, 5))

            
            loss = loss_my + 0.3*loss_ucf;

            losses.update(loss.data, data.size(0))
            top1.update(prec1_my, data.size(0))
            top5.update(prec1, data.size(0))
            # top5.update(prec5, data.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        #Save details in csv
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'MyPrec':[round(top1.avg,4)],
                'UCFPrec':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/test9/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        my_top1 = AverageMeter()
        my_top5 = AverageMeter()

        ucf_top1 = AverageMeter()
        ucf_top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.my_dic_video_level_preds={}
        self.ucf_dic_video_level_preds={}
        end = time.time()
        batch_count = 0
        loss_sum_my = 0
        prec1_sum_my = 0
        loss_sum_ucf = 0
        prec1_sum_ucf = 0

        # progress = tqdm(self.test_loader_my)
        with torch.no_grad():
            for (my_batch,ucf_batch) in tqdm(zip(self.test_loader_my, self.test_loader)):

         #----------------------------My Data--------------------------------------------

                keys = my_batch[0]
                data = my_batch[1]
                label = my_batch[2]
                
                label = label.cuda(async=True)
                data_var = Variable(data).cuda(async=True)
                target_var = Variable(label).cuda(async=True)

                # compute output
                output = self.model(data_var)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                #Frame-level accuracy and loss
                batch_loss_my = self.criterion(output, target_var)
                batch_prec1_my,batch_prec5_my = accuracy(output.data, label, topk=(1, 5))

                loss_sum_my += batch_loss_my
                prec1_sum_my += batch_prec1_my


                #Calculate MY video level prediction
                my_preds = output.data.cpu().numpy()
                my_nb_data = my_preds.shape[0]
                for j in range(my_nb_data):
                    videoName = keys[j].split('/',1)[0]
                    if videoName not in self.my_dic_video_level_preds.keys():
                        self.my_dic_video_level_preds[videoName] = my_preds[j,:]
                    else:
                        self.my_dic_video_level_preds[videoName] += my_preds[j,:]


        #----------------------------UCF Data--------------------------------------------
                keys = ucf_batch[0]
                data = ucf_batch[1]
                label = ucf_batch[2]

                label = label.cuda(async=True)
                data_var = Variable(data).cuda(async=True)
                target_var = Variable(label).cuda(async=True)

                # compute output
                output = self.model(data_var)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                #Frame-level accuracy and loss
                batch_loss_ucf = self.criterion(output, target_var)
                batch_prec1_ucf,batch_prec5_ucf  = accuracy(output.data, label, topk=(1, 5))

                loss_sum_ucf += batch_loss_ucf
                prec1_sum_ucf += batch_prec1_ucf

                #Calculate MY video level prediction
                ucf_preds = output.data.cpu().numpy()
                ucf_nb_data = ucf_preds.shape[0]
                for j in range(ucf_nb_data):
                    videoName = keys[j].split('/',1)[0]
                    if videoName not in self.ucf_dic_video_level_preds.keys():
                        self.ucf_dic_video_level_preds[videoName] = ucf_preds[j,:]
                    else:
                        self.ucf_dic_video_level_preds[videoName] += ucf_preds[j,:]

                batch_count += 1


        loss_my = loss_sum_my/batch_count
        loss_ucf = loss_sum_ucf/batch_count
        total_loss = loss_my + 0.3*loss_ucf

        prec1_my = prec1_sum_my/batch_count
        prec1_ucf = prec1_sum_ucf/batch_count

        my_video_top1, ucf_video_top1, total_video_loss = self.frame2_video_level_accuracy()
        print "My Video Level Accuracy Is: ", my_video_top1
        print "UCF Video Level Accuracy Is: ", ucf_video_top1


        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(total_loss,5)],
                'MyPrec':[round(prec1_my,4)],
                'UCFPrec':[round(prec1_ucf,4)]}
        record_info(info, 'record/test9/rgb_test.csv','test')
        return prec1_my, total_loss

    def frame2_video_level_accuracy(self):

        my_video_correct = 0
        ucf_video_correct = 0
        my_video_level_preds = np.zeros((len(self.my_dic_video_level_preds),101))
        my_video_level_labels = np.zeros(len(self.my_dic_video_level_preds))

        ucf_video_level_preds = np.zeros((len(self.ucf_dic_video_level_preds),101))
        ucf_video_level_labels = np.zeros(len(self.ucf_dic_video_level_preds))
        ii=0
        jj=0

        for name in sorted(self.ucf_dic_video_level_preds.keys()):

            preds = self.ucf_dic_video_level_preds[name]
            label = int(self.test_video[name])-1

            ucf_video_level_preds[ii,:] = preds
            ucf_video_level_labels[ii] = label
            ii+=1
            if np.argmax(preds) == (label):
                ucf_video_correct+=1

        for name in sorted(self.my_dic_video_level_preds.keys()):

            preds = self.my_dic_video_level_preds[name]
            label = int(self.test_video_my[name])-1

            my_video_level_preds[jj,:] = preds
            my_video_level_labels[jj] = label
            jj+=1
            if np.argmax(preds) == (label):
                my_video_correct+=1

        #top1 top5 and loss for my data
        my_video_level_labels = torch.from_numpy(my_video_level_labels).long()
        my_video_level_preds = torch.from_numpy(my_video_level_preds).float()

        my_video_top1,my_video_top5 = accuracy(my_video_level_preds, my_video_level_labels, topk=(1,5))
        my_video_loss = self.criterion(Variable(my_video_level_preds).cuda(), Variable(my_video_level_labels).cuda())

        my_video_top1 = float(my_video_top1.numpy())
        my_video_top5 = float(my_video_top5.numpy())


        #top1 top5 and loss for ucf data
        ucf_video_level_labels = torch.from_numpy(ucf_video_level_labels).long()
        ucf_video_level_preds = torch.from_numpy(ucf_video_level_preds).float()
        
        ucf_video_top1,ucf_video_top5 = accuracy(ucf_video_level_preds, ucf_video_level_labels, topk=(1,5))
        ucf_video_loss = self.criterion(Variable(ucf_video_level_preds).cuda(), Variable(ucf_video_level_labels).cuda())

        ucf_video_top1 = float(ucf_video_top1.numpy())
        ucf_video_top5 = float(ucf_video_top5.numpy())

        # print "my correct is ",my_correct
        # print "UCF correct is ", ucf_correct

        video_loss = my_video_loss + ucf_video_loss

        # top1 = float(top1.numpy())
        # top5 = float(top5.numpy())

        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return my_video_top1,ucf_video_top1,video_loss.data.cpu().numpy()



if __name__=='__main__':
    main()
