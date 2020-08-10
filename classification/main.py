import sys
import numpy as np
import os
import time
import shutil
import random
import argparse
from imblearn import under_sampling, over_sampling

# Add the parent directory to the path such that all modules can be found
filePath = os.path.abspath(__file__)
fileDir = os.path.dirname(filePath)
parentDir = os.path.dirname(fileDir)
sys.path.insert(0, parentDir)

localtime = time.localtime(time.time())
localtime = time.asctime(localtime)
default_experiment = localtime.replace(' ', '_')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Touch-Classification.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default=('/scratch1/msc20f10/data/3kOhm_FB/'
                                          + 'data_MT_FabianGeiger_5sess.mat'),
                    help="Path to metadata.mat file.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True,
                    default=False,
                    help="Start from scratch (do not load weights).")
parser.add_argument('--test', type=str2bool, nargs='?', const=True,
                    default=False, help="Just run test and quit.")
parser.add_argument('--snapshotDir',
                    default='/scratch1/msc20f10/stag/training_checkpoints',
                    help="Where to store checkpoints during training.")
parser.add_argument('--gpu', type=int, default=None,
                    help=("ID number of the GPU to use [0--4].\n"
                          + "If left unspecified all visible CPUs."))
parser.add_argument('--experiment', default=default_experiment,
                    help="Name of the current experiment.")
parser.add_argument('--nfilters', type=int, default=16,
                    help="Number of filters for the first convolution.")
parser.add_argument('--dropout', type=float, default=0,
                    help="Dropout between the two ResNet blocks.")
parser.add_argument('--epochs', type=int, default=30,
                    help="Number of epochs to train.")
parser.add_argument('--kfoldCV', type=int, default=None,
                    help=("performs 6 fold cross validation.\n"
                          + "Only has an effect with --dataSplit 'random'"))
parser.add_argument('--dataSplit', type=str, default='random',
                    help=("Can either be 'session' or 'random', relating"
                          + " to the way the data is split into training and test set."))
parser.add_argument('--makeRepeatable', type=str2bool, nargs='?', const=True,
                    default=False,
                    help="Makes experiments repeatable by using a constant seed.")
args = parser.parse_args()

# This line makes only the chosen GPU visible.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
#________________________________________________________________________________________#

import torch
import torch.backends.cudnn as cudnn

from CustomDataLoader import CustomDataLoader
from shared.dataset_tools import load_data

nClasses = 17
epochs = args.epochs
batch_size = 32
workers = 0
experiment = args.experiment
balance_per_epoch = False
oversample = False
undersample = ~oversample
split = args.dataSplit

metaFile = args.dataset
if split == 'random':
    kFoldCV = args.kfoldCV
else:
    kFoldCV = None

if args.makeRepeatable:
    seed = 333
else:
    seed = int(333 + time.time() + os.getpid())
 

class Trainer(object):
    def __init__(self, data_set):
        self.init(data_set)
        super(Trainer, self).__init__()


    def init(self, data_set):
        # Init model
        self.data = {}
        # This data was loaded with the custom_dataloader
        self.data['train'] = data_set[0:2]
        self.data['test'] = data_set[2:4]

        self.val_loader = self.loadDatasets('test', True)

        self.initModel()


    def loadDatasets(self, split='train', shuffle=True):
        # With custom data, the previously loaded data is used to
        # instantiate a dataloader
        # Expect data to be of shape (set_size, 1024)
        return torch.utils.data.DataLoader(
            CustomDataLoader(self.data[split][0],
                             self.data[split][1], augment=(split=='train'),
                             split=split,
                             nclasses=nClasses, balance=balance_per_epoch),
            batch_size=batch_size, shuffle=shuffle, num_workers=workers)


    def run(self):
        """
        Implements one complete training for a given number of epochs.

        Returns
        -------
        None.

        """
        # These lists are used to save all precision and loss values through
        # the training
        trainloss_history = []
        trainprec_history = []
        testloss_history = []
        testprec_history = []
        print('[Trainer] Starting...')

        self.counters = {
            'train': 0,
            'test': 0,
        }

        if args.test:
            # Just loads the best model and tests it on the given test data
            self.doSink()
            return

        # The validation loader contains the test data
        val_loader = self.val_loader
        train_loader = self.loadDatasets('train', True)

        for epoch in range(epochs):
            print('Epoch %d/%d....' % (epoch, epochs))
            self.model.updateLearningRate(epoch)

            trainprec1, _, trainloss, cm_train = self.step(train_loader,
                                                           self.model.epoch,
                                                           isTrain=True)
            prec1, _, testloss, _ = self.step(val_loader,
                                           self.model.epoch,
                                           isTrain=False,
                                           sinkName=None)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.model.bestPrec
            if is_best:
                self.model.bestPrec = prec1
            self.model.epoch = epoch + 1
            if not args.snapshotDir is None:
                self.saveCheckpoint(self.model.exportState(), is_best)
      
            trainloss_history.append(trainloss)
            trainprec_history.append(trainprec1)
            testloss_history.append(testloss)
            testprec_history.append(prec1)

        # Final results
        res, cm_test = self.doSink()

        savedir = '/home/msc20f10/Python_Code/results/stag_realData/'
        savedir = savedir + 'np' + str(self.model.nParams) + '_'
        np.save(savedir + experiment + '_history.npy',
                np.array([trainloss_history, trainprec_history,
                          testloss_history, testprec_history, res,
                          cm_train.numpy(), cm_test.numpy(),
                          self.model.ntParams, self.model.nParams]))

        print('DONE')
        return res


    def doSink(self):
        """
        Does the final test on the given test data. If the flag for test was
        set in the command-line argument, this function is run after loading
        a previously trained model.

        Returns
        -------
        res : dictionary
            python dictionary containing the results for top1 and top3
            precision for random and clustering.
        conf_mat : numpy array
            confusion matrix for the random collation of frames.
        conf_mat_cluster : numpy array
            confusion matrix for the random collation of frames.

        """
        res = {}

        print('Running test...')
        res['test-top1'], res['test-top3'],\
            _, conf_mat = self.step(self.val_loader, self.model.epoch,
                                    isTrain=False, sinkName='test')

        print('--------------\nResults:')
        for k,v in res.items():
            print('\t%s: %.3f %%' % (k,v))
    
        return res, conf_mat


    def initModel(self):
        cudnn.benchmark = True

        from ClassificationModel import ClassificationModel as Model # the main model

        initShapshot = os.path.join(args.snapshotDir, experiment,
                                    'model_best.pth.tar')
        if args.reset:
            initShapshot = None

        # Initialize the model with all the command-line arguments
        self.model = Model(numClasses=nClasses, inplanes=args.nfilters,
                           dropout=args.dropout)
        self.model.epoch = 0
        self.model.bestPrec = -1e20

        if not initShapshot is None:
            state = torch.load(initShapshot)
            assert not state is None, 'Warning: Could not read checkpoint %s!' % initShapshot
            print('Loading checkpoint %s...' % (initShapshot))
            self.model.importState(state)


    def step(self, data_loader, epoch, isTrain=True, sinkName=None):
        """
        Implements one step through all batches.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Either training or test data loader.
        epoch : int
            Current training epoch.
        isTrain : bool, optional
            Whether this step is used for training or test. The default is True.
        sinkName : string, optional
            Name of the results. The default is None.

        Returns
        -------
        top1.avg : AverageMeter
            Average value of top1 precision after current step.
        top3.avg : AverageMeter
            Average value of top3 precision after current step.
        losses.avg : AverageMeter
            Average value of loss after current step.
        conf_matrix : numpy array
            Confusion matrix after current step.

        """
        if isTrain:
            # If more than one frame, this generates new combinations of frames
            data_loader.dataset.refresh()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        end = time.time()
        conf_matrix = torch.zeros(nClasses, nClasses).cpu()
        for i, (inputs) in enumerate(data_loader):
            data_time.update(time.time() - end)
          
            inputsDict = {
                'pressure': inputs[0],
                'objectId': inputs[1],
                }

            res, loss = self.model.step(inputsDict, isTrain,
                                        params = {'debug': True})

            losses.update(loss['Loss'], batch_size)
            top1.update(loss['Top1'], batch_size)
            top3.update(loss['Top3'], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if isTrain:
                self.counters['train'] = self.counters['train'] + 1

            # Clear line before printing
            sys.stdout.write('\033[K')
            sys.stdout.flush()
            print('{phase}: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'
                  .format(epoch, i, len(data_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top3=top3,
                          phase=('Train' if isTrain else 'Test')),
                  end='\r', flush=True)

            # Calculate the confusion matrix
            for t, p in zip(inputs[1].view(-1), res['pred'].view(-1)):
                conf_matrix[t.long(), p.long()] += 1
      
        print('')
        self.counters['test'] = self.counters['test'] + 1

        return top1.avg, top3.avg, losses.avg, conf_matrix


    def saveCheckpoint(self, state, is_best):
        snapshotDir = os.path.join(args.snapshotDir, experiment)
        if not os.path.isdir(snapshotDir):
            os.makedirs(snapshotDir, 0o777)
        chckFile = os.path.join(snapshotDir, 'checkpoint.pth.tar')
        print('Writing checkpoint to %s...' % chckFile)
        torch.save(state, chckFile)
        if is_best:
            bestFile = os.path.join(snapshotDir, 'model_best.pth.tar')
            shutil.copyfile(chckFile, bestFile)
        print('\t...Done.\n')


    @staticmethod
    def make(data_set):
        if args.makeRepeatable:
            random.seed(454878)
            np.random.seed(12683)
            torch.manual_seed(23142)
        else:
            random.seed(454878 + time.time() + os.getpid())
            np.random.seed(int(12683 + time.time() + os.getpid()))
            torch.manual_seed(23142 + time.time() + os.getpid())

        ex = Trainer(data_set)
        res = ex.run()
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

if __name__ == "__main__":
    # Load the data with a random stratified train/test split if split='random'
    # or split into the recording sessions if split='session'
    data_set = load_data(filename=metaFile, kfold=kFoldCV, seed=seed,
                         undersample=undersample, split=split)

    # k fold cross validation
    if kFoldCV:
        train_data = data_set[0]
        train_labels = data_set[1]
        test_data = data_set[2]
        test_labels = data_set[3]
        skf = data_set[4]
        
        res_top1 = []
        res_top3 = []
        for train_index, val_index in skf.split(train_data, train_labels):
            dataset = [train_data[train_index], train_labels[train_index],
                       train_data[val_index], train_labels[val_index]]
            res = Trainer.make(dataset)
            res_top1.append(res['test-top1'])
            res_top3.append(res['test-top3'])
        
        print('\nResults for 6 fold cross validation:')
        print('\tTop 1: mean {:.3f}% std. dev. {:.3f}%'
              .format(np.mean(res_top1), np.std(res_top1)))
        print('\tTop 3: mean {:.3f}% std. dev. {:.3f}%'
              .format(np.mean(res_top3), np.std(res_top3)))

    elif split == 'session':
        # The data set returns lists for tactile data and labels. Each element
        # in the list corresponds to a recording session
        # Balance the data once and use the same data for all tests
        if(oversample or undersample):
            data_balanced = []
            labels_balanced = []
            for i in range(len(data_set[1])):
                if oversample:
                    osr = over_sampling.KMeansSMOTE(random_state=seed,
                                                    kmeans_estimator=15,
                                                    k_neighbors=3)
                    sampled_data,\
                        sampled_labels = osr.fit_resample(data_set[0][i],
                                                          data_set[1][i])
                elif undersample:
                    usr = under_sampling.RandomUnderSampler(random_state=seed,
                                                            sampling_strategy='not minority')
                    sampled_data,\
                        sampled_labels = usr.fit_resample(data_set[0][i],
                                                          data_set[1][i])
                data_balanced.append(sampled_data)
                labels_balanced.append(sampled_labels)
            data_set = [data_balanced, labels_balanced]
        
        res_top1 = []
        res_top3 = []
        for i in range(len(data_set[1])):
            train_data = np.delete(data_set[0], i)
            train_data = np.concatenate(train_data)
            train_labels = np.delete(data_set[1], i)
            train_labels = np.concatenate(train_labels)
            test_data = data_set[0][i]
            test_labels = data_set[1][i]
                
            dataset = [train_data, train_labels, test_data, test_labels]
            res = Trainer.make(dataset)
            res_top1.append(res['test-top1'])
            res_top3.append(res['test-top3'])
        
        print('\nResults for inter-session cross validation:')
        for i in range(len(data_set[1])):
            print('\tSession', i,'as test data Top 1: {:.3f}%\t Top 3: {:.3f}%'
                  .format(res_top1[i], res_top3[i]))
        print('\t=> Top 1 mean {:.3f}% std. dev. {:.3f}%'
              .format(np.mean(res_top1), np.std(res_top1)))
        print('\t=> Top 3 mean {:.3f}% std. dev. {:.3f}%'
              .format(np.mean(res_top3), np.std(res_top3)))
    
    else:
        train_data = data_set[0]
        train_labels = data_set[1]
        test_data = data_set[2]
        test_labels = data_set[3]

        data_set = [train_data, train_labels, test_data, test_labels]
        Trainer.make(data_set)
