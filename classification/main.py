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

parser = argparse.ArgumentParser(description='Touch-Classification Slim12.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default=('/scratch1/msc20f10/data/'
                                          + 'classification/metadata.mat'),
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
parser.add_argument('--dropout', type=float, default=0.4,
                    help="Dropout between the two ResNet blocks.")
parser.add_argument('--epochs', type=int, default=30,
                    help="Number of epochs to train.")
parser.add_argument('--customData', type=str, default=None,
                    help=("Which split to use in the custom data loader.\n"
                          + "If left to None the original data loader is used.\n" 
                          + "Options: 'random', 'original', 'recording'"))
parser.add_argument('--kfoldCV', type=int, default=None,
                    help=("performs 6 fold cross validation.\n"
                          + "Only has an effect with --customData 'random'"))
parser.add_argument('--makeRepeatable', type=str2bool, nargs='?', const=True,
                    default=False,
                    help="Makes experiments repeatable by using a constant seed.")
parser.add_argument('--dropClasses', type=str2bool, nargs='?', const=True,
                    default=False, help="Drop the 7 worst performing classes.")
args = parser.parse_args()

# This line makes only the chosen GPU visible.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
#________________________________________________________________________________________#

import torch
import torch.backends.cudnn as cudnn

from CustomDataLoader import CustomDataLoader
from shared.dataset_tools import load_data


if args.dropClasses:
    # classes with accuracy < 50% in confusion matrix
    #to_drop = [0, 1, 5, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 26]
    # classes with worst accuracy in confusion matrix
    # to_drop = [1, 18, 19, 21, 23, 24]
    to_drop = [19]
    # classes that take away accuracy
    # to_drop = [2, 5, 7, 11, 12, 16, 17, 20, 25]
    # fuse confusing classes
    to_fuse = [12, 13]
    nClasses = 27 - len(to_drop)
    if len(to_fuse) != 0:
        nClasses -= len(to_fuse)-1
else:
    nClasses = 27
epochs = args.epochs
batch_size = 32
workers = 0
experiment = args.experiment
balance_per_epoch = True

metaFile = args.dataset
doFilter = True
if args.customData is None:
    kFoldCV = None
else:
    kFoldCV = args.kfoldCV
 

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

        self.val_loader = self.loadDatasets('test', False, False)

        self.initModel()


    def loadDatasets(self, split='train', shuffle=True,
                     useClusterSampling=False):
        # With custom data, the previously loaded data is used to
        # instantiate a dataloader
        set_size = len(self.data[split][1])
        return torch.utils.data.DataLoader(
            CustomDataLoader(self.data[split][0].reshape((set_size, 32, 32)),
                             self.data[split][1], augment=(split=='train'),
                             use_clusters=False, split=split,
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
        train_loader = self.loadDatasets('train', True, False)

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
        res, cm_test, cm_test_cl = self.doSink()

        savedir = '/home/msc20f10/Python_Code/results/stag/'
        if args.experiment == 'slim16':
            savedir = savedir + 'slim16/'
        else:
            savedir = savedir + 'np' + str(self.model.nParams) + '_'
        np.save(savedir + experiment + '_history.npy',
                np.array([trainloss_history, trainprec_history,
                          testloss_history, testprec_history, res,
                          cm_train.numpy(), cm_test.numpy(), cm_test_cl.numpy(),
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

        # So far clustering is only used if the custom data loading is NOT used
        print('Running test with clustering...')
        val_loader_cluster = self.loadDatasets('test', False, args.customData is None)
        res['test_cluster-top1'], res['test_cluster-top3'],\
            _, conf_mat_cluster = self.step(val_loader_cluster,
                                            self.model.epoch,
                                            isTrain=False,
                                            sinkName='test_cluster')

        print('--------------\nResults:')
        for k,v in res.items():
            print('\t%s: %.3f %%' % (k,v))
    
        return res, conf_mat, conf_mat_cluster


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
                'image': inputs[1],
                'pressure': inputs[2],
                'objectId': inputs[3],
                }

            res, loss = self.model.step(inputsDict, isTrain,
                                        params = {'debug': True})

            losses.update(loss['Loss'], inputs[0].size(0))
            top1.update(loss['Top1'], inputs[0].size(0))
            top3.update(loss['Top3'], inputs[0].size(0))

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
            for t, p in zip(inputs[3].view(-1), res['pred'].view(-1)):
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
        

def drop_classes(in_data, in_labels, to_drop):
    """
    Drops the specified classes from data and label arrays. Reorders the label
    array such that the class numbering is a continuous range.

    Parameters
    ----------
    in_data : numpy array
        Data from which the classes should be dropped.
    in_labels : numpy array
        Labels from which the classes should be dropped.
    to_drop : list, array
        Classes to drop.

    Returns
    -------
    data : numpy array
        Data with specified classes removed.
    labels : numpy array
        Labels with specified classes removed.

    """
    mask = np.zeros((len(in_labels),))
    for j in range(len(to_drop)):
        drop_mask = in_labels == to_drop[j]
        mask = np.logical_or(mask, drop_mask)
    mask = np.invert(mask)
    
    data = in_data[mask]
    labels = in_labels[mask]
    
    # need to extend the list to make the for loop work
    to_drop += [100]
    for k in range(1, len(to_drop)):
        mask = np.logical_and(labels > to_drop[k-1], labels < to_drop[k])
        labels[mask] -= k
        
    return data, labels


def fuse_classes(in_labels, to_fuse):
    labels = np.copy(in_labels)
    # Sort the array such that the smalles value is first
    to_fuse.sort()
    for i in range(1, len(to_fuse)):
        fuse_mask = in_labels == to_fuse[i]
        labels[fuse_mask] = to_fuse[0]

    to_fuse += [100]
    for k in range(1, len(to_fuse)-1):
        mask = np.logical_and(labels > to_fuse[k], labels < to_fuse[k+1])
        labels[mask] -= k
    
    return labels


if __name__ == "__main__":
    # Load the data either with the original train/test split if split='original'
    # or with a random stratified train/test split if split='random'
    # or split into the recording sessions if split='recording'
    if args.makeRepeatable:
        seed = 333
    else:
        seed = int(333 + time.time() + os.getpid())
        
    undersample = False
    data_set = load_data(filename=metaFile, kfold=kFoldCV, split=args.customData,
                         seed=seed, undersample=undersample)

    # k fold cross validation
    if kFoldCV and args.customData == 'random':
        train_data = data_set[0]
        train_labels = data_set[1]
        test_data = data_set[2]
        test_labels = data_set[3]
        skf = data_set[4]
        if args.dropClasses:
            train_data, train_labels = drop_classes(train_data, train_labels,
                                                    to_drop)
            test_data, test_labels = drop_classes(test_data, test_labels,
                                                  to_drop)
            if len(to_fuse) != 0:
                train_labels = fuse_classes(train_labels, to_fuse)
                test_labels = fuse_classes(test_labels, to_fuse)
        
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

    elif args.customData == 'recording':
        cross_validate = True
        intra_session = False
        
        x = np.array([data_set[0], data_set[2], data_set[4]])
        y = np.array([data_set[1], data_set[3], data_set[5]])
        if cross_validate:
            res_top1 = []
            res_top3 = []
            for i in range(3):                    
                test_data = x[i]
                test_labels = y[i]

                if not undersample:
                    undersampler = under_sampling.RandomUnderSampler(random_state=seed+i)
                    # Balance the test data
                    test_data, test_labels = undersampler.fit_resample(test_data,
                                                                       test_labels)
                # Remove ith element from array and concatenate the rest into one
                train_data = np.delete(x, i)
                train_data = np.concatenate((train_data[0], train_data[1]), axis=0)
                train_labels = np.delete(y, i)
                train_labels = np.concatenate((train_labels[0], train_labels[1]), axis=0)

                if not balance_per_epoch and not undersample:
                    # Undersample majority class to the second largest class
                    # class 0 is always the biggest class (empty hand)
                    nmax = 0
                    for i in range(1, 27):
                        mask = train_labels == i
                        n = np.count_nonzero(mask)
                        if n > nmax:
                            nmax = n
                    strat = {0: nmax}
                    usr = under_sampling.RandomUnderSampler(random_state=seed,
                                                            sampling_strategy=strat)
                    traindata_resampled,\
                        trainlabels_resampled = usr.fit_resample(train_data,
                                                                  train_labels)
                    
                    
                    # Then oversample the rest of the classes such that the set is balanced.
                    # KMeansSMOTE oversampling. This generates NEW samples!
                    # Can be seen as data augmentation. kmeans_estimator tells the sampler
                    # how many clusters to generate
                    osr = over_sampling.KMeansSMOTE(random_state=seed+2,
                                                    kmeans_estimator=15,
                                                    k_neighbors=3)
                    train_data,\
                        train_labels = osr.fit_resample(traindata_resampled,
                                                        trainlabels_resampled)    
                                        
                    # # Try to oversample without undersampling class 0 first
                    # train_data, train_labels = osr.fit_resample(train_data,
                    #                                             train_labels)
                    
                if args.dropClasses:
                    train_data, train_labels = drop_classes(train_data, train_labels,
                                                            to_drop)
                    test_data, test_labels = drop_classes(test_data, test_labels,
                                                          to_drop)
                    if len(to_fuse) != 0:
                        train_labels = fuse_classes(train_labels, to_fuse)
                        test_labels = fuse_classes(test_labels, to_fuse)

                dataset = [train_data, train_labels, test_data, test_labels]
                res = Trainer.make(dataset)
                res_top1.append(res['test-top1'])
                res_top3.append(res['test-top3'])
    
            print('\nResults for recording cross validation:')
            print('\tTop 1: {:.3f}% + {:.3f}% + {:.3f}% = {:.3f}%'
                  .format(res_top1[0], res_top1[1], res_top1[2], np.mean(res_top1)))
            print('\tTop 3: {:.3f}% + {:.3f}% + {:.3f}% = {:.3f}%\n'
                  .format(res_top3[0], res_top3[1], res_top3[2], np.mean(res_top3)))
        elif intra_session:
            from sklearn.model_selection import StratifiedKFold
            top1 = []
            top3 = []
            for i in range(3):                    
                data = x[i]
                labels = y[i]

                skf = StratifiedKFold(n_splits=6, shuffle=True,
                                      random_state=seed+i)
                res_top1 = []
                res_top3 = []
                for train_index, val_index in skf.split(data, labels):
                    dataset = [data[train_index], labels[train_index],
                               data[val_index], labels[val_index]]
                    res = Trainer.make(dataset)
                    res_top1.append(res['test-top1'])
                    res_top3.append(res['test-top3'])

                top1.append(res_top1)
                top3.append(res_top3)
            
            print('\nResults for intra-session cross validation:')
            print('\tTop 1: {:.3f}% + {:.3f}% + {:.3f}% = {:.3f}%'
                  .format(np.mean(top1[0]), np.mean(top1[1]),
                          np.mean(top1[2]), np.mean(top1)))
            print('\tStandard deviation top 1: {:.3f}, {:.3f}, {:.3f}'
                  .format(np.std(top1[0]), np.std(top1[1]), np.std(top1[2])))
            print('\tTop 3: {:.3f}% + {:.3f}% + {:.3f}% = {:.3f}%'
                  .format(np.mean(top3[0]), np.mean(top3[1]),
                          np.mean(top3[2]), np.mean(top3)))
            print('\tStandard deviation top 3: {:.3f}, {:.3f}, {:.3f}\n'
                  .format(np.std(top3[0]), np.std(top3[1]), np.std(top3[2])))
        else:
            test_data = x[1]
            test_labels = y[1]

            train_data = np.concatenate((x[0], x[2]), axis=0)
            train_labels = np.concatenate((y[0], y[2]), axis=0)
            
            if not undersample:
                undersampler = under_sampling.RandomUnderSampler(random_state=seed)
                # Balance the test data
                test_data, test_labels = undersampler.fit_resample(test_data,
                                                                   test_labels)

            if args.dropClasses:
                train_data, train_labels = drop_classes(train_data, train_labels,
                                                        to_drop)
                test_data, test_labels = drop_classes(test_data, test_labels,
                                                  to_drop)
                if len(to_fuse) != 0:
                    train_labels = fuse_classes(train_labels, to_fuse)
                    test_labels = fuse_classes(test_labels, to_fuse)
                    
            dataset = [train_data, train_labels, test_data, test_labels]
            res = Trainer.make(dataset)

    else:
        train_data = data_set[0]
        train_labels = data_set[1]
        test_data = data_set[2]
        test_labels = data_set[3]
        if args.dropClasses:
            train_data, train_labels = drop_classes(train_data, train_labels,
                                                    to_drop)
            test_data, test_labels = drop_classes(test_data, test_labels,
                                                  to_drop)
            if len(to_fuse) != 0:
                    train_labels = fuse_classes(train_labels, to_fuse)
                    test_labels = fuse_classes(test_labels, to_fuse)

        data_set = [train_data, train_labels, test_data, test_labels]
        Trainer.make(data_set)
