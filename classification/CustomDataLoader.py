#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:37:27 2020

@author: fabian
"""


import numpy as np
import random
from sklearn.utils import shuffle
import torch.utils.data as data
import torch        
from imblearn import over_sampling, under_sampling, combine
from scipy.ndimage import gaussian_filter


class CustomDataLoader(data.Dataset):
    def __init__(self, data, labels, augment=False, nclasses=27,
                 balance=False, split='train'):
        self.nclasses = nclasses
        self.data = data
        self.labels = labels
        self.augment = augment
        self.balance = balance
        self.split = split
        if balance:
            self.original_data = data
            self.original_labels = labels
            self.balance_data()
        self.collate_data()
        

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        pressure = self.collated_data[idx].reshape((32,32))
        if self.augment:
            #noise = torch.randn_like(pressure) * 0.015#0.015
            noise = np.random.normal(size=pressure.shape) * 0.015
            pressure += noise
# =============================================================================
#         pressure = gaussian_filter(pressure, sigma=0.6)
#         mask = np.array([np.ones(32), np.ones(32), np.ones(32),
#                  np.concatenate((np.zeros(14), np.ones(18))),
#                  np.concatenate((np.zeros(14), np.ones(18))),
#                  np.concatenate((np.zeros(14), np.ones(18))),
#                  np.ones(32), np.ones(32), np.ones(32),
#                  np.concatenate((np.zeros(14), np.ones(18))),
#                  np.ones(32), np.ones(32), np.ones(32),
#                  np.concatenate((np.zeros(14), np.ones(18))),
#                  np.concatenate((np.zeros(14), np.ones(18))),
#                  np.ones(32), np.ones(32), np.ones(32),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
#                  np.concatenate((np.zeros(25), np.ones(4), np.zeros(3)))]).astype(np.bool)
#         pressure[~mask] = 0.0
#         pressure = np.clip(pressure, 0.0, 1.0)
# =============================================================================
        pressure = np.expand_dims(pressure, axis=0)
        pressure = torch.from_numpy(pressure)
        object_id = torch.LongTensor([int(self.collated_labels[idx])])
        return pressure, object_id


    def collate_data(self):
        """
        Function to collate the training or test data into blocks that are
        sized corresponding to the number of used input frames
        e.g. if 4 input frames are used, one block has the shape (4,32,32)

        Returns
        -------
        None.

        """
        self.collated_data = self.data
        self.collated_labels = self.labels
        self.collated_data,\
            self.collated_labels = shuffle(self.collated_data,
                                           self.collated_labels)

        
    def balance_data(self):
        # Randomize for every refresh call
        seed = random.randint(0,1000)
        if self.split == 'train':
            # # Randomize for every refresh call
            # neighbors = random.randint(2,4)
            # clusters = random.randint(16,24)
    
            # oversampler = over_sampling.KMeansSMOTE(random_state=seed,
            #                                         kmeans_estimator=clusters,
            #                                         k_neighbors=neighbors,
            #                                         sampling_strategy='not majority')
            # # oversampler = over_sampling.SVMSMOTE(random_state=seed, out_step=step,
            # #                                      k_neighbors=5, m_neighbors=10)
            
            # # Undersample majority class to the second largest class
            # # class 0 is always the biggest class (empty hand)
            # nmax = 0
            # for i in range(1, self.nclasses):
            #     mask = self.original_labels == i
            #     n = np.count_nonzero(mask)
            #     if n > nmax:
            #         nmax = n
            # strat = {0: nmax}
            # undersampler = under_sampling.RandomUnderSampler(random_state=seed,
            #                                                   sampling_strategy=strat)
            # sampler = combine.SMOTETomek(random_state=seed, kmeans_estimator=clusters,
            #                               k_neighbors=neighbors)
            
            undersampler = under_sampling.RandomUnderSampler(random_state=seed,
                                                             sampling_strategy='not minority')
            
            resampled_data,\
                resampled_labels = undersampler.fit_resample(self.original_data,
                                                             self.original_labels)
            # try:
            #     resampled_data,\
            #         resampled_labels = oversampler.fit_resample(resampled_data,
            #                                                     resampled_labels)
            # except:
            #     neighbors = 2
            #     clusters = 24
            #     oversampler = over_sampling.KMeansSMOTE(random_state=seed,
            #                                             kmeans_estimator=clusters,
            #                                             k_neighbors=neighbors,
            #                                             sampling_strategy='not majority')
            #     resampled_data,\
            #         resampled_labels = oversampler.fit_resample(resampled_data,
            #                                                     resampled_labels)
                
        elif self.split == 'test':
            resampled_data = self.original_data
            resampled_labels = self.original_labels
                
        self.data = resampled_data
        self.labels = resampled_labels
 
    
    def refresh(self):
        print('Refreshing dataset...')
        if self.balance:
            self.balance_data()
        self.collate_data()
