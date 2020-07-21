import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling


def load_data(filename, kfold=3, seed=333, split='random', undersample=True):
    """
    Function to load the pressure data with stratified k fold

    Parameters
    ----------
    filename : string
        Path to the data file.
    augment : bool, optional
        Whether or not to add random noise to the pressure data.
        The default is False.
    kFold : int, optional
        Number of folds to use in stratified k fold. The default is 3.
    seed : int, optional
        Seed used for shuffling the train test split and stratified k fold.
        The default is None.

    Returns
    -------
    train_data : numpy array
        DESCRIPTION.
    train_labels : numpy array
        DESCRIPTION.
    train_ind : numpy array
        DESCRIPTION.
    val_ind : numpy array
        DESCRIPTION.
    test_data : numpy array
        DESCRIPTION.
    test_labels : numpy array
        DESCRIPTION.
    """

    data = sio.loadmat(filename)
    valid_idx = data['hasValidLabel'].flatten() == 1
    balanced_idx = data['isBalanced'].flatten() == 1
    # indices now gives a subset of the data set that contains only valid
    # pressure frames and the same number of frames for each class
    indices = np.logical_and(valid_idx, balanced_idx)
    pressure = np.transpose(data['pressure'], axes=(0, 2, 1))
    # Reshape the data into 1D feature vectors for sklearn utility functions
    pressure = np.reshape(pressure, (-1, 32*32))
    object_id = data['objectId'].flatten()
    
    if split == 'original':
        # pressure = normalize(pressure.astype(np.float32))
        # Prepare the data the same way as in the paper
        pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)
        # Find the samples that were used for training in the paper
        split_idx = data['splitId'].flatten() == 0
        train_indices = np.logical_and(indices, split_idx)
        pressure_train = pressure[train_indices]

        train_data = pressure_train
        train_labels = object_id[train_indices]

        # Find the samples that were used for testing in the paper
        split_idx = data['splitId'].flatten() == 1
        test_indices = np.logical_and(indices, split_idx)
        pressure_test = pressure[test_indices]

        test_data = pressure_test
        test_labels = object_id[test_indices]

        # # Add the rest of the valid data to the test set
        # unbalanced_idx = np.logical_xor(valid_idx, balanced_idx)
        # rest_pressure = pressure[unbalanced_idx]
        # rest_object_id = object_id[unbalanced_idx]
        # test_data = np.append(test_data, rest_pressure, axis=0)
        # test_labels = np.append(test_labels, rest_object_id, axis=0)
        
        #_____________________________________________________________________#
        # # Just to test if the accuracy in the test set itself stays high
        # train_data, test_data,\
        #     train_labels, test_labels = train_test_split(pressure_train,
        #                                                  train_labels,
        #                                                  test_size=0.2,
        #                                                  random_state=seed,
        #                                                  shuffle=True,
        #                                                  stratify=train_labels)
        #_____________________________________________________________________#

        return train_data, train_labels, test_data, test_labels

    elif split == 'random':
        # Prepare the data the same way as in the paper
        pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)
        # # Add the rest of the valid data to the test set
        # unbalanced_idx = np.logical_xor(valid_idx, balanced_idx)
        # rest_pressure = pressure[unbalanced_idx]
        # rest_object_id = object_id[unbalanced_idx]

        # indices give balanced and valid data
        pressure = pressure[indices]
        object_id = object_id[indices]

        if kfold is not None:
            # Decrease the test size if cross validation is used
            test_size = 0.15
        else:
            kfold = 3
            test_size = 0.05 #0.306

        # Split the already balanced dataset in a stratified way -> training
        # and test set will still be balanced
        train_data, test_data,\
            train_labels, test_labels = train_test_split(pressure, object_id,
                                                         test_size=test_size,
                                                         random_state=seed,
                                                         shuffle=True,
                                                         stratify=object_id)
        #print(train_data.shape, train_labels.shape)
        # This generates a k fold split in a stratified way.
        # Easy way to do k fold cross validation
        skf = StratifiedKFold(n_splits=kfold, shuffle=True,
                              random_state=seed+1)
        # train_ind, val_ind = skf.split(train_data, train_labels)
        # skf_gen = skf.split(train_data, train_labels)
        
        # # Add the rest of the valid data to the test set
        # test_data = np.append(test_data, rest_pressure, axis=0)
        # test_labels = np.append(test_labels, rest_object_id, axis=0)
    
        return train_data, train_labels, test_data, test_labels, skf

    elif split == 'recording':
        # Normalize pressure data
        # pressure = normalize(pressure.astype(np.float32))
        # Prepare the data the same way as in the paper
        pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)

        # Each class has three recording IDs that correspond to the different
        # experiment days. There are 81 recording IDs (3*27)
        # 0  - 26 belong to the first recording
        # 27 - 53 belong to the second recording
        # 54 - 81 belong to the third recording
        recording_id = data['recordingId'].flatten()
        recordings = []
        for i in range(3):
            # Find valid samples from the different recording days
            recording_mask = np.logical_and(recording_id >= i*27,
                                            recording_id < (i+1)*27)
            recording_mask = np.logical_and(recording_mask, valid_idx)
            
            # The data is not yet balanced!
            recordings.append([pressure[recording_mask],
                               object_id[recording_mask]])
            
        x1, y1 = recordings[0][0], recordings[0][1]
        x2, y2 = recordings[1][0], recordings[1][1]
        x3, y3 = recordings[2][0], recordings[2][1]
        
        # # Normalize each recording pixelwise
        # x1 = normalize_per_pixel(x1.reshape((x1.astype(np.float32))
        # x2 = normalize_per_pixel(x2.reshape((x2.astype(np.float32))
        # x3 = normalize_per_pixel(x3.reshape((x3.astype(np.float32))
        
        # # Normalize each recording
        # x1 = normalize_per_session(x1.astype(np.float32))
        # x2 = normalize_per_session(x2.astype(np.float32))
        # x3 = normalize_per_session(x3.astype(np.float32))
        
        if undersample:
            # Balance data using the python package 'imbalanced-learn'
            # Random undersampling.
            undersampler = under_sampling.RandomUnderSampler(random_state=seed+2,
                                                              sampling_strategy='not minority')
            x1_resampled, y1_resampled = undersampler.fit_resample(x1, y1)
            x2_resampled, y2_resampled = undersampler.fit_resample(x2, y2)
            x3_resampled, y3_resampled = undersampler.fit_resample(x3, y3)
    
            return x1_resampled, y1_resampled, x2_resampled, y2_resampled,\
                x3_resampled, y3_resampled
        else:
            return x1, y1, x2, y2, x3, y3
        

def normalize(pressure):
    normalized_p = np.copy(pressure)
    for i in range(pressure.shape[0]):
        min_p = np.min(pressure[i])
        normalized_p[i] = (pressure[i] - min_p) / np.max(pressure[i] - min_p)
    
    return normalized_p


def normalize_per_pixel(pressure):
    normalized_p = np.copy(pressure)
    # First scale values to [0, 1]
    min_p = np.min(pressure)
    normalized_p = (pressure - min_p) / np.max(pressure - min_p)
    # Then subtract the mean for each pixel
    pixel_mean = np.mean(normalized_p, axis=0)
    # pixel_mean should be shaped like normalized_p
    normalized_p = normalized_p - pixel_mean
    
    return normalized_p


def normalize_per_session(pressure):
    normalized_p = np.copy(pressure)
    # First scale values to [0, 1]
    min_p = np.min(pressure)
    normalized_p = (pressure - min_p) / np.max(pressure - min_p)
    # Then subtract the mean for the whole session
    mean_p = np.mean(normalized_p)
    normalized_p = normalized_p - mean_p
    
    return normalized_p
    