import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling


# Mask that selects only the physically present taxels
mask = np.array([np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3)))])
finger_mask = np.zeros((32,32))
finger_mask[:, 0:4] = 1
finger_mask[28:32, :] = 1
finger_mask = np.logical_and(mask, finger_mask)
mask = mask.reshape((1024,)).astype(np.bool)
finger_mask = finger_mask.reshape((1024,)).astype(np.bool)
palm_only = False


def load_data(filename, kfold=3, seed=333, undersample=True, split='random'):

    data = sio.loadmat(filename, squeeze_me=True)
    # Use only frames in which objects were touched
    valid_mask = data['valid_flag'] == 1
    # Use threshold on the fingertips to find frames in which the object was
    # only touched with the palm
    if(palm_only):
        finger_threshold = data['threshold'][finger_mask]
        for i, press in enumerate(data['tactile_data']):
            if(np.any(press[finger_mask] > finger_threshold)):
                valid_mask[i] = False
    pressure = data['tactile_data'][valid_mask]
    # Scale data to the range [0, 1]
    pressure = np.clip((pressure.astype(np.float32)-1500)/(2700-1500), 0.0, 1.0)
    #pressure = normalize(pressure.astype(np.float32))
    #pressure = np.exp2(pressure)
    #pressure = np.clip((pressure - 1), 0.0, 1.0)
    #pressure = boost(pressure)
    #pressure = np.clip(pressure, 0.0, 1.0)
    object_id = data['object_id'][valid_mask]
    # Set taxels that are not physically present to zero
    #pressure[:, ~mask] = 0.0
                                                             
    if kfold is not None:
        # Decrease the test size if cross validation is used
        test_size = 0.15
    else:
        kfold = 3
        test_size = 0.33

    if(split == 'random'):
        if(undersample):
            us = under_sampling.RandomUnderSampler(random_state=seed,
                                                   sampling_strategy='not minority')
            us_pressure, us_object_id = us.fit_resample(pressure, object_id)
            
            pressure, object_id = us_pressure, us_object_id
    
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
                              random_state=seed)
        # train_ind, val_ind = skf.split(train_data, train_labels)
        # skf_gen = skf.split(train_data, train_labels)
        
        return train_data, train_labels, test_data, test_labels, skf
    
    elif(split == 'session'):
        num_sessions = len(np.unique(data['session_id']))
        x = []
        y = []
        valid_sessions = data['session_id'][valid_mask]
        for i in range(num_sessions):
            session_mask = valid_sessions == i
            x.append(pressure[session_mask])
            y.append(object_id[session_mask])
            
        return x, y 
        

def normalize(pressure):
    normalized_p = np.copy(pressure)
    for i, press in enumerate(pressure):
        min_p = np.min(press)
        normalized_p[i] = (press - min_p) / np.max(press - min_p)
    
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


def boost(pressure):
    boosted_p = np.copy(pressure)
    for i, press in enumerate(boosted_p):
        mean_p = np.mean(press[mask])
        boost_mask = press > mean_p
        press[boost_mask] = list(map(lambda x: 4*(x-mean_p), press[boost_mask]))
    
    return boosted_p


def mix_real_MIT(kfold=3, seed=333, undersample=True, split='random'):
    realData_path = ('/scratch1/msc20f10/data/3kOhm_FB/'
                     + 'data_MT_FabianGeiger_5sess.mat')
    MITdata_path = ('/scratch1/msc20f10/data/classification/metadata.mat')
    
    # These two lists will contain valid data split into recording sessions
    x = []
    y = []
    
    realData = sio.loadmat(realData_path, squeeze_me=True)
    real_valid_mask = realData['valid_flag'] == 1
    # Scale all data to the range [0, 1]
    realPressure = realData['tactile_data'][real_valid_mask].astype(np.float32)
    realPressure = np.clip((realPressure - 1500)/(2700 - 1500), 0.0, 1.0)
    #realPressure = normalize(realPressure.astype(np.float32))
    realObject_id = realData['object_id'][real_valid_mask]
    realSession_id = realData['session_id'][real_valid_mask]
    # Split into sessions
    num_sessions = len(np.unique(realSession_id))
    for i in range(num_sessions):
        session_mask = realSession_id == i
        x.append(realPressure[session_mask])
        y.append(realObject_id[session_mask])

    MITdata = sio.loadmat(MITdata_path, squeeze_me=True)
    MIT_valid_mask = MITdata['hasValidLabel'] == 1
    MITpressure = MITdata['pressure'].reshape((-1, 32*32)).astype(np.float32)
    MITpressure = MITpressure[MIT_valid_mask]
    MITpressure = np.clip((MITpressure - 500)/(650 - 500), 0.0, 1.0)
    #MITpressure = normalize(MITpressure.astype(np.float32))
    MITobject_id = MITdata['objectId'][MIT_valid_mask]    
    # Only use the same objects as in the real data set
    MITobjects = list(MITdata['objects'])
    used_objects = list(map(str.strip, realData['objects']))
    # Each class has three recording IDs that correspond to the different
    # experiment days. There are 81 recording IDs (3*27)
    # 0  - 26 belong to the first recording
    # 27 - 53 belong to the second recording
    # 54 - 81 belong to the third recording
    MITrecording_id = MITdata['recordingId'][MIT_valid_mask]
    for i in range(3):
        # Find valid samples from the different recording days
        recording_mask = np.logical_and(MITrecording_id >= i*27,
                                        MITrecording_id < (i+1)*27)
        
        used_pressure = []
        used_object_id = []
        for i, obj in enumerate(used_objects):
            idx = MITobjects.index(obj)
            used_mask = np.logical_and(MITobject_id == idx, recording_mask)
            used_pressure.append(MITpressure[used_mask])
            # Assign same class to equal objects
            used_object_id.append(np.full(len(MITobject_id[used_mask]), i))
            
        x.append(np.concatenate(used_pressure))
        y.append(np.concatenate(used_object_id))
    
    
    if kfold is not None:
        # Decrease the test size if cross validation is used
        test_size = 0.15
    else:
        kfold = 3
        test_size = 0.33

    if(split == 'random'):
        pressure = np.concatenate(x)
        object_id = np.concatenate(y)
        if(undersample):
            us = under_sampling.RandomUnderSampler(random_state=seed,
                                               sampling_strategy='not minority')
            us_pressure, us_object_id = us.fit_resample(pressure, object_id)
            
            pressure, object_id = us_pressure, us_object_id
    
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
                              random_state=seed)
        # train_ind, val_ind = skf.split(train_data, train_labels)
        # skf_gen = skf.split(train_data, train_labels)
        
        return train_data, train_labels, test_data, test_labels, skf
    
    elif(split == 'session'):
        return x, y
    