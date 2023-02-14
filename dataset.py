import os
from pathlib import Path
import cv2
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset as ms_dataset
from mindspore.dataset import SequentialSampler

from distributed_sampler import DistributedSampler

class KineticsDataset():
    '''
    自己实现的MindSpore用的数据集类

    A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list 
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 16. 
    '''

    def __init__(self, directory, mode='train', clip_len=16):
        folder = Path(directory)/mode  # get the directory of the specified split

        self.clip_len = clip_len

        # the following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128  
        self.resize_width = 171
        self.crop_size = 112

        # obtain all the filenames of files inside all the class folders 
        # going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)     

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int) 
        #super().__init__()

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideo(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    
        
    def loadvideo(self, fname):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('[TestDebug] ', fname, frame_count)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True

        # read in each frame, one at a time into the numpy buffer array
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            # 如果frame是None的话，先跳过
            if(type(frame) == type(None)):
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # will resize frames if not already final size
            # NOTE: strongly recommended to resize them during the download process. This script
            # will process videos of any size, but will take longer the larger the video file.
            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[count] = frame
            count += 1

        # release the VideoCapture once it is no longer needed
        capture.release()

        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        buffer = buffer.transpose((3, 0, 1, 2))

        return buffer 
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[1] - clip_len)
        # randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[2] - crop_size)
        width_index = np.random.randint(buffer.shape[3] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[:, time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size]

        return buffer                

    def normalize(self, buffer):
        # Normalize the buffer
        # NOTE: Default values of RGB images normalization are used, as precomputed 
        # mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel 
        # free to push to and edit this section to replace them if found. 
        buffer = (buffer - 128)/128
        return buffer

    def __len__(self):
        return len(self.fnames)




class VideoDataset1M(KineticsDataset):
    r"""Dataset that implements VideoDataset, and produces exactly 1M augmented
    training samples every epoch.
        
        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 8. 
        """
    def __init__(self, directory, mode='train', clip_len=8):
        # Initialize instance of original dataset class
        super(VideoDataset1M, self).__init__(directory, mode, clip_len)

    def __getitem__(self, index):
        # if we are to have 1M samples on every pass, we need to shuffle
        # the index to a number in the original range, or else we'll get an 
        # index error. This is a legitimate operation, as even with the same 
        # index being used multiple times, it'll be randomly cropped, and
        # be temporally jitterred differently on each pass, properly
        # augmenting the data. 
        index = np.random.randint(len(self.fnames))

        buffer = self.loadvideo(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    

    def __len__(self):
        return 1000000  # manually set the length to 1 million


class DataLoader():
    def __init__(self, datas, batch_size, is_shuffle = True, num_workers = 1):
        self.dataset = datas
        self.batch_size = batch_size
        #self.count = len(datas.frames)
        self.count = datas.label_array.shape[0]
        self.is_shuffle = is_shuffle
        pass

    def GetBatchGenerator(self):
        data_sample = self.dataset.__getitem__(0)
        data_inputs = np.zeros((self.batch_size, ) + data_sample[0].shape)
        data_labels = np.zeros((self.batch_size, ) + data_sample[1].shape)
        num = 0
        epoch = 0
        while(True):
            if(num >= self.count):
                epoch += 1
                self.__do_shuffle__()
            num = num % self.count

            for i in range(0, self.batch_size):
                data_item = self.dataset.__getitem__(num % self.count)
                data_inputs[i] = data_item[0]
                data_labels[i] = data_item[1]
                num += 1
            yield epoch, data_inputs, data_labels
        pass
    
    def __do_shuffle__(self):
        
        pass

def Create_Kinetics400_Dataset(data_path, mode='train', batch_size = 16, shuffle = True, device_num = 1, is_distributed = False, rank = 0):

    dataset = KineticsDataset(data_path)
    current_sampler = None
    if(is_distributed):
        distributed_sampler = DistributedSampler(len(dataset), device_num, rank, shuffle=shuffle)
        current_sampler = distributed_sampler
    else:
        current_sampler = DistributedSampler(len(dataset), device_num, rank, shuffle=shuffle)
        #current_sampler = SequentialSampler(start_index=0, num_samples=len(dataset))
    #hwc_to_chw = CV.HWC2CHW()
    op_none = lambda x: x

    data_set = ds.GeneratorDataset(dataset, column_names=["frames","label"], shuffle=True,sampler=current_sampler)
    data_set = data_set.map(input_columns=["frames"], operations=op_none, num_parallel_workers=8)
    data_set = data_set.map(input_columns=["label"], operations=op_none, num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set