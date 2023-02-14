import os
from pathlib import Path
import cv2
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset as ms_dataset
from mindspore.dataset import SequentialSampler

from distributed_sampler import DistributedSampler

class UCF101Dataset():
    '''
    自己实现的MindSpore用的数据集类

        Args:
            directory_dataset (str): The path to the directory containing the datasets
            directory_datalist (str): The path to the directory containing the train/test datalists
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 16. 
    '''

    def __init__(self, directory_dataset, directory_datalist, mode='train', clip_len=16, keep_in_memory = False):
        self.directory_dataset = directory_dataset
        self.directory_datalist = directory_datalist
        self.mode = mode
        self.clip_len = clip_len
        self.keep_in_memory = keep_in_memory

        self.index_to_class = dict() # index to label text
        self.class_to_index = dict() # label text to index
        self.datalist = list() # dataset list
        self.data_in_memory = dict()
        #self.data_in_memory = list()
        #self.index_dic = dict()

        # obtain all the filenames of files inside all the class folders 
        # going through each class folder one at a time
        self.fnames, self.labels = [], []
        self._load_classlist_(self.directory_datalist + 'classlist.txt')
        if(self.mode == 'train'):
            self._load_datalist_(self.directory_datalist + 'trainlist.txt')
            pass
        elif(self.mode == 'test'):
            self._load_datalist_(self.directory_datalist + 'testlist.txt')
            pass
        else:
            raise ValueError('There only are two modes in this dataset, which is `train` and `test`. ')

        # the following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128  
        self.resize_width = 171
        self.crop_size = 112

        
        #for label in sorted(os.listdir(folder)):
        #    for fname in os.listdir(os.path.join(folder, label)):
        #        self.fnames.append(os.path.join(folder, label, fname))
        #        labels.append(label)     

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(self.labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in self.labels], dtype=int) 
        #super().__init__()
    
    def _load_classlist_(self, filename):
        f = open(filename, 'r', encoding='utf-8')
        txt = f.read()
        f.close()
        txt_split = txt.split('\n')
        for line in txt_split:
            if(len(line)==0 or ' ' not in line):
                continue
            line_split = line.split(' ')
            tmp_index = int(line_split[0])
            tmp_class = line_split[1]
            self.index_to_class[tmp_index] = tmp_class
            self.class_to_index[tmp_class] = tmp_index

        pass

    def _load_datalist_(self, filename):
        f = open(filename, 'r', encoding='utf-8')
        txt = f.read()
        f.close()
        txt_split = txt.split('\n')
        for line in txt_split:
            if(len(line)==0 or ' ' not in line):
                continue
            line_split = line.split(' ')
            tmp_filepath = line_split[0]
            tmp_fileclass = int(line_split[1])
            self.datalist.append([self.directory_dataset + '/' + tmp_filepath, tmp_fileclass])
            self.fnames.append(self.directory_dataset + '/' + tmp_filepath)
            self.labels.append(tmp_fileclass)

        pass

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        if(self.keep_in_memory == True and index in self.data_in_memory):
            buffer = self.data_in_memory[index]
        else:
            buffer = self.loadvideo(self.fnames[index])
        
            # if keep_in_memory is true, save it in `self.data_in_memory`
            if(self.keep_in_memory == True):
                self.data_in_memory[index] = buffer
                #self.data_in_memory.append(buffer)
                #index_new = len(self.data_in_memory) - 1
                #self.index_dic[index] = index_new

        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    
        
    def loadvideo(self, fname):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #print('[TestDebug] ', fname, frame_count)
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



def Create_UCF101_Dataset(data_path, datapath_list, mode='train', batch_size = 16, shuffle = True, device_num = 1, is_distributed = False, rank = 0):

    dataset = UCF101Dataset(data_path, datapath_list, mode = mode)
    current_sampler = None
    if(is_distributed):
        distributed_sampler = DistributedSampler(len(dataset), device_num, rank, shuffle=shuffle)
        current_sampler = distributed_sampler
    else:
        current_sampler = DistributedSampler(len(dataset), device_num, rank, shuffle=shuffle)
        #current_sampler = SequentialSampler(start_index=0, num_samples=len(dataset))
    #hwc_to_chw = CV.HWC2CHW()
    op_none = lambda x: x
    op_print = lambda x: myprint(x)
    def myprint(x):
        print('输出的x：', x)
        return x

    data_set = ds.GeneratorDataset(dataset, column_names=["frames","label"], shuffle=shuffle,sampler=current_sampler, num_parallel_workers=8)
    data_set = data_set.map(input_columns=["frames"], operations=op_none, num_parallel_workers=8)
    data_set = data_set.map(input_columns=["label"], operations=op_none, num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set