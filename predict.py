import os
import cv2
import numpy as np

import mindspore as ms
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor

from network import R2Plus1DClassifier

class R2Plus1D_Model():
    def __init__(self, num_classes, layer_sizes=[2, 2, 2, 2], device_target='Ascend', device_id = '0'):
        self.num_classes = num_classes
        self.layer_sizes = layer_sizes

        device_id = int(os.getenv('DEVICE_ID', device_id))
        context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, device_target=device_target, save_graphs=False, device_id=device_id)

        # the following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128  
        self.resize_width = 171
        self.crop_size = 112

        self.model = None
        
        pass

    def LoadCheckPoint(self, filename):
        self.model = R2Plus1DClassifier(num_classes=self.num_classes, layer_sizes=self.layer_sizes)
        self.model.set_train(mode=False)
        if os.path.exists(filename):
            # loads the checkpoint
            param_dict = load_checkpoint(filename)
            param_dict_new = {}
            for key, values in param_dict.items():
                if key.startswith('moments.'):
                    continue
                elif key.startswith('r2plus1d_network.'):
                    param_dict_new[key[13:]] = values
                else:
                    param_dict_new[key] = values
            load_param_into_net(self.model, param_dict_new)
            print("[Info] Reloading from previously saved checkpoint")
        else:
            print('[Warning] Can not find exist check point file, this predict will run with initial weight params. ')
        pass

    def LoadCheckPointNew(self, filename):
        self.model = R2Plus1DClassifier(num_classes=self.num_classes, layer_sizes=self.layer_sizes)
        self.model.set_train(mode=False)
        if os.path.exists(filename):
            # loads the checkpoint
            param_dict = load_checkpoint(filename)
            
            load_param_into_net(self.model, param_dict)
            print("[Info] Reloading from previously saved checkpoint")
        else:
            print('[Warning] Can not find exist check point file, this predict will run with initial weight params. ')
        pass
    
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

    def crop(self, buffer, time_start, clip_len, crop_size):
        # select time index for temporal jittering
        time_index = time_start
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

    def Forward(self, x):
        x_in = Tensor(x, dtype=ms.float32)
        y_out = self.model(x_in)
        y = y_out.asnumpy()
        return y

    def Predict(self, x, clip_len = 16):
        batch_size = x.shape[1] // clip_len
        x_in = np.zeros((batch_size, 3, clip_len, self.crop_size, self.crop_size))
        for i in range(0, batch_size):
            x_tmp = self.crop(x, 16 * i, clip_len, self.crop_size)
            x_tmp = self.normalize(x_tmp)
            x_in[i] = x_tmp
        
        y_out = self.Forward(x_in)

        label_predict = np.argmax(y_out, axis=1)
        return label_predict

    def PredictFromFile(self, filename):
        data = self.loadvideo(filename)
        y = self.Predict(data)
        return list(y)

if(__name__ == '__main__'):
    num_class = 101
    model = R2Plus1D_Model(num_class)
    model.LoadCheckPoint('/home/XidianUniversity/nl/R2Plus1D-MindSpore/save_model/ckpt_0/0-7_224.ckpt')
    y = model.PredictFromFile('/data/XDU/UCF101_hmdb51/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')

    data_path, datapath_list = '/data/XDU/UCF101_hmdb51/UCF-101/', 'datalist/ucf101/'
    from dataset_ucf101 import UCF101Dataset
    dataset = UCF101Dataset(data_path, datapath_list)
    r = list()
    for pre in y:
        r.append(dataset.index_to_class[pre+1])
    print('Predict class label: ', r)
