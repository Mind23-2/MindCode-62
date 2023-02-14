
import cv2
import numpy as np
import os

_resize_height = 112
_resize_width = 112

filename_datalist = 'datalist/kinetics400/trainlist.txt'
path_origin_dataset = '/data/nl/kinetics400_data/'
path_deal_dataset = '/opt/npu/data/kinetics400_data_deal/'

def LoadVideoFromFileName(filename):
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(filename)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #print('[TestDebug] ', fname, frame_count)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
    buffer = np.empty((frame_count, _resize_height, _resize_width, 3), np.dtype('float32'))

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
        if (frame_height != _resize_height) or (frame_width != _resize_width):
            frame = cv2.resize(frame, (_resize_width, _resize_height))
        buffer[count] = frame
        count += 1

    # release the VideoCapture once it is no longer needed
    capture.release()

    # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    #buffer = buffer.transpose((3, 0, 1, 2))

    return buffer 

def SaveImage(filename, img):
    f = open(filename, 'wb')
    for i in range(0, img.shape[0]):
        f.write(img[i])
    f.close()

def SaveImageAsNumpy(filename, imgArray):
    if(filename == None or filename == ""):
        filename = "filename.npy"
    np.save(filename, imgArray)

def JpgEncode(imgArray):
    return cv2.imencode(".jpg", imgArray)

def ConvertDataset(loadpath_datalist, loadpath_origin_dataset, savepath_deal_dataset):
    if(os.path.exists(savepath_deal_dataset) == False):
        os.makedirs(savepath_deal_dataset)
    
    # load datalist
    f = open(loadpath_datalist, 'r')
    txt = f.read()
    f.close()
    txt_split = txt.split('\n')

    count = 0
    for line in txt_split:
        line_split = line.split(' ')
        filepath_data = line_split[0]
        filepath_split = filepath_data.split('/')

        for i in range(0, len(filepath_split)):
            tmppath = savepath_deal_dataset + '/' + '/'.join(filepath_split[0 : i + 1])
            if(os.path.exists(tmppath) == False):
                os.makedirs(tmppath)
        
        filename = loadpath_origin_dataset + '/' + filepath_data
        output_path = savepath_deal_dataset + '/' + filepath_data + '/'
        videodata = LoadVideoFromFileName(filename)

        f = open(output_path + 'length.txt', 'w')
        txt = f.write(str(videodata.shape[0]))
        f.close()

        for i in range(0, videodata.shape[0]):
            tmp = JpgEncode(videodata[i])
            SaveImageAsNumpy(output_path + str(i) + '.npy', tmp[1])
        
        count += 1
        print('[进度]', count, ':', filename)
       
    pass



def __TestVideoToJpgsNumpys():
    filename = '/data/nl/kinetics400_data/train_256/abseiling/0347ZoDXyP0_000095_000105.mp4'
    output_path = 'test_out'
    videodata = LoadVideoFromFileName(filename)
    for i in range(0, videodata.shape[0]):
        tmp = JpgEncode(videodata[i])
        SaveImage(output_path + '/out_' + str(i) + '.jpg', tmp[1])
        SaveImageAsNumpy(output_path + '/out_' + str(i) + '.npy', tmp[1])
    pass

if(__name__ == '__main__'):
    ConvertDataset(filename_datalist, path_origin_dataset, path_deal_dataset)
    pass