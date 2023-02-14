import numpy as np
import time

from predict import R2Plus1D_Model
from dataset_ucf101 import UCF101Dataset

class ModelEval():
    def __init__(self, num_classes, layer_sizes=[2, 2, 2, 2], device_target='Ascend', device_id = '0'):
        self.r2plus1d_model = R2Plus1D_Model(num_classes, layer_sizes, device_target, device_id)
        self.dataset = None
        self.data_mode = None
        pass

    def SetDataset(self, data_path, datapath_list, data_mode='test'):
        self.data_mode = data_mode
        self.dataset = UCF101Dataset(data_path, datapath_list, data_mode)
        pass

    def LoadCheckPoint(self, filename):
        self.r2plus1d_model.LoadCheckPoint(filename)
        pass

    def Eval(self, save_report = True):
        lst_result = list()
        tmp_result = 'number\tlabel number\tlabel\tpredict number\tpredict\tfilename\n\n'
        lst_result.append(tmp_result)
        localtime = time.localtime(time.time())

        data_count = len(self.dataset)
        if(data_count == 0):
            raise Exception("The data count in this dataset is " + str(data_count))
        
        correct_count = 0
        for i in range(0, data_count):
            #if(i % 100 == 0):
            print('[Eval Info]', i, '/', data_count, 'correct:', correct_count, end = '\r', flush = True)
            
            data_in, data_label = self.dataset[i]
            x_in = np.zeros((1,) + data_in.shape)
            x_in[0] = data_in
            y_out = self.r2plus1d_model.Forward(x_in)
            y_pre = np.argmax(y_out, axis=1)
            label_predict = y_pre[0]

            if(label_predict == data_label):
                correct_count += 1

            tmp_result = str(i) + '\t' + str(data_label + 1) + '\t' + self.dataset.index_to_class[data_label + 1] + '\t' + str(label_predict + 1) + '\t' + self.dataset.index_to_class[label_predict + 1] + '\t' + self.dataset.fnames[i]
            lst_result.append(tmp_result)

        print('[Eval Info]', i, '/', data_count, 'correct:', correct_count)

        acc = correct_count / data_count * 100
        tmp_result = 'Accuracy: ' + str(acc) + ' %. '
        print('[Info] R(2+1)D Model Eval Accuracy:', str(acc) , '%.')
        lst_result.append('')
        lst_result.append(tmp_result)

        if(save_report == True):
            txt = '\n'.join(lst_result)
            filename = 'eval_report_' + self.data_mode + '_' + str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec) + '.txt'

            f = open(filename, 'w', encoding='utf-8')
            f.write(txt)
            f.close()

        pass

if(__name__ == '__main__'):
    eval_model = ModelEval(101)
    eval_model.LoadCheckPoint('/home/XidianUniversity/nl/R2Plus1D-MindSpore/save_model/ckpt_0/0-14_252.ckpt')
    data_path, datapath_list = '/data/XDU/UCF101_hmdb51/UCF-101/', 'datalist/ucf101/'
    eval_model.SetDataset(data_path, datapath_list, data_mode='test')
    eval_model.Eval()


