import os
import time
import numpy as np
from tqdm import tqdm

#from dataset import KineticsDataset, VideoDataset1M, DataLoader, Create_Kinetics400_Dataset
#from dataset_ucf101 import Create_UCF101_Dataset
from dataset_kinetics400 import Create_Kinetics400_Dataset
from network import R2Plus1DClassifier
from eval_callback2 import EvalCallBack
#from utils import AverageMeter

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig
from mindspore.train.callback import LearningRateScheduler, TimeMonitor,LossMonitor

args = {}
args['device_target'] = "Ascend"
args['is_distributed'] = bool(os.getenv('IS_DISTRIBUTED', False))
args['device_num'] = 1
args['device_id'] = '0'

device_id = int(os.getenv('DEVICE_ID', args['device_id']))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=args['device_target'], save_graphs=False, device_id=device_id)
# context.PYNATIVE_MODE

args['rank'] = 0
args['group_size'] = 1

if(args['is_distributed']):
    if args['device_target'] == "Ascend":
        init()
    else:
        init("nccl")
    
    args['rank'] = get_rank()
    args['group_size'] = get_group_size()
    device_num = args['group_size']
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
else:
    pass

args['datapath'] = '/opt/npu/data/kinetics400_data_deal/'
#args['datapath'] = '/data/nl/UCF-101/'
args['savepath'] = 'r2plus1d.model'
args['weight_save_path'] = 'save_model/'
args['batch_size_train'] = 10 # 原文作者用的10

args['ckpt_interval'] = 10000
args['lr'] = 0.001 # 原文作者使用的0.01
args['epoch'] = 45

args['is_modelarts'] = bool(os.getenv('IS_MODELARTS', False))

localtime = time.localtime(time.time())
timestamp = str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
if(args['is_modelarts']):
    import moxing as mox
    import argparse

    parser = argparse.ArgumentParser('mindspore stgcn training')
    # Path for data and checkpoint
    parser.add_argument('--data_url', type=str, help='数据集路径.')
    parser.add_argument('--train_url', type=str, help='训练输出存储位置.')
    args, _ = parser.parse_known_args()

    local_data_url = '/cache/data'
    local_train_url = '/cache/train'

    args['datapath'] = local_data_url
    args['weight_save_path'] = local_train_url + '/save_model_' + timestamp + '/'

    mox.file.make_dirs(local_train_url)
    mox.file.make_dirs(args['weight_save_path'])
    mox.file.copy_parallel(args.data_url, local_data_url)

    pass

global epoch_current
epoch_current = 0

def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]

def step_decay(lr, cur_step_num):
    step_per_lr = int(234549 / args['batch_size_train'] / args['device_num']) * 10
    #initial_lr = args['lr']
    #xx = epoch // 10
    #lr = initial_lr
    #for i in range(0, xx):
    #    lr = lr / 10

    if(cur_step_num % step_per_lr == 0):
        lr = lr / 10
    
    return lr

def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds, dataset_sink_mode=False)
    return res[metrics_name]

def train_model_ms(num_classes, directory, batch_size = 10, layer_sizes=[3, 4, 6, 3], num_epochs=45, save=True, path="model_data.pth.tar"):
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the MindSpore tutorial found here: https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/index.html

        Args:
            num_classes (int): Number of classes in the data
            directory (str): Directory where the data is to be loaded from
            layer_sizes (list, optional): Number of blocks in each layer. Defaults to [3, 4, 6, 3], equivalent to ResNet18.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """

    # initalize the ResNet 18 version of this model
    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes)

    #criterion = ms.ops.SigmoidCrossEntropyWithLogits()

    #lr = 0.01
    #optimizer = ms.nn.SGD(get_param_groups(model), learning_rate=args['lr'], weight_decay=0.0)
    #optimizer = ms.nn.Adam(get_param_groups(model), learning_rate=args['lr'], beta1=0.9, beta2=0.999, eps=1e-08, use_locking=False, use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
    optimizer = ms.nn.Adam(model.trainable_params(), learning_rate=args['lr'], beta1=0.9, beta2=0.999, eps=1e-08, use_locking=False, use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
    

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    # check if there was a previously saved checkpoint
    if os.path.exists(path):
        # loads the checkpoint
        #checkpoint = torch.load(path)
        param_dict = load_checkpoint(path)
        #param_dict_new = {}
        #for key, values in param_dict.items():
        #    if key.startswith('moments.'):
        #        continue
        #    elif key.startswith('r2plus1d_network.'):
        #        param_dict_new[key[13:]] = values
        #    else:
        #        param_dict_new[key] = values
        #load_param_into_net(model, param_dict_new)
        load_param_into_net(model, param_dict)
        print("Reloading from previously saved checkpoint")
    else:
        print('Can not find exist check point file, this run will start with begining. ')
    
    
    # Check Point Setting
    ckpt_max_num = num_epochs
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args['ckpt_interval'],
                                       keep_checkpoint_max=ckpt_max_num)
    save_ckpt_path = os.path.join(args['weight_save_path'], 'ckpt_' + str(args['rank']) + '/')
    ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(args['rank']))
    cb_params = _InternalCallbackParam()
    cb_params.train_network = model
    cb_params.epoch_num = ckpt_max_num
    cb_params.cur_epoch_num = 1
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    # Callbacks
    time_cb = TimeMonitor(data_size=1)
    loss_cb = LossMonitor(per_print_times=1)


    
    #loss_meter = AverageMeter('loss')
    #epoch_current = 0
    #epoch_last = 0
    #step = 0
    #t_end = time.time()


    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    #from crossentropy import CrossEntropy
    #loss = CrossEntropy(num_classes=101)
    train_model = Model(model, loss_fn = loss, optimizer=optimizer, metrics={'acc', 'loss'})

    callbacks = [time_cb, loss_cb, LearningRateScheduler(step_decay), ckpt_cb]

    #if(args['rank'] == 0):
    # EvalCallBack
    #dataset_eval = Create_UCF101_Dataset(args['datapath'], 'datalist/ucf101/', mode = 'test', batch_size = batch_size, shuffle=True)
    dataset_eval = Create_Kinetics400_Dataset(args['datapath'], 'datalist/kinetics400/', mode = 'test', batch_size = batch_size, shuffle=False)
    eval_param_dict = {"model": train_model, "dataset": dataset_eval, "metrics_name": "acc"}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=5,
                           eval_start_epoch=1, save_best_ckpt=True,
                           ckpt_directory=save_ckpt_path, besk_ckpt_name="best_acc.ckpt",
                           metrics_name="acc")
    callbacks.append(eval_cb)
    


    #dataset = Create_Kinetics400_Dataset(args['datapath'], mode = 'train', batch_size = batch_size, shuffle=True)
    #dataset = Create_UCF101_Dataset(args['datapath'], 'datalist/ucf101/', mode = 'train', batch_size = batch_size, shuffle=True, device_num = args['group_size'], is_distributed = True, rank = args['rank'])
    dataset = Create_Kinetics400_Dataset(args['datapath'], 'datalist/kinetics400/', mode = 'train', batch_size = batch_size, shuffle=True, device_num = args['group_size'], is_distributed = True, rank = args['rank'])
    train_model.train(num_epochs, dataset, callbacks=callbacks, dataset_sink_mode=False)

    if(args['rank'] == 0):
        train_model.eval(dataset, dataset_sink_mode=False)
        train_model.eval(dataset_eval, dataset_sink_mode=False)
    


    #ckpt_cb.end(run_context)
    

    # 模型训练结束
    

    '''
    # 直接开始测试

    from eval import ModelEval
    from predict import R2Plus1D_Model

    eval_model_mem = ModelEval(101)
    eval_model_mem.r2plus1d_model = R2Plus1D_Model(101, [2,2,2,2], 'Ascend', device_id = '7')
    eval_model_mem.r2plus1d_model.model = model
    data_path, datapath_list = '/opt/npu/pvc/ucf101/UCF-101/', 'datalist/ucf101/'
    eval_model_mem.SetDataset(data_path, datapath_list, data_mode='train')
    eval_model_mem.Eval()



    eval_model = ModelEval(101)
    eval_model.LoadCheckPoint('/home/XidianUniversity/nl/R2Plus1D-MindSpore/save_model/ckpt_0/0-45_953.ckpt')
    data_path, datapath_list = '/opt/npu/pvc/ucf101/UCF-101/', 'datalist/ucf101/'
    eval_model.SetDataset(data_path, datapath_list, data_mode='train')
    eval_model.Eval()
    '''
    
    pass






if(__name__ == '__main__'):
    num_classes = 400
    datapath = args['datapath']
    savepath = args['savepath']
    train_model_ms(num_classes, datapath, args['batch_size_train'], num_epochs = args['epoch'], save = True, path = savepath)

    if(args['is_modelarts']):
        import moxing as mox
        import argparse

        parser = argparse.ArgumentParser('mindspore stgcn training')
        # Path for data and checkpoint
        parser.add_argument('--data_url', type=str, help='数据集路径.')
        parser.add_argument('--train_url', type=str, help='训练输出存储位置.')
        args, _ = parser.parse_known_args()

        local_data_url = '/cache/data'
        local_train_url = '/cache/train'

        #args['datapath'] = local_data_url
        #args['weight_save_path'] = local_train_url + '/save_model' + timestamp + '/'

        mox.file.make_dirs(local_train_url)
        mox.file.copy_parallel(local_train_url, args.train_url)

        pass
