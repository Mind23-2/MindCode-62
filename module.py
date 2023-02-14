'''
用3D卷积实现了一个时空卷积模块
'''

import math

from myutils import _triple

import mindspore as ms
import mindspore.nn as nn

#from mindspore.ops.operations.nn_ops import Conv3D
from mindspore.nn import Conv3d
from mindspore.ops import Shape, Reshape
#from mindspore.nn.layer.normalization import BatchNorm3d

class BatchNorm3d(nn.Cell):
    def __init__(self, num_features):
        super().__init__()
        self.reshape = Reshape()
        self.shape = Shape()
        self.bn2d = nn.BatchNorm2d(num_features, momentum=0.98, data_format="NCHW", use_batch_statistics=True)

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4]))
        bn2d_out = self.bn2d(x)
        bn3d_out = self.reshape(bn2d_out, x_shape)
        return bn3d_out

class SpatioTemporalConv(nn.Cell):
    '''
    使用华为MindSpore复现的SpatioTemporalConv单元
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        #spatial_padding =  [0, padding[1], padding[2]]
        spatial_padding =  [0, 0, padding[1], padding[1], padding[2], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        #temporal_padding =  [padding[0], 0, 0]
        temporal_padding =  [padding[0], padding[0], 0, 0, 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        
        self.spatial_conv =  Conv3d(in_channels = in_channels, out_channels = intermed_channels, kernel_size = tuple(spatial_kernel_size), pad_mode='pad', padding=tuple(spatial_padding), stride=tuple(spatial_stride), weight_init='he_normal', has_bias=True, data_format="NCDHW")
        self.bn = BatchNorm3d(num_features=intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase

        self.temporal_conv = Conv3d(in_channels = intermed_channels, out_channels = out_channels, kernel_size = tuple(temporal_kernel_size), pad_mode='pad', padding=tuple(temporal_padding), stride=tuple(temporal_stride), weight_init='he_normal', has_bias=True, data_format="NCDHW")
        
        pass
    def construct(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x
        
    pass

class R2Plus1dStem(nn.Cell):
    '''
    使用华为MindSpore复现的R2Plus1dStem单元
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(R2Plus1dStem, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        #spatial_padding =  [0, padding[1], padding[2]]
        spatial_padding =  [0, 0, padding[1], padding[1], padding[2], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        #temporal_padding =  [padding[0], 0, 0]
        temporal_padding =  [padding[0], padding[0], 0, 0, 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        #intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
        #                    (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        intermed_channels = 45
        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU

        self.spatial_conv =  Conv3d(in_channels = in_channels, out_channels = intermed_channels, kernel_size = tuple(spatial_kernel_size), pad_mode='pad', padding=tuple(spatial_padding), stride=tuple(spatial_stride), weight_init='he_normal', has_bias=False, data_format="NCDHW")
        self.bn = BatchNorm3d(num_features=intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase

        self.temporal_conv = Conv3d(in_channels = intermed_channels, out_channels = out_channels, kernel_size = tuple(temporal_kernel_size), pad_mode='pad', padding=tuple(temporal_padding), stride=tuple(temporal_stride), weight_init='he_normal', has_bias=False, data_format="NCDHW")
        self.bn2 = BatchNorm3d(num_features=out_channels)
        pass
    def construct(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x
        
    pass
