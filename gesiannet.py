from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.param_attr import ParamAttr

# gesian net
# https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1
# Note: Tiny modification is made: each conv layer is followed by a BN layer.
class GesianNet():
    def net(self, input, class_dim=1000):
        conv_1 = fluid.layers.conv2d(
                input,
                num_filters=32,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=fluid.ParamAttr(name='conv1_w', initializer=fluid.initializer.Xavier(uniform=False)),
                bias_attr=ParamAttr(name='conv1_b'))
        conv1_bn_w_attr = fluid.ParamAttr(name='conv1_bn_w', initializer=fluid.initializer.Xavier(uniform=False))
        conv1_bn_b_attr = fluid.ParamAttr(name='conv1_bn_b', initializer=fluid.initializer.Xavier(uniform=False))
        conv1_bn = fluid.layers.batch_norm(input=conv_1, param_attr=conv1_bn_w_attr, bias_attr=conv1_bn_b_attr)
        conv1_bn_relu = fluid.layers.relu(conv1_bn, name='conv1_bn_relu')
        pool_1 = fluid.layers.pool2d(
                conv1_bn_relu, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max', name='max1')
        conv = self.make_fire(pool_1, 16, 64, 64, name='fire2')
        pool_2 = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_type='max', name='max2')
        conv = self.make_fire(pool_2, 32, 128, 128, name='fire3')
        pool_3 = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max', name='max3')
        conv = self.make_fire(pool_3, 64, 192, 192, name='fire4')
        drop_1 = fluid.layers.dropout(conv, dropout_prob=0.5)
        conv = fluid.layers.conv2d(
            drop_1,
            num_filters=class_dim,
            filter_size=1,
            act='relu',
            param_attr=fluid.ParamAttr(name='conv8_w', initializer=fluid.initializer.Xavier(uniform=False)),
            bias_attr=ParamAttr(name='conv8_b'))
        conv8_bn_w_attr = fluid.ParamAttr(name='conv8_bn_w', initializer=fluid.initializer.Xavier(uniform=False))
        conv8_bn_b_attr = fluid.ParamAttr(name='conv8_bn_b', initializer=fluid.initializer.Xavier(uniform=False))
        conv = fluid.layers.batch_norm(input=conv, param_attr=conv8_bn_w_attr, bias_attr=conv8_bn_b_attr)
        pool_4 = fluid.layers.pool2d(conv, pool_type='avg', global_pooling=True)
        pool_4_flat = fluid.layers.flatten(pool_4)
        out = fluid.layers.softmax(input=pool_4_flat)
        return out, conv

    def make_fire_conv_bn(self,
                       input,
                       num_filters,
                       filter_size,
                       padding=0,
                       name=None):
        conv_attr = fluid.ParamAttr(name=name + '_w', initializer=fluid.initializer.Xavier(uniform=False))
        conv = fluid.layers.conv2d(
            input,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=padding,
            param_attr=conv_attr,
            bias_attr=ParamAttr(name=name + '_b'))
        bn_w_attr = fluid.ParamAttr(name=name + '_bn_w', initializer=fluid.initializer.Xavier(uniform=False))
        bn_b_attr = fluid.ParamAttr(name=name + '_bn_b', initializer=fluid.initializer.Xavier(uniform=False))
        conv = fluid.layers.batch_norm(input=conv, param_attr=bn_w_attr, bias_attr=bn_b_attr)
        conv = fluid.layers.relu(conv)
        return conv

    def make_fire(self,
                  input,
                  squeeze_channels,
                  expand1x1_channels,
                  expand3x3_channels,
                  name=None):
        conv = self.make_fire_conv_bn(
            input, squeeze_channels, 1, name=name + '_squeeze1x1')
        conv_path1 = self.make_fire_conv_bn(
            conv, expand1x1_channels, 1, name=name + '_expand1x1')
        conv_path2 = self.make_fire_conv_bn(
            conv, expand3x3_channels, 3, 1, name=name + '_expand3x3')
        out = fluid.layers.concat([conv_path1, conv_path2], axis=1)
        return out


if __name__ == '__main__':
    model = GesianNet()
    image = fluid.layers.data(name='image', shape=[1, 112, 112], dtype='float32')
    net, layer = model.net(image,2)
    print("model output shape:")
    print(net.shape)
    print(layer.shape)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    model_path = "./model/gesiannet/"
    fluid.io.save_inference_model(dirname=model_path, feeded_var_names=[image.name], target_vars=[net], executor=exe)
    # To see the model graph(no tensor dimension info)
    # 1.run visualdl --model_pb ./model --logdir ./ in terminal
    # 2.open http://0.0.0.0:8040/static/index.html#/graphs in browser