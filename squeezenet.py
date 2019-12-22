#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.param_attr import ParamAttr

# SqueezeNet_v1.1
# https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1
class SqueezeNet():
    def net(self, input, class_dim=1000):
        conv_1 = fluid.layers.conv2d(
                input,
                num_filters=64,
                filter_size=3,
                stride=2,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(name="conv1_w"),
                bias_attr=ParamAttr(name='conv1_b'))
        conv1_bn_w_attr = fluid.ParamAttr(name='conv1_bn_w', initializer=fluid.initializer.Xavier(uniform=False))
        conv1_bn_b_attr = fluid.ParamAttr(name='conv1_bn_b', initializer=fluid.initializer.Xavier(uniform=False))
        conv1_bn = fluid.layers.batch_norm(input=conv_1, param_attr=conv1_bn_w_attr, bias_attr=conv1_bn_b_attr)
        pool_1 = fluid.layers.pool2d(
                conv1_bn, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        conv = self.make_fire(pool_1, 16, 64, 64, name='fire2')
        conv = self.make_fire(conv, 16, 64, 64, name='fire3')
        pool_2 = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_type='max')
        conv = self.make_fire(pool_2, 32, 128, 128, name='fire4')
        conv = self.make_fire(conv, 32, 128, 128, name='fire5')
        pool_3 = fluid.layers.pool2d(
                conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        conv = self.make_fire(pool_3, 48, 192, 192, name='fire6')
        conv = self.make_fire(conv, 48, 192, 192, name='fire7')
        conv = self.make_fire(conv, 64, 256, 256, name='fire8')
        conv = self.make_fire(conv, 64, 256, 256, name='fire9')
        conv = fluid.layers.dropout(conv, dropout_prob=0.5)
        conv = fluid.layers.conv2d(
            conv,
            num_filters=class_dim,
            filter_size=1,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(name="conv10_w"),
            bias_attr=ParamAttr(name='conv10_b'))
        conv10_bn_w_attr = fluid.ParamAttr(name='conv10_bn_w', initializer=fluid.initializer.Xavier(uniform=False))
        conv10_bn_b_attr = fluid.ParamAttr(name='conv10_bn_b', initializer=fluid.initializer.Xavier(uniform=False))
        conv = fluid.layers.batch_norm(input=conv, param_attr=conv10_bn_w_attr, bias_attr=conv10_bn_b_attr)
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
            act='relu',
            param_attr=conv_attr,
            bias_attr=ParamAttr(name=name + '_b'))
        bn_w_attr = fluid.ParamAttr(name=name + '_bn_w', initializer=fluid.initializer.Xavier(uniform=False))
        bn_b_attr = fluid.ParamAttr(name=name + '_bn_b', initializer=fluid.initializer.Xavier(uniform=False))
        conv = fluid.layers.batch_norm(input=conv, param_attr=bn_w_attr, bias_attr=bn_b_attr)
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
    model = SqueezeNet()
    image = fluid.layers.data(name='image', shape=[1, 112, 112], dtype='float32')
    net, layer = model.net(image,2)
    print("model output shape:")
    print(net.shape)
    print(layer.shape)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    model_path = "./model"
    fluid.io.save_inference_model(dirname=model_path, feeded_var_names=[image.name], target_vars=[net], executor=exe)
    # To see the model graph(no tensor dimension info)
    # 1.run visualdl --model_pb ./model --logdir ./ in terminal
    # 2.open http://0.0.0.0:8040/static/index.html#/graphs in browser