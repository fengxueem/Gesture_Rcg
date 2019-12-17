from argparse import ArgumentParser
import paddle
import paddle.fluid as fluid
import numpy as np
import os, sys
from multiprocessing import cpu_count
from squeezenet import SqueezeNet

# global variables
input_channel_mean = [127]
input_variance = 128.0
input_h = input_w = 112

# define reader for training data
def train_reader(train_list, batch, buffered_size=1024):
    def mapper(sample):
        '''
        :param sample: a tuple contains image path and label
        :type sample: tuple
        '''
        img, label = sample
        # load image from disk by opencv, img is in HWC format
        img = paddle.dataset.image.load_image(img,is_color=False)
        img = paddle.dataset.image.simple_transform(im=img,          #输入图片是HWC   
                                                    resize_size=input_h, # 剪裁图片
                                                    crop_size=input_h, 
                                                    is_color=False,
                                                    is_train=True,
                                                    mean=input_channel_mean)
        img= img.flatten().astype('float32')/input_variance
        return img, label
    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lebel = line.strip().split('\t')
                yield img_path, int(lebel) 
    return paddle.batch(paddle.reader.shuffle(paddle.reader.xmap_readers(mapper,
                                                                         reader,
                                                                         cpu_count(),
                                                                         buffered_size),
                                              buf_size=buffered_size),
                        batch_size=batch)

# define reader for testing data
def test_reader(test_list, batch, buffered_size=1024):
    def mapper(sample):
        '''
        :param sample: a tuple contains image path and label
        :type sample: tuple
        '''
        img_path, label = sample
        img = paddle.dataset.image.load_image(img_path,is_color=False)
        img = paddle.dataset.image.simple_transform(im=img,
                                                    resize_size=input_h,
                                                    crop_size=input_h,
                                                    is_color=False,
                                                    is_train=False,
                                                    mean=input_channel_mean)
        img= img.flatten().astype('float32')/input_variance
        return img, label
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)
    return paddle.batch(paddle.reader.xmap_readers(mapper, reader, cpu_count(), buffered_size), batch_size=batch)

def main(args):
    image = fluid.layers.data(name='image', shape=[1, input_h, input_w], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    model = SqueezeNet()
    prediction, _ = model.net(input=image, class_dim=2)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    optimizer = fluid.optimizer.Adam(learning_rate=float(args.learning_rate))
    optimizer.minimize(loss)
    # use CPU to train
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
    # batch
    batch_size = int(args.batch_size)
    train_data_reader = train_reader(args.train_list_path, batch_size)
    test_data_reader = test_reader(args.test_list_path, batch_size)
    epoch = int(args.epoch)
    test_program = fluid.default_main_program().clone(for_test=True)
    print('Start training...')
    for pass_id in range(epoch):
        train_cost = 0
        for batch_id, data in enumerate(train_data_reader()):
            train_cost, train_acc = exe.run(
                program=fluid.default_main_program(),                            
                feed=feeder.feed(data),                                         
                fetch_list=[loss, accuracy])                                
            if batch_id % 10 == 0:                                              
                print("\nPass %d, Step %d, Cost %f, Acc %f" % 
                (pass_id, batch_id, train_cost[0], train_acc[0]))
        # test after every 5 epoch
        if pass_id % 5 == 0:
            test_accs = []                                                           
            test_costs = []                                                           
            for batch_id, data in enumerate(test_data_reader()):
                test_cost, test_acc = exe.run(program=test_program,
                                            feed=feeder.feed(data),                
                                            fetch_list=[avg_cost, accuracy])      
                test_accs.append(test_acc[0])                                        
                test_costs.append(test_cost[0])                                      

            test_cost = (sum(test_costs) / len(test_costs))                           
            test_acc = (sum(test_accs) / len(test_accs))                              
            print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
        # save model every 2 epoch
        if pass_id % 2 == 0:
            model_save_dir = args.model_save_path
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            fluid.io.save_inference_model(dirname=model_save_dir, 
                                            feeded_var_names=["image"],
                                            target_vars=[prediction],
                                            executor=exe)


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a model")
    parser.add_argument('train_list_path', help="Path of test list")
    parser.add_argument('test_list_path', help="Path of test list")
    parser.add_argument('learning_rate', help="Learning rate")
    parser.add_argument('batch_size', help="Batch size")
    parser.add_argument('epoch', help="Epoch")
    parser.add_argument('model_save_path', help="Path to save the model")
    args = parser.parse_args()
    main(args)