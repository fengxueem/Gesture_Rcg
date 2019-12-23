from argparse import ArgumentParser
import math
from multiprocessing import cpu_count
import numpy as np
import os, sys
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
from squeezenet import SqueezeNet
import time

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
    with open(train_list, 'r') as f:
        lines = [line.strip() for line in f]
        total_batch = len(lines) / batch
    return paddle.batch(paddle.reader.shuffle(paddle.reader.xmap_readers(mapper,
                                                                         reader,
                                                                         cpu_count(),
                                                                         buffered_size),
                                              buf_size=buffered_size),
                        batch_size=batch), total_batch

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
    os.environ['CPU_NUM'] = args.cpu_num
    image = fluid.layers.data(name='image', shape=[1, input_h, input_w], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    model = SqueezeNet()
    prediction, _ = model.net(input=image, class_dim=2)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    train_program = compiler.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)
    test_program = compiler.CompiledProgram(fluid.default_main_program().clone(for_test=True)).with_data_parallel(loss_name=loss.name)
    warmup_steps = int(args.warmup_steps)
    lr = fluid.layers.learning_rate_scheduler.noam_decay(1 / (warmup_steps *(float(args.learning_rate) ** 2)), warmup_steps)
    optimizer = fluid.optimizer.Adagrad(learning_rate=lr,
                                        regularization=fluid.regularizer.L2Decay(regularization_coeff=0.0002))
    optimizer.minimize(loss)
    # use CPU or GPU to train
    use_gpu = True if args.device == "GPU" else False
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
    # batch
    batch_size = int(args.batch_size)
    train_data_reader, total_batch = train_reader(args.train_list_path, batch_size)
    test_data_reader = test_reader(args.test_list_path, batch_size)
    epoch = int(args.epoch)
    print('Start training...', flush=True)
    for pass_id in range(epoch):
        train_loss = 0
        train_loss_list = []
        train_acc_list = []
        progress_bar_len = 10
        batch_len = total_batch / progress_bar_len
        last_work_len = 0
        for batch_id, data in enumerate(train_data_reader()):
            work_len = math.ceil(batch_id / batch_len)
            batch_start_timestamp = time.time()
            train_loss, train_acc = exe.run(
                program=train_program,                            
                feed=feeder.feed(data),                                         
                fetch_list=[loss, accuracy])    
            train_loss_list.append(train_loss[0])
            train_acc_list.append(train_acc[0])
            if work_len > last_work_len + 1:
                batch_end_timestamp = time.time()   
                last_work_len = work_len
                print("|" + ">" * work_len + "-" * (progress_bar_len - work_len) + "| Epoch[%d] %.3fs, Current batch loss: %.6f, acc: %.3f%%" % 
                    (pass_id, batch_end_timestamp - batch_start_timestamp, train_loss[0], 100.0*train_acc[0]), flush=True)
        # print average loss and accuracy during this epoch
        print("Sum: Epoch[%d], Avg loss: %.6f, Avg acc %.3f%%" % (pass_id, np.mean(train_loss_list), 100.0 * np.mean(train_acc_list)), flush=True)
        # test every 5 epochs
        if pass_id % 5 == 0:
            test_accs = []                                                           
            test_costs = []                                                           
            for batch_id, data in enumerate(test_data_reader()):
                test_loss, test_acc = exe.run(program=test_program,
                                            feed=feeder.feed(data),                
                                            fetch_list=[loss, accuracy])  
                test_accs.append(test_acc[0])                                        
                test_costs.append(test_loss[0])                                      

            #test_loss = (sum(test_costs) / len(test_costs))                           
            #test_acc = (sum(test_accs) / len(test_accs))                              
            print('Test:%d, Loss:%0.6f, Acc:%0.3f%%' % (pass_id / 5, np.mean(test_loss), 100.0*np.mean(test_acc)), flush=True)
        # save model every 2 epochs
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
    parser.add_argument('warmup_steps', help="Noam decay warm up steps")
    parser.add_argument('batch_size', help="Batch size")
    parser.add_argument('epoch', help="Epoch")
    parser.add_argument('model_save_path', help="Path to save the model")
    parser.add_argument('device', help="Training device type, CPU or GPU")
    parser.add_argument('cpu_num', help="Number of CPU cores for training")
    args = parser.parse_args()
    main(args)