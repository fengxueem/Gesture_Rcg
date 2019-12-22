import os
import json
from argparse import ArgumentParser


def main(args):
    # 设置要生成文件的路径
    data_root_path = args.dataset_path
    # 所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    class_dirs = os.listdir(data_root_path)
    # 类别标签
    class_label = 0
    # 把生产的数据列表都放在自己的总类别文件夹中
    data_list_path = args.output_list_path
    # 如果不存在这个文件夹,就创建
    isexist = os.path.exists(data_list_path)
    if not isexist:
        os.makedirs(data_list_path)
    # 清空原来的数据
    with open(data_list_path + "test.list", 'w') as f:
        pass
    with open(data_list_path + "train.list", 'w') as f:
        pass
    # 总图像数
    all_class_images = 0
    # 总类别数
    all_class_sum = 0
    # 读取每个类别
    for class_dir in class_dirs:
        # 获取类别路径
        path = os.path.join(data_root_path,class_dir)
        if not os.path.isdir(path):
            continue
        # 每个类别的信息
        class_detail_list = {}
        test_sum = 0
        train_sum = 0
        # 统计每个类别有多少张图片
        # 获取所有图片
        class_sum = 0
        img_paths = os.listdir(path)
        class_test_list = []
        class_train_list = []
        for img_name in img_paths:                                  # 遍历文件夹下的每个图片
            name_path = os.path.join(path, img_name)                       # 每张图片的路径
            if class_sum % 6 == 0:                                 # 每6张图片取一个做测试数据
                test_sum += 1                                       #test_sum测试数据的数目
                class_test_list.append(name_path)
            else:
                train_sum += 1                                    
                class_train_list.append(name_path)
            class_sum += 1                                          #每类图片的数目
        with open(data_list_path + "test.list", 'a') as f:
            for img in class_test_list:
                f.write(img + "\t%d" % class_label + "\n")#class_label 标签：0,1,2
        with open(data_list_path + "train.list", 'a') as f:
            for img in class_train_list:
                f.write(img + "\t%d" % class_label + "\n") #class_label 标签：0,1,2
        # 说明的json文件的class_detail数据
        class_detail_list['class_name'] = class_dir             #类别名称，如jiangwen
        class_detail_list['class_label'] = class_label          #类别标签，0,1,2
        class_detail_list['class_test_images'] = test_sum       #该类数据的测试集数目
        class_detail_list['class_train_images'] = train_sum #该类数据的训练集数目
        class_detail.append(class_detail_list)         
        class_label += 1                                        #class_label 标签：0,1,2
        all_class_images += class_sum                           #所有类图片的数目
        all_class_sum += 1
    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = args.data_category
    readjson['all_class_sum'] = all_class_sum
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(data_list_path + "readme.json",'w') as f:
        f.write(jsons)
    print ('Done:)')


if __name__ == '__main__':
    parser = ArgumentParser(description="Preprocess raw data and make dataset file")
    parser.add_argument('output_list_path', help="Path of output training & testing list")
    parser.add_argument('dataset_path', help="Path of dataset")
    parser.add_argument('data_category', help="Category of dataset")
    args = parser.parse_args()
    main(args)
