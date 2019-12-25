from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


def main(args):
    # 获取所有 log 文件
    logs = os.listdir(args.log_path)
    data = {}
    for log in logs:
        # 获取 log 完整路径
        path = os.path.join(args.log_path,log)
        if os.path.isdir(path):
            continue
        losses = []
        with open(path) as f:
            for line in f.readlines():
                if "Current batch loss" in line:
                    pattern = re.compile("(?<=Current batch loss: )\d+\.?\d*")
                    this_loss = float(pattern.findall(line)[0])
                    losses.append(this_loss)
        # 检查 losses 合法性
        if len(losses) > 0:
            data[log] = losses
    #draw
    data = pd.DataFrame(data)
    data.boxplot()
    plt.ylabel("loss")
    plt.xlabel("learning rate")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description="Analyze training logs where loss doesn't converge during the first 10 epochs")
    parser.add_argument("log_path", help="Path of log files")
    args = parser.parse_args()
    main(args)

