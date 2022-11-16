import time
import datetime
import os
import csv
import torch
import math
import numpy as np
import pandas as pd
from model import MLTPModel
from torch import nn
from torch.utils.data import DataLoader
from loss_functions import SuperHeroLoss, NewSuperHeroLoss
from train import get_linear_schedule_with_warmup, DataTrain, evaluate, CosineScheduler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']


def PadEncode(data, label, max_len):  # 序列编码
    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0
        # data_e.append(elemt)
        # length = len(data_e)
        # seq_length.append(length)
        if length <= max_len and sign == 0:
            temp.append(elemt)
            seq_length.append(len(temp[b]))
            b += 1
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    return np.array(data_e), np.array(label_e), np.array(seq_length)


def getSequenceData(first_dir, file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)

    return data, label


def staticTrainandTest(y_train, y_test):
    data_size_tr = np.zeros(len(filenames))
    data_size_te = np.zeros(len(filenames))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            if y_train[i][j] > 0:
                data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    # print("TrainingSet:\n")
    # for i in range(len(filenames)):
    #     print('{}:{}\n'.format(filenames[i], int(data_size_tr[i])))
    #
    # print("TestingSet:\n")
    # for i in range(len(filenames)):
    #     print('{}:{}\n'.format(filenames[i], int(data_size_te[i])))
    return data_size_tr


def main(num, data):
    first_dir = 'dataset'

    max_length = 50  # the longest length of the peptide sequence

    # getting train data and test data
    train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
    test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')

    # Converting the list collection to an array
    y_train = np.array(train_sequence_label)
    y_test = np.array(test_sequence_label)

    # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
    x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
    x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)

    # x_train_np = np.array(x_train)
    # y_train_np = np.array(y_train)
    # y_test_np = np.array(y_test)
    # x_test_np = np.array(x_test)
    # data_size = staticTrainandTest(y_train_np, y_test_np)

    x_train = torch.LongTensor(x_train)  # torch.Size([7872, 50])
    x_test = torch.LongTensor(x_test)  # torch.Size([1969, 50])
    train_length = torch.LongTensor(train_length)

    y_test = torch.Tensor(y_test)
    y_train = torch.Tensor(y_train)
    test_length = torch.LongTensor(test_length)

    """Create a dataset and split it into train/dev/test."""
    dataset_train = list(zip(x_train, y_train, train_length))
    dataset_test = list(zip(x_test, y_test, test_length))
    dataset_train = DataLoader(dataset_train, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
    dataset_test = DataLoader(dataset_test, batch_size=data['batch_size'], shuffle=True, pin_memory=True)

    # 设置训练参数
    vocab_size = 50
    output_size = 21

    # 类别权重

    # class_weights = []  # 类别权重
    # sumx = sum(data_size)
    #
    # m1 = (np.max(data_size) / sumx)
    # for m in range(len(data_size)):
    #     # x = {m: 18*math.pow(int((math.log((sumx / data_size[m]), 2))),2)}
    #     # x = int(sumx / (data_size[m]))
    #     # x = int((math.log((sumx / data_size[m]), 2)))
    #     # x = 8 * math.pow(int((math.log((sumx / data_size[m]), 2))), 2)
    #     x = math.pow(int((math.log((sumx / data_size[m]), 2))), 2)
    #     class_weights.append(x)  # 更新权重
    # class_weights = torch.Tensor(class_weights).to(DEVICE)

    # 初始化参数训练模型相关参数
    model = MLTPModel(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'],
                      data['num_heads'], data['max_pool'])
    rate_learning = data['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)
    lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
    # lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
    # 学习率预热 https://zhuanlan.zhihu.com/p/452448670

    # criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss(weight=class_weights).to(DEVICE)
    # criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    # BCELoss https://www.jianshu.com/p/63e255e3232f
    # criterion = BCEFocalLoss(gamma=10)
    # criterion = BCEFocalLoss(class_weight=class_weights)
    # criterion = GHMC(label_weight=class_weights)
    # criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.2, reduction='sum')
    # criterion = GHMC(label_weight=class_weights, class_weight=class_weights)
    # criterion = BinaryDiceLoss()
    # criterion = SuperDiceLoss()
    # criterion = HeroLoss()
    # criterion = NewSuperDiceLoss()
    # criterion = SuperLoss()
    # criterion = NewDiceLoss()

    criterion = SuperHeroLoss(clip_pos=data['clip_pos'], clip_neg=data['clip_neg'], pos_weight=data['pos_weight']).to(DEVICE)
    # criterion = NewSuperHeroLoss()

    # 创建初始化训练类
    Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

    a = time.time()
    Train.train_step(dataset_train, epochs=data['epochs'], plot_picture=False)
    b = time.time()
    test_score = evaluate(model, dataset_test, device=DEVICE)
    runtime = b - a
    "-------------------------------------------保存模型参数-----------------------------------------------"
    """
    staic_dict 就是一个简单的Python字典，它将模型中的可训练参数（比如：weights和biases, batchnorm的running_mean、
    torch.optim参数等）通过将模型每层与层的参数张量之间一一映射，实现保存、更新、变化和再存储。
    """
    PATH = os.getcwd()
    each_model = os.path.join(PATH, 'Model', 'model' + str(num) + '.pth')
    torch.save(model.state_dict(), each_model)
    "---------------------------------------------------------------------------------------------------"

    "-------------------------------------------输出模型结果-----------------------------------------------"
    print(f"runtime:{runtime:.3f}s")
    print("测试集：")
    print(f'aiming: {test_score["aiming"]:.3f}')
    print(f'coverage: {test_score["coverage"]:.3f}')
    print(f'accuracy: {test_score["accuracy"]:.3f}')
    print(f'absolute_true: {test_score["absolute_true"]:.3f}')
    print(f'absolute_false: {test_score["absolute_false"]:.3f}')
    "---------------------------------------------------------------------------------------------------"

    "-------------------------------------------保存模型结果-----------------------------------------------"
    title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']

    # model = "embedding%s,fan%s,head%s" % (data['embedding_size'], data['fan_epochs'], data['num_heads'])
    model = "TCFANSHL"
    test_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = [[model, '%.3f' % test_score["aiming"],
                '%.3f' % test_score["coverage"],
                '%.3f' % test_score["accuracy"],
                '%.3f' % test_score["absolute_true"],
                '%.3f' % test_score["absolute_false"],
                '%.3f' % runtime,
                test_time]]

    path = "{}/{}.csv".format('result', 'test')

    if os.path.exists(path):
        data1 = pd.read_csv(path, header=None)
        one_line = list(data1.iloc[0])
        if one_line == title:
            with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
                writer = csv.writer(t)  # 这一步是创建一个csv的写入器
                writer.writerows(content)  # 写入样本数据
        else:
            with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
                writer = csv.writer(t)  # 这一步是创建一个csv的写入器
                writer.writerow(title)  # 写入标签
                writer.writerows(content)  # 写入样本数据
    else:
        with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
            writer = csv.writer(t)  # 这一步是创建一个csv的写入器

            writer.writerow(title)  # 写入标签
            writer.writerows(content)  # 写入样本数据
    "---------------------------------------------------------------------------------------------------"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clip_pos = 1.0
    clip_neg = 0.7
    # pos_weight = 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    pos_weight = 0.3

    batch_size = 192
    epochs = 200
    learning_rate = 0.0018

    embedding_size = 192
    dropout = 0.6
    fan_epochs = 1
    num_heads = 8
    max_pool = 5

    para = {'clip_pos': clip_pos,
            'clip_neg': clip_neg,
            'pos_weight': pos_weight,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'embedding_size': embedding_size,
            'dropout': dropout,
            'fan_epochs': fan_epochs,
            "num_heads": num_heads,
            'max_pool': max_pool}

    for i in range(10):
        main(i, para)
