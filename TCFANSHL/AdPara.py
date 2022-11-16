#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/12 15:27
# @Author : fhh
# @FileName: AdPara.py
# @Software: PyCharm

import datetime
import os
import csv
import torch
import numpy as np
import pandas as pd
from model import MLTPModel
from loss_functions import SuperHeroLoss
from torch.utils.data import DataLoader
from train import get_linear_schedule_with_warmup, DataTrain, evaluate, CosineScheduler
import optuna


DEVICE = '%s' % (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def PadEncode(data, label, max_len):  # 序列编码
    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e = [], []
    sign = 0
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

        if length <= max_len and sign == 0:
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    return np.array(data_e), np.array(label_e)


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


def train(dataset_train, dataset_test, batch_size, epochs, rate_learning, embedding_size, fan_epoch,
          dropout, num_heads, max_pool, clip_pos, clip_neg, pos_weight):
    dataset_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataset_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 设置训练参数
    vocab_size = 50
    output_size = 21

    # 初始化参数训练模型相关参数
    model = MLTPModel(vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=100000)
    lr_scheduler = CosineScheduler(10000, base_lr=rate_learning, warmup_steps=500)
    criterion = SuperHeroLoss(clip_pos=clip_pos, clip_neg=clip_neg, pos_weight=pos_weight).to(DEVICE)

    # 创建初始化训练类
    Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

    Train.train_step(dataset_train, epochs=epochs, plot_picture=False)

    test_score = evaluate(model, dataset_test, device=DEVICE)

    return test_score


def get_k_fold_data(k, i, X, y):
    global X_valid, y_valid
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)

    all_dataset_train = list(zip(X_train, y_train))
    all_dataset_test = list(zip(X_valid, y_valid))
    return all_dataset_train, all_dataset_test


# 交叉验证
def k_fold(k, X_train, y_train, batch_size, epochs, rate_learning, embedding_size, fan_epoch,
           dropout, num_heads, max_pool, clip_pos, clip_neg, pos_weight):
    valid_l_aiming, valid_l_coverage, valid_l_accuracy, valid_l_absolute_true, valid_l_absolute_false = 0, 0, 0, 0, 0
    for i in range(k):
        train_data, test_data = get_k_fold_data(k, i, X_train, y_train)
        valid_ls = train(train_data, test_data, batch_size, epochs, rate_learning, embedding_size, fan_epoch,
                         dropout, num_heads, max_pool, clip_pos, clip_neg, pos_weight)

        valid_l_aiming += valid_ls["aiming"]
        valid_l_coverage += valid_ls["coverage"]
        valid_l_accuracy += valid_ls["accuracy"]
        valid_l_absolute_true += valid_ls["absolute_true"]
        valid_l_absolute_false += valid_ls["absolute_false"]
        print(f'折{i + 1}')
        print("验证集")
        print(f'aiming: {valid_ls["aiming"]}')
        print(f'coverage: {valid_ls["coverage"]}')
        print(f'accuracy: {valid_ls["accuracy"]}')
        print(f'absolute_true: {valid_ls["absolute_true"]}')
        print(f'absolute_false: {valid_ls["absolute_false"]}')

        "-------------------------------------------保存模型结果-----------------------------------------------"
        title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'Test_Time']

        model = "clip_pos %s, clip_neg %s, pos_weight %s" % (clip_pos, clip_neg, pos_weight)

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = [[model, '%.3f' % valid_ls["aiming"],
                    '%.3f' % valid_ls["coverage"],
                    '%.3f' % valid_ls["accuracy"],
                    '%.3f' % valid_ls["absolute_true"],
                    '%.3f' % valid_ls["absolute_false"],
                    now]]

        path = "{}/{}.csv".format('result', 'param1')

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

    return valid_l_aiming / k, valid_l_coverage / k, valid_l_accuracy / k, valid_l_absolute_true / k, valid_l_absolute_false / k


def main(trial):
    # 网格搜索
    batch_size = 192
    epochs = 200
    # rate_learning = trial.suggest_float('rate_learning', 5e-4, 1e-2, log=True)
    rate_learning = 0.0018

    embedding_size = 192
    fan_epoch = 1
    # fan_epoch = 1
    # dropout = trial.suggest_float('dropout', 0.5, 0.6, step=0.1)
    dropout = 0.6
    # num_heads = trial.suggest_int('num_heads', 4, 8, step=4)
    num_heads = 8
    max_pool = 5
    # max_pool = trial.suggest_int('max_pool', 2, 6)

    clip_pos = trial.suggest_float('clip_pos', 0., 1., step=0.1)
    clip_neg = trial.suggest_float('clip_neg', 0., 1, step=0.1)
    pos_weight = trial.suggest_float('pos_weight', 0., 1., step=0.1)
    # clip_pos = 0.50
    # clip_neg = 0.50
    # pos_weight = 0.50
    first_dir = 'dataset'

    max_length = 50  # the longest length of the peptide sequence

    # getting train data and test data
    train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')

    # Converting the list collection to an array
    y_train = np.array(train_sequence_label)

    # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
    x_train, y_train = PadEncode(train_sequence_data, y_train, max_length)

    x_train = torch.LongTensor(x_train)  # torch.Size([7872, 50])

    y_train = torch.Tensor(y_train)

    k = 5
    ai, co, acc, abt, abf = k_fold(k, x_train, y_train, batch_size, epochs, rate_learning, embedding_size, fan_epoch,
                                   dropout, num_heads, max_pool, clip_pos, clip_neg, pos_weight)
    print(f'{k}-折验证,平均验证:')
    print(f'aiming: {ai}')
    print(f'coverage: {co}')
    print(f'accuracy: {acc}')
    print(f'absolute_true: {abt}')
    print(f'absolute_false: {abf}\n')
    return abt


if __name__ == '__main__':
    study_space = {"clip_pos": [0., 0.3, 0.5, 0.7, 1.0],
                   "clip_neg": [0., 0.3, 0.5, 0.7, 1.0],
                   "pos_weight": [0., 0.3, 0.5, 0.7, 1.0]}
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(study_space), direction='maximize')
    study.optimize(main)
    # fig1 = optuna.visualization.plot_slice(study, params=["embedding_size", "fan_epoch"])
    fig2 = optuna.visualization.plot_param_importances(study)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('para_image/optimization_history.png')
    # fig1.write_image('para_image/slice.png')
    fig2.write_image("para_image/param_importance.png")
