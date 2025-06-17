import traj_utils
import torch
import split_data
from torchvision import transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
# import data_utils

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

def getdata(file0_1, file1_1, file1_2, batch_size):
    _, y_all, X_all = traj_utils.traj_pre(file0_1, file1_1, file1_2, file1_2)
    print("Preprocess Done.\n")

    group = 20
    cap = int(len(X_all) / 20)
    k = 1
    X_train, y_train, X_test, y_test = split_data.split_by_group(X_all, y_all, group, cap, k)

    X_train = X_train[0].astype('float32')
    X_test = X_test[0].astype('float32')

    X_train /= 255
    X_test /= 255

    # num_samples = len(X_train)
    # num_samples1 = len(X_test)

    # # 生成随机排列的索引
    # random_indices = np.random.permutation(num_samples)
    # random_indices1 = np.random.permutation(num_samples1)
    #
    # # 根据随机索引重新排序数据样本和标签
    # shuffled_X_train = X_train[random_indices]
    # shuffled_y_train = y_train[0][random_indices]
    #
    # # shuffled_X_test = X_test[random_indices1]
    # # shuffled_y_test = y_test[0][random_indices1]

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train[0])
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test[0])

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)



    #return train_loader, test_loader, y_all, X_all, X_test, y_test
    return train_loader, test_loader, y_all, X_all, X_test, y_test, X_train, y_train


