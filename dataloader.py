import scipy.io as sio
import math
from sklearn.decomposition import PCA
import numpy as np
from sklearn import preprocessing

def load_dataset(Dataset, is_pca):
    data_path = './data/'
    if Dataset == 'IN':
        data_path = data_path + 'IndianPines/'
        mat_data = sio.loadmat(data_path + 'Indian_pines_corrected.mat')#读取训练样本
        mat_gt = sio.loadmat(data_path + 'Indian_pines_gt.mat')  #读取标签
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        # 总共21025个像素。10249是地物像素，其余10776是背景像素（需要剔除），除去背景共16类
        # K = 30
        K = 40

    elif Dataset == 'PU':
        data_path = data_path + 'PaviaU/'
        uPavia = sio.loadmat(data_path + 'PaviaU.mat')
        gt_uPavia = sio.loadmat(data_path + 'PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        # K = 15
        K = 30
    elif Dataset == 'PC':
        data_path = data_path + 'PaviaC/'
        uPavia = sio.loadmat(data_path + 'Pavia.mat')
        gt_uPavia = sio.loadmat(data_path + 'Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        K = 15
    elif Dataset == 'XZ':
        data_path = data_path + 'Xuzhou/'
        uPavia = sio.loadmat(data_path + 'xuzhou.mat')
        gt_uPavia = sio.loadmat(data_path + 'xuzhou_gt.mat')
        data_hsi = uPavia['xuzhou']
        gt_hsi = gt_uPavia['xuzhou_gt']
        K = 15
        # K = 15

    elif Dataset == 'HS2018':
        data_path = data_path + 'Houston2018/'
        HS = sio.loadmat(data_path + 'HoutonU2018_img.mat')
        gt_HS = sio.loadmat(data_path + 'HoutonU2018_gt.mat')
        # S = whosmat(data_path + 'Houston_gt.mat')
        # for i in S:
        #     print(i)
        data_hsi = HS['HoutonU2018_img']
        gt_hsi = gt_HS['HoutonU2018_gt']
        K = 16
    elif Dataset == 'SC':
        data_path = data_path + 'Salinas/'
        SV = sio.loadmat(data_path + 'Salinas_corrected.mat')
        gt_SV = sio.loadmat(data_path + 'Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        K = 15

    elif Dataset == 'XA':
        data_path = data_path + 'Xiongan/'
        SV = sio.loadmat(data_path + 'xiongan.mat')
        gt_SV = sio.loadmat(data_path + 'xiongan_gt.mat')
        data_hsi = SV['xiongan']
        gt_hsi = gt_SV['xiongan_gt']
        K = 50
    elif Dataset == 'HC':
        data_path = data_path + 'WHU-Hi-HanChuan/'
        HC = sio.loadmat(data_path + 'WHU_Hi_HanChuan.mat')
        gt_HC = sio.loadmat(data_path + 'WHU_Hi_HanChuan_gt.mat')
        # S = whosmat(data_path + 'Houston_gt.mat')
        # for i in S:
        #     print(i)
        data_hsi = HC['WHU_Hi_HanChuan']
        gt_hsi = gt_HC['WHU_Hi_HanChuan_gt']
        K = 20
    elif Dataset == 'HH':
        data_path = data_path + 'WHU-Hi-HongHu/'
        HC = sio.loadmat(data_path + 'WHU_Hi_HongHu.mat')
        gt_HC = sio.loadmat(data_path + 'WHU_Hi_HongHu_gt.mat')
        data_hsi = HC['WHU_Hi_HongHu']
        gt_hsi = gt_HC['WHU_Hi_HongHu_gt']
        # K = 30
        K = 30
    # shapeor = list(data_hsi.shape)
    shapeor1 = list(data_hsi.shape)
    data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])

    if is_pca:
        data_hsi_pca = PCA(n_components=K).fit_transform(data_hsi)
        shapeor1[-1] = K
        data_pca = preprocessing.scale(data_hsi_pca)
        shapeor_pca = np.array(shapeor1)
        data_pca = data_pca.reshape(shapeor_pca)

    # data = preprocessing.scale(data_hsi)
    # shapeor = np.array(shapeor)

    # data = data.reshape(shapeor)
    TOTAL_SIZE = np.count_nonzero(gt_hsi) #统计非零元素个数
    CLASSES_NUM = gt_hsi.max()
    # print(data.shape,data_pca.shape)
    return data_pca,gt_hsi, TOTAL_SIZE, CLASSES_NUM


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def select(Dataset, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)

    if Dataset == 'IN':
        # amount = [15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50]  # IN
        # amount = [35, 80, 80, 80, 80, 25, 80, 25, 80, 80, 80, 80, 80, 80, 80, 80]
        amount = [3, 71, 42, 12, 24, 36, 2, 25, 2, 50, 123, 30, 10, 63, 20, 5]  # 5%
    elif Dataset == 'PC':
        amount = [15, 15, 15, 15, 15, 15, 15, 15, 15]  # PC
    elif Dataset == 'PU':
        # amount = [66, 186, 21, 30, 13, 50, 13, 36, 9]  # PU
        amount = [120, 240, 52, 70, 40, 120, 40, 90, 30]
    elif Dataset == 'SC':
        amount = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
    elif Dataset == 'XA':
        amount = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    elif Dataset == 'HH':
        # amount = [140, 35, 218, 1632, 62,
        #           445, 241, 40, 108, 124,
        #           110, 89, 225, 73, 10,
        #           72, 30, 32, 87, 34,
        #           13, 40]  # 1%
        amount = [160, 50, 240, 1632, 100,
                  445, 241, 60, 108, 124,
                  110, 100, 225, 100, 20,
                  100, 50, 52, 100, 50,
                  20, 50]
    elif Dataset == 'XZ':
        # amount = [132, 20, 14, 26, 66, 12, 35, 24, 15]  # XZ  0.5%
        amount = [264, 40, 28, 52, 132, 24, 70, 48, 30]  # XZ  1%
    elif Dataset == 'HS2018':
        amount = [500, 1625, 34, 680, 246,
                  226, 13, 1989, 11188, 2293,
                  1701, 84, 2317, 493, 347,
                  575, 7, 327, 268, 341]  # HS2018  5%
    elif Dataset == 'HC':
        # amount = [223, 113, 51, 27, 6, 23, 30, 90, 47, 52, 85, 19, 45, 93, 6, 377]  # 0.5%
        # amount = [447, 227, 103, 54, 12,45, 59, 180, 95, 105,169, 37, 91, 186, 11,754]  #1%
        # amount = [339,226,153,81,18,69,90,270,141,156,255,57,135,279,17,1131]    #1.5%
        # amount = [894, 454, 206, 108, 24,90, 118, 360, 190, 210,338, 74, 182, 372, 22,1508]  #2%
        # amount = [1115, 565, 255, 135, 30, 115, 150, 450, 235, 260, 425, 95, 225, 465, 30, 1885]  # 2.5%
        amount = [1678,602,306,300,160,138,180,540,282,312,510,114,270,558,34,2262]    #3%

    else:
        raise NotImplementedError('Dataset name is ' + str(Dataset) + ', which should be in [IN, PU, PC, SC,XZ,HS2018,HC,HH].')
    for i in range(m):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1
        ]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(amount[i])
        train[i] = indices[:nb_val]
        test[i] = indices[nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
