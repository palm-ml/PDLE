import os
import os.path as p
import random
import pickle

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


SRC_PATH = p.join(p.dirname(__file__),'src')

def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data(dataset,train=True, src_path = SRC_PATH):
    root = p.join(src_path, dataset)
    if not p.isdir(root):
        os.mkdir(root)
    if dataset == 'cifar10':
        datas = datasets.CIFAR10(root, train=train, download=True)
    elif dataset == 'cifar100':
        datas = datasets.CIFAR100(root, train=train, download=True)
    else:
        raise ValueError('Unknown dataset: %s' %dataset)
    version = torchvision.__version__
    if train:
        if version[:3] == '0.2':
            features = np.array(datas.train_data)
            labels = np.array(datas.train_labels)
        else:
            features = np.array(datas.data)
            labels = np.array(datas.targets)
    else:
        if version[:3] == '0.2':
            features = np.array(datas.test_data)
            labels = np.array(datas.test_labels)
        else:
            features = np.array(datas.data)
            labels = np.array(datas.targets)
    return features, labels

def prepare_data(features, labels, args):
    # shuffle data
    n_samples = len(labels)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    # split data
    n_clean = int(n_samples * args['fraction'])
    while (len(set(labels[indices][:n_clean])) < args['num_class']):
        np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]
    features_clean = features[:n_clean]
    labels_clean = labels[:n_clean]
    features_noisy = features[n_clean:]
    labels_noisy = labels[n_clean:]
    # construct dataset
    clean_data = CleanData(num_class = args['num_class'], features = features_clean, labels = labels_clean, transform = args['transform'])
    noisy_info = {'type':args['noisy_type'],'strength':args['noisy_strength']}
    noisy_data = NoisyData(num_class = args['num_class'], features = features_noisy, labels = labels_noisy, noisy_info = noisy_info, transform = args['transform'] )
    return clean_data, noisy_data

class CleanData(Dataset):
    def __init__(self, num_class, features, labels, transform = None):
        super(CleanData, self).__init__()
        self.num_class = num_class
        self.features = features
        self.labels = labels
        self.transform = transform
        self._generateSupportSets()

    def _generateSupportSets(self):
        support_sets = [[] for _ in range(self.num_class)]
        n_samples = len(self.features)
        for i in range(n_samples):
            support_sets[self.labels[i]].append(self.features[i])
        self.support_sets = support_sets
    
    def getSupportSets(self, n_shot):
        n_dims = self.features[0].shape
        support_features = np.zeros((self.num_class*n_shot, *n_dims), dtype = np.uint8)
        support_labels = np.zeros(self.num_class*n_shot, dtype=np.int64)
        for i in range(self.num_class):
            cand_set = self.support_sets[i]
            support_features[i*n_shot : (i+1)*n_shot] = random.sample(list(cand_set), n_shot)
            support_labels[i*n_shot : (i+1)*n_shot] = i
        if not self.transform == None:
            support_set_new = torch.zeros((self.num_class*n_shot, 3, *n_dims[:2]))
            for i in range(len(support_features)):
                img = support_features[i]
                img = Image.fromarray(img.squeeze())
                support_set_new[i] = self.transform(img)
            support_features = support_set_new
            support_labels = torch.from_numpy(support_labels)
        return support_features, support_labels

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        if not self.transform == None:
            feature = Image.fromarray(feature.squeeze())
            feature = self.transform(feature)
        
        return feature, label

    def __len__(self):
        return len(self.labels)


class NoisyData(Dataset):
    def __init__(self, num_class, features, labels, noisy_info, transform = None):
        self.num_class = num_class
        self.features = features
        self.labels = labels
        self.transform = transform
        self.noisy_strength = noisy_info['strength']
        self.noisy_type = noisy_info['type']
        self.label_confidence = np.zeros(len(self.labels))
        self._generateNoisyLabels()

    def _generateNoisyLabels(self):
        self.noisy_labels = self.labels.copy()
        # construct noise-tranction-matrix C
        C = np.eye(self.num_class) * (1 - self.noisy_strength)
        if self.noisy_type == 'unif':
            row_indices = np.arange(self.num_class)
            for i in range(self.num_class):
                C[i][row_indices[row_indices != i]] = self.noisy_strength / (self.num_class - 1)
        elif self.noisy_type == 'flip':
            row_indices = np.arange(self.num_class)
            for i in range(self.num_class):
                C[i][np.random.choice(row_indices[row_indices != i])] = self.noisy_strength
        else:
            raise ValueError('Undefined noisy type: %s' %self.noisy_type)
        self.C = C 
        # generate noisy labels base on C
        for i in range(len(self.labels)):
            self.noisy_labels[i] = np.random.choice(self.num_class, p = self.C[self.labels[i]])


    def updateLabelConfidence(self, score, idx):
        assert idx < len(self.labels)
        self.label_confidence[idx] = score

    def __getitem__(self, index):
        feature = self.features[index]
        clean_label = self.labels[index]
        noisy_label = self.noisy_labels[index]
        label_confidence = self.label_confidence[index]
        if not self.transform == None:
            feature = Image.fromarray(feature.squeeze())
            feature = self.transform(feature)
        return index,feature,clean_label,noisy_label,label_confidence
        
    def __len__(self):
        return len(self.labels)


class TrainData(Dataset):
    def __init__(self, num_class, features, labels, transform = None):
        super(TrainData, self).__init__()
        self.num_class = num_class
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        if not self.transform == None:
            feature = Image.fromarray(feature.squeeze())
            feature = self.transform(feature)

        return feature, label

    def __len__(self):
        return len(self.labels)


class TrainDataAll(Dataset):
    def __init__(self, num_class, features, labels, grounds, label_confidence, transform = None, threshold = 0.4):
        super(TrainDataAll, self).__init__()
        self.num_class = num_class
        self.features = features
        self.labels = labels
        self.grounds = grounds
        self.transform = transform
        self.label_confidence = label_confidence
        self.threshold = threshold

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        ground = self.grounds[index]
        confidence = self.label_confidence[index]
        if confidence >= self.threshold:
            data_type = 1
        elif confidence < 0:
            data_type = 2
        else:
            data_type = 0
        if not self.transform == None:
            feature = Image.fromarray(feature.squeeze())
            feature = self.transform(feature)

        return feature, label, ground, data_type, confidence

    def __len__(self):
        return len(self.labels)

class VerifyData(TrainData):
    def __init__(self, num_class, features, labels, noisy_info, transform = None):
        super(VerifyData, self).__init__(num_class, features, labels, transform)
        self.noisy_labels = labels.copy()
        self.noisy_strength = noisy_info['strength']
        self.noisy_type = noisy_info['type']

    def reset_noise(self, C):
        self.C = C 
        # generate noisy labels base on C
        for i in range(len(self.labels)):
            p_sum = self.C[self.labels[i]].sum()
            self.noisy_labels[i] = np.random.choice(self.num_class, p =
            self.C[self.labels[i]] / p_sum)
    
    def __getitem__(self, index):
        feature = self.features[index]
        clean_label = self.labels[index]
        noisy_label = self.noisy_labels[index]
        if not self.transform == None:
            feature = Image.fromarray(feature.squeeze())
            feature = self.transform(feature)

        return feature, clean_label, noisy_label

    def __len__(self):
        return len(self.labels)
    



if __name__ == '__main__':
    mean = [0.4914,0.4822,0.4465]
    std = [0.2023,0.1994,0.2010]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(), transforms.Normalize(mean, std)])
    features, labels = load_data()
    args = {}
    args['fraction'] = 0.1
    args['transform'] = train_transform
    args['noisy_type'] = 'unif'
    args['noisy_strength'] = 0.5
    args['num_class'] = 100
    c_data, n_data = prepare_data(features, labels, args)
    c_data.getSupportSets(5)
