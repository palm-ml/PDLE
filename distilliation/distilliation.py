import random
import pickle
import sys

import numpy as np 
import torch
import torch.nn.functional as F 
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from utils import * 
from wideresnet import WideResNet


class TrainingDataWithSoftLabels(Dataset):
    def __init__(self, num_class, features, labels, soft_labels, transform=None):
        super(TrainingDataWithSoftLabels, self).__init__()
        self.num_class = num_class
        self.features = features
        self.labels = labels
        self.soft_labels = soft_labels
        self.transform = transform
    
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        soft_label = self.soft_labels[index]
        if not self.transform == None:
            feature = Image.fromarray(feature.squeeze())
            feature = self.transform(feature)

        return feature, label, soft_label

    def __len__(self):
        return len(self.features)


def train_clean_net(epoch, c_data, model, optimizer, args):
    device = args['device']
    data_loader = torch.utils.data.DataLoader(c_data,batch_size=args['batch_size'],shuffle=True, drop_last=True)
    model.train()
    for idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)
        # forward
        outputs = model(batch_x)
        # backward
        loss = F.cross_entropy(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def gen_soft_labels(args):
    device = torch.device("cuda:"+str(args['gpu']) if torch.cuda.is_available() else "cpu")
    args['device'] = device
    setup_seed(args['seed'])
    # setup data
    if args['dataset'] == 'cifar10':
        args['num_class'] = 10
    elif args['dataset'] == 'cifar100':
        args['num_class'] = 100
    else:
        raise ValueError('Unknown dataset: %s' %args['dataset'])
    mean = [0.4914,0.4822,0.4465]
    std = [0.2023,0.1994,0.2010]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # load data from file
    folder_path = p.join(p.dirname(__file__), args['save_folder'], args['dataset'])
    data_path = p.join(folder_path,'datas'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
    
    with open(data_path,'rb') as f:
        clean_data = pickle.load(f)
        noisy_data = pickle.load(f)
    
    method_map = ["Distilliation", "Distilliation + Detected"]

    clean_features = clean_data.features
    noisy_features = noisy_data.features
    clean_labels = clean_data.labels 
    noiy_labels = noisy_data.noisy_labels
        
    clean_confidence = np.ones(clean_labels.shape, dtype=np.float32)
    noisy_confidence = noisy_data.label_confidence
    clean_grounds = clean_data.labels
    noisy_grounds = noisy_data.labels

    features = np.vstack((clean_features, noisy_features))
    labels = np.hstack((clean_labels, noiy_labels))
    grounds = np.hstack((clean_grounds, noisy_grounds))
    confidence = np.hstack((clean_confidence, noisy_confidence))
    train_data = TrainDataAll(num_class=args['num_class'],features=features,labels=labels,grounds=grounds,label_confidence=confidence,transform=train_transform,threshold=args['thre'])


    if args['method'] == 0:
        features = clean_data.features
        labels = clean_data.labels
        ctrain_data = CleanData(num_class=args['num_class'], features=features, labels=labels, transform=train_transform)
    elif args['method'] == 1:
        sel_idxes = np.where(noisy_data.label_confidence >= args['thre'])[0]
        new_clean_features = noisy_data.features[sel_idxes]
        new_clean_labels = noisy_data.noisy_labels[sel_idxes]
        features = np.vstack((clean_data.features, new_clean_features))
        labels = np.hstack((clean_data.labels, new_clean_labels))
        ctrain_data = CleanData(num_class=args['num_class'], features=features, labels=labels, transform=train_transform)
    # setup model
    model = WideResNet(depth=40, num_classes=args['num_class'], widen_factor=2, dropRate=0.3).to(device)
    optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum=0.9, weight_decay=0.0005, nesterov=True)
    # train model
    for epoch in range(max(args['epochs'], 2000//args['batch_size']+1)):
        train_clean_net(epoch, ctrain_data, model, optimizer, args)
    # estimate lambda
    model.eval()
    tp_count = np.zeros(args['num_class'])
    tp_fp_count = np.zeros(args['num_class'])
    y_br = np.zeros(args['num_class'])
    y_tilde_br = np.zeros(args['num_class'])
    for i in range(len(train_data)):
        data, n_label, c_label, _, _ = train_data[i]
        data = data.unsqueeze(0).to(device)
        output = model(data)
        pred = output.argmax().data.cpu().numpy()
        if pred == c_label:
            tp_count[pred] += 1
        if n_label == c_label:
            y_tilde_br[c_label] += 1
        tp_fp_count[c_label] += 1
        y_br[c_label] += 1
    precision = tp_count / (tp_fp_count + 1e-8)
    clean_ap = np.mean(precision)
    n_precision = y_tilde_br / (y_br + 1e-8)
    noisy_ap = np.mean(n_precision)
    comb_lambda = float(clean_ap / (noisy_ap + clean_ap))
    # generate soft labels
    model.eval()
    clean_soft_labels = np.zeros((len(clean_data), args['num_class']))
    noisy_soft_labels = np.zeros((len(noisy_data), args['num_class']))

    clean_data.transform = test_transform
    noisy_data.transform = test_transform
    
    for i in range(len(clean_data)):
        x, label = clean_data[i]
        label_one_hot = np.zeros(args['num_class'])
        label_one_hot[label] = 1

        x = x.unsqueeze(0).to(device)
        label_soft = F.softmax(model(x)).data.cpu().numpy()

        target = comb_lambda * label_soft + (1-comb_lambda)*label_one_hot
        clean_soft_labels[i] = target
    
    print("Generate Soft Label:" + method_map[args['method']])
    for i in range(len(noisy_data)):
        _,x,_,label,_ = noisy_data[i]
        label_one_hot = np.zeros(args['num_class'])
        label_one_hot[label] = 1

        x = x.unsqueeze(0).to(device)
        label_soft = F.softmax(model(x)).data.cpu().numpy()
        target = comb_lambda * label_soft + (1-comb_lambda)*label_one_hot
        noisy_soft_labels[i] = target
    # save soft labels
    save_file = p.join(folder_path,'soft_labels'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_method'+str(args['method'])+'_seed'+str(args['seed']))
    with open(save_file, 'wb') as f:
        pickle.dump(clean_soft_labels, f)
        pickle.dump(noisy_soft_labels, f)
 

def train(epoch, datas, model, optimizer, args):
    device = args['device']
    model.train()
    data_loader = torch.utils.data.DataLoader(datas, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    
    train_loss = []
    for idx,(batch_x, _, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.float().to(device)
        # forward
        outputs = F.softmax(model(batch_x))
        loss = -(batch_y * torch.log(outputs + 1e-12)).sum(1).mean(0)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data.cpu())
    if (epoch+1) % 50 == 0:
        print('Epoch {:04d}'.format(epoch + 1))
        print('train_loss: {:.03f} '.format(np.mean(train_loss)))


def test(data, model, args):
    device = args['device']
    model.eval()
    data_loader = torch.utils.data.DataLoader(data, batch_size=args['batch_size'], shuffle=False)
    n_correct = 0.0
    for idx,(batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        outputs = model(batch_x)
        preds = outputs.argmax(1)
        n_correct += preds.eq(batch_y).sum().float().cpu().numpy()
    acc = n_correct / len(data)
    print("Test acc: {:.03f}".format(acc))

def main(args):
    device = torch.device("cuda:"+str(args['gpu']) if torch.cuda.is_available() else "cpu")
    args['device'] = device
    setup_seed(args['seed'])
    # setup data
    if args['dataset'] == 'cifar10':
        args['num_class'] = 10
    elif args['dataset'] == 'cifar100':
        args['num_class'] = 100
    else:
        raise ValueError('Unknown dataset: %s' %args['dataset'])
    mean = [0.4914,0.4822,0.4465]
    std = [0.2023,0.1994,0.2010]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # load data from file
    folder_path = p.join(p.dirname(__file__), args['save_folder'], args['dataset'])
    data_path = p.join(folder_path,'datas'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
    s_data_path = p.join(folder_path,'soft_labels'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_method'+str(args['method'])+'_seed'+str(args['seed']))
    with open(data_path,'rb') as f:
        clean_data = pickle.load(f)
        noisy_data = pickle.load(f)
    with open(s_data_path, 'rb') as f:
        clean_soft_labels = pickle.load(f)
        noisy_soft_labels = pickle.load(f)
    # combine data
    clean_features = clean_data.features
    noisy_features = noisy_data.features
    features = np.vstack((clean_features, noisy_features))
    clean_labels = clean_data.labels
    noisy_labels = noisy_data.labels
    labels = np.hstack((clean_labels, noisy_labels))
    soft_labels = np.vstack((clean_soft_labels, noisy_soft_labels))
    train_data = TrainingDataWithSoftLabels(num_class=args['num_class'], features=features, labels=labels, soft_labels=soft_labels, transform=train_transform)
    # load C_hat
    C_path = p.join(folder_path,'C_glc_'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
    with open(C_path, 'rb') as f:
        C_hat = pickle.load(f)
    print(np.round(C_hat, 3))
    C_hat = torch.from_numpy(C_hat).float().to(device)

    # setup model
    model = WideResNet(depth=40, num_classes=args['num_class'], widen_factor=2, dropRate=0.3).to(device)
    optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60,80,90], gamma=0.2, last_epoch=-1)

    method_map = ["Distilliation", "Distilliation + Detected"]
    # training
    for epoch in range(args['epochs']):
        train(epoch, train_data, model, optimizer,args)
        scheduler.step()

    # testing
    test_features, test_labels = load_data(args['dataset'], train=False)
    test_data = TrainData(args['num_class'], test_features, test_labels, transform=test_transform)
    test(test_data, model, args)
    
