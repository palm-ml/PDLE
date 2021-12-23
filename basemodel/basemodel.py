# --- Base model: wide-resnet

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

def train(epoch, datas, model, optimizer, args):
    device = args['device']
    model.train()
    data_loader = torch.utils.data.DataLoader(datas, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    train_loss = []
    for idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        # forward
        outputs = model(batch_x)
        # loss
        loss = F.cross_entropy(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data.cpu())
    if (epoch + 1) % 10 == 0: 
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
    
    with open(data_path,'rb') as f:
        clean_data = pickle.load(f)
        noisy_data = pickle.load(f)
    # prepare data
    clean_features = clean_data.features
    noisy_features = noisy_data.features
    clean_labels = clean_data.labels 
    noiy_labels = noisy_data.noisy_labels
        
    features = np.vstack((clean_features, noisy_features))
    labels = np.hstack((clean_labels, noiy_labels))
    train_data = CleanData(num_class=args['num_class'], features=features, labels=labels, transform=train_transform)
    # setup model
    model = WideResNet(depth=40, num_classes=args['num_class'], widen_factor=2, dropRate=0.3).to(device)
    optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60,80,90], gamma=0.2, last_epoch=-1)
    # training
    for epoch in range(args['epochs']):
        train(epoch, train_data, model, optimizer, args)
        sys.stdout.flush()
        scheduler.step()
    
    # testing
    test_features, test_labels = load_data(args['dataset'], train=False)
    test_data = TrainData(args['num_class'], test_features, test_labels, transform=test_transform)
    test(test_data, model, args)
