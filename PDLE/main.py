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

def train_with_trusted(epoch, datas, model, optimizer, args):
    # train with trusted data (including the detected data 
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
    if (epoch + 1) % 20 == 0: 
        print('Epoch {:04d}'.format(epoch + 1))
        print('train_loss: {:.03f} '.format(np.mean(train_loss)))

def train_glc(epoch, datas, C_hat, model, optimizer, args):
    device = args['device']
    model.train()
    data_loader = torch.utils.data.DataLoader(datas, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    train_loss = []
    train_loss_g = []
    train_loss_n = []
    n_correct = 0.0

    for idx,(batch_x, batch_y, batch_ground, data_types, _) in enumerate(data_loader):
        batch_ground = batch_ground.long().to(device)
        batch_x = batch_x.to(device)
        # split data
        trusted_y = batch_y[data_types == 1].long().to(device)
        untrusted_y = batch_y[data_types == 0].long().to(device)
        n_trused = len(trusted_y)
        n_untrusted = len(untrusted_y)
        # forward
        outputs = model(batch_x)
        preds = outputs.argmax(1)
        n_correct += preds.eq(batch_ground).sum().data.cpu().numpy()

        loss_t = 0.0
        if n_trused > 0:
            output_t = outputs[data_types == 1]
            loss_t += F.cross_entropy(output_t, trusted_y, size_average=False)
        loss_n = 0.0
        if n_untrusted > 0:
            output_n = outputs[data_types == 0]
            pre1 = C_hat[untrusted_y]
            pre2 = torch.mul(F.softmax(output_n, 1), pre1).sum(1)
            loss_n += -(torch.log(pre2)).sum(0)
        # loss 
        loss = (loss_t + loss_n) / (n_trused + n_untrusted)
        # backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # records
        train_loss.append(loss.data.cpu())
        if n_trused > 0:
            loss_t_avg = loss_t / n_trused
            train_loss_g.append(loss_t_avg.data.cpu())
        if n_untrusted > 0:
            loss_n_avg = loss_n / n_untrusted
            train_loss_n.append(loss_n_avg.data.cpu())
    acc = n_correct / len(datas)
    # display result
    if (epoch + 1) % 20 == 0: 
        print('Epoch {:04d}'.format(epoch + 1))
        print('train_loss: {:.03f} '.format(np.mean(train_loss)),
            'train_loss_g: {:.03f} '.format(np.mean(train_loss_g)),
            'train_loss_n: {:.03f} '.format(np.mean(train_loss_n)),
            'train_acc: {:.03f}'.format(acc))

def train_cdd_soft(epoch, datas, C_hat, model, optimizer, args):
    device = args['device']
    model.train()
    data_loader = torch.utils.data.DataLoader(datas, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    
    train_loss = []
    n_correct = 0.0 
    for idx,(batch_x, batch_y, batch_ground, _, batch_confidence) in enumerate(data_loader):
        batch_ground = batch_ground.long().to(device)
        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)
        batch_confidence = batch_confidence.float().to(device)
        # train
        outputs = model(batch_x)
        preds = outputs.argmax(1)
        n_correct += preds.eq(batch_ground).sum().data.cpu().numpy()
        
        loss_ce = F.cross_entropy(outputs, batch_y, reduce=False)
        pre1 = C_hat[batch_y]
        pre2 = torch.mul(F.softmax(outputs, 1), pre1).sum(1)
        loss_for = -(torch.log(pre2))

        loss = torch.mul(loss_ce, batch_confidence) + torch.mul(loss_for,1-batch_confidence)
        loss = loss.mean()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data.cpu())
    acc = n_correct / len(datas)
    if (epoch + 1) % 5 == 0: 
        print('Epoch {:04d}'.format(epoch + 1),
            'train_loss: {:.03f} '.format(np.mean(train_loss)),
            'train_acc: {:.03f}'.format(acc))

def train_cdd_hard(epoch, datas, C_hat, model, optimizer, args):
    device = args['device']
    model.train()
    data_loader = torch.utils.data.DataLoader(datas, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    train_loss = []
    train_loss_t = []
    train_loss_d = []
    train_loss_u = []
    n_correct = 0.0
    for idx,(batch_x, batch_y, batch_ground, data_types, _) in enumerate(data_loader):
        batch_ground = batch_ground.long().to(device)
        batch_x = batch_x.to(device)
        # split data
        trust_y = batch_y[data_types == 2].long().to(device)
        detected_y = batch_y[data_types == 1].long().to(device)
        untrusted_y = batch_y[data_types == 0].long().to(device)
        n_trusted = len(trust_y)
        n_detected = len(detected_y)
        n_untrusted = len(untrusted_y)
        # forward
        outputs = model(batch_x)
        preds = outputs.argmax(1)
        n_correct += preds.eq(batch_ground).sum().data.cpu().numpy()
        
        # loss
        loss_t = 0.0
        if n_trusted > 0:
            output_t = outputs[data_types == 2]
            loss_t = F.cross_entropy(output_t, trust_y, size_average=False)
        loss_d = 0.0
        if n_detected > 0:
            output_d = outputs[data_types == 1]
            l1_pre1 = C_hat[detected_y]
            l1_pre2 = torch.mul(F.softmax(output_d, 1), l1_pre1).sum(1)
            l1 = -(torch.log(l1_pre2)).sum(0)
            l2 = F.cross_entropy(output_d, detected_y, size_average=False)
            loss_d = args['beta']*l1 + (1-args['beta'])*l2
        loss_u = 0.0
        if n_untrusted > 0:
            output_u = outputs[data_types == 0]
            pre1 = C_hat[untrusted_y]
            pre2 = torch.mul(F.softmax(output_u, 1), pre1).sum(1)
            loss_u = -(torch.log(pre2)).sum(0)
        loss = (loss_t + loss_d + loss_u) / (n_trusted + n_detected + n_untrusted)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # records
        train_loss.append(loss.data.cpu())
        if n_trusted > 0:
            loss_t_avg = loss_t / n_trusted
            train_loss_t.append(loss_t_avg)
        if n_detected > 0:
            loss_d_avg = loss_d / n_trusted
            train_loss_d.append(loss_d_avg)
        if n_untrusted > 0:
            loss_u_avg = loss_u / n_untrusted
            train_loss_u.append(loss_u_avg)

    acc = n_correct / len(datas)
    # display result
    if (epoch + 1) % 20 == 0: 
        print('Epoch {:04d}'.format(epoch + 1))
        print('train_loss: {:.03f} '.format(np.mean(train_loss)),
            'train_acc: {:.03f}'.format(acc))

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
    """
    args[method]: The selected methods
        0: Trusted only
        1: Trusted + Detected
        2: GLC
        3: GLC + Detected
        4: CDD_hard
        5: CDD_soft
    """
    method_map = ["Trusted only", "Trusted + Detected","GLC","GLC + Detected","CDD_hard","CDD_soft"]
    if args['method'] == 0:
        features = clean_data.features
        labels = clean_data.labels
        train_data = CleanData(num_class=args['num_class'], features=features, labels=labels, transform=train_transform)
    elif args['method'] == 1:
        sel_idxes = np.where(noisy_data.label_confidence >= args['thre'])[0]
        new_clean_features = noisy_data.features[sel_idxes]
        new_clean_labels = noisy_data.noisy_labels[sel_idxes]
        features = np.vstack((clean_data.features, new_clean_features))
        labels = np.hstack((clean_data.labels, new_clean_labels))
        train_data = CleanData(num_class=args['num_class'], features=features, labels=labels, transform=train_transform)
    elif args['method'] == 2:
        C_path = p.join(folder_path,'C_glc_'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
        with open(C_path, 'rb') as f:
            C_hat = pickle.load(f)
        print(np.round(C_hat, 3))
        C_hat = torch.from_numpy(C_hat).float().to(device)

        clean_features = clean_data.features
        noisy_features = noisy_data.features
        clean_labels = clean_data.labels 
        noiy_labels = noisy_data.noisy_labels
        clean_confidence = np.ones(clean_labels.shape, dtype=np.float32)
        noisy_confidence = np.zeros(noiy_labels.shape, dtype=np.float32)
        clean_grounds = clean_data.labels
        noisy_grounds = noisy_data.labels

        features = np.vstack((clean_features, noisy_features))
        labels = np.hstack((clean_labels, noiy_labels))
        grounds = np.hstack((clean_grounds, noisy_grounds))
        confidence = np.hstack((clean_confidence, noisy_confidence))
        train_data = TrainDataAll(num_class=args['num_class'],features=features,labels=labels,grounds=grounds,label_confidence=confidence,transform=train_transform,threshold=args['thre'])
    elif args['method'] >= 3 :
        C_path = p.join(folder_path,'C_glc_'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
        with open(C_path, 'rb') as f:
            C_hat = pickle.load(f)
        print(np.round(C_hat, 3))
        C_hat = torch.from_numpy(C_hat).float().to(device)
        clean_features = clean_data.features
        noisy_features = noisy_data.features
        clean_labels = clean_data.labels 
        noiy_labels = noisy_data.noisy_labels
        
        clean_confidence = np.ones(clean_labels.shape, dtype=np.float32)
        if args['method'] == 4:
            clean_confidence = -clean_confidence
        noisy_confidence = noisy_data.label_confidence
        clean_grounds = clean_data.labels
        noisy_grounds = noisy_data.labels

        features = np.vstack((clean_features, noisy_features))
        labels = np.hstack((clean_labels, noiy_labels))
        grounds = np.hstack((clean_grounds, noisy_grounds))
        confidence = np.hstack((clean_confidence, noisy_confidence))
        train_data = TrainDataAll(num_class=args['num_class'],features=features,labels=labels,grounds=grounds,label_confidence=confidence,transform=train_transform,threshold=args['thre'])

    # setup model
    model = WideResNet(depth=40, num_classes=args['num_class'], widen_factor=2, dropRate=0.3).to(device)
    optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60,80,90], gamma=0.2, last_epoch=-1)

    print("Training Model:" + method_map[args['method']])
    # training
    for epoch in range(args['epochs']):
        if args['method'] <= 1:
            train_with_trusted(epoch, train_data, model, optimizer, args)
        elif args['method'] <= 3:
            train_glc(epoch, train_data, C_hat, model, optimizer, args)
        elif args['method'] == 4:
            train_cdd_hard(epoch, train_data, C_hat, model, optimizer, args)
        elif args['method'] == 5:
            train_cdd_soft(epoch, train_data, C_hat, model, optimizer, args)
        sys.stdout.flush()
        scheduler.step()
    
    # testing
    test_features, test_labels = load_data(args['dataset'], train=False)
    test_data = TrainData(args['num_class'], test_features, test_labels, transform=test_transform)
    test(test_data, model, args)
