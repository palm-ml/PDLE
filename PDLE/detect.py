import random
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

from models import MLP, resnet32
from utils import *

def construct_task_map(num_class, n_way):
    if (num_class == n_way):
        return np.ones((num_class,num_class))
    task_map = np.eye(num_class)
    for i in range(num_class):
        row_indices = np.arange(num_class)
        sel_idx = random.sample(list(row_indices[row_indices != i]),n_way-1)
        task_map[i][sel_idx] = 1
    return task_map

def train(epoch, c_data, n_data, enc, fc, optimizer, args):
    device = args['device']
    c_data_loader = torch.utils.data.DataLoader(c_data, 
        batch_size=args['query_batch_size'], shuffle=True, drop_last=True)
    n_data_loader = torch.utils.data.DataLoader(n_data,
        batch_size=args['noisy_batch_size'], shuffle=True, drop_last=True)
    
    enc.train()
    fc.train()
    # construct task
    task_map = construct_task_map(args['num_class'], args['n_way'])
    train_loss = []
    n_select = 0.0      # number of detected samples
    n_correct = 0.0     # number of correct samples in the detected samples
    n_all = 0.0         # number of correct samples in the whole set

    # train detecting-model
    for idx, (batch_idx, batch_x, batch_ground, batch_y, batch_confidence) in enumerate(n_data_loader):
        batch_ground = batch_ground.long().to(device)
        batch_confidence = batch_confidence.float().to(device)
        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        n_batch = batch_x.size(0)
        # sample query data
        query_x, query_y = c_data_loader.__iter__().next()
        query_x, query_y = query_x.to(device), query_y.long().to(device)
        n_query = query_x.size(0)
        # sample support data
        support_x, support_y = c_data.getSupportSets(args['n_shot'])
        support_x, support_y = support_x.to(device), support_y.long().to(device)
        n_support = support_x.size(0)
        # forward----------------------------------------------
        # extract features
        batch_data = torch.cat((torch.cat((support_x, query_x),0), batch_x),0)
        features = enc(batch_data)
        support_features = features[:n_support]
        query_features = features[n_support:n_support + n_query]
        batch_features = features[n_support + n_query:]
        # mlp out
        outputs = fc(features)
        # -----------------------------------------------------
        # calcuate similarity
        support_features_ext = support_features.unsqueeze(0).repeat(n_query, 1, 1)
        support_features_ext = support_features_ext.view(n_support*n_query, -1)
        query_features_ext = query_features.unsqueeze(1).repeat(1, n_support, 1)
        query_features_ext = query_features_ext.view(n_support*n_query, -1)
        query_dists = torch.pow(support_features_ext - query_features_ext, 2).mean(1)
        query_dists = query_dists.view(n_query, -1, args['n_shot'])
        query_dists = 1 / query_dists.mean(2)

        # reviced sim_loss
        sub_query_probs = torch.zeros(n_query, args['n_way']).float().to(device)
        sub_query_y = np.zeros(n_query, dtype=np.int64)
        for i in range(n_query):
            sel_list = np.where(task_map[query_y[i]])[0]
            sel_dists = query_dists[i][sel_list]
            sub_query_probs[i] = sel_dists
            sub_query_y[i] = np.where(sel_list == query_y[i].data.cpu().numpy())[0]
        sub_query_y = torch.from_numpy(sub_query_y).to(device)
        loss_sim = F.cross_entropy(sub_query_probs, sub_query_y)

        support_features_ext = support_features.unsqueeze(0).repeat(n_batch, 1, 1).detach()
        support_features_ext = support_features_ext.view(n_support*n_batch, -1)
        batch_features_ext = batch_features.unsqueeze(1).repeat(1, n_support, 1).detach()
        batch_features_ext = batch_features_ext.view(n_support*n_batch, -1)
        batch_dists = torch.pow(support_features_ext - batch_features_ext, 2).mean(1)
        batch_dists = batch_dists.view(n_batch, -1, args['n_shot'])
        batch_dists = 1 / batch_dists.mean(2)

        # calculate attention for batch_data
        batch_attention = batch_confidence.clone()
        n_valid = 0
        for i in range(n_batch):
            if (batch_y[i] == batch_ground[i]): n_all += 1
            label_confidence = batch_dists[i][batch_y[i]]
            sel_dist = batch_dists[i][np.where(task_map[batch_y[i]])]
            if torch.sum(torch.exp(sel_dist)) != 0:
                label_confidence = torch.exp(label_confidence) / torch.sum(torch.exp(sel_dist))
            else:
                label_confidence = 0.0
            batch_confidence[i] = args['alpha']*batch_confidence[i] + (1-args['alpha'])*label_confidence
            n_data.updateLabelConfidence(batch_confidence[i], batch_idx[i])
            if batch_confidence[i] > args['thre']:
                batch_attention[i] = batch_confidence[i]
                n_valid += 1
                n_select += 1
                if batch_y[i] == batch_ground[i]: n_correct += 1
            else:
                batch_attention[i] = 0
        batch_attention = batch_attention.to(device)
        # calcuate loss
        output_c = outputs[:n_support + n_query]
        output_n = outputs[n_support + n_query:]
        target_c = torch.cat((support_y, query_y), 0)
        loss_c = F.cross_entropy(output_c, target_c, size_average=False)
        loss_n = F.cross_entropy(output_n, batch_y, reduce=False)
        loss_n = loss_n.mul(batch_attention).sum()
        loss_cls = (loss_c + loss_n) / (n_valid + n_support + n_query)
        loss = loss_cls + loss_sim
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.data.cpu().numpy())
    # display
    if (epoch + 1) % 5 == 0: 
        print('Epoch {:04d}'.format(epoch + 1))
        LP = 0.0
        if (n_select > 0): LP = n_correct / n_select
        print('train_loss: {:.03f} '.format(np.mean(train_loss)),
            'label precision: {:.03f} '.format(LP),
            'label recall: {:.03f} '.format(n_correct / n_all))

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
    args['transform'] = train_transform
    train_features, train_labels = load_data(args['dataset'],train=True)
    clean_data, noisy_data = prepare_data(train_features, train_labels, args)

    # setup model
    encoder = resnet32().to(device)
    fc = MLP(in_dim = 64, out_dim = args['num_class']).to(device)
    optimizer = optim.SGD(list(encoder.parameters())+list(fc.parameters()),lr = args['learning_rate'], momentum=0.9, weight_decay=1e-4)
    schedule = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    # training
    for epoch in range(args['epochs']):
        train(epoch, clean_data, noisy_data, encoder, fc, optimizer, args)
        sys.stdout.flush()
        schedule.step()
    # save data
    save_folder = p.join(p.dirname(__file__),args['save_folder'])
    if not p.isdir(save_folder):
        os.mkdir(save_folder)
    save_path = p.join(save_folder,args['dataset'],'datas'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
    with open(save_path, 'wb') as f:
        pickle.dump(clean_data, f)
        pickle.dump(noisy_data, f)
    
