
import random
import pickle
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F 
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import * 
from wideresnet import WideResNet
from mentornet import MentorNet


def get_v(epoch, loss, labels, mentornet,args):
    loss_moving_avg = args['loss_moving_avg']
    # set epochs according current epoch
    cur_epoch = min(epoch, 18)
    v_ones = torch.ones(len(loss), dtype=torch.float32)
    v_zeros = torch.zeros(len(loss), dtype=torch.float32)
    # 对于小于burn_in_epoch的epoch, 强制选择所有样本
    if cur_epoch < 18:
        upper_bound = v_ones
    else:
        upper_bound = v_zeros
    # 根据epoch选择example_drip_rate和loss_p_percentile
    if cur_epoch < 18:
        drop_rate = 0.5
    elif cur_epoch < 96:
        drop_rate = 0.05
    else:
        drop_rate = 0.9
    p_percentile = 0.7
    # 更新loss_moving_avg并计算loss_diff
    np_loss = loss.data.cpu().numpy()
    percentile_loss = np.percentile(np_loss, p_percentile*100)
    percentile_loss = torch.Tensor([percentile_loss]).to(args['device'])
    alpha = args['loss_moving_avg_decay']
    loss_moving_avg = loss_moving_avg * alpha + (1-alpha) * percentile_loss
    args['loss_moving_avg'] = loss_moving_avg
    loss_diff = loss - loss_moving_avg
    # 将cur_epoch扩成向量形式
    epoch_vec = torch.ones(len(loss)) * cur_epoch
    labels = labels.float()
    # 拼接特征
    loss = loss.unsqueeze(-1)
    loss_diff = loss_diff.unsqueeze(-1)
    labels = labels.unsqueeze(-1).to(args['device'])
    epoch_vec = epoch_vec.unsqueeze(-1).to(args['device'])
    input_data = torch.cat([loss,loss_diff,labels,epoch_vec],-1)
    v = mentornet(input_data)
    # sampleing v
    p = np.copy(v.data.cpu().numpy())
    p = p.reshape(-1)
    # np.random.seed(args['seed'])
    ids = np.random.choice(p.shape[0], int(p.shape[0]*(1-drop_rate)), replace=False)
    v_dropout = np.zeros(v.size(), dtype = np.float32)
    v_dropout[ids,0] = v.data.cpu().numpy()[ids,0]
    v_dropout[np.isnan(v_dropout)] = 0
    v_dropout = torch.from_numpy(v_dropout).to(args['device'])

    v = torch.mul(v,v_dropout)
    return v

def train_basemodel(epoch, n_datas, model, mentornet, optimizer, args):
    device = args['device']
    model.train()
    mentornet.eval()
    data_loader = torch.utils.data.DataLoader(n_datas, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    train_loss = []
    for idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        # forward
        outputs = model(batch_x)
        loss = F.cross_entropy(outputs, batch_y,reduce=False)
        
        v = get_v(epoch, loss, batch_y, mentornet, args).detach()
        weighted_loss = torch.mul(loss, v).mean()

        decay_loss = model.get_decay_loss()
        weighted_decay_loss = decay_loss * (v.sum() / len(batch_x))

        total_loss = weighted_loss + weighted_decay_loss
        total_loss = weighted_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss.append(total_loss.data.cpu())
        # print(total_loss.data.cpu())
    
    # display
    if (epoch + 1) % 50 == 0: 
        print('Epoch {:04d}'.format(epoch + 1))
        print('train_loss: {:.03f} '.format(np.mean(train_loss)))

def train_mentornet(epoch, c_datas, C, model, mentornet, m_optimizer, args):
    device = args['device']
    model.eval()
    mentornet.train()
    c_datas.reset_noise(C)
    data_loader = torch.utils.data.DataLoader(c_datas, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    for idx, (batch_x, batch_ground, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        # forward
        outputs = model(batch_x).detach()
        targets = batch_y.long().to(device)
        loss = F.cross_entropy(outputs, targets,reduce=False)
        v = get_v(epoch, loss, batch_y, mentornet, args)
        v_ref = 1-v
        v_res = torch.cat((v_ref,v), -1)
        labels = batch_ground.eq(batch_y).long().to(device)

        m_loss = F.cross_entropy(v_res, labels)
        # backward
        m_optimizer.zero_grad()
        m_loss.backward()
        m_optimizer.step()

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
    C_path = p.join(folder_path,'C_glc_'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
    
    with open(data_path,'rb') as f:
        clean_data = pickle.load(f)
        noisy_data = pickle.load(f)
    with open(C_path, 'rb') as f:
        C_hat = pickle.load(f)
        C_hat = C_hat.transpose()
    noisy_features = noisy_data.features
    clean_features = clean_data.features
    noisy_labels = noisy_data.noisy_labels
    clean_labels = clean_data.labels
    features = np.vstack((clean_features, noisy_features))
    labels = np.hstack((clean_labels, noisy_labels))
    train_data = TrainData(num_class = args['num_class'], features = features, labels = labels, transform=train_transform)
    noisy_info = {'type':args['noisy_type'],'strength':args['noisy_strength']}
    verify_data = VerifyData(num_class = args['num_class'], features = clean_features, labels = clean_labels,  noisy_info = noisy_info, transform=train_transform)

    # setup model
    model = WideResNet(depth=40, num_classes=args['num_class'], widen_factor=2, dropRate=0.3).to(device)
    optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,64,77], gamma=0.1, last_epoch=-1)
    loss_moving_avg = Variable(torch.zeros(1)).to(device)
    args['loss_moving_avg'] = loss_moving_avg

    mentornet = MentorNet(args['num_class']).to(device)
    m_optimizer = optim.SGD(mentornet.parameters(), lr = 0.001)

    for epoch in range(args['epochs']):
        if epoch in [50,64,77]:
            for j in range(2):
                train_mentornet(epoch, verify_data, C_hat, model, mentornet, m_optimizer, args)
        train_basemodel(epoch, train_data, model, mentornet, optimizer, args)
        sys.stdout.flush()
        scheduler.step()

    # testing
    test_features, test_labels = load_data(args['dataset'], train=False)
    test_data = TrainData(args['num_class'], test_features, test_labels, transform=test_transform)
    test(test_data, model, args)
