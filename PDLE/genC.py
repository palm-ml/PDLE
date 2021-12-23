import random 
import pickle
import random
import pickle

import numpy as np 
import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

from utils import * 
from wideresnet import WideResNet

def train(epoch, data_loader, model, optimizer, args):
    device = args['device']
    model.train()
    # train model with noisy data
    # train_loss = []
    for idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        # forward
        output = model(batch_x)
        # backward
        loss = F.cross_entropy(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_loss.append(loss.data.cpu().numpy())
    # print('Epoch {}'.format(epoch+1),'train_loss {:.3f}'.format(np.mean(train_loss)))
    
def get_C_hat_transpose(c_data, model, args):
    device = args['device']
    batch_size = args['batch_size']
    model.eval()
    
    data_loader = torch.utils.data.DataLoader(c_data,batch_size=batch_size,shuffle=False)
    n_samples = len(c_data)
    probs = np.zeros((n_samples, args['num_class']))

    for idx, (batch_x, _) in enumerate(data_loader):
        offset = idx * batch_size
        batch_x = batch_x.to(device)
        output = F.softmax(model(batch_x)).data.cpu().numpy()
        probs[offset:offset+len(batch_x)] = output
    
    C_hat = np.zeros((args['num_class'], args['num_class']))
    labels = c_data.labels
    for i in range(args['num_class']):
        indices = np.arange(n_samples)[labels == i]
        C_hat[i] = np.mean(probs[indices], axis=0, keepdims=True)
    return C_hat.T

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
    args['transform'] = train_transform
    # load data from file
    folder_path = p.join(p.dirname(__file__), args['save_folder'], args['dataset'])
    data_path = p.join(folder_path,'datas'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
    with open(data_path,'rb') as f:
        clean_data = pickle.load(f)
        noisy_data = pickle.load(f)

    train_data = TrainData(num_class=args['num_class'], features=noisy_data.features, labels=noisy_data.noisy_labels, transform=train_transform)
    data_loader = torch.utils.data.DataLoader(train_data,batch_size=args['batch_size'],shuffle=True)
    # add the samples with high confidence into the clean set
    if args['choice'] == 'cdd':
        sel_idxes = np.where(noisy_data.label_confidence >= args['thre'])[0]
        new_clean_features = noisy_data.features[sel_idxes]
        new_clean_labels = noisy_data.noisy_labels[sel_idxes]
        new_clean_features = np.vstack((new_clean_features, clean_data.features))
        new_clean_labels = np.hstack((new_clean_labels, clean_data.labels))
        clean_data = CleanData(num_class=args['num_class'], features = new_clean_features, labels=new_clean_labels, transform=test_transform)  

    # setup model
    model = WideResNet(depth=40, num_classes=args['num_class'], widen_factor=2, dropRate=0.3).to(device)
    optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60,80,90], gamma=0.2, last_epoch=-1)
    for epoch in range(args['epochs']):
        train(epoch, data_loader, model, optimizer, args)
        scheduler.step()
    C_hat = get_C_hat_transpose(clean_data, model, args)
    # save C_hat    
    save_file = p.join(folder_path,'C_'+args['choice']+'_'+str(args['fraction'])+'_'+args['noisy_type']+'_'+str(args['noisy_strength'])+'_seed'+str(args['seed']))
    with open(save_file, 'wb') as f:
        pickle.dump(C_hat, f)
    print(np.round(C_hat, 3))
