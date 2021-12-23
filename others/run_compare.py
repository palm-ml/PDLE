import argparse
from compare import main 


parser = argparse.ArgumentParser(description='Train detection networks')
# data args
default_dataset = 'cifar10'
parser.add_argument('--dataset', '-DS', type = str, default=default_dataset, 
                    help = 'dataset name (default: {:s})'.format(default_dataset))
parser.add_argument('--noisy_type', '-type', type = str, default='flip', 
                    help='Noisy type of data [unif or flip] (default: unif) ') 
parser.add_argument('--noisy_strength', '-strength', type = float, default=0.0, 
                    help='Strength of corruption process (default: 0.5)')

# train args
parser.add_argument('--epochs', '-epochs', type=int, default=100,
                    help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                    help = 'learning rate (default: 0.1)')
parser.add_argument('--batch_size','-batch_size', type = int, default = 128,
                    help = 'batch_size of training process (default: 64)')
# other args
parser.add_argument('--gpu', '-gpu', type = int, default = 0, 
                    help = 'device of gpu id (default: 0)')
parser.add_argument('--seed', '-seed', type = int, default = 0,
                    help = 'random seed (default: 0)')
parser.add_argument('--save_folder', '-save', type = str, default='data',
                    help = 'save folder name (default: data)')
parser.add_argument('--method', '-method', type = int, default = 0, 
                    help = 'choice of method')

args = vars(parser.parse_args())



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    for i in range(6,9):
        strength = (i)*0.1
        args['noisy_strength'] = strength
        # show noisy pattern
        print("------------------------------------------")
        print("Training on: {}".format(args['dataset']))
        print("Noisy Settings:")
        print("\t Noisy type:\t{}\n".format(args['noisy_type']),
              "\t Noisy strength:{}".format(args['noisy_strength']))
        print("Random seed: {}".format(args['seed']))
        print("------------------------------------------")
        print("Training process...")
        
        args['method'] = 0
        main(args)
        args['method'] = 1
        main(args)
  
