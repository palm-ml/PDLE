import argparse

from train import main


parser = argparse.ArgumentParser(description='Train detection networks')
# data args
default_dataset = 'cifar100'
parser.add_argument('--dataset', '-DS', type = str, default=default_dataset, 
                    help = 'dataset name (default: {:s})'.format(default_dataset))
parser.add_argument('--fraction', '-frac', type = float, default=0.1, 
                    help = 'Fraction of trusted data (default: 0.1)')
parser.add_argument('--noisy_type', '-type', type = str, default='unif', 
                    help='Noisy type of data [unif or flip] (default: unif) ') 
parser.add_argument('--noisy_strength', '-strength', type = float, default=0.5, 
                    help='Strength of corruption process (default: 0.5)')

# train args
parser.add_argument('--epochs', '-epochs', type=int, default=100,
                    help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                    help = 'learning rate (default: 0.1)')
parser.add_argument('--batch_size','-batch_size', type = int, default = 128,
                    help = 'batch_size of training process (default: 64)')

parser.add_argument('--loss_moving_avg_decay', type = float, default=0.5)

# other args
parser.add_argument('--gpu', '-gpu', type = int, default = 0, 
                    help = 'device of gpu id (default: 0)')
parser.add_argument('--seed', '-seed', type = int, default = 1,
                    help = 'random seed (default: 0)')
parser.add_argument('--save_folder', '-save', type = str, default='data',
                    help = 'save folder name (default: data)')

args = vars(parser.parse_args())


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    # show noisy pattern
    print("------------------------------------------")
    print("Training on: {}".format(args['dataset']))
    print("Noisy Settings:")
    print("\t Noisy type:\t{}\n".format(args['noisy_type']),
          "\t Noisy strength:{}\n".format(args['noisy_strength']),
          "\t Data fraction:\t{}".format(args['fraction']))
    print("Random seed: {}".format(args['seed']))
    print("------------------------------------------")
    main(args)
