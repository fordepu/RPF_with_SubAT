import argparse

import torch

from utils import get_model
from torchsummary import summary

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet18',
                    help='model architecture (default: PreActResNet18)')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='training datasets')
parser.add_argument('--rp',default='true', action='store_true', help='if random projection')
parser.add_argument('--rp_block', default=[-1, -1], type=int, nargs='*',
                        help='block schedule of rp')
parser.add_argument('--rp_out_channel', default=48, type=int, help='number of rp output channels')

def main():
    global args
    args = parser.parse_args()
    args.num_classes = 10
    model = torch.nn.DataParallel(get_model(args))
    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

main()