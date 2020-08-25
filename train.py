import os
import  numpy as np
import torch
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from multiprocessing.dummy import Pool as ThreadPool
from medpy.io import load
import argparse
from denseunet import denseUnet
from dataset import LiTSDataset
from torch.autograd import Variable
from sklearn import metrics
from loss import *
import time


parser = argparse.ArgumentParser(description='PyTorch 2d denseunet Training')
#  data folder
parser.add_argument('-data', type=str, default='data/', help='test images')
parser.add_argument('-save_path', type=str, default='experiments/')
#  other paras
parser.add_argument('-b', type=int, default=40)
parser.add_argument('-input_size', type=int, default=224)
parser.add_argument('-model_weight', type=str, default='./model/densenet161_weights_tf.h5')
parser.add_argument('-input_cols', type=int, default=3)
parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPUs for training')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='', help="specify the path for saving the model")
parser.add_argument('--batch_size', type=int, default=1, help="specify the batch size")
#  data augment
parser.add_argument('-mean', type=int, default=48)
parser.add_argument('-thread_num', type=int, default=14)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpus)
#writer = SummaryWriter()

model = denseUnet(reduction=0.5)
model = model.cuda()

# run the model on multiple gpus
model = torch.nn.DataParallel(model, device_ids=list(
    range(torch.cuda.device_count()))).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
])

train_set = LiTSDataset('{:s}/train.h5'.format(args.data), data_transform)
test_set = LiTSDataset('{:s}/test.h5'.format(args.data), data_transform)

idx_list = torch.randperm(len(train_set)).tolist()

model.train()
start = time.time()
for i in range(0, len(idx_list), args.batch_size):
    N = min(args.batch_size, len(idx_list) - i)
    loss = 0
    for k in range(N):
        idx = idx_list[i + k]
        images, masks = train_set[idx]
        images = Variable(images.cuda())
        #masks = masks.long()
        masks = Variable(masks.long().cuda())        
        output = model(images)
        masks = masks.squeeze(1)
        output = output.squeeze(1)
        loss += criterion(output, masks)
        print('loss: ', loss)
    loss /= N
    #losses.update(loss.item(), N)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    #batch_time.update(time.time() - end)
    end = time.time()
    print('batch Loss: ', loss)
