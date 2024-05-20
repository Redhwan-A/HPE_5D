
# python3 train_5D_Resnet.py   --dataset CMU --data_dir /home/redhwan/2/HPE/DirectMHP/exps/sixdrepnet/datasets/CMU/train/   --filename_list /home/redhwan/2/HPE/DirectMHP/exps/sixdrepnet/datasets/CMU/files_train.txt --output_string CMU
import time
import datetime
import math
import re
import sys
import os
import argparse
import csv
import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

matplotlib.use('TkAgg')

from model import RepNet6D, RepNet5D,Resnet5D
import utils
import datasets
from loss import GeodesicLoss


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Train a deep network to predict 3D expression.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=800, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=200, #40
        # default=16,
        type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.001, type=float) #0.00001
    parser.add_argument('--scheduler',
                        default=False,
                        # default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='',
        # default='/home/redhwan/2/HPE/quat/output/snapshots/Pose_300W_LP_20240215174446_bs1/300W_LP_epoch_27.tar',
        type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler
    dataset_name = args.dataset
    snapshot_name = args.snapshot
    # =====================learn_info tar ==================
    datetime_ = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    time_ = time.time()
    print("datetime_",datetime_,"time_",time_)

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots') #Redhwan added exist_ok=True

    summary_name = '{}_{}_bs{}'.format(
        dataset_name, datetime_, args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name)) #Redhwan added exist_ok=True
    #=====================learn_info txt==================
    if not os.path.exists('output/learn_info'):
        os.makedirs('output/learn_info') #Redhwan added exist_ok=True

    name_txt = '{}_{}'.format(
        dataset_name, datetime_)

    if not os.path.exists('output/learn_info/{}'.format(name_txt)):
        os.makedirs('output/learn_info/{}'.format(name_txt)) #Redhwan added exist_ok=True
    backbone_name = 'resnet50'
    model = Resnet5D(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 5)

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(
                                              224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)
    print('pose_dataset_____________', pose_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    model.cuda(gpu)
    # crit = GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    # crit = GeodesicLoss().cuda(gpu)
    crit = torch.nn.MSELoss().cuda(gpu)
    # crit = quat_chordal_squared_loss #quat_squared_loss
    # crit = rotmat_frob_squared_norm_loss
    # crit = quat_loss
    # crit = quat_squ_loss
    # crit =  opal_loss
    # crit = quaternion_geodesic_loss_Red
    # crit = quat_consistency_loss
    # crit = quat_self_supervised_primal_loss
    # crit = bingham_likelihood_loss
    # crit = bingham_loss
    # crit = quaternion_geodesic_loss
    # crit = quat_geodesic_loss_antipodal
    # crit = quaternion_Anti_Geodesic_loss
    # crit = quaternion_mse_loss
    # crit = quaternion_loss
    # crit = frobenius_squared_norm_loss
    # crit = unit_quaternion_regularization_loss
    # crit = lie_algebra_loss
    # crit = quaternion_angular_error_loss
    # crit = quaternion_distance_loss
    # crit =quat_squared_loss

    # softmax = nn.Softmax().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr) #Adam
    # optimizer = torch.optim.AdamW(model.parameters(), args.lr) #AdamW
    # optimizer = torch.optim.SGD(model.parameters(), args.lr)  # SGD
    print('optimizer', optimizer)


    if not args.snapshot == '':
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])

    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    outfile = open('output/learn_info/' + name_txt + '/' + args.output_string + '.txt', "a+")
    outfileplot = open('output/learn_info/' + name_txt + '/' + args.output_string + 'plot.txt', "a+")
    b= ('optimizer: %s, crit: %s, dataset_name: %s, backbone_name: %s,  batch_size: %d , lr: ' '%.7f' % ( optimizer, crit, dataset_name, backbone_name,batch_size, args.lr)  )
    outfile.write('\n')
    outfile.write(b)
    outfileplot.write(b)
    min_loss = 9999
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0

        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images).cuda(gpu)

            # Forward pass
            pred_mat = model(images)
            # print('pred_mat', type(pred_mat), pred_mat.shape )
            # print('labels.cuda(gpu)', type(labels.cuda(gpu)), labels.cuda(gpu).shape)
            # print('cont_labels.cuda(gpu)', type(cont_labels.cuda(gpu)), cont_labels.cuda(gpu).shape)

            # Calc loss
            # print('pred_mat.shape', pred_mat.shape, 'labels.cuda(gpu)', labels.cuda(gpu).shape)
            loss = crit(labels.cuda(gpu), pred_mat)

            # MSE loss
            # pred_mat_predicted = softmax(pred_mat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()


            if (i+1) % int(len(train_loader)//3) == 0: #if (i+1) % 100 == 0:
                a= ('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.7f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                      )
                      )
                print(a)
                # with open('distance.csv', "wb") as f:
                #     wr = csv.writer(f, dialect='excel')
                #     wr.writerow(a)

                outfile.write('\n')
                outfile.write(a)

        avg_loss = loss_sum / (i + 1)
        if min_loss > avg_loss:
            min_loss = avg_loss


        b = ("Epoch: %d, avg_loss: %.7f, min_loss: %.7f" % (epoch + 1, avg_loss, min_loss))
        c = ('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.7f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          avg_loss,
                      )
                      )
        print(b)
        outfile.write('\n')
        outfile.write(b)
        outfileplot.write('\n')
        outfileplot.write(c)
        if b_scheduler:
            print('kkkkkkkkkkkkkkkkkk')
            scheduler.step()

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                      '_epoch_' + str(epoch+1) + '.pkl')
                  )

    outfile.close()
    outfileplot.close()



"""
/home/redhwan/.local/lib/python3.8/site-packages/torch/nn/modules/loss.py:528: UserWarning: Using a target size (torch.Size([16, 4])) that is different to the input size (torch.Size([16, 3, 3])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
Traceback (most recent call last):
  File "train_qurt.py", line 203, in <module>
    loss = crit(labels.cuda(gpu), pred_mat)
  File "/home/redhwan/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/redhwan/.local/lib/python3.8/site-packages/torch/nn/modules/loss.py", line 528, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "/home/redhwan/.local/lib/python3.8/site-packages/torch/nn/functional.py", line 2928, in mse_loss
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
  File "/home/redhwan/.local/lib/python3.8/site-packages/torch/functional.py", line 74, in broadcast_tensors
    return _VF.broadcast_tensors(tensors)  # type: ignore
RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 2

the link is https://github.com/utiasSTARS/bingham-rotation-learning and https://david-m-rosen.github.io/slides/3D_Rotation_Learning_RSS.pdf and https://arxiv.org/pdf/2006.01031.pdf
"""