#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_partseg.py
@Time: 2021/7/20 7:49 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart
from model import DGCNN_partseg,CustomHeader
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from CustomDataset import CustomDataset
global class_cnts
class_indexs = np.zeros((16,), dtype=int)
global visual_warning
visual_warning = True
import trimesh
from pytorch3d.loss import chamfer_distance
from torch.utils.tensorboard import SummaryWriter

class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'):
        os.makedirs('outputs/'+args.exp_name+'/'+'visualization')
    os.system('cp main_partseg.py outputs'+'/'+args.exp_name+'/'+'main_partseg.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, visual=False):
    if not visual:
        label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def visualization(visu, visu_format, data, pred, seg, label, partseg_colors, class_choice):
    global class_indexs
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        classname = class_choices[int(label[i])]
        class_index = class_indexs[int(label[i])]
        if visu[0] != 'all':
            if len(visu) != 1:
                if visu[0] != classname or visu[1] != str(class_index):
                    skip = True 
                else:
                    visual_warning = False
            elif visu[0] != classname:
                skip = True 
            else:
                visual_warning = False
        elif class_choice != None:
            skip = True
        else:
            visual_warning = False
        if skip:
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1
        else:  
            if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname):
                os.makedirs('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname)
            for j in range(0, data.shape[2]):
                RGB.append(partseg_colors[int(pred[i][j])])
                RGB_gt.append(partseg_colors[int(seg[i][j])])
            pred_np = []
            seg_np = []
            pred_np.append(pred[i].cpu().numpy())
            seg_np.append(seg[i].cpu().numpy())
            xyz_np = data[i].cpu().numpy()
            xyzRGB = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB_gt)), axis=1)
            IoU = calculate_shape_IoU(np.array(pred_np), np.array(seg_np), label[i].cpu().numpy(), class_choice, visual=True)
            IoU = str(round(IoU[0], 4))
            filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_pred_'+IoU+'.'+visu_format
            filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_gt.'+visu_format
            if visu_format=='txt':
                np.savetxt(filepath, xyzRGB, fmt='%s', delimiter=' ') 
                np.savetxt(filepath_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
                print('TXT visualization file saved in', filepath)
                print('TXT visualization file saved in', filepath_gt)
            elif visu_format=='ply':
                xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath)
                vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath_gt)
                print('PLY visualization file saved in', filepath)
                print('PLY visualization file saved in', filepath_gt)
            else:
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                (visu_format))
                exit()
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1


def train(args, io,tolerance=100):
    log_dir = os.path.join(f'outputs/{args.exp_name}/models/', "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    train_dataset=CustomDataset("train") # !!!!!!!!!
    val_dataset=CustomDataset("val")
    print("args batch size",args.batch_size)

    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    writer = SummaryWriter(log_dir = log_dir)
    train_loader = DataLoader(train_dataset, num_workers=2, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
    val_loader= DataLoader(val_dataset, num_workers=2, batch_size=args.batch_size, shuffle=False, drop_last=drop_last)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    
    seg_start_index = train_loader.dataset.seg_start_index
    seg_num_all=3 
    class_nums=1
    
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, 50).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)

    print(args.model_path)
    model.load_state_dict(torch.load(args.dgcnn_pretrained_model_path,map_location=torch.device(device)))
 
    model.module.conv1=nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
    model.module.transform_net.transform=nn.Linear(256, 6*6)
    
    model.module.transform_net.conv1=nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
    model.module.conv8= nn.Sequential(nn.Conv1d(1024+3*64, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
    model.module.conv11=nn.Conv1d(128, 3, kernel_size=1, bias=False)
    waiting=0
    checkpoint=None
    if len(args.resume_model_path)>0:
        checkpoint = torch.load(args.resume_model_path,map_location=torch.device(device))
        print("Resume training from epoch",args.continue_epoch)
        model.load_state_dict(checkpoint["model_state_dict"])
        waiting=checkpoint["waiting"]

    model=model.to(device)
    model.train()
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if len(args.resume_model_path)>0:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.scheduler == 'cos':
        #scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
        scheduler = CosineAnnealingLR(opt, args.epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    #criterion=nn.L1Loss() 
    criterion=nn.MSELoss()
    criterion.to(device)
    
    train_loss = 0.0
    best_loss = torch.inf
    for epoch in range(args.continue_epoch,args.epochs):
        
        count = 0.0
        epoch_loss=0.
        for data,seg in train_loader:
            seg=seg.permute(0, 2, 1).float()
            data=data.float()
            data,seg = data.to(device), seg.to(device)
            
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            loss_criterion=criterion(seg_pred,seg)
            #loss_chamfer,_=chamfer_distance(seg.permute(0, 2, 1),seg_pred.permute(0, 2, 1))
            #loss=loss_chamfer+loss_criterion
            loss=loss_criterion
            
            loss.backward()
            opt.step()
            curr_loss=loss.item()
            epoch_loss+=curr_loss
       
            count+=batch_size
        
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        
        train_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        print("Epoch: ",str(epoch),", current epoch train loss: ",train_loss)
        waiting += 1
        
        if epoch>0 and epoch%args.validate_every_n==0:
            
            model.eval()
            print("Validation starts...")
            with torch.no_grad():
                loss_val=0.
                for data_val, seg_val in val_loader:
                    seg_val=seg_val.permute(0, 2, 1).float()
                    data_val=data_val.float()
                    data_val,seg_val = data_val.to(device), seg_val.to(device)
                    
                    batch_size = data_val.size()[0]
                    seg_pred_val = model(data_val)
                    opt.zero_grad()
                    loss_criterion_val=criterion(seg_pred_val,seg_val)
                    loss=loss_criterion_val
                    loss_val+=loss.item()
                    
                print("Validation loss:",loss_val/ len(val_loader))
                writer.add_scalar('Loss/val', loss_val/ len(val_loader), epoch)
            model.train()
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = model.state_dict()
                waiting = 0
                torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),"waiting":waiting}, 'outputs/%s/models/model%s.t7' % (args.exp_name,"_"+str(epoch)))
                
            if waiting >= tolerance > 0:
                print("Early stopping, training ends here. Epoch:",epoch)
                print("Saving model")
                torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),"waiting":waiting}, 'outputs/%s/models/model%s.t7' % (args.exp_name,"_"+str(epoch)))
                break
        if epoch%5==0:
            print("Saving model")
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),"waiting":waiting}, 'outputs/%s/models/model%s.t7' % (args.exp_name,"_"+str(epoch)))
        writer.add_scalar("Waiting",waiting,epoch)
    torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': opt.state_dict(),"waiting":waiting}, 'outputs/%s/models/model_final.t7' % args.exp_name)
    
def test(args,io):
    test_dataset=CustomDataset("test")
    test_loader = DataLoader(test_dataset, num_workers=8,batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device)
    
    seg_start_index = test_loader.dataset.seg_start_index
    seg_num_all=3
    class_nums=1
    
    seg_num_all = test_loader.dataset.seg_num_all
    
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, 50).to(device)
    else:
        raise Exception("Not implemented")
    model = nn.DataParallel(model)

    print(args.model_path)
    model.load_state_dict(torch.load(args.dgcnn_pretrained_model_path,map_location=torch.device(device)))

    model.module.conv1=nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
    model.module.transform_net.transform=nn.Linear(256, 6*6)
    
    model.module.transform_net.conv1=nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
    model.module.conv8= nn.Sequential(nn.Conv1d(1024+3*64, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
    model.module.conv11=nn.Conv1d(128, 3, kernel_size=1, bias=False)
    checkpoint = torch.load(args.model_path,map_location=torch.device(device))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model=model.to(device)
    model.eval()

    test_loss=0.
    total_l2_loss=0.
    total_chamfer_loss=0.
    count = 0
    
    #criterion=nn.L1Loss()
    criterion=nn.MSELoss()
    criterion.to(device)

    if not os.path.exists(args.predicted_pc):
        os.makedirs(args.predicted_pc)
    with torch.no_grad():
        for data, seg in test_loader:
            
            seg=seg.permute(0, 2, 1).float()
            data=data.float()
            data,seg = data.to(device), seg.to(device)
            count+=1 
            batch_size = data.size()[0]
            seg_pred = model(data)
            
            
            loss_criterion=criterion(seg_pred,seg)
            loss_chamfer_,_=chamfer_distance(seg.permute(0, 2, 1),seg_pred.permute(0, 2, 1))
            loss_chamfer=loss_chamfer_.detach().cpu()
            curr_loss=loss_criterion.item()

            total_l2_loss+=curr_loss
            total_chamfer_loss+=loss_chamfer
            
            print("Test loss for the batch: ",curr_loss,"chamfer_distance:",loss_chamfer)
            seg_pred=seg_pred.permute(0,2,1)
            for i in range(seg_pred.size()[0]):
                pred_np = seg_pred[i].detach().cpu().numpy()
                pred_pc = trimesh.PointCloud(pred_np)
                pred_pc.export(os.path.join(args.predicted_pc,str(count)+"_"+str(i)+".ply"))
            
    print("L2 loss for test set:",total_l2_loss/len(test_loader),"Chamfer loss for test set:",total_chamfer_loss/len(test_loader))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    parser.add_argument('--predicted_pc', type=str, 
                        help='path of the predicted pointclouds')
    parser.add_argument('--continue_epoch', type=int, default=0,
                        help='the number of epoch to continue training')
    parser.add_argument('--resume_model_path', type=str, default="",
                        help='the path of the model to resume training on top of it')
    parser.add_argument("--dgcnn_pretrained_model_path",type=str,help="path of the pretrained dgcnn")
    parser.add_argument("--validate_every_n",type=int,default=5,help="validate every nth epoch")
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    args.cuda = not args.no_cuda
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
