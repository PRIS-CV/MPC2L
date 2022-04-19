from __future__ import print_function, absolute_import
import os
import sys
import time
import math
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from losses import AssociationLoss
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler, FramePairSampler


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='/data/data/per-id/data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',  # dukemtmcreid  market1501
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=8, type=int,    # default=16
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split_id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03_labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03_classic_split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use_metric_cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max_epoch', default=1800, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start_epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--association_batch', default=16*4, type=int,
                    help="association batch size")
parser.add_argument('--train_batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test_batch', default=512, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.00035, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[900, 1500], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon for label smooth")
parser.add_argument('--num_instances', type=int, default=4,
                    help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='mbcresnet', choices=models.get_names())
parser.add_argument('--dropout', type=float, default=0, help="dropout for FC")
parser.add_argument('--s', type=float, default=16)
parser.add_argument('--m', type=float, default=0.2)
# Miscs
parser.add_argument('--print-interval', type=int, default=20)
parser.add_argument('--distance', type=str, default='consine', help="euclidean or consine")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval_step', type=int, default=5,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='/data/data/per-id/cycas_modify/log-arcface')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--sample-type', type=str, default='intra')

args = parser.parse_args()
def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    set_seed(7)
    # torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    # Data augmentation
    transform_train = T.Compose([
        T.Resize((args.height, args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability = 0.5)
    ])
    transform_unsup = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False
    # import pdb
    # pdb.set_trace()
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=FramePairSampler(dataset, batch_size=args.association_batch, type='inter'),
        batch_size=args.association_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )
    associationloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=FramePairSampler(dataset, batch_size=args.association_batch, type='intra'),
        batch_size=args.association_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, dataset=dataset,
                              dropout=args.dropout)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_xent = AssociationLoss() 
    criterion_CE = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    criterion_unsup = AssociationLoss() 
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        with torch.no_grad():
            test(model, queryloader, galleryloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        scheduler.step()

        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_unsup, criterion_CE, optimizer, trainloader, associationloader, use_gpu)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            with torch.no_grad():
                rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion_xent, criterion_unsup, criterion_CE, optimizer, trainloader, unsuperviseloader, use_gpu):
    batch_xent_loss = AverageMeter()
    batch_unsup_loss = AverageMeter()
    batch_losses_1 = AverageMeter()
    batch_losses_2 = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    unsup_data_time = AverageMeter()

    model.train()

    end = time.time()
    unsuperviseloader = iter(unsuperviseloader)
    for batch_idx, (imgs, pids, cids) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        # if epoch > 0:
            # zero the parameter gradients
        ###############################################################################
        optimizer.zero_grad()
        # forward
        outputs, features, labels, p = model(imgs, cids, pids)
        _, preds = torch.max(torch.cat(outputs,0).data, 1)
        # import pdb 
        # pdb.set_trace()
        label = torch.cat((labels[0],labels[1],labels[2],labels[3],labels[4],labels[5]),0)


        loss2 = criterion_xent(features, [args.association_batch//4,]*2)
        loss = loss2

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics

        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))

        batch_losses_2.update(loss2.item(), pids.size(0))
        batch_xent_loss.update(loss.item(), pids.size(0))
        ################################################################


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.print_interval == 0:
    
            print('Epoch{0}: {1}/{2} '
                'Time:{batch_time.avg:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'xentLoss:{xent_loss.avg:.4f} '
                'loss_xent{batch_losses_2.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                 epoch+1, batch_idx, len(trainloader), batch_time=batch_time,
                 data_time=data_time, xent_loss=batch_xent_loss, batch_losses_2=batch_losses_2, 
                 acc=corrects))

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'xentLoss:{xent_loss.avg:.4f} '
          # 'unsupLoss:{unsup_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time,
           data_time=data_time, xent_loss=batch_xent_loss, #unsup_loss=batch_unsup_loss,          
           acc=corrects))

def fliplr(img, use_gpu):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    if use_gpu: inv_idx = inv_idx.cuda()
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()
    
    model.eval()

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        end = time.time()

        n, c, h, w = imgs.size()
        features = torch.FloatTensor(n, model.module.feat_dim).zero_()
        for i in range(2):
            if(i==1):
                imgs = fliplr(imgs, use_gpu)
            if use_gpu: imgs = imgs.cuda()
            _, outputs, _, _ = model(imgs, camids, pids)
            f = outputs.data.cpu()
            features = features+f

        batch_time.update(time.time() - end)

        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids = [], [], []
    end = time.time()
    for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
        end = time.time()

        n, c, h, w = imgs.size()
        features = torch.FloatTensor(n, model.module.feat_dim).zero_()
        for i in range(2):
            if(i==1):
                imgs = fliplr(imgs, use_gpu)
            if use_gpu: imgs = imgs.cuda()
            _, outputs, _, _ = model(imgs, camids, pids)
            f = outputs.data.cpu()
            features = features+f

        batch_time.update(time.time() - end)

        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    distmat = torch.zeros((m,n))
    if args.distance == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    distmat = distmat.numpy()



    print("Computing CMC and mAP")
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]

if __name__ == '__main__':
    main()
