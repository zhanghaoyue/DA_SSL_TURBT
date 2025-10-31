#!/usr/bin/env python

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import model.model as module_pool
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from utils import NativeScalerWithGradNormCount
from utils.util import convert_bag_to_cluster_representation
import data_loader.dataset as SSL_loader
import model.arch.simsiam_builder as simsiam_builder


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Medical Training')
parser.add_argument('--train_level', default='slide', type=str, help='training level: patch or slide')
parser.add_argument('-a', '--arch', metavar='ARCH', default='acmil_ga',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: acmil_ga)')
parser.add_argument('--fmodel', default='uni_v1',type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--et', '--encoder-type', default='conv1d', type=str)
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 5)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--cluster', default=False, action='store_true',
                    help='Use cluster representation')
parser.add_argument('--multiprocessing-distributed', default=False, action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--proj-dim', default=1024, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=1024, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', default=True, action='store_true',
                    help='Fix learning rate for the predictor')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    # if standard model, call: models.__dict__[args.arch]
    # if customized model, call:
    MIL_model = getattr(module_pool, args.arch)
    if args.train_level == 'patch':
        kwargs = {"out_channels": args.pred_dim,
                  "embed_dim": args.pred_dim,
                  "n_masked_patch": 10,
                  "pretrained_model_path": None}
    elif args.train_level == 'slide':
        kwargs = {
            "out_channels": args.pred_dim,
            "embed_dim":args.pred_dim,
            "D_inner":int(args.pred_dim/2),
            "n_token":6,
            "mask_drop":0,
            "n_masked_patch":20,
            "pretrained_model_path":None}
    
    model = simsiam_builder.SimSiam(et_type=args.et,
        MIL_encoder=MIL_model(**kwargs),feature_dim=kwargs['out_channels'],
        proj_dim=args.proj_dim, pred_dim=args.pred_dim)

    # saved path
    saved_model_path = "/raid/hzhang/training/CSSL/"+args.fmodel+"/20x/"+args.et+"/saved/"
    os.makedirs(saved_model_path, exist_ok=True)
    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 16

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model)  # print model after SyncBatchNorm

    if args.fix_pred_lr:
        optim_params = [{'params': model._mil_encoder.parameters(), 'fix_lr': False},
                        {'params': model.feature_encoder.parameters(), 'fix_lr': False},
                        {'params': model.prediction_head.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code, change foundational model features
    data_path = "features_"+args.fmodel+"/h5_files"
    cuda_device = torch.device(f"cuda:{args.gpu}")
    train_dataset = SSL_loader.Dataset_slide_CSSL(data_path=data_path, device=cuda_device,num_patches=5000)

    def collate_fn(batch):
        q_batch = [item[0] for item in batch]  # Extract q features
        k_batch = [item[1] for item in batch]  # Extract k features
        coords = [item[2] for item in batch]  # Extract coordinates
        filename = [item[3] for item in batch]
        
        batch_size = len(q_batch)
        
        # Find max number of patches in this batch
        max_patches = max(f.shape[0] for f in q_batch)
        feature_dim = q_batch[0].shape[1]
        coord_dim = coords[0].shape[1]
        
        device = q_batch[0].device
        
        # Initialize padded tensors
        padded_q = torch.full((batch_size, max_patches, feature_dim), 0.0, device=device)
        padded_k = torch.full((batch_size, max_patches, feature_dim), 0.0, device=device)
        padded_coords = torch.full((batch_size, max_patches, coord_dim), 0.0, device=device)
        
        mask = torch.zeros((batch_size, max_patches), dtype=torch.bool, device=device)
        # Populate with actual data
        for i, (q, k, c) in enumerate(zip(q_batch, k_batch, coords)):
            num_patches = q.shape[0]
            
            padded_q[i, :num_patches] = q
            padded_k[i, :num_patches] = k
            padded_coords[i, :num_patches] = c
            mask[i, :num_patches] = 1  # Valid patches marked as 1

        return padded_q, padded_k, padded_coords, mask, filename

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=False,collate_fn=None,
                                               sampler=train_sampler, drop_last=True)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print(f'mem {memory_used:.0f}MB')
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, model_path=saved_model_path,filename='/checkpoint_{:04d}.pth'.format(epoch))


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader), 
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()

    end = time.time()
    i_batch = 0

    for batch_idx, (query, key, coords, filename) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        query = query.cuda()
        key = key.cuda()
        i_batch += 1

        # multi batch training
        if args.batch_size == 1:
            
            accumulation_steps = 16
            loss_scaler = NativeScalerWithGradNormCount()
            with (torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)):
                
                # Forward pass through SimSiam and MIL-based encoders
                z1_proj, z2_proj, p1, p2, z1, z2  = model(x1=query, x2=key, coords=coords)

                # Compute the SimSiam loss
                loss_simsiam = simsiam_loss(z1_proj, z2_proj, p1, p2)
                # Compute the consistency loss (feature embedding improvement)
                loss_c = consistency_loss(z1, z2, lambda_sparse=0.1)
                # Total loss: Combine SimSiam loss and feature embedding improvement loss
                loss = loss_simsiam + 0.5 * loss_c  # Balance the losses with a hyperparameter
            
            loss = loss/accumulation_steps + 1e-10
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            grad_norm = loss_scaler(loss, optimizer, clip_grad=5.0,
                                         parameters=model.parameters(), create_graph=is_second_order,
                                         update_grad=(batch_idx + 1) % accumulation_steps == 0)
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]
            torch.cuda.synchronize()

            losses.update(loss.item(), query.data.size(0))
            
        elif args.batch_size > 1:            
            if args.cluster:
                query = convert_bag_to_cluster_representation(query, num_clusters)
                key = convert_bag_to_cluster_representation(key, num_clusters)

            # compute output and loss
            z1_proj, z2_proj, p1, p2, z1, z2 = model(x1=query, x2=key, coords=coords)
            loss_simsiam = simsiam_loss(z1_proj, z2_proj, p1, p2)
            loss_c = consistency_loss(z1, z2, lambda_sparse=0.1)
            loss = loss_simsiam + 0.5 * loss_c 

            losses.update(loss.item(), query.data.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        else:
            raise NotImplementedError('wrong batch size specified')
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)


def simsiam_loss(z1_proj, z2_proj, p1, p2, mask=None):
    """
    SimSiam loss with optional masking for padded values.
    Inputs:
      - z1_proj, z2_proj: [B, N, D] — projection outputs
      - p1, p2: [B, N, D] — prediction outputs
      - mask: [B, N] — binary mask (1 for valid patch, 0 for padding)
    """
    # Cosine similarity (stop gradient on the targets)
    sim_1 = -torch.nn.functional.cosine_similarity(p1, z2_proj.detach(), dim=-1)  # [B, N]
    sim_2 = -torch.nn.functional.cosine_similarity(p2, z1_proj.detach(), dim=-1)  # [B, N]

    if mask is not None:
        sim_1 = sim_1 * mask  # [B, N]
        sim_2 = sim_2 * mask
        sim_1 = sim_1.sum() / mask.sum().clamp(min=1)  # [B]
        sim_2 = sim_2.sum() / mask.sum().clamp(min=1)
    else:
        if sim_1.dim() == 2:
            sim_1 = sim_1.mean(dim=1)  # mean over patches -> [B]
            sim_2 = sim_2.mean(dim=1)
        else:
            sim_1 = sim_1.mean()  # for unbatched [N]
            sim_2 = sim_2.mean()
    loss = 0.5 * (sim_1.mean() + sim_2.mean())  # scalar

    return loss


def consistency_loss(z1, z2, mask=None, lambda_sparse=1e-4):
    """
    consistency loss with masked MSE reconstruction loss and sparse regularization.
    
    z1: (B, L, D) - output from the encoder
    z2: (B, L, D) - input to the consistency (reconstruction target)
    mask: (B, L) - mask indicating valid patches
    lambda_sparse: Regularization factor for sparsity penalty (L1 penalty)
    """
    # Compute MSE loss for reconstruction
    mse_loss = torch.nn.functional.mse_loss(z1, z2, reduction="none")  # (B, L, D)
    
    if mask is not None:
        # Apply mask to both MSE and sparse penalty
        mask_exp = mask.unsqueeze(-1)  # (B, L, 1)
        mse_loss = mse_loss * mask_exp
        mse_loss = mse_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mse_loss = mse_loss.mean() 
        sparse_penalty = lambda_sparse * (torch.abs(z1) * mask_exp).sum() / mask.sum().clamp(min=1)
    else:
        mse_loss = mse_loss.mean()
        sparse_penalty = lambda_sparse * torch.abs(z1).mean()
    return mse_loss + sparse_penalty
    

def save_checkpoint(state, is_best, model_path, filename='checkpoint.pth'):
    torch.save(state, model_path + filename)
    if is_best:
        shutil.copyfile(model_path + filename, model_path + '/model_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries),flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()