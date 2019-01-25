from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import shutil
import time
import subprocess
import yaml
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

# from torch.autograd import Variable

from model import cifar10_WideResNet,mnist_mlp,imagenet_resnet50
from parameterized_tensors import SparseTensor,TiedTensor

# env
from dotenv import load_dotenv
load_dotenv(verbose=True)

# data path
logger = logging.getLogger(__name__)
DATAPATH = os.getenv("DATAPATH")
if DATAPATH is None:
    logger.warning(
        "Dataset directory is not configured. Please set the "
        "DATAPATH env variable or create an .env file."
    )
    DATAPATH='./data'  # default


class split_dataset(torch.utils.data.Dataset):
      def __init__(self,parent_dataset,split_start=-1,split_end= -1):
          split_start = split_start if split_start != -1 else 0
          split_end = split_end if split_end != -1 else len(parent_dataset)          
          assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and split_start < split_end , "invalid dataset split"

          self.parent_dataset = parent_dataset
          self.split_start = split_start
          self.split_end = split_end

      def __len__(self):
          return self.split_end - self.split_start


      def __getitem__(self,index):
          assert index < len(self),"index out of bounds in split_datset"
          return self.parent_dataset[index + self.split_start]


best_prec1 = 0
def main():
    global args, best_prec1

    # args
    from arg_parse import parser
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True}
    dataset = args.model.split('_')[0]
    if dataset == 'mnist':
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

        if not(args.validate_set):
            train_loader = torch.utils.data.DataLoader(full_dataset,batch_size=args.batch_size, shuffle=True,**kwargs)
            val_loader = None
        else:
            train_dataset = split_dataset(full_dataset,split_end = 50000)
            val_dataset = split_dataset(full_dataset,split_start = 50000)         
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True,**kwargs)
            val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False,**kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.batch_size, shuffle=False,**kwargs)

    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                   std=[x/255.0 for x in [63.0, 62.1, 66.7]])




        if args.augment:
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                            (4,4,4,4),mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

        full_dataset = datasets.CIFAR10('~/wr_data', train=True, download=True,
                             transform=transform_train)

        if not(args.validate_set):
            train_loader = torch.utils.data.DataLoader(full_dataset,
                                                       batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = None
        else:
            train_dataset = split_dataset(full_dataset,split_end = 45000)
            val_dataset = split_dataset(full_dataset,split_start = 45000)         
            train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('~/wr_data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    elif dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        full_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = None


        if not(args.validate_set):
            train_loader = torch.utils.data.DataLoader(
                full_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)
            val_loader = None

        else:
            train_dataset = split_dataset(full_dataset,split_end = len(full_dataset) - 10000)
            val_dataset = split_dataset(full_dataset,split_start = len(full_dataset) - 10000)         
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)

            
        
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    else:
       raise RuntimeError('Unknown dataset {}. Dataset is first segment of network name'.format(dataset))

    print(args)
    with open(args.schedule_file, 'r') as stream:
        try:
            loaded_schedule = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
   
    if args.model == 'mnist_mlp':
        model = mnist_mlp(initial_sparsity = args.initial_sparsity_fc,sparse = not(args.tied))
    elif args.model == 'cifar10_WideResNet':
        model = cifar10_WideResNet(args.layers,widen_factor = args.widen_factor,initial_sparsity_conv = args.initial_sparsity_conv,initial_sparsity_fc = args.initial_sparsity_fc,
                                   sub_kernel_granularity = args.sub_kernel_granularity,sparse = not(args.tied))
    elif args.model == 'imagenet_resnet50':
        model = imagenet_resnet50(initial_sparsity_conv = args.initial_sparsity_conv,initial_sparsity_fc = args.initial_sparsity_fc,vanilla_downsample = not(args.sparse_resnet_downsample),
                                   sub_kernel_granularity = args.sub_kernel_granularity,sparse = not(args.tied),widen_factor = args.widen_factor)
        
    else:
        raise RuntimeError('unrecognized model name ' + repr(args.model))

    model = model.cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), get_schedule_val(loaded_schedule['lr_schedule'],0),
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)


    if dataset == 'imagenet':
        print('setting up data parallel')
        model = torch.nn.DataParallel(model).cuda()        
        base_model = model.module
    else:
        base_model = model
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


        
    # get the number of model parameters
    model_size = base_model.get_model_size()
    print(model_size)
    
    print(model)
        
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    train_loss_l = []
    test_loss_l = []
    train_prec1_l = []
    test_prec1_l = []
    train_prec5_l = []
    test_prec5_l = []

    val_loss_l = []
    val_prec1_l = []
    val_prec5_l = []
    

    prune_mode = args.prune

    start_pruning_after_epoch_n = args.start_pruning_after_epoch
    prune_every_epoch_n = args.prune_epoch_frequency
    prune_iterations = args.prune_iterations
    post_prune_epochs  = args.post_prune_epochs

    filename = args.model + '_' + repr(args.job_idx)
    n_prunes_done  = 0

    if prune_mode:
        ## Special consideration so that pruning mnist_mlp does not use less than 100 parameters in the top layer after pruning
        if args.prune_target_sparsity_fc > 0.9 and args.model == 'mnist_mlp':
              total_available_weights = (1. - args.prune_target_sparsity_fc) * (784*300 + 300 * 100 + 100 * 10) - 100
              prune_target_sparsity_special = 0.9
              prune_target_sparsity_fc = 1. - total_available_weights / (784*300 + 300 * 100)
        else:
              prune_target_sparsity_fc = prune_target_sparsity_special = args.prune_target_sparsity_fc

              
        prune_fraction_fc = 1.0 - (1 - prune_target_sparsity_fc) ** (1.0/prune_iterations)
        prune_fraction_conv = 1.0 - (1 - args.prune_target_sparsity_conv) ** (1.0/prune_iterations)        

        prune_fraction_fc_special = 1.0 - (1 - prune_target_sparsity_special) ** (1.0/prune_iterations)

        
        cubic_pruning_multipliers  = (1 - np.arange(prune_iterations+1)/prune_iterations)**3.0
        def get_prune_fraction_cubic(current_prune_iter,final_sparsity):
              return 1 - (1 - final_sparsity  + final_sparsity * cubic_pruning_multipliers[current_prune_iter+1]) / (1 - final_sparsity + final_sparsity * cubic_pruning_multipliers[current_prune_iter])
        
        nEpochs_to_prune = int(start_pruning_after_epoch_n + prune_every_epoch_n * (prune_iterations -1 ) ) + post_prune_epochs
        print('prune fraction fc : {} , prune_fraction conv : {} '.format(prune_fraction_fc,prune_fraction_conv))
        print('nepochs ' +repr(nEpochs_to_prune))
    
        filename += '_target_' + repr(args.prune_target_sparsity_fc) + ',' + repr(args.prune_target_sparsity_conv)
        validate(test_loader, model, criterion, 1,'validate')

    save_checkpoint({
            'model_size' : base_model.get_model_size(),
            'model_name' : args.model,
            'state_dict' : model.state_dict(),
            'args' : args
        }, filename = filename+'_initial')
    
        
    current_iteration = 0
    lr_schedule = loaded_schedule['lr_schedule']
    rewire_schedule = loaded_schedule['rewire_period_schedule']
    DeepR_temperature_schedule = loaded_schedule['DeepR_temperature_schedule']
    threshold = 1.0e-3
    for epoch in range(0, nEpochs_to_prune if prune_mode else args.epochs):
        adjust_learning_rate(optimizer, epoch,lr_schedule)
        rewire_period = get_schedule_val(rewire_schedule,epoch)
        DeepR_temperature = get_schedule_val(DeepR_temperature_schedule,epoch)
        print('rewiring every {} iterations'.format(rewire_period))

        t1 = time.time()
        current_iteration,threshold = train(train_loader, model, criterion, optimizer,epoch,current_iteration,rewire_period,DeepR_temperature,threshold)
        print('epoch time ' + repr(time.time() - t1))
        
        if prune_mode and epoch >= start_pruning_after_epoch_n and (epoch - start_pruning_after_epoch_n) % prune_every_epoch_n == 0 and n_prunes_done < prune_iterations:
            if args.cubic_prune_schedule:
                  base_model.prune(get_prune_fraction_cubic(n_prunes_done,prune_target_sparsity_fc),
                                   get_prune_fraction_cubic(n_prunes_done,args.prune_target_sparsity_conv),
                                   get_prune_fraction_cubic(n_prunes_done,prune_target_sparsity_special)
                  )
            else:
                  base_model.prune(prune_fraction_fc,prune_fraction_conv,prune_fraction_fc_special)                  
            n_prunes_done += 1
            print(base_model.get_model_size())
        
        if not(args.no_validate_train):
            prec1_train,prec5_train,loss_train = validate(train_loader, model, criterion, epoch,'train')
        else:
            prec1_train,prec5_train,loss_train = 0.0,0.0,0.0

        if args.validate_set:
            prec1_val,prec5_val,loss_val = validate(val_loader, model, criterion, epoch,'validate')
        else:
            prec1_val,prec5_val,loss_val = 0.0,0.0,0.0
            
        prec1_test,prec5_test,loss_test = validate(test_loader, model, criterion, epoch,'test')

        test_loss_l.append(loss_test)
        train_loss_l.append(loss_train)            
        val_loss_l.append(loss_val)
        
        test_prec1_l.append(prec1_test)
        train_prec1_l.append(prec1_train)            
        val_prec1_l.append(prec1_val)
        
        test_prec5_l.append(prec5_test)
        train_prec5_l.append(prec5_train)
        val_prec5_l.append(prec5_val)
        
        print(model)
        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'model_size' : base_model.get_model_size(),
            'test_loss' : test_loss_l,
            'train_loss' : train_loss_l,
            'val_loss' : val_loss_l,
            
            'test_prec1' : test_prec1_l,
            'train_prec1' : train_prec1_l,
            'val_prec1' : val_prec1_l,            

            'test_prec5' : test_prec5_l,
            'train_prec5' : train_prec5_l,
            'val_prec5' : train_prec5_l,            

            'model_name' : args.model,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch': epoch + 1,
            'args' : args
        }, filename = filename)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch,current_iteration,rewire_period,DeepR_temperature,threshold):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    
    # switch to train mode
    model.train()
    total_iters = len(train_loader)
    #all_sparse_tensors = [x for x in model.modules() if isinstance( x,SparseTensor) and x.get_sparsity()[0] != x.s_tensor.numel()]
    all_sparse_tensors = [x for x in model.modules() if isinstance( x,SparseTensor)]

    end = time.time()
    epoch_start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input = input.cuda()
        
        # compute output
        output = model(input)

        loss = criterion(output, target)

        L1Loss = 0.0
        for st in [x for x in model.modules() if isinstance(x,SparseTensor)]:
            if args.conv_group_lasso and st.conv_tensor:
                L1Loss += torch.sqrt((st.s_tensor**2).sum(-1).sum(-1) + 1.0e-12).sum()
            else:
                L1Loss += st.s_tensor.abs().sum()

        for st in [x for x in model.modules() if isinstance(x,TiedTensor)]:
            if args.conv_group_lasso and st.conv_tensor:
                L1Loss += torch.sqrt((st()**2).sum(-1).sum(-1) + 1.0e-12).sum()
            else:
                L1Loss += st.bank.abs().sum()
                
        loss += L1Loss * args.L1_loss_coeff

                
        # measure accuracy and record loss
        prec1,prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        adjusted_loss = loss
        adjusted_loss.backward()

        DeepR_std = np.sqrt(2 * args.DeepR_eta * DeepR_temperature)
        if args.DeepR and epoch < args.stop_rewire_epoch:
            for st in [x for x in model.modules() if isinstance(x,SparseTensor)]:
                if (st.get_sparsity()[0] != st.s_tensor.numel()):
                    st.s_tensor.grad.add_(torch.zeros_like(st.s_tensor).normal_(0.0,DeepR_std) * st.mask)

        optimizer.step()

            
        n_pruned = 0
        if args.DeepR and epoch < args.stop_rewire_epoch:
            enable_print =  (i % args.print_freq == 0)
            for st in [x for x in model.modules() if isinstance(x,SparseTensor)]:
                if (st.get_sparsity()[0] != st.s_tensor.numel()):
                    pruned_indices = st.prune_sign_change(not(args.big_new_weights),enable_print = enable_print)
                    st.grow_random(None,pruned_indices,enable_print = enable_print)
        elif args.rewire and (i + current_iteration != 0) and (i + current_iteration) % rewire_period == 0 and epoch < args.stop_rewire_epoch:
            print('rewiring at iteration ' + repr(i+current_iteration))
            n_pruned_indices = np.zeros(len(all_sparse_tensors))
            all_pruned_indices = []            
            for i,st in enumerate(all_sparse_tensors):
              if args.threshold_prune:
                  pruned_indices = st.prune_threshold(threshold,not(args.big_new_weights))
              else:
                  pruned_indices = st.prune_small_connections(args.rewire_fraction,not(args.big_new_weights))
              all_pruned_indices.append(pruned_indices)
              n_pruned_indices[i] = pruned_indices.size(0)

            if args.grow_across_layers:
                sparse_tensor_nonzeros = np.array([x.mask.sum() for x in all_sparse_tensors])            

                pruned_tensor_fraction = n_pruned_indices / sparse_tensor_nonzeros


                adjusted_pruned_tensor_fraction = np.ones_like(pruned_tensor_fraction) * pruned_tensor_fraction.mean()
                adjusted_pruned_tensor_fraction = np.clip(adjusted_pruned_tensor_fraction,0.0,1.0)


                n_grown = 0
                grow_backs = adjusted_pruned_tensor_fraction * sparse_tensor_nonzeros
                grow_backs /= grow_backs.sum()
                

                while n_grown < n_pruned_indices.sum():
                      weights_to_grow = n_pruned_indices.sum() - n_grown
                      grow_backs_count = grow_backs * weights_to_grow
                      grow_backs_count = np.round(grow_backs_count).astype('int32')

                      rounding_discrepency = weights_to_grow - grow_backs_count.sum()
                      if rounding_discrepency != 0:
                        rnd_idx = np.random.randint(len(grow_backs_count))    
                        grow_backs_count[rnd_idx] = max(0,grow_backs_count[rnd_idx] + rounding_discrepency)
                            
                      print(list(zip(grow_backs_count,n_pruned_indices)))

                      for i,st in enumerate(all_sparse_tensors):
                            n_grown += st.grow_random(None,all_pruned_indices[i],n_to_add = grow_backs_count[i])

                print('n_grown : {} , n_pruned : {}'.format(n_grown,n_pruned_indices.sum()))
                if n_grown != n_pruned_indices.sum():
                      print('*********** discrepency between pruned and grown')
            
            else:
                for i,st in enumerate(all_sparse_tensors):
                      st.grow_random(None,all_pruned_indices[i])

            if args.threshold_prune:
                if n_pruned_indices.sum() > 1.1 * args.n_realloc_params:
                    threshold /= 2.0
                elif n_pruned_indices.sum() < 0.9 * args.n_realloc_params:
                    threshold *= 2.0
                print('threshold is ' + repr(threshold))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            print('elapsed time ' + repr(time.time() - epoch_start_time))
                              

    # log to TensorBoard

    return current_iteration + len(train_loader),threshold

def validate(val_loader, model, criterion, epoch,pre_text):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1,prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(pre_text + ' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    # log to TensorBoard
    return top1.avg,top5.avg,losses.avg


def save_checkpoint(state,  filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def get_schedule_val(schedule,query):
    val = list(schedule[-1].values())[0]
    for i,entry in enumerate(schedule):
        if query < list(entry)[0]:
            val = list(schedule[i-1].values())[0]
            break
    return val
        
def adjust_learning_rate(optimizer, epoch,schedule):
    """Sets the learning rate to the initial LR divided by 5 at 30th, 60th and 90th epochs"""
    lr = get_schedule_val(schedule,epoch)

    print('setting learning rate to ' + repr(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == '__main__':
    main()
    
