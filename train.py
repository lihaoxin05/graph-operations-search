import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, arch_optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    op_losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, proposal, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        proposal = Variable(proposal.float())
        targets = Variable(targets)
        
        outputs, op_loss = model(inputs, proposal, training=True)
        cls_loss = criterion(outputs, targets)
        loss = cls_loss + opt.op_loss_weight * op_loss.mean()
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data.cpu()[0], inputs.size(0))
        cls_losses.update(cls_loss.data.cpu()[0], inputs.size(0))
        op_losses.update(op_loss.data.cpu()[0], inputs.size(0))
        accuracies.update(acc.cpu(), inputs.size(0))

        if i % 2 == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            arch_optimizer.zero_grad()
            loss.backward()
            arch_optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % opt.log_step == 0:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'cls_loss': cls_losses.val,
                'op_loss': op_losses.val,
                'acc': accuracies.val,
                'lr': max([optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))])
            })
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLSLoss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'OPLoss {op_loss.val:.4f} ({op_loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      cls_loss=cls_losses,
                      op_loss=op_losses,
                      acc=accuracies), flush=True)

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': max([optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))])
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
