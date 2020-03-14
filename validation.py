import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    op_losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, proposal, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        # if not opt.no_cuda:
            # targets = targets.cuda(async=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            proposal = Variable(proposal.float())
            outputs, op_loss = model(inputs, proposal, training=False)
        # targets = Variable(targets, volatile=True)
        # targets = targets.data.cpu()
        
        outputs = outputs.data.cpu()
        cls_loss = criterion(outputs, targets)
        op_loss = op_loss.data.cpu()
        loss = cls_loss + opt.op_loss_weight * op_loss.mean()
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        cls_losses.update(cls_loss.data.cpu()[0], inputs.size(0))
        op_losses.update(op_loss.data.cpu()[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % opt.log_step == 0:
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

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg
