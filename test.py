import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
import json

from utils import AverageMeter, calculate_accuracy


def calculate_video_results(output_buffer, video_id, test_results, label, criterion, opt):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0, keepdim=True)
    label = torch.unsqueeze(label, 0)
    loss = criterion(average_scores, label)
    acc = calculate_accuracy(average_scores, label)
    sorted_scores, locs = torch.topk(F.softmax(average_scores,-1), k=10)

    video_results = []
    video_results.append({
        'groundtruth': label.numpy().tolist(),
        'label': locs[0].numpy().tolist(),
        'score': sorted_scores[0].numpy().tolist()
    })
    test_results['results'][video_id] = video_results
    
    return loss, acc, locs[0][0]


def test(data_loader, model, criterion, opt):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    previous_label = ''
    test_results = {'results': {}}
    confusion_matrix = np.zeros((opt.n_classes, opt.n_classes))                                                                            
    for i, (inputs, proposal, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            inputs = Variable(inputs)
            proposal = Variable(proposal.float())
            outputs, _ = model(inputs, proposal, training=False)

        for j in range(outputs.size(0)):
            if (not (i == 0 and j == 0) and targets[0][j] != previous_video_id) or (i == len(data_loader)-1 and j == outputs.size(0)-1):
                loss, acc, locs = calculate_video_results(output_buffer, previous_video_id,
                                        test_results, previous_label, criterion, opt)
                confusion_matrix[previous_label, locs] += 1.0                                             
                output_buffer = []
                losses.update(loss)
                accuracies.update(acc)
            elif not (i == 0 and j == 0):
                assert targets[1][j] == previous_label
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[0][j]
            previous_label = targets[1][j]

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % opt.log_step == 0:
            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies), flush=True)
    print('Loss {loss.avg:.4f}\tAcc {acc.avg:.4f}'.format(loss=losses, acc=accuracies), flush=True)
    if opt.save_test_result:
        with open(os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)), 'w') as f:
            json.dump(test_results, f)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, -1, keepdims=True)
    np.save('{}/confusion_matrix.npy'.format(opt.result_path), confusion_matrix)                                                                                 
