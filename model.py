import torch
from torch import nn
import torchvision

from models.i3res import I3ResNet
from models.GCN import GCN


def generate_model(opt):
    assert opt.model_depth in [50, 101, 152]

    if opt.model_depth == 50:
        model = torchvision.models.resnet50(pretrained=True)
        model = I3ResNet(model, opt.sample_duration)
    elif opt.model_depth == 101:
        model = torchvision.models.resnet101(pretrained=True)
        model = I3ResNet(model, opt.sample_duration)
    elif opt.model_depth == 152:
        model = torchvision.models.resnet152(pretrained=True)
        model = I3ResNet(model, opt.sample_duration)
    
    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        assert opt.arch == pretrain['arch'], 'Unmatched model from pretrained path.'
        state_dict = pretrain['state_dict']
        local_state = model.state_dict()
        for name, param in local_state.items():
            if 'fc' not in name:
                key = 'module.' + name
                if key in state_dict:
                    input_param = state_dict[key].data
                    param.copy_(input_param)
    
    base_model = model
    model = GCN(opt.n_box_per_frame, opt.step_per_layer, opt.n_classes, opt.basenet_fixed_layers)
    model.create_architecture(base_model)
    del base_model
    
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        parameters = []
        arch_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'arch_weights' not in key:
                    parameters += [{'params':[value]}]
                else:
                    arch_parameters += [{'params':[value]}]
        return model, parameters, arch_parameters
    
    else:
        parameters = []
        arch_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'arch_weights' not in key:
                    parameters += [{'params':[value]}]
                else:
                    arch_parameters += [{'params':[value]}]
        return model, parameters, arch_parameters

