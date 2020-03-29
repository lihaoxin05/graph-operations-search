import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from roi_pooling.modules.roi_pool import _RoIPooling
from roi_align.modules.roi_align import RoIAlign
from .operations import PRIMITIVES, MixedOp
from .boxes_handling import *

class NASGCN(nn.Module):
    def __init__(self, n_box_per_frame, step_per_layer, num_classes):
        super(NASGCN, self).__init__()
        self.n_box_per_frame = n_box_per_frame
        
        ## define roialign
        self.roi_pool = RoIAlign(7, 7, 1.0/32.0)
        ## define base feature transformation layer
        self.base_feat_transform = nn.Linear(2048, 256)
        self.activate = nn.LeakyReLU(inplace=True)
        ## define graph operations search layer
        k = sum(1 for i in range(step_per_layer) for n in range(1+i))
        self.num_ops = len(PRIMITIVES)
        self.arch_weights = nn.Linear(256, self.num_ops*k)
        self.graph_layer = NAS_layer(256, 16*n_box_per_frame, step_per_layer)
        ## define classification layer
        self.cls_fc = nn.Linear(1024, num_classes)
        
    def forward(self, video, box, training=True):
        batch_size = video.size(0)
        time_length = video.size(2)
        height = video.size(3)
        width = video.size(4)

        video = video.data #shape: (b*3*T*H*W)
        box = box.data #shape: (b*T*n*4)
        
        ## feed video data to base model to obtain base feature map
        base_feat = self.CNN_base(video) #shape: (b*C*t*h*w)
        final_time_length = base_feat.size(2)
        h = base_feat.size(3)
        w = base_feat.size(4)
        
        ## base feature map transform to low dimension features
        base_feat = self.activate(self.base_feat_transform(base_feat.permute(0,2,3,4,1)).permute(0,1,4,2,3).view(batch_size*final_time_length, -1, h, w).contiguous()) #shape: (bt*c*h*w)
        c = base_feat.size(1)
        
        ## global nodes
        global_nodes = base_feat.view(batch_size, final_time_length, c, h*w).permute(0,1,3,2).contiguous() #shape: (b*t*m*c)
        
        ## preprossing box
        box = box[:,::time_length//final_time_length,:,:].contiguous() #shape: (b*t*n*4)
        pos = local_layout(box.clone(), width, height).contiguous()
        box = add_index(box.view(box.size(0)*box.size(1), box.size(2), box.size(3)).contiguous()).contiguous() #shape: (btn*5)
        ## roi align
        pooled_feat = self.roi_pool(base_feat, box)
        ## local nodes
        local_nodes = torch.squeeze(torch.nn.AvgPool2d(7)(pooled_feat)) #shape: (btn*c)
        local_nodes = local_nodes.view(batch_size, final_time_length, self.n_box_per_frame, -1).contiguous() #shape: (b*t*n*c)
        
        ## graph operations search
        ops_weights = self.arch_weights(global_nodes.mean(1).mean(1)).view(batch_size, -1, self.num_ops)
        local_nodes, op_loss = self.graph_layer(local_nodes, global_nodes, pos, ops_weights, training)
        
        ## classification
        feat = torch.cat((local_nodes.mean(1).mean(1), global_nodes.mean(1).mean(1)), dim=-1)
        score = self.cls_fc(feat)

        return score, op_loss

    def _init_weights(self):
        nn.init.kaiming_normal_(self.base_feat_transform.weight)
        nn.init.constant_(self.base_feat_transform.bias, 0)
        
        nn.init.constant_(self.arch_weights.weight, 0)
        
        nn.init.kaiming_normal_(self.cls_fc.weight)
        nn.init.constant_(self.cls_fc.bias, 0)

    def create_architecture(self, base_model):
        self._init_modules(base_model)
        self._init_weights()
        
    
class NAS_layer(nn.Module):
    def __init__(self, local_channel, num_nodes, steps):
        super(NAS_layer, self).__init__()
        self._ops = nn.ModuleList()
        self.steps = steps
        for i in range(steps):
            for j in range(1+i):
                op = MixedOp(local_channel, num_nodes)
                self._ops.append(op)


    def forward(self, local_feat, global_feat, pos, ops_weights, training=True):
        batch = global_feat.size(0)
        time = global_feat.size(1)
        box_per_frame = local_feat.size(2)
        
        ## weights
        if training:
            ops_weights = F.softmax(ops_weights - torch.max(ops_weights, -1, keepdim=True)[0], -1)
        else:
            ops_weights_max = torch.max(ops_weights, -1, keepdim=True)[0]
            ops_weight = torch.where(ops_weights < ops_weights_max, torch.zeros_like(ops_weight), torch.ones_like(ops_weight))
        var_loss = torch.var(torch.sum(ops_weights, 1), 1).mean()
        ## MixOps
        states = [local_feat]
        offset = 0
        for i in range(self.steps):
            l_f = []
            for j, h in enumerate(states):
                tmp_l_f = self._ops[offset+j](h, global_feat, pos, ops_weights[:,offset+j,:])
                l_f.append(tmp_l_f)
            l_f = torch.sum(torch.stack(l_f, -1), -1)
            offset += len(states)
            states.append(l_f)

        local_feat = []
        for s in states[1:]:
            local_feat.append(s)
        local_feat = torch.cat(local_feat, -1)
        
        return local_feat, var_loss
        
