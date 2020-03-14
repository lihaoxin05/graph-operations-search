import torch
import torch.nn as nn
import torch.nn.functional as F

PRIMITIVES = [
    'none',
    'skip',
    'comb1',
    'comb2',
    'comb3'
]


NODES_OPS = {
    'none': lambda channel, num_nodes: Zero(),
    'skip': lambda channel, num_nodes: Skip(),
    'feat_aggr' : lambda channel, num_nodes: Feat_aggr(channel),
    'diff_prop' : lambda channel, num_nodes: Diff_prop(channel),
    'temp_conv' : lambda channel, num_nodes: Temp_conv(channel),
    'back_incor' : lambda channel, num_nodes: Back_incor(channel),
    'node_att': lambda channel, num_nodes: Node_att(channel, num_nodes),
    'comb1': lambda channel, num_nodes: Comb1(channel, num_nodes),
    'comb2': lambda channel, num_nodes: Comb2(channel, num_nodes),
    'comb3': lambda channel, num_nodes: Comb3(channel, num_nodes),
    'comb4': lambda channel, num_nodes: Comb4(channel, num_nodes),
    'comb5': lambda channel, num_nodes: Comb5(channel, num_nodes),
    'comb6': lambda channel, num_nodes: Comb6(channel, num_nodes)
}


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, local_feat, global_feat, pos):
        return local_feat.mul(0.)


class Skip(nn.Module):

    def __init__(self):
        super(Skip, self).__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, local_feat, global_feat, pos):
        return self.dropout(local_feat)


class Feat_aggr(nn.Module):

    def __init__(self, channel):
        super(Feat_aggr, self).__init__()
        self.adj_weight = nn.Linear(channel, channel, bias=False)
        self.feature_affine = nn.Linear(channel, channel)
        self.ln = nn.LayerNorm(channel)
        self.activate = nn.LeakyReLU(inplace=True)
        
        nn.init.eye_(self.adj_weight.weight)
        nn.init.kaiming_normal_(self.feature_affine.weight)
        nn.init.constant_(self.feature_affine.bias, 0)

    def forward(self, local_feat, global_feat, pos):
        batch = local_feat.shape[0]
        time = local_feat.shape[1]
        nodes = local_feat.shape[2]

        local_feat = local_feat.view(batch, time*nodes, -1).contiguous()

        norm_feat = F.normalize(local_feat, 2, -1)
        A = torch.einsum('bnc,cc,bcm->bnm', (norm_feat, self.adj_weight.weight, torch.transpose(norm_feat, 1, 2)))
        A = 5 * (A - torch.max(A, -1, keepdim=True)[0])
        A = F.softmax(A, -1)

        local_feat = self.feature_affine(local_feat)
        local_feat = torch.einsum('bij,bjk->bik', (A, local_feat))
        local_feat = self.activate(self.ln(local_feat))
        local_feat = local_feat.view(batch, time, nodes, -1).contiguous()

        return local_feat


class Diff_prop(nn.Module):

    def __init__(self, channel):
        super(Diff_prop, self).__init__()
        self.adj_weight = nn.Linear(channel, channel, bias=False)
        self.feature_affine = nn.Linear(channel, channel)
        self.ln = nn.LayerNorm(channel)
        self.activate = nn.LeakyReLU(inplace=True)
        
        nn.init.eye_(self.adj_weight.weight)
        nn.init.kaiming_normal_(self.feature_affine.weight)
        nn.init.constant_(self.feature_affine.bias, 0)

    def forward(self, local_feat, global_feat, pos):
        batch = local_feat.shape[0]
        time = local_feat.shape[1]
        nodes = local_feat.shape[2]

        local_feat = local_feat.view(batch, time*nodes, -1).contiguous()

        norm_feat = F.normalize(local_feat, 2, -1)
        A = torch.einsum('bnc,cc,bcm->bnm', (norm_feat, self.adj_weight.weight, torch.transpose(norm_feat, 1, 2)))
        A = torch.exp(5*(A - torch.max(A, -1, keepdim=True)[0]))
        A = (1.0 - torch.stack([torch.eye(A.size(1))]*A.size(0)).cuda()) * A
        A = F.normalize(A, 1, -1)

        n = local_feat.shape[1]
        diff_feat = torch.stack([local_feat]*n, 2) - torch.stack([local_feat]*n, 1)
        diff_feat = self.feature_affine(diff_feat.view(diff_feat.shape[0], n*n, -1).contiguous()).view(diff_feat.shape[0], n, n, -1).contiguous()
        diff_feat = torch.sum(torch.unsqueeze(A, -1) * diff_feat, 2)
        local_feat = self.activate(self.ln(diff_feat))
        local_feat = local_feat.view(batch, time, nodes, -1).contiguous()

        return local_feat
        
        
class Temp_conv(nn.Module):

    def __init__(self, channel):
        super(Temp_conv, self).__init__()
        self.adj_weight = nn.Linear(channel, channel, bias=False)
        self.temp_conv1 = nn.Conv1d(channel, channel, 7, groups=channel)
        self.ln = nn.LayerNorm(channel)
        self.activate = nn.LeakyReLU(inplace=True)
        
        nn.init.eye_(self.adj_weight.weight)
        nn.init.kaiming_normal_(self.temp_conv1.weight)
        nn.init.constant_(self.temp_conv1.bias, 0)

    def forward(self, local_feat, global_feat, pos):
        batch = local_feat.shape[0]
        time = local_feat.shape[1]
        nodes = local_feat.shape[2]

        local_feat = local_feat.view(batch, time*nodes, -1).contiguous()

        norm_feat = F.normalize(local_feat, 2, -1)
        A = torch.einsum('bnc,cc,bcm->bnm', (norm_feat, self.adj_weight.weight, torch.transpose(norm_feat, 1, 2)))
        A = A.view(batch, time*nodes, time, nodes).contiguous()
        _, top_A = torch.max(A, dim=-1)

        n = local_feat.shape[1]
        feat_channel = local_feat.shape[-1]
        local_feat = torch.stack([local_feat]*n, 1)
        local_feat = local_feat.view(batch, n, time, nodes, -1)
        feat_index = torch.stack([top_A.unsqueeze(-1)]*feat_channel, dim=-1)
        top_feat = torch.gather(local_feat, 3, feat_index).view(batch*n, time, feat_channel).contiguous()
        top_feat = top_feat.permute(0,2,1).contiguous()

        local_feat = self.temp_conv1(top_feat)
        local_feat = self.activate(self.ln(local_feat.permute(0,2,1).contiguous())).permute(0,2,1).contiguous()
        local_feat = local_feat.mean(-1)
        local_feat = local_feat.view(batch, time, nodes, -1).contiguous()

        return local_feat


class Back_incor(nn.Module):

    def __init__(self, channel):
        super(Back_incor, self).__init__()
        self.adj_weight = nn.Linear(channel, channel, bias=False)
        self.feature_affine = nn.Linear(channel+49, channel)
        self.ln = nn.LayerNorm(channel)
        self.activate = nn.LeakyReLU(inplace=True)
        
        nn.init.eye_(self.adj_weight.weight)
        nn.init.kaiming_normal_(self.feature_affine.weight)
        nn.init.constant_(self.feature_affine.bias, 0)

    def forward(self, local_feat, global_feat, pos):
        batch = local_feat.shape[0]
        time = local_feat.shape[1]
        nodes = local_feat.shape[2]
        bg_bins = global_feat.shape[2]

        local_feat = local_feat.view(batch*time, nodes, -1).contiguous()
        global_feat = global_feat.view(batch*time, bg_bins, -1).contiguous()

        norm_feat = F.normalize(local_feat, 2, -1)
        norm_global_feat = F.normalize(global_feat, 2, -1)
        A_raw = torch.einsum('bnc,cc,bcm->bnm', (norm_feat, self.adj_weight.weight, torch.transpose(norm_global_feat, 1, 2)))
        A = 5 * (A_raw - torch.max(A_raw, -1, keepdim=True)[0])
        A = F.softmax(A, -1)
        local_feat = torch.cat((torch.einsum('bnm,bmc->bnc', (A, global_feat)), A_raw), dim=-1)
        local_feat = self.feature_affine(local_feat)
        local_feat = self.activate(self.ln(local_feat))
        local_feat = local_feat.view(batch, time, nodes, -1).contiguous()

        return local_feat
        
        
class Node_att(nn.Module):

    def __init__(self, channel, num_nodes):
        super(Node_att, self).__init__()
        self.node_att = nn.Linear(num_nodes+9, 1)
        self.activate = nn.Sigmoid()
        
        nn.init.kaiming_normal_(self.node_att.weight)
        nn.init.constant_(self.node_att.bias, 0)

    def forward(self, local_feat, global_feat, pos):
        batch = local_feat.shape[0]
        time = local_feat.shape[1]
        nodes = local_feat.shape[2]

        local_feat = local_feat.view(batch, time*nodes, -1).contiguous()
        pos = pos.view(batch, time*nodes, -1).contiguous()

        norm_feat = F.normalize(local_feat, 2, -1)
        A = torch.einsum('bnc,bcm->bnm', (norm_feat, torch.transpose(norm_feat, 1, 2)))
        attention_source = torch.cat((A, pos), -1)
        att_weight = self.activate(self.node_att(attention_source).view(batch, time, nodes, -1).contiguous())
        
        local_feat = local_feat.view(batch, time, nodes, -1).contiguous()
        local_feat = local_feat * att_weight

        return local_feat
        
        
class Comb1(nn.Module):

    def __init__(self, channel, num_nodes):
        super(Comb1, self).__init__()
        self.base_op = Diff_prop(channel)
        self.conv = Feat_aggr(channel)
        self.att = Node_att(channel, num_nodes)
        
    def forward(self, local_feat, global_feat, pos):
        local_feat = self.base_op(local_feat, global_feat, pos)
        local_feat = self.conv(local_feat, global_feat, pos)
        local_feat = self.att(local_feat, global_feat, pos)
        return local_feat
        
        
class Comb2(nn.Module):

    def __init__(self, channel, num_nodes):
        super(Comb2, self).__init__()
        self.base_op = Temp_conv(channel)
        self.conv = Feat_aggr(channel)
        self.att = Node_att(channel, num_nodes)
        
    def forward(self, local_feat, global_feat, pos):
        local_feat = self.base_op(local_feat, global_feat, pos)
        local_feat = self.conv(local_feat, global_feat, pos)
        local_feat = self.att(local_feat, global_feat, pos)
        return local_feat
        
        
class Comb3(nn.Module):

    def __init__(self, channel, num_nodes):
        super(Comb3, self).__init__()
        self.base_op = Back_incor(channel)
        self.conv = Feat_aggr(channel)
        self.att = Node_att(channel, num_nodes)
        
    def forward(self, local_feat, global_feat, pos):
        local_feat = self.base_op(local_feat, global_feat, pos)
        local_feat = self.conv(local_feat, global_feat, pos)
        local_feat = self.att(local_feat, global_feat, pos)
        return local_feat
        
        
class Comb4(nn.Module):

    def __init__(self, channel, num_nodes):
        super(Comb4, self).__init__()
        self.base_op1 = Feat_aggr(channel)
        self.base_op2 = Diff_prop(channel)
        self.reduce = nn.Linear(2*channel, channel)
        self.activate = nn.LeakyReLU(inplace=True)
        self.att = Node_att(channel, num_nodes)
        
    def forward(self, local_feat, global_feat, pos):
        local_feat_1 = self.base_op1(local_feat, global_feat, pos)
        local_feat_2 = self.base_op2(local_feat, global_feat, pos)
        local_feat = self.activate(self.reduce(torch.cat((local_feat_1, local_feat_2), -1)))
        local_feat = self.att(local_feat, global_feat, pos)
        return local_feat   


class Comb5(nn.Module):

    def __init__(self, channel, num_nodes):
        super(Comb5, self).__init__()
        self.base_op1 = Temp_conv(channel)
        self.base_op2 = Back_incor(channel)
        self.reduce = nn.Linear(2*channel, channel)
        self.activate = nn.LeakyReLU(inplace=True)
        self.att = Node_att(channel, num_nodes)
        
    def forward(self, local_feat, global_feat, pos):
        local_feat_1 = self.base_op1(local_feat, global_feat, pos)
        local_feat_2 = self.base_op2(local_feat, global_feat, pos)
        local_feat = self.activate(self.reduce(torch.cat((local_feat_1, local_feat_2), -1)))
        local_feat = self.att(local_feat, global_feat, pos)
        return local_feat
        

class Comb6(nn.Module):

    def __init__(self, channel, num_nodes):
        super(Comb6, self).__init__()
        self.base_op1 = Feat_aggr(channel)
        self.base_op2 = Back_incor(channel)
        self.reduce = nn.Linear(2*channel, channel)
        self.activate = nn.LeakyReLU(inplace=True)
        self.att = Node_att(channel, num_nodes)
        
    def forward(self, local_feat, global_feat, pos):
        local_feat_1 = self.base_op1(local_feat, global_feat, pos)
        local_feat_2 = self.base_op2(local_feat, global_feat, pos)
        local_feat = self.activate(self.reduce(torch.cat((local_feat_1, local_feat_2), -1)))
        local_feat = self.att(local_feat, global_feat, pos)
        return local_feat 

  
class MixedOp(nn.Module):

    def __init__(self, channel, num_nodes):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = NODES_OPS[primitive](channel, num_nodes)
            self._ops.append(op)

    def forward(self, local_feat, global_feat, pos, weights):
        out = []
        weights = torch.chunk(weights, weights.shape[-1], -1)
        for w, op in zip(weights, self._ops):
            out_feat = op(local_feat, global_feat, pos)
            out.append(w.unsqueeze(-2).unsqueeze(-2) * out_feat)
        return torch.sum(torch.stack(out, -1), -1)

