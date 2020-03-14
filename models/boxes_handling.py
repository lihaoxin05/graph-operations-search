import numpy as np
import skimage
import torch
    
def add_index(box):
    # input: box with shape (b,n,4): [c1,r1,c2,r2]
    # output: box with shape (bn,5): [ind,c1,r1,c2,r2]
    ind = np.array(range(0, box.size(0)))[:,np.newaxis]
    ind = np.repeat(ind, box.size(1), axis=1)
    ind = ind.reshape(-1,1)
    box = box.view(-1,4)
    box = torch.cat((torch.from_numpy(ind).float().cuda(), box), dim=-1).contiguous()
    return box

def corner2cs(box):
    # input: box with shape (b,n,4): [c1,r1,c2,r2]
    # output: center and scale with shape (b,n,4): [c0,r0,sc,sr]
    cs = box.clone()
    cs[:,:,0] = (box[:,:,0] + box[:,:,2]) / 2
    cs[:,:,1] = (box[:,:,1] + box[:,:,3]) / 2
    cs[:,:,2] = -box[:,:,0] + box[:,:,2]
    cs[:,:,3] = -box[:,:,1] + box[:,:,3]
    return cs

def box_norm(box, width, height):
    # input: box with shape (b,n,4): [c1,r1,c2,r2] or [c0,r0,sc,sr]
    #        width, height: normalize with width and height
    # output: center and scale with shape (b,n,4): [c1,r1,c2,r2] or [c0,r0,sc,sr]
    box[:,:,0] = box[:,:,0] / width
    box[:,:,1] = box[:,:,1] / height
    box[:,:,2] = box[:,:,2] / width
    box[:,:,3] = box[:,:,3] / height
    return box
    
def local_layout(layout, width, height):
    # input: layout with shape (b,t,n,4): [c1,r1,c2,r2]
    # output: layout feature with shape (b,t,n,9): [c1,r1,c2,r2,c0,r0,sc,sr,t]
    batches = layout.size(0)
    time = layout.size(1)
    nodes = layout.size(2)
    layout = box_norm(layout.contiguous().view(batches, time*nodes, -1).contiguous().clone(), width, height)
    cs = corner2cs(layout)
    timedim = torch.arange(time, dtype=torch.float).cuda().view(time, 1)
    timedim = timedim / time
    timedim = torch.stack([torch.stack([timedim]*nodes, dim=1)]*batches).view(batches, time*nodes, -1)
    layout = torch.cat((layout, cs, timedim), dim=-1).view(batches, time, nodes, -1)
    return layout
    
    
