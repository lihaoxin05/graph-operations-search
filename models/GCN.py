import torch
from .NGCN import NASGCN


class GCN(NASGCN):
  def __init__(self, n_box_per_frame, step_per_layer, num_classes, basenet_fixed_layers):
    self.basenet_fixed_layers = basenet_fixed_layers
    
    NASGCN.__init__(self, n_box_per_frame, step_per_layer, num_classes)

  def _init_modules(self, base_model):
    # Build base_model
    self.CNN_base = torch.nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool1, base_model.layer1, base_model.maxpool2, base_model.layer2, base_model.layer3, base_model.layer4)

    # Fix blocks
    for p in self.CNN_base[0].parameters(): p.requires_grad=False
    for p in self.CNN_base[1].parameters(): p.requires_grad=False

    if self.basenet_fixed_layers >= 4:
      for p in self.CNN_base[8].parameters(): p.requires_grad=False
    if self.basenet_fixed_layers >= 3:
      for p in self.CNN_base[7].parameters(): p.requires_grad=False
    if self.basenet_fixed_layers >= 2:
      for p in self.CNN_base[6].parameters(): p.requires_grad=False
    if self.basenet_fixed_layers >= 1:
      for p in self.CNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1 or classname.find('bn') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.CNN_base[0].apply(set_bn_fix)
    self.CNN_base[1].apply(set_bn_fix)
    if self.basenet_fixed_layers >= 4:
      self.CNN_base[8].apply(set_bn_fix)
    if self.basenet_fixed_layers >= 3:
      self.CNN_base[7].apply(set_bn_fix)
    if self.basenet_fixed_layers >= 2:
      self.CNN_base[6].apply(set_bn_fix)
    if self.basenet_fixed_layers >= 1:
      self.CNN_base[4].apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    torch.nn.Module.train(self, mode)
    if mode:
      self.CNN_base.eval()
      if self.basenet_fixed_layers < 4:
        self.CNN_base[8].train()
      if self.basenet_fixed_layers < 3:
        self.CNN_base[7].train()
      if self.basenet_fixed_layers < 2:
        self.CNN_base[6].train()
      if self.basenet_fixed_layers < 1:
        self.CNN_base[4].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1 or classname.find('bn') != -1:
          m.eval()

        self.CNN_base.apply(set_bn_eval)