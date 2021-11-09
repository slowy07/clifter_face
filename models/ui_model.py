import torch
from torch.autorgrad import Variable
from collections import OrderedDict
import numpy as np
import os
from PIL import Image
import util.util as util
from .base_model import BaseModel
from . import networks

class UIModel(BaseModel):
    def name(self):
        return "UIModel"

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.use_features = opt.instance_feat or opt.label_feat

        netG_input_nc = opt.label_nc
        if not opt.not_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num

        self.netG = networks.define_G(netG_input_nc, opt.output_nc,
            opt.netG, opt.n_downsample_global, opt.n_blocks_global,
            opt.n_local_enhancers, opt.n_blocks_local, gpu_ids = self.gpu_ids
        )

        self.load_network(self.netG, "G", opt.which_epoch)

        print("network initialized")

    def toTensor(self, img, normalize=False):
        tensor = torch.from_numpy(np.array(img, np.int32, copy=False))
        tensor = tensor.view(1, img.size[1], img.size[0], len(img.mode))
        tensor = tensor.transpose(1,2).transpose(1, 3).contiguous()
        if normalize:
            return (tensor.float() / 255.0 - 0.5) / 0.5
        
        return tensor.float()