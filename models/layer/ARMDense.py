import math

import torch
import torch.nn as nn
from config import opt
import torch.nn.functional as F


class ARMDense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_decay=1e-4, lamba=0.001, droprate_init=.5,
                 local_rep=True, **kwargs):
        super(ARMDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decay = weight_decay
        self.lamba = lamba
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features, ))
        self.z_phi = nn.Parameter(torch.Tensor(in_features))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))

        self.floatTensor = torch.FloatTensor if not opt.use_gpu else torch.cuda.FloatTensor
        self.droprate_init = droprate_init
        self.u = None
        self.forward_mode = True
        self.local_rep = local_rep
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weights, mode='fan_out')
        self.z_phi.data.normal_((math.log(1 - self.droprate_init) - math.log(self.droprate_init)) / opt.k, 1e-2 / opt.k)
        if self.use_bias:
            self.bias.data.fill_(0)

    def update_phi_gradient(self, f1, f2):
        # only deal with first part of gradient
        # regularization part will be handled by pytorch
        k = opt.k
        if opt.ar:
            e = k * (f2 * (1 - 2 * self.u)).mean(dim=0)
        else:
            e = k * ((f1 - f2) * (self.u - .5)).mean(dim=0)
        self.z_phi.grad = e

    def regularization(self):
        ''' similar with L0 paper'''
        if opt.hardsigmoid:
            pi = F.hardtanh(opt.k * self.z_phi / 7. + .5, 0, 1)
        else:
            pi = torch.sigmoid(opt.k * self.z_phi)

        l0 = self.lamba * pi.sum() * self.out_features
        logpw_col = torch.sum(.5 * self.weight_decay * self.weights.pow(2), 1)
        logpw = torch.sum(pi * logpw_col)
        logpb = 0 if not self.use_bias else torch.sum(.5 * self.weight_decay * self.bias.pow(2))
        l2 = logpw + logpb
        return l0 + l2

    def count_expected_flops_and_l0(self):
        '''Measures the expected floating point operations (FLOPs) and the expected L0 norm
        dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        + the bias addition for each neuron
        total_flops = (2 * in_features - 1) * out_features + out_features'''

        if opt.hardsigmoid:
            ppos = F.hardtanh(opt.k * self.z_phi / 7. + .5, 0, 1).sum()
        else:
            ppos = torch.sigmoid(opt.k * self.z_phi).sum()
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features

        if not opt.ar:
            expected_flops *= 2
            expected_l0 *= 2
        return expected_flops.data, expected_l0.data

    def sample_z(self, batch_size):
        if opt.hardsigmoid:
            pi = F.hardtanh(opt.k * self.z_phi / 7. + .5, 0, 1).detach()
        else:
            pi = torch.sigmoid(opt.k * self.z_phi).detach()
        if self.forward_mode:
            z = self.floatTensor(batch_size, self.in_features).zero_()
            if self.training:
                if self.local_rep:
                    self.u = self.floatTensor(self.in_features).uniform_(0, 1).expand(batch_size, self.in_features)
                else:
                    self.u = self.floatTensor(batch_size, self.in_features).uniform_(0, 1)

                z[self.u < pi.expand(batch_size, self.in_features)] = 1
                if opt.use_t_in_training:
                    z[pi.expand(batch_size, self.in_features) < opt.t] = 0
                self.train_z = z
            else:
                z[self.z_phi.expand(batch_size, self.in_features) > 0] = 1

                if opt.use_t_in_testing:
                    z = pi.expand(batch_size, self.in_features)
                    z[z < opt.t] = 0
                self.test_z = z
        else:
            pi2 = 1 - pi
            if self.u is None:
                raise Exception('Forward pass first')
            z = self.floatTensor(self.u.size()).zero_()
            z[self.u > pi2.expand(batch_size, self.in_features)] = 1
            if opt.use_t_in_training:
                z[pi.expand(batch_size, self.in_features) < opt.t] = 0
        return z

    def forward(self, input):
        """ forward for fc """
        xin = input.mul(self.sample_z(input.size(0)))
        output = xin.mm(self.weights)
        if self.use_bias:
            output += self.bias
        return output

    def masked_weight(self):
        return self.weights * self.test_z[0].reshape(self.in_features, 1)

    def activated_neurons(self):
        return (self.test_z > 0).sum() / self.test_z.size(0)

    def expected_activated_neurons(self):
        return (self.train_z > 0).sum() / self.train_z.size(0)

    def __repr__(self):
        s = ('{name}({in_features} -> {out_features},'
             'lamba={lamba}, weight_decay={weight_decay}, ')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
