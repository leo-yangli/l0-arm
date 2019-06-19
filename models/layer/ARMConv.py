import math

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair as pair
import torch.nn.functional as F
from config import opt


class ArmConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 weight_decay=1.e-4,
                 lamba=0.1 / 6e5, droprate_init=.5, local_rep=True, **kwargs):
        super(ArmConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.weight_decay = weight_decay
        self.lamba = lamba
        self.floatTensor = torch.FloatTensor if not opt.use_gpu else torch.cuda.FloatTensor
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        self.weights = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.z_phi = Parameter(torch.Tensor(out_channels))
        self.dim_z = out_channels
        self.input_shape = None
        self.u = torch.Tensor(self.dim_z).uniform_(0, 1)
        self.droprate_init = droprate_init
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
        # only handle first part of phi's gradient
        k = opt.k
        if opt.ar:
            e = k * (f2 * (1 - 2 * self.u)).mean(dim=0)
        else:
            e = k * ((f1 - f2) * (self.u - .5)).mean(dim=0)
        self.z_phi.grad = e

    def regularization(self):
        # similar with L0 paper
        if opt.hardsigmoid:
            pi = F.hardtanh(opt.k * self.z_phi / 7. + .5, 0, 1)
        else:
            pi = torch.sigmoid(opt.k * self.z_phi)

        l0 = self.lamba * pi.sum() * self.weights.view(-1).size()[0] / self.weights.size(0)
        wd_col = .5 * self.weight_decay * self.weights.pow(2).sum(3).sum(2).sum(1)
        wd = torch.sum(pi * wd_col)
        wb = 0 if not self.use_bias else torch.sum(pi * (.5 * self.weight_decay * self.bias.pow(2)))
        l2 = wd + wb
        return l0 + l2

    def count_expected_flops_and_l0(self):
        '''
        Measures the expected floating point operations (FLOPs) and the expected L0 norm
        '''
        if opt.hardsigmoid:
            ppos = F.hardtanh(opt.k * self.z_phi / 7. + .5, 0, 1).sum()
        else:
            ppos = torch.sigmoid(opt.k * self.z_phi).sum()

        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = ((self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[
            0]) + 1  # for rows
        num_instances_per_filter *= ((self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[
            1]) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos
        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

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
            z = self.floatTensor(batch_size, self.dim_z).zero_()
            if self.training:
                if self.local_rep:
                    self.u = self.floatTensor(self.dim_z).uniform_(0, 1).expand(batch_size, self.dim_z)
                else:
                    self.u = self.floatTensor(batch_size, self.dim_z).uniform_(0, 1)

                z[self.u < pi.expand(batch_size, self.dim_z)] = 1
                if opt.use_t_in_training:
                    z[(pi.expand(batch_size, self.dim_z)) < opt.t] = 0
                self.train_z = z
            else:
                z[self.z_phi.expand(batch_size, self.dim_z) > 0] = 1
                # z = torch.sigmoid(self.z_phi.data).expand(batch_size, self.dim_z)
                if opt.use_t_in_testing:
                    z = pi.expand(batch_size, self.dim_z)
                    z[z < opt.t] = 0
                self.test_z = z
        else:
            # pi2 = torch.sigmoid(-opt.k * self.z_phi)
            pi2 = 1 - pi

            if self.u is None:
                raise Exception('Forward pass first')
            z = self.floatTensor(batch_size, self.dim_z).zero_()
            z[self.u > pi2.expand(batch_size, self.dim_z)] = 1
            if opt.use_t_in_training:
                z[pi.expand(batch_size, self.dim_z) < opt.t] = 0
        return z.view(batch_size, self.dim_z, 1, 1)

    def forward(self, input_):
        """ forward for fc """
        if self.input_shape is None:
            self.input_shape = input_.size()
        b = None if not self.use_bias else self.bias
        output = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)
        z = self.sample_z(output.size(0))
        output = output.mul(z)
        return output

    def activated_neurons(self):
        return (self.test_z > 0).sum() / self.test_z.size(0)

    def expected_activated_neurons(self):
        return (self.train_z > 0).sum() / self.train_z.size(0)

    def masked_weight(self):
        return self.weights * self.test_z[0].reshape(self.out_channels, 1, 1, 1)

    def __repr__(self):
        s = (
            '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, lamba={lamba}, weight_decay={weight_decay}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
