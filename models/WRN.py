from models.layer import ArmConv2d
from models.layer.MAPConv2D import MAPConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from models.layer.MAPDense import MAPDense

from config import opt


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, droprate_init=0.0, weight_decay=0., lamba=0.01, local_rep=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = ArmConv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
                               droprate_init=droprate_init, weight_decay=weight_decay / (1 - 0.3),
                               local_rep=local_rep, lamba=lamba)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = MAPConv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_decay=weight_decay)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            MAPConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                      weight_decay=weight_decay) or None

    def forward(self, x):
        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.add(out, x if self.equalInOut else self.convShortcut(x))


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate_init=0.0, weight_decay=0., lamba=0.01,
                 local_rep=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate_init,
                                      weight_decay=weight_decay, lamba=lamba, local_rep=local_rep)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate_init,
                    weight_decay=0., lamba=0.01, local_rep=False):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,
                                droprate_init, weight_decay, lamba, local_rep))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ARMWideResNet(nn.Module):
    # droprate_init = 0.3
    def __init__(self, depth=28, num_classes=10, widen_factor=10, N=50000, beta_ema=0.99, weight_decay=5e-4,
                 lambas=0.001):
        super(ARMWideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        self.n = (depth - 4) // 6
        self.N = N
        self.beta_ema = beta_ema
        block = BasicBlock
        droprate_init = opt.wrn_dr

        self.weight_decay = N * weight_decay
        self.lamba = lambas

        # 1st conv before any network block
        self.conv1 = MAPConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
                               weight_decay=self.weight_decay)
        # 1st block
        self.block1 = NetworkBlock(self.n, nChannels[0], nChannels[1], block, 1, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=opt.local_rep)
        # 2nd block
        self.block2 = NetworkBlock(self.n, nChannels[1], nChannels[2], block, 2, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=opt.local_rep)
        # 3rd block
        self.block3 = NetworkBlock(self.n, nChannels[2], nChannels[3], block, 2, droprate_init, self.weight_decay,
                                   self.lamba, local_rep=opt.local_rep)
        # bn, relu and classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fcout = MAPDense(nChannels[3], num_classes, weight_decay=self.weight_decay)

        self.layers, self.bn_params, self.l0_layers = [], [], []
        for m in self.modules():
            if isinstance(m, MAPDense) or isinstance(m, MAPConv2d) or isinstance(m, ArmConv2d):
                self.layers.append(m)
                if isinstance(m, ArmConv2d):
                    self.l0_layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        print('Using weight decay: {}'.format(self.weight_decay))

    def score(self, x, y=None):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fcout(out)

    def update_phi_gradient(self, f1, f2):
        for layer in self.l0_layers:
            layer.update_phi_gradient(f1, f2)

    def forward_mode(self, mode):
        for layer in self.l0_layers:
            layer.forward_mode = mode

    def forward(self, x, y=None):
        if self.training:
            self.forward_mode(True)
            score = self.score(x)

            self.eval() if opt.gpus <= 1 else self.module.eval()
            if opt.ar is not True:
                self.forward_mode(False)
                score2 = self.score(x).data
                f1 = nn.CrossEntropyLoss()(score2, y).data
            else:
                f1 = 0
            f2 = nn.CrossEntropyLoss()(score, y).data

            self.update_phi_gradient(f1, f2)
            self.train() if opt.gpus <= 1 else self.module.train()
        else:
            self.forward_mode(True)
            score = self.score(x)
        return score

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += (1. / self.N) * layer.regularization()
        for bnw in self.bn_params:
            if self.weight_decay > 0:
                regularization += (self.weight_decay / self.N) * .5 * torch.sum(bnw.pow(2))
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            try:
                e_fl, e_l0 = layer.count_expected_flops_and_l0()
                expected_flops += e_fl
                expected_l0 += e_l0
            except:
                pass
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_activated_neurons(self):
        return [layer.activated_neurons() for layer in self.l0_layers]

    def get_expected_activated_neurons(self):
        return [layer.expected_activated_neurons() for layer in self.l0_layers]

    def prune_rate(self):
        l = [layer.activated_neurons().cpu().numpy() for layer in self.l0_layers]
        return 100 - 100. * (l[0] * 16 + (l[1] + l[2] + l[3] + l[4]) * 160 + (l[5] + l[6] + l[7] + l[8]) * 320 + (
                    l[9] + l[10] + l[11]) * 640) \
               / (16 * 160 + 160 * 160 * 3 + 160 * 320 + 320 * 320 * 3 + 320 * 640 + 640 * 640 * 3)

    def z_phis(self):
        return [layer.z_phi for layer in self.l0_layers]
