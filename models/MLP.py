import torch
import torch.nn as nn

from models.layer import ARMDense

from copy import deepcopy

from config import opt


class ARMMLP(nn.Module):
    def __init__(self, input_dim=784, num_classes=10, N=60000, layer_dims=(300, 100), beta_ema=0.999,
                 weight_decay=5e-4, lambas=(.1, .1, .1), local_rep=True):
        super(ARMMLP, self).__init__()

        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.N = N
        self.weight_decay = N * weight_decay
        self.lambas = lambas

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            droprate_init, lamba = 0.2 if i == 0 else opt.mlp_dr, lambas[i] if len(lambas) > 1 else lambas[0]
            layers += [ARMDense(inp_dim, dimh, droprate_init=droprate_init, weight_decay=self.weight_decay,
                                lamba=lamba, local_rep=opt.local_rep), nn.ReLU()]

        layers.append(ARMDense(self.layer_dims[-1], num_classes, droprate_init=opt.mlp_dr,
                               weight_decay=self.weight_decay, lamba=lambas[-1], local_rep=opt.local_rep))

        self.output = nn.Sequential(*layers)
        self.layers = []
        for m in self.modules():
            if isinstance(m, ARMDense):
                self.layers.append(m)

        self.beta_ema = beta_ema
        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def score(self, x):
        return self.output(x.view(-1, self.input_dim))

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

    def update_phi_gradient(self, f1, f2):
        for layer in self.layers:
            layer.update_phi_gradient(f1, f2)

    def forward_mode(self, mode):
        for layer in self.layers:
            layer.forward_mode = mode

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += (1. / self.N) * layer.regularization()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def get_activated_neurons(self):
        return [layer.activated_neurons() for layer in self.layers]

    def get_expected_activated_neurons(self):
        return [layer.expected_activated_neurons() for layer in self.layers]

    def prune_rate(self):
        l = [layer.activated_neurons().cpu().numpy() for layer in self.layers]
        return 100 - 100.0 * (l[0] * l[1] + l[1] * l[2] + l[2] * 10.) / (784. * 300. + 30000. + 1000.)

    def z_phis(self):
        return [layer.z_phi for layer in self.layers]

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
