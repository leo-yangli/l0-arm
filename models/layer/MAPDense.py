import torch
from torch.nn import Parameter


class MAPDense(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, weight_decay=1., **kwargs):
        super(MAPDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_out')

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def constrain_parameters(self, **kwargs):
        pass

    def _reg_w(self, **kwargs):
        logpw = torch.sum(self.weight_decay * .5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        expected_flops = (2 * self.in_features - 1) * self.out_features
        expected_l0 = self.in_features * self.out_features
        if self.bias is not None:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops, expected_l0

    def forward(self, input):
        output = input.mm(self.weight)
        if self.bias is not None:
            output.add_(self.bias.view(1, self.out_features).expand_as(output))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', weight_decay: ' \
            + str(self.weight_decay) + ')'

    def activated_neurons(self):
        return self.in_features

    def expected_activated_neurons(self):
        return self.in_features
