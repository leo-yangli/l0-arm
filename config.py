import warnings


class DefaultConfig:
    """ model: ARMLeNet5 | ARMMLP | ARMWideResNet (default: ARMLeNet5)
        optimizer: adam | momentum (default: adam)
        dataset: mnist | cifar10 | cifar100 (default: mnist)
        lambas: L0 regularization strength (default: [10, 0.5, 0.1, 10])
        ar: use AR if True, else use ARM (default: False)
        hardsigmoid: use hardsigmoid if True, else use sigmoid
        k: the hyper-parameter that controls distribution over gates (default: 7)
        log_dir: directory for Tensorboard log (default: log)
        checkpoints_dir: directory for checkpoints (default: 'checkpoints')
        seed: seed for initializing training (default: None)
        max_epoch: number of total epochs to run (default: 200)
        start_epoch: manual epoch number (useful on restarts)
        use_gpu: use GPU or not (default: True)
        load_file: path to checkpoint (default: '')
        batch_size: mini-batch size (default: 128)
        lr: initial learning rate (default: 0.001)
        lr_decay: learning rate decay (default: 0.2)
        weight_decay: weight decay (default: 5e-4)
        momentum: momentum (default: 0.9)
        schedule_milestone: schedule for learning rate decay (default: [])
        t: threshold for gate. gate = 1 if gate > t; else gate = 0. (default: 0.5)
        use_t_in_training: use binary gate for training if True, else use continuous value (default: False)
        use_t_in_testing: use binary gate for testing if True, else use continuous value (default: True)
        lenet_dr: initial dropout rate for LeNet model (default: 0.5)
        mlp_dr: initial dropout rate for MLP model (default: 0.5)
        wrn_dr: initial dropout rate for WRN model (default: 0.01)
        local_rep: stochastic level (default: True)
        gpus: number of gpus (default: 1)
        note: note shown in log title (default: '')
        verbose: verbose mode. (default: True)
        print_freq: print frequency (default: 100) """

    model = 'ARMLeNet5'
    optimizer = 'adam'
    dataset = 'mnist'
    # lambas = [.1, .1, .1]  #MLP
    lambas = [10, 0.5, 0.1, 10]  # LeNet
    #lambas = 0.1   # WRN
    ar = False
    hardsigmoid = False
    k = 7

    log_dir = 'log'
    checkpoints_dir = 'checkpoints'
    seed = None
    use_gpu = True
    load_file = ''
    batch_size = 128
    start_epoch = 0
    max_epoch = 200
    lr = 0.001
    lr_decay = 0.2
    weight_decay = 5e-4
    momentum = 0.9
    schedule_milestone = []
    t = 0.5
    use_t_in_training = False
    use_t_in_testing = True
    lenet_dr = 0.5
    mlp_dr = 0.5
    wrn_dr = 0.01
    local_rep = True
    gpus = 1
    note = ''
    verbose = True
    print_freq = 100


def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    self.title_note = 'ar={} hs={} wd={} lr={} lambda={} k={} t={}'.format(self.ar, self.hardsigmoid,
                                                                           self.weight_decay, self.lr, self.lambas,
                                                                           self.k, self.t)
    str = ''
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))
            str += "{}: {}<br/>".format(k, getattr(self, k))
    return str


DefaultConfig.parse = parse
opt = DefaultConfig()
