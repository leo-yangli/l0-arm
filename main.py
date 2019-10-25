import torch
from torchnet import meter
from torch import nn
from tqdm import tqdm
import data.dataset as dataset
import models
from utils.visualization import Visualizer
import numpy as np
import time
from time import localtime
from config import opt
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


device = None
cudnn.benchmark = True
current_time = time.strftime('%Y-%m-%d %H:%M:%S', localtime())
print(current_time)
vis = None


def train(**kwargs):
    global device, vis
    if opt.seed is not None:
        setup_seed(opt.seed)
    config_str = opt.parse(kwargs)
    device = torch.device("cuda" if opt.use_gpu else "cpu")

    vis = Visualizer(opt.log_dir, opt.model, current_time, opt.title_note)
    # log all configs
    vis.log('config', config_str)

    # load data set
    train_loader, val_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size * opt.gpus)
    # load model
    model = getattr(models, opt.model)(lambas=opt.lambas, num_classes=num_classes, weight_decay=opt.weight_decay).to(
        device)

    if opt.gpus > 1:
        model = nn.DataParallel(model)

    # define loss function
    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var)
        reg_loss = model.regularization() if opt.gpus <= 1 else model.module.regularization()
        total_loss = (loss + reg_loss).to(device)
        return total_loss

    # load optimizer and scheduler
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters() if opt.gpus <= 1 else model.module.parameters(), opt.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.lr_decay, patience=15)
        scheduler = None
        print('Optimizer: Adam, lr={}'.format(opt.lr))
    elif opt.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters() if opt.gpus <= 1
                                    else model.module.parameters(), opt.lr, momentum=opt.momentum, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.schedule_milestone,
                                                         gamma=opt.lr_decay)
        print('Optimizer: Momentum, lr={}, momentum'.format(opt.lr, opt.momentum))
    else:
        print('No optimizer')
        return

    loss_meter = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(accuracy=True)
    # create checkpoints dir
    directory = '{}/{}_{}'.format(opt.checkpoints_dir, opt.model, current_time)
    if not os.path.exists(directory):
        os.makedirs(directory)
    total_steps = 0
    for epoch in range(opt.start_epoch, opt.max_epoch) if opt.verbose else tqdm(range(opt.start_epoch, opt.max_epoch)):
        model.train() if opt.gpus <= 1 else model.module.train()
        loss_meter.reset()
        accuracy_meter.reset()
        for ii, (input_, target) in enumerate(train_loader):
            input_, target = input_.to(device), target.to(device)
            optimizer.zero_grad()
            score = model(input_, target)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.cpu().data)
            accuracy_meter.add(score.data, target.data)

            e_fl, e_l0 = model.get_exp_flops_l0() if opt.gpus <= 1 else model.module.get_exp_flops_l0()
            vis.plot('stats_comp/exp_flops', e_fl, total_steps)
            vis.plot('stats_comp/exp_l0', e_l0, total_steps)
            total_steps += 1

            if (model.beta_ema if opt.gpus <= 1 else model.module.beta_ema) > 0.:
                model.update_ema() if opt.gpus <= 1 else model.module.update_ema()

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('train/loss', loss_meter.value()[0])
                vis.plot('train/accuracy', accuracy_meter.value()[0])
                if opt.verbose:
                    print("epoch:{epoch},lr:{lr},loss:{loss:.2f},train_acc:{train_acc:.2f}"
                      .format(epoch=epoch, loss=loss_meter.value()[0],
                              train_acc=accuracy_meter.value()[0],
                              lr=optimizer.param_groups[0]['lr']))

        # save model
        if epoch % 10 == 0 or epoch == opt.max_epoch - 1:
            torch.save(model.state_dict(), directory + '/{}.model'.format(epoch))
        # validate model
        val_accuracy, val_loss = val(model, val_loader, criterion)

        vis.plot('val/loss', val_loss)
        vis.plot('val/accuracy', val_accuracy)

        # update lr
        if scheduler is not None:
            if isinstance(optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step(epoch)
        if opt.verbose:
            print("epoch:{epoch},lr:{lr},loss:{loss:.2f},val_acc:{val_acc:.2f},prune_rate:{pr:.2f}"
                  .format(epoch=epoch, loss=loss_meter.value()[0], val_acc=val_accuracy, lr=optimizer.param_groups[0]['lr'],
                          pr=model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate()))
        for (i, num) in enumerate(model.get_expected_activated_neurons() if opt.gpus <= 1
                                  else model.module.get_expected_activated_neurons()):
            vis.plot("Training_layer/{}".format(i), num)
        vis.plot('lr', optimizer.param_groups[0]['lr'])


def val(model, dataloader, criterion):
    model.eval() if opt.gpus <= 1 else model.module.eval()
    loss_meter = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(accuracy=True)
    for ii, data in enumerate(dataloader):
        input_, label = data
        input_, label = input_.to(device), label.to(device)
        score = model(input_)
        accuracy_meter.add(score.data.squeeze(), label.long())
        loss = criterion(score, label)
        loss_meter.add(loss.cpu().data)

    for (i, num) in enumerate(model.get_activated_neurons() if opt.gpus <= 1 else model.module.get_activated_neurons()):
        vis.plot("val_layer/{}".format(i), num)

    for (i, z_phi) in enumerate(model.z_phis()):
        if opt.hardsigmoid:
            vis.hist("hard_sigmoid(phi)/{}".format(i), F.hardtanh(opt.k * z_phi / 7. + .5, 0, 1).cpu().detach().numpy())
        else:
            vis.hist("sigmoid(phi)/{}".format(i), torch.sigmoid(opt.k * z_phi).cpu().detach().numpy())

    vis.plot("prune_rate", model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate())
    return accuracy_meter.value()[0], loss_meter.value()[0]


def test(**kwargs):
    opt.parse(kwargs)
    global device, vis
    device = torch.device("cuda" if opt.use_gpu else "cpu")
    vis = Visualizer(opt.log_dir, opt.model, current_time)
    # load model
    model = getattr(models, opt.model)(lambas=opt.lambas).to(device)
    # load data set
    train_loader, val_loader, num_classes = getattr(dataset, opt.dataset)(opt.batch_size * opt.gpus)

    # define loss function
    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var)
        total_loss = (loss + model.regularization() if opt.gpus <= 1 else model.module.regularization()).to(device)
        return total_loss

    if len(opt.load_file) > 0:
        model.load_state_dict(torch.load(opt.load_file))
        val_accuracy, val_loss = val(model, val_loader, criterion)
        print("loss:{loss:.2f},val_acc:{val_acc:.2f},prune_rate:{pr:.2f}"
              .format(loss=val_loss, val_acc=val_accuracy,
                      pr=model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate()))
        # print(model.get_activated_neurons())


def help():
    '''help'''
    print('''
    usage : python main.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --model=ARMLeNet5 --dataset=mnist --lambas="[.1,.1,.1,.1]" --optimizer=adam --lr=0.001
            python {0} test --model=ARMLeNet5 --dataset=mnist --lambas="[.1,.1,.1,.1]" --load_file="checkpoints/ARMLeNet5_2019-06-19 14:27:03/0.model"
            python {0} train --model=ARMWideResNet --dataset=cifar10 --lambas=.001 --optimizer=momentum --lr=0.1 --schedule_milestone="[60,120]"
            python {0} help
            
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire({'train': train, 'test': test, 'help': help})
