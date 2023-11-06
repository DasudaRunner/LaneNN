#coding:utf-8
import torch
import math
import torch.nn as nn

def conv1x1(in_planes, out_planes, stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)

def activation(act_type='prelu'):
    if act_type == 'prelu':
        act = nn.PReLU()
    else:
        act = nn.ReLU(inplace=True)
    return act

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_weights(_modules):
    for m in _modules:
        if isinstance(m, nn.Conv2d):
            # init.xavier_normal(m.weight.data)
            fan_in = m.out_channels * m.kernel_size[0] * m.kernel_size[1]
            scale = math.sqrt(2. / fan_in)
            m.weight.data.uniform_(-scale, scale)
            if m.bias is not None:
                m.bias.data.zero_()
        # Xavier can not be applied to less than 2D.
        elif isinstance(m, nn.BatchNorm2d):
            if not m.weight is None:
                m.weight.data.fill_(1)
            if not m.bias is None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # n = m.weight.size(1)
            m.weight.data.normal_(0, 0.001)
            if m.bias is not None:
                m.bias.data.zero_()
                
def count_parameters_num(model):
    count = 0
    count_fc = 0
    count_others = 0
    param_dict = {name:param for name,param in model.named_parameters()}
    param_keys = param_dict.keys()
    for m_name, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count += temp_params.data.nelement()
        elif isinstance(m, nn.Linear):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count_fc += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count_fc += temp_params.data.nelement()
        else:
            num = sum(p.numel() for p in m.parameters())
            count_others += num
    total_count = count + count_fc + count_others
    print('Number of params, total:%.2fM, conv/bn: %.2fM, fc: %.2fM, others: %.2fM'
               % (total_count/1e6, count / 1e6,count_fc / 1e6, count_others/ 1e6))
    return (total_count/1e6, count / 1e6, count_fc / 1e6, count_others/ 1e6)


