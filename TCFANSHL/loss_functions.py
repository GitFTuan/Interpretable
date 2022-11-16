import torch
import torch.nn as nn
import torch.nn.functional as F


# def FocalLoss(y_pred, y_true, gamma=2):
#     # y_pred is the logits without Sigmoid
#     assert y_pred.shape == y_true.shape
#     # pt = torch.exp(-F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')).detach()
#     # loss = nn.BCEWithLogitsLoss(reduction='none').detach
#     # pt = loss(y_pred, y_true)
#     m = nn.Sigmoid()
#     pt = m(y_pred).detach()
#     label_weight = torch.exp((2*(1 - pt)) ** gamma)
#     # label_weight = (2*(1 - pt)) ** 10
#     # print('label_weight\n', label_weight)
#     focal_loss = nn.BCEWithLogitsLoss(weight=label_weight, reduction='mean')
#     b = nn.BCEWithLogitsLoss(weight=label_weight, reduction='none')
#     # nn.BCEWithLogitsLoss(y_pred, y_true, weight=sample_weight, pos_weight=sample_weight, reduction=reduction)
#     return focal_loss(y_pred, y_true)


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean', class_weight=None):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.reducation = reduction
        self.class_weight = class_weight

    def forward(self, data, label):
        sigmoid = nn.Sigmoid()
        pt = sigmoid(data).detach()
        if self.class_weight is not None:
            label_weight = ((1-pt)**self.gamma)*self.class_weight
            # label_weight = torch.exp((1 - pt)) * self.class_weight
            # label_weight = torch.exp((2 * (1 - pt)) ** self.gamma) * self.class_weight
        else:
            label_weight = (1 - pt) ** self.gamma
            # label_weight = torch.exp((1 - pt))
            # label_weight = torch.exp((2 * (1 - pt)) ** self.gamma)

        focal_loss = nn.BCEWithLogitsLoss(weight=label_weight, reduction='mean')
        return focal_loss(data, label)


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, class_weight=None, label_weight=None, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.class_weight = class_weight
        self.label_weight = label_weight
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        # if pred.dim() != target.dim():
        #     target, label_weight = _expand_binary_labels(
        #     target, label_weight, pred.size(-1))
        target, label_weight = target.float(), self.label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        # sigmoid梯度计算
        g = torch.abs(pred.sigmoid().detach() - target)
        # 有效的label的位置
        valid = label_weight > 0
        # 有效的label的数量
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            # 将对应的梯度值划分到对应的bin中， 0-1
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid

            # 该bin中存在多少个样本
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    # moment计算num bin
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    # 权重等于总数/num bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            # scale系数
            weights = weights / n

        if self.class_weight is not None:
            weights = weights*self.class_weight
        else:
            weights = weights
        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_Loss):
    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, reduction='mean'):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
            # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
            # https://blog.csdn.net/weixin_39504171/article/details/106069230

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        if self.reduction == 'mean':
            loss = -loss.mean()
        else:
            loss = -loss.sum()
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)

        loss = 1 - (2*num) / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class NewDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(NewDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num_pos = torch.sum(torch.mul(predict, target), dim=1)
        # dim=1 按行相加
        den_pos = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)
        num_neg = torch.sum(torch.mul((1-predict), (1-target)), dim=1)
        den_neg = torch.sum((1-predict).pow(self.p) + (1-target).pow(self.p), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class SuperDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(SuperDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num_pos = torch.sum(torch.mul((predict-0.5), target), dim=1)
        # dim=1 按行相加
        # den_pos = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)
        den_pos = torch.sum(torch.mul((predict-0.5), target) + torch.mul((0.5-predict), (1-target)), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos

        loss = loss_pos
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class NewSuperDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        p_pos, p_neg: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        clip:
        class_weight:
        input: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, p_pos=2, p_neg=2, clip=0.05, class_weight=None, reduction='mean'):
        super(NewSuperDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip = clip
        self.class_weight = class_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num_pos = torch.sum(torch.mul(predict, target), dim=1)
        # dim=1 按行相加
        den_pos = torch.sum(predict.pow(self.p_pos) + target.pow(self.p_pos), dim=1)
        xs_neg = 1-predict
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        num_neg = torch.sum(torch.mul(xs_neg, (1 - target)), dim=1)
        den_neg = torch.sum((1-predict).pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos + loss_neg
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class SuperLoss(nn.Module):
    """Dice loss of binary class
    Args:
        p_pos, p_neg: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        clip:
        class_weight:
        input: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, p_pos=2, p_neg=2, clip=0.05, class_weight=None, reduction='mean'):
        super(SuperLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip = clip
        self.class_weight = class_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num_pos = torch.sum(torch.mul(predict, target), dim=1)
        # dim=1 按行相加
        den_pos = torch.sum(predict.pow(self.p_pos) + target.pow(self.p_pos), dim=1)
        xs_neg = 1-predict
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        num_neg = torch.sum(torch.mul(xs_neg, (1 - target)), dim=1)
        den_neg = torch.sum(xs_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2*num_pos) / den_pos
        loss_neg = 1 - (2*num_neg) / den_neg
        loss = (loss_pos*loss_neg)/(loss_pos+loss_neg)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class HeroLoss(nn.Module):
    """Dice loss of binary class
    Args:
        p_pos, p_neg: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        clip:
        class_weight:
        input: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.05, clip_neg=0.05, class_weight=None, reduction='mean'):
        super(HeroLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.class_weight = class_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        m_pos = predict
        if self.clip_neg is not None and self.clip_neg > 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
        num_pos = torch.sum(torch.mul(m_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(m_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1-predict
        if self.clip_neg is not None and self.clip_neg > 0:
            xs_neg = ((xs_neg + self.clip_neg).clamp(max=1))*xs_neg
        num_neg = torch.sum(torch.mul(xs_neg, (1 - target)), dim=1)
        den_neg = torch.sum(xs_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2*num_pos) / den_pos
        loss_neg = 1 - (2*num_neg) / den_neg
        loss = loss_pos+loss_neg
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class SuperHeroLoss(nn.Module):

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.50, clip_neg=0.50, pos_weight=0.50, reduction='mean'):
        super(SuperHeroLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos*self.pos_weight + loss_neg*(1-self.pos_weight)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class NewSuperHeroLoss(nn.Module):
    """Dice loss of binary class
    Args:
        p_pos, p_neg: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        clip:
        class_weight:
        input: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.5, clip_neg=0.5, smooth=0.05, class_weight=None, reduction='mean'):
        super(NewSuperHeroLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.class_weight = class_weight
        self.smooth = smooth

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_neg is not None and self.clip_neg > 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg > 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
            p_neg = (p_neg + self.smooth).clamp(max=1)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos + loss_neg
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


if __name__ == '__main__':
    m = nn.Sigmoid()
    batch_size = 3
    num_class = 3
    label_weight = torch.Tensor([3, 1, 2])
    input = torch.randn((3, 3), requires_grad=True)
    target = torch.empty((3, 3)).random_(2)
    print("traget\n", target)
    print(target*label_weight)
    print(torch.mul(target, label_weight))
    # 可以利用广播机制为样本赋予权重
    # loss2 = nn.BCELoss(reduction='mean')
    # loss3 = nn.BCEWithLogitsLoss(reduction='mean', weight=label_weight)
    # loss4 = BCEFocalLoss(gamma=0, class_weight=label_weight)
    # loss5 = GHMC(label_weight=label_weight)
    # loss6 = AsymmetricLoss()
    # loss7 = BinaryDiceLoss()
    # loss8 = NewDiceLoss()
    # loss9 = SuperDiceLoss()
    loss10 = SuperHeroLoss()

    # weight 是指每个标签的权重； pos_weight 是指正负样本的权重

    # output2 = loss2(m(input), target)
    # output3 = loss3(input, target)
    # # output1 = FocalLoss(m(input), target)
    # output4 = loss4(input, target)
    # output5 = loss5(input, target)
    # output6 = loss6(input, target)
    # output7 = loss7(input, target)
    # output8 = loss8(input, target)
    # output9 = loss9(input, target)
    output10 = loss10(input, target)
    print(output10)
    # print('focal_loss\n', output1)
    # print('BCELoss\n', output2)
    # print('BCEWithLogitsLoss\n', output3)
    # print('BCEFocalLoss\n', output4)
    # print('GHMC\n', output5)
    # print('AsymmetricLoss\n', output6)
    # print(output7)
    # print(output8)
    # output1.backward()
    # output2.backward()
    # output3.backward()
