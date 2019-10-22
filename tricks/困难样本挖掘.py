# OHEM
# focal loss把loss作用在类别上，目标检测OHEM把loss 作用在ROI上
import torch
import torch.nn.functional as F
from torch import nn


class NLL_OHEM(torch.nn.NLLLoss):
    """ Online hard example mining.
    Needs input from nn.LogSotmax() """

    def __init__(self, ratio):
        super(NLL_OHEM, self).__init__(None, True)
        self.ratio = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
            # loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn)



def ohem_loss(batch_size, cls_pred, cls_target, loc_pred, loc_target, smooth_l1_sigma=1.0):
    """
    Arguments:
        batch_size (int): number of sampled rois for bbox head training
        loc_pred (FloatTensor): [R, 4], location of positive rois
        loc_target (FloatTensor): [R, 4], location of positive rois
        pos_mask (FloatTensor): [R], binary mask for sampled positive rois
        cls_pred (FloatTensor): [R, C]
        cls_target (LongTensor): [R]
    Returns:
        cls_loss, loc_loss (FloatTensor)
    """
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
    ohem_loc_loss = F.smooth_l1_loss(loc_pred, loc_target, sigma=smooth_l1_sigma, reduce=False)
    # 这里先暂存下正常的分类loss和回归loss
    loss = ohem_cls_loss + ohem_loc_loss
    # 然后对分类和回归loss求和

    sorted_ohem_loss, idx = torch.sort(loss, descending=True)
    # 再对loss进行降序排列
    keep_num = min(sorted_ohem_loss.size()[0], batch_size)
    # 得到需要保留的loss数量
    if keep_num < sorted_ohem_loss.size()[0]:
        # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
        keep_idx_cuda = idx[:keep_num]
        # 保留到需要keep的数目
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        ohem_loc_loss = ohem_loc_loss[keep_idx_cuda]
        # 分类和回归保留相同的数目
    cls_loss = ohem_cls_loss.sum() / keep_num
    loc_loss = ohem_loc_loss.sum() / keep_num
    # 然后分别对分类和回归loss求均值
    return cls_loss, loc_loss

# TopK Loss
# Focal Loss 1,2