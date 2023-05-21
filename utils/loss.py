# Loss functions

import torch
import torch.nn as nn
import numpy as np

from utils.general import bbox_iou
from utils.torch_utils import is_parallel
from utils.plots import plot_samples
from torch.autograd import Variable
from descriptor.LSS import denseLSS
from descriptor.CFOG import denseCFOG


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# varifocal loss
class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=2.0, alpha=0.25):
        super(VFLoss, self).__init__()
        # 传递 nn.BCEWithLogitsLoss() 损失函数  must be nn.BCEWithLogitsLoss()
        self.loss_fcn = loss_fcn  #
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply VFL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits

        focal_weight = true * (true > 0.0).float() + self.alpha * (pred_prob - true).abs().pow(self.gamma) * (true <= 0.0).float()
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Ranking Loss
class RankingLoss2(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, true):
        loss = 0
        bs, c, h, w = pred.shape

        for x, y in zip(pred, true):
            x = self.sigmoid(x)

            loss -= y * torch.log(1 - (y - x) + 1e-7) + (1 - y) * torch.log(torch.where(self.threshold - (1 - x) > 0, 1 - self.threshold + (1 - x) + 1e-7, torch.ones([1], dtype=x.dtype, device='cuda')))
        return loss.sum() / (bs * c * h * w)


# Ranking Loss
class RankingLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, true):
        loss = 0.0
        bs, c, h, w = pred.shape

        for pred_i, true_i in zip(pred, true):
            for x, y in zip(pred_i, true_i):
                x = self.sigmoid(x)
                mask_negative = y < 0.3
                mask_positive = y > 0.5
                s = x[mask_positive]
                if len(s) == 0:
                    val = 0
                else:
                    pos_pred = x[mask_positive].min()
                    neg_pred = x[mask_negative].max()
                    val = torch.exp(neg_pred - pos_pred)
                    #val = (((1 + neg_pred - pos_pred) / 2) ** self.gamma) * torch.exp(neg_pred - pos_pred)
                    #val = torch.log(1 - (1 + neg_pred - pos_pred) / 2.0 + 1e-7)
                    '''
                    pos_pred = x[mask_positive].mean()
                    neg_pred = x[~mask_positive].mean()
                    if (pos_pred - neg_pred).item() >= 0.7:
                        val = 0
                    else:
                        val = torch.exp(neg_pred - pos_pred)
                    '''

                loss += val
        return loss / (bs * c)


# Similarity Loss
class SimLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def des_SSD(self, i, j, descriptor):
        mask_i = torch.ge(i.squeeze(0).squeeze(0), 1)
        mask_i = torch.tensor(mask_i, dtype=torch.float32)
        mask_j = torch.ge(j.squeeze(0).squeeze(0), 1)
        mask_j = torch.tensor(mask_j, dtype=torch.float32)
        mask = torch.mul(mask_i, mask_j)
        num = mask[mask.ge(1)].size()[0]
        if descriptor == 'CFOG':
            des_i = denseCFOG(i)
            des_j = denseCFOG(j)
        elif descriptor == 'LSS':
            des_i = denseLSS(i)
            des_j = denseLSS(j)
        des_i = torch.mul(des_i, mask)
        des_j = torch.mul(des_j, mask)
        SSD_loss = nn.MSELoss(reduction='sum')
        loss = SSD_loss(des_i, des_j) / num
        return loss

    def des_NCC(self, i, j, descriptor):
        mask_i = torch.ge(i.squeeze(0).squeeze(0), 1)
        mask_i = torch.tensor(mask_i, dtype=torch.float32)
        mask_j = torch.ge(j.squeeze(0).squeeze(0), 1)
        mask_j = torch.tensor(mask_j, dtype=torch.float32)
        mask = torch.mul(mask_i, mask_j)
        num = mask[mask.ge(1)].size()[0]
        if descriptor == 'CFOG':
            des_i = denseCFOG(i)
            des_j = denseCFOG(j)
        elif descriptor == 'LSS':
            des_i = denseLSS(i)
            des_j = denseLSS(j)
        des_i = torch.mul(des_i, mask)
        des_j = torch.mul(des_j, mask)
        loss = self.gncc_loss(des_i, des_j) * 512 * 512 / num
        return loss

    def gradient_loss(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

    def mse_loss(self, x, y):
        return torch.mean((x - y) ** 2)

    def DSC(self, pred, target):
        smooth = 1e-5
        m1 = pred.flatten()
        m2 = target.flatten()
        intersection = (m1 * m2).sum()
        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    def gncc_loss(self, I, J, eps=1e-5):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        I_ave, J_ave = I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()
        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)
        cc = cross / (I_var.sqrt() * J_var.sqrt() + eps)  # 1e-5
        return -1.0 * cc + 1

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2, J2, IJ = I * I, J * J, I * J
        I_sum = nn.functional.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = nn.functional.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = nn.functional.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = nn.functional.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = nn.functional.conv2d(IJ, filt, stride=stride, padding=padding)
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def cc_loss(self, x, y):
        dim = [2, 3, 4]
        mean_x = torch.mean(x, dim, keepdim=True)
        mean_y = torch.mean(y, dim, keepdim=True)
        mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
        mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
        stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
        stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
        return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

    def Get_Ja(self, flow):
        D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
        D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
        D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
        D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
        D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
        D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
        return D1 - D2 + D3

    def NJ_loss(self, ypred):
        Neg_Jac = 0.5 * (torch.abs(self.Get_Ja(ypred)) - self.Get_Ja(ypred))
        return torch.sum(Neg_Jac)

    def lncc_loss(self, i, j, win=[9, 9], eps=1e-5):
        I = i
        J = j
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        filters = Variable(torch.ones(1, 1, win[0], win[1])).cuda()
        padding = (win[0] // 2, win[1] // 2)
        I_sum = nn.functional.conv2d(I, filters, stride=1, padding=padding)
        J_sum = nn.functional.conv2d(J, filters, stride=1, padding=padding)
        I2_sum = nn.functional.conv2d(I2, filters, stride=1, padding=padding)
        J2_sum = nn.functional.conv2d(J2, filters, stride=1, padding=padding)
        IJ_sum = nn.functional.conv2d(IJ, filters, stride=1, padding=padding)
        win_size = win[0] * win[1]
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        cc = cross * cross / (I_var * J_var + eps)
        lcc = -1.0 * torch.mean(cc) + 1
        return lcc

    def forward(self, reference, sensed_tran, sensed, reference_inv_tran, descriptor, similarity):
        if similarity == 'SSD':  # Similarity: SSD or NCC based on descriptors
            loss1 = Variable(self.des_SSD(reference, sensed_tran, descriptor), requires_grad=True)
            loss2 = Variable(self.des_SSD(sensed, reference_inv_tran, descriptor), requires_grad=True)
        elif similarity == 'NCC':
            loss1 = Variable(self.des_NCC(reference, sensed_tran, descriptor), requires_grad=True)
            loss2 = Variable(self.des_NCC(sensed, reference_inv_tran, descriptor), requires_grad=True)

        loss = (loss1 + loss2) * 0.5
        return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = True  # 在计算objectness时是否对ciou进行排序
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        #BCEobj = VFLoss(BCEobj)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.RKobj = RankingLoss(2.0)
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj, lrk = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, offsets = self.build_targets(p, targets)  # targets

        #plot_samples(batch_index, imgs, path, tcls, tbox, indices, anchors, offsets, targets)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)  # 将iou按照从小到大进行排序，返回下标索引
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # 根据model.gr设置真实框的标签值

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            # Ranking
            #lrk += self.RKobj(pi[..., 4], tobj)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lrk *= 0.1
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lrk
        return loss * bs, torch.cat((lbox, lobj, lcls, lrk)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, offset = [], [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            offset.append(offsets)

        return tcls, tbox, indices, anch, offset
