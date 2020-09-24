import torch.nn as nn


class SmooothL1Loss(nn.Module):
    def __init__(self, alpha_1, alpha_2):
        super(SmooothL1Loss, self).__init__()
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.loss = nn.L1Loss()

    def cal_l1_loss(self, est, target, mix):
        est_ = mix - est
        loss_1 = self.loss(est, target[:,0,:])
        loss_2 = self.loss(est_, target[:,1,:])
        return loss_1 + loss_2

    def forward(self, est, target, mix):
        est_inter = est[0].squeeze()
        est_intra = est[1].squeeze()
        loss_inter = self.alpha_1*self.cal_l1_loss(est_inter, target, mix)
        loss_intra = self.alpha_2*self.cal_l1_loss(est_intra, target, mix)
        return loss_inter + loss_intra, loss_inter.item(), loss_intra.item()

