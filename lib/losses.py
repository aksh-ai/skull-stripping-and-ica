import torch as th
from torch.nn import Module

class DiceLoss(Module):
    def init(self):
        super(DiceLoss, self).init()
    
    def forward(self, y_pred, y_true, smooth=1.0):       
       y_pred = y_pred.contiguous().view(-1)
       y_true = y_true.contiguous().view(-1)
       
       intersection = (y_pred * y_pred).sum()
       
       A_sum = th.sum(y_pred * y_pred)
       B_sum = th.sum(y_true * y_true)
       
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

class StandardSegmentationLoss(Module):
    def __init__(self, loss_type="L1", num_classes=1):
        from torch.nn import L1Loss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss, MSELoss

        if loss_type == 'L1':
            self.loss = L1Loss() 
        elif loss == 'smooth_L1':
            self.loss = SmoothL1Loss()
        elif loss == 'MSE':
            self.loss = MSELoss()
        elif loss == 'BCE':
            self.loss = BCELoss()
        elif loss == 'BCE_logits':
            self.loss = BCEWithLogitsLoss()
        else:
            raise Exception("Invalid Loss function defined.")

        self.num_classes = num_classes
    
    def forward(self, y_pred, y_true):
        assert y_pred.shape[0] == y_true.shape[0], "Tensors are of different batch sizes"
        assert y_pred.size() == y_true.size(), "Tensors don't have the same shape"

        y_true = y_true.reshape(-1, self.num_classes)
        y_pred = y_pred.reshape(-1, self.num_classes)

        loss = self.loss(y_pred, y_true)

        return loss

class VAELoss(Module):
    def __init__(self, mu=None, log_var=None, loss_type="L1", h1=0.1, h2=0.1):
        self.mu = mu
        self.log_var = log_var

        from torch.nn import L1Loss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss

        if loss_type == 'L1':
            self.loss = L1Loss() 
        elif loss == 'smooth_L1':
            self.loss = SmoothL1Loss()
        elif loss == 'MSE':
            self.loss = MSE
        elif loss == 'BCE':
            self.loss = BCELoss()
        elif loss == 'BCE_logits':
            self.loss = BCEWithLogitsLoss()
        else:
            raise Exception("Invalid Loss function defined.")

        self.h1, self.h2 = h1, h2
    
    def forward(self, pred, x, mu, logvar):
        assert pred.shape[0] == x.shape[0], "Tensors are of different batch sizes"
        assert pred.size() == x.size(), "Tensors don't have the same shape"

        batch_size = x.shape[0]

        loss_val = self.loss(pred.view(batch_size, -1), x.view(batch_size, -1))

        kl_divergence = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return kl_divergence * self.h2 + loss_val * self.h1

class GANLoss:
    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        raise NotImplementedError("gen_loss method has not been implemented")

class StandardGANLoss(GANLoss):
    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps):
        assert real_samps.device == fake_samps.device, "Real and Fake samples are not on the same device"

        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # calculate the real loss:
        real_loss = self.criterion(
            th.squeeze(r_preds),
            th.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            th.squeeze(f_preds),
            th.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps):
        preds, _, _ = self.dis(fake_samps)
        return self.criterion(th.squeeze(preds),
                              th.ones(fake_samps.shape[0]).to(fake_samps.device))

class WGAN_GP_Loss(GANLoss):
    def __init__(self, dis, drift=0.001, use_gp=False):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, reg_lambda=10):
        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)
        merged.requires_grad = True

        # forward pass
        op = self.dis(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = th.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=th.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps)
        real_out = self.dis(real_samps)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps))

        return loss

class LSGAN_Loss(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        return 0.5 * (((th.mean(self.dis(real_samps)) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps))) ** 2)

    def gen_loss(self, _, fake_samps):
        return 0.5 * ((th.mean(self.dis(fake_samps)) - 1) ** 2)

class LSGAN_SIGMOID_Loss(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        from torch.nn.functional import sigmoid
        real_scores = th.mean(sigmoid(self.dis(real_samps)))
        fake_scores = th.mean(sigmoid(self.dis(fake_samps)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def gen_loss(self, _, fake_samps):
        from torch.nn.functional import sigmoid
        scores = th.mean(sigmoid(self.dis(fake_samps)))
        return 0.5 * ((scores - 1) ** 2)

class HingeGANLoss(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        r_preds, r_mus, r_sigmas = self.dis(real_samps)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps)

        loss = (th.mean(th.nn.ReLU()(1 - r_preds)) +
                th.mean(th.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps):
        return -th.mean(self.dis(fake_samps))

class RelativisticAverageHingeGANLoss(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))