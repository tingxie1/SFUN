
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class TVLoss(nn.Module):

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)


class SpectralAngleLoss(nn.Module):

    def __init__(self, eps=1e-8, reduction='mean', use_arccos=False):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.use_arccos = use_arccos  

    def forward(self, pred, target):

        dot_product = torch.sum(pred * target, dim=1)


        pred_norm = torch.norm(pred, dim=1) + self.eps
        target_norm = torch.norm(target, dim=1) + self.eps


        cos_sim = dot_product / (pred_norm * target_norm)


        cos_sim = torch.clamp(cos_sim, -1 + self.eps, 1 - self.eps)

        if self.use_arccos:

            sam = torch.acos(cos_sim)
        else:

            sam = 1 - cos_sim

        if self.reduction == 'mean':
            return sam.mean()
        elif self.reduction == 'sum':
            return sam.sum()
        else:
            return sam



SAMLoss = SpectralAngleLoss



class SpectralTVLoss(nn.Module):


    def __init__(self, weight=1.0, order=1):
        super().__init__()
        self.weight = weight
        self.order = order

    def forward(self, x, target=None):

        x_grad = x[:, 1:, :, :] - x[:, :-1, :, :]

        if target is not None:
            target_grad = target[:, 1:, :, :] - target[:, :-1, :, :]
            loss = F.l1_loss(x_grad, target_grad)

            if self.order >= 2:
                x_grad2 = x_grad[:, 1:, :, :] - x_grad[:, :-1, :, :]
                target_grad2 = target_grad[:, 1:, :, :] - target_grad[:, :-1, :, :]
                loss = loss + 0.5 * F.l1_loss(x_grad2, target_grad2)
        else:
            loss = torch.abs(x_grad).mean()

            if self.order >= 2:
                x_grad2 = x_grad[:, 1:, :, :] - x_grad[:, :-1, :, :]
                loss = loss + 0.5 * torch.abs(x_grad2).mean()

        return self.weight * loss



SpectralGradientLoss = SpectralTVLoss



class SSIMLoss(nn.Module):

    def __init__(self, window_size=11, channel=28, reduction='mean'):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.reduction = reduction
        self.register_buffer('window', self._create_window(window_size, channel))

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map

    def forward(self, pred, target):
        channel = pred.size(1)

        if self.window.shape[0] != channel:
            self.window = self._create_window(self.window_size, channel).to(pred.device)

        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)

        ssim_map = self._ssim(pred, target, self.window, self.window_size, channel)

        if self.reduction == 'mean':
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map


class FrequencyLoss(nn.Module):

    def __init__(self, loss_type='l1', amp_weight=1.0, phase_weight=0.0):
        super().__init__()
        self.loss_type = loss_type
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight

    def forward(self, pred, target):

        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')


        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        if self.loss_type == 'l1':
            amp_loss = F.l1_loss(pred_amp, target_amp)
        else:
            amp_loss = F.mse_loss(pred_amp, target_amp)


        if self.phase_weight > 0:
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)
            phase_diff = torch.abs(pred_phase - target_phase)
            phase_diff = torch.minimum(phase_diff, 2 * math.pi - phase_diff)
            phase_loss = phase_diff.mean()
        else:
            phase_loss = 0

        return self.amp_weight * amp_loss + self.phase_weight * phase_loss


class FocalFrequencyLoss(nn.Module):


    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        freq_diff = (pred_amp - target_amp) ** 2

        weight = freq_diff ** self.alpha
        weight = weight / (weight.sum() + 1e-8)

        loss = (weight * freq_diff).sum()
        return loss


class HSIRestoreLoss(nn.Module):


    def __init__(
        self,
        channels=28,
        charbonnier_weight=None,
        ssim_weight=None,
        sam_weight=None,
        spectral_tv_weight=None,
        freq_weight=None,
        tv_weight=None,
        w_char=None,
        w_ssim=None,
        w_sam=None,
        w_spec_tv=None,
        w_spectv=None,
        w_freq=None,
        w_tv=None,
        return_dict=False,
        char_eps=1e-3,
        ssim_window=11,
        sam_eps=1e-8,
    ):
        super().__init__()


        self.charbonnier_weight = w_char if w_char is not None else (charbonnier_weight if charbonnier_weight is not None else 1.0)
        self.ssim_weight = w_ssim if w_ssim is not None else (ssim_weight if ssim_weight is not None else 0.2)
        self.sam_weight = w_sam if w_sam is not None else (sam_weight if sam_weight is not None else 0.1)
        self.spectral_tv_weight = w_spec_tv if w_spec_tv is not None else (w_spectv if w_spectv is not None else (spectral_tv_weight if spectral_tv_weight is not None else 0.05))
        self.freq_weight = w_freq if w_freq is not None else (freq_weight if freq_weight is not None else 0.0)
        self.tv_weight = w_tv if w_tv is not None else (tv_weight if tv_weight is not None else 0.0)

        self.return_dict = return_dict
        self.channels = channels

        self.charbonnier = CharbonnierLoss(eps=char_eps)

        if self.ssim_weight > 0:
            self.ssim = SSIMLoss(window_size=ssim_window, channel=channels)
        else:
            self.ssim = None

        if self.sam_weight > 0:
            self.sam = SpectralAngleLoss(eps=sam_eps, use_arccos=False)
        else:
            self.sam = None

        if self.spectral_tv_weight > 0:
            self.spectral_tv = SpectralTVLoss(weight=1.0, order=1)
        else:
            self.spectral_tv = None

        if self.freq_weight > 0:
            self.freq = FrequencyLoss(loss_type='l1')
        else:
            self.freq = None

        if self.tv_weight > 0:
            self.tv = TVLoss(weight=1.0)
        else:
            self.tv = None

    def forward(self, pred, target):
        losses = {}

        losses['charbonnier'] = self.charbonnier(pred, target)
        total_loss = self.charbonnier_weight * losses['charbonnier']

        if self.ssim is not None and self.ssim_weight > 0:
            losses['ssim'] = self.ssim(pred, target)
            total_loss = total_loss + self.ssim_weight * losses['ssim']

        if self.sam is not None and self.sam_weight > 0:
            losses['sam'] = self.sam(pred, target)
            total_loss = total_loss + self.sam_weight * losses['sam']

        if self.spectral_tv is not None and self.spectral_tv_weight > 0:
            losses['spectral_tv'] = self.spectral_tv(pred, target)
            total_loss = total_loss + self.spectral_tv_weight * losses['spectral_tv']

        if self.freq is not None and self.freq_weight > 0:
            losses['freq'] = self.freq(pred, target)
            total_loss = total_loss + self.freq_weight * losses['freq']

        if self.tv is not None and self.tv_weight > 0:
            losses['tv'] = self.tv(pred)
            total_loss = total_loss + self.tv_weight * losses['tv']

        if self.return_dict:
            losses['total'] = total_loss
            return total_loss, losses
        else:
            return total_loss



HybridLossV3 = HSIRestoreLoss



class StagedLoss(nn.Module):


    def __init__(self, channels=28, total_epochs=100):
        super().__init__()
        self.total_epochs = total_epochs
        self.channels = channels

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss(channel=channels)
        self.sam = SpectralAngleLoss()
        self.spectral_tv = SpectralTVLoss()

    def get_weights(self, epoch):

        progress = epoch / self.total_epochs

        if progress < 0.3:

            return {
                'charbonnier': 1.0,
                'ssim': 0.05,
                'sam': 0.0,
                'spectral_tv': 0.0,
            }
        elif progress < 0.7:

            return {
                'charbonnier': 1.0,
                'ssim': 0.2,
                'sam': 0.05,
                'spectral_tv': 0.02,
            }
        else:

            return {
                'charbonnier': 1.0,
                'ssim': 0.3,
                'sam': 0.15,
                'spectral_tv': 0.05,
            }

    def forward(self, pred, target, epoch=0):
        weights = self.get_weights(epoch)

        losses = {
            'charbonnier': self.charbonnier(pred, target),
            'ssim': self.ssim(pred, target),
            'sam': self.sam(pred, target),
            'spectral_tv': self.spectral_tv(pred, target),
        }

        total_loss = sum(weights[k] * losses[k] for k in weights if weights[k] > 0)
        losses['total'] = total_loss
        losses['weights'] = weights

        return total_loss, losses


def get_loss(loss_type='hybrid', channels=28, **kwargs):

    if loss_type == 'charbonnier':
        return CharbonnierLoss()

    elif loss_type == 'hybrid':
        return HSIRestoreLoss(
            channels=channels,
            charbonnier_weight=1.0,
            ssim_weight=0.2,
            sam_weight=0.1,
            spectral_tv_weight=0.05,
            **kwargs
        )

    elif loss_type == 'psnr_focus':

        return HSIRestoreLoss(
            channels=channels,
            charbonnier_weight=1.0,
            ssim_weight=0.1,
            sam_weight=0.05,
            spectral_tv_weight=0.02,
            **kwargs
        )

    elif loss_type == 'sam_focus':

        return HSIRestoreLoss(
            channels=channels,
            charbonnier_weight=1.0,
            ssim_weight=0.1,
            sam_weight=0.2,
            spectral_tv_weight=0.1,
            **kwargs
        )

    elif loss_type == 'staged':
        return StagedLoss(channels=channels, **kwargs)

    else:
        return CharbonnierLoss()