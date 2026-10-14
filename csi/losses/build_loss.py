

from .l1_loss import (
    CharbonnierLoss,
    TVLoss,
    SpectralAngleLoss,
    SpectralTVLoss,
    SSIMLoss,
    FrequencyLoss,
    HSIRestoreLoss,
    StagedLoss,
)


def build_loss(cfg):
    loss_type = getattr(cfg.LOSSES, 'TYPE', 'hybrid')
    channels = cfg.DATASETS.WAVE_LENS

    if loss_type == 'legacy':
        return _build_legacy_loss(cfg)

    elif loss_type == 'staged':
        total_epochs = getattr(cfg.LOSSES, 'STAGED_TOTAL_EPOCHS', cfg.OPTIMIZER.MAX_EPOCH)
        return StagedLoss(channels=channels, total_epochs=total_epochs)

    else:  
        return HSIRestoreLoss(
            channels=channels,
            charbonnier_weight=getattr(cfg.LOSSES, 'CHARBONNIER_WEIGHT', cfg.LOSSES.L1_WEIGHT),
            ssim_weight=cfg.LOSSES.SSIM_WEIGHT if getattr(cfg.LOSSES, 'SSIM_LOSS', False) else 0.0,
            sam_weight=cfg.LOSSES.SAM_WEIGHT if getattr(cfg.LOSSES, 'SAM_LOSS', False) else 0.0,
            spectral_tv_weight=cfg.LOSSES.SPECTRAL_TV_WEIGHT if getattr(cfg.LOSSES, 'SPECTRAL_TV_LOSS', False) else 0.0,
            freq_weight=cfg.LOSSES.FREQ_WEIGHT if getattr(cfg.LOSSES, 'FREQ_LOSS', False) else 0.0,
            tv_weight=cfg.LOSSES.TV_WEIGHT if getattr(cfg.LOSSES, 'TV_LOSS', False) else 0.0,
            return_dict=False,
        )


def _build_legacy_loss(cfg):
    class LegacyLoss:
        def __init__(self, cfg):
            self.losses = []
            self.weights = []

            if cfg.LOSSES.L1_LOSS:
                eps = getattr(cfg.LOSSES, 'CHARBONNIER_EPS', getattr(cfg.LOSSES, 'CHAR_EPS', 1e-3))
                self.losses.append(CharbonnierLoss(eps=eps))
                weight = getattr(cfg.LOSSES, 'CHARBONNIER_WEIGHT', cfg.LOSSES.L1_WEIGHT)
                self.weights.append(weight)

            if cfg.LOSSES.TV_LOSS:
                self.losses.append(TVLoss())
                self.weights.append(cfg.LOSSES.TV_WEIGHT)

            if getattr(cfg.LOSSES, 'SAM_LOSS', False):
                eps = getattr(cfg.LOSSES, 'SAM_EPS', 1e-8)
                self.losses.append(SpectralAngleLoss(eps=eps))
                self.weights.append(cfg.LOSSES.SAM_WEIGHT)

            if getattr(cfg.LOSSES, 'SPECTRAL_TV_LOSS', False) or getattr(cfg.LOSSES, 'SPEC_TV_LOSS', False):
                weight = getattr(cfg.LOSSES, 'SPECTRAL_TV_WEIGHT', getattr(cfg.LOSSES, 'SPEC_TV_WEIGHT', 0.0))
                self.losses.append(SpectralTVLoss())
                self.weights.append(weight)

        def __call__(self, pred, target):
            total_loss = 0
            for loss_fn, weight in zip(self.losses, self.weights):
                if isinstance(loss_fn, TVLoss):
                    total_loss = total_loss + weight * loss_fn(pred)
                elif isinstance(loss_fn, SpectralTVLoss):
                    total_loss = total_loss + weight * loss_fn(pred, target)
                else:
                    total_loss = total_loss + weight * loss_fn(pred, target)
            return total_loss

    return LegacyLoss(cfg)


def build_loss_with_details(cfg):
    loss_type = getattr(cfg.LOSSES, 'TYPE', 'hybrid')
    channels = cfg.DATASETS.WAVE_LENS

    if loss_type == 'staged':
        total_epochs = getattr(cfg.LOSSES, 'STAGED_TOTAL_EPOCHS', cfg.OPTIMIZER.MAX_EPOCH)
        return StagedLoss(channels=channels, total_epochs=total_epochs)

    else:  
        return HSIRestoreLoss(
            channels=channels,
            charbonnier_weight=getattr(cfg.LOSSES, 'CHARBONNIER_WEIGHT', cfg.LOSSES.L1_WEIGHT),
            ssim_weight=cfg.LOSSES.SSIM_WEIGHT if getattr(cfg.LOSSES, 'SSIM_LOSS', False) else 0.0,
            sam_weight=cfg.LOSSES.SAM_WEIGHT if getattr(cfg.LOSSES, 'SAM_LOSS', False) else 0.0,
            spectral_tv_weight=cfg.LOSSES.SPECTRAL_TV_WEIGHT if getattr(cfg.LOSSES, 'SPECTRAL_TV_LOSS', False) else 0.0,
            freq_weight=cfg.LOSSES.FREQ_WEIGHT if getattr(cfg.LOSSES, 'FREQ_LOSS', False) else 0.0,
            tv_weight=cfg.LOSSES.TV_WEIGHT if getattr(cfg.LOSSES, 'TV_LOSS', False) else 0.0,
            return_dict=True,  
        )