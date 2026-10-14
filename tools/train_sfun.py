import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from csi.architectures import sfun
from csi.config import get_cfg
from csi.data import CSITrainDataset, generate_mask_3d_shift, gen_meas_torch_batch
from csi.engine.defaults import seed_everything
from csi.losses.build_loss import build_loss
from csi.metrics import torch_psnr, torch_ssim, torch_sam
from csi.utils.schedulers import get_cosine_schedule_with_warmup


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, ema, scenes, mask, cfg):
    model.eval()
    rows = []
    with ema.average_parameters():
        for scene in scenes["hsi"]:
            gt = torch.from_numpy(scene).permute(2, 0, 1).unsqueeze(0).cuda()
            meas = gen_meas_torch_batch(
                gt, mask, cfg.DATASETS.STEP, cfg.DATASETS.WAVE_LENS,
                mask_type=cfg.DATASETS.MASK_TYPE, with_pan=True,
                with_noise=False, direction="right",
            )
            meas.update(mask=mask, PhiC=mask, PhiP=torch.ones_like(gt))
            pred = model(meas)[0]
            rows.append([float(torch_psnr(pred, gt[0])),
                         float(torch_ssim(pred, gt[0])),
                         float(torch_sam(pred, gt[0]))])
    model.train()
    values = np.asarray(rows)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite validation metric")
    return dict(zip(("psnr", "ssim", "sam"), values.mean(0).tolist())), rows


def meets_target(metrics):
    return (metrics["psnr"] >= 44.525 and metrics["ssim"] >= 0.9915
            and metrics["sam"] <= 4.5645)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/sfun.yaml")
    parser.add_argument("--output", default="outputs/train")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stop-on-target", action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.epochs is not None:
        cfg.OPTIMIZER.MAX_EPOCH = args.epochs
    cfg.freeze()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if cfg.DATASETS.DISPERSION_DIRECTION != "right":
        raise ValueError("This reference SFUN implementation uses right dispersion")
    seed_everything(cfg.SEED, deterministic=cfg.DETERMINISTIC)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.eval_only and (out / "sfun_last.pth").exists():
        raise FileExistsError("Use a new output directory; previous training exists")
    (out / "config.yaml").write_text(cfg.dump())
    log = (out / ("evaluation.log" if args.eval_only else "train.log")).open("a", buffering=1)

    def report(message):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + message
        print(line, flush=True)
        print(line, file=log, flush=True)

    model = sfun(cfg).cuda()
    source = args.checkpoint or cfg.PRETRAINED_CKPT_PATH
    ckpt = torch.load(source, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
        ema.to(device="cuda")
    else:
        report("Checkpoint contains inference parameters; initialize EMA from loaded model")
    report(f"Strictly loaded {len(model.state_dict())} keys from {source}; class={type(model).__module__}.{type(model).__name__}")
    dataset = CSITrainDataset(cfg, crop_size=cfg.DATASETS.TRAIN.CROP_SIZE)
    if dataset.len_images < 2:
        raise ValueError("At least two CAVE scenes are required for training and validation")
    val_indices = set(random.Random(3407).sample(
        range(dataset.len_images), max(1, int(dataset.len_images * 0.1))))
    scenes = {"hsi": [img for i, img in enumerate(dataset.imgs) if i in val_indices]}
    dataset.imgs = [img for i, img in enumerate(dataset.imgs) if i not in val_indices]
    dataset.len_images = len(dataset.imgs)
    mask = generate_mask_3d_shift(cfg.DATASETS.VAL.MASK_PATH, wave_len=28, band_start=0).cuda().unsqueeze(0)
    height = mask.shape[-2]
    width = mask.shape[-1] - 27 * cfg.DATASETS.STEP
    for i, img in enumerate(scenes["hsi"]):
        if img.shape[0] < height or img.shape[1] < width:
            raise ValueError(f"CAVE scene {img.shape} is smaller than validation size {(height, width)}")
        top = (img.shape[0] - height) // 2
        left = (img.shape[1] - width) // 2
        scenes["hsi"][i] = np.ascontiguousarray(img[top:top + height, left:left + width, :])
    metrics, rows = evaluate(model, ema, scenes, mask, cfg)
    report(f"Initial validation: {json.dumps(metrics)}")
    (out / "initial_metrics.json").write_text(json.dumps({"metrics": metrics, "per_scene": rows, "source": str(source)}, indent=2))
    if args.eval_only:
        log.close()
        return

    ema.copy_to(model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR, betas=(0.9, 0.999))
    steps = cfg.DATASETS.TRAIN.ITERATION // cfg.DATALOADER.BATCH_SIZE
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=steps,
        num_training_steps=steps * cfg.OPTIMIZER.MAX_EPOCH, eta_min=1e-6)
    loss_fn = build_loss(cfg).cuda()
    best = dict(metrics)
    best_epoch = 0
    best_psnr = metrics["psnr"]
    history = [{"epoch": 0, **metrics}]

    def export(name, epoch, scores):
        with ema.average_parameters():
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        atomic_save({"model": state, "model_name": "sfun", "model_class": "csi.architectures.sfun.sfun",
                     "epoch": epoch, "metrics": scores, "config": cfg.dump(),
                     "weights_kind": "ema", "pretrained_source": str(source)}, out / name)

    export("sfun.pth", 0, metrics)
    export("sfun_best_psnr.pth", 0, metrics)
    loader = DataLoader(dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                        num_workers=cfg.DATALOADER.NUM_WORKERS,
                        pin_memory=cfg.DATALOADER.PIN_MEMORY, drop_last=True)
    report(f"Training {len(dataset.imgs)} scenes; {steps} iterations/epoch; max_epochs={cfg.OPTIMIZER.MAX_EPOCH}")
    final_epoch = 0
    for epoch in range(1, cfg.OPTIMIZER.MAX_EPOCH + 1):
        begin = time.monotonic()
        total = 0.0
        model.train()
        for step, data in enumerate(loader, 1):
            if step > steps:
                break
            data = {k: v.cuda(non_blocking=True) for k, v in data.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(data)
            loss = loss_fn(pred, data["hsi"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            loss.backward()
            if cfg.OPTIMIZER.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2, error_if_nonfinite=True)
            optimizer.step()
            ema.update()
            scheduler.step()
            total += loss.item()
            if step == 1 or step % 100 == 0:
                report(f"Epoch {epoch} step {step}/{steps}: loss={total / step:.6f}, lr={scheduler.get_last_lr()[0]:.8f}, elapsed={time.monotonic()-begin:.1f}s")
        metrics, rows = evaluate(model, ema, scenes, mask, cfg)
        final_epoch = epoch
        report(f"Epoch {epoch} validation: {json.dumps(metrics)}")
        history.append({"epoch": epoch, "loss": total / steps, **metrics})
        if metrics["sam"] < best["sam"]:
            best, best_epoch = dict(metrics), epoch
            export("sfun.pth", epoch, metrics)
            report(f"Saved sfun.pth: best SAM at epoch {epoch}")
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            export("sfun_best_psnr.pth", epoch, metrics)
        atomic_save({"model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "epoch": epoch, "model_name": "sfun", "config": cfg.dump(),
                     "metrics": metrics, "best_metrics": best,
                     "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}},
                    out / "sfun_last.pth")
        (out / "metrics.json").write_text(json.dumps(history, indent=2))
        if args.stop_on_target and best_epoch > 0 and meets_target(best):
            report("Reference target reached by newly trained weights; stopping.")
            break
    summary = {"status": "completed", "epochs_trained": final_epoch, "best_epoch": best_epoch,
               "best_metrics": best, "target_met": meets_target(best),
               "selection": "minimum validation SAM (same as reference)",
               "checkpoint": str((out / "sfun.pth").resolve()),
               "stop_on_target": args.stop_on_target}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report(json.dumps(summary))
    log.close()


if __name__ == "__main__":
    main()
