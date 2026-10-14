from collections import defaultdict

import os
import random

import torch
from torch.utils.data import Dataset

import numpy as np
from scipy import io as sio

from glob import glob

from box import Box


def _cfg_get(obj, name, default=None):
    return getattr(obj, name, default)


DEFAULT_SOURCE_WAVE_MIN = 450.0
DEFAULT_SOURCE_WAVE_MAX = 650.0
DEFAULT_TARGET_WAVE_MIN = 450.0
DEFAULT_TARGET_WAVE_MAX = 650.0


def spectral_resample_linear(
    img,
    source_wavelengths,
    target_wavelengths,
):
    img = np.asarray(img, dtype=np.float32)
    source_wavelengths = np.asarray(
        source_wavelengths, dtype=np.float32
    ).reshape(-1)
    target_wavelengths = np.asarray(
        target_wavelengths, dtype=np.float32
    ).reshape(-1)

    if img.ndim != 3:
        raise ValueError(
            f"HSI should be 3D [H, W, C], but got shape {img.shape}"
        )

    if img.shape[-1] != source_wavelengths.size:
        raise ValueError(
            f"HSI has {img.shape[-1]} bands, but source_wavelengths "
            f"contains {source_wavelengths.size} values."
        )

    if source_wavelengths.size < 2:
        raise ValueError("At least two source wavelengths are required.")

    if target_wavelengths.size < 1:
        raise ValueError("At least one target wavelength is required.")

    if np.any(np.diff(source_wavelengths) <= 0):
        raise ValueError("source_wavelengths must be strictly increasing.")

    if target_wavelengths.size > 1 and np.any(
        np.diff(target_wavelengths) <= 0
    ):
        raise ValueError("target_wavelengths must be strictly increasing.")

    eps = 1e-6
    if target_wavelengths[0] < source_wavelengths[0] - eps:
        raise ValueError(
            f"Target minimum wavelength {target_wavelengths[0]:.4f} nm "
            f"is outside source range "
            f"[{source_wavelengths[0]:.4f}, "
            f"{source_wavelengths[-1]:.4f}] nm."
        )

    if target_wavelengths[-1] > source_wavelengths[-1] + eps:
        raise ValueError(
            f"Target maximum wavelength {target_wavelengths[-1]:.4f} nm "
            f"is outside source range "
            f"[{source_wavelengths[0]:.4f}, "
            f"{source_wavelengths[-1]:.4f}] nm."
        )

    right = np.searchsorted(
        source_wavelengths,
        target_wavelengths,
        side="left",
    )
    right = np.clip(right, 1, source_wavelengths.size - 1)
    left = right - 1

    lambda_left = source_wavelengths[left]
    lambda_right = source_wavelengths[right]

    weight_right = (
        (target_wavelengths - lambda_left)
        / (lambda_right - lambda_left + 1e-12)
    ).astype(np.float32)

    output = (
        img[..., left] * (1.0 - weight_right)
        + img[..., right] * weight_right
    )

    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(output, 0.0, 1.0).astype(np.float32)


def convert_hsi_to_target_bands(
    img,
    wave_len=26,
    source_wave_min=DEFAULT_SOURCE_WAVE_MIN,
    source_wave_max=DEFAULT_SOURCE_WAVE_MAX,
    target_wave_min=DEFAULT_TARGET_WAVE_MIN,
    target_wave_max=DEFAULT_TARGET_WAVE_MAX,
):
    img = np.asarray(img, dtype=np.float32)

    if img.ndim != 3:
        raise ValueError(
            f"HSI should be [H, W, C], but got shape {img.shape}"
        )

    source_bands = img.shape[-1]

    if source_bands == wave_len:
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    if source_bands != 28:
        raise ValueError(
            f"Expected a 28-band source HSI or an already converted "
            f"{wave_len}-band HSI, but got {source_bands} bands."
        )

    if wave_len != 26:
        raise ValueError(
            f"This interpolation setup is defined for 28 -> 26 bands, "
            f"but WAVE_LENS={wave_len}."
        )

    source_wavelengths = np.linspace(
        float(source_wave_min),
        float(source_wave_max),
        source_bands,
        dtype=np.float32,
    )
    target_wavelengths = np.linspace(
        float(target_wave_min),
        float(target_wave_max),
        wave_len,
        dtype=np.float32,
    )

    return spectral_resample_linear(
        img,
        source_wavelengths=source_wavelengths,
        target_wavelengths=target_wavelengths,
    )


def select_bands(img, wave_len=26, band_start=1):
    del band_start
    return convert_hsi_to_target_bands(
        img,
        wave_len=wave_len,
    )


def _to_float32_hsi(img, divisor=None, clip=True):
    img = np.asarray(img, dtype=np.float32)

    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise ValueError("HSI contains no finite values.")

    raw_max = float(finite.max())

    if divisor is not None and raw_max > 2.0:
        img = img / float(divisor)

    img = np.nan_to_num(
        img,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    if clip:
        img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32)


def _center_start(full, target):
    if full < target:
        raise ValueError(f"Cannot crop target={target} from full={full}")
    return (full - target) // 2


def generate_mask_3d_shift(mask_path, wave_len=26, band_start=1):
    mask = sio.loadmat(mask_path)
    if "mask_3d_shift" not in mask:
        raise KeyError(f"'mask_3d_shift' not found in {mask_path}")

    mask3d_shift = mask["mask_3d_shift"]  
    mask_bands = mask3d_shift.shape[-1]
    if mask_bands != wave_len:
        raise ValueError(
            f"Pre-shifted mask has {mask_bands} bands, but "
            f"WAVE_LENS={wave_len}. A shifted mask must not be spectrally "
            f"interpolated or cropped because every channel has a different "
            f"spatial offset. Please regenerate a {wave_len}-band shifted mask."
        )
    mask3d_shift = np.transpose(mask3d_shift, [2, 0, 1])  
    mask3d_shift = torch.from_numpy(mask3d_shift).to(torch.float32)

    return mask3d_shift


def generate_mask_3d(mask_path, wave_len):
    mask = sio.loadmat(mask_path)
    if "mask" not in mask:
        raise KeyError(f"'mask' not found in {mask_path}")

    mask = mask["mask"]
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, wave_len))
    mask3d = np.transpose(mask3d, [2, 0, 1])  
    mask3d = torch.from_numpy(mask3d).to(torch.float32)

    return mask3d


def _ensure_hwc(img, scene_path):
    img = np.asarray(img)

    if img.ndim != 3:
        raise ValueError(
            f"HSI in {scene_path} should be 3D, but got shape {img.shape}"
        )

    if img.shape[-1] in (26, 28):
        return img

    if img.shape[0] in (26, 28):
        return np.transpose(img, (1, 2, 0))

    raise ValueError(
        f"Cannot infer spectral dimension for {scene_path}. "
        f"Expected 26 or 28 bands, got shape {img.shape}."
    )


def _load_hsi_from_mat(scene_path):
    img_dict = sio.loadmat(scene_path)

    candidates = [
        ("img_expand", 65536.0),
        ("img", 65536.0),
        ("hsi", None),
        ("HSI", None),
        ("data_slice", 65536.0),
        ("cave_512_28", 65536.0),
        ("cave_512_26", 65536.0),
        ("ICVL_26", 4096.0),
    ]

    selected_key = None
    img = None

    for key, divisor in candidates:
        if key not in img_dict:
            continue

        selected_key = key
        img = _ensure_hwc(img_dict[key], scene_path)
        img = _to_float32_hsi(
            img,
            divisor=divisor,
            clip=True,
        )
        break

    if img is None:
        valid_keys = [
            k for k in img_dict.keys()
            if not k.startswith("__")
        ]
        raise KeyError(
            f"No valid HSI key found in {scene_path}. "
            f"Available keys: {valid_keys}"
        )

    return img, selected_key


def _collect_training_scenes(paths):
    scene_list = []

    for path in paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Training dataset directory does not exist: {path}"
            )

        current = sorted(glob(os.path.join(path, "*.mat")))
        if not current:
            raise FileNotFoundError(
                f"No .mat files found in training directory: {path}"
            )

        scene_list.extend(current)

    return sorted(scene_list)


def _print_spectral_grid(
    source_bands,
    target_bands,
    source_wave_min,
    source_wave_max,
    target_wave_min,
    target_wave_max,
):
    source_step = (
        (source_wave_max - source_wave_min)
        / max(source_bands - 1, 1)
    )
    target_step = (
        (target_wave_max - target_wave_min)
        / max(target_bands - 1, 1)
    )

    print(
        "spectral interpolation: "
        f"{source_bands} bands "
        f"[{source_wave_min:.1f}, {source_wave_max:.1f}] nm, "
        f"step={source_step:.6f} nm "
        f"-> {target_bands} bands "
        f"[{target_wave_min:.1f}, {target_wave_max:.1f}] nm, "
        f"step={target_step:.6f} nm"
    )


def LoadTraining(
    paths,
    debug=False,
    wave_len=26,
    band_start=1,
    source_wave_min=DEFAULT_SOURCE_WAVE_MIN,
    source_wave_max=DEFAULT_SOURCE_WAVE_MAX,
    target_wave_min=DEFAULT_TARGET_WAVE_MIN,
    target_wave_max=DEFAULT_TARGET_WAVE_MAX,
):
    del band_start

    imgs = []
    scene_list = _collect_training_scenes(paths)

    if debug:
        scene_list = scene_list[:20]

    print("training scenes:", len(scene_list))

    if wave_len == 26:
        _print_spectral_grid(
            source_bands=28,
            target_bands=26,
            source_wave_min=float(source_wave_min),
            source_wave_max=float(source_wave_max),
            target_wave_min=float(target_wave_min),
            target_wave_max=float(target_wave_max),
        )

    for scene_path in scene_list:
        img, hsi_key = _load_hsi_from_mat(scene_path)

        original_bands = img.shape[-1]

        img = convert_hsi_to_target_bands(
            img,
            wave_len=wave_len,
            source_wave_min=source_wave_min,
            source_wave_max=source_wave_max,
            target_wave_min=target_wave_min,
            target_wave_max=target_wave_max,
        )

        img = np.ascontiguousarray(
            np.clip(img, 0.0, 1.0).astype(np.float32)
        )

        imgs.append(img)

        dataset_name = os.path.basename(
            os.path.dirname(scene_path)
        )
        print(
            f"Scene {dataset_name}/{os.path.basename(scene_path)} "
            f"is loaded. key={hsi_key}, "
            f"bands={original_bands}->{img.shape[-1]}, "
            f"shape={img.shape}"
        )

    if len(imgs) == 0:
        raise RuntimeError("No training scenes were loaded.")

    return imgs


def LoadVal(
    path_val,
    wave_len=26,
    band_start=1,
    source_wave_min=DEFAULT_SOURCE_WAVE_MIN,
    source_wave_max=DEFAULT_SOURCE_WAVE_MAX,
    target_wave_min=DEFAULT_TARGET_WAVE_MIN,
    target_wave_max=DEFAULT_TARGET_WAVE_MAX,
):
    del band_start

    images = []
    data = {}

    if not os.path.isdir(path_val):
        raise FileNotFoundError(
            f"Validation directory does not exist: {path_val}"
        )

    scene_list = sorted(glob(os.path.join(path_val, "*.mat")))

    if not scene_list:
        raise FileNotFoundError(
            f"No .mat files found in validation directory: {path_val}"
        )

    for scene_path in scene_list:
        img, hsi_key = _load_hsi_from_mat(scene_path)
        original_bands = img.shape[-1]

        img = convert_hsi_to_target_bands(
            img,
            wave_len=wave_len,
            source_wave_min=source_wave_min,
            source_wave_max=source_wave_max,
            target_wave_min=target_wave_min,
            target_wave_max=target_wave_max,
        )

        img = np.ascontiguousarray(
            np.clip(img, 0.0, 1.0).astype(np.float32)
        )

        images.append(img)

        print(
            f"Validation scene {os.path.basename(scene_path)} "
            f"is loaded. key={hsi_key}, "
            f"bands={original_bands}->{img.shape[-1]}, "
            f"shape={img.shape}"
        )

    data["hsi"] = images
    return data


def LoadTSATestMeas(path_test):
    measurements = []
    meas_paths = sorted(glob(os.path.join(path_test, "*.mat")))

    for meas_path in meas_paths:
        meas = sio.loadmat(meas_path)["meas_real"].astype(np.float32)
        meas[meas < 0] = 0.0
        meas[meas > 1] = 1.0
        measurements.append(meas)

    measurements = torch.from_numpy(np.stack(measurements, axis=0)).to(torch.float32)

    return measurements


def LoadTestMeas(path_test, with_PAN=True):
    res = defaultdict()
    MeasCs = []
    MeasPs = [] if with_PAN else None

    meas_paths = sorted(glob(os.path.join(path_test, "*.mat")))

    for meas_path in meas_paths:
        meas = sio.loadmat(meas_path)

        if "MeasC" in meas:
            MeasC = meas["MeasC"].astype(np.float32)
        elif "meas_real" in meas:
            MeasC = meas["meas_real"].astype(np.float32)
        else:
            raise KeyError(f"No valid coded measurement key found in {meas_path}")

        MeasC[MeasC < 0] = 0.0
        MeasC[MeasC > 1] = 1.0
        MeasCs.append(MeasC)

        if with_PAN:
            if "MeasP" not in meas:
                raise KeyError(
                    f"WITH_PAN=True but 'MeasP' not found in {meas_path}. "
                    f"Set DATASETS.WITH_PAN=False or provide MeasP."
                )
            MeasP = meas["MeasP"].astype(np.float32)
            MeasP[MeasP < 0] = 0.0
            MeasP[MeasP > 1] = 1.0
            MeasPs.append(MeasP)

    res["MeasC"] = torch.from_numpy(np.stack(MeasCs, axis=0)).to(torch.float32)

    if with_PAN:
        res["MeasP"] = torch.from_numpy(np.stack(MeasPs, axis=0)).to(torch.float32)

    return res


def crop_mask_for_hsi(
    mask,
    target_h,
    target_w,
    step=1,
    wave_len=26,
    mask_type="mask_3d",
    direction="up",
    random_crop=False,
):
    if mask.dim() != 3:
        raise ValueError(f"mask should be [C, H, W], but got shape {tuple(mask.shape)}")

    shift_size = (wave_len - 1) * step
    _, mask_h, mask_w = mask.shape

    if mask_type == "mask_3d":
        need_h, need_w = target_h, target_w
    elif mask_type == "mask_3d_shift":
        if direction == "up":
            need_h, need_w = target_h + shift_size, target_w
        elif direction == "right":
            need_h, need_w = target_h, target_w + shift_size
        else:
            raise ValueError(f"Unsupported dispersion direction: {direction}")
    else:
        raise ValueError(f"Unsupported MASK_TYPE: {mask_type}")

    if mask_h == need_h and mask_w == need_w:
        return mask

    if mask_h < need_h or mask_w < need_w:
        raise ValueError(
            f"Mask size {tuple(mask.shape)} is smaller than required "
            f"[C, {need_h}, {need_w}] for target HSI [{target_h}, {target_w}], "
            f"mask_type={mask_type}, direction={direction}."
        )

    if random_crop:
        x = np.random.randint(0, mask_h - need_h + 1)
        y = np.random.randint(0, mask_w - need_w + 1)
    else:
        x = _center_start(mask_h, need_h)
        y = _center_start(mask_w, need_w)

    return mask[:, x:x + need_h, y:y + need_w]


class CSITrainDataset(Dataset):
    def __init__(self, cfg, crop_size=(350, 260)):
        super().__init__()

        self.cfg = cfg
        self.iteration = cfg.DATASETS.TRAIN.ITERATION
        self.crop_size = tuple(crop_size)
        self.augment = cfg.DATASETS.TRAIN.AUGMENT

        self.wave_len = cfg.DATASETS.WAVE_LENS
        self.step = cfg.DATASETS.STEP
        self.mask_type = cfg.DATASETS.MASK_TYPE
        self.direction = _cfg_get(cfg.DATASETS, "DISPERSION_DIRECTION", "up")

        self.band_start = _cfg_get(
            cfg.DATASETS.TRAIN,
            "BAND_START",
            _cfg_get(cfg.DATASETS, "BAND_START", 1)
        )

        self.source_wave_min = float(
            _cfg_get(
                cfg.DATASETS,
                "SOURCE_WAVE_MIN",
                DEFAULT_SOURCE_WAVE_MIN,
            )
        )
        self.source_wave_max = float(
            _cfg_get(
                cfg.DATASETS,
                "SOURCE_WAVE_MAX",
                DEFAULT_SOURCE_WAVE_MAX,
            )
        )
        self.target_wave_min = float(
            _cfg_get(
                cfg.DATASETS,
                "TARGET_WAVE_MIN",
                DEFAULT_TARGET_WAVE_MIN,
            )
        )
        self.target_wave_max = float(
            _cfg_get(
                cfg.DATASETS,
                "TARGET_WAVE_MAX",
                DEFAULT_TARGET_WAVE_MAX,
            )
        )

        self.imgs = LoadTraining(
            cfg.DATASETS.TRAIN.PATHS,
            cfg.DEBUG,
            wave_len=self.wave_len,
            band_start=self.band_start,
            source_wave_min=self.source_wave_min,
            source_wave_max=self.source_wave_max,
            target_wave_min=self.target_wave_min,
            target_wave_max=self.target_wave_max,
        )

        if cfg.DATASETS.MASK_TYPE == "mask_3d":
            self.mask = generate_mask_3d(cfg.DATASETS.TRAIN.MASK_PATH, self.wave_len)
        elif cfg.DATASETS.MASK_TYPE == "mask_3d_shift":
            self.mask = generate_mask_3d_shift(
                cfg.DATASETS.TRAIN.MASK_PATH,
                wave_len=self.wave_len,
                band_start=self.band_start
            )
        else:
            raise ValueError(f"Unsupported MASK_TYPE: {cfg.DATASETS.MASK_TYPE}")

        _, self.mask_h, self.mask_w = self.mask.shape
        self.len_images = len(self.imgs)

    def random_crop_hsi(self, img, crop_size=None):
        if crop_size is None:
            crop_size = self.crop_size

        crop_h, crop_w = crop_size
        h, w, c = img.shape

        if c != self.wave_len:
            raise ValueError(f"Expected {self.wave_len} bands, but got {c} bands.")

        if h < crop_h or w < crop_w:
            raise ValueError(f"Image size {img.shape} is smaller than crop size {crop_size}")

        x_index = np.random.randint(0, h - crop_h + 1)
        y_index = np.random.randint(0, w - crop_w + 1)

        patch = img[x_index:x_index + crop_h, y_index:y_index + crop_w, :]
        patch = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float()

        return patch

    def get_train_mask(self):
        crop_h, crop_w = self.crop_size

        mask = crop_mask_for_hsi(
            self.mask,
            target_h=crop_h,
            target_w=crop_w,
            step=self.step,
            wave_len=self.wave_len,
            mask_type=self.mask_type,
            direction=self.direction,
            random_crop=self.cfg.DATASETS.TRAIN.RANDOM_MASK,
        )

        return mask

    def __getitem__(self, idx):
        data = {}

        if self.augment:
            flag = random.randint(0, 1)

            if flag:
                index = np.random.randint(0, self.len_images)
                img = self.imgs[index]
                processed_image = self.random_crop_hsi(img)
                processed_image = augment_1(processed_image)

            else:
                crop_h, crop_w = self.crop_size
                half_h, half_w = crop_h // 2, crop_w // 2

                processed_image = torch.zeros(
                    (self.wave_len, crop_h, crop_w),
                    dtype=torch.float32
                )

                sample_list = np.random.randint(0, self.len_images, 4)
                patches = []
                for j in range(4):
                    patch = self.random_crop_hsi(
                        self.imgs[sample_list[j]],
                        crop_size=(half_h, half_w)
                    )
                    patches.append(patch)

                processed_image[:, :half_h, :half_w] = patches[0]
                processed_image[:, :half_h, half_w:] = patches[1]
                processed_image[:, half_h:, :half_w] = patches[2]
                processed_image[:, half_h:, half_w:] = patches[3]
                processed_image = augment_1(processed_image)

        else:
            index = np.random.randint(0, self.len_images)
            img = self.imgs[index]
            processed_image = self.random_crop_hsi(img)

        mask = self.get_train_mask()

        data["hsi"] = processed_image
        data["mask"] = mask

        return data

    def __len__(self):
        return self.iteration


def augment_1(x):
    rot180 = random.randint(0, 1)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)

    if rot180:
        x = torch.rot90(x, k=2, dims=(1, 2))

    if vFlip:
        x = torch.flip(x, dims=(1,))

    if hFlip:
        x = torch.flip(x, dims=(2,))

    return x


def augment_2(generate_gt, crop_size):
    c, h, w = generate_gt.shape[1], crop_size[0], crop_size[1]
    divid_point_h = crop_size[0] // 2
    divid_point_w = crop_size[1] // 2

    output_img = torch.zeros(c, h, w)
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]

    return output_img



def shift_batch(inputs, nC=26, step=1, direction="up"):
    B, C, row, col = inputs.shape
    if C != nC:
        raise ValueError(f"Input channels C={C} but nC={nC}")

    shift_size = (nC - 1) * step

    if direction == "right":
        outputs = torch.zeros(
            (B, nC, row, col + shift_size),
            dtype=inputs.dtype,
            device=inputs.device
        )
        for i in range(nC):
            outputs[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]

    elif direction == "up":
        outputs = torch.zeros(
            (B, nC, row + shift_size, col),
            dtype=inputs.dtype,
            device=inputs.device
        )
        for i in range(nC):
            start = (nC - 1 - i) * step
            outputs[:, i, start:start + row, :] = inputs[:, i, :, :]

    else:
        raise ValueError(f"Unsupported dispersion direction: {direction}")

    return outputs


def shift_back_batch(inputs, nC=26, step=1, direction="up"):
    B, row, col = inputs.shape
    shift_size = (nC - 1) * step

    if direction == "right":
        out_w = col - shift_size
        if out_w <= 0:
            raise ValueError(f"Invalid shifted width: col={col}, shift_size={shift_size}")

        output = torch.zeros(
            (B, nC, row, out_w),
            dtype=inputs.dtype,
            device=inputs.device
        )
        for i in range(nC):
            output[:, i, :, :] = inputs[:, :, step * i:step * i + out_w]

    elif direction == "up":
        out_h = row - shift_size
        if out_h <= 0:
            raise ValueError(f"Invalid shifted height: row={row}, shift_size={shift_size}")

        output = torch.zeros(
            (B, nC, out_h, col),
            dtype=inputs.dtype,
            device=inputs.device
        )
        for i in range(nC):
            start = (nC - 1 - i) * step
            output[:, i, :, :] = inputs[:, start:start + out_h, :]

    else:
        raise ValueError(f"Unsupported dispersion direction: {direction}")

    return output



def gen_meas_torch_batch(
    inputs,
    Phi,
    step,
    wave_len,
    mask_type="mask_3d",
    with_pan=True,
    with_noise=False,
    direction="up"
):
    data = {}
    B, nC, H, W = inputs.shape

    if nC != wave_len:
        raise ValueError(f"Input channels nC={nC}, but wave_len={wave_len}")

    if Phi.dim() == 3:
        Phi = Phi.unsqueeze(0)

    if mask_type == "mask_3d":
        modulated_hsi = Phi * inputs
        modulated_hsi_shift = shift_batch(
            modulated_hsi,
            step=step,
            nC=wave_len,
            direction=direction
        )

    elif mask_type == "mask_3d_shift":
        hsi_shift = shift_batch(
            inputs,
            step=step,
            nC=wave_len,
            direction=direction
        )
        modulated_hsi_shift = Phi * hsi_shift

    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")

    MeasC = torch.sum(modulated_hsi_shift, dim=1)

    if with_pan:
        MeasP = torch.sum(inputs, dim=1)

    if with_noise:
        MeasC = MeasC / nC * 2 * 1.2
        QE, bit = 0.4, 2048
        MeasC_noise = torch.tensor(
            np.random.binomial((MeasC.detach().cpu().numpy() * bit / QE).astype(int), QE)
        ).float()
        MeasC = MeasC_noise / bit
        MeasC = MeasC.to(inputs.device)

        data["MeasC"] = MeasC
        MeasH = shift_back_batch(MeasC, nC=wave_len, step=step, direction=direction)
        data["MeasH"] = MeasH

        if with_pan:
            MeasP = MeasP / nC * 1.2
            QE, bit = 0.4, 2048
            MeasP_noise = torch.tensor(
                np.random.binomial((MeasP.detach().cpu().numpy() * bit / QE).astype(int), QE)
            ).float()
            MeasP = MeasP_noise / bit
            MeasP = MeasP.to(inputs.device)
            data["MeasP"] = MeasP

    else:
        data["MeasC"] = MeasC

        MeasC_norm = MeasC / nC * 2
        MeasH = shift_back_batch(MeasC_norm, nC=wave_len, step=step, direction=direction)
        data["MeasH"] = MeasH

        if with_pan:
            data["MeasP"] = MeasP

    return data


if __name__ == "__main__":
    cfg = Box(
        {
            "DEBUG": True,
            "DATASETS":
            {
                "STEP": 1,
                "WAVE_LENS": 26,
                "BAND_START": 1,
                "SOURCE_WAVE_MIN": 450.0,
                "SOURCE_WAVE_MAX": 650.0,
                "TARGET_WAVE_MIN": 450.0,
                "TARGET_WAVE_MAX": 650.0,
                "DISPERSION_DIRECTION": "up",
                "MASK_TYPE": "mask_3d",
                "WITH_PAN": True,
                "TRAIN":
                {
                    "PATHS": [
                        "../../../datasets/CSI/CAVE_512_28",
                        "../../../datasets/CSI/KAIST_CVPR2021",
                    ],
                    "ITERATION": 1000,
                    "MASK_PATH": "../../../datasets/CSI/TSA_simu_data/mask.mat",
                    "RANDOM_MASK": False,
                    "AUGMENT": True,
                    "WITH_NOISE": False,
                    "CROP_SIZE": [350, 260],
                    "BAND_START": 1,
                },
                "VAL":
                {
                    "PATH": "../../../datasets/CSI/TSA_simu_data/Truth",
                    "MASK_PATH": "../../../datasets/CSI/TSA_simu_data/mask.mat"
                }
            },
            "DATALOADER":
            {
                "BATCH_SIZE": 1
            }
        }
    )

    train_mask = generate_mask_3d(cfg.DATASETS.TRAIN.MASK_PATH, wave_len=cfg.DATASETS.WAVE_LENS)
    print("raw mask:", train_mask.shape)

    dataset = CSITrainDataset(cfg, crop_size=cfg.DATASETS.TRAIN.CROP_SIZE)

    data = dataset[0]
    print("hsi.shape: ", data["hsi"].shape)
    print("mask.shape:", data["mask"].shape)

    meas = gen_meas_torch_batch(
        data["hsi"].unsqueeze(0),
        data["mask"].unsqueeze(0),
        step=cfg.DATASETS.STEP,
        wave_len=cfg.DATASETS.WAVE_LENS,
        mask_type=cfg.DATASETS.MASK_TYPE,
        with_pan=cfg.DATASETS.WITH_PAN,
        with_noise=False,
        direction=cfg.DATASETS.DISPERSION_DIRECTION
    )
    print("MeasC.shape:", meas["MeasC"].shape)
    print("MeasH.shape:", meas["MeasH"].shape)
    print("MeasP.shape:", meas["MeasP"].shape if "MeasP" in meas else None)
