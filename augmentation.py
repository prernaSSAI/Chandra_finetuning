import random
from typing import Any, Dict, List
import numpy as np
from PIL import Image
import albumentations as A
from augraphy.augmentations import (
    BadPhotoCopy,
    DirtyDrum,
    LightingGradient,
    ShadowCast,
    Folding,
    NoisyLines,
)

NOISE_PARAMS = {
    "clahe": {
        "clip_limit": 4.0,
        "tile_grid_size": (8, 8),
    },
    "gamma": {
        "gamma_limit": (80, 130),
    },
    "brightness": {
        "brightness_limit": (-0.05,0.12),
    },
    "contrast": {
        "contrast_limit": (0.25, 0.35),
    },
    "sharpen": {
        "alpha": (0.25, 0.45),
        "lightness": (0.9, 1.1),
    },
}
NOISE_NAMES = [noise for noise in NOISE_PARAMS.keys()]  # names used by augmentation.py



def _build_transform(name: str, params: Dict[str, Any]) -> A.BasicTransform:

    if name == "gauss_noise":
        return A.GaussNoise(
            std_range=params.get("std_range", (0.04, 0.12)),
            p=1.0,
        )

    elif name == "blur":
        return A.Blur(
            blur_limit=params.get("blur_limit", (3, 7)),
            p=1.0,
        )

    elif name == "gaussian_blur":
        return A.GaussianBlur(
            blur_limit=params.get("blur_limit", (3, 7)),
            p=1.0,
        )

    elif name == "motion_blur":
        return A.MotionBlur(
            blur_limit=params.get("blur_limit", 7),
            p=1.0,
        )

    elif name == "brightness":
        return A.RandomBrightnessContrast(
            brightness_limit=params.get("brightness_limit", 0.2),
            contrast_limit=0,
            p=1.0,
        )

    elif name == "contrast":
        return A.RandomBrightnessContrast(
            brightness_limit=0,
            contrast_limit=params.get("contrast_limit", 0.2),
            p=1.0,
        )

    elif name == "saturation":
        return A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=params.get("sat_shift_limit", 30),
            val_shift_limit=0,
            p=1.0,
        )

    elif name == "hue":
        return A.HueSaturationValue(
            hue_shift_limit=params.get("hue_shift_limit", 20),
            sat_shift_limit=0,
            val_shift_limit=0,
            p=1.0,
        )

    elif name == "rotate":
        return A.Rotate(
            limit=params.get("limit", 15),
            border_mode=params.get("border_mode", 0),
            p=1.0,
        )

    elif name == "random_crop":
        if "height" not in params or "width" not in params:
            raise ValueError("random_crop requires height and width")

        return A.RandomCrop(
            height=params["height"],
            width=params["width"],
            p=1.0,
        )
    elif name == "clahe":
        return A.CLAHE(**params, p=1.0)

    elif name == "gamma":
        return A.RandomGamma(**params, p=1.0)

    elif name == "sharpen":
        return A.Sharpen(**params, p=1.0)

    else:
        raise ValueError(f"Unknown augmentation name: {name}")


def _pil_to_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype("uint8"), mode="RGB")


def apply_noise(
    image: Image.Image,
    noise_names: List[str],
    *,
    # rng: random.Random,
    noise_params: Dict[str, Dict[str, Any]] | None = None,
    # mode: str = "one",
) -> Image.Image:
    """
    Apply albumentations-based augmentations to a PIL image.

    - noise_names: list of augmentation names (see _build_transform)
    - rng: random.Random instance for deterministic sampling
    - noise_params: mapping name -> kwargs to pass to transform builder
    """
    noise_params = noise_params or {}
    if not noise_names:
        return image

    # Convert to numpy array
    img_arr = _pil_to_array(image)

    transforms = [
        _build_transform(name, noise_params.get(name, {})) for name in noise_names
    ]
    transform = A.Compose(transforms)
    # seed = rng.randrange(0, 2**32 - 1)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    augmented = transform(image=img_arr)["image"]
    return _array_to_pil(augmented)


def apply_augraphy_augmentation(
    image: Image.Image,
    augmenter,
    seed: int | None = None,
) -> Image.Image:
    """Apply exactly one Augraphy augmenter to a PIL image."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    arr = np.array(image.convert("RGB"))
    augmented = augmenter(arr)
    return Image.fromarray(augmented.astype("uint8"), mode="RGB")



def build_augraphy_augmenters():
    return [
        BadPhotoCopy(
            noise_type=1,
            noise_side="none",
            noise_iteration=(2, 3),
            noise_size=(2, 3),
            noise_sparsity=(0.15, 0.15),
            noise_concentration=(0.3, 0.3),
            blur_noise=-1,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,
            edge_effect=0,
        ),
        LightingGradient(
            light_position=None,
            direction=None,
            max_brightness=255,
            min_brightness=0,
            mode="gaussian",
            transparency=0.8,
            p=1,
        ),
        ShadowCast(
            shadow_side="right",
            shadow_vertices_range=(10,15),
            shadow_width_range=(0.8, 0.95),
            shadow_height_range=(0.05, 0.08),
            shadow_color=(0, 0, 0),
            shadow_opacity_range=(0.3, 0.4),
            shadow_iterations_range=(3, 5),
            shadow_blur_kernel_range=(401, 601),
            p=1.0,
        ),
        DirtyDrum(
            line_width_range=(0, 1),
            line_concentration=0.01,
            direction=2,
            noise_intensity=0.2,
            noise_value=(0, 1),
            ksize=(3, 3),
            sigmaX=0,
        ),
        Folding(
            fold_x=None,
            fold_deviation=(0, 0),
            fold_count=2,
            fold_noise=0.01,
            fold_angle_range=(0, 0),
            gradient_width=(0.1, 0.2),
            gradient_height=(0.01, 0.02),
            backdrop_color=(255, 255, 255),
            p=1,
        ),
        NoisyLines(
            noisy_lines_direction="random",
            noisy_lines_location="random",
            noisy_lines_number_range=(5, 10),
            noisy_lines_color=(0, 0, 0),
            noisy_lines_thickness_range=(1, 2),
            noisy_lines_random_noise_intensity_range=(0.01, 0.1),
            noisy_lines_length_interval_range=(0, 100),
            noisy_lines_gaussian_kernel_value_range=(3, 5),
            noisy_lines_overlay_method="ink_to_paper",
            p=1,
        ),
    ]
