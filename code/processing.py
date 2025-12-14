"""
processing.py

Core signal/image processing utilities for:
- preprocessing
- noise injection
- kernel construction
- 2D discrete convolution (LTI, shift-invariant)
- metrics (MSE, PSNR, SSIM)
- FFT spectrum analysis
- parameter study

All functions are intentionally explicit and well-documented to match
a Signals & Systems / DSP reporting style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for noise injection experiments."""
    mode: str  # "none" | "gaussian" | "salt_pepper"
    gaussian_sigma: float = 15.0  # standard deviation in intensity domain (0..255)
    salt_pepper_prob: float = 0.02  # probability that a pixel becomes 0 or 255


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for smoothing filters."""
    kernel_size: int  # K (odd)
    gaussian_sigma: float  # σ in Gaussian kernel


@dataclass(frozen=True)
class BoundaryConfig:
    """
    Boundary handling for discrete convolution.

    "reflect" is recommended for image processing to reduce border artifacts.
    """
    pad_mode: str = "reflect"  # "reflect" | "constant" | "edge"


# -----------------------------
# Preprocessing
# -----------------------------
def load_and_preprocess_image(
    pil_img: Image.Image,
    size: Tuple[int, int] = (500, 500),
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """
    Load image, resize to fixed dimensions, convert to grayscale, return:
    - resized RGB image (PIL)
    - grayscale image (PIL)
    - grayscale as float64 ndarray f[m,n] in [0,255]

    In report terms, f[m,n] is the 2D discrete input signal.
    """
    img_resized = pil_img.resize(size)
    img_gray = ImageOps.grayscale(img_resized)
    f = np.asarray(img_gray, dtype=np.float64)
    return img_resized, img_gray, f


# -----------------------------
# Noise models
# -----------------------------
def add_noise(image: np.ndarray, cfg: NoiseConfig, seed: int | None = 42) -> np.ndarray:
    """
    Inject controlled noise into image for repeatable experiments.

    Gaussian noise model:
        f_noisy[m,n] = f[m,n] + w[m,n],  w ~ N(0, σ^2)

    Salt-and-pepper model:
        with probability p, set pixel to 0 or 255.

    Returns clipped image in [0,255] as float64.
    """
    if cfg.mode == "none":
        return image.copy()

    rng = np.random.default_rng(seed)

    if cfg.mode == "gaussian":
        noise = rng.normal(loc=0.0, scale=cfg.gaussian_sigma, size=image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 255)

    if cfg.mode == "salt_pepper":
        noisy = image.copy()
        p = cfg.salt_pepper_prob
        mask = rng.random(image.shape)
        noisy[mask < (p / 2)] = 0.0
        noisy[(mask >= (p / 2)) & (mask < p)] = 255.0
        return noisy

    raise ValueError(f"Unknown noise mode: {cfg.mode}")


# -----------------------------
# Kernels (impulse responses)
# -----------------------------
def box_kernel(kernel_size: int) -> np.ndarray:
    """
    Normalized Box (Average) kernel:

        h[i,j] = 1 / K^2  for all i,j in KxK

    Sum(h) = 1 ensures DC (average brightness) is preserved.
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer.")
    k = np.ones((kernel_size, kernel_size), dtype=np.float64)
    return k / (kernel_size**2)


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Normalized 2D Gaussian kernel sampled on a discrete grid.

    Continuous form:
        G(x,y) = (1/(2πσ^2)) exp( -(x^2 + y^2) / (2σ^2) )

    Discrete kernel is sampled and normalized so Sum(h)=1.
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    r = kernel_size // 2
    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    h = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    h /= np.sum(h)
    return h


# -----------------------------
# 2D Discrete Convolution (Spatial Domain)
# -----------------------------
def convolve2d(image: np.ndarray, kernel: np.ndarray, boundary: BoundaryConfig) -> np.ndarray:
    """
    2D discrete convolution (spatial domain), implemented explicitly.

    Implements (conceptually):
        g[m,n] = Σ_i Σ_j  f[m - i, n - j] * h[i,j]

    Notes:
    - For symmetric kernels used in smoothing (box/gaussian), correlation vs convolution
      differences are negligible in effect. We keep "convolution" naming for Signals & Systems context.
    - Boundary handling uses padding to define pixels outside the image support.
    """
    ih, iw = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Reflect padding reduces dark-border artifacts compared to zero-padding
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode=boundary.pad_mode)

    out = np.zeros_like(image, dtype=np.float64)
    for m in range(ih):
        for n in range(iw):
            region = padded[m : m + kh, n : n + kw]
            out[m, n] = np.sum(region * kernel)
    return out


# -----------------------------
# Metrics (MSE, PSNR, SSIM)
# -----------------------------
def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Squared Error: MSE = mean((x - y)^2)."""
    return float(np.mean((x - y) ** 2))


def psnr(x: np.ndarray, y: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio:
        PSNR = 20 log10(MAX / sqrt(MSE))
    Higher is better.
    """
    m = mse(x, y)
    if m == 0:
        return float("inf")
    return float(20.0 * np.log10(max_pixel / math.sqrt(m)))


def ssim(x: np.ndarray, y: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Structural Similarity Index (global SSIM approximation).

    This is a simplified SSIM computed over the whole image (not windowed SSIM).
    It is acceptable for course-level quantitative comparison without extra dependencies.

    SSIM(x,y) = ((2μxμy + C1)(2σxy + C2)) / ((μx^2 + μy^2 + C1)(σx^2 + σy^2 + C2))
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (0.01 * max_pixel) ** 2
    c2 = (0.03 * max_pixel) ** 2

    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(num / den)


# -----------------------------
# Frequency domain analysis (FFT)
# -----------------------------
def magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute log-magnitude spectrum of a 2D signal via FFT:

        S(u,v) = log(1 + |FFTshift(FFT2(f[m,n]))|)

    The log scale improves visibility.
    """
    F = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    return np.log1p(mag)


# -----------------------------
# Pipeline
# -----------------------------
@dataclass(frozen=True)
class PipelineResult:
    """All artifacts produced by a pipeline run."""
    resized_rgb: Image.Image
    gray_pil: Image.Image
    clean_gray: np.ndarray
    noisy_gray: np.ndarray
    box_out: np.ndarray
    gauss_out: np.ndarray
    box_kernel: np.ndarray
    gauss_kernel: np.ndarray
    spectra: Dict[str, np.ndarray]
    metrics: Dict[str, Dict[str, float]]


def run_pipeline(
    pil_img: Image.Image,
    filt: FilterConfig,
    noise: NoiseConfig,
    boundary: BoundaryConfig,
    seed: int = 42,
) -> PipelineResult:
    """
    End-to-end pipeline:
    1) preprocess (resize + grayscale)
    2) noise injection (optional)
    3) build kernels (box + gaussian)
    4) spatial convolution with boundary handling
    5) spectra for analysis
    6) metrics against clean grayscale reference
    """
    resized_rgb, gray_pil, f_clean = load_and_preprocess_image(pil_img)

    f_noisy = add_noise(f_clean, noise, seed=seed)

    h_box = box_kernel(filt.kernel_size)
    h_g = gaussian_kernel(filt.kernel_size, filt.gaussian_sigma)

    g_box = convolve2d(f_noisy, h_box, boundary)
    g_g = convolve2d(f_noisy, h_g, boundary)

    # Spectra (clean vs noisy vs filtered)
    spectra = {
        "clean": magnitude_spectrum(f_clean),
        "noisy": magnitude_spectrum(f_noisy),
        "box": magnitude_spectrum(g_box),
        "gaussian": magnitude_spectrum(g_g),
    }

    # Metrics w.r.t. clean reference (f_clean)
    metrics = {
        "noisy_vs_clean": {
            "MSE": mse(f_clean, f_noisy),
            "PSNR": psnr(f_clean, f_noisy),
            "SSIM": ssim(f_clean, f_noisy),
        },
        "box_vs_clean": {
            "MSE": mse(f_clean, g_box),
            "PSNR": psnr(f_clean, g_box),
            "SSIM": ssim(f_clean, g_box),
        },
        "gaussian_vs_clean": {
            "MSE": mse(f_clean, g_g),
            "PSNR": psnr(f_clean, g_g),
            "SSIM": ssim(f_clean, g_g),
        },
    }

    return PipelineResult(
        resized_rgb=resized_rgb,
        gray_pil=gray_pil,
        clean_gray=f_clean,
        noisy_gray=f_noisy,
        box_out=g_box,
        gauss_out=g_g,
        box_kernel=h_box,
        gauss_kernel=h_g,
        spectra=spectra,
        metrics=metrics,
    )


# -----------------------------
# Parameter study
# -----------------------------
def parameter_study(
    pil_img: Image.Image,
    kernel_sizes: List[int],
    sigmas: List[float],
    noise: NoiseConfig,
    boundary: BoundaryConfig,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """
    Sweep kernel size and sigma for Gaussian smoothing, compute metrics.

    Returns a list of rows (dicts), easy to render as a table in UI.
    """
    _, _, f_clean = load_and_preprocess_image(pil_img)
    f_noisy = add_noise(f_clean, noise, seed=seed)

    rows: List[Dict[str, float]] = []
    for k in kernel_sizes:
        for s in sigmas:
            h = gaussian_kernel(k, s)
            out = convolve2d(f_noisy, h, boundary)

            rows.append({
                "kernel_size": float(k),
                "sigma": float(s),
                "MSE_vs_clean": mse(f_clean, out),
                "PSNR_vs_clean": psnr(f_clean, out),
                "SSIM_vs_clean": ssim(f_clean, out),
            })
    return rows
