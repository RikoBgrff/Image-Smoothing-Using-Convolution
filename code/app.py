from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from processing import (
    BoundaryConfig,
    FilterConfig,
    NoiseConfig,
    parameter_study,
    run_pipeline,
)
from history import HistoryItem, append_history, load_history, now_iso


APP_TITLE = "GROUP 7 - Image Smoothing Using Convolution"
GALLERY_DIR = "gallery"
RESULTS_DIR = "results"

os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normalize an array to [0,1] for visualization purposes.
    This does NOT change the underlying data, only display.
    """
    min_val = img.min()
    max_val = img.max()
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)


# -----------------------------
# Helpers
# -----------------------------
def list_gallery_images(gallery_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for name in os.listdir(gallery_dir):
        if name.lower().endswith(exts):
            files.append(name)
    files.sort()
    return files


def save_array_as_image(arr: np.ndarray, path: str) -> None:
    from PIL import Image
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)


def render_kernel(kernel: np.ndarray, title: str) -> None:
    st.markdown(f"**{title}**")
    st.code(np.array2string(kernel, precision=6, suppress_small=True))


def render_metrics(metrics: dict) -> None:
    st.markdown("### Quantitative Metrics (vs. clean grayscale reference)")
    for k, v in metrics.items():
        st.markdown(f"**{k}**")
        st.write(v)


def read_docs_md() -> str:
    path = os.path.join(os.path.dirname(__file__), "docs.md")
    if not os.path.exists(path):
        return "docs.md not found."
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("EEE309 / EEE311 Signals and Systems – Convolution-based smoothing with analysis, metrics, FFT, parameter study, and run history.")


# -----------------------------
# Sidebar: Input selection
# -----------------------------
st.sidebar.header("Input Selection")

source_mode = st.sidebar.radio(
    "Choose image source",
    ["Select from gallery", "Upload image"],
    index=0
)

gallery_files = list_gallery_images(GALLERY_DIR)
selected_gallery = None
uploaded = None

if source_mode == "Select from gallery":
    if not gallery_files:
        st.sidebar.warning("No images found in code/assets/. Add at least one image (e.g., realPhoto.jpg).")
    selected_gallery = st.sidebar.selectbox("Gallery images (code/gallery/)", gallery_files if gallery_files else ["(none)"])
else:
    uploaded = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])


# -----------------------------
# Sidebar: Processing parameters
# -----------------------------
st.sidebar.header("Processing Parameters")

kernel_size = st.sidebar.selectbox("Kernel size (odd)", [3, 5, 7, 9], index=0)
gauss_sigma = st.sidebar.slider("Gaussian σ", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

boundary_mode = st.sidebar.selectbox("Boundary handling", ["reflect", "edge", "constant"], index=0)

noise_mode = st.sidebar.selectbox("Noise injection", ["none", "gaussian", "salt_pepper"], index=0)
noise_gaussian_sigma = st.sidebar.slider("Noise σ (Gaussian)", 1.0, 50.0, 15.0, 1.0)
salt_pepper_prob = st.sidebar.slider("Salt/Pepper probability", 0.0, 0.2, 0.02, 0.005)


# -----------------------------
# Sidebar: Parameter study configuration
# -----------------------------
st.sidebar.header("Parameter Study")
do_study = st.sidebar.checkbox("Run Gaussian parameter study", value=False)
study_kernel_sizes = st.sidebar.multiselect("Kernel sizes", [3, 5, 7, 9], default=[3, 5, 7])
study_sigmas = st.sidebar.multiselect("Sigmas", [0.7, 1.0, 1.5, 2.0, 3.0], default=[0.7, 1.0, 1.5])


# -----------------------------
# Load image based on selection
# -----------------------------
def get_pil_image() -> Tuple[Image.Image | None, str, str]:
    """
    Returns (PIL image, source_type, source_name).
    """
    if source_mode == "Select from gallery":
        if not gallery_files or selected_gallery in (None, "(none)"):
            return None, "gallery", ""
        path = os.path.join(GALLERY_DIR, selected_gallery)
        return Image.open(path), "gallery", selected_gallery

    # upload
    if uploaded is None:
        return None, "upload", ""
    return Image.open(uploaded), "upload", getattr(uploaded, "name", "uploaded_image")


pil_img, source_type, source_name = get_pil_image()


# -----------------------------
# Run button
# -----------------------------
run = st.sidebar.button("Run Processing", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Run & Results", "History", "See the Documentation"])

# -----------------------------
# Tab 1: Run & Results
# -----------------------------
with tab1:
    if pil_img is None:
        st.info("Select a gallery image or upload an image to begin.")
    elif run:
        filt = FilterConfig(kernel_size=int(kernel_size), gaussian_sigma=float(gauss_sigma))
        noise_cfg = NoiseConfig(
            mode=noise_mode,
            gaussian_sigma=float(noise_gaussian_sigma),
            salt_pepper_prob=float(salt_pepper_prob),
        )
        boundary = BoundaryConfig(pad_mode=boundary_mode)

        result = run_pipeline(
            pil_img=pil_img,
            filt=filt,
            noise=noise_cfg,
            boundary=boundary,
            seed=42,
        )

        # Save outputs into assets/ with run timestamp prefix for uniqueness
        stamp = now_iso().replace(":", "-")
        out_resized = os.path.join(RESULTS_DIR, f"{stamp}_resized_500x500.jpg")
        out_gray    = os.path.join(RESULTS_DIR, f"{stamp}_grayscale.jpg")
        out_noisy   = os.path.join(RESULTS_DIR, f"{stamp}_noisy.jpg")
        out_box     = os.path.join(RESULTS_DIR, f"{stamp}_box_filtered.jpg")
        out_gauss   = os.path.join(RESULTS_DIR, f"{stamp}_gaussian_filtered.jpg")


        result.resized_rgb.save(out_resized)
        result.gray_pil.save(out_gray)
        save_array_as_image(result.noisy_gray, out_noisy)
        save_array_as_image(result.box_out, out_box)
        save_array_as_image(result.gauss_out, out_gauss)

        # Record history
        item = HistoryItem(
            timestamp_iso=stamp,
            source_type=source_type,
            source_name=source_name,
            kernel_size=int(kernel_size),
            gaussian_sigma=float(gauss_sigma),
            noise_mode=noise_mode,
            noise_gaussian_sigma=float(noise_gaussian_sigma),
            salt_pepper_prob=float(salt_pepper_prob),
            boundary_mode=boundary_mode,
            outputs={
                "resized": os.path.basename(out_resized),
                "grayscale": os.path.basename(out_gray),
                "noisy": os.path.basename(out_noisy),
                "box": os.path.basename(out_box),
                "gaussian": os.path.basename(out_gauss),
            },
            metrics=result.metrics,
        )
        append_history(RESULTS_DIR, item)

        # Layout: images
        st.subheader("Images")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.image(result.resized_rgb, caption="Resized RGB (500×500)")
        with c2:
            st.image(np.uint8(result.clean_gray), caption="Grayscale (clean reference)")
        with c3:
            st.image(np.uint8(result.noisy_gray), caption=f"Noisy image (mode: {noise_mode})")
        with c4:
            st.image(np.uint8(result.box_out), caption="Box filter output")
        with c5:
            st.image(np.uint8(result.gauss_out), caption="Gaussian filter output")

        st.divider()

        # Kernels
        st.subheader("Kernels (Impulse Responses)")
        kc1, kc2 = st.columns(2)
        with kc1:
            render_kernel(result.box_kernel, f"Box kernel (K={kernel_size})")
        with kc2:
            render_kernel(result.gauss_kernel, f"Gaussian kernel (K={kernel_size}, σ={gauss_sigma})")

        st.divider()

        # Metrics
        render_metrics(result.metrics)

        st.divider()

        # FFT Spectra
        st.subheader("Frequency-Domain Analysis (FFT log-magnitude spectra)")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.image(normalize_for_display(result.spectra["clean"]),caption="Spectrum: clean grayscale")
        with sc2:
           st.image(normalize_for_display(result.spectra["noisy"]), caption="Spectrum: noisy")
        with sc3:
           st.image(normalize_for_display(result.spectra["box"]), caption="Spectrum: box-filtered")
        with sc4:
            st.image(normalize_for_display(result.spectra["gaussian"]), caption="Spectrum: gaussian-filtered")


        st.divider()

        # Parameter study
        if do_study:
            st.subheader("Gaussian Parameter Study (Kernel Size & Sigma Sweep)")
            rows = parameter_study(
                pil_img=pil_img,
                kernel_sizes=[int(x) for x in study_kernel_sizes],
                sigmas=[float(x) for x in study_sigmas],
                noise=noise_cfg,
                boundary=boundary,
                seed=42,
            )

            # Sort by best PSNR desc
            rows_sorted = sorted(rows, key=lambda r: r["PSNR_vs_clean"], reverse=True)
            st.write("Top results (sorted by PSNR):")
            st.dataframe(rows_sorted, use_container_width=True)

            best = rows_sorted[0] if rows_sorted else None
            if best:
                st.info(
                    f"Best PSNR in sweep: K={int(best['kernel_size'])}, σ={best['sigma']}, "
                    f"PSNR={best['PSNR_vs_clean']:.3f}, SSIM={best['SSIM_vs_clean']:.4f}"
                )

    else:
        st.info("Set parameters in the sidebar and click 'Run Processing'.")


# -----------------------------
# Tab 2: History
# -----------------------------
with tab2:
    st.subheader("Past Runs")
    items = load_history(RESULTS_DIR)
    if not items:
        st.info("No history yet. Run processing at least once.")
    else:
        # list runs
        for idx, it in enumerate(items[:20]):
            with st.expander(f"{idx+1}) {it.timestamp_iso} | source={it.source_type}:{it.source_name} | K={it.kernel_size} σ={it.gaussian_sigma} | noise={it.noise_mode}", expanded=False):
                st.write({
                    "kernel_size": it.kernel_size,
                    "gaussian_sigma": it.gaussian_sigma,
                    "noise_mode": it.noise_mode,
                    "noise_gaussian_sigma": it.noise_gaussian_sigma,
                    "salt_pepper_prob": it.salt_pepper_prob,
                    "boundary_mode": it.boundary_mode,
                })

                st.markdown("**Metrics**")
                st.write(it.metrics)

                st.markdown("**Saved outputs (in code/results/)**")
                st.write(it.outputs)

                # show images if available
                cols = st.columns(5)
                keys = ["resized", "grayscale", "noisy", "box", "gaussian"]
                for c, k in zip(cols, keys):
                    filename = it.outputs.get(k, "")
                    path = os.path.join(RESULTS_DIR, filename)
                    if filename and os.path.exists(path):
                        c.image(path, caption=f"{k}: {filename}")
                    else:
                        c.write(f"{k}: (missing)")


# -----------------------------
# Tab 3: Documentation
# -----------------------------
with tab3:
    st.subheader("Documentation")
    st.markdown(read_docs_md())
