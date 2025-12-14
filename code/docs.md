# Project Documentation: Image Smoothing Using Convolution

## What this project does
This project demonstrates convolution-based noise reduction on a real-world image. The input photograph is resized to 500×500, converted to grayscale, optionally corrupted with controlled noise, and then smoothed using two low-pass filters:

- Box (Average) filter
- Gaussian filter

Outputs, metrics, and frequency-domain spectra are generated to support objective and theoretical analysis.

## Mathematical basis
### 2D discrete convolution
For an input image f[m,n] and a kernel h[i,j], the filtered image g[m,n] is obtained via:

g[m,n] = Σ_i Σ_j f[m - i, n - j] · h[i,j]

This corresponds to an LTI (shift-invariant) filtering operation in the spatial domain. Low-pass filters suppress high-frequency components (often associated with noise) while preserving low-frequency structure.

### Box filter
A K×K box kernel assigns equal weights:

h[i,j] = 1 / K²

### Gaussian filter
A Gaussian kernel assigns higher weight near the center:

h[i,j] ∝ exp(-(i² + j²) / (2σ²)), normalized so ΣΣ h[i,j] = 1.

## Boundary handling
The implementation uses reflect padding by default to reduce border artifacts compared to zero-padding.

## Quantitative metrics
The application reports MSE, PSNR, and SSIM comparing filtered outputs against the clean grayscale reference.

- MSE: mean squared error
- PSNR: peak signal-to-noise ratio (higher is better)
- SSIM: structural similarity (closer to 1 is better)

## Frequency-domain analysis (FFT)
The log-magnitude spectrum is computed using FFT2 and FFTshift:

S(u,v) = log(1 + |FFTshift(FFT2(f[m,n]))|)

This helps visually confirm low-pass behavior.

## How to run
1) Install dependencies:
   pip install -r requirements.txt

2) Run the app:
   streamlit run app.py

## Outputs and history
All results are stored under code/results/ and each run is recorded in results/history.json for reproducibility.
