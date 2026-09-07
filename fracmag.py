from __future__ import annotations

import cv2
import numpy as np
from scipy import signal

def gl_coefficients(nu: float, K: int) -> np.ndarray:
    if K < 1:
        raise ValueError("K must be >= 1")
    w = np.zeros(K + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, K + 1):
        w[k] = w[k - 1] * (k - 1 - nu) / k
    return w

def gl_memory_length(nu: float, tol: float = 1e-3, max_K: int = 512) -> int:
    w = gl_coefficients(nu, max_K)
    mag = np.abs(w)
    cola = np.cumsum(mag[::-1])[::-1]
    idx = np.argmax(cola < tol * mag[0])
    return int(idx) if idx > 0 else max_K

def fractional_bandpass(
    x: np.ndarray,
    fps: float,
    f_lo: float,
    f_hi: float,
    nu: float = 0.7,
    axis: int = 0,
) -> np.ndarray:
    if not (0.0 <= nu <= 1.0):
        raise ValueError("nu must be in [0, 1]")

    nyq = fps / 2.0
    lo = max(f_lo / nyq, 1e-6)
    hi = min(f_hi / nyq, 0.999)
    if lo >= hi:
        raise ValueError(f"invalid band: [{f_lo}, {f_hi}] Hz with fps={fps}")

    x = np.asarray(x, dtype=np.float64)
    x = np.moveaxis(x, axis, 0)
    T = x.shape[0]

    b, a = signal.butter(1, [lo, hi], btype='bandpass')
    y = signal.filtfilt(b, a, x, axis=0)

    if nu > 1e-9:
        K = min(gl_memory_length(nu), T - 1)
        w = gl_coefficients(nu, K)
        out = np.zeros_like(y)
        for k, wk in enumerate(w):
            if abs(wk) < 1e-12:
                continue
            if k == 0:
                desplazado = y
            else:
                desplazado = np.empty_like(y)
                desplazado[k:] = y[:-k]
                desplazado[:k] = y[0]
            out += wk * desplazado
        y = out

    return np.moveaxis(y, 0, axis)

def fractional_response(nu: float, fps: float, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    f = np.linspace(1e-6, fps / 2, n)
    w = 2 * np.pi * f / fps
    H = (1 - np.exp(-1j * w)) ** nu
    return f, np.abs(H)

def build_laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gauss = [img.astype(np.float64)]
    for _ in range(levels):
        gauss.append(cv2.pyrDown(gauss[-1]))
    pyr = []
    for i in range(levels):
        siguiente = cv2.pyrUp(gauss[i + 1], dstsize=(gauss[i].shape[1], gauss[i].shape[0]))
        pyr.append(gauss[i] - siguiente)
    pyr.append(gauss[-1])
    return pyr

def collapse_laplacian_pyramid(pyr: list[np.ndarray]) -> np.ndarray:
    img = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(pyr[i].shape[1], pyr[i].shape[0])) + pyr[i]
    return img

def riesz_transform(band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kx = np.array([[-0.5, 0.0, 0.5]], dtype=np.float64)
    ky = kx.T
    r1 = cv2.filter2D(band, -1, kx, borderType=cv2.BORDER_REFLECT)
    r2 = cv2.filter2D(band, -1, ky, borderType=cv2.BORDER_REFLECT)
    return r1, r2

def local_amplitude_phase(band: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1, r2 = riesz_transform(band)
    amp = np.sqrt(band ** 2 + r1 ** 2 + r2 ** 2)
    r_mag = np.sqrt(r1 ** 2 + r2 ** 2)
    fase = np.arctan2(r_mag, band)
    orient = np.arctan2(r2, r1)
    return amp, fase, orient

def phase_reliability(amp: np.ndarray, percentil: float = 75.0) -> np.ndarray:
    sigma = np.percentile(amp, percentil) + 1e-8
    return amp ** 2 / (amp ** 2 + sigma ** 2)

def adaptive_gain(alpha: float, amp: np.ndarray, nu: float,
                  percentil: float = 75.0) -> np.ndarray:
    rho = phase_reliability(amp, percentil)
    return alpha * rho / (1.0 + nu * (1.0 - rho))

def fram_magnify(
    frames: np.ndarray,
    fps: float,
    f_lo: float,
    f_hi: float,
    alpha: float = 20.0,
    nu: float = 0.7,
    levels: int = 4,
    adaptive: bool = True,
    percentil: float = 75.0,
) -> np.ndarray:
    T = frames.shape[0]

    piramides = [build_laplacian_pyramid(frames[t], levels) for t in range(T)]
    n_bandas = len(piramides[0])
    salida = [[None] * n_bandas for _ in range(T)]

    for b in range(n_bandas):
        pila = np.stack([piramides[t][b] for t in range(T)], axis=0)

        if b == n_bandas - 1:
            for t in range(T):
                salida[t][b] = pila[t]
            continue

        filtrada = fractional_bandpass(pila, fps, f_lo, f_hi, nu=nu, axis=0)

        if adaptive:

            amp_media, _, _ = local_amplitude_phase(pila.mean(axis=0))
            g = adaptive_gain(alpha, amp_media, nu, percentil)
        else:
            g = alpha

        for t in range(T):
            salida[t][b] = pila[t] + g * filtrada[t]

    return np.stack([collapse_laplacian_pyramid(salida[t]) for t in range(T)], axis=0)

def evm_magnify(
    frames: np.ndarray,
    fps: float,
    f_lo: float,
    f_hi: float,
    alpha: float = 20.0,
    levels: int = 4,
    order: int = 1,
) -> np.ndarray:
    T = frames.shape[0]
    piramides = [build_laplacian_pyramid(frames[t], levels) for t in range(T)]
    n_bandas = len(piramides[0])
    salida = [[None] * n_bandas for _ in range(T)]

    nyq = fps / 2.0
    b_coef, a_coef = signal.butter(
        order, [max(f_lo / nyq, 1e-6), min(f_hi / nyq, 0.999)], btype='bandpass')

    for b in range(n_bandas):
        pila = np.stack([piramides[t][b] for t in range(T)], axis=0)
        if b == n_bandas - 1:
            for t in range(T):
                salida[t][b] = pila[t]
            continue
        filtrada = signal.filtfilt(b_coef, a_coef, pila, axis=0)
        for t in range(T):
            salida[t][b] = pila[t] + alpha * filtrada[t]

    return np.stack([collapse_laplacian_pyramid(salida[t]) for t in range(T)], axis=0)
