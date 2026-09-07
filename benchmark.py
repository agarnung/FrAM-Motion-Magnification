from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def make_sequence(
    T: int = 96, H: int = 128, W: int = 128, fps: float = 30.0,
    f_signal: float = 2.0, amp_px: float = 0.2,
    noise_std: float = 0.0, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)

    d = amp_px * np.sin(2 * np.pi * f_signal * np.arange(T) / fps)

    frames = np.zeros((T, H, W))
    for t in range(T):

        patron = 0.5 + 0.35 * np.sin(2 * np.pi * (x - d[t]) / 16.0)

        fondo = 0.10 * np.sin(2 * np.pi * y / 37.0)
        frames[t] = np.clip(patron + fondo, 0, 1)

    if noise_std > 0:
        frames = np.clip(frames + rng.normal(0, noise_std, frames.shape), 0, 1)

    return frames, d

def estimate_displacement(frames: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    base = frames.mean(axis=0) if ref is None else ref
    Ix = np.gradient(base, axis=1)
    denom = np.sum(Ix ** 2)
    out = np.zeros(len(frames))
    for t in range(len(frames)):
        It = frames[t] - base
        out[t] = -np.sum(It * Ix) / (denom + 1e-12)
    return out

def amplification_factor(d_out: np.ndarray, d_in: np.ndarray) -> float:
    di = d_in - d_in.mean()
    do = d_out - d_out.mean()
    return float(np.sum(di * do) / (np.sum(di ** 2) + 1e-12))

def band_snr(frames: np.ndarray, fps: float, f_signal: float,
             ancho: float = 0.5) -> float:
    serie = estimate_displacement(frames)
    serie = serie - serie.mean()
    espectro = np.abs(np.fft.rfft(serie)) ** 2
    freqs = np.fft.rfftfreq(len(serie), 1.0 / fps)

    en_banda = (freqs > f_signal - ancho) & (freqs < f_signal + ancho)
    fuera = (~en_banda) & (freqs > 0)
    p_sig = espectro[en_banda].sum()
    p_noise = espectro[fuera].sum() + 1e-20
    return float(10 * np.log10(p_sig / p_noise))

def noise_gain(frames_out: np.ndarray, frames_in: np.ndarray) -> float:
    def hf_std(f):
        return np.std(np.diff(f, axis=0))
    return float(hf_std(frames_out) / (hf_std(frames_in) + 1e-12))
