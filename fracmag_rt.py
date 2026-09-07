from __future__ import annotations

import cv2
import numpy as np
from fracmag import gl_coefficients, gl_memory_length
from scipy import signal

class BandpassState:

    def __init__(self, shape, fps: float, f_lo: float, f_hi: float):
        nyq = fps / 2.0
        lo = max(f_lo / nyq, 1e-4)
        hi = min(f_hi / nyq, 0.999)
        if lo >= hi:
            hi = min(lo + 1e-3, 0.999)
        b, a = signal.butter(1, [lo, hi], btype='bandpass')
        self.b = (b / a[0]).astype(np.float32)
        self.a = (a / a[0]).astype(np.float32)
        z = np.zeros(shape, np.float32)
        self.x1, self.x2 = z.copy(), z.copy()
        self.y1, self.y2 = z.copy(), z.copy()

    def push(self, x: np.ndarray) -> np.ndarray:
        b, a = self.b, self.a
        y = (b[0] * x + b[1] * self.x1 + b[2] * self.x2
             - a[1] * self.y1 - a[2] * self.y2)
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y

class FractionalState:

    def __init__(self, shape, nu: float, K: int):
        self.nu = float(nu)
        self.K = int(max(1, K))
        self.w = gl_coefficients(self.nu, self.K).astype(np.float32)
        self.buf = np.zeros((self.K + 1,) + tuple(shape), np.float32)
        self.pos = 0
        self.n = 0

    def push(self, x: np.ndarray) -> np.ndarray:
        self.buf[self.pos] = x
        self.pos = (self.pos + 1) % (self.K + 1)
        self.n += 1

        idx = (self.pos - 1 - np.arange(self.K + 1)) % (self.K + 1)
        return np.tensordot(self.w, self.buf[idx], axes=(0, 0))

def pyr_down_up(img: np.ndarray, levels: int):
    g = [img]
    for _ in range(levels):
        g.append(cv2.pyrDown(g[-1]))
    bandas = []
    for i in range(levels):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        bandas.append(g[i] - up)
    return bandas, g[-1]

def pyr_collapse(bandas: list[np.ndarray], residuo: np.ndarray) -> np.ndarray:
    img = residuo
    for b in reversed(bandas):
        img = cv2.pyrUp(img, dstsize=(b.shape[1], b.shape[0])) + b
    return img

class FrAMStream:

    def __init__(self, fps: float = 30.0, f_lo: float = 0.7, f_hi: float = 2.0,
                 alpha: float = 25.0, nu: float = 0.3, levels: int = 3,
                 adaptive: bool = True, percentil: float = 75.0,
                 K_max: int = 24, tau: float = 0.9, evm: bool = False):
        self.fps, self.f_lo, self.f_hi = fps, f_lo, f_hi
        self.alpha, self.nu, self.levels = alpha, nu, levels
        self.adaptive, self.percentil = adaptive, percentil
        self.K_max = K_max
        self.tau = tau
        self.evm = evm
        self._shape = None
        self._clave_actual = None
        self._bp: list[BandpassState] = []
        self._fr: list[FractionalState] = []
        self._amp_ema: list[np.ndarray | None] = []
        self._sigma: list[float] = []
        self.K = 0
        self.n = 0

    def _clave(self, shape):
        return (shape, round(self.f_lo, 4), round(self.f_hi, 4),
                round(self.nu, 4), self.levels, self.K_max, round(self.fps, 3))

    def _reset(self, ejemplo: np.ndarray):
        bandas, _ = pyr_down_up(ejemplo, self.levels)
        K = int(min(self.K_max, max(1, gl_memory_length(self.nu))))
        self._bp = [BandpassState(b.shape, self.fps, self.f_lo, self.f_hi) for b in bandas]
        self._fr = [FractionalState(b.shape, self.nu, K) for b in bandas]
        self._amp_ema = [None] * len(bandas)
        self._sigma = [1e-3] * len(bandas)
        self._shape = ejemplo.shape
        self._clave_actual = self._clave(ejemplo.shape)
        self.K = K
        self.n = 0

    def set_params(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(f'unknown parameter: {k}')
            setattr(self, k, v)

    def push(self, frame: np.ndarray) -> np.ndarray:
        f = np.ascontiguousarray(frame, dtype=np.float32)

        if self._shape is None or self._clave(f.shape) != self._clave_actual:
            self._reset(f)

        bandas, residuo = pyr_down_up(f, self.levels)
        salida = []

        for i, b in enumerate(bandas):
            filt = self._bp[i].push(b)

            if not self.evm and self.nu > 1e-6:
                filt = self._fr[i].push(filt)

            if self.adaptive and not self.evm:
                amp = self._amplitud(b)
                if self._amp_ema[i] is None:
                    self._amp_ema[i] = amp
                    self._sigma[i] = float(np.percentile(amp, self.percentil)) + 1e-8
                else:
                    t = self.tau
                    self._amp_ema[i] = t * self._amp_ema[i] + (1.0 - t) * amp

                    if self.n % 8 == 0:
                        s = float(np.percentile(self._amp_ema[i], self.percentil)) + 1e-8
                        self._sigma[i] = 0.7 * self._sigma[i] + 0.3 * s
                a2 = self._amp_ema[i] ** 2
                rho = a2 / (a2 + self._sigma[i] ** 2)
                g = self.alpha * rho / (1.0 + self.nu * (1.0 - rho))
            else:
                g = self.alpha

            salida.append(b + g * filt)

        self.n += 1
        return pyr_collapse(salida, residuo)

    @staticmethod
    def _amplitud(banda: np.ndarray) -> np.ndarray:
        r1 = cv2.Sobel(banda, cv2.CV_32F, 1, 0, ksize=3) * 0.25
        r2 = cv2.Sobel(banda, cv2.CV_32F, 0, 1, ksize=3) * 0.25
        return cv2.sqrt(banda * banda + r1 * r1 + r2 * r2)

    def mascara(self, indice: int = 0) -> np.ndarray | None:
        if indice >= len(self._amp_ema) or self._amp_ema[indice] is None:
            return None
        a2 = self._amp_ema[indice] ** 2
        return a2 / (a2 + self._sigma[indice] ** 2)

class FrAMStreamColor:

    def __init__(self, chroma: float = 1.0, **kw):
        self.y = FrAMStream(**kw)
        self.chroma = chroma

    def set_params(self, **kw):
        if 'chroma' in kw:
            self.chroma = kw.pop('chroma')
        self.y.set_params(**kw)

    def push(self, bgr: np.ndarray) -> np.ndarray:
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
        Y = self.y.push(ycc[:, :, 0])
        ycc[:, :, 0] = np.clip(Y, 0, 1)
        if self.chroma != 1.0:
            ycc[:, :, 1] = 0.5 + (ycc[:, :, 1] - 0.5) * self.chroma
            ycc[:, :, 2] = 0.5 + (ycc[:, :, 2] - 0.5) * self.chroma
        return cv2.cvtColor((np.clip(ycc, 0, 1) * 255).astype(np.uint8),
                            cv2.COLOR_YCrCb2BGR)

    def mascara(self, indice: int = 0):
        return self.y.mascara(indice)
