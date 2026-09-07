from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fracmag_rt as rt

VENTANA = 'FrAM - real-time motion magnification'

PANEL_W = 330
FILA_H = 46
BARRA_X0 = 14
BARRA_H = 7

PARAMS = [
    ('alpha',     'Gain',                     0.0, 200.0, 25.0, 1.0,   0,
     'Amplification factor'),
    ('nu',        'Fractional order',         0.0,   1.0,  0.30, 0.01, 2,
     'GL: 0=band-pass only, 1=derivative'),
    ('f_lo',      'Lower band (Hz)',          0.05,  5.0,  0.70, 0.05, 2,
     'Band-pass low cutoff'),
    ('f_hi',      'Upper band (Hz)',          0.10,  6.0,  2.00, 0.05, 2,
     'Band-pass high cutoff'),
    ('levels',    'Pyramid levels',           1,     5,    3,    1,    0,
     'Spatial frequency bands'),
    ('K',         'Fractional memory',        2,     48,   8,    1,    0,
     'Terms of the GL series'),
    ('percentil', 'Percentile of rho',        40,    95,   75,   1,    0,
     'Phase mask threshold'),
    ('chroma',    'Saturation',               0.0,   2.0,  1.00, 0.05, 2,
     'Color mode only'),
]

CASILLAS = [
    ('fram',   'FrAM  (unchecked: EVM)',         True),
    ('adapt',  'Adaptive gain',                  True),
    ('masc',   'Show rho mask',                  False),
    ('color',  'Color  (unchecked: grayscale)',  True),
    ('freeze', 'Freeze capture',                 False),
]

PRESETS = {
    '1': ('Facial pulse', dict(f_lo=0.70, f_hi=2.00, alpha=40.0, nu=0.25, levels=3)),
    '2': ('Breathing',    dict(f_lo=0.20, f_hi=0.80, alpha=30.0, nu=0.20, levels=4)),
    '3': ('Vibration',    dict(f_lo=1.00, f_hi=5.00, alpha=20.0, nu=0.30, levels=3)),
}

def alto_panel_necesario() -> int:
    return (48 + len(PARAMS) * FILA_H
            + 6 + 28 + len(CASILLAS) * 24
            + 4 + 30 + 5 * 15
            + 3 * 14 + 24)

class Control:

    def __init__(self):
        self.val = {c[0]: c[4] for c in PARAMS}
        self.chk = {c[0]: c[2] for c in CASILLAS}
        self.sel = 0
        self.arrastrando = None
        self._geom = {}
        self._geom_chk = {}
        self._alto = 0
        self.escala = 1.0
        self.escala_y = 1.0
        self.off_x = 0

    def _spec(self, clave):
        for c in PARAMS:
            if c[0] == clave:
                return c
        raise KeyError(clave)

    def fijar(self, clave, valor):
        _, _, mn, mx, _, paso, _, _ = self._spec(clave)
        v = round(float(valor) / paso) * paso
        self.val[clave] = float(np.clip(v, mn, mx))

    def mover(self, clave, pasos):
        _, _, _, _, _, paso, _, _ = self._spec(clave)
        self.fijar(clave, self.val[clave] + pasos * paso)

    def texto_valor(self, clave):
        _, _, _, _, _, _, dec, _ = self._spec(clave)
        v = self.val[clave]
        return f'{v:.{dec}f}' if dec else f'{int(round(v))}'

    def preset(self, tecla):
        if tecla in PRESETS:
            nombre, vals = PRESETS[tecla]
            for k, v in vals.items():
                self.fijar(k, v)
            return nombre
        return None

    def params(self, fps):
        p = dict(self.val)
        nyq = fps / 2.0

        p['f_hi'] = min(p['f_hi'], nyq * 0.98)
        p['f_lo'] = max(0.05, min(p['f_lo'], p['f_hi'] - 0.05))
        p['levels'] = int(round(p['levels']))
        p['K'] = int(round(p['K']))
        return p

    def on_mouse(self, evento, x, y, flags, _):

        if self.escala != 1.0:
            x = int(x * self.escala)
            y = int(y * self.escala_y)

        x -= self.off_x
        if evento == cv2.EVENT_LBUTTONDOWN:
            for i, (clave, g) in enumerate(self._geom.items()):
                x0, x1, yb = g
                if abs(y - yb) <= 12 and x0 - 6 <= x <= x1 + 6:
                    self.arrastrando = clave
                    self.sel = i
                    self._set_por_x(clave, x)
                    return
            for clave, (cx0, cx1, cy0, cy1) in self._geom_chk.items():
                if cx0 <= x <= cx1 and cy0 <= y <= cy1:
                    self.chk[clave] = not self.chk[clave]
                    return
        elif evento == cv2.EVENT_MOUSEMOVE and self.arrastrando:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self._set_por_x(self.arrastrando, x)
            else:
                self.arrastrando = None
        elif evento == cv2.EVENT_LBUTTONUP:
            self.arrastrando = None

    def _set_por_x(self, clave, x):
        x0, x1, _ = self._geom[clave]
        _, _, mn, mx, _, _, _, _ = self._spec(clave)
        t = float(np.clip((x - x0) / max(x1 - x0, 1), 0.0, 1.0))
        self.fijar(clave, mn + t * (mx - mn))

    def render(self, alto, extra=None):
        img = np.full((alto, PANEL_W, 3), 32, np.uint8)
        F = cv2.FONT_HERSHEY_SIMPLEX
        self._geom.clear()
        self._geom_chk.clear()

        cv2.putText(img, 'PARAMETERS', (BARRA_X0, 22), F, 0.52,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.line(img, (BARRA_X0, 30), (PANEL_W - BARRA_X0, 30), (85, 85, 85), 1)

        y = 48
        for i, (clave, nombre, mn, mx, _, _, dec, desc) in enumerate(PARAMS):
            activo = (i == self.sel)
            gris_txt = (255, 235, 170) if activo else (215, 215, 215)

            if activo:
                cv2.rectangle(img, (4, y - 13), (PANEL_W - 5, y + FILA_H - 26),
                              (70, 60, 30), -1)
                cv2.rectangle(img, (4, y - 13), (PANEL_W - 5, y + FILA_H - 26),
                              (150, 130, 60), 1)

            cv2.putText(img, nombre, (BARRA_X0, y), F, 0.43, gris_txt, 1, cv2.LINE_AA)
            tv = self.texto_valor(clave)
            tw = cv2.getTextSize(tv, F, 0.46, 1)[0][0]
            cv2.putText(img, tv, (PANEL_W - BARRA_X0 - tw, y), F, 0.46,
                        (120, 235, 160), 1, cv2.LINE_AA)

            bx0, bx1, by = BARRA_X0, PANEL_W - BARRA_X0, y + 13
            cv2.rectangle(img, (bx0, by - BARRA_H // 2), (bx1, by + BARRA_H // 2),
                          (62, 62, 62), -1)
            t = (self.val[clave] - mn) / float(mx - mn)
            xf = int(bx0 + t * (bx1 - bx0))
            cv2.rectangle(img, (bx0, by - BARRA_H // 2), (xf, by + BARRA_H // 2),
                          (95, 175, 100) if not activo else (130, 210, 135), -1)
            cv2.circle(img, (xf, by), 6, (245, 245, 245), -1)
            cv2.circle(img, (xf, by), 6, (30, 30, 30), 1)
            self._geom[clave] = (bx0, bx1, by)

            cv2.putText(img, desc, (BARRA_X0, y + 28), F, 0.34,
                        (150, 150, 150), 1, cv2.LINE_AA)
            y += FILA_H

        y += 6
        cv2.line(img, (BARRA_X0, y - 8), (PANEL_W - BARRA_X0, y - 8), (85, 85, 85), 1)
        cv2.putText(img, 'OPTIONS', (BARRA_X0, y + 10), F, 0.46,
                    (235, 235, 235), 1, cv2.LINE_AA)
        y += 28
        for clave, etiqueta, _ in CASILLAS:
            on = self.chk[clave]
            x0, x1 = BARRA_X0, BARRA_X0 + 14
            cv2.rectangle(img, (x0, y - 10), (x1, y + 4),
                          (110, 220, 120) if on else (80, 80, 80), -1 if on else 1)
            if on:
                cv2.line(img, (x0 + 3, y - 3), (x0 + 6, y + 1), (25, 25, 25), 2)
                cv2.line(img, (x0 + 6, y + 1), (x1 - 2, y - 8), (25, 25, 25), 2)
            cv2.putText(img, etiqueta, (x1 + 10, y + 2), F, 0.40,
                        (225, 225, 225) if on else (150, 150, 150), 1, cv2.LINE_AA)
            self._geom_chk[clave] = (x0 - 4, PANEL_W - BARRA_X0, y - 12, y + 6)
            y += 24

        y += 4
        cv2.line(img, (BARRA_X0, y - 6), (PANEL_W - BARRA_X0, y - 6), (85, 85, 85), 1)
        cv2.putText(img, 'PRESETS   1 pulse  2 breath.  3 vibr.', (BARRA_X0, y + 12),
                    F, 0.38, (170, 200, 235), 1, cv2.LINE_AA)
        y += 30
        ayuda = ['TAB/up-down selects', 'left-right or +/- adjusts',
                 'e FrAM/EVM   m mask   a adapt.',
                 'g color/gray  SPACE freezes',
                 's save   r reset   q quit']

        limite = alto - (14 * len(extra) + 22 if extra else 8)
        for linea in ayuda:
            if y > limite:
                break
            cv2.putText(img, linea, (BARRA_X0, y), F, 0.35, (145, 145, 145), 1,
                        cv2.LINE_AA)
            y += 15

        if extra:
            y = alto - 14 * len(extra) - 6
            cv2.line(img, (BARRA_X0, y - 13), (PANEL_W - BARRA_X0, y - 13),
                     (85, 85, 85), 1)
            for linea in extra:
                cv2.putText(img, linea, (BARRA_X0, y), F, 0.36,
                            (200, 215, 235), 1, cv2.LINE_AA)
                y += 14
        return img

def main():
    ap = argparse.ArgumentParser(description='FrAM in real time (unified viewer)')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--video', default=None)
    ap.add_argument('--width', type=int, default=0,
                    help='processing width; 0 = native resolution')
    ap.add_argument('--gray', '--gris', dest='gris', action='store_true',
                    help='start in grayscale')
    ap.add_argument('--fps', type=float, default=None)
    args = ap.parse_args()

    if args.video:
        cap, fuente = cv2.VideoCapture(args.video), os.path.basename(args.video)
    else:
        cap, fuente = cv2.VideoCapture(args.camera, cv2.CAP_V4L2), f'webcam {args.camera}'
    if not cap.isOpened():
        print(f'ERROR: could not open {fuente}')
        return 1

    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps < 240.0):
        fps = 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ctl = Control()
    ctl.chk['color'] = not args.gris

    cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(VENTANA, ctl.on_mouse)

    motor, clave = None, None
    hist = deque(maxlen=30)
    ultimo, n_guard, aviso, t_aviso = None, 0, '', 0.0

    print(f'Source: {fuente}  {w0}x{h0} @ {fps:.1f} fps  '
          f'(processing width {args.width or w0})')
    print('Single window: video + parameter panel. Mouse or keyboard.')

    while True:
        t0 = time.perf_counter()

        if not ctl.chk['freeze']:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if args.width and args.width != frame.shape[1]:
                e = args.width / float(frame.shape[1])
                frame = cv2.resize(frame, (args.width, max(2, int(frame.shape[0] * e))),
                                   interpolation=cv2.INTER_AREA)
            ultimo = frame
        if ultimo is None:
            continue

        p = ctl.params(fps)
        color = ctl.chk['color']
        fram = ctl.chk['fram']
        entrada = ultimo if color else \
            cv2.cvtColor(ultimo, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        c = (round(p['f_lo'], 3), round(p['f_hi'], 3), round(p['nu'], 3),
             p['levels'], p['K'], color, entrada.shape)
        if motor is None or c != clave:
            cls = rt.FrAMStreamColor if color else rt.FrAMStream
            kw = dict(fps=fps, f_lo=p['f_lo'], f_hi=p['f_hi'], alpha=p['alpha'],
                      nu=p['nu'], levels=p['levels'], adaptive=ctl.chk['adapt'],
                      percentil=p['percentil'], K_max=p['K'], evm=not fram)
            if color:
                kw['chroma'] = p['chroma']
            motor = cls(**kw)
            clave = c
        else:
            motor.set_params(alpha=p['alpha'], adaptive=ctl.chk['adapt'],
                             percentil=p['percentil'], evm=not fram,
                             **({'chroma': p['chroma']} if color else {}))

        try:
            sal = motor.push(entrada)
            err = None
        except Exception as e:
            sal, err = entrada, f'{type(e).__name__}: {e}'

        a8 = lambda x: (x.copy() if x.dtype == np.uint8
                        else (np.clip(x, 0, 1) * 255).astype(np.uint8))
        izq, der = a8(entrada), a8(sal)
        if izq.ndim == 2:
            izq = cv2.cvtColor(izq, cv2.COLOR_GRAY2BGR)
        if der.ndim == 2:
            der = cv2.cvtColor(der, cv2.COLOR_GRAY2BGR)

        if ctl.chk['masc']:
            m = motor.mascara(0)
            if m is not None:
                m = cv2.resize(np.clip(m, 0, 1), (der.shape[1], der.shape[0]))
                calor = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                der = cv2.addWeighted(der, 0.45, calor, 0.55, 0)

        cv2.putText(izq, 'ORIGINAL', (8, izq.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(der, ('FrAM' if fram else 'EVM') + (' + rho' if ctl.chk['masc'] else ''),
                    (8, der.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (120, 255, 160) if fram else (140, 140, 255), 2, cv2.LINE_AA)

        video = np.hstack([izq, np.full((izq.shape[0], 3, 3), 70, np.uint8), der])
        obj = 1180
        f = obj / float(video.shape[1])
        if abs(f - 1.0) > 0.02:
            video = cv2.resize(video, None, fx=f, fy=f,
                               interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)

        dt = time.perf_counter() - t0
        hist.append(1.0 / max(dt, 1e-6))
        estado = [
            f"{'FrAM' if fram else 'EVM'} | {'color' if color else 'gray'} | "
            f"{'adaptive' if ctl.chk['adapt'] else 'fixed'}",
            f"{np.mean(hist):.1f} fps | {entrada.shape[1]}x{entrada.shape[0]}",
        ]
        if aviso and time.time() - t_aviso < 2.0:
            estado.append(aviso)
        elif err:
            estado.append(err[:38])

        alto = max(video.shape[0], alto_panel_necesario())
        ctl.off_x = video.shape[1]
        panel = ctl.render(alto, extra=estado)
        if video.shape[0] < alto:
            video = np.vstack([video,
                               np.full((alto - video.shape[0], video.shape[1], 3),
                                       32, np.uint8)])
        vista = np.hstack([video, panel])
        cv2.imshow(VENTANA, vista)

        try:
            _, _, wv, hv = cv2.getWindowImageRect(VENTANA)
            if wv > 0 and hv > 0:
                ctl.escala = vista.shape[1] / float(wv)
                ctl.escala_y = vista.shape[0] / float(hv)
        except Exception:
            ctl.escala = ctl.escala_y = 1.0

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == 9:
            ctl.sel = (ctl.sel + 1) % len(PARAMS)
        elif k in (82, ord('w')):
            ctl.sel = (ctl.sel - 1) % len(PARAMS)
        elif k in (84, ord('x')):
            ctl.sel = (ctl.sel + 1) % len(PARAMS)
        elif k in (81, ord('-'), ord('_')):
            ctl.mover(PARAMS[ctl.sel][0], -1)
        elif k in (83, ord('+'), ord('=')):
            ctl.mover(PARAMS[ctl.sel][0], +1)
        elif k == ord('e'):
            ctl.chk['fram'] = not ctl.chk['fram']
        elif k == ord('m'):
            ctl.chk['masc'] = not ctl.chk['masc']
        elif k == ord('a'):
            ctl.chk['adapt'] = not ctl.chk['adapt']
        elif k == ord('g'):
            ctl.chk['color'] = not ctl.chk['color']
            motor = None
        elif k == ord(' '):
            ctl.chk['freeze'] = not ctl.chk['freeze']
        elif k == ord('r'):
            motor = None
            aviso, t_aviso = 'engine reset', time.time()
        elif 32 <= k < 127 and chr(k) in PRESETS:
            nombre = ctl.preset(chr(k))
            aviso, t_aviso = f'preset: {nombre}', time.time()
        elif k == ord('s'):
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
            os.makedirs(d, exist_ok=True)
            ruta = os.path.join(d, f'live_captura_{n_guard:02d}.png')
            cv2.imwrite(ruta, vista)
            print('saved', ruta)
            aviso, t_aviso = 'snapshot saved', time.time()
            n_guard += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    sys.exit(main())
