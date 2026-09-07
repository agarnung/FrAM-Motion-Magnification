from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:
    sys.exit('ERROR: tkinter is missing.  Install it with:  sudo apt install python3-tk\n'
             '(tkinter-free alternative: python3 live_fram.py)')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fracmag_rt as rt

PARAMS = [
    ('alpha',     'Gain  α',                0.0, 200.0, 25.0, 1.0,
     'Base amplification factor'),
    ('nu',        'Fractional order  ν',     0.0,   1.0,  0.30, 0.01,
     'ν=0 band-pass only · ν=1 full derivative · useful 0.0–0.3'),
    ('f_lo',      'Lower band  f_lo',        0.05,  5.0,  0.70, 0.05,
     'Lower cutoff of the temporal band (Hz)'),
    ('f_hi',      'Upper band  f_hi',        0.10,  6.0,  2.00, 0.05,
     'Upper cutoff of the temporal band (Hz)'),
    ('levels',    'Pyramid levels',          1,     5,    3,    1,
     'Spatial frequency bands'),
    ('K',         'Fractional memory  K',    2,    48,    8,    1,
     'Terms of the Grünwald-Letnikov series; controls the cost'),
    ('percentil', 'Percentile of ρ',        40,    95,   75,    1,
     'σ threshold of the phase reliability mask'),
]

PRESETS = {
    'Facial pulse':      dict(f_lo=0.70, f_hi=2.00, alpha=30.0, nu=0.25, levels=3),
    'Breathing':         dict(f_lo=0.20, f_hi=0.80, alpha=20.0, nu=0.20, levels=4),
    'Mech. vibration':   dict(f_lo=1.00, f_hi=5.00, alpha=15.0, nu=0.30, levels=3),
}

class Estado:

    def __init__(self):
        self.lock = threading.Lock()

        self.pendiente: np.ndarray | None = None
        self.ultimo_original: np.ndarray | None = None
        self.color = True
        self.resultado: np.ndarray | None = None
        self.mascara: np.ndarray | None = None
        self.parar = False
        self.congelado = False
        self.metodo_fram = True
        self.ver_mascara = False
        self.adaptativa = True
        self.params = {k: v for k, _, _, _, v, _, _ in PARAMS}
        self.fps = 30.0
        self.t_proceso = 0.0
        self.n_proceso = 0
        self.error: str | None = None

    def leer_params(self):
        with self.lock:
            return dict(self.params), self.metodo_fram, self.adaptativa

def hilo_captura(est: Estado, cap, ancho_proc: int, es_video: bool):
    while True:
        with est.lock:
            if est.parar:
                return
            congelado = est.congelado
        if congelado:
            time.sleep(0.03)
            continue

        ok, frame = cap.read()
        if not ok:
            if es_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        if ancho_proc and ancho_proc != w:
            esc = ancho_proc / float(w)
            frame = cv2.resize(frame, (ancho_proc, max(2, int(h * esc))),
                               interpolation=cv2.INTER_AREA)

        with est.lock:
            color = est.color

            est.pendiente = frame if color else \
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            est.ultimo_original = frame
        time.sleep(0.001)

def hilo_proceso(est: Estado):
    motor = None
    clave = None
    while True:
        with est.lock:
            if est.parar:
                return
            p = dict(est.params)
            fram, adaptativa = est.metodo_fram, est.adaptativa
            ver_masc = est.ver_mascara
            frame = est.pendiente
            est.pendiente = None
            fps, color = est.fps, est.color

        if frame is None:
            time.sleep(0.002)
            continue

        f_hi = min(p['f_hi'], fps / 2.0 * 0.98)
        f_lo = max(0.05, min(p['f_lo'], f_hi - 0.05))
        c = (round(f_lo, 3), round(f_hi, 3), round(p['nu'], 3),
             int(p['levels']), int(p['K']), color, frame.shape)
        if motor is None or c != clave:
            cls = rt.FrAMStreamColor if color else rt.FrAMStream
            motor = cls(fps=fps, f_lo=f_lo, f_hi=f_hi, alpha=p['alpha'],
                        nu=p['nu'], levels=int(p['levels']),
                        adaptive=adaptativa, percentil=p['percentil'],
                        K_max=int(p['K']), evm=not fram)
            clave = c
        else:
            motor.set_params(alpha=p['alpha'], adaptive=adaptativa,
                             percentil=p['percentil'], evm=not fram)

        t0 = time.perf_counter()
        try:
            sal = motor.push(frame)
            masc = motor.mascara(0) if ver_masc else None
            err = None
        except Exception as e:
            sal, masc, err = None, None, f'{type(e).__name__}: {e}'
        dt = time.perf_counter() - t0

        with est.lock:
            if sal is not None:
                est.resultado = sal
                est.mascara = masc
            est.t_proceso = dt
            est.n_proceso += 1
            est.error = err

class App:
    def __init__(self, root, est: Estado):
        self.root = root
        self.est = est
        self.escalas = {}
        self.etiquetas_valor = {}
        self._foto = None

        root.title('FrAM — live motion magnification')
        root.protocol('WM_DELETE_WINDOW', self.cerrar)

        marco = ttk.Frame(root, padding=8)
        marco.grid(row=0, column=0, sticky='nsew')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        izq = ttk.Frame(marco)
        izq.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.lienzo = tk.Label(izq, background='#101010')
        self.lienzo.grid(row=0, column=0, sticky='nsew')
        self.estado_txt = ttk.Label(izq, text='starting…', font=('TkDefaultFont', 9))
        self.estado_txt.grid(row=1, column=0, sticky='w', pady=(6, 0))

        der = ttk.Frame(marco)
        der.grid(row=0, column=1, sticky='ns')

        ttk.Label(der, text='Parameters', font=('TkDefaultFont', 11, 'bold')) \
            .grid(row=0, column=0, sticky='w', pady=(0, 6))

        fila = 1
        for clave, etiqueta, mn, mx, ini, res, desc in PARAMS:
            cab = ttk.Frame(der)
            cab.grid(row=fila, column=0, sticky='ew', pady=(6, 0))
            ttk.Label(cab, text=etiqueta, font=('TkDefaultFont', 9, 'bold')) \
                .grid(row=0, column=0, sticky='w')
            val = ttk.Label(cab, text=self._fmt(clave, ini),
                            font=('TkMonospaceFont', 9), foreground='#0a58ca')
            val.grid(row=0, column=1, sticky='e', padx=(10, 0))
            cab.columnconfigure(1, weight=1)
            self.etiquetas_valor[clave] = val

            esc = tk.Scale(der, from_=mn, to=mx, resolution=res,
                           orient='horizontal', length=290, showvalue=False,
                           command=lambda v, c=clave: self.cambiar(c, v))
            esc.set(ini)
            esc.grid(row=fila + 1, column=0, sticky='ew')
            self.escalas[clave] = esc

            ttk.Label(der, text=desc, font=('TkDefaultFont', 8),
                      foreground='#666', wraplength=290) \
                .grid(row=fila + 2, column=0, sticky='w')
            fila += 3

        sep = ttk.Separator(der, orient='horizontal')
        sep.grid(row=fila, column=0, sticky='ew', pady=8)
        fila += 1

        self.var_metodo = tk.BooleanVar(value=True)
        self.var_adapt = tk.BooleanVar(value=True)
        self.var_masc = tk.BooleanVar(value=False)
        self.var_congelar = tk.BooleanVar(value=False)

        ttk.Checkbutton(der, text='FrAM  (uncheck = classic EVM)',
                        variable=self.var_metodo, command=self.conmutar) \
            .grid(row=fila, column=0, sticky='w'); fila += 1
        ttk.Checkbutton(der, text='Adaptive gain  (uncheck = fixed)',
                        variable=self.var_adapt, command=self.conmutar) \
            .grid(row=fila, column=0, sticky='w'); fila += 1
        ttk.Checkbutton(der, text='Show reliability mask  ρ',
                        variable=self.var_masc, command=self.conmutar) \
            .grid(row=fila, column=0, sticky='w'); fila += 1
        ttk.Checkbutton(der, text='Freeze capture',
                        variable=self.var_congelar, command=self.conmutar) \
            .grid(row=fila, column=0, sticky='w'); fila += 1
        self.var_color = tk.BooleanVar(value=est.color)
        ttk.Checkbutton(der, text='Color  (uncheck = grayscale)',
                        variable=self.var_color, command=self.conmutar) \
            .grid(row=fila, column=0, sticky='w'); fila += 1

        ttk.Separator(der, orient='horizontal') \
            .grid(row=fila, column=0, sticky='ew', pady=8); fila += 1
        ttk.Label(der, text='Presets',
                  font=('TkDefaultFont', 9, 'bold')) \
            .grid(row=fila, column=0, sticky='w'); fila += 1
        pf = ttk.Frame(der); pf.grid(row=fila, column=0, sticky='ew', pady=(2, 0))
        for i, nombre in enumerate(PRESETS):
            ttk.Button(pf, text=nombre, width=17,
                       command=lambda n=nombre: self.aplicar_preset(n)) \
                .grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky='ew')
        fila += 1

        ttk.Separator(der, orient='horizontal') \
            .grid(row=fila, column=0, sticky='ew', pady=8); fila += 1
        af = ttk.Frame(der); af.grid(row=fila, column=0, sticky='ew')
        ttk.Button(af, text='Save snapshot', command=self.guardar) \
            .grid(row=0, column=0, padx=2, sticky='ew')
        ttk.Button(af, text='Reset buffer', command=self.reiniciar) \
            .grid(row=0, column=1, padx=2, sticky='ew')
        af.columnconfigure(0, weight=1); af.columnconfigure(1, weight=1)

        self.n_guardadas = 0
        self.refrescar()

    def _fmt(self, clave, v):
        if clave in ('levels', 'T', 'percentil'):
            return f'{int(float(v))}'
        if clave == 'nu':
            return f'{float(v):.2f}'
        if clave in ('f_lo', 'f_hi'):
            return f'{float(v):.2f} Hz'
        return f'{float(v):.0f}'

    def cambiar(self, clave, valor):
        v = float(valor)
        with self.est.lock:
            self.est.params[clave] = v
        self.etiquetas_valor[clave].config(text=self._fmt(clave, v))

    def conmutar(self):
        with self.est.lock:
            self.est.metodo_fram = self.var_metodo.get()
            self.est.adaptativa = self.var_adapt.get()
            self.est.ver_mascara = self.var_masc.get()
            self.est.congelado = self.var_congelar.get()

            if self.var_color.get() != self.est.color:
                self.est.color = self.var_color.get()
                self.est.pendiente = None
                self.est.resultado = None

    def aplicar_preset(self, nombre):
        for clave, valor in PRESETS[nombre].items():
            self.escalas[clave].set(valor)

    def reiniciar(self):
        with self.est.lock:
            self.est.resultado = None
            self.est.pendiente = None

    def guardar(self):
        if self._ultima_vista is None:
            return
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        os.makedirs(d, exist_ok=True)
        ruta = os.path.join(d, f'live_tk_{self.n_guardadas:02d}.png')
        cv2.imwrite(ruta, cv2.cvtColor(self._ultima_vista, cv2.COLOR_RGB2BGR))
        print('saved', ruta)
        self.n_guardadas += 1

    def cerrar(self):
        with self.est.lock:
            self.est.parar = True
        self.root.after(120, self.root.destroy)

    _ultima_vista = None

    def refrescar(self):
        est = self.est
        with est.lock:
            orig = est.ultimo_original
            res = est.resultado
            masc = est.mascara
            tp = est.t_proceso
            color = est.color
            npr = est.n_proceso
            err = est.error
            fram = est.metodo_fram
            p = dict(est.params)

        if orig is not None:
            def a_rgb(x):
                if x.dtype != np.uint8:
                    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
                if x.ndim == 2:
                    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
                return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

            izq = a_rgb(orig)
            der = a_rgb(res) if res is not None else np.zeros_like(izq)

            if masc is not None and res is not None:
                m = cv2.resize(masc, (der.shape[1], der.shape[0]))
                calor = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                der = cv2.addWeighted(der, 0.45,
                                      cv2.cvtColor(calor, cv2.COLOR_BGR2RGB), 0.55, 0)

            etiq = 'FrAM' if fram else 'EVM'
            cv2.putText(izq, 'ORIGINAL', (8, izq.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(der, etiq, (8, der.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (120, 255, 160) if fram else (255, 170, 120), 2, cv2.LINE_AA)

            vista = np.hstack([izq, np.full((izq.shape[0], 3, 3), 70, np.uint8), der])

            objetivo = max(480, min(1180, self.root.winfo_width() - 380))
            f = objetivo / float(vista.shape[1])
            if abs(f - 1.0) > 0.02:
                vista = cv2.resize(vista, None, fx=f, fy=f,
                                   interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)

            self._ultima_vista = vista
            cab = f'P6 {vista.shape[1]} {vista.shape[0]} 255 '.encode()
            self._foto = tk.PhotoImage(data=cab + np.ascontiguousarray(vista).tobytes(),
                                       format='PPM')
            self.lienzo.config(image=self._foto)

        est_txt = (
            f"{'FrAM' if fram else 'EVM'} · α={p['alpha']:.0f} · ν={p['nu']:.2f} · "
            f"band {p['f_lo']:.2f}–{p['f_hi']:.2f} Hz · levels {int(p['levels'])} · "
            f"K={int(p['K'])}\n"
            f"processing {tp*1000:.1f} ms → {1/tp if tp else 0:.1f} fps · "
            f"{'color' if color else 'gray'} · {npr} frames processed"
        )
        if err:
            est_txt += f'\n{err[:90]}'
        self.estado_txt.config(text=est_txt)

        self.root.after(60, self.refrescar)

def main():
    ap = argparse.ArgumentParser(description='FrAM live demo (Tkinter)')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--video', default=None)
    ap.add_argument('--width', type=int, default=0,
                    help='processing width; 0 = native resolution (default)')
    ap.add_argument('--gray', '--gris', dest='gris', action='store_true',
                    help='start in grayscale')
    ap.add_argument('--fps', type=float, default=None)
    args = ap.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        fuente = os.path.basename(args.video)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        fuente = f'webcam {args.camera}'
    if not cap.isOpened():
        print(f'ERROR: could not open {fuente}')
        return 1

    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps < 240.0):
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_w = args.width or w
    print(f'Source: {fuente}  {w}x{h} @ {fps:.1f} fps')
    print(f'Processing width: {proc_w} px' + ('  (native)' if not args.width else ''))
    print('Incremental causal engine (fracmag_rt): constant cost per frame.')

    est = Estado()
    est.fps = fps
    est.color = not args.gris

    t_cap = threading.Thread(target=hilo_captura,
                             args=(est, cap, args.width, bool(args.video)), daemon=True)
    t_prc = threading.Thread(target=hilo_proceso, args=(est,), daemon=True)
    t_cap.start(); t_prc.start()

    root = tk.Tk()
    root.geometry('1500x820')
    App(root, est)
    root.mainloop()

    with est.lock:
        est.parar = True
    time.sleep(0.15)
    cap.release()
    return 0

if __name__ == '__main__':
    sys.exit(main())
