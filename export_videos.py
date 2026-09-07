from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_video as rv

DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(DIR, 'assets')
os.makedirs(ASSETS, exist_ok=True)

REPETICIONES = 3

RECORTE = 12

def a_uint8(f: np.ndarray) -> np.ndarray:
    return (np.clip(f, 0.0, 1.0) * 255.0).round().astype(np.uint8)

def etiquetar(img: np.ndarray, texto: str) -> np.ndarray:
    h, w = img.shape[:2]
    escala = max(0.45, min(1.0, w / 520.0))
    grosor = max(1, int(round(escala * 2)))
    alto_banda = int(round(30 * escala)) + 8

    out = img.copy()
    banda = out[h - alto_banda:h].astype(np.float32) * 0.25
    out[h - alto_banda:h] = banda.astype(np.uint8)

    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    org = ((w - tw) // 2, h - (alto_banda - th) // 2 - 2)
    cv2.putText(out, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala,
                (255, 255, 255), grosor, cv2.LINE_AA)
    return out

def componer(paneles: list, separador: int = 4) -> np.ndarray:
    h = paneles[0].shape[0]
    sep = np.full((h, separador, 3), 40, np.uint8)
    tira = []
    for i, p in enumerate(paneles):
        if i:
            tira.append(sep)
        tira.append(p)
    return np.hstack(tira)

def escribir_mp4(ruta: str, fotogramas, fps: float) -> str:
    h, w = fotogramas[0].shape[:2]

    vw = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f'VideoWriter could not open {ruta}')
    for _ in range(REPETICIONES):
        for f in fotogramas:
            vw.write(f)
    vw.release()
    return ruta

def exportar(nombre: str, nu: float = 0.3) -> dict:
    escala = 0.5 if nombre == 'face' else 0.35
    r = rv.run_video(nombre, max_frames=180, scale=escala, nu=nu)

    fr, evm, fram = r['frames'], r['vid_evm'], r['vid_fram']
    fps = r['fps']
    ini, fin = RECORTE, len(fr) - RECORTE

    etiquetas = ['original',
                 f'EVM  alpha={r["alpha_evm"]:.0f}',
                 f'FrAM nu={nu:g}  alpha={r["alpha_fram"]:.0f}']

    fotogramas = []
    for t in range(ini, fin):
        paneles = [etiquetar(cv2.cvtColor(a_uint8(v[t]), cv2.COLOR_GRAY2BGR), et)
                   for v, et in zip([fr, evm, fram], etiquetas)]
        fotogramas.append(componer(paneles))

    ruta = escribir_mp4(os.path.join(ASSETS, f'video_{nombre}_comparativa.mp4'),
                        fotogramas, fps)

    ruta2 = escribir_mp4(
        os.path.join(ASSETS, f'video_{nombre}_original_vs_fram.mp4'),
        [componer([
            etiquetar(cv2.cvtColor(a_uint8(fr[t]), cv2.COLOR_GRAY2BGR), 'original (before)'),
            etiquetar(cv2.cvtColor(a_uint8(fram[t]), cv2.COLOR_GRAY2BGR),
                      f'FrAM nu={nu:g} (after)'),
        ]) for t in range(ini, fin)],
        fps)

    return {'nombre': nombre, 'fps': fps, 'n': fin - ini,
            'shape': fotogramas[0].shape, 'alpha_fram': r['alpha_fram'],
            'rutas': [ruta, ruta2]}

def main():
    pedidos = sys.argv[1:] or ['face', 'face2', 'baby', 'subway']
    desconocidos = [v for v in pedidos if v not in rv.CONFIG]
    if desconocidos:
        raise SystemExit(f'unknown videos: {desconocidos}; '
                         f'available: {list(rv.CONFIG)}')

    print('=' * 78)
    print('Export of comparison videos (offline validation)')
    print('=' * 78)

    for nombre in pedidos:
        print(f'\nProcessing {nombre} ...', flush=True)
        info = exportar(nombre)
        h, w = info['shape'][:2]
        print(f'  {info["n"]} frames x{REPETICIONES}  {w}x{h}  '
              f'{info["fps"]:.1f} fps  alpha_fram={info["alpha_fram"]:.0f}')
        for p in info['rutas']:
            print(f'  -> {os.path.relpath(p, DIR)}  '
                  f'({os.path.getsize(p) / 1e6:.1f} MB)')

    print('\nDone.')

if __name__ == '__main__':
    main()
