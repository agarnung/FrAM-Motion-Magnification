from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fracmag as fm

DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(DIR, 'assets')
os.makedirs(ASSETS, exist_ok=True)

SRC = os.environ.get('FRAM_VIDEO_SRC', os.path.join(ASSETS, 'source'))

def load_video(nombre: str, max_frames: int = 240, scale: float = 0.5):
    ruta = os.path.join(SRC, f'{nombre}.mp4')
    if not os.path.isfile(ruta):
        raise SystemExit(
            f'could not find {ruta}\n'
            'Fetch the clips with:  python3 fetch_videos.py\n'
            'or point FRAM_VIDEO_SRC at a directory that contains them.')
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0)
    cap.release()
    if not frames:
        raise RuntimeError(f'could not read {ruta}')
    return np.array(frames), float(fps)

def split_regions(frames: np.ndarray, percentil: float = 80.0):
    actividad = frames.std(axis=0)

    actividad = cv2.GaussianBlur(actividad, (0, 0), 3.0)
    alto = np.percentile(actividad, percentil)
    bajo = np.percentile(actividad, 30.0)
    return actividad >= alto, actividad <= bajo

def band_energy_ratio(frames, fps, mask, f_lo, f_hi):
    serie = np.array([f[mask].mean() for f in frames])
    serie = serie - serie.mean()
    esp = np.abs(np.fft.rfft(serie)) ** 2
    freqs = np.fft.rfftfreq(len(serie), 1.0 / fps)
    en = (freqs >= f_lo) & (freqs <= f_hi)
    fuera = (freqs > 0) & (~en)
    return float(esp[en].sum() / (esp[en].sum() + esp[fuera].sum() + 1e-20))

def dominant_frequency(frames, fps, mask, f_lo=0.7, f_hi=2.0):
    serie = np.array([f[mask].mean() for f in frames])
    serie = serie - serie.mean()
    esp = np.abs(np.fft.rfft(serie)) ** 2
    freqs = np.fft.rfftfreq(len(serie), 1.0 / fps)
    m = (freqs >= f_lo) & (freqs <= f_hi)
    return float(freqs[m][np.argmax(esp[m])] * 60.0)

def temporal_noise(frames, mask):
    d = np.diff(frames, axis=0)
    return float(np.std(d[:, mask]))

def spatial_selectivity(out, inp, m_act, m_est):
    g_act = temporal_noise(out, m_act) / (temporal_noise(inp, m_act) + 1e-12)
    g_est = temporal_noise(out, m_est) / (temporal_noise(inp, m_est) + 1e-12)
    return float(g_act / (g_est + 1e-12)), g_act, g_est

CONFIG = {

    'face':   ((0.7, 2.0), 20.0, 'facial pulse (physiological band)'),
    'face2':  ((0.7, 2.0), 20.0, 'facial pulse (physiological band)'),
    'baby':   ((0.5, 2.0), 15.0, 'infant breathing / pulse'),
    'subway': ((1.0, 4.0), 10.0, 'structural vibration'),
}

def run_video(nombre: str, levels: int = 4, max_frames: int = 240,
              scale: float = 0.5, nu: float = 0.3):
    (f_lo, f_hi), alpha, desc = CONFIG[nombre]
    frames, fps = load_video(nombre, max_frames=max_frames, scale=scale)
    m_act, m_est = split_regions(frames)

    evm = fm.evm_magnify(frames, fps, f_lo, f_hi, alpha=alpha, levels=levels)

    g_evm = temporal_noise(evm, m_act) / (temporal_noise(frames, m_act) + 1e-12)
    f0 = fm.fram_magnify(frames, fps, f_lo, f_hi, alpha=alpha, nu=nu, levels=levels)
    g_f0 = temporal_noise(f0, m_act) / (temporal_noise(frames, m_act) + 1e-12)

    factor = (g_evm - 1.0) / max(g_f0 - 1.0, 1e-6)
    alpha_fram = alpha * float(np.clip(factor, 0.1, 200.0))
    fram = fm.fram_magnify(frames, fps, f_lo, f_hi, alpha=alpha_fram, nu=nu, levels=levels)

    res = {'nombre': nombre, 'desc': desc, 'fps': fps, 'shape': frames.shape,
           'banda': (f_lo, f_hi), 'alpha_evm': alpha, 'alpha_fram': alpha_fram,
           'nu': nu, 'frames': frames,

           'vid_evm': evm, 'vid_fram': fram,
           'm_act': m_act, 'm_est': m_est}

    for etiqueta, vid in [('input', frames), ('evm', evm), ('fram', fram)]:
        sel, g_a, g_e = spatial_selectivity(vid, frames, m_act, m_est)
        res[etiqueta] = {
            'ber_act': band_energy_ratio(vid, fps, m_act, f_lo, f_hi),
            'ruido_est': temporal_noise(vid, m_est),
            'ruido_act': temporal_noise(vid, m_act),
            'selectividad': sel, 'realce_act': g_a, 'realce_est': g_e,
            'lpm': dominant_frequency(vid, fps, m_act) if nombre.startswith('face') else None,
        }
    return res

def activity_contrast(frames, m_act, m_est):
    return temporal_noise(frames, m_act) / (temporal_noise(frames, m_est) + 1e-12)

def figura_real(resultados):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(resultados)
    fig = plt.figure(figsize=(16, 3.6 * n + 1.2))

    for i, r in enumerate(resultados):
        fr, evm, fram = r['frames'], r['vid_evm'], r['vid_fram']
        m_act, m_est = r['m_act'], r['m_est']

        ax = fig.add_subplot(n, 4, 4 * i + 1)
        base = np.dstack([fr[0]] * 3)
        base[m_act] = base[m_act] * [1.0, 0.55, 0.55]
        base[m_est] = base[m_est] * [0.55, 0.75, 1.0]
        ax.imshow(np.clip(base, 0, 1))
        ax.set_title(f'{r["nombre"]} — masks\nred=active, blue=static',
                     fontsize=9.5)
        ax.axis('off')

        ax = fig.add_subplot(n, 4, 4 * i + 2)
        for vid, et, c in [(fr, 'input', 'gray'), (evm, 'EVM', 'crimson'),
                           (fram, 'FrAM', 'seagreen')]:
            serie = np.array([f[m_act].mean() for f in vid])
            serie = serie - serie.mean()
            esp = np.abs(np.fft.rfft(serie)) ** 2
            fq = np.fft.rfftfreq(len(serie), 1.0 / r['fps'])
            ax.semilogy(fq[1:], esp[1:] + 1e-18, color=c, lw=1.5, label=et)
        ax.axvspan(*r['banda'], color='gold', alpha=0.25, label='target band')
        ax.set_xlim(0, 6); ax.set_xlabel('Hz'); ax.set_ylabel('power')
        ax.set_title('Spectrum in the active region', fontsize=9.5)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = fig.add_subplot(n, 4, 4 * i + 3)
        et = ['input', 'EVM', 'FrAM']
        act = [r['input']['ruido_act'], r['evm']['ruido_act'], r['fram']['ruido_act']]
        est = [r['input']['ruido_est'], r['evm']['ruido_est'], r['fram']['ruido_est']]
        x = np.arange(3); w = 0.36
        ax.bar(x - w / 2, act, w, label='active region', color='steelblue')
        ax.bar(x + w / 2, est, w, label='static region', color='indianred')
        ax.set_xticks(x); ax.set_xticklabels(et, fontsize=8)
        ax.set_ylabel('temporal noise')
        ax.set_title('The background should NOT be amplified', fontsize=9.5)
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

        ax = fig.add_subplot(n, 4, 4 * i + 4)
        n_evm = np.std(np.diff(evm, axis=0), axis=0)
        n_frm = np.std(np.diff(fram, axis=0), axis=0)
        d = n_evm - n_frm
        v = np.percentile(np.abs(d), 99) + 1e-9
        h = ax.imshow(d, cmap='RdBu_r', vmin=-v, vmax=v)
        ax.set_title('EVM noise − FrAM noise\n(red: FrAM injects less)', fontsize=9.5)
        ax.axis('off'); plt.colorbar(h, ax=ax, fraction=0.046)

    fig.suptitle('Real-video validation: EVM vs. FrAM at matched enhancement '
                 'in the active region', fontsize=13)
    fig.tight_layout()
    p = os.path.join(ASSETS, 'fig4_video_real.png')
    fig.savefig(p, dpi=125, bbox_inches='tight'); plt.close(fig)
    return p

def main():
    videos = ['face', 'face2', 'baby', 'subway']
    print('=' * 78)
    print('FrAM validation on real video')
    print('=' * 78)

    resultados = []
    for nombre in videos:
        escala = 0.5 if nombre == 'face' else 0.35
        print(f'\nProcessing {nombre} ...', flush=True)
        r = run_video(nombre, max_frames=180, scale=escala)
        r['contraste'] = activity_contrast(r['frames'], r['m_act'], r['m_est'])
        resultados.append(r)

    print('\n--- Table 4: real video, at matched enhancement in the active region ---')
    cab = (f'{"video":8s}{"method":9s}{"BER":>7}{"n.bkgnd":>10}'
           f'{"n.active":>10}{"selectiv":>10}{"bpm":>6}')
    print(cab); print('-' * len(cab))
    for r in resultados:
        for k in ['input', 'evm', 'fram']:
            d = r[k]
            lpm = f'{d["lpm"]:.0f}' if d['lpm'] else '-'
            print(f'{r["nombre"] if k == "input" else "":8s}{k:9s}'
                  f'{d["ber_act"]:>7.3f}{d["ruido_est"]:>10.5f}'
                  f'{d["ruido_act"]:>10.5f}{d["selectividad"]:>10.2f}{lpm:>6}')

    print('\n--- Table 5: activity contrast vs. the observed advantage ---')
    cab = (f'{"video":8s}{"contrast":>10}{"EVM n.bkgnd":>13}'
           f'{"FrAM n.bkgnd":>14}{"gain":>10}')
    print(cab); print('-' * len(cab))
    for r in sorted(resultados, key=lambda x: -x['contraste']):
        g = r['evm']['ruido_est'] / (r['fram']['ruido_est'] + 1e-12)
        print(f'{r["nombre"]:8s}{r["contraste"]:>10.2f}'
              f'{r["evm"]["ruido_est"]:>13.5f}{r["fram"]["ruido_est"]:>14.5f}'
              f'{g:>9.2f}x')

    p = figura_real(resultados)
    print(f'\nFigure: {p}')

    with open(os.path.join(DIR, 'results_real.md'), 'w') as fh:
        fh.write('# Real-video validation (generated by real_video.py)\n\n')
        fh.write('## Table 4 — At matched enhancement in the active region\n\n')
        fh.write('| Video | Method | Band BER | Background noise | Active noise | '
                 'Selectivity | bpm |\n|---|---|---|---|---|---|---|\n')
        for r in resultados:
            for k in ['input', 'evm', 'fram']:
                d = r[k]
                lpm = f'{d["lpm"]:.0f}' if d['lpm'] else '—'
                fh.write(f'| {r["nombre"]} | {k} | {d["ber_act"]:.3f} | '
                         f'{d["ruido_est"]:.5f} | {d["ruido_act"]:.5f} | '
                         f'{d["selectividad"]:.2f} | {lpm} |\n')
        fh.write('\n## Table 5 — Activity contrast vs. the observed advantage\n\n')
        fh.write('| Video | Activity contrast | EVM background noise | '
                 'FrAM background noise | Gain |\n|---|---|---|---|---|\n')
        for r in sorted(resultados, key=lambda x: -x['contraste']):
            g = r['evm']['ruido_est'] / (r['fram']['ruido_est'] + 1e-12)
            fh.write(f'| {r["nombre"]} | {r["contraste"]:.2f} | '
                     f'{r["evm"]["ruido_est"]:.5f} | {r["fram"]["ruido_est"]:.5f} | '
                     f'{g:.2f}x |\n')
    print('results_real.md')
    print('\nDone.')

if __name__ == '__main__':
    main()
