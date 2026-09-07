from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as bm
import fracmag as fm

FPS, F_SIG = 30.0, 2.0
BANDA = (1.5, 2.5)
DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(DIR, 'assets')
os.makedirs(ASSETS, exist_ok=True)

def sequence_mixed(T=96, H=128, W=128, amp_px=0.2, noise=0.04, seed=2):
    rng = np.random.default_rng(seed)
    _, x = np.mgrid[0:H, 0:W].astype(float)
    d = amp_px * np.sin(2 * np.pi * F_SIG * np.arange(T) / FPS)
    fr = np.zeros((T, H, W))
    for t in range(T):
        im = np.full((H, W), 0.5)
        im[:, :W // 2] = 0.5 + 0.35 * np.sin(2 * np.pi * (x[:, :W // 2] - d[t]) / 16.0)
        fr[t] = np.clip(im + rng.normal(0, noise, (H, W)), 0, 1)
    return fr, d

def ruido_plano(o):  return float(np.std(np.diff(o[:, :, 70:], axis=0)))
def ruido_textura(o): return float(np.std(np.diff(o[:, :, :58], axis=0)))

def tabla_principal():
    fr, d = sequence_mixed()
    amp = lambda o: bm.amplification_factor(bm.estimate_displacement(o), d)

    filas = []
    evm = fm.evm_magnify(fr, FPS, *BANDA, alpha=20, levels=4)
    A_ref = amp(evm)
    filas.append(('Input (unprocessed)', '--', amp(fr), ruido_plano(fr),
                  ruido_textura(fr), bm.band_snr(fr, FPS, F_SIG)))
    filas.append(('EVM (Butterworth)', '20', A_ref, ruido_plano(evm),
                  ruido_textura(evm), bm.band_snr(evm, FPS, F_SIG)))

    for nu in [0.0, 0.3, 0.5, 0.7]:

        o1 = fm.fram_magnify(fr, FPS, *BANDA, alpha=20, nu=nu, levels=4)
        al = 20 * A_ref / amp(o1)
        o = fm.fram_magnify(fr, FPS, *BANDA, alpha=al, nu=nu, levels=4)
        filas.append((f'FrAM (nu={nu})', f'{al:.0f}', amp(o), ruido_plano(o),
                      ruido_textura(o), bm.band_snr(o, FPS, F_SIG)))
    return filas

def tabla_ablacion():
    fr, d = sequence_mixed()
    amp = lambda o: bm.amplification_factor(bm.estimate_displacement(o), d)
    filas = []
    for nu in [0.0, 0.5, 0.7]:
        for ad in [False, True]:
            o = fm.fram_magnify(fr, FPS, *BANDA, alpha=20, nu=nu, levels=4, adaptive=ad)
            filas.append((f'nu={nu}', 'adaptive' if ad else 'constant',
                          amp(o), ruido_plano(o), ruido_textura(o)))
    return filas

def tabla_robustez():
    filas = []
    for ns in [0.01, 0.02, 0.04, 0.08]:
        fr, d = sequence_mixed(noise=ns)
        amp = lambda o: bm.amplification_factor(bm.estimate_displacement(o), d)
        evm = fm.evm_magnify(fr, FPS, *BANDA, alpha=20, levels=4)
        A = amp(evm)
        o1 = fm.fram_magnify(fr, FPS, *BANDA, alpha=20, nu=0.3, levels=4)
        o = fm.fram_magnify(fr, FPS, *BANDA, alpha=20 * A / amp(o1), nu=0.3, levels=4)
        filas.append((ns, A, ruido_plano(evm), amp(o), ruido_plano(o)))
    return filas

def figura_respuesta():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for nu in [0.0, 0.25, 0.5, 0.75, 1.0]:
        f, H = fm.fractional_response(nu, FPS)
        ax[0].plot(f, H, lw=2, label=f'$\\nu$={nu}')
    ax[0].set_xlabel('frequency (Hz)'); ax[0].set_ylabel('$|H_\\nu(\\omega)|$')
    ax[0].set_title('Grünwald-Letnikov operator response\n'
                    '$H_\\nu(\\omega)=(1-e^{-i\\omega})^\\nu$', fontsize=11)
    ax[0].legend(); ax[0].grid(alpha=0.3)

    for nu in [0.2, 0.5, 0.9]:
        w = fm.gl_coefficients(nu, 40)
        ax[1].plot(np.abs(w), 'o-', ms=3.5, lw=1.4, label=f'$\\nu$={nu}')
    ax[1].set_yscale('log'); ax[1].set_xlabel('memory lag $k$ (frames)')
    ax[1].set_ylabel('$|w_k|$')
    ax[1].set_title('Operator memory: coefficients decay\n'
                    'as $O(k^{-(1+\\nu)})$ — low $\\nu$ = long memory', fontsize=11)
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(ASSETS, 'fig1_operador.png')
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    return p

def figura_mascara():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fr, _ = sequence_mixed()
    banda = fm.build_laplacian_pyramid(fr.mean(axis=0), 4)[0]
    amp, fase, _ = fm.local_amplitude_phase(banda)
    rho = fm.phase_reliability(amp)

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for a, (im, t, cm) in zip(ax, [
            (fr[0], 'Input frame\n(left textured, right flat)', 'gray'),
            (amp, 'Local amplitude $A$\n(monogenic signal)', 'viridis'),
            (fase, 'Local phase $\\phi$\n(noise where $A\\approx 0$)', 'twilight'),
            (rho, 'Reliability $\\rho=A^2/(A^2+\\sigma^2)$\n(the proposed mask)', 'magma')]):
        h = a.imshow(im, cmap=cm); a.set_title(t, fontsize=10); a.axis('off')
        plt.colorbar(h, ax=a, fraction=0.046)
    fig.suptitle('Why adaptive gain is needed: phase is noise in flat regions',
                 fontsize=12.5)
    fig.tight_layout()
    p = os.path.join(ASSETS, 'fig2_mascara.png')
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    return p

def figura_resultados():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fr, d = sequence_mixed()
    amp = lambda o: bm.amplification_factor(bm.estimate_displacement(o), d)
    evm = fm.evm_magnify(fr, FPS, *BANDA, alpha=20, levels=4)
    A = amp(evm)
    o1 = fm.fram_magnify(fr, FPS, *BANDA, alpha=20, nu=0.3, levels=4)
    fram = fm.fram_magnify(fr, FPS, *BANDA, alpha=20 * A / amp(o1), nu=0.3, levels=4)

    fig = plt.figure(figsize=(16, 9))

    for i, (im, t) in enumerate([(fr, 'Input'), (evm, 'EVM'), (fram, 'FrAM ($\\nu$=0.3)')]):
        a = fig.add_subplot(2, 3, i + 1)
        a.imshow(im[:, 64, :], cmap='gray', aspect='auto')
        a.axvline(64, color='red', ls='--', lw=1.2)
        a.set_title(f'{t} — space-time slice\n(row 64; red = texture|flat boundary)',
                    fontsize=10)
        a.set_xlabel('x'); a.set_ylabel('time (frame)')

    a = fig.add_subplot(2, 3, 4)
    for im, t, c in [(fr, 'Input', 'gray'), (evm, 'EVM', 'crimson'),
                     (fram, 'FrAM', 'seagreen')]:
        a.plot(np.std(np.diff(im, axis=0), axis=(0, 1)), label=t, color=c, lw=1.6)
    a.axvline(64, color='k', ls='--', lw=1.0)
    a.set_xlabel('column $x$'); a.set_ylabel('temporal noise')
    a.set_title('Injected noise per column\nEVM contaminates the flat region; FrAM does not', fontsize=10)
    a.legend(); a.grid(alpha=0.3)

    a = fig.add_subplot(2, 3, 5)
    a.plot(d / np.abs(d).max(), 'k--', lw=2, label='ground truth (normalized)')
    for im, t, c in [(evm, 'EVM', 'crimson'), (fram, 'FrAM', 'seagreen')]:
        e = bm.estimate_displacement(im)
        a.plot(e / np.abs(e).max(), color=c, lw=1.5, label=t)
    a.set_xlabel('frame'); a.set_ylabel('norm. displacement')
    a.set_title('Phase fidelity: both track the true\noscillation without phase lag', fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = fig.add_subplot(2, 3, 6)
    nus, amps, rp = [], [], []
    for nu in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        o = fm.fram_magnify(fr, FPS, *BANDA, alpha=20, nu=nu, levels=4)
        nus.append(nu); amps.append(amp(o)); rp.append(ruido_plano(o))
    a2 = a.twinx()
    a.plot(nus, amps, 'o-', color='steelblue', lw=2, label='amplification')
    a2.plot(nus, rp, 's--', color='darkorange', lw=2, label='flat-region noise')
    a.axhline(A, color='crimson', ls=':', lw=1.5)
    a.text(0.42, A * 0.93, f'EVM ({A:.1f}x)', color='crimson', fontsize=9)
    a.set_xlabel('fractional order $\\nu$')
    a.set_ylabel('amplification', color='steelblue')
    a2.set_ylabel('flat-region noise', color='darkorange')
    a.set_title('Trade-off controlled by $\\nu$\n(fixed $\\alpha$ = 20)', fontsize=10)
    a.grid(alpha=0.3)

    fig.suptitle('FrAM vs. EVM at matched effective amplification', fontsize=13)
    fig.tight_layout()
    p = os.path.join(ASSETS, 'fig3_resultados.png')
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    return p

def main():
    print('=' * 70); print('FrAM experiments'); print('=' * 70)

    t1 = tabla_principal()
    print('\n--- Table 1: at matched effective amplification ---')
    print(f'{"method":<24}{"alpha":>7}{"amp":>7}{"n.flat":>10}{"n.text":>9}{"SNR":>8}')
    for f in t1:
        print(f'{f[0]:<24}{f[1]:>7}{f[2]:>7.2f}{f[3]:>10.4f}{f[4]:>9.4f}{f[5]:>8.2f}')

    t2 = tabla_ablacion()
    print('\n--- Table 2: gain ablation ---')
    print(f'{"order":<10}{"gain":<14}{"amp":>7}{"n.flat":>10}{"n.text":>9}')
    for f in t2:
        print(f'{f[0]:<10}{f[1]:<14}{f[2]:>7.2f}{f[3]:>10.4f}{f[4]:>9.4f}')

    t3 = tabla_robustez()
    print('\n--- Table 3: robustness to noise ---')
    print(f'{"sigma_in":>9}{"amp EVM":>9}{"n.fl EVM":>10}{"amp FrAM":>10}{"n.fl FrAM":>11}')
    for f in t3:
        print(f'{f[0]:>9.2f}{f[1]:>9.2f}{f[2]:>10.4f}{f[3]:>10.2f}{f[4]:>11.4f}')

    print('\n--- Figures ---')
    for p in [figura_respuesta(), figura_mascara(), figura_resultados()]:
        print('  ', p)

    with open(os.path.join(DIR, 'results.md'), 'w') as fh:
        fh.write('# FrAM results (generated by experiments.py)\n\n')
        fh.write('## Table 1 — At matched effective amplification\n\n')
        fh.write('| Method | alpha | Amplification | Flat-region noise | Textured-region noise | SNR (dB) |\n')
        fh.write('|---|---|---|---|---|---|\n')
        for f in t1:
            fh.write(f'| {f[0]} | {f[1]} | {f[2]:.2f}x | {f[3]:.4f} | {f[4]:.4f} | {f[5]:+.2f} |\n')
        fh.write('\n## Table 2 — Adaptive-gain ablation\n\n')
        fh.write('| Order | Gain | Amplification | Flat-region noise | Textured-region noise |\n')
        fh.write('|---|---|---|---|---|\n')
        for f in t2:
            fh.write(f'| {f[0]} | {f[1]} | {f[2]:.2f}x | {f[3]:.4f} | {f[4]:.4f} |\n')
        fh.write('\n## Table 3 — Robustness to input noise\n\n')
        fh.write('| input sigma | EVM amp. | EVM flat noise | FrAM amp. | FrAM flat noise |\n')
        fh.write('|---|---|---|---|---|\n')
        for f in t3:
            fh.write(f'| {f[0]:.2f} | {f[1]:.2f}x | {f[2]:.4f} | {f[3]:.2f}x | {f[4]:.4f} |\n')
    print('\n   results.md')
    print('\nDone.')

if __name__ == '__main__':
    main()
