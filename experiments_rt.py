from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as bm
import experiments as ex
import fracmag as fm
import fracmag_rt as rt

DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(DIR, 'assets')
os.makedirs(ASSETS, exist_ok=True)

FPS, BANDA, F_SIG = 30.0, (1.5, 2.5), 2.0
TRANSITORIO = 40

r_plano = lambda o: float(np.std(np.diff(o[:, :, 70:], axis=0)))
r_text = lambda o: float(np.std(np.diff(o[:, :, :58], axis=0)))

def tabla_velocidad():
    filas = []
    rng = np.random.default_rng(0)
    for (H, W) in [(240, 320), (480, 640), (720, 1280), (1080, 1920)]:
        img = rng.random((H, W)).astype(np.float32)

        st = rt.FrAMStream(fps=FPS, f_lo=BANDA[0], f_hi=BANDA[1],
                           levels=3, nu=0.3, K_max=24)
        for _ in range(6):
            st.push(img)
        t = time.perf_counter(); N = 20
        for _ in range(N):
            st.push(img)
        t_rt = (time.perf_counter() - t) / N

        T = 32
        vol = rng.random((T, H, W))
        t = time.perf_counter()
        fm.fram_magnify(vol, FPS, BANDA[0], BANDA[1], alpha=20, nu=0.3, levels=3)
        t_off = time.perf_counter() - t

        filas.append((f'{W}x{H}', t_off, t_rt, t_off / t_rt, 1.0 / t_rt))
    return filas

def tabla_color():
    filas = []
    rng = np.random.default_rng(0)
    for (H, W) in [(480, 640), (720, 1280), (1080, 1920)]:
        bgr = (rng.random((H, W, 3)) * 255).astype(np.uint8)
        st = rt.FrAMStreamColor(fps=FPS, f_lo=BANDA[0], f_hi=BANDA[1],
                                levels=3, nu=0.3, K_max=24)
        for _ in range(6):
            st.push(bgr)
        t = time.perf_counter(); N = 20
        for _ in range(N):
            st.push(bgr)
        d = (time.perf_counter() - t) / N
        filas.append((f'{W}x{H}', d, 1.0 / d))
    return filas

def _run_stream(fr, alpha, nu, evm=False, K=31):
    st = rt.FrAMStream(fps=FPS, f_lo=BANDA[0], f_hi=BANDA[1], alpha=alpha,
                       nu=nu, levels=4, K_max=K, evm=evm)
    return np.array([st.push(f.astype(np.float32)) for f in fr])[TRANSITORIO:]

def tabla_equivalencia():
    fr, d = ex.sequence_mixed(T=140)
    dt = d[TRANSITORIO:]
    amp = lambda o: bm.amplification_factor(bm.estimate_displacement(o), dt)

    filas = [('Input', '--', bm.amplification_factor(
        bm.estimate_displacement(fr), d), r_plano(fr), r_text(fr))]

    o_evm = _run_stream(fr, 20, 0.3, evm=True)
    a_ref = amp(o_evm)
    filas.append(('EVM (streaming)', '20', a_ref, r_plano(o_evm), r_text(o_evm)))

    for nu in [0.0, 0.15, 0.3, 0.5]:
        o1 = _run_stream(fr, 20, nu)
        al = 20 * a_ref / max(amp(o1), 1e-6)
        o = _run_stream(fr, al, nu)
        filas.append((f'FrAM RT (nu={nu})', f'{al:.0f}', amp(o),
                      r_plano(o), r_text(o)))
    return filas

def tabla_causal_vs_offline():
    fr, d = ex.sequence_mixed(T=140)
    dt = d[TRANSITORIO:]
    off = fm.fram_magnify(fr, FPS, BANDA[0], BANDA[1], alpha=20, nu=0.3, levels=4)
    rtm = _run_stream(fr, 20, 0.3)
    return [
        ('offline (filtfilt, zero phase)',
         bm.amplification_factor(bm.estimate_displacement(off), d),
         r_plano(off), r_text(off), bm.band_snr(off, FPS, F_SIG)),
        ('streaming (causal)',
         bm.amplification_factor(bm.estimate_displacement(rtm), dt),
         r_plano(rtm), r_text(rtm), bm.band_snr(rtm, FPS, F_SIG)),
    ]

def tabla_memoria():
    fr, d = ex.sequence_mixed(T=140)
    dt = d[TRANSITORIO:]
    filas = []
    for K in [4, 8, 16, 24, 31]:
        o = _run_stream(fr, 20, 0.3, K=K)
        img = np.random.default_rng(0).random((480, 640)).astype(np.float32)
        st = rt.FrAMStream(fps=FPS, f_lo=BANDA[0], f_hi=BANDA[1],
                           levels=3, nu=0.3, K_max=K)
        for _ in range(5):
            st.push(img)
        t = time.perf_counter()
        for _ in range(15):
            st.push(img)
        ms = (time.perf_counter() - t) / 15 * 1000
        filas.append((K, bm.amplification_factor(bm.estimate_displacement(o), dt),
                      r_plano(o), ms))
    return filas

def figura_rt(t_vel, t_mem, t_eq):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    et = [f[0] for f in t_vel]
    off = [f[1] * 1000 for f in t_vel]
    rtv = [f[2] * 1000 for f in t_vel]
    x = np.arange(len(et)); w = 0.38
    ax[0].bar(x - w / 2, off, w, label='offline (reprocesses buffer)', color='indianred')
    ax[0].bar(x + w / 2, rtv, w, label='streaming (causal)', color='seagreen')
    ax[0].axhline(1000 / 30, color='k', ls='--', lw=1.2, label='33 ms = 30 fps')
    ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels(et, fontsize=8)
    ax[0].set_ylabel('ms per frame (log)')
    ax[0].set_title('Cost per frame\nthe causal reformulation crosses the 30 fps threshold',
                    fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis='y')
    for i, (o, r) in enumerate(zip(off, rtv)):
        ax[0].text(i, max(o, r) * 1.4, f'{o/r:.0f}x', ha='center', fontsize=8.5,
                   fontweight='bold')

    K = [f[0] for f in t_mem]
    a = [f[1] for f in t_mem]
    ms = [f[3] for f in t_mem]
    a2 = ax[1].twinx()
    ax[1].plot(K, a, 'o-', color='steelblue', lw=2, label='amplification')
    a2.plot(K, ms, 's--', color='darkorange', lw=2, label='ms/frame')
    ax[1].set_xlabel('truncated memory $K$'); ax[1].set_ylabel('amplification', color='steelblue')
    a2.set_ylabel('ms per frame (640x480)', color='darkorange')
    ax[1].set_title('Truncating the fractional tail\ncosts almost no amplification', fontsize=10)
    ax[1].grid(alpha=0.3)

    nom = [f[0] for f in t_eq[1:]]
    pl = [f[3] for f in t_eq[1:]]
    tx = [f[4] for f in t_eq[1:]]
    x = np.arange(len(nom))
    ax[2].bar(x - 0.2, pl, 0.4, label='flat region', color='steelblue')
    ax[2].bar(x + 0.2, tx, 0.4, label='textured region', color='indianred')
    ax[2].axhline(t_eq[0][3], color='k', ls=':', lw=1.4, label='input level')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([n.replace('FrAM RT ', '').replace(' (streaming)', '')
                           for n in nom], fontsize=8, rotation=15)
    ax[2].set_ylabel('temporal noise')
    ax[2].set_title('At matched amplification (~20x), in streaming\nthe advantage is preserved',
                    fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis='y')

    fig.suptitle('FrAM-RT: the same mathematics, recast in causal incremental form',
                 fontsize=12.5)
    fig.tight_layout()
    p = os.path.join(ASSETS, 'fig5_tiempo_real.png')
    fig.savefig(p, dpi=135, bbox_inches='tight'); plt.close(fig)
    return p

def main():
    print('=' * 76)
    print('FrAM-RT (real-time) experiments')
    print('=' * 76)

    print('\n--- Table 6: cost per frame, offline vs streaming ---')
    t_vel = tabla_velocidad()
    print(f'{"resolution":>12}{"offline":>12}{"streaming":>12}{"speedup":>10}{"fps RT":>9}')
    for r, o, s, sp, f in t_vel:
        print(f'{r:>12}{o*1000:>11.0f}m{s*1000:>11.1f}m{sp:>9.0f}x{f:>9.1f}')

    print('\n--- Table 7: color variant (YCrCb) ---')
    t_col = tabla_color()
    print(f'{"resolution":>12}{"ms/frame":>11}{"fps":>8}')
    for r, d, f in t_col:
        print(f'{r:>12}{d*1000:>11.1f}{f:>8.1f}')

    print('\n--- Table 8: streaming at matched amplification ---')
    t_eq = tabla_equivalencia()
    print(f'{"method":>20}{"alpha":>7}{"amplif":>8}{"n.flat":>10}{"n.text":>9}')
    for n, al, a, p, x in t_eq:
        print(f'{n:>20}{al:>7}{a:>8.2f}{p:>10.4f}{x:>9.4f}')

    print('\n--- Table 9: cost of causality ---')
    t_cv = tabla_causal_vs_offline()
    print(f'{"mode":>32}{"amplif":>8}{"n.flat":>10}{"n.text":>9}{"SNR":>8}')
    for n, a, p, x, s in t_cv:
        print(f'{n:>32}{a:>8.2f}{p:>10.4f}{x:>9.4f}{s:>8.2f}')

    print('\n--- Table 10: memory truncation K ---')
    t_mem = tabla_memoria()
    print(f'{"K":>5}{"amplif":>9}{"n.flat":>10}{"ms/frame":>11}')
    for k, a, p, ms in t_mem:
        print(f'{k:>5}{a:>9.2f}{p:>10.4f}{ms:>11.1f}')

    p = figura_rt(t_vel, t_mem, t_eq)
    print(f'\nFigure: {p}')

    with open(os.path.join(DIR, 'results_rt.md'), 'w') as fh:
        fh.write('# FrAM-RT results (generated by experiments_rt.py)\n\n')
        fh.write('## Table 6 — Cost per frame\n\n')
        fh.write('| Resolution | Offline | Streaming | Speedup | Streaming fps |\n')
        fh.write('|---|---|---|---|---|\n')
        for r, o, s, sp, f in t_vel:
            fh.write(f'| {r} | {o*1000:.0f} ms | {s*1000:.1f} ms | '
                     f'**{sp:.0f}×** | {f:.1f} |\n')
        fh.write('\n## Table 7 — Color variant (YCrCb)\n\n')
        fh.write('| Resolution | ms/frame | fps |\n|---|---|---|\n')
        for r, d, f in t_col:
            fh.write(f'| {r} | {d*1000:.1f} | {f:.1f} |\n')
        fh.write('\n## Table 8 — Streaming at matched amplification\n\n')
        fh.write('| Method | alpha | Amplification | Flat noise | Textured noise |\n')
        fh.write('|---|---|---|---|---|\n')
        for n, al, a, p_, x in t_eq:
            fh.write(f'| {n} | {al} | {a:.2f}× | {p_:.4f} | {x:.4f} |\n')
        fh.write('\n## Table 9 — Cost of causality\n\n')
        fh.write('| Mode | Amplification | Flat noise | Textured noise | SNR (dB) |\n')
        fh.write('|---|---|---|---|---|\n')
        for n, a, p_, x, s in t_cv:
            fh.write(f'| {n} | {a:.2f}× | {p_:.4f} | {x:.4f} | {s:+.2f} |\n')
        fh.write('\n## Table 10 — Memory truncation\n\n')
        fh.write('| K | Amplification | Flat noise | ms/frame (480×640) |\n')
        fh.write('|---|---|---|---|\n')
        for k, a, p_, ms in t_mem:
            fh.write(f'| {k} | {a:.2f}× | {p_:.4f} | {ms:.1f} |\n')
    print('results_rt.md')
    print('\nDone.')

if __name__ == '__main__':
    main()
