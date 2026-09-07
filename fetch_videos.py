from __future__ import annotations

import os
import sys
import urllib.request

DIR = os.path.dirname(os.path.abspath(__file__))
DESTINO = os.path.join(DIR, 'assets', 'source')

BASES = ['https://people.csail.mit.edu/mrub/vidmag/video',
         'https://people.csail.mit.edu/mrub/evm/video']

VIDEOS = ['face', 'face2', 'baby', 'subway']

def descargar(nombre: str) -> bool:
    ruta = os.path.join(DESTINO, f'{nombre}.mp4')
    if os.path.isfile(ruta) and os.path.getsize(ruta) > 100_000:
        print(f'  {nombre:8s} already present ({os.path.getsize(ruta) / 1e6:.1f} MB)')
        return True

    for base in BASES:
        url = f'{base}/{nombre}.mp4'
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                datos = r.read()
        except Exception:
            continue

        if len(datos) < 100_000 or datos[4:8] != b'ftyp':
            continue
        with open(ruta, 'wb') as fh:
            fh.write(datos)
        print(f'  {nombre:8s} {len(datos) / 1e6:.1f} MB  <- {base}')
        return True

    print(f'  {nombre:8s} NOT AVAILABLE at any known URL')
    return False

def main():
    pedidos = sys.argv[1:] or VIDEOS
    os.makedirs(DESTINO, exist_ok=True)
    print(f'Downloading to {os.path.relpath(DESTINO, DIR)}')
    ok = sum(descargar(v) for v in pedidos)
    print(f'\n{ok}/{len(pedidos)} available.')
    return 0 if ok == len(pedidos) else 1

if __name__ == '__main__':
    sys.exit(main())
