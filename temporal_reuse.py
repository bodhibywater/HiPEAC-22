#!/usr/bin/env python3
"""Analyse exact spike-window reuse in HiPEAC-22 SNNMNST1/2 traces.
No third-party dependencies; results are theoretical explicit vector operations,
not measured cache traffic or simulated cycles.
"""
import argparse
import csv
import hashlib
import io
import math
import struct
from pathlib import Path

HEADER = struct.Struct('<8sIIIIIff')

def analyse(path, vlen=256):
    raw = Path(path).read_bytes()
    magic, n, t, ni, nh, no, beta, threshold = HEADER.unpack_from(raw)
    if magic not in (b'SNNMNST1', b'SNNMNST2') or min(n,t,ni,nh,no) <= 0:
        raise ValueError('Unsupported trace format or dimensions')
    offset = HEADER.size + 4*(ni*nh + nh + nh*no + no)
    sample_size = 2 + t*(ni+nh+no)
    if len(raw) != offset + n*sample_size:
        raise ValueError('Trace length does not match header')
    rows = []
    layers = [('L1', ni, nh, 2), ('L2', nh, no, 2+t*ni)]
    for name, inputs, outputs, start in layers:
        # Each binary spike occupies a byte. OR merges active input sets;
        # bit_count counts active inputs because each byte is exactly 0 or 1.
        images = []
        for i in range(n):
            p = offset + i*sample_size + start
            spikes = raw[p:p+t*inputs]
            if spikes.translate(None, b'\x00\x01'):
                raise ValueError('Nonbinary spike data')
            images.append([int.from_bytes(spikes[k*inputs:(k+1)*inputs], 'little') for k in range(t)])
        s = sum(x.bit_count() for img in images for x in img)
        blocks = math.ceil(outputs/(vlen//32))  # LMUL=1, FP32
        for b in (1,2,4,8,16):
            if b > t:
                continue
            u = 0
            for img in images:
                for k in range(0,t,b):
                    union = 0
                    for x in img[k:k+b]:
                        union |= x
                    u += union.bit_count()
            # Membrane load + bias load + membrane store per output block.
            before = 3*n*t*blocks
            after = 3*n*math.ceil(t/b)*blocks
            rows.append(dict(layer=name,B=b,samples=n,timesteps=t,inputs=inputs,outputs=outputs,
                             density_pct=round(100*s/(n*t*inputs),4),S=s,U_B=u,
                             weight_load_reduction_pct=round(100*(1-u/s),4) if s else '',
                             weight_vector_loads_before=s*blocks,weight_vector_loads_after=u*blocks,
                             state_bias_vector_ops_before=before,state_bias_vector_ops_after=after,
                             state_bias_reduction_pct=round(100*(1-after/before),4)))
    return rows, hashlib.sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace'); p.add_argument('--vlen',type=int,default=256)
    p.add_argument('--csv',default='temporal_reuse.csv')
    a=p.parse_args()
    if a.vlen < 32 or a.vlen % 32: p.error('VLEN must be a positive multiple of 32')
    rows, sha = analyse(a.trace,a.vlen)
    dest=Path(a.csv); dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Trace SHA256:',sha)
    print('FP32, LMUL=1, VLEN=',a.vlen,sep='')
    print(f"{'Layer':5} {'B':>3} {'Density%':>9} {'S':>10} {'U_B':>10} {'Weight saved%':>14} {'State/bias saved%':>18}")
    for r in rows:
        print(f"{r['layer']:5} {r['B']:3} {r['density_pct']:9.2f} {r['S']:10} {r['U_B']:10} {str(r['weight_load_reduction_pct']):>14} {r['state_bias_reduction_pct']:18.2f}")
    print('CSV:',dest)
    print('These are ideal explicit operation counts, not cache misses or speedups.')
if __name__=='__main__': main()
