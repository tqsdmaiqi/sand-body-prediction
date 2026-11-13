from pathlib import Path
from dataset import read_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def _find_pred_csv(*, params) -> Path:
    return params['channel_output'] / params['pred_basename']

def wells_dedup(path: Path, *, params) -> pd.DataFrame:
    d = read_csv(path, options=params['csv_options'])
    return d.sort_values(['source_well']).groupby('source_well', as_index=False).first()[['source_well', 'X_center', 'Y_center']]

def add_scalebar(ax, length_m: float=20000.0, label: str='20 km') -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = x0 + 0.06 * (x1 - x0)
    ys = y0 + 0.06 * (y1 - y0)
    ax.plot([xs, xs + length_m], [ys, ys], color='k', lw=2.2, solid_capstyle='butt')
    ax.text(xs + length_m / 2, ys + 0.012 * (y1 - y0), label, ha='center', va='bottom', fontsize=9)

def run(*, params) -> None:
    params['out_dir'].mkdir(parents=True, exist_ok=True)
    pred_path = _find_pred_csv(params=params)
    d = read_csv(pred_path, options=params['csv_options'])
    fin_col = params['col_fin']
    delta_col = params['col_delta']
    for c in (params['col_raw'], fin_col, delta_col, 'channel_zone', 'X_center', 'Y_center'):
        if c not in d.columns:
            raise ValueError(f"{params['pred_basename']} Missing columns {c}")
    x = pd.to_numeric(d['X_center'], errors='coerce').to_numpy(float)
    y = pd.to_numeric(d['Y_center'], errors='coerce').to_numpy(float)
    raw = pd.to_numeric(d[params['col_raw']], errors='coerce').to_numpy(float)
    fin = pd.to_numeric(d[fin_col], errors='coerce').to_numpy(float)
    delta = pd.to_numeric(d[delta_col], errors='coerce').to_numpy(float)
    zone = pd.to_numeric(d['channel_zone'], errors='coerce').fillna(-1).to_numpy(int)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(raw) & np.isfinite(fin)
    xmin, xmax, ymin, ymax = (x[ok].min(), x[ok].max(), y[ok].min(), y[ok].max())
    tr_w = wells_dedup(params['train_csv'], params=params)
    va_w = wells_dedup(params['valid_csv'], params=params)
    va_set = set(va_w['source_well'].astype(str))
    tr_w = tr_w[~tr_w['source_well'].astype(str).isin(va_set)]
    for nm, w in (('training wells', tr_w), ('validation wells', va_w)):
        wx = pd.to_numeric(w['X_center'], errors='coerce').to_numpy(float)
        wy = pd.to_numeric(w['Y_center'], errors='coerce').to_numpy(float)
        inside = ((wx >= xmin) & (wx <= xmax) & (wy >= ymin) & (wy <= ymax)).all()
        assert inside, f'{nm} coordinates outside map extent'
    assert xmax - xmin > 100000.0 and ymax - ymin > 100000.0, 'Invalid figure extent'
    span_x = xmax - xmin
    nz = int((np.abs(delta) > 1e-09).sum())
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.6), constrained_layout=True)
    vmax = float(np.nanpercentile(fin, 99))
    cmap = plt.get_cmap('YlGnBu')
    sc_kw = dict(s=9, marker='s', linewidths=0, rasterized=True)
    for ax, val, ttl in ((axes[0, 0], raw, '(a) Raw prediction'), (axes[0, 1], fin, '(b) Graded channel constraint + Channel continuity adjustmentafter')):
        m = ax.scatter(x[ok], y[ok], c=val[ok], cmap=cmap, vmin=0.0, vmax=vmax, **sc_kw)
        fig.colorbar(m, ax=ax, shrink=0.82, label='sand_thickness_cn (m)')
        ax.set_title(ttl, fontsize=12)
        add_scalebar(ax)
    m = axes[1, 0].scatter(x[ok], y[ok], c=delta[ok], cmap='Reds', vmin=0.0, vmax=float(max(np.nanpercentile(delta, 99), 1e-06)), **sc_kw)
    fig.colorbar(m, ax=axes[1, 0], shrink=0.82, label='Thickness change = final - raw (m)')
    axes[1, 0].set_title(f'(c) Change from raw prediction ({nz} nonzero cells, uplift only)', fontsize=12)
    add_scalebar(axes[1, 0])
    ax = axes[1, 1]
    zone_cmap = ListedColormap(['#f0f0f0', '#d9d9d9', '#a6cee3', '#1f78b4'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], zone_cmap.N)
    m = ax.scatter(x[ok], y[ok], c=zone[ok], cmap=zone_cmap, norm=norm, **sc_kw)
    cb = fig.colorbar(m, ax=ax, shrink=0.82, ticks=[-1, 0, 1, 2])
    cb.ax.set_yticklabels(['no data', 'zone0 Outside', 'zone1 Influence zone', 'zone2 Core zone'])
    ax.scatter(pd.to_numeric(tr_w['X_center'], errors='coerce'), pd.to_numeric(tr_w['Y_center'], errors='coerce'), s=42, marker='o', facecolors='none', edgecolors='k', linewidths=1.0, label=f'training wells {len(tr_w)} wells')
    ax.scatter(pd.to_numeric(va_w['X_center'], errors='coerce'), pd.to_numeric(va_w['Y_center'], errors='coerce'), s=95, marker='^', facecolors='none', edgecolors='#d62728', linewidths=1.6, label=f'validation wells {len(va_w)} wells')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.set_title('(d) Channel zones and wells', fontsize=12)
    add_scalebar(ax)
    for ax in axes.ravel():
        ax.set_aspect('equal')
        ax.set_xlim(xmin - 3000.0, xmax + 3000.0)
        ax.set_ylim(ymin - 3000.0, ymax + 3000.0)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(alpha=0.15, lw=0.4)
        ax.ticklabel_format(style='plain', useOffset=False)
        for lb in ax.get_xticklabels():
            lb.set_rotation(30)
            lb.set_ha('right')
    fig.suptitle(f'Full-grid thickness prediction (36 wells, 430 rows; five seismic attributes and channel distance)\n38611 cells; prediction area about {span_x / 1000:.0f} km wide; C2 constraint plus continuity adjustment (low values uplifted; zone0 unchanged)', fontsize=13)
    for ext, kw in (('png', {'dpi': 300}), ('pdf', {})):
        p = params['out_dir'] / f'18_grid_predictions.{ext}'
        fig.savefig(p, **kw)
    plt.close(fig)
