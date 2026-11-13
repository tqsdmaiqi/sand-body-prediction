from pathlib import Path
from dataset import read_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.measure import label
from skimage.morphology import skeletonize
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n')

def neighbor_count(mask: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    return convolve(mask.astype(np.uint8), k, mode='constant', cval=0)

def structure_metrics(mask: np.ndarray, valid: np.ndarray, *, params) -> dict:
    out = {'sand_area_km2': float(mask.sum()) * params['km2_per_cell']}
    lab = label(mask, connectivity=2)
    n_comp = int(lab.max())
    out['connected_components'] = n_comp
    if n_comp:
        sizes = np.bincount(lab.ravel())[1:]
        out['largest_component_fraction'] = float(sizes.max() / sizes.sum())
        out['components_ge_25km2'] = int((sizes * params['km2_per_cell'] >= 25.0).sum())
    else:
        out['largest_component_fraction'] = np.nan
        out['components_ge_25km2'] = 0
    skel = skeletonize(mask)
    nbr = neighbor_count(skel)
    out['skeleton_length_km'] = float(skel.sum() * params['cell'] / 1000.0)
    out['skeleton_endpoints'] = int(((nbr == 1) & skel).sum())
    out['skeleton_branchpoints'] = int(((nbr >= 3) & skel).sum())
    out['sand_fraction_of_prediction_area'] = float(mask.sum() / max(valid.sum(), 1))
    return out

def run(*, params) -> None:
    params['out_dir'].mkdir(parents=True, exist_ok=True)
    d = read_csv(params['pred_csv'], options=params['csv_options'])
    need = ['grid_row', 'grid_col', 'pred_sand_thickness_raw', 'pred_sand_thickness_channel_constrained', 'delta_channel_minus_raw', 'channel_zone', 'distance_to_channel', 'channel_weight_2000']
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f'05_Grid prediction Missing columns: {missing}\nRun predict_channel_brt.py first.')
    for c in need[1:]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    ri = d['grid_row'].to_numpy(int)
    ci = d['grid_col'].to_numpy(int)

    def to_grid(v: np.ndarray) -> np.ndarray:
        a = np.full((params['row_n'], params['col_n']), np.nan)
        a[ri, ci] = v
        return a
    g_raw = to_grid(d['pred_sand_thickness_raw'].to_numpy(float))
    g_con = to_grid(d['pred_sand_thickness_channel_constrained'].to_numpy(float))
    g_del = to_grid(d['delta_channel_minus_raw'].to_numpy(float))
    g_zone = to_grid(d['channel_zone'].to_numpy(float))
    valid = np.isfinite(g_raw)
    labels = read_csv(params['label_csv'], options=params['csv_options'])['sand_thickness']
    cuts = [float(labels.quantile(q)) for q in params['cutoff_quantiles']]
    src = 'observed well label quantiles'
    rows = []
    for q, cut in zip(params['cutoff_quantiles'], cuts):
        m_raw = valid & (g_raw >= cut)
        m_con = valid & (g_con >= cut)
        r_raw = structure_metrics(m_raw, valid, params=params)
        r_con = structure_metrics(m_con, valid, params=params)
        r = {'threshold_quantile': f'P{q * 100:.0f}', 'threshold_m': round(cut, 3)}
        for k in r_raw:
            r[f'raw_{k}'] = r_raw[k]
            r[f'constrained_{k}'] = r_con[k]
            r[f'delta_{k}'] = r_con[k] - r_raw[k]
        for zn in (0, 1, 2):
            zm = valid & (g_zone == zn)
            if zm.sum() == 0:
                continue
            r[f'zone{zn}sand_fraction_raw'] = float(m_raw[zm].mean())
            r[f'zone{zn}sand_fraction_constrained'] = float(m_con[zm].mean())
            r[f'zone{zn}sand_fraction_change'] = r[f'zone{zn}sand_fraction_constrained'] - r[f'zone{zn}sand_fraction_raw']
        rows.append(r)
    sdf = pd.DataFrame(rows)
    write_csv(sdf, params['out_dir'] / '04_structure_metrics.csv')
    zrows = []
    for zn in sorted(pd.unique(d['channel_zone'].dropna().astype(int))):
        sel = d[d['channel_zone'] == zn]
        raw = sel['pred_sand_thickness_raw']
        con = sel['pred_sand_thickness_channel_constrained']
        ms = valid & (g_zone == zn)
        m_raw = ms & (g_raw >= cuts[1])
        m_con = ms & (g_con >= cuts[1])
        zrows.append({'zone': f'zone{zn}', 'zone_name': params['zone_name'].get(zn, f'zone{zn}'), 'cell_count': int(len(sel)), 'area_km2': round(len(sel) * params['km2_per_cell'], 2), 'raw_mean_thickness_m': round(float(raw.mean()), 3), 'constrained_mean_thickness_m': round(float(con.mean()), 3), 'mean_adjustment_m': round(float(sel['delta_channel_minus_raw'].mean()), 4), 'adjusted_cell_count': int((sel['delta_channel_minus_raw'].abs() > 1e-09).sum()), 'rawsand_fraction_P50': round(float(m_raw.sum() / max(ms.sum(), 1)), 4), 'constrainedsand_fraction_P50': round(float(m_con.sum() / max(ms.sum(), 1)), 4)})
    zdf = pd.DataFrame(zrows)
    write_csv(zdf, params['out_dir'] / '05_structure_by_zone.csv')
    stride = 1
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), constrained_layout=True)
    cmap_m = plt.matplotlib.colors.ListedColormap(['#f2f2f2', '#2b2b2b'])
    cut = cuts[1]
    for ax, g, title in ((axes[0], g_raw, f'(a) raw BRT (>= {cut:.2f} m)'), (axes[1], g_con, f'(b) Graded channel constraintafter (>= {cut:.2f} m)')):
        ax.imshow(valid[::stride, ::stride], cmap=cmap_m, origin='lower', interpolation='nearest', vmin=0, vmax=1)
        ax.imshow((valid & (g >= cut))[::stride, ::stride], cmap=cmap_m, origin='lower', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(title)
    im = axes[2].imshow(np.where(valid, g_del, np.nan)[::stride, ::stride], cmap='turbo', origin='lower', interpolation='nearest')
    axes[2].set_title(f'(c) Channel adjustment (mean {float(np.nanmean(g_del)):+.3f} m, maximum {float(np.nanmax(g_del)):+.3f} m)')
    fig.colorbar(im, ax=axes[2], shrink=0.85, label='Thickness change (m)')
    for ax in axes:
        ax.set_xlabel('grid_col')
        ax.set_ylabel('grid_row')
    fig.suptitle('Graded channel constraint: raw and constrained masks and changes (500 m grid)', fontsize=14)
    for ext, kw in (('png', {'dpi': params['fig_dpi']}), ('pdf', {})):
        p = params['out_dir'] / f'06_structure_comparison.{ext}'
        fig.savefig(p, **kw)
    plt.close(fig)
