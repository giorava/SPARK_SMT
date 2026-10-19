import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
import time
import io
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except ImportError:
    _HAS_CANVAS = False

# --- 1. CONSTANTS ---

# Binding site strength defaults
STRENGTH_MULTIPLIERS_DEFAULT = [0.1, 0.3, 1.0, 3.0, 10.0]
STRENGTH_PRESETS = {
    "Uniform": [20, 20, 20, 20, 20],
    "Mostly Weak": [60, 25, 10, 4, 1],
    "Mostly Strong": [1, 4, 10, 25, 60],
    "Bimodal": [45, 5, 0, 5, 45],
    "All Baseline (1x)": [0, 0, 100, 0, 0],
}

BRIGHTNESS_VMAX = [1.0, 0.5, 0.25, 0.1, 0.05]
BRIGHTNESS_LABELS = ["1.0 (darkest)", "0.5", "0.25", "0.1", "0.05 (brightest)"]

# Sidebar parameter defaults
SIDEBAR_DEFAULTS = {
    'tf_concentration_nM': 100.0,
    'nucleus_diameter_nm': 5000.0,
    'num_binding_sites': 3000,
    'num_synthetic_sites': 0,
    'synthetic_strength': 1.0,
    'synthetic_cluster_radius_nm': 150.0,
    'bs_diff_mean': 0.01,
    'bs_diff_std': 0.005,
    'bs_jiggle_std': 0.001,
    'bs_hurst': 0.25,
    'num_frames': 60,
    'sim_dt': 0.01,
    'display_dt': 1.0,
    'cluster_size_0': 1,
    'cluster_size_1': 2,
    'cluster_size_2': 5,
    'cluster_size_3': 10,
    'cluster_size_4': 20,
    'cluster_pct_0': 20.0,
    'cluster_pct_1': 20.0,
    'cluster_pct_2': 20.0,
    'cluster_pct_3': 20.0,
    'cluster_pct_4': 20.0,
    'cluster_radius_nm': 150.0,
    'pct_free_init': 60.0,
    'pct_inter_init': 40.0,
    'pct_ns_init': 0.0,
    'd_free': 0.3,
    'd_free_std': 0.05,
    'd_inter': 0.1,
    'd_inter_std': 0.05,
    'd_ns': 0.01,
    'd_ns_std': 0.005,
    'k_fi': 0.1,
    'k_if': 0.1,
    'k_fns': 0.05,
    'k_ins': 0.1,
    'ns_res': 0.5,
    'k_on_prob': 1.0,
    'res_time_mean': 2.0,
    'res_time_std': 0.5,
    'capture_rad_nm': 50.0,
    'voxel_size_nm': 108.0,
    'lateral_fwhm_nm': 200.0,
    'axial_fwhm_nm': 600.0,
    'molecule_diameter_nm': 15.0,
    'image_dpi': 80,
    'upscale_factor': 12,
    'slice_spacing_nm': 2500.0,
    'num_slices': 1,
    'exposure_time_ms': 50.0,
    'strength_mult_0': 0.1,
    'strength_mult_1': 0.3,
    'strength_mult_2': 1.0,
    'strength_mult_3': 3.0,
    'strength_mult_4': 10.0,
    'strength_pct_0': 20.0,
    'strength_pct_1': 20.0,
    'strength_pct_2': 20.0,
    'strength_pct_3': 20.0,
    'strength_pct_4': 20.0,
    'cluster_correlated_strength': False,
    'enable_cooperativity': False,
    'coop_strength': 1.0,
    'coop_hill': 1.0,
    'two_state_mode': True,
    'au_calibration_enabled': False,
    'display_noise_enabled': False,
    'display_noise_floor_pct': 2.0,
    'display_noise_std_pct': 1.0,
    'au_ref_concentration_nM': 500.0,
    'au_ref_signal': 80.0,
    'au_background': 100.0,
    'au_noise_std': 5.0,
}

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_CSV_PATH = os.path.normpath(os.path.join(_APP_DIR, '..', 'examples', 'interactive_presets.csv'))

# --- 2. UTILITY FUNCTIONS ---

def safe_rerun():
    # Only catch AttributeError (function doesn't exist), not RerunException —
    # swallowing RerunException prevents the page from actually rerunning.
    for _fn_name in ("rerun", "experimental_rerun"):
        _fn = getattr(st, _fn_name, None)
        if _fn is not None:
            _fn()
            return

def _load_presets_df():
    if not os.path.exists(PRESETS_CSV_PATH):
        return pd.DataFrame(columns=['label'])
    try:
        df = pd.read_csv(PRESETS_CSV_PATH)
        if 'label' not in df.columns:
            return pd.DataFrame(columns=['label'])
        return df.dropna(subset=['label'])
    except Exception:
        return pd.DataFrame(columns=['label'])


def _apply_preset_row(row):
    def _get(col):
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        s = str(v).strip()
        return None if s.lower() in ('', 'nan', 'none') else s

    def _cast(key, raw):
        default = SIDEBAR_DEFAULTS.get(key)
        if default is None:
            return raw
        if isinstance(default, bool):
            return str(raw).strip().lower() in ('true', '1', 'yes')
        if isinstance(default, int):
            return int(float(raw))
        return float(raw)

    # nucleus diameter first (needed for conc/mol conversion)
    diam_raw = _get('nucleus_diameter_nm')
    diam = float(diam_raw) if diam_raw is not None else SIDEBAR_DEFAULTS['nucleus_diameter_nm']
    st.session_state['gp_nucleus_diam'] = diam

    conc_raw = _get('tf_concentration_nM')
    mol_raw = _get('num_molecules')
    if conc_raw is not None:
        conc = float(conc_raw)
    elif mol_raw is not None:
        conc = molecule_count_to_conc(int(float(mol_raw)), diam)
    else:
        conc = SIDEBAR_DEFAULTS['tf_concentration_nM']
    st.session_state['gp_conc'] = conc
    st.session_state['gp_num_mols'] = conc_to_molecule_count(conc, diam)

    _plain = [
        'molecule_diameter_nm', 'num_frames', 'sim_dt', 'display_dt',
        'pct_free_init', 'pct_inter_init', 'pct_ns_init',
        'd_free', 'd_free_std', 'd_inter', 'd_inter_std', 'd_ns', 'd_ns_std',
        'k_fi', 'k_if', 'k_fns', 'k_ins', 'ns_res',
        'k_on_prob', 'res_time_mean', 'res_time_std', 'capture_rad_nm',
        'coop_strength', 'coop_hill',
        'num_binding_sites', 'bs_diff_mean', 'bs_diff_std', 'bs_jiggle_std', 'bs_hurst',
        'cluster_radius_nm',
        'num_synthetic_sites', 'synthetic_strength', 'synthetic_cluster_radius_nm',
        'lateral_fwhm_nm', 'axial_fwhm_nm', 'voxel_size_nm', 'image_dpi', 'upscale_factor',
        'slice_spacing_nm', 'num_slices', 'exposure_time_ms',
        'display_noise_floor_pct', 'display_noise_std_pct',
        'au_ref_concentration_nM', 'au_ref_signal', 'au_background', 'au_noise_std',
    ]
    for key in _plain:
        raw = _get(key)
        st.session_state[key] = _cast(key, raw) if raw is not None else SIDEBAR_DEFAULTS[key]

    for key in ('enable_cooperativity', 'cluster_correlated_strength',
                'display_noise_enabled', 'au_calibration_enabled'):
        raw = _get(key)
        if raw is not None:
            st.session_state[key] = str(raw).strip().lower() in ('true', '1', 'yes')
        else:
            st.session_state[key] = SIDEBAR_DEFAULTS[key]

    tsm_raw = _get('two_state_mode')
    if tsm_raw is not None:
        two_state = str(tsm_raw).strip().lower() in ('true', '1', 'yes')
    else:
        two_state = SIDEBAR_DEFAULTS['two_state_mode']
    st.session_state['three_state_mode'] = not two_state

    for i in range(5):
        raw = _get(f'strength_mult_{i}')
        st.session_state[f'mult_{i}'] = float(raw) if raw is not None else SIDEBAR_DEFAULTS[f'strength_mult_{i}']
        raw = _get(f'strength_pct_{i}')
        st.session_state[f'str_pct_{i}'] = float(raw) if raw is not None else SIDEBAR_DEFAULTS[f'strength_pct_{i}']
        raw_s = _get(f'cluster_size_{i}')
        st.session_state[f'cluster_size_{i}'] = int(float(raw_s)) if raw_s is not None else SIDEBAR_DEFAULTS[f'cluster_size_{i}']
        raw_p = _get(f'cluster_pct_{i}')
        st.session_state[f'cluster_pct_{i}'] = float(raw_p) if raw_p is not None else SIDEBAR_DEFAULTS[f'cluster_pct_{i}']
    st.session_state['strength_preset'] = 'Custom'


def _capture_current_as_preset_row(name):
    ss = st.session_state

    def _ss(key, default):
        v = ss.get(key, default)
        return v if v is not None else default

    row = {'label': name}
    row['tf_concentration_nM'] = _ss('gp_conc', SIDEBAR_DEFAULTS['tf_concentration_nM'])
    row['nucleus_diameter_nm'] = _ss('gp_nucleus_diam', SIDEBAR_DEFAULTS['nucleus_diameter_nm'])
    row['num_molecules'] = _ss('gp_num_mols', 0)
    row['molecule_diameter_nm'] = _ss('molecule_diameter_nm', SIDEBAR_DEFAULTS['molecule_diameter_nm'])
    row['num_frames'] = _ss('num_frames', SIDEBAR_DEFAULTS['num_frames'])
    row['sim_dt'] = _ss('sim_dt', SIDEBAR_DEFAULTS['sim_dt'])
    row['display_dt'] = _ss('display_dt', SIDEBAR_DEFAULTS['display_dt'])
    row['two_state_mode'] = not _ss('three_state_mode', not SIDEBAR_DEFAULTS['two_state_mode'])
    row['pct_free_init'] = _ss('pct_free_init', SIDEBAR_DEFAULTS['pct_free_init'])
    row['pct_inter_init'] = _ss('pct_inter_init', SIDEBAR_DEFAULTS['pct_inter_init'])
    row['pct_ns_init'] = _ss('pct_ns_init', SIDEBAR_DEFAULTS['pct_ns_init'])
    row['d_free'] = _ss('d_free', SIDEBAR_DEFAULTS['d_free'])
    row['d_free_std'] = _ss('d_free_std', SIDEBAR_DEFAULTS['d_free_std'])
    row['d_inter'] = _ss('d_inter', SIDEBAR_DEFAULTS['d_inter'])
    row['d_inter_std'] = _ss('d_inter_std', SIDEBAR_DEFAULTS['d_inter_std'])
    row['d_ns'] = _ss('d_ns', SIDEBAR_DEFAULTS['d_ns'])
    row['d_ns_std'] = _ss('d_ns_std', SIDEBAR_DEFAULTS['d_ns_std'])
    row['k_fi'] = _ss('k_fi', SIDEBAR_DEFAULTS['k_fi'])
    row['k_if'] = _ss('k_if', SIDEBAR_DEFAULTS['k_if'])
    row['k_fns'] = _ss('k_fns', SIDEBAR_DEFAULTS['k_fns'])
    row['k_ins'] = _ss('k_ins', SIDEBAR_DEFAULTS['k_ins'])
    row['ns_res'] = _ss('ns_res', SIDEBAR_DEFAULTS['ns_res'])
    row['k_on_prob'] = _ss('k_on_prob', SIDEBAR_DEFAULTS['k_on_prob'])
    row['res_time_mean'] = _ss('res_time_mean', SIDEBAR_DEFAULTS['res_time_mean'])
    row['res_time_std'] = _ss('res_time_std', SIDEBAR_DEFAULTS['res_time_std'])
    row['capture_rad_nm'] = _ss('capture_rad_nm', SIDEBAR_DEFAULTS['capture_rad_nm'])
    row['enable_cooperativity'] = _ss('enable_cooperativity', SIDEBAR_DEFAULTS['enable_cooperativity'])
    row['coop_strength'] = _ss('coop_strength', SIDEBAR_DEFAULTS['coop_strength'])
    row['coop_hill'] = _ss('coop_hill', SIDEBAR_DEFAULTS['coop_hill'])
    row['num_binding_sites'] = _ss('num_binding_sites', SIDEBAR_DEFAULTS['num_binding_sites'])
    row['bs_diff_mean'] = _ss('bs_diff_mean', SIDEBAR_DEFAULTS['bs_diff_mean'])
    row['bs_diff_std'] = _ss('bs_diff_std', SIDEBAR_DEFAULTS['bs_diff_std'])
    row['bs_jiggle_std'] = _ss('bs_jiggle_std', SIDEBAR_DEFAULTS['bs_jiggle_std'])
    row['bs_hurst'] = _ss('bs_hurst', SIDEBAR_DEFAULTS['bs_hurst'])
    row['cluster_radius_nm'] = _ss('cluster_radius_nm', SIDEBAR_DEFAULTS['cluster_radius_nm'])
    row['cluster_correlated_strength'] = _ss('cluster_correlated_strength', SIDEBAR_DEFAULTS['cluster_correlated_strength'])
    for i in range(5):
        row[f'cluster_size_{i}'] = _ss(f'cluster_size_{i}', SIDEBAR_DEFAULTS[f'cluster_size_{i}'])
        row[f'cluster_pct_{i}'] = _ss(f'cluster_pct_{i}', SIDEBAR_DEFAULTS[f'cluster_pct_{i}'])
        row[f'strength_mult_{i}'] = _ss(f'mult_{i}', SIDEBAR_DEFAULTS[f'strength_mult_{i}'])
        row[f'strength_pct_{i}'] = _ss(f'str_pct_{i}', SIDEBAR_DEFAULTS[f'strength_pct_{i}'])
    row['num_synthetic_sites'] = _ss('num_synthetic_sites', SIDEBAR_DEFAULTS['num_synthetic_sites'])
    row['synthetic_strength'] = _ss('synthetic_strength', SIDEBAR_DEFAULTS['synthetic_strength'])
    row['synthetic_cluster_radius_nm'] = _ss('synthetic_cluster_radius_nm', SIDEBAR_DEFAULTS['synthetic_cluster_radius_nm'])
    row['lateral_fwhm_nm'] = _ss('lateral_fwhm_nm', SIDEBAR_DEFAULTS['lateral_fwhm_nm'])
    row['axial_fwhm_nm'] = _ss('axial_fwhm_nm', SIDEBAR_DEFAULTS['axial_fwhm_nm'])
    row['voxel_size_nm'] = _ss('voxel_size_nm', SIDEBAR_DEFAULTS['voxel_size_nm'])
    row['image_dpi'] = _ss('image_dpi', SIDEBAR_DEFAULTS['image_dpi'])
    row['upscale_factor'] = _ss('upscale_factor', SIDEBAR_DEFAULTS['upscale_factor'])
    row['slice_spacing_nm'] = _ss('slice_spacing_nm', SIDEBAR_DEFAULTS['slice_spacing_nm'])
    row['num_slices']       = _ss('num_slices',       SIDEBAR_DEFAULTS['num_slices'])
    row['exposure_time_ms'] = _ss('exposure_time_ms', SIDEBAR_DEFAULTS['exposure_time_ms'])
    row['display_noise_enabled'] = _ss('display_noise_enabled', SIDEBAR_DEFAULTS['display_noise_enabled'])
    row['display_noise_floor_pct'] = _ss('display_noise_floor_pct', SIDEBAR_DEFAULTS['display_noise_floor_pct'])
    row['display_noise_std_pct'] = _ss('display_noise_std_pct', SIDEBAR_DEFAULTS['display_noise_std_pct'])
    row['au_calibration_enabled'] = _ss('au_calibration_enabled', SIDEBAR_DEFAULTS['au_calibration_enabled'])
    row['au_ref_concentration_nM'] = _ss('au_ref_concentration_nM', SIDEBAR_DEFAULTS['au_ref_concentration_nM'])
    row['au_ref_signal'] = _ss('au_ref_signal', SIDEBAR_DEFAULTS['au_ref_signal'])
    row['au_background'] = _ss('au_background', SIDEBAR_DEFAULTS['au_background'])
    row['au_noise_std'] = _ss('au_noise_std', SIDEBAR_DEFAULTS['au_noise_std'])
    return row


def _count_mode_str(radio_label):
    """Map UI radio label to calculate_count_grid mode string."""
    return {
        "Full exposure":            "full",
        "Snapshot (last step)":     "snapshot",
        "Centroid (mean position)": "centroid",
    }.get(radio_label, "full")

def _resolve_batch_slice_params(row):
    """Return (slice_spacing_nm, num_slices_or_None) for a batch CSV row.

    slice_spacing_nm has priority. num_slices is used only when
    slice_spacing_nm is absent or NaN, in which case spacing is derived from
    num_slices and the nucleus radius. Falls back to SIDEBAR_DEFAULTS.
    """
    import math as _math
    def _valid(v):
        try:
            return not _math.isnan(float(v))
        except (TypeError, ValueError):
            return False

    _spc = row.get('slice_spacing_nm')
    _ns  = row.get('num_slices')
    if _valid(_spc):
        return float(_spc), None
    if _valid(_ns):
        _n = max(1, int(float(_ns)))
        _n_half = _n // 2
        _nuc_r  = float(row.get('nucleus_diameter_nm', 5000)) / 2
        _spacing = (_nuc_r / _n_half) if _n_half > 0 else _nuc_r
        return _spacing, _n
    return float(SIDEBAR_DEFAULTS['slice_spacing_nm']), None


_AVOGADRO = 6.022e23

def _nucleus_volume_um3(diameter_nm):
    """Spherical nucleus volume in µm³ from a diameter given in nm."""
    radius_um = (float(diameter_nm) / 1000.0) / 2.0
    return (4.0 / 3.0) * np.pi * (radius_um ** 3)

def conc_to_molecule_count(conc_nM, diameter_nm):
    """Convert TF concentration (nM) to an integer molecule count.

    Uses truncation (int()) rather than rounding, to exactly match the
    original `run_full_simulation` formula:
        int(((conc_nM * 1e-9) * (vol_um3 * 1e-15)) * 6.022e23)
    e.g. 0.1 nM in a 5000 nm nucleus -> 3.94 -> 3 molecules.
    """
    vol_um3 = _nucleus_volume_um3(diameter_nm)
    return int((float(conc_nM) * 1e-9) * (vol_um3 * 1e-15) * _AVOGADRO)

def molecule_count_to_conc(count, diameter_nm):
    """Convert an integer molecule count back to TF concentration (nM)."""
    vol_um3 = _nucleus_volume_um3(diameter_nm)
    denom = (vol_um3 * 1e-15) * _AVOGADRO * 1e-9
    return (float(count) / denom) if denom > 0 else 0.0


def upscale_image(img, scale_factor):
    """
    Upscale image by exact integer factor using nearest neighbor.
    Ensures each source pixel becomes exactly scale_factor x scale_factor pixels.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    width, height = img.size
    new_size = (width * scale_factor, height * scale_factor)
    upscaled = img.resize(new_size, Image.NEAREST)
    return upscaled

def draw_boundary_overlay(img_path_or_pil, nucleus_radius_nm, z_nm, crop_min, crop_max, 
                           line_color=(0, 255, 255), line_width=3):
    """
    Draw nuclear boundary circle overlay on an image for display only.
    Returns a new RGB PIL image; original file/image is not modified.
    """
    from PIL import ImageDraw
    
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil)
    else:
        img = img_path_or_pil.copy()
    
    img_rgb = img.convert('RGB')
    
    cross_section_r = np.sqrt(max(0, nucleus_radius_nm**2 - z_nm**2))
    if cross_section_r <= 0:
        return img_rgb
    
    draw = ImageDraw.Draw(img_rgb)
    w, h = img_rgb.size
    data_range = crop_max - crop_min
    
    cx = (0 - crop_min) / data_range * w
    cy = (crop_max - 0) / data_range * h
    
    r_px = cross_section_r / data_range * w
    
    bbox = [cx - r_px, cy - r_px, cx + r_px, cy + r_px]
    draw.ellipse(bbox, outline=line_color, width=line_width)
    
    return img_rgb

def calculate_synthetic_centroid_pixels(binding_sites_pos, is_synthetic, frames, 
                                        nucleus_diameter_nm, voxel_size_nm):
    """Calculate centroid of synthetic binding sites in pixel coordinates (native grid)."""
    if is_synthetic is None or not np.any(is_synthetic):
        return []
    
    NUCLEUS_RADIUS_NM = nucleus_diameter_nm / 2
    plot_limit = NUCLEUS_RADIUS_NM * 1.1
    GRID_SIZE = int(np.ceil(nucleus_diameter_nm / voxel_size_nm)) + 2
    if GRID_SIZE % 2 == 0: GRID_SIZE += 1
    half_span = (GRID_SIZE * voxel_size_nm) / 2.0
    
    voxel_centers_1d = np.arange(GRID_SIZE) * voxel_size_nm - half_span + voxel_size_nm/2
    in_view = (voxel_centers_1d >= -plot_limit) & (voxel_centers_1d <= plot_limit)
    view_indices = np.where(in_view)[0]
    if len(view_indices) > 0:
        i_start, i_end = view_indices[0], view_indices[-1] + 1
    else:
        i_start, i_end = 0, GRID_SIZE
    
    synthetic_sites_nm = binding_sites_pos[is_synthetic] * 1000
    centroid_nm = np.mean(synthetic_sites_nm, axis=0)
    
    grid_x = (centroid_nm[0] + half_span) / voxel_size_nm
    grid_y = (centroid_nm[1] + half_span) / voxel_size_nm
    grid_z = (centroid_nm[2] + half_span) / voxel_size_nm
    
    pixel_x = grid_x - i_start
    pixel_y = (i_end - 1) - grid_y
    pixel_z = grid_z
    
    results = []
    for frame_idx in frames:
        results.append({'Frame': frame_idx, 'Z': pixel_z, 'Y': pixel_y, 'X': pixel_x})
    return results


def _row_group_str(row, col):
    """Read an optional group label column ('sim_group' / 'site_group') from a
    pandas row, treating NaN / empty / 'nan' / 'None' as missing. Mirrors
    sim_repro._nonempty_str semantics."""
    if col not in row:
        return ''
    v = row[col]
    if v is None:
        return ''
    if isinstance(v, float) and np.isnan(v):
        return ''
    s = str(v).strip()
    if s == '' or s.lower() in ('nan', 'none'):
        return ''
    return s


def resolve_row_seed(row, row_idx, base_seed=None):
    """
    Resolve the per-row RNG seed for batch mode, mirroring the SLURM convention
    used in batch_hpc_fast.py / sim_repro.get_sim_seed():

      1. If the row has a usable integer in the 'sim_seed' column, use it.
      2. Otherwise, if a base_seed is given (UI), fall back to base_seed + row_idx.
      3. Otherwise return None (non-deterministic, OS entropy — current behavior).

    Accepts NaN, empty string, and 'nan'/'none' as "missing" so a partially-filled
    sim_seed column behaves cleanly in pandas-parsed CSVs.
    """
    if 'sim_seed' in row:
        v = row['sim_seed']
        if v is not None:
            if isinstance(v, float) and np.isnan(v):
                pass
            elif isinstance(v, str):
                s = v.strip()
                if s and s.lower() not in ('nan', 'none'):
                    try:
                        return int(float(s))
                    except (TypeError, ValueError):
                        pass
            else:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
    if base_seed is not None:
        return int(base_seed) + int(row_idx)
    return None


def _derive_motion_seed(sim_seed):
    """
    Deterministically derive a motion seed from a single root sim_seed.

    Uses numpy.random.SeedSequence with spawn_key=(1,) — the SAME convention
    sim_repro.make_rngs() uses for its mol_rng stream. This means:

    - In batch mode, a row in a `site_group` (but not a `sim_group`) gets
      a motion seed that is deterministic given its sim_seed but completely
      decoupled from the site-setup RNG state. Re-running the same CSV
      produces bit-identical motion, and changing site_physics columns
      (which would otherwise shift the global RNG state during site setup)
      does not affect the motion RNG.

    - In the HPC code, mol_rng = SeedSequence(seed, spawn_key=(1,)) is fed
      into a PCG64 Generator. Here we feed the same SeedSequence into the
      legacy MT19937 via np.random.seed(). The bytes differ between app2
      and HPC (different RNG algorithms), but the conceptual pairing —
      "this is the motion stream of seed N" — is preserved.

    Returns an int suitable for np.random.seed().
    """
    ss = np.random.SeedSequence(int(sim_seed), spawn_key=(1,))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def resolve_row_motion_seed(row, row_seed):
    """
    Decide the `motion_seed` argument for a batch row based on its
    sim_group / site_group columns and the resolved per-row sim_seed.
    Mirrors the HPC sim_repro tier semantics:

      sim_group set (full-lock tier):
          motion_seed = None  -> single-seed path; sites + motion both
          driven by row_seed via the global np.random state.

      site_group set but not sim_group (site-only tier):
          motion_seed = _derive_motion_seed(row_seed)  -> deterministic
          motion stream decoupled from site-setup state. Two rows with
          the same row_seed but different motion physics columns will
          produce different motion (because num_molecules etc. differ),
          but the *randomness* in motion will be reproducible across
          re-runs — and identical site_physics across rows still yields
          identical sites + identical cluster_D.

      neither set:
          motion_seed = None  -> existing behavior (single-seed lock if
          row_seed is set, OS entropy otherwise).

    A row_seed of None forces motion_seed = None regardless of grouping.
    """
    if row_seed is None:
        return None
    sim_group = _row_group_str(row, 'sim_group')
    site_group = _row_group_str(row, 'site_group')
    if sim_group:
        return None  # full lock: single-seed path
    if site_group:
        return _derive_motion_seed(row_seed)  # site lock with deterministic motion
    return None  # neither column set


def extract_batch_row_params(row):
    """Extract simulation and rendering parameters from a batch CSV row."""
    p = {}
    p['cluster_dist'] = {
        int(row.get(f'cluster_size_{i}', i + 1)): float(row.get(f'cluster_pct_{i}', 20.0))
        for i in range(5)
    }
    p['strength_multipliers'] = [
        row.get(f'strength_mult_{i}', STRENGTH_MULTIPLIERS_DEFAULT[i]) for i in range(5)
    ]
    p['strength_percentages'] = [row.get(f'strength_pct_{i}', 20.0) for i in range(5)]
    p['cluster_correlated'] = row.get('cluster_correlated_strength', False)
    p['cluster_radius_nm'] = float(row.get('cluster_radius_nm', 150.0))
    p['strength_range'] = (min(p['strength_multipliers']), max(p['strength_multipliers']))
    p['bs_hurst'] = row.get('bs_hurst', 0.25)
    p['exposure_time_s'] = row.get('exposure_time_ms', 50.0) / 1000.0
    p['num_synthetic'] = int(row.get('num_synthetic_sites', 0))
    p['synthetic_strength'] = row.get('synthetic_strength', 1.0)
    p['synthetic_radius'] = row.get('synthetic_cluster_radius_nm', 150.0)
    p['upscale_factor'] = int(row.get('upscale_factor', 10))
    p['lateral_fwhm'] = float(row.get('lateral_fwhm_nm', 200.0))
    p['axial_fwhm'] = float(row.get('axial_fwhm_nm', 600.0))
    p['enable_coop'] = bool(row.get('enable_cooperativity', False))
    p['coop_strength'] = float(row.get('coop_strength', 1.0))
    p['coop_hill'] = float(row.get('coop_hill', 1.0))
    p['two_state_mode'] = bool(row.get('two_state_mode', True))
    
    # AU calibration parameters
    p['au_calibration_enabled'] = bool(row.get('au_calibration_enabled', False))
    p['au_ref_concentration_nM'] = float(row.get('au_ref_concentration_nM', 500.0))
    p['au_ref_signal'] = float(row.get('au_ref_signal', 80.0))
    p['au_background'] = float(row.get('au_background', 100.0))
    p['au_noise_std'] = float(row.get('au_noise_std', 5.0))
    
    row_label = str(row.get('label', ''))
    p['row_label'] = "".join([c for c in row_label if c.isalnum() or c in (' ', '_', '-')]).strip()
    return p


# --- 3. PHYSICS ENGINE ---

def generate_clustered_sites(num_sites, nucleus_radius_um, cluster_probs, cluster_radius_nm=150.0, 
                             num_synthetic=0, synthetic_radius_nm=150.0):
    cluster_radius_um = cluster_radius_nm / 1000.0
    synthetic_radius_um = synthetic_radius_nm / 1000.0
    
    sites = []
    cluster_ids = []
    is_synthetic = []
    current_cluster_id = 0
    
    total_p = sum(cluster_probs.values())
    if total_p == 0 and num_sites > 0:
        normalized_probs = {1: 1.0}
    elif total_p > 0:
        normalized_probs = {k: v/total_p for k,v in cluster_probs.items()}
    else:
        normalized_probs = {}
    
    for size, prob in normalized_probs.items():
        target_sites_in_cat = int(num_sites * prob)
        n_clusters = int(target_sites_in_cat / size)
        
        for _ in range(n_clusters):
            center = np.random.normal(size=3)
            center /= np.linalg.norm(center)
            center *= (np.random.random() ** (1/3)) * nucleus_radius_um
            
            for _ in range(size):
                off = np.random.normal(size=3)
                off /= np.linalg.norm(off)
                off *= (np.random.random() ** (1/3)) * cluster_radius_um
                pos = center + off
                if np.linalg.norm(pos) > nucleus_radius_um:
                    pos = pos / np.linalg.norm(pos) * nucleus_radius_um
                sites.append(pos)
                cluster_ids.append(current_cluster_id)
                is_synthetic.append(False)
            current_cluster_id += 1
                
    while len(sites) < num_sites:
        pos = np.random.normal(size=3)
        pos /= np.linalg.norm(pos)
        pos *= (np.random.random() ** (1/3)) * nucleus_radius_um
        sites.append(pos)
        cluster_ids.append(current_cluster_id)
        is_synthetic.append(False)
        current_cluster_id += 1
    
    if num_synthetic > 0:
        synthetic_center = np.random.normal(size=3)
        synthetic_center /= np.linalg.norm(synthetic_center)
        synthetic_center *= (np.random.random() ** (1/3)) * nucleus_radius_um * 0.7
        synthetic_cluster_id = current_cluster_id
        
        for _ in range(num_synthetic):
            off = np.random.normal(size=3)
            off /= np.linalg.norm(off)
            off *= (np.random.random() ** (1/3)) * synthetic_radius_um
            pos = synthetic_center + off
            if np.linalg.norm(pos) > nucleus_radius_um:
                pos = pos / np.linalg.norm(pos) * nucleus_radius_um
            sites.append(pos)
            cluster_ids.append(synthetic_cluster_id)
            is_synthetic.append(True)
        
    return np.array(sites), np.array(cluster_ids), np.array(is_synthetic)


def assign_site_strengths(num_sites, cluster_ids, strength_multipliers, strength_percentages, 
                          is_synthetic, synthetic_strength=1.0, cluster_correlated=False):
    site_strengths = np.zeros(num_sites)
    site_strengths[is_synthetic] = synthetic_strength
    
    endogenous_mask = ~is_synthetic
    num_endogenous = np.sum(endogenous_mask)
    
    if num_endogenous > 0:
        total_pct = sum(strength_percentages)
        if total_pct == 0:
            probs = [1.0 / len(strength_multipliers)] * len(strength_multipliers)
        else:
            probs = [p / total_pct for p in strength_percentages]
        
        if cluster_correlated:
            endogenous_clusters = np.unique(cluster_ids[endogenous_mask])
            cluster_strengths_dict = {}
            for cid in endogenous_clusters:
                cluster_strengths_dict[cid] = np.random.choice(strength_multipliers, p=probs)
            for idx in np.where(endogenous_mask)[0]:
                site_strengths[idx] = cluster_strengths_dict[cluster_ids[idx]]
        else:
            endogenous_indices = np.where(endogenous_mask)[0]
            site_strengths[endogenous_indices] = np.random.choice(
                strength_multipliers, size=num_endogenous, p=probs
            )
    return site_strengths


def _ensure_compact(frame_data):
    """Convert old list-of-dicts exposure format to compact dict (idempotent)."""
    if isinstance(frame_data, dict):
        return frame_data
    from scipy import stats as _sc_stats
    pos_steps = np.array([s['pos'] for s in frame_data], dtype=np.float32)
    ss = np.array([s['states'] for s in frame_data])
    si = np.array([s['sites'] for s in frame_data], dtype=np.float32)
    oo = np.array([s['occupancy'] for s in frame_data])
    return {
        'pos_steps':     pos_steps,
        'modal_states':  _sc_stats.mode(ss, axis=0, keepdims=False)[0].astype(np.int8),
        'site_centroid': si.mean(axis=0),
        'any_occupied':  oo.any(axis=0),
    }


def compute_exposure_frames(exposure_data, site_strengths):
    """Compute exposure-averaged molecule positions and binding site data."""
    exposure_mol_data = {}
    exposure_site_data = {}

    for display_frame, exp in exposure_data.items():
        if exp is None:
            continue
        exp = _ensure_compact(exp)
        pos_steps = exp['pos_steps']          # (steps, N, 3)
        N = pos_steps.shape[1]
        centroid_pos = pos_steps.mean(axis=0) # (N, 3)
        mol_df = pd.DataFrame({
            'trajectory': range(N),
            'x': centroid_pos[:, 0], 'y': centroid_pos[:, 1], 'z': centroid_pos[:, 2],
            'state': exp['modal_states'], 'frame': display_frame
        })
        exposure_mol_data[display_frame] = mol_df
        exposure_site_data[display_frame] = (exp['site_centroid'], exp['any_occupied'])

    return exposure_mol_data, exposure_site_data


def calculate_exposure_grid_fast(exposure_data, display_frame, nucleus_diameter_nm, voxel_size_nm, lateral_fwhm_nm, axial_fwhm_nm):
    if display_frame not in exposure_data:
        return None, None, None
    exp = exposure_data[display_frame]
    if exp is None:
        return None, None, None
    exp = _ensure_compact(exp)
    all_positions = exp['pos_steps'].reshape(-1, 3).astype(float)
    combined_df = pd.DataFrame({
        'x': all_positions[:, 0], 'y': all_positions[:, 1], 'z': all_positions[:, 2]
    })
    grid, half_span, grid_size = calculate_grid_for_frame(
        combined_df, nucleus_diameter_nm, voxel_size_nm, lateral_fwhm_nm, axial_fwhm_nm
    )
    return grid, half_span, grid_size


def _make_frap_exposure(exposure_data_raw, frame, bleached_ids):
    """Return a one-frame exposure_data dict with bleached molecule indices removed."""
    exp = exposure_data_raw.get(frame)
    if exp is None:
        return {frame: None}
    exp = _ensure_compact(exp)
    N = exp['pos_steps'].shape[1]
    mask = np.ones(N, dtype=bool)
    mask[list(bleached_ids)] = False
    return {frame: {
        'pos_steps':     exp['pos_steps'][:, mask, :],
        'modal_states':  exp['modal_states'][mask],
        'site_centroid': exp['site_centroid'],
        'any_occupied':  exp['any_occupied'],
    }}


def calculate_count_grid(exposure_data, display_frame, nucleus_diameter_nm,
                         voxel_size_nm, slice_spacing_nm, mode='full'):
    """
    Compute a raw molecule count grid (no PSF convolution).

    Each molecule is assigned **directly** to the z-slab whose center is
    nearest its physical z position — no intermediate 108 nm z-voxelization.
    XY binning still uses ``voxel_size_nm`` (108 nm), matching the saved
    image pixel size and the convolved image's pixel grid.

    The third axis of the returned grid is therefore the **slab index**
    (length ``n_slabs``), not a fine z-voxel axis. This eliminates the
    boundary-band slab-assignment mismatch that arose when round-to-108-nm
    then round-to-300-nm-slab gave a different answer from direct
    round-to-300-nm-slab — see ``positions.txt`` for the ground truth.

    Parameters
    ----------
    slice_spacing_nm : float
        Spacing between z-slabs in nm. Each molecule is binned to the slab
        whose center is closest to its z (i.e. ``round(z / slice_spacing_nm)``).
    mode : {'full', 'snapshot', 'centroid'}
        'full'     - concatenate every sub-step (a molecule present for N steps
                     is counted N times). Matches the convolved image's
                     time-integrated geometry.
        'snapshot' - use only the last sub-step. One count per molecule at its
                     end-of-exposure position.
        'centroid' - use the mean position across all sub-steps. One count per
                     molecule at its time-averaged position. Matches
                     exposure_mol_data and the positions.txt file.

    Returns
    -------
    count_grid   : ndarray (GRID_SIZE, GRID_SIZE, n_slabs), int32
        Third axis is the slab index, not a 108 nm z-voxel index.
    half_span    : float
    grid_size    : int
        Size of the lateral (XY) grid before in-view cropping.
    slab_centers : ndarray (n_slabs,), float
        Z positions (nm) of each slab center, in ascending order.
    """
    if display_frame not in exposure_data:
        return None, None, None, None
    exp = exposure_data[display_frame]
    if exp is None:
        return None, None, None, None
    exp = _ensure_compact(exp)
    pos_steps = exp['pos_steps'].astype(float)  # (steps, N, 3)

    NUCLEUS_RADIUS_NM = nucleus_diameter_nm / 2

    if mode == 'centroid':
        positions = pos_steps.mean(axis=0)
    elif mode == 'snapshot':
        positions = pos_steps[-1]
    elif mode == 'full':
        positions = pos_steps.reshape(-1, 3)
    else:
        raise ValueError("mode must be 'full', 'snapshot', or 'centroid'")

    coords = positions * 1000  # µm → nm

    # Lateral (XY) grid — matches convolved image pixel grid
    GRID_SIZE = int(np.ceil(nucleus_diameter_nm / voxel_size_nm)) + 2
    if GRID_SIZE % 2 == 0:
        GRID_SIZE += 1
    half_grid_span = (GRID_SIZE * voxel_size_nm) / 2.0

    # XY: 108 nm voxel binning (unchanged)
    gx = np.floor((coords[:, 0] + half_grid_span) / voxel_size_nm).astype(int)
    gy = np.floor((coords[:, 1] + half_grid_span) / voxel_size_nm).astype(int)

    # Z: direct nearest-slab assignment
    slab_centers = np.sort(np.unique(np.concatenate([
        np.arange(0, NUCLEUS_RADIUS_NM, slice_spacing_nm),
        -np.arange(0, NUCLEUS_RADIUS_NM, slice_spacing_nm)
    ])))
    n_slabs = len(slab_centers)
    center_slab = n_slabs // 2  # index of slab centered at z=0

    # floor(z/spacing + 0.5) gives round-half-up — assigns z to nearest slab.
    # Boundary band [k*spacing - spacing/2, k*spacing + spacing/2) maps to slab k.
    sz_offset = np.floor(coords[:, 2] / slice_spacing_nm + 0.5).astype(int)
    sz = sz_offset + center_slab

    valid = (
        (gx >= 0) & (gx < GRID_SIZE) &
        (gy >= 0) & (gy < GRID_SIZE) &
        (sz >= 0) & (sz < n_slabs)
    )

    count_grid = np.zeros((GRID_SIZE, GRID_SIZE, n_slabs), dtype=np.int32)
    np.add.at(count_grid, (gx[valid], gy[valid], sz[valid]), 1)

    # Apply spherical nucleus mask per slab using the slab's z-center for
    # the z² term. This matches the cross-section semantics of the renderer
    # (the convolved image masks per slab too via NUCLEUS_RADIUS² - z_nm²).
    voxel_centers_xy = np.arange(GRID_SIZE) * voxel_size_nm - half_grid_span + voxel_size_nm / 2
    xn, yn = np.meshgrid(voxel_centers_xy, voxel_centers_xy, indexing='ij')
    xy_sq = xn**2 + yn**2
    R_sq = NUCLEUS_RADIUS_NM**2
    for s_idx, zc in enumerate(slab_centers):
        cross_section_r_sq = max(0.0, R_sq - zc**2)
        slab_mask = xy_sq <= cross_section_r_sq
        count_grid[:, :, s_idx] *= slab_mask

    return count_grid, half_grid_span, GRID_SIZE, slab_centers


def render_count_slices(count_grid, nucleus_diameter_nm, voxel_size_nm,
                        half_span, grid_size, slice_spacing_nm,
                        upscale_factor=1, colormap_name='turbo',
                        save_to_disk=False, output_dir=None, frame_idx=0,
                        save_prefix="", z_projection='slab'):
    """
    Render molecule-count z-slices from a count grid whose third axis is
    already the slab index (as produced by the updated
    ``calculate_count_grid``).

    Produces both a 16-bit integer TIFF (raw counts) and an RGB colormapped
    image for each z-slab.

    Parameters
    ----------
    count_grid : ndarray, int32
        Shape (GRID_SIZE, GRID_SIZE, n_slabs). Third axis is the slab index,
        NOT a 108 nm z-voxel index.
    colormap_name : str
        Matplotlib colormap name for the preview images.
    z_projection : str
        Accepted for backward compatibility but ignored. With the new direct
        slab-assignment binning, each molecule is already in exactly one slab,
        so 'slab' and 'single' would be identical.

    Returns
    -------
    result_dict : dict
        ``{z_idx: {'raw': PIL.Image (16-bit), 'color': PIL.Image (RGB)}}``
    global_max : int
        Maximum count across all slabs (useful for consistent LUT scaling).
    """
    import matplotlib.cm as cm

    NUCLEUS_RADIUS_NM = nucleus_diameter_nm / 2
    plot_limit = NUCLEUS_RADIUS_NM * 1.1

    voxel_centers_1d = np.arange(grid_size) * voxel_size_nm - half_span + voxel_size_nm / 2
    in_view = (voxel_centers_1d >= -plot_limit) & (voxel_centers_1d <= plot_limit)
    view_indices = np.where(in_view)[0]
    if len(view_indices) > 0:
        i_start, i_end = view_indices[0], view_indices[-1] + 1
    else:
        i_start, i_end = 0, grid_size

    n_slabs = count_grid.shape[2]

    # Find global max for consistent colormap scaling
    if n_slabs > 0:
        global_max = int(count_grid[i_start:i_end, i_start:i_end, :].max())
    else:
        global_max = 0

    cmap = cm.get_cmap(colormap_name)
    result_dict = {}

    for z_idx in range(n_slabs):
        # Crop to in-view XY, transpose to (rows=y, cols=x)
        slice_2d = count_grid[i_start:i_end, i_start:i_end, z_idx].T
        # Flip to match matplotlib origin='lower' (same convention as convolved)
        slice_flipped = np.flipud(slice_2d)

        # 16-bit raw count image
        raw_array = np.clip(slice_flipped, 0, 65535).astype(np.uint16)
        if upscale_factor > 1:
            raw_up = np.repeat(np.repeat(raw_array, upscale_factor, axis=0), upscale_factor, axis=1)
        else:
            raw_up = raw_array
        raw_img = Image.fromarray(raw_up, mode='I;16')

        # Colormapped RGB image
        if global_max > 0:
            normed = slice_flipped.astype(float) / global_max
        else:
            normed = np.zeros_like(slice_flipped, dtype=float)
        rgba = cmap(normed)
        rgb_array = (rgba[:, :, :3] * 255).astype(np.uint8)
        if upscale_factor > 1:
            rgb_array = np.repeat(np.repeat(rgb_array, upscale_factor, axis=0), upscale_factor, axis=1)
        color_img = Image.fromarray(rgb_array, mode='RGB')

        if save_to_disk and output_dir:
            raw_path = os.path.join(output_dir, f"{save_prefix}f{frame_idx}_z{z_idx}_count.png")
            color_path = os.path.join(output_dir, f"{save_prefix}f{frame_idx}_z{z_idx}_count_color.png")
            raw_img.save(raw_path)
            color_img.save(color_path)
            result_dict[z_idx] = {'raw': raw_path, 'color': color_path}
        else:
            result_dict[z_idx] = {'raw': raw_img, 'color': color_img}

    return result_dict, global_max


def generate_fbm_step(prev_step, hurst, num_clusters, cluster_D, sim_dt_s):
    step_std = np.sqrt(2 * cluster_D * sim_dt_s)
    new_random = np.random.normal(0, 1, (num_clusters, 3))
    
    rho = 2.0 ** (2.0 * hurst - 1.0) - 1.0
    rho = np.clip(rho, -0.99, 0.99)
    
    if prev_step is not None:
        prev_norms = np.linalg.norm(prev_step, axis=1, keepdims=True)
        prev_norms = np.where(prev_norms > 1e-10, prev_norms, 1.0)
        prev_direction = prev_step / prev_norms
        combined = rho * prev_direction + np.sqrt(1.0 - rho**2) * new_random
    else:
        combined = new_random
    
    step = combined * step_std[:, np.newaxis]
    return step


def generate_all_fbm_steps_dh(total_steps, num_clusters, cluster_D, hurst, sim_dt_s):
    """
    Pre-generate all cluster fBm increments using the Davies-Harte / Wood & Chan
    (1994) exact method, fully vectorised across all clusters and spatial dims.

    All num_clusters * 3 independent fGn series are produced in a single batched
    FFT call, making this much faster than looping over individual series.

    Returns
    -------
    steps : np.ndarray, shape (total_steps, num_clusters, 3)
    """
    n = int(total_steps)
    H = float(np.clip(hurst, 0.01, 0.99))
    step_stds = np.sqrt(2.0 * np.maximum(cluster_D, 0.0) * sim_dt_s)  # (num_clusters,)

    if n < 2 or num_clusters == 0:
        out = np.zeros((n, num_clusters, 3))
        for d in range(3):
            out[:, :, d] = np.random.normal(size=(n, num_clusters)) * step_stds[np.newaxis, :]
        return out

    N = 2 * n

    # fGn autocovariance gamma(k), k = 0..n
    k = np.arange(n + 1, dtype=float)
    autocov = 0.5 * (
        np.abs(k - 1) ** (2.0 * H)
        - 2.0 * np.abs(k) ** (2.0 * H)
        + np.abs(k + 1) ** (2.0 * H)
    )

    # Circulant first row of length 2n
    r = np.empty(N)
    r[:n + 1] = autocov
    r[n + 1:] = autocov[1:n][::-1]

    # Real eigenvalues of the circulant
    lam = np.fft.fft(r).real
    lam = np.maximum(lam, 0.0)

    # Batched Wood & Chan spectral construction
    # All num_clusters * 3 series at once — one FFT call total
    n_series = num_clusters * 3
    w = np.zeros((n_series, N), dtype=complex)

    # Boundary terms: real Gaussians at j=0 and j=n
    w[:, 0] = np.sqrt(lam[0] / N) * np.random.normal(size=n_series)
    w[:, n] = np.sqrt(lam[n] / N) * np.random.normal(size=n_series)

    # Interior terms: complex Gaussians with Hermitian symmetry
    scale_mid = np.sqrt(lam[1:n] / (2.0 * N))
    w[:, 1:n] = scale_mid[np.newaxis, :] * (
        np.random.normal(size=(n_series, n - 1))
        + 1j * np.random.normal(size=(n_series, n - 1))
    )
    w[:, n + 1:] = np.conj(w[:, 1:n][:, ::-1])

    # Single batched FFT -> unit-variance fGn, shape (n_series, n)
    fgn = np.fft.fft(w, axis=1).real[:, :n]

    # (n_series, n) -> (num_clusters, 3, n) -> (n, num_clusters, 3)
    fgn = fgn.reshape(num_clusters, 3, n).transpose(2, 0, 1)

    # Scale to physical displacements
    steps = fgn * step_stds[np.newaxis, :, np.newaxis]

    return steps


def run_full_simulation(
    num_display_frames, num_binding_sites, nucleus_diam_nm, tf_concentration_nM,  
    sim_dt_s, display_dt_s, 
    pct_free_init, pct_inter_init, pct_ns_init,
    d_inter, d_free, d_ns, 
    d_free_std, d_inter_std, d_ns_std,
    bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
    k_free_to_inter, k_inter_to_free,
    k_free_to_ns, k_inter_to_ns,
    ns_res_time, 
    cluster_distribution,
    res_time_mean, res_time_std,
    capture_radius_nm, k_on_prob,
    strength_multipliers=None, strength_percentages=None,
    cluster_correlated_strength=False,
    num_synthetic_sites=0, synthetic_strength=1.0, synthetic_cluster_radius_nm=150.0, cluster_radius_nm=150.0,
    exposure_time_s=0.01,
    enable_cooperativity=False, coop_strength=1.0, coop_hill=1.0,
    two_state_mode=False,
    show_progress=True,
    binding_check_interval=1,
    progress_callback=None,
    sim_seed=None,
    motion_seed=None,
    num_molecules_override=None,
):
    # === REPRODUCIBILITY ====================================================
    # Two-tier seeding mirroring the HPC sim_group / site_group pattern in
    # batch_hpc_fast.py + sim_repro.py:
    #
    #   sim_seed alone (motion_seed is None):
    #       Single seed locks everything. Equivalent to the HPC `sim_group`
    #       behavior — sites + strengths + cluster_D + fBm/jiggle + initial
    #       molecule positions + state transitions + binding dice are all
    #       deterministic from this one seed.
    #
    #   sim_seed + motion_seed='fresh':
    #       Sites locked, motion re-seeded from OS entropy after site setup.
    #       Site positions, strengths, and initial cluster_D are reproducible;
    #       molecule motion is fresh each run. Useful for interactive
    #       exploration where you want to vary conc / kinetics and see new
    #       motion realizations on a fixed binding-site landscape.
    #
    #   sim_seed + motion_seed=int:
    #       Sites locked from sim_seed, motion locked from motion_seed
    #       (independently). Equivalent to the HPC `site_group`-only behavior
    #       once the motion seed is derived deterministically via
    #       _derive_motion_seed(). Same sim_seed + same motion_seed = same
    #       motion regardless of whether site_physics columns shifted the
    #       site-setup RNG state.
    #
    # The motion-seed boundary is applied AFTER site setup completes
    # (sites, strengths, cluster_D) and BEFORE the first molecule-related
    # draw. See the second seeding block below in this function.
    if sim_seed is not None:
        np.random.seed(int(sim_seed))
    # ========================================================================
    radius_um = (nucleus_diam_nm / 2) / 1000.0
    capture_radius_um = capture_radius_nm / 1000.0
    
    if sim_dt_s <= 0: sim_dt_s = 0.013
    steps_per_display = int(display_dt_s / sim_dt_s)
    if steps_per_display < 1: steps_per_display = 1
    steps_per_exposure = max(1, int(exposure_time_s / sim_dt_s))
    
    vol_um3 = (4/3) * np.pi * (radius_um**3)
    if num_molecules_override is not None:
        num_molecules = int(num_molecules_override)
    else:
        num_molecules = int(((tf_concentration_nM * 1e-9) * (vol_um3 * 1e-15)) * 6.022e23)

    sites, cluster_ids, is_synthetic = generate_clustered_sites(
        num_binding_sites, radius_um, cluster_distribution, cluster_radius_nm=cluster_radius_nm,
        num_synthetic=num_synthetic_sites, synthetic_radius_nm=synthetic_cluster_radius_nm
    )
    site_occupancy = np.full(len(sites), -1, dtype=int)
    
    if strength_multipliers is None: strength_multipliers = [1.0]
    if strength_percentages is None: strength_percentages = [100.0]
    
    site_strengths = assign_site_strengths(
        len(sites), cluster_ids, strength_multipliers, strength_percentages,
        is_synthetic, synthetic_strength, cluster_correlated_strength
    )
    
    inner_radius = radius_um - capture_radius_um
    s_dists = np.linalg.norm(sites, axis=1)
    m_out = s_dists > inner_radius
    if np.any(m_out):
        sites[m_out] *= (inner_radius / s_dists[m_out])[:, np.newaxis]

    # sites_base holds the rigid (jiggle-free) cluster center positions.
    # fBm diffusion updates sites_base; sites = sites_base + fresh jiggle each step.
    sites_base = sites.copy()
    
    unique_clusters = np.unique(cluster_ids)
    num_clusters = len(unique_clusters)
    cluster_D = np.maximum(0.005, np.random.normal(bs_diff_mean, bs_diff_std, num_clusters))
    
    cluster_id_to_idx = {cid: i for i, cid in enumerate(unique_clusters)}
    site_cluster_idx = np.array([cluster_id_to_idx[cid] for cid in cluster_ids])
    cluster_occupancy_count = np.zeros(num_clusters, dtype=int)

    # === MOTION-SEED BOUNDARY ==============================================
    # Site setup (positions, strengths, cluster_D) is complete. Optionally
    # re-seed the global RNG before any molecule-related draws so that the
    # motion stream is decoupled from the cumulative state of the site
    # stream. See the docstring at the top of this function for the three
    # supported modes.
    #
    # Caveat: app2.py uses the global np.random throughout, so once we
    # re-seed here, any subsequent np.random calls from the per-step site
    # dynamics (fBm, jiggle) will also draw from the motion stream. This is
    # an intentional simplification — the HPC engine separates these into
    # site_rng and mol_rng via Generator instances; here we lock initial
    # site setup (which is what callers actually need) and accept that
    # site dynamics inside the loop ride on the motion RNG.
    if motion_seed is not None:
        if isinstance(motion_seed, str) and motion_seed.lower() == 'fresh':
            np.random.seed()  # OS entropy
        else:
            np.random.seed(int(motion_seed))
    # ========================================================================

    pos = np.random.normal(size=(num_molecules, 3))
    pos /= np.linalg.norm(pos, axis=1)[:, np.newaxis]
    pos *= (np.random.random(num_molecules) ** (1/3) * radius_um)[:, np.newaxis]
    
    if two_state_mode:
        # 2-state: only free (0) and NS-bound (3), no intermediate
        total_pct = pct_free_init + pct_ns_init
        if total_pct == 0: p_dist_2s = [1, 0]
        else: p_dist_2s = [pct_free_init/total_pct, pct_ns_init/total_pct]
        states = np.random.choice([0, 3], size=num_molecules, p=p_dist_2s)
        p_dest_free = 1.0
        p_dest_inter = 0.0
    else:
        total_pct = pct_free_init + pct_inter_init + pct_ns_init
        if total_pct == 0: p_dist = [1, 0, 0]
        else: p_dist = [pct_free_init/total_pct, pct_inter_init/total_pct, pct_ns_init/total_pct]
        states = np.random.choice([0, 1, 3], size=num_molecules, p=p_dist)
        sum_mobile = pct_free_init + pct_inter_init
        if sum_mobile > 0:
            p_dest_free = pct_free_init / sum_mobile
            p_dest_inter = pct_inter_init / sum_mobile
        else:
            p_dest_free, p_dest_inter = 0.5, 0.5

    bound_site_indices = np.full(num_molecules, -1, dtype=int)
    unbind_times = np.zeros(num_molecules)
    
    p_fi = k_free_to_inter * sim_dt_s
    p_if = k_inter_to_free * sim_dt_s
    p_fns = k_free_to_ns * sim_dt_s
    p_ins = k_inter_to_ns * sim_dt_s
    p_ns_out = (1.0 / ns_res_time) * sim_dt_s if ns_res_time > 0 else 1.0

    def sample_ds(state_arr):
        ds = np.zeros_like(state_arr, dtype=float)
        m0 = (state_arr == 0)
        ds[m0] = np.maximum(0, np.random.normal(d_free, d_free_std, size=np.sum(m0)))
        if not two_state_mode:
            m1 = (state_arr == 1)
            ds[m1] = np.maximum(0, np.random.normal(d_inter, d_inter_std, size=np.sum(m1)))
        m_ns = (state_arr == 3)
        ds[m_ns] = np.maximum(0, np.random.normal(d_ns, d_ns_std, size=np.sum(m_ns)))
        return ds

    current_ds = sample_ds(states)
    exposure_data = {f: [] for f in range(num_display_frames)}
    current_time = 0.0
    bs_hurst = np.clip(bs_hurst, 0.01, 0.99)
    prev_cluster_step = None

    # ---- Per-frame fBm with exact conditioning at boundaries ----
    # Instead of pre-generating ALL fBm steps upfront, we generate a fresh
    # Davies-Harte block per display frame and condition it on the previous
    # frame's last step via an exact Gaussian bridge.  This keeps the correct
    # long-range autocorrelation within each frame while using less memory.
    _use_dh_fbm = (bs_diff_mean > 0 or bs_diff_std > 0)
    precomputed_bs_steps = None
    _frame_step_base = 0
    if _use_dh_fbm:
        _step_stds = np.sqrt(2.0 * np.maximum(cluster_D, 0.0) * sim_dt_s)
        _H = float(bs_hurst)
        _r1 = 0.5 * (2.0 ** (2.0 * _H) - 2.0)  # fGn autocovariance at lag 1

    if show_progress:
        sim_progress = st.progress(0)
    
    physics_step = 0
    binding_check_interval = max(1, int(binding_check_interval))
    
    sites_are_static = (bs_diff_mean <= 0 and bs_diff_std <= 0 and bs_jiggle_std <= 0)
    cached_tree = cKDTree(sites) if sites_are_static else None
    
    for display_frame in range(num_display_frames):
        # Generate fBm steps for this frame only
        if _use_dh_fbm:
            _nf = steps_per_display + steps_per_exposure + 2
            _dh = generate_all_fbm_steps_dh(
                _nf, num_clusters, cluster_D, bs_hurst, sim_dt_s)
            if display_frame > 0 and prev_cluster_step is not None:
                # Exact Gaussian bridge: r(1)*prev + sqrt(1-r(1)^2)*std*Z
                _bridge = (_r1 * prev_cluster_step
                           + np.sqrt(1.0 - _r1 ** 2)
                           * _step_stds[:, np.newaxis]
                           * np.random.normal(size=prev_cluster_step.shape))
                # Condition DH block on first element = bridge
                _k = np.arange(1, _nf, dtype=float)
                _rk = 0.5 * (np.abs(_k - 1) ** (2 * _H)
                             - 2 * np.abs(_k) ** (2 * _H)
                             + np.abs(_k + 1) ** (2 * _H))
                _dh[1:] += _rk[:, np.newaxis, np.newaxis] * (
                    _bridge[np.newaxis, :, :] - _dh[0:1, :, :])
                _dh[0] = _bridge
            precomputed_bs_steps = _dh
            _frame_step_base = physics_step

        target_step = display_frame * steps_per_display
        
        while physics_step < target_step:
            check_binding = (physics_step % binding_check_interval == 0)
            _dh_step = (
                precomputed_bs_steps[physics_step - _frame_step_base]
                if precomputed_bs_steps is not None
                and (physics_step - _frame_step_base) < len(precomputed_bs_steps)
                else None
            )
            prev_cluster_step, cached_tree = _run_physics_step(
                pos, states, sites, sites_base, site_occupancy, bound_site_indices, unbind_times,
                current_ds, sample_ds, cluster_D, site_cluster_idx, site_strengths,
                p_fi, p_if, p_fns, p_ins, p_ns_out, p_dest_free, p_dest_inter,
                bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
                radius_um, capture_radius_um, inner_radius, sim_dt_s, current_time,
                res_time_mean, res_time_std, k_on_prob, num_clusters,
                prev_cluster_step,
                enable_cooperativity, coop_strength, coop_hill, cluster_occupancy_count,
                two_state_mode=two_state_mode,
                check_binding=check_binding, cached_tree=cached_tree,
                precomputed_cluster_step=_dh_step
            )
            current_time += sim_dt_s
            physics_step += 1
        
        for exp_step in range(steps_per_exposure):
            exposure_data[display_frame].append({
                'pos': pos.copy(), 'states': states.copy(),
                'sites': sites.copy(), 'occupancy': (site_occupancy >= 0).copy()
            })
            _dh_step = (
                precomputed_bs_steps[physics_step - _frame_step_base]
                if precomputed_bs_steps is not None
                and (physics_step - _frame_step_base) < len(precomputed_bs_steps)
                else None
            )
            prev_cluster_step, cached_tree = _run_physics_step(
                pos, states, sites, sites_base, site_occupancy, bound_site_indices, unbind_times,
                current_ds, sample_ds, cluster_D, site_cluster_idx, site_strengths,
                p_fi, p_if, p_fns, p_ins, p_ns_out, p_dest_free, p_dest_inter,
                bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
                radius_um, capture_radius_um, inner_radius, sim_dt_s, current_time,
                res_time_mean, res_time_std, k_on_prob, num_clusters,
                prev_cluster_step,
                enable_cooperativity, coop_strength, coop_hill, cluster_occupancy_count,
                two_state_mode=two_state_mode,
                check_binding=True, cached_tree=cached_tree,
                precomputed_cluster_step=_dh_step
            )
            current_time += sim_dt_s
            physics_step += 1
        
        if show_progress:
            sim_progress.progress((display_frame + 1) / num_display_frames)
        if progress_callback:
            progress_callback(display_frame + 1, num_display_frames)
    
    if show_progress:
        sim_progress.empty()
    
    return exposure_data, num_molecules, site_strengths, is_synthetic, steps_per_exposure


def _run_physics_step(
    pos, states, sites, sites_base, site_occupancy, bound_site_indices, unbind_times,
    current_ds, sample_ds, cluster_D, site_cluster_idx, site_strengths,
    p_fi, p_if, p_fns, p_ins, p_ns_out, p_dest_free, p_dest_inter,
    bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
    radius_um, capture_radius_um, inner_radius, sim_dt_s, current_time,
    res_time_mean, res_time_std, k_on_prob, num_clusters,
    prev_cluster_step,
    enable_cooperativity, coop_strength, coop_hill, cluster_occupancy_count,
    two_state_mode=False,
    check_binding=True, cached_tree=None,
    precomputed_cluster_step=None
):
    # --- State transitions ---
    changed_indices = []
    
    if two_state_mode:
        # 2-state mode: Free ↔ NS-bound + Specific-bound (no intermediate)
        # Free -> NS transitions
        mask_0 = (states == 0)
        if np.any(mask_0):
            r = np.random.random(size=np.sum(mask_0))
            to_ns = r < p_fns
            idx = np.where(mask_0)[0]
            ns_changers = idx[to_ns]
            states[ns_changers] = 3
            if len(ns_changers) > 0:
                changed_indices.append(ns_changers)
        
        # NS-bound -> Free (always free, never intermediate)
        mask_3 = (states == 3)
        if np.any(mask_3):
            r = np.random.random(size=np.sum(mask_3))
            leaving = r < p_ns_out
            idx = np.where(mask_3)[0]
            leavers = idx[leaving]
            states[leavers] = 0
            if len(leavers) > 0:
                changed_indices.append(leavers)
    else:
        # 3-state mode: Free ↔ Intermediate ↔ NS-bound + Specific-bound
        mask_0 = (states == 0)
        if np.any(mask_0):
            r = np.random.random(size=np.sum(mask_0))
            to_inter = r < p_fi
            to_ns = (r >= p_fi) & (r < (p_fi + p_fns))
            idx = np.where(mask_0)[0]
            changed_0 = np.concatenate([idx[to_inter], idx[to_ns]])
            states[idx[to_inter]] = 1
            states[idx[to_ns]] = 3
            if len(changed_0) > 0:
                changed_indices.append(changed_0)
            
        mask_1 = (states == 1)
        if np.any(mask_1):
            r = np.random.random(size=np.sum(mask_1))
            to_free = r < p_if
            to_ns = (r >= p_if) & (r < (p_if + p_ins))
            idx = np.where(mask_1)[0]
            changed_1 = np.concatenate([idx[to_free], idx[to_ns]])
            states[idx[to_free]] = 0
            states[idx[to_ns]] = 3
            if len(changed_1) > 0:
                changed_indices.append(changed_1)
            
        mask_3 = (states == 3)
        if np.any(mask_3):
            r = np.random.random(size=np.sum(mask_3))
            leaving = r < p_ns_out
            idx = np.where(mask_3)[0]
            leavers = idx[leaving]
            new_dest = np.random.choice([0, 1], size=len(leavers), p=[p_dest_free, p_dest_inter])
            states[leavers] = new_dest
            if len(leavers) > 0:
                changed_indices.append(leavers)
    
    if changed_indices:
        all_changed = np.concatenate(changed_indices)
        current_ds[all_changed] = sample_ds(states[all_changed])

    # --- Binding site motion ---
    sites_moved = False
    current_cluster_step = None
    if bs_diff_mean > 0 or bs_diff_std > 0 or bs_jiggle_std > 0:
        sites_moved = True
        if bs_diff_mean > 0 or bs_diff_std > 0:
            if precomputed_cluster_step is not None:
                cluster_step = precomputed_cluster_step  # shape (num_clusters, 3)
            else:
                cluster_step = generate_fbm_step(
                    prev_cluster_step, bs_hurst, num_clusters, cluster_D, sim_dt_s
                )
            # fBm diffusion updates the rigid base positions only
            sites_base += cluster_step[site_cluster_idx]
            current_cluster_step = cluster_step

        # Boundary enforcement on base positions
        s_dists = np.linalg.norm(sites_base, axis=1)
        m_out = s_dists > inner_radius
        sites_base[m_out] *= (inner_radius / s_dists[m_out])[:, np.newaxis]

        if bs_jiggle_std > 0:
            # Fresh independent noise each step — amplitude is always bs_jiggle_std,
            # never accumulates. sites_base is the cluster center; sites is what
            # molecules and the kd-tree see.
            sites[:] = sites_base + np.random.normal(0, bs_jiggle_std, sites.shape)
            # Clamp sites to radius_um after jiggle
            _sd = np.linalg.norm(sites, axis=1)
            _so = _sd > radius_um
            if np.any(_so):
                sites[_so] *= (radius_um / _sd[_so])[:, np.newaxis]
        else:
            sites[:] = sites_base

    # --- Unbinding ---
    mask_bound = (states == 2)
    if np.any(mask_bound):
        releasing = mask_bound & (unbind_times <= current_time)
        if np.any(releasing):
            rel_idx = np.where(releasing)[0]
            target_sites = bound_site_indices[rel_idx]
            valid_releases = (site_occupancy[target_sites] == rel_idx)
            final_target_sites = target_sites[valid_releases]
            final_rel_idx = rel_idx[valid_releases]
            site_occupancy[final_target_sites] = -1
            if enable_cooperativity:
                c_indices = site_cluster_idx[final_target_sites]
                np.subtract.at(cluster_occupancy_count, c_indices, 1)
                cluster_occupancy_count[cluster_occupancy_count < 0] = 0
            if two_state_mode:
                # 2-state: unbound molecules become free or NS-bound (no intermediate)
                states[final_rel_idx] = np.random.choice([0, 3], size=len(final_rel_idx),
                    p=[p_dest_free, 1.0 - p_dest_free])
            else:
                # 3-state: use same population-derived probabilities as NS unbinding
                states[final_rel_idx] = np.random.choice([0, 1], size=len(final_rel_idx),
                    p=[p_dest_free, p_dest_inter])
            bound_site_indices[final_rel_idx] = -1
            current_ds[final_rel_idx] = sample_ds(states[final_rel_idx])

    # --- Diffusion ---
    mobile_mask = (states != 2)
    mobile_idx = np.where(mobile_mask)[0]
    if len(mobile_idx) > 0:
        pos[mobile_idx] += np.random.normal(0, np.sqrt(2 * current_ds[mobile_idx] * sim_dt_s)[:, np.newaxis], (len(mobile_idx), 3))
    
    bound_mask = (states == 2)
    if np.any(bound_mask):
        pos[bound_mask] = sites[bound_site_indices[bound_mask]] 

    # --- Binding check ---
    new_tree = cached_tree
    if check_binding and len(mobile_idx) > 0:
        if cached_tree is None or sites_moved:
            new_tree = cKDTree(sites)
        
        dists, s_ids = new_tree.query(pos[mobile_idx], k=1, distance_upper_bound=capture_radius_um)
        
        valid_check = (dists < capture_radius_um) & (s_ids < len(sites))
        if np.any(valid_check):
            cand_mol = mobile_idx[valid_check]
            cand_site = s_ids[valid_check]
            
            site_free = (site_occupancy[cand_site] == -1)
            cand_mol = cand_mol[site_free]
            cand_site = cand_site[site_free]
            
            if len(cand_mol) > 0:
                if enable_cooperativity:
                    c_ids = site_cluster_idx[cand_site]
                    n_occ = cluster_occupancy_count[c_ids]
                    mult = 1.0 + (coop_strength * n_occ) ** coop_hill
                    probs = np.minimum(1.0, k_on_prob * mult)
                else:
                    probs = k_on_prob
                
                binds = np.random.random(len(cand_mol)) < probs
                cand_mol = cand_mol[binds]
                cand_site = cand_site[binds]
                
                if len(cand_mol) > 0:
                    shuffle = np.random.permutation(len(cand_mol))
                    cand_mol = cand_mol[shuffle]
                    cand_site = cand_site[shuffle]
                    _, unique_site_idx = np.unique(cand_site, return_index=True)
                    cand_mol = cand_mol[unique_site_idx]
                    cand_site = cand_site[unique_site_idx]
                    
                    states[cand_mol] = 2
                    bound_site_indices[cand_mol] = cand_site
                    pos[cand_mol] = sites[cand_site]
                    site_occupancy[cand_site] = cand_mol
                    current_ds[cand_mol] = 0.0
                    
                    if enable_cooperativity:
                        np.add.at(cluster_occupancy_count, site_cluster_idx[cand_site], 1)
                    
                    site_str = site_strengths[cand_site]
                    unbind_times[cand_mol] = current_time + np.maximum(0.1,
                        np.random.normal(res_time_mean * site_str, res_time_std * site_str))

    # --- Boundary enforcement (mobile molecules only - bound ones are at site positions) ---
    if len(mobile_idx) > 0:
        _bd = np.linalg.norm(pos[mobile_idx], axis=1)
        _bo = _bd > radius_um
        if np.any(_bo):
            _oi = mobile_idx[_bo]
            pos[_oi] *= (radius_um / _bd[_bo])[:, np.newaxis]
    
    return current_cluster_step, new_tree

# --- 4. PROCESSING & VISUALIZATION ---

def calculate_grid_for_frame(frame_df, nucleus_diameter_nm, voxel_size_nm, lateral_fwhm_nm, axial_fwhm_nm):
    """Calculate 3D grid with anisotropic Gaussian PSF."""
    NUCLEUS_RADIUS_NM = nucleus_diameter_nm / 2
    coords = frame_df[['x', 'y', 'z']].values * 1000
    
    GRID_SIZE = int(np.ceil(nucleus_diameter_nm / voxel_size_nm)) + 2
    if GRID_SIZE % 2 == 0: GRID_SIZE += 1
    half_grid_span = (GRID_SIZE * voxel_size_nm) / 2.0
    
    coords_shifted = coords + half_grid_span
    grid_indices = np.floor(coords_shifted / voxel_size_nm).astype(int)
    molecule_grid_3d = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=float)
    
    SIGMA_LATERAL_NM = lateral_fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    SIGMA_AXIAL_NM = axial_fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    SIGMA_LATERAL_VOXELS = SIGMA_LATERAL_NM / voxel_size_nm
    SIGMA_AXIAL_VOXELS = SIGMA_AXIAL_NM / voxel_size_nm
    
    sigma_xy = max(SIGMA_LATERAL_VOXELS, 0.5)
    sigma_z = max(SIGMA_AXIAL_VOXELS, 0.5)
    
    kernel_radius_xy = int(np.ceil(3 * sigma_xy))
    kernel_radius_z = int(np.ceil(3 * sigma_z))
    
    ax = np.arange(-kernel_radius_xy, kernel_radius_xy + 1)
    ay = np.arange(-kernel_radius_xy, kernel_radius_xy + 1)
    az = np.arange(-kernel_radius_z, kernel_radius_z + 1)
    xx, yy, zz = np.meshgrid(ax, ay, az, indexing='ij')
    
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_xy**2) - zz**2 / (2 * sigma_z**2))
    kernel /= kernel.sum()

    for gx, gy, gz in grid_indices:
        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE and 0 <= gz < GRID_SIZE:
             x_start, x_end = max(0, gx-kernel_radius_xy), min(GRID_SIZE, gx+kernel_radius_xy+1)
             y_start, y_end = max(0, gy-kernel_radius_xy), min(GRID_SIZE, gy+kernel_radius_xy+1)
             z_start, z_end = max(0, gz-kernel_radius_z), min(GRID_SIZE, gz+kernel_radius_z+1)
             kx_start, kx_end = x_start-(gx-kernel_radius_xy), x_end-(gx-kernel_radius_xy)
             ky_start, ky_end = y_start-(gy-kernel_radius_xy), y_end-(gy-kernel_radius_xy)
             kz_start, kz_end = z_start-(gz-kernel_radius_z), z_end-(gz-kernel_radius_z)
             if kx_end > kx_start and ky_end > ky_start and kz_end > kz_start:
                 molecule_grid_3d[x_start:x_end, y_start:y_end, z_start:z_end] += \
                     kernel[kx_start:kx_end, ky_start:ky_end, kz_start:kz_end]

    voxel_centers = np.arange(GRID_SIZE) * voxel_size_nm - half_grid_span + voxel_size_nm/2
    x_nm, y_nm, z_nm = np.meshgrid(voxel_centers, voxel_centers, voxel_centers, indexing='ij')
    mask = (x_nm**2 + y_nm**2 + z_nm**2) <= NUCLEUS_RADIUS_NM**2
    molecule_grid_3d *= mask
            
    return molecule_grid_3d, half_grid_span, GRID_SIZE


def render_ground_truth_for_frame(frame_df, output_dir, frame_idx, nucleus_diameter_nm, molecule_diameter_nm,
                                   image_dpi, slice_spacing_nm, binding_sites_pos, site_occupied,
                                   site_strengths=None, strength_range=(0.1, 10.0), is_synthetic=None,
                                   save_to_disk=True, save_prefix="", draw_boundary=True, upscale_factor=12,
                                   voxel_size_nm=108.0, num_slices=None):
    """Render ground truth at high resolution, then scale to match convolved image dimensions."""
    NUCLEUS_RADIUS_NM = nucleus_diameter_nm / 2
    coords = frame_df[['x', 'y', 'z']].values * 1000
    plot_limit = NUCLEUS_RADIUS_NM * 1.1
    
    GRID_SIZE = int(np.ceil(nucleus_diameter_nm / voxel_size_nm)) + 2
    if GRID_SIZE % 2 == 0: GRID_SIZE += 1
    half_grid_span = (GRID_SIZE * voxel_size_nm) / 2.0
    
    voxel_centers_1d = np.arange(GRID_SIZE) * voxel_size_nm - half_grid_span + voxel_size_nm/2
    in_view = (voxel_centers_1d >= -plot_limit) & (voxel_centers_1d <= plot_limit)
    view_indices = np.where(in_view)[0]
    if len(view_indices) > 0:
        i_start, i_end = view_indices[0], view_indices[-1] + 1
    else:
        i_start, i_end = 0, GRID_SIZE
    
    convolved_native_pixels = i_end - i_start
    target_final_pixels = convolved_native_pixels * upscale_factor
    native_pixels_gt = max(target_final_pixels, int(np.ceil(2 * plot_limit / molecule_diameter_nm)))
    
    crop_min = voxel_centers_1d[i_start] - voxel_size_nm/2
    crop_max = voxel_centers_1d[i_end-1] + voxel_size_nm/2
    
    paths_or_images = []
    if num_slices is not None:
        _n_half_gt = int(num_slices) // 2
        _cand_gt = sorted({i * slice_spacing_nm for i in range(-_n_half_gt, _n_half_gt + 1)})
        all_z_positions = [z for z in _cand_gt if abs(z) <= NUCLEUS_RADIUS_NM]
    else:
        all_z_positions = sorted(np.unique(np.concatenate([
            np.arange(0, NUCLEUS_RADIUS_NM, slice_spacing_nm),
            -np.arange(0, NUCLEUS_RADIUS_NM, slice_spacing_nm)
        ])).tolist())
    
    log_min = np.log10(max(strength_range[0], 1e-6))
    log_max = np.log10(max(strength_range[1], 1e-6))
    
    for i, z_nm in enumerate(sorted(all_z_positions)):
        native_fig_size = native_pixels_gt / image_dpi
        fig_gt, ax_gt = plt.subplots(figsize=(native_fig_size, native_fig_size), facecolor='black')
        ax_gt.set_position([0, 0, 1, 1])
        ax_gt.set_axis_off()
        
        nuc_cross_section_r = np.sqrt(max(0, NUCLEUS_RADIUS_NM**2 - z_nm**2))
        if draw_boundary:
            ax_gt.add_patch(plt.Circle((0, 0), nuc_cross_section_r, fill=False, edgecolor='cyan', ls='--', lw=1.5))
        
        mol_rad = molecule_diameter_nm / 2
        dists = np.abs(coords[:, 2] - z_nm)
        m_slice = dists < mol_rad
        if np.any(m_slice):
            c_map = {0: 'cyan', 1: 'yellow', 2: 'red', 3: 'magenta'}
            states = frame_df['state'].values[m_slice].astype(int)
            colors = [c_map.get(s, 'white') for s in states]
            data_width_nm = crop_max - crop_min
            fig_width_points = native_fig_size * 72
            points_per_nm = fig_width_points / data_width_nm
            marker_size_points = molecule_diameter_nm * points_per_nm
            marker_area = marker_size_points ** 2
            ax_gt.scatter(coords[m_slice, 0], coords[m_slice, 1], c=colors, s=marker_area, linewidths=0)

        if binding_sites_pos is not None:
            s_coords = binding_sites_pos * 1000
            s_slice = np.abs(s_coords[:, 2] - z_nm) < 50
            slice_indices = np.where(s_slice)[0]
            for idx in slice_indices:
                s = s_coords[idx]
                r = np.sqrt(max(0, 50**2 - (s[2]-z_nm)**2))
                is_synth = (is_synthetic is not None and is_synthetic[idx])
                if is_synth:
                    edge_color = (0.2, 1.0, 0.2)
                else:
                    if site_strengths is not None and idx < len(site_strengths):
                        strength = site_strengths[idx]
                        log_strength = np.log10(np.clip(strength, strength_range[0], strength_range[1]))
                        norm_strength = (log_strength - log_min) / (log_max - log_min) if log_max > log_min else 0.5
                    else:
                        norm_strength = 0.5
                    r_channel = 1.0 - 0.4 * norm_strength
                    g_channel = 0.7 * (1 - norm_strength)
                    b_channel = 0.7 * (1 - norm_strength)
                    edge_color = (r_channel, g_channel, b_channel)
                is_occupied = (site_occupied is not None and site_occupied[idx])
                if is_occupied:
                    ax_gt.add_patch(plt.Circle((s[0], s[1]), r, fill=True, facecolor=(*edge_color, 0.6), edgecolor=edge_color, lw=1.5))
                else:
                    ax_gt.add_patch(plt.Circle((s[0], s[1]), r, fill=False, edgecolor=edge_color, lw=1.5))

        ax_gt.set_xlim(crop_min, crop_max)
        ax_gt.set_ylim(crop_min, crop_max)
        ax_gt.set_aspect('equal')
        
        buf = io.BytesIO()
        plt.savefig(buf, dpi=image_dpi, facecolor='black', pad_inches=0, format='png')
        buf.seek(0)
        native_img = Image.open(buf).convert('RGB')
        plt.close()
        
        final_img = native_img.resize((target_final_pixels, target_final_pixels), Image.LANCZOS)
        
        if save_to_disk and output_dir:
            filename = f"{save_prefix}f{frame_idx}_z{i}_gt.png"
            gt_path = os.path.join(output_dir, filename)
            final_img.save(gt_path)
            paths_or_images.append(gt_path)
        else:
            paths_or_images.append(final_img)
        
    return paths_or_images


def render_convolved_slices_fast(grid_3d, nucleus_diameter_nm, voxel_size_nm,
                                  half_span, grid_size, slice_spacing_nm,
                                  brightness_vmax_list, upscale_factor=12,
                                  save_to_disk=False, output_dir=None, frame_idx=0, save_prefix="",
                                  bit_depth=8, au_calibration=None,
                                  noise_floor_frac=0.0, noise_std_frac=0.0,
                                  num_slices=None):
    """
    Fast convolved slice rendering without matplotlib.
    
    Produces normalized grayscale images at the specified bit depth.
    Upscaling is done on the numpy array via np.repeat for compatibility
    with both 8-bit and 16-bit modes.
    
    Parameters
    ----------
    bit_depth : int
        8 for uint8 (0-255) or 16 for uint16 (0-65535).
    au_calibration : dict or None
        If provided, renders in absolute AU units instead of vmax normalization.
        Expected keys:
            'ref_concentration' : float - Reference TF concentration (nM)
            'ref_signal'        : float - Signal above background at ref concentration (AU)
            'background'        : float - Background intensity (AU)
            'noise_std'         : float - Gaussian noise std dev (AU)
            'sim_concentration' : float - Current simulation TF concentration (nM)
    noise_floor_frac : float
        Mean of a Gaussian noise floor, as a fraction of full scale (vmax -> 1.0).
        Only applied in normalized (non-AU) mode. 0 disables it.
    noise_std_frac : float
        Standard deviation of the Gaussian noise floor, as a fraction of full
        scale. Only applied in normalized (non-AU) mode. 0 disables it.
    """
    NUCLEUS_RADIUS_NM = nucleus_diameter_nm / 2
    plot_limit = NUCLEUS_RADIUS_NM * 1.1
    
    voxel_centers_1d = np.arange(grid_size) * voxel_size_nm - half_span + voxel_size_nm / 2
    in_view = (voxel_centers_1d >= -plot_limit) & (voxel_centers_1d <= plot_limit)
    view_indices = np.where(in_view)[0]
    if len(view_indices) > 0:
        i_start, i_end = view_indices[0], view_indices[-1] + 1
    else:
        i_start, i_end = 0, grid_size
    
    if num_slices is not None:
        _n_half = int(num_slices) // 2
        _cand = sorted({i * slice_spacing_nm for i in range(-_n_half, _n_half + 1)})
        sorted_z_positions = [z for z in _cand if abs(z) <= NUCLEUS_RADIUS_NM]
    else:
        sorted_z_positions = sorted(np.unique(np.concatenate([
            np.arange(0, NUCLEUS_RADIUS_NM, slice_spacing_nm),
            -np.arange(0, NUCLEUS_RADIUS_NM, slice_spacing_nm)
        ])).tolist())
    
    if bit_depth == 16:
        max_val = 65535
        dtype = np.uint16
        pil_mode = 'I;16'
    else:
        max_val = 255
        dtype = np.uint8
        pil_mode = 'L'
    
    # --- AU calibration: compute scale factor from 3D grid ---
    au_scale_factor = None
    au_background = 0.0
    au_noise_std = 0.0
    if au_calibration is not None:
        # Compute nucleus mask on the 3D grid
        voxel_centers_3d = np.arange(grid_size) * voxel_size_nm - half_span + voxel_size_nm / 2
        x_nm, y_nm, z_nm_3d = np.meshgrid(voxel_centers_3d, voxel_centers_3d, voxel_centers_3d, indexing='ij')
        nucleus_mask_3d = (x_nm**2 + y_nm**2 + z_nm_3d**2) <= NUCLEUS_RADIUS_NM**2
        
        # Mean raw intensity within nucleus
        masked_vals = grid_3d[nucleus_mask_3d]
        mean_nucleus = np.mean(masked_vals) if len(masked_vals) > 0 else 1.0
        if mean_nucleus <= 0:
            mean_nucleus = 1e-10  # avoid division by zero
        
        # Expected signal above background for this concentration
        ref_conc = au_calibration['ref_concentration']
        ref_signal = au_calibration['ref_signal']
        sim_conc = au_calibration['sim_concentration']
        expected_signal = (sim_conc / ref_conc) * ref_signal if ref_conc > 0 else 0.0
        
        au_scale_factor = expected_signal / mean_nucleus
        au_background = au_calibration['background']
        au_noise_std = au_calibration['noise_std']

    # Pre-compute 2D coordinate grids for nucleus mask (used in AU mode)
    if au_calibration is not None:
        cropped_centers = voxel_centers_1d[i_start:i_end]
        mask_x, mask_y = np.meshgrid(cropped_centers, cropped_centers)  # after .T, rows=y, cols=x
        mask_xy_sq = mask_x**2 + mask_y**2  # x²+y² grid, invariant to flipud since squared
    
    result_dict = {}
    
    for z_idx, z_nm in enumerate(sorted_z_positions):
        grid_z_idx = np.clip(int(np.floor((z_nm + half_span) / voxel_size_nm)), 0, grid_size - 1)
        slice_full = grid_3d[:, :, grid_z_idx]
        slice_cropped = slice_full[i_start:i_end, i_start:i_end].T
        
        # Flip to match matplotlib origin='lower'
        slice_flipped = np.flipud(slice_cropped)
        
        result_dict[z_idx] = {}
        
        if au_calibration is not None:
            # --- AU calibration mode: absolute intensity, no vmax normalization ---
            # Compute 2D nucleus cross-section mask for this z-slice
            cross_section_r_sq = max(0, NUCLEUS_RADIUS_NM**2 - z_nm**2)
            nucleus_mask_2d = mask_xy_sq <= cross_section_r_sq
            
            scaled = np.zeros_like(slice_flipped)
            scaled[nucleus_mask_2d] = slice_flipped[nucleus_mask_2d] * au_scale_factor + au_background
            if au_noise_std > 0:
                noise = np.random.normal(0, au_noise_std, scaled.shape)
                scaled[nucleus_mask_2d] += noise[nucleus_mask_2d]
            # Pixels outside nucleus remain 0
            img_array = np.clip(scaled, 0, max_val).astype(dtype)
            
            # Upscale on numpy array
            if upscale_factor > 1:
                img_array = np.repeat(np.repeat(img_array, upscale_factor, axis=0), upscale_factor, axis=1)
            
            final_img = Image.fromarray(img_array, mode=pil_mode)
            
            if save_to_disk and output_dir:
                filename = f"{save_prefix}f{frame_idx}_z{z_idx}_au_conv.png"
                out_path = os.path.join(output_dir, filename)
                final_img.save(out_path)
                result_dict[z_idx][0] = out_path
            else:
                result_dict[z_idx][0] = final_img
        else:
            # --- Original vmax normalization mode ---
            for b_idx, vmax_val in enumerate(brightness_vmax_list):
                normalized = np.clip(slice_flipped / vmax_val, 0.0, 1.0)
                # Optional Gaussian noise floor (fraction of full scale)
                if noise_floor_frac > 0 or noise_std_frac > 0:
                    noise = np.random.normal(noise_floor_frac, noise_std_frac, normalized.shape)
                    normalized = np.clip(normalized + noise, 0.0, 1.0)
                img_array = (normalized * max_val).astype(dtype)
                
                # Upscale on numpy array (works for both 8-bit and 16-bit)
                if upscale_factor > 1:
                    img_array = np.repeat(np.repeat(img_array, upscale_factor, axis=0), upscale_factor, axis=1)
                
                final_img = Image.fromarray(img_array, mode=pil_mode)
                
                if save_to_disk and output_dir:
                    filename = f"{save_prefix}f{frame_idx}_z{z_idx}_b{b_idx}_conv.png"
                    out_path = os.path.join(output_dir, filename)
                    final_img.save(out_path)
                    result_dict[z_idx][b_idx] = out_path
                else:
                    result_dict[z_idx][b_idx] = final_img
    
    return result_dict


# --- 5. STREAMLIT UI ---

st.set_page_config(page_title="SPARK", layout="wide")

_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
_spark_logo = os.path.join(_assets_dir, 'Logo.png')
_lab_logo   = os.path.join(_assets_dir, 'LabLogo.png')

@st.cache_data
def _b64_img(path):
    import base64
    with open(path, 'rb') as _f:
        return base64.b64encode(_f.read()).decode()

_spark_img_tag = (f'<img src="data:image/png;base64,{_b64_img(_spark_logo)}" style="width:100%;display:block;" />'
                  if os.path.exists(_spark_logo) else '')
_lab_img_tag   = (f'<img src="data:image/png;base64,{_b64_img(_lab_logo)}" style="width:100%;display:block;" />'
                  if os.path.exists(_lab_logo) else '')

st.markdown(f"""
<div style="display:flex;align-items:flex-end;gap:1.5rem;padding-bottom:0.75rem;">
  <div style="flex:5;">
    <h1 style="font-size:4.5rem;font-weight:800;margin-bottom:0;line-height:1;">SPARK</h1>
    <p style="font-size:1.4rem;color:#555;margin-top:0;margin-bottom:0;">Simulation of Protein Accumulation from Reaction Kinetics</p>
  </div>
  <div style="flex:2;">{_spark_img_tag}</div>
  <div style="flex:1;">{_lab_img_tag}</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar styling: larger fonts + compact expander spacing ---
st.markdown("""
<style>
/* Base font size bump for everything in the sidebar */
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] *,
section[data-testid="stSidebar"] .stAlert,
section[data-testid="stSidebar"] .stAlert p,
section[data-testid="stSidebar"] button {
    font-size: 16px !important;
}

/* Captions slightly smaller than body but still larger than default */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] small {
    font-size: 14px !important;
}

/* Section headers inside the sidebar (h1–h4) */
section[data-testid="stSidebar"] h1 { font-size: 22px !important; }
section[data-testid="stSidebar"] h2 { font-size: 20px !important; }
section[data-testid="stSidebar"] h3 { font-size: 18px !important; }
section[data-testid="stSidebar"] h4 { font-size: 17px !important; }

/* Expander header (the section title you click to open/close) */
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
section[data-testid="stSidebar"] details summary,
section[data-testid="stSidebar"] details summary * {
    font-size: 30px !important;
    font-weight: 400 !important;
}

/* Reduce vertical gap between stacked expanders */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    margin-bottom: 6px !important;
}

/* Smaller molecule-state metrics in the main area */
[data-testid="stMetricValue"] { font-size: 14px !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'traj_df' not in st.session_state: st.session_state['traj_df'] = None
if 'viz_grids' not in st.session_state: st.session_state['viz_grids'] = {}
if 'gt_paths' not in st.session_state: st.session_state['gt_paths'] = {}
if 'gt_rendered' not in st.session_state: st.session_state['gt_rendered'] = True
if 'conv_paths' not in st.session_state: st.session_state['conv_paths'] = {}
if 'count_paths' not in st.session_state: st.session_state['count_paths'] = {}
if 'count_max' not in st.session_state: st.session_state['count_max'] = 0
if 'calib_max' not in st.session_state: st.session_state['calib_max'] = 1.0
if 'temp_viz_dir' not in st.session_state: st.session_state['temp_viz_dir'] = None
if 't_idx' not in st.session_state: st.session_state['t_idx'] = 0
if 'z_idx' not in st.session_state: st.session_state['z_idx'] = 0
if 'autoplay' not in st.session_state: st.session_state['autoplay'] = False
if 'autoplay_z' not in st.session_state: st.session_state['autoplay_z'] = False
if 'brightness_idx' not in st.session_state: st.session_state['brightness_idx'] = 4
if 'site_strengths' not in st.session_state: st.session_state['site_strengths'] = None
if 'strength_multipliers' not in st.session_state: st.session_state['strength_multipliers'] = STRENGTH_MULTIPLIERS_DEFAULT.copy()
if 'frap_bleach_frame' not in st.session_state: st.session_state['frap_bleach_frame'] = None
if 'frap_bleach_z' not in st.session_state: st.session_state['frap_bleach_z'] = 0
if 'frap_roi' not in st.session_state: st.session_state['frap_roi'] = None
if 'frap_bleached_ids' not in st.session_state: st.session_state['frap_bleached_ids'] = None
if 'frap_conv_paths' not in st.session_state: st.session_state['frap_conv_paths'] = {}
if 'frap_curve' not in st.session_state: st.session_state['frap_curve'] = None
if 'frap_temp_dir' not in st.session_state: st.session_state['frap_temp_dir'] = None
if 'frap_t_idx' not in st.session_state: st.session_state['frap_t_idx'] = 0
if 'frap_z_idx' not in st.session_state: st.session_state['frap_z_idx'] = 0
if 'frap_autoplay_t' not in st.session_state: st.session_state['frap_autoplay_t'] = False
if 'frap_autoplay_z' not in st.session_state: st.session_state['frap_autoplay_z'] = False
if 'frap_display_dt' not in st.session_state: st.session_state['frap_display_dt'] = None
if 'frap_import_r_px' not in st.session_state: st.session_state['frap_import_r_px'] = 20
if 'frap_saved_curves' not in st.session_state: st.session_state['frap_saved_curves'] = []
if 'frap_img_zip_bytes' not in st.session_state: st.session_state['frap_img_zip_bytes'] = None
if '_sim_params_for_pkl' not in st.session_state: st.session_state['_sim_params_for_pkl'] = None
if 'sim_num_molecules' not in st.session_state: st.session_state['sim_num_molecules'] = 0
if 'exposure_pkl_bytes' not in st.session_state: st.session_state['exposure_pkl_bytes'] = None

# Slice spacing <-> slice count synced inputs
if 'slice_spacing_nm' not in st.session_state:
    st.session_state['slice_spacing_nm'] = SIDEBAR_DEFAULTS['slice_spacing_nm']
if 'num_slices' not in st.session_state:
    st.session_state['num_slices'] = SIDEBAR_DEFAULTS['num_slices']

# Concentration <-> molecule-count synced inputs (General Parameters)
if 'gp_nucleus_diam' not in st.session_state:
    st.session_state['gp_nucleus_diam'] = SIDEBAR_DEFAULTS['nucleus_diameter_nm']
if 'gp_conc' not in st.session_state:
    st.session_state['gp_conc'] = SIDEBAR_DEFAULTS['tf_concentration_nM']
if 'gp_num_mols' not in st.session_state:
    st.session_state['gp_num_mols'] = conc_to_molecule_count(
        st.session_state['gp_conc'], st.session_state['gp_nucleus_diam'])
if '_preset_feedback' not in st.session_state:
    st.session_state['_preset_feedback'] = ''

def _sync_count_from_conc():
    st.session_state['gp_num_mols'] = conc_to_molecule_count(
        st.session_state['gp_conc'], st.session_state['gp_nucleus_diam'])

def _sync_conc_from_count():
    st.session_state['gp_conc'] = molecule_count_to_conc(
        st.session_state['gp_num_mols'], st.session_state['gp_nucleus_diam'])

def _sync_on_diam_change():
    # Treat concentration as authoritative when the nucleus volume changes.
    st.session_state['gp_num_mols'] = conc_to_molecule_count(
        st.session_state['gp_conc'], st.session_state['gp_nucleus_diam'])

def _sync_slices_from_spacing():
    if st.session_state.get('custom_slice_override', False):
        return
    _nuc_r = float(st.session_state.get('gp_nucleus_diam', SIDEBAR_DEFAULTS['nucleus_diameter_nm'])) / 2
    _sp = float(st.session_state.get('slice_spacing_nm', SIDEBAR_DEFAULTS['slice_spacing_nm']))
    if _sp > 0:
        st.session_state['num_slices'] = 2 * max(1, int(_nuc_r / _sp)) + 1

def _sync_spacing_from_slices():
    if st.session_state.get('custom_slice_override', False):
        return
    _nuc_r = float(st.session_state.get('gp_nucleus_diam', SIDEBAR_DEFAULTS['nucleus_diameter_nm'])) / 2
    _ns = int(st.session_state.get('num_slices', SIDEBAR_DEFAULTS['num_slices']))
    _n_half = _ns // 2
    if _n_half > 0:
        st.session_state['slice_spacing_nm'] = _nuc_r / _n_half


tab_interactive, tab_batch = st.tabs(["Interactive Simulation", "Batch Processing (CSV)"])

# ==========================================
# TAB 1: INTERACTIVE MODE
# ==========================================
with tab_interactive:
    # --- PRESET PANEL ---
    with st.sidebar:
        st.markdown("### Parameter Presets")
        _presets_df = _load_presets_df()
        _preset_names = _presets_df['label'].tolist() if len(_presets_df) > 0 else []
        if _preset_names:
            _sel = st.selectbox("Select preset", _preset_names, key='_selected_preset',
                                label_visibility='collapsed')
            _pcol1, _pcol2 = st.columns(2)
            if _pcol1.button("Load", key='_load_preset_btn', use_container_width=True):
                _row = _presets_df[_presets_df['label'] == _sel].iloc[0]
                _apply_preset_row(_row)
                st.session_state['_preset_feedback'] = f"Loaded \"{_sel}\""
                st.session_state['_rerun_pending'] = True
            with _pcol2.popover("Save as...", use_container_width=True):
                _save_name = st.text_input("Preset name", key='_preset_save_name',
                                           placeholder="My preset")
                if st.button("Save", key='_save_preset_btn'):
                    _name = (_save_name or '').strip()
                    if _name:
                        _new_row = _capture_current_as_preset_row(_name)
                        _df2 = _load_presets_df()
                        _df2 = _df2[_df2['label'] != _name]
                        _df2 = pd.concat([_df2, pd.DataFrame([_new_row])], ignore_index=True)
                        try:
                            _df2.to_csv(PRESETS_CSV_PATH, index=False)
                            st.session_state['_preset_feedback'] = f"Saved \"{_name}\""
                        except Exception as _e:
                            st.session_state['_preset_feedback'] = f"Save error: {_e}"
                        safe_rerun()
                    else:
                        st.warning("Enter a name first.")
        else:
            st.caption("No presets found — save one to get started.")
            with st.popover("Save as...", use_container_width=True):
                _save_name = st.text_input("Preset name", key='_preset_save_name',
                                           placeholder="My preset")
                if st.button("Save", key='_save_preset_btn'):
                    _name = (_save_name or '').strip()
                    if _name:
                        _new_row = _capture_current_as_preset_row(_name)
                        _df2 = pd.DataFrame([_new_row])
                        try:
                            _df2.to_csv(PRESETS_CSV_PATH, index=False)
                            st.session_state['_preset_feedback'] = f"Saved \"{_name}\""
                        except Exception as _e:
                            st.session_state['_preset_feedback'] = f"Save error: {_e}"
                        safe_rerun()
                    else:
                        st.warning("Enter a name first.")
        if st.session_state.get('_preset_feedback'):
            st.success(st.session_state.pop('_preset_feedback'))
        st.divider()

    with st.sidebar.expander("1. General Parameters", expanded=False):
        nucleus_diameter_nm = st.number_input(
            "Nucleus Diameter (nm)", key='gp_nucleus_diam',
            on_change=_sync_on_diam_change)

        conc_col, mol_col = st.columns(2)
        tf_concentration_nM = conc_col.number_input(
            "TF Concentration (nM)", key='gp_conc', min_value=0.0, format="%.4f",
            on_change=_sync_count_from_conc,
            help="Transcription Factor concentration (nM) in nucleus. "
                 "Edit this or the molecule count — the other updates automatically.")
        num_molecules_input = mol_col.number_input(
            "Number of Molecules", key='gp_num_mols', min_value=0, step=1,
            on_change=_sync_conc_from_count,
            help="Total TF molecules in the nucleus. "
                 "Edit this or the concentration — the other updates automatically.")
        st.caption(f"≈ {num_molecules_input} molecules in a {nucleus_diameter_nm:.0f} nm nucleus "
                   f"({_nucleus_volume_um3(nucleus_diameter_nm):.3f} µm³)")

        molecule_diameter_nm = st.number_input("Molecule Diameter (nm)", value=SIDEBAR_DEFAULTS['molecule_diameter_nm'], key='molecule_diameter_nm')

        st.caption("Time Settings")
        num_frames = st.number_input("Number of Display Frames", value=SIDEBAR_DEFAULTS['num_frames'], step=10, key='num_frames')
        sim_dt = st.number_input("Physics Time Step (s)", value=SIDEBAR_DEFAULTS['sim_dt'], format="%.4f", key='sim_dt', help="Time step at which molecule positions are calculated.")
        display_dt = st.number_input("Display Interval (s)", value=SIDEBAR_DEFAULTS['display_dt'], format="%.2f", key='display_dt', help="Time interval between display frames.")

        st.caption("Reproducibility")
        use_seed = st.checkbox(
            "Use random seed (reproducible)",
            value=False,
            help="When checked, the simulation uses the seed below. When unchecked, the "
                 "simulation uses fresh OS entropy each run (default behavior)."
        )
        if use_seed:
            sim_seed_value = st.number_input(
                "Seed (integer)",
                value=42, min_value=0, max_value=2**31 - 1, step=1,
                help="Any non-negative integer. Used as the root seed."
            )
            seed_lock_mode = st.radio(
                "What to lock with this seed",
                options=["Lock everything (sites + motion)",
                         "Lock sites only (motion uses fresh entropy)"],
                index=0,
                help="**Lock everything** — same seed + same parameters ⇒ byte-identical "
                     "molecule trajectories, sites, and AU noise. Equivalent to a `sim_group` "
                     "row in the HPC batch script.\n\n"
                     "**Lock sites only** — same seed ⇒ same binding-site layout, strengths, "
                     "and initial per-cluster diffusion coefficients. Molecule motion is "
                     "re-randomized each run from OS entropy. Equivalent to a `site_group` "
                     "(without `sim_group`) row in the HPC batch script. Useful when you want "
                     "to vary TF concentration / kinetics / diffusion and see how motion "
                     "rearranges itself on a fixed binding-site landscape."
            )
            motion_seed_value = ('fresh'
                                 if seed_lock_mode.startswith("Lock sites only")
                                 else None)
        else:
            sim_seed_value = None
            motion_seed_value = None

    with st.sidebar.expander("2. Diffusion", expanded=False):
        three_state_mode = st.checkbox(
            "3-State Model (Free + Inter + NS ↔ Bound)",
            value=not SIDEBAR_DEFAULTS['two_state_mode'],
            key='three_state_mode',
            help="When enabled, the Intermediate state (state 1) is added: "
                 "Free ↔ Intermediate ↔ Specifically Bound, plus NS-Bound. "
                 "When disabled (default), only Free (state 0), NS-Bound (state 3), and "
                 "Specifically Bound (state 2) exist — after unbinding, molecules return "
                 "to Free or NS-bound (never Intermediate)."
        )
        two_state_mode = not three_state_mode

        if two_state_mode:
            st.info("**2-State Mode:** Free ↔ Specific Bound + NS-bound. No intermediate state.")
        else:
            st.info("**3-State Mode:** Free ↔ Intermediate ↔ Specific Bound + NS-bound.")

        st.markdown("#### Initial Populations (%)")
        if two_state_mode:
            c1, c2 = st.columns(2)
            pct_free_init = c1.number_input("Free", 0.0, 100.0, 60.0, key='pct_free_init')
            pct_ns_init = c2.number_input("NS Bound", 0.0, 100.0, 40.0, key='pct_ns_init')
            pct_inter_init = 0.0
        else:
            c1, c2, c3 = st.columns(3)
            pct_free_init = c1.number_input("Free", 0.0, 100.0, SIDEBAR_DEFAULTS['pct_free_init'], key='pct_free_init')
            pct_inter_init = c2.number_input("Inter", 0.0, 100.0, SIDEBAR_DEFAULTS['pct_inter_init'], key='pct_inter_init')
            pct_ns_init = c3.number_input("NS Bound", 0.0, 100.0, SIDEBAR_DEFAULTS['pct_ns_init'], key='pct_ns_init')

        st.markdown("#### Diffusion Coefficients (µm²/s)")
        c_d1, c_d2 = st.columns(2)
        d_free = c_d1.number_input("DC Free", value=SIDEBAR_DEFAULTS['d_free'], format="%.2f", key='d_free')
        d_free_std = c_d2.number_input("DC Free σ", value=SIDEBAR_DEFAULTS['d_free_std'], format="%.2f", key='d_free_std')
        if not two_state_mode:
            c_d3, c_d4 = st.columns(2)
            d_inter = c_d3.number_input("DC Inter", value=SIDEBAR_DEFAULTS['d_inter'], format="%.4f", key='d_inter')
            d_inter_std = c_d4.number_input("DC Inter σ", value=SIDEBAR_DEFAULTS['d_inter_std'], format="%.4f", key='d_inter_std')
        else:
            d_inter = SIDEBAR_DEFAULTS['d_inter']
            d_inter_std = SIDEBAR_DEFAULTS['d_inter_std']
        c_d5, c_d6 = st.columns(2)
        d_ns = c_d5.number_input("DC NS Bound", value=SIDEBAR_DEFAULTS['d_ns'], format="%.4f", key='d_ns')
        d_ns_std = c_d6.number_input("DC NS Bound σ", value=SIDEBAR_DEFAULTS['d_ns_std'], format="%.4f", key='d_ns_std')

        if not two_state_mode:
            st.caption("State Switching Rates (Hz)")
            c1, c2 = st.columns(2)
            k_fi = c1.number_input("Free -> Inter", value=SIDEBAR_DEFAULTS['k_fi'], key='k_fi')
            k_if = c2.number_input("Inter -> Free", value=SIDEBAR_DEFAULTS['k_if'], key='k_if')
            k_fns = c1.number_input("Free -> NS Bound", value=SIDEBAR_DEFAULTS['k_fns'], key='k_fns')
            k_ins = c2.number_input("Inter -> NS Bound", value=SIDEBAR_DEFAULTS['k_ins'], key='k_ins')
            ns_res = st.number_input("NS Bound Residence Time (s)", value=SIDEBAR_DEFAULTS['ns_res'], key='ns_res')
        else:
            k_fi = 0.0
            k_if = 0.0
            k_ins = 0.0
            st.caption("State Switching Rates (Hz)")
            k_fns = st.number_input("Free -> NS Bound", value=SIDEBAR_DEFAULTS['k_fns'], key='k_fns')
            ns_res = st.number_input("NS Bound Residence Time (s)", value=SIDEBAR_DEFAULTS['ns_res'], key='ns_res')

        st.markdown("#### Specific Binding Kinetics")
        k_on_prob = st.number_input("Binding Prob (k_on)", 0.0, 1.0, SIDEBAR_DEFAULTS['k_on_prob'], 0.1, key='k_on_prob', help="Probability of binding when a molecule is within the capture radius of an unoccupied site")
        res_time_mean = st.number_input("Base Residence Time Mean (s)", value=SIDEBAR_DEFAULTS['res_time_mean'], key='res_time_mean', help="Base residence time - will be multiplied by site strength")
        res_time_std = st.number_input("Base Residence Time Std (s)", value=SIDEBAR_DEFAULTS['res_time_std'], key='res_time_std', help="Base std dev - will be multiplied by site strength")
        capture_rad_nm = st.number_input("Capture Rad (nm)", value=SIDEBAR_DEFAULTS['capture_rad_nm'], key='capture_rad_nm', help="Radius within which a molecule can bind to an unoccupied site")

        st.markdown("#### Cooperativity / Cooperative Binding")
        enable_coop = st.checkbox("Enable Cooperative Binding", value=SIDEBAR_DEFAULTS['enable_cooperativity'], key='enable_cooperativity', help="If checked, occupied sites in a cluster increase the binding probability of remaining sites in that cluster.")
        c_coop1, c_coop2 = st.columns(2)
        coop_strength = c_coop1.number_input("Coop Strength", value=SIDEBAR_DEFAULTS['coop_strength'], min_value=0.0, step=0.1, key='coop_strength', disabled=not enable_coop)
        coop_hill = c_coop2.number_input("Hill Coeff (n)", value=SIDEBAR_DEFAULTS['coop_hill'], min_value=0.1, step=0.1, key='coop_hill', disabled=not enable_coop)
        if enable_coop:
            st.info(f"Formula: $P_{{bind}}' = P_{{bind}} \\times (1 + ({coop_strength} \\times N_{{occupied}})^{{{coop_hill}}})$")

    with st.sidebar.expander("3. Binding Sites", expanded=False):
        num_binding_sites = st.number_input("Number of Binding Sites", value=SIDEBAR_DEFAULTS['num_binding_sites'], step=100, key='num_binding_sites')
        
        st.caption("Binding Site Clustering")
        st.caption("Cluster sizes (sites per cluster):")
        size_cols = st.columns(5)
        cluster_sizes = []
        for i, col in enumerate(size_cols):
            size = col.number_input(f"S{i+1}", value=i+1, min_value=1, max_value=1000, step=1, key=f"cluster_size_{i}", label_visibility="collapsed")
            cluster_sizes.append(int(size))
        
        st.caption("% of sites in each cluster size:")
        pct_cols = st.columns(5)
        cluster_percentages = []
        for i, col in enumerate(pct_cols):
            pct = col.number_input(f"P{i+1}", value=20.0, min_value=0.0, max_value=100.0, step=5.0, key=f"cluster_pct_{i}", label_visibility="collapsed")
            cluster_percentages.append(pct)
        
        cluster_dist = {cluster_sizes[i]: cluster_percentages[i] for i in range(5)}
        total_cluster_pct = sum(cluster_percentages)
        if abs(total_cluster_pct - 100.0) > 0.1:
            st.warning(f"⚠️ Total: {total_cluster_pct:.1f}% (should be 100%)")
        else:
            st.success(f"✓ Total: {total_cluster_pct:.1f}%")

        cluster_radius_nm = st.number_input("Binding Site Cluster radius (nm)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0, key="cluster_radius_nm", help="Radius within which binding sites in the same cluster are distributed")

        st.caption("Binding Site Diffusion")
        c_bs1, c_bs2 = st.columns(2)
        bs_diff_mean = c_bs1.number_input("Binding Site DC (µm²/s)", value=SIDEBAR_DEFAULTS['bs_diff_mean'], format="%.4f", key='bs_diff_mean', help="Diffusion Coefficient for each cluster of binding sites.")
        bs_diff_std = c_bs2.number_input("Binding Site DC σ", value=SIDEBAR_DEFAULTS['bs_diff_std'], format="%.4f", key='bs_diff_std')
        c_bs3, c_bs4 = st.columns(2)
        bs_jiggle_std = c_bs3.number_input("Binding Site Jiggle σ (µm)", value=SIDEBAR_DEFAULTS['bs_jiggle_std'], format="%.4f", key='bs_jiggle_std', help="Random local fluctuation added to each binding site.")
        bs_hurst = c_bs4.number_input("H (Hurst exponent)", value=SIDEBAR_DEFAULTS['bs_hurst'], min_value=0.01, max_value=0.99, format="%.2f", key='bs_hurst', help="Hurst exponent for fractional Brownian motion.")
        st.caption("H=0.5: Brownian | H=0.25: Rouse | H<0.5: sub-diffusive | H>0.5: super-diffusive")

        st.markdown("#### Binding Site Strength Distribution")
        st.caption("Strength multiplier affects residence time at each site")
        
        def on_preset_change():
            preset = st.session_state['strength_preset']
            if preset != "Custom" and preset in STRENGTH_PRESETS:
                for i, val in enumerate(STRENGTH_PRESETS[preset]):
                    st.session_state[f'str_pct_{i}'] = float(val)
        
        strength_preset = st.selectbox("Preset", ["Custom"] + list(STRENGTH_PRESETS.keys()), index=1, key="strength_preset", on_change=on_preset_change)
        
        st.caption("Strength Multipliers (residence time ×)")
        mult_cols = st.columns(5)
        strength_multipliers = []
        for i, col in enumerate(mult_cols):
            mult = col.number_input(f"M{i+1}", value=STRENGTH_MULTIPLIERS_DEFAULT[i], format="%.2f", key=f"mult_{i}", label_visibility="collapsed", min_value=0.01)
            strength_multipliers.append(mult)
        
        st.caption("% of Sites in Each Bin")
        mult_label_cols = st.columns(5)
        for i, col in enumerate(mult_label_cols):
            col.caption(f"{strength_multipliers[i]}×")

        for i in range(5):
            if f'str_pct_{i}' not in st.session_state:
                st.session_state[f'str_pct_{i}'] = 20.0
        
        pct_cols = st.columns(5)
        strength_percentages = []
        for i, col in enumerate(pct_cols):
            pct = col.number_input(f"P{i+1}", min_value=0.0, max_value=100.0, step=5.0, key=f"str_pct_{i}", label_visibility="collapsed")
            strength_percentages.append(pct)
        
        cluster_correlated_strength = st.checkbox("Sites in same cluster share strength", value=False, key='cluster_correlated_strength', help="When enabled, all binding sites in a cluster inherit the same strength")
        
        if sum(strength_percentages) > 0:
            fig_preview, ax_preview = plt.subplots(figsize=(4, 1.5))
            bar_colors = ['#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#cc0000']
            ax_preview.bar([f"{m}×" for m in strength_multipliers], strength_percentages, color=bar_colors)
            ax_preview.set_ylabel("% Sites", fontsize=8)
            ax_preview.tick_params(axis='both', labelsize=7)
            ax_preview.set_ylim(0, max(strength_percentages) * 1.2 if max(strength_percentages) > 0 else 100)
            ax_preview.set_facecolor('#0e1117')
            fig_preview.patch.set_facecolor('#0e1117')
            ax_preview.tick_params(colors='white')
            ax_preview.yaxis.label.set_color('white')
            ax_preview.spines['bottom'].set_color('white')
            ax_preview.spines['left'].set_color('white')
            ax_preview.spines['top'].set_visible(False)
            ax_preview.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_preview)
            plt.close()

        st.markdown("#### Synthetic Binding Site Array")
        synth_col1, synth_col2 = st.columns(2)
        num_synthetic_sites = synth_col1.number_input("# Synthetic Sites", value=SIDEBAR_DEFAULTS['num_synthetic_sites'], min_value=0, step=10, key='num_synthetic_sites', help="Number of synthetic binding sites that move as a single cluster (colored green)")
        synthetic_strength = synth_col2.number_input("Synth Strength", value=SIDEBAR_DEFAULTS['synthetic_strength'], min_value=0.01, format="%.2f", key='synthetic_strength', help="Residence time multiplier for synthetic sites")
        synthetic_cluster_radius_nm = st.number_input("Synth Cluster Radius (nm)", value=SIDEBAR_DEFAULTS['synthetic_cluster_radius_nm'], min_value=10.0, key='synthetic_cluster_radius_nm', help="Spatial extent of synthetic cluster")

    st.session_state['strength_multipliers'] = strength_multipliers

    with st.sidebar.expander("4. Visualization", expanded=False):
        st.caption("Point Spread Function (PSF)")
        psf_col1, psf_col2 = st.columns(2)
        lateral_fwhm_nm = psf_col1.number_input("Lateral FWHM (nm)", value=SIDEBAR_DEFAULTS['lateral_fwhm_nm'], min_value=1.0, format="%.1f", key='lateral_fwhm_nm')
        axial_fwhm_nm = psf_col2.number_input("Axial FWHM (nm)", value=SIDEBAR_DEFAULTS['axial_fwhm_nm'], min_value=1.0, format="%.1f", key='axial_fwhm_nm')

        st.caption("Imaging Parameters")
        voxel_size_nm = st.number_input("XY Pixel Size (nm)", value=SIDEBAR_DEFAULTS['voxel_size_nm'], key='voxel_size_nm')
        image_dpi = st.number_input("Image DPI", value=SIDEBAR_DEFAULTS['image_dpi'], key='image_dpi')
        upscale_factor = st.number_input("Upscale Factor", value=SIDEBAR_DEFAULTS['upscale_factor'], min_value=1, max_value=20, step=1, key='upscale_factor')
        _custom_slicing = st.checkbox("Set spacing and count independently", key='custom_slice_override', value=False)
        _sl_col, _ns_col = st.columns(2)
        slice_spacing_nm = _sl_col.number_input(
            "Z Spacing (nm)", min_value=1.0,
            key='slice_spacing_nm', on_change=_sync_slices_from_spacing)
        num_slices = _ns_col.number_input(
            "Num Slices", min_value=1, step=2,
            key='num_slices', on_change=_sync_spacing_from_slices)
        st.caption(f"{int(num_slices)} slices · {slice_spacing_nm:.0f} nm spacing")

        st.caption("Camera Exposure")
        exposure_time_ms = st.number_input("Exposure Time (ms)", value=SIDEBAR_DEFAULTS['exposure_time_ms'], min_value=1.0, key='exposure_time_ms')
        exposure_time_s = exposure_time_ms / 1000.0
        physics_steps_per_exposure = max(1, int(exposure_time_s / sim_dt))
        st.caption(f"≈ {physics_steps_per_exposure} physical steps per exposure")

        st.caption("Display Noise Floor (normalized mode only)")
        enable_display_noise = st.checkbox(
            "Add noise floor",
            value=SIDEBAR_DEFAULTS['display_noise_enabled'],
            key='display_noise_enabled',
            help="Adds a Gaussian noise floor to the convolved display/export when AU "
                 "calibration is OFF. Levels are given as a percentage of vmax "
                 "(the global max intensity used for normalization). Ignored in AU mode, "
                 "which has its own background + noise model."
        )
        if enable_display_noise:
            noise_col1, noise_col2 = st.columns(2)
            noise_floor_pct = noise_col1.number_input(
                "Noise floor (% of vmax)", value=SIDEBAR_DEFAULTS['display_noise_floor_pct'],
                min_value=0.0, max_value=100.0, step=0.5, format="%.2f",
                key='display_noise_floor_pct',
                help="Mean background level added to every pixel.")
            noise_std_pct = noise_col2.number_input(
                "Noise std (% of vmax)", value=SIDEBAR_DEFAULTS['display_noise_std_pct'],
                min_value=0.0, max_value=100.0, step=0.5, format="%.2f",
                key='display_noise_std_pct',
                help="Standard deviation of the Gaussian noise.")
        else:
            noise_floor_pct = 0.0
            noise_std_pct = 0.0
        display_noise_floor_frac = noise_floor_pct / 100.0
        display_noise_std_frac = noise_std_pct / 100.0

    with st.sidebar.expander("5. AU Intensity Calibration (Optional)", expanded=False):
        st.caption("Calibrate convolved images to absolute Arbitrary Units using experimental reference data")
        au_calibration_enabled = st.checkbox(
            "Enable AU Calibration",
            value=SIDEBAR_DEFAULTS['au_calibration_enabled'],
            key='au_calibration_enabled',
            help="When enabled, convolved images are rendered in absolute AU instead of normalized intensity. "
                 "Provide a reference concentration and its measured signal above background."
        )
        if au_calibration_enabled:
            st.info(
                "**AU mode:** Pixel values = absolute AU.\n\n"
                "**Formula:** `pixel = raw × (expected_signal / mean_raw) + background + noise`\n\n"
                "Where `expected_signal = (sim_conc / ref_conc) × ref_signal`"
            )
            au_col1, au_col2 = st.columns(2)
            au_ref_concentration = au_col1.number_input("Ref Concentration (nM)", value=SIDEBAR_DEFAULTS['au_ref_concentration_nM'], min_value=0.01, format="%.1f", key='au_ref_concentration_nM', help="TF concentration at which the reference signal was measured")
            au_ref_signal = au_col2.number_input("Ref Signal (AU above bg)", value=SIDEBAR_DEFAULTS['au_ref_signal'], min_value=0.01, format="%.1f", key='au_ref_signal', help="Signal above background at the reference concentration")
            au_col3, au_col4 = st.columns(2)
            au_background = au_col3.number_input("Background (AU)", value=SIDEBAR_DEFAULTS['au_background'], min_value=0.0, format="%.1f", key='au_background', help="Constant background intensity (e.g. autofluorescence)")
            au_noise_std = au_col4.number_input("Noise Std (AU)", value=SIDEBAR_DEFAULTS['au_noise_std'], min_value=0.0, format="%.1f", key='au_noise_std', help="Gaussian noise standard deviation per pixel")
            
            expected_signal_preview = (float(tf_concentration_nM) / au_ref_concentration) * au_ref_signal if au_ref_concentration > 0 else 0
            st.caption(f"At {tf_concentration_nM} nM → expected signal: {expected_signal_preview:.1f} AU above background → mean pixel ≈ {expected_signal_preview + au_background:.1f} AU")
        else:
            au_ref_concentration = SIDEBAR_DEFAULTS['au_ref_concentration_nM']
            au_ref_signal = SIDEBAR_DEFAULTS['au_ref_signal']
            au_background = SIDEBAR_DEFAULTS['au_background']
            au_noise_std = SIDEBAR_DEFAULTS['au_noise_std']


    # --- RUN OPTIONS ---
    render_gt_display = st.checkbox(
        "Render ground truth display",
        value=True,
        help="Uncheck to skip the slow matplotlib ground truth render. GT can still be exported as TIFF."
    )

    render_count_maps = st.checkbox(
        "Render Molecule Count Maps",
        value=False,
        help="Generate per-voxel molecule count grids (no PSF). "
             "Same geometry as convolved images — overlay in FIJI to see how many molecules are in each bright spot. "
             "Exports as 16-bit TIFF; preview uses Turbo colormap."
    )
    count_mode = "Full exposure"
    count_z_proj = "slab"
    if render_count_maps:
        count_mode = st.radio(
            "Count mode",
            ["Full exposure", "Snapshot (last step)", "Centroid (mean position)"],
            index=0,
            horizontal=True,
            help="**Full exposure:** count every molecule at every physics sub-step during the exposure "
                 "(a molecule present for all N steps is counted N times). "
                 "**Snapshot:** count each molecule exactly once using only the last physics step of the exposure window. "
                 "**Centroid:** count each molecule exactly once at its mean position averaged over the exposure window — "
                 "matches the positions in positions.txt."
        )
        count_z_proj = st.radio(
            "Z-slice projection",
            ["Z-slab (±half slice spacing)", "Single voxel"],
            index=0,
            horizontal=True,
            help="**Z-slab:** sum all voxels within ±(slice spacing / 2) of each z-plane. "
                 "Every molecule falls into exactly one slab — no gaps, no double-counting. "
                 "**Single voxel:** count only the single voxel-thick layer at the exact z-plane."
        )
        count_z_proj = "slab" if "slab" in count_z_proj.lower() else "single"

    # --- LOAD FROM HPC ---
    with st.expander("Load HPC result (exposure_data.pkl)"):
        st.caption("Upload the `exposure_data.pkl` produced by `--save-exposure` "
                   "to visualise and FRAP a batch run without re-simulating.")
        _hpc_file = st.file_uploader("exposure_data.pkl", type=["pkl"],
                                     key="hpc_pkl_upload", label_visibility='collapsed')
        if _hpc_file and st.button("Load", key="hpc_load_btn", type="primary"):
            import pickle as _pkl
            _hpc = _pkl.load(_hpc_file)
            _hp  = _hpc['params']
            _hpc_exp_data = _hpc['exposure_data']
            _hpc_mol_data, _hpc_site_data = compute_exposure_frames(
                _hpc_exp_data, _hpc.get('site_strengths'))
            st.session_state['exposure_data']      = _hpc_exp_data
            st.session_state['site_strengths']     = _hpc.get('site_strengths')
            st.session_state['is_synthetic']       = _hpc.get('is_synthetic')
            st.session_state['exposure_mol_data']  = _hpc_mol_data
            st.session_state['exposure_site_data'] = _hpc_site_data
            st.session_state['steps_per_exposure'] = 1
            # Store params so rendering + FRAP use the correct PSF/geometry
            st.session_state['_hpc_lateral_fwhm_nm'] = _hp['lateral_fwhm_nm']
            st.session_state['_hpc_axial_fwhm_nm']  = _hp['axial_fwhm_nm']
            st.session_state['voxel_size_nm_hpc']  = _hp['voxel_size_nm']
            st.session_state['nucleus_diameter_nm_hpc'] = _hp['nucleus_diameter_nm']
            st.session_state['slice_spacing_nm_hpc']    = _hp['slice_spacing_nm']
            st.session_state['num_slices_hpc']          = _hp.get('num_slices')
            st.session_state['_sim_num_slices']         = _hp.get('num_slices')
            st.session_state['upscale_factor_hpc']      = _hp['upscale_factor']
            st.session_state['_hpc_display_dt']     = _hp['display_dt']
            st.session_state['_sim_params_for_pkl'] = _hp
            st.session_state['sim_num_molecules']   = _hpc.get('num_molecules', 0)
            st.session_state['exposure_pkl_bytes']  = None
            # Clear stale viz + FRAP state
            st.session_state['viz_grids'] = {}
            st.session_state['grid_info'] = {}
            if st.session_state.get('frap_temp_dir') and os.path.exists(st.session_state['frap_temp_dir']):
                shutil.rmtree(st.session_state['frap_temp_dir'], ignore_errors=True)
            st.session_state['frap_conv_paths'] = {}
            st.session_state['frap_curve']      = None
            st.session_state['frap_roi']        = None
            st.session_state['frap_bleached_ids'] = None
            st.session_state['frap_temp_dir']   = None
            st.session_state['frap_t_idx'] = 0
            st.session_state['frap_z_idx'] = 0
            st.session_state['frap_saved_curves'] = []
            st.session_state['last_run_info'] = [("success",
                f"Loaded HPC result: {len(_hpc_mol_data)} frames, "
                f"{_hpc['num_molecules']} molecules.")]
            st.session_state['_needs_render'] = True
            st.session_state['_rerun_pending'] = True

    # --- RUN SIMULATION (runs, renders, and displays automatically) ---
    _run_sim_clicked = st.button("Run Simulation", type="primary")
    _needs_render_flag = st.session_state.pop('_needs_render', False)

    if _run_sim_clicked:
        with st.spinner("Simulating..."):
            exposure_data, n_mols, site_strengths, is_synthetic, steps_per_exposure = run_full_simulation(
                int(num_frames), int(num_binding_sites),
                float(nucleus_diameter_nm), float(tf_concentration_nM),
                float(sim_dt), float(display_dt),
                float(pct_free_init), float(pct_inter_init), float(pct_ns_init),
                float(d_inter), float(d_free), float(d_ns),
                float(d_free_std), float(d_inter_std), float(d_ns_std),
                float(bs_diff_mean), float(bs_diff_std), float(bs_jiggle_std), float(bs_hurst),
                float(k_fi), float(k_if), float(k_fns), float(k_ins), float(ns_res),
                cluster_dist, res_time_mean, res_time_std, capture_rad_nm, k_on_prob,
                strength_multipliers, strength_percentages, cluster_correlated_strength,
                num_synthetic_sites=int(num_synthetic_sites),
                synthetic_strength=float(synthetic_strength),
                synthetic_cluster_radius_nm=float(synthetic_cluster_radius_nm),
                cluster_radius_nm=float(cluster_radius_nm),
                exposure_time_s=float(exposure_time_s),
                enable_cooperativity=enable_coop,
                coop_strength=float(coop_strength),
                coop_hill=float(coop_hill),
                two_state_mode=two_state_mode,
                show_progress=True,
                sim_seed=sim_seed_value,
                motion_seed=motion_seed_value,
                num_molecules_override=int(num_molecules_input),
            )
            exposure_mol_data, exposure_site_data = compute_exposure_frames(exposure_data, site_strengths)

            st.session_state['exposure_data'] = exposure_data
            st.session_state['site_strengths'] = site_strengths
            st.session_state['is_synthetic'] = is_synthetic
            st.session_state['exposure_mol_data'] = exposure_mol_data
            st.session_state['exposure_site_data'] = exposure_site_data
            st.session_state['steps_per_exposure'] = steps_per_exposure
            st.session_state['sim_num_molecules'] = n_mols
            st.session_state['_sim_display_dt'] = float(display_dt)
            st.session_state['_sim_slice_spacing_nm'] = float(slice_spacing_nm)
            st.session_state['_sim_num_slices'] = int(num_slices)
            st.session_state['_sim_params_for_pkl'] = {
                'nucleus_diameter_nm': float(nucleus_diameter_nm),
                'voxel_size_nm':       float(voxel_size_nm),
                'lateral_fwhm_nm':     float(lateral_fwhm_nm),
                'axial_fwhm_nm':       float(axial_fwhm_nm),
                'slice_spacing_nm':    float(slice_spacing_nm),
                'num_slices':          int(num_slices),
                'upscale_factor':      int(upscale_factor),
                'display_dt':          float(display_dt),
                'sim_dt':              float(sim_dt),
                'num_frames':          int(num_frames),
                'exposure_time_ms':    float(exposure_time_ms),
            }
            st.session_state['exposure_pkl_bytes'] = None
            st.session_state['viz_grids'] = {}
            # Clear any stale FRAP results from a previous run
            if st.session_state.get('frap_temp_dir') and os.path.exists(st.session_state['frap_temp_dir']):
                shutil.rmtree(st.session_state['frap_temp_dir'], ignore_errors=True)
            st.session_state['frap_conv_paths'] = {}
            st.session_state['frap_curve'] = None
            st.session_state['frap_roi'] = None
            st.session_state['frap_bleached_ids'] = None
            st.session_state['frap_temp_dir'] = None
            st.session_state['frap_t_idx'] = 0
            st.session_state['frap_z_idx'] = 0
            st.session_state['frap_autoplay_t'] = False
            st.session_state['frap_autoplay_z'] = False
            st.session_state['frap_display_dt'] = None
            st.session_state['frap_saved_curves'] = []
            mode_str = "2-state (Free + NS ↔ Specific Bound)" if two_state_mode else "3-state (Free + Inter + NS ↔ Specific Bound)"
            _run_info = [("success", f"Simulation complete. {n_mols} molecules, {len(exposure_mol_data)} display frames, {steps_per_exposure} physical steps per exposure. Mode: {mode_str}")]
            if sim_seed_value is not None:
                if motion_seed_value == 'fresh':
                    _run_info.append(("info", f"Sites locked — seed = {int(sim_seed_value)}; motion uses fresh OS entropy."))
                else:
                    _run_info.append(("info", f"Reproducible run — seed = {int(sim_seed_value)} (sites + motion locked)."))
            if num_synthetic_sites > 0:
                _run_info.append(("info", f"✨ {num_synthetic_sites} synthetic sites (green) with {synthetic_strength}× strength"))
            st.session_state['last_run_info'] = _run_info

    if _run_sim_clicked or _needs_render_flag:
        exposure_data      = st.session_state.get('exposure_data', {})
        exposure_mol_data  = st.session_state.get('exposure_mol_data', {})
        exposure_site_data = st.session_state.get('exposure_site_data', {})
        # When loaded from an HPC pkl, use the params stored in the file
        # rather than whatever the sidebar currently shows.
        stored_lateral_fwhm = st.session_state.get('_hpc_lateral_fwhm_nm', lateral_fwhm_nm)
        stored_axial_fwhm   = st.session_state.get('_hpc_axial_fwhm_nm',  axial_fwhm_nm)
        _hpc_nuc  = st.session_state.get('nucleus_diameter_nm_hpc')
        _hpc_vox  = st.session_state.get('voxel_size_nm_hpc')
        _hpc_spc  = st.session_state.get('slice_spacing_nm_hpc')
        _hpc_ns   = st.session_state.get('num_slices_hpc')
        _hpc_ups  = st.session_state.get('upscale_factor_hpc')
        if _hpc_nuc: nucleus_diameter_nm = _hpc_nuc
        if _hpc_vox: voxel_size_nm       = _hpc_vox
        if _hpc_spc: slice_spacing_nm    = _hpc_spc
        if _hpc_ns:  num_slices          = int(_hpc_ns)
        if _hpc_ups: upscale_factor      = _hpc_ups
        display_frames = sorted(exposure_mol_data.keys())
        if st.session_state['temp_viz_dir'] and os.path.exists(st.session_state['temp_viz_dir']):
            shutil.rmtree(st.session_state['temp_viz_dir'], ignore_errors=True)
        temp_dir = tempfile.mkdtemp(prefix="sim_viz_")
        st.session_state['temp_viz_dir'] = temp_dir
        
        grids = {}
        grid_info = {}
        gt_paths = {}
        conv_paths = {}
        count_paths = {}
        count_max_global = 0
        site_strengths = st.session_state.get('site_strengths', None)
        is_synthetic = st.session_state.get('is_synthetic', None)
        strength_mults = st.session_state.get('strength_multipliers', STRENGTH_MULTIPLIERS_DEFAULT)
        strength_range = (min(strength_mults), max(strength_mults))
        
        # Build AU calibration dict if enabled
        au_cal = None
        if au_calibration_enabled:
            au_cal = {
                'ref_concentration': float(au_ref_concentration),
                'ref_signal': float(au_ref_signal),
                'background': float(au_background),
                'noise_std': float(au_noise_std),
                'sim_concentration': float(tf_concentration_nM),
            }
        
        prog = st.progress(0)
        status = st.empty()

        global_max_val = 0.0
        n_frames = len(display_frames)
        for i, disp_frame in enumerate(display_frames):
            status.text(f"Calibrating brightness... frame {i+1}/{n_frames}")
            prog.progress((i + 1) / (n_frames * 2))
            grid, half_span, grid_size = calculate_exposure_grid_fast(
                exposure_data, disp_frame, float(nucleus_diameter_nm), float(voxel_size_nm),
                stored_lateral_fwhm, stored_axial_fwhm
            )
            if grid is not None:
                grids[disp_frame] = grid
                grid_info[disp_frame] = {'half_span': half_span, 'grid_size': grid_size}
                if grid.max() > global_max_val:
                    global_max_val = grid.max()

        st.session_state['calib_max'] = global_max_val if global_max_val > 0 else 1.0
        status.text(f"Max sim intensity: {st.session_state['calib_max']:.4f}. Rendering...")

        total_steps = n_frames
        for i, disp_frame in enumerate(display_frames):
            if disp_frame not in grids:
                continue
            mol_data = exposure_mol_data[disp_frame]
            if disp_frame in exposure_site_data:
                sites_pos, site_occ = exposure_site_data[disp_frame]
            else:
                sites_pos, site_occ = None, None
            
            # Ground Truth (no boundary baked in - overlay added at display time)
            if render_gt_display:
                paths = render_ground_truth_for_frame(
                    mol_data, temp_dir, int(disp_frame), float(nucleus_diameter_nm), float(molecule_diameter_nm),
                    int(image_dpi), float(slice_spacing_nm), sites_pos, site_occ,
                    site_strengths=site_strengths, strength_range=strength_range, is_synthetic=is_synthetic,
                    save_to_disk=True, draw_boundary=False, upscale_factor=int(upscale_factor),
                    voxel_size_nm=float(voxel_size_nm), num_slices=int(num_slices)
                )
                gt_paths[disp_frame] = paths

            # Convolved (fast path - saves to disk, always 8-bit for display)
            # AU off  -> single image normalized to the global max/min across all
            #            frames (min-max normalized, since empty space is ~0).
            # AU on   -> absolute AU intensity (index 0).
            grid_3d = grids[disp_frame]
            g_info = grid_info[disp_frame]
            conv_vmax_list = [st.session_state['calib_max']] if au_cal is None else BRIGHTNESS_VMAX
            c_paths = render_convolved_slices_fast(
                grid_3d, float(nucleus_diameter_nm), float(voxel_size_nm),
                g_info['half_span'], g_info['grid_size'],
                float(slice_spacing_nm), conv_vmax_list,
                upscale_factor=int(upscale_factor),
                save_to_disk=True, output_dir=temp_dir, frame_idx=int(disp_frame),
                bit_depth=8, au_calibration=au_cal,
                noise_floor_frac=display_noise_floor_frac,
                noise_std_frac=display_noise_std_frac,
                num_slices=int(num_slices)
            )
            conv_paths[disp_frame] = c_paths
            
            # Molecule count map (if enabled)
            if render_count_maps:
                _count_mode_v = _count_mode_str(count_mode)
                cnt_grid, cnt_hs, cnt_gs, cnt_slab_centers = calculate_count_grid(
                    exposure_data, disp_frame,
                    float(nucleus_diameter_nm), float(voxel_size_nm),
                    float(slice_spacing_nm),
                    mode=_count_mode_v
                )
                if cnt_grid is not None:
                    cnt_result, cnt_max = render_count_slices(
                        cnt_grid, float(nucleus_diameter_nm), float(voxel_size_nm),
                        cnt_hs, cnt_gs, float(slice_spacing_nm),
                        upscale_factor=int(upscale_factor),
                        save_to_disk=True, output_dir=temp_dir, frame_idx=int(disp_frame),
                        z_projection=count_z_proj
                    )
                    count_paths[disp_frame] = cnt_result
                    if cnt_max > count_max_global:
                        count_max_global = cnt_max

            prog.progress(0.5 + (i + 1) / (total_steps * 2))

        st.session_state['viz_grids'] = grids
        st.session_state['grid_info'] = grid_info
        st.session_state['gt_paths'] = gt_paths
        st.session_state['gt_rendered'] = render_gt_display
        st.session_state['conv_paths'] = conv_paths
        st.session_state['count_paths'] = count_paths
        st.session_state['count_max'] = count_max_global
        st.session_state['count_mode'] = count_mode if render_count_maps else None
        st.session_state['count_z_proj'] = count_z_proj if render_count_maps else None
        st.session_state['au_calibration_dict'] = au_cal
        
        # Store geometry for boundary overlay at display time.
        # Mirror render_convolved_slices_fast: generate candidates from num_slices
        # (slice_spacing_nm sets the grid, num_slices caps how many we take),
        # then filter to those actually inside the nucleus — exactly the set that
        # was rendered — so z-index 0..N-1 maps correctly to rendered slices.
        _NR = float(nucleus_diameter_nm) / 2
        _nsl = int(num_slices)
        _n_half_ss = _nsl // 2
        _sorted_z = sorted(
            z for z in {i * float(slice_spacing_nm) for i in range(-_n_half_ss, _n_half_ss + 1)}
            if abs(z) <= _NR
        )
        st.session_state['sorted_z_positions'] = _sorted_z
        _GS = int(np.ceil(float(nucleus_diameter_nm) / float(voxel_size_nm))) + 2
        if _GS % 2 == 0: _GS += 1
        _hs = (_GS * float(voxel_size_nm)) / 2.0
        _vc = np.arange(_GS) * float(voxel_size_nm) - _hs + float(voxel_size_nm)/2
        _pl = _NR * 1.1
        _iv = np.where((_vc >= -_pl) & (_vc <= _pl))[0]
        _i0, _i1 = _iv[0], _iv[-1] + 1
        st.session_state['crop_extent'] = (
            float(_vc[_i0] - float(voxel_size_nm)/2),
            float(_vc[_i1-1] + float(voxel_size_nm)/2)
        )
        
        # Default the Z view to the center slice for every fresh render
        _ref_paths = gt_paths if gt_paths else conv_paths
        _n_z_slices = len(next(iter(_ref_paths.values()))) if _ref_paths else 0
        if _n_z_slices > 0:
            st.session_state['z_idx'] = _n_z_slices // 2

        status.empty()
        prog.empty()
        st.session_state['_rerun_pending'] = True

    # show the summary from the most recent run
    if st.session_state.get('last_run_info'):
        for _kind, _msg in st.session_state['last_run_info']:
            getattr(st, _kind)(_msg)

    # pkl download — lazy-serialize once per simulation/load, then cache bytes
    if st.session_state.get('exposure_data') and st.session_state.get('_sim_params_for_pkl'):
        if st.session_state.get('exposure_pkl_bytes') is None:
            import pickle as _pkl_mod
            _pkl_payload = {
                'exposure_data':  st.session_state['exposure_data'],
                'num_molecules':  st.session_state.get('sim_num_molecules', 0),
                'site_strengths': st.session_state.get('site_strengths'),
                'is_synthetic':   st.session_state.get('is_synthetic'),
                'params':         st.session_state['_sim_params_for_pkl'],
            }
            st.session_state['exposure_pkl_bytes'] = _pkl_mod.dumps(
                _pkl_payload, protocol=_pkl_mod.HIGHEST_PROTOCOL)
        st.download_button(
            "Download exposure_data.pkl",
            data=st.session_state['exposure_pkl_bytes'],
            file_name="exposure_data.pkl",
            mime="application/octet-stream",
            key="dl_exp_pkl",
        )

    # PLAYBACK UI
    if st.session_state['viz_grids']:
        st.divider()
        st.markdown("### Playback & Metrics")
        
        frames = sorted(st.session_state['viz_grids'].keys())
        if st.session_state['t_idx'] >= len(frames): st.session_state['t_idx'] = 0
        cur_frame = frames[st.session_state['t_idx']]
        
        exposure_mol_data = st.session_state.get('exposure_mol_data', {})
        steps_per_exposure = st.session_state.get('steps_per_exposure', 1)
        stored_sim_dt = st.session_state.get('sim_dt', 0.013)
        stored_display_dt = st.session_state.get('_hpc_display_dt', display_dt)
        
        if cur_frame in exposure_mol_data:
            current_data = exposure_mol_data[cur_frame]
            counts = current_data['state'].value_counts()
        else:
            counts = pd.Series()
        
        total_seconds = cur_frame * stored_display_dt
        mins, secs = int(total_seconds // 60), int(total_seconds % 60)
        millis = int((total_seconds * 1000) % 1000)
        exposure_ms = steps_per_exposure * stored_sim_dt * 1000
        
        n_free  = int(counts.get(0, 0))
        n_inter = int(counts.get(1, 0))
        n_spec  = int(counts.get(2, 0))
        n_ns    = int(counts.get(3, 0))
        total_mols = n_free + n_inter + n_spec + n_ns

        def _pct(n):
            return f"{(100.0 * n / total_mols):.1f}%" if total_mols > 0 else "0.0%"

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Display Frame", cur_frame)
        m2.metric("Time", f"{mins:02d}:{secs:02d}.{millis:03d}")
        m3.metric("Free (Cyan)", f"{n_free} ({_pct(n_free)})")
        m4.metric("Inter (Yel)", f"{n_inter} ({_pct(n_inter)})")
        m5.metric("Bound Spec (Red)", f"{n_spec} ({_pct(n_spec)})")
        m6.metric("Non-Spec (Magenta)", f"{n_ns} ({_pct(n_ns)})")
        st.caption(f"Total molecules: {total_mols} | Exposure: {exposure_ms:.1f}ms ({steps_per_exposure} physics steps)")

        # Transport controls
        c_play, c_prev, c_slide, c_next = st.columns([1.5, 1, 8, 1])
        with c_play:
            if st.button("Stop T" if st.session_state['autoplay'] else "Play T", key="pt"):
                st.session_state['autoplay'] = not st.session_state['autoplay']
                st.session_state['autoplay_z'] = False
                safe_rerun()
        with c_prev:
            if st.button("◀ T", key="prevt"): 
                st.session_state['t_idx'] = max(0, st.session_state['t_idx']-1)
                safe_rerun()
        with c_slide:
            st.session_state['t_idx'] = st.slider("T", 0, len(frames)-1, st.session_state['t_idx'], label_visibility='collapsed')
        with c_next:
            if st.button("T ▶", key="nextt"): 
                st.session_state['t_idx'] = min(len(frames)-1, st.session_state['t_idx']+1)
                safe_rerun()

        _gt_rendered = st.session_state.get('gt_rendered', True)
        gt_paths = st.session_state['gt_paths'].get(cur_frame, {}) if _gt_rendered else {}
        if gt_paths:
            num_slices = len(gt_paths)
        else:
            num_slices = len(st.session_state['conv_paths'].get(cur_frame, {}))
        if num_slices == 0: num_slices = 1
        if st.session_state['z_idx'] >= num_slices: st.session_state['z_idx'] = num_slices // 2
        
        z_play, z_prev, z_slide, z_next = st.columns([1.5, 1, 8, 1])
        with z_play:
            if st.button("Stop Z" if st.session_state['autoplay_z'] else "Play Z", key="pz"):
                st.session_state['autoplay_z'] = not st.session_state['autoplay_z']
                st.session_state['autoplay'] = False
                safe_rerun()
        with z_prev:
            if st.button("◀ Z", key="prevz"):
                st.session_state['z_idx'] = max(0, st.session_state['z_idx']-1)
                safe_rerun()
        with z_slide:
            if num_slices > 1:
                st.session_state['z_idx'] = st.slider("Z", 0, num_slices-1, min(st.session_state['z_idx'], num_slices-1), label_visibility='collapsed')
            else:
                st.session_state['z_idx'] = 0
                st.caption("1 slice")
        with z_next:
            if st.button("Z ▶", key="nextz"):
                st.session_state['z_idx'] = min(max(num_slices-1, 0), st.session_state['z_idx']+1)
                safe_rerun()

        cur_z = st.session_state['z_idx']
        
        # Boundary overlay toggle
        show_boundary = st.checkbox("Show nucleus boundary", value=True, key="show_boundary")
        
        _overlay_z_nm = None
        _overlay_crop = st.session_state.get('crop_extent', None)
        _overlay_z_list = st.session_state.get('sorted_z_positions', None)
        _overlay_nuc_r = float(nucleus_diameter_nm) / 2
        _overlay_lw = max(2, int(upscale_factor) // 3)
        if show_boundary and _overlay_z_list is not None and _overlay_crop is not None and cur_z < len(_overlay_z_list):
            _overlay_z_nm = _overlay_z_list[cur_z]
        
        _has_count = bool(st.session_state.get('count_paths')) and cur_frame in st.session_state.get('count_paths', {})

        if _gt_rendered and _has_count:
            colA, colB, colC = st.columns(3)
        elif _gt_rendered:
            colA, colB = st.columns(2)
        elif _has_count:
            colB, colC = st.columns(2)
        else:
            colB = st.container()

        # Ground Truth display
        if _gt_rendered and gt_paths:
            if show_boundary and _overlay_z_nm is not None:
                gt_display = draw_boundary_overlay(gt_paths[cur_z], _overlay_nuc_r, _overlay_z_nm,
                                                    _overlay_crop[0], _overlay_crop[1], line_width=_overlay_lw)
            else:
                gt_display = gt_paths[cur_z]
            colA.image(gt_display, caption=f"Ground Truth (T={cur_frame}, Z={cur_z})")
            colA.markdown(
                "<div style='font-size:0.88rem;color:#555;margin-top:0.3rem;'>"
                "<div style='display:flex;align-items:center;gap:16px;flex-wrap:wrap;'>"
                "<span style='display:inline-flex;align-items:center;gap:5px;'>"
                "<span style='display:inline-block;width:5px;height:5px;border-radius:50%;background:#333;flex-shrink:0;'></span>Molecule</span>"
                "<span style='display:inline-flex;align-items:center;gap:5px;'>"
                "<span style='display:inline-block;width:13px;height:13px;border-radius:50%;border:2.5px solid #cc2200;background:transparent;flex-shrink:0;'></span>Unbound binding site</span>"
                "<span style='display:inline-flex;align-items:center;gap:5px;'>"
                "<span style='display:inline-block;width:13px;height:13px;border-radius:50%;border:2.5px solid #cc2200;background:rgba(204,34,0,0.4);flex-shrink:0;'></span>Bound binding site</span>"
                "</div>"
                "<div style='font-style:italic;color:#777;margin-top:0.2rem;'>Colour indicates binding site strength (light red → weak, dark red → strong)</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        
        # Convolved display
        if 'conv_paths' in st.session_state and cur_frame in st.session_state['conv_paths']:
            is_au_display = st.session_state.get('au_calibration_enabled', False)
            
            if is_au_display:
                # AU mode: single image per z-slice, absolute AU units
                conv_img_path = st.session_state['conv_paths'][cur_frame][cur_z][0]
                if show_boundary and _overlay_z_nm is not None:
                    conv_display = draw_boundary_overlay(conv_img_path, _overlay_nuc_r, _overlay_z_nm,
                                                          _overlay_crop[0], _overlay_crop[1], line_width=_overlay_lw)
                else:
                    conv_display = conv_img_path
                colB.image(conv_display, caption=f"Convolved AU (T={cur_frame}, Z={cur_z})")
            else:
                # Normalized mode: single min-max normalized image, no brightness selector
                conv_img_path = st.session_state['conv_paths'][cur_frame][cur_z][0]
                if show_boundary and _overlay_z_nm is not None:
                    conv_display = draw_boundary_overlay(conv_img_path, _overlay_nuc_r, _overlay_z_nm,
                                                          _overlay_crop[0], _overlay_crop[1], line_width=_overlay_lw)
                else:
                    conv_display = conv_img_path
                colB.image(conv_display, caption=f"Convolved (normalized, T={cur_frame}, Z={cur_z})")
        
        # Molecule Count Map display
        if _has_count:
            count_data = st.session_state['count_paths'][cur_frame]
            cnt_max_display = st.session_state.get('count_max', 0)
            cnt_mode_display = st.session_state.get('count_mode', 'Full exposure')
            cnt_z_proj_display = st.session_state.get('count_z_proj', 'slab')
            _mode_tag = "snapshot" if "Snapshot" in str(cnt_mode_display) else "full exp"
            _z_tag = "z-slab" if cnt_z_proj_display == "slab" else "single voxel"
            colC.caption(f"🔢 Count per voxel — {_mode_tag}, {_z_tag} (max={cnt_max_display})")
            if cur_z in count_data:
                colC.image(count_data[cur_z]['color'], caption=f"Count Map (T={cur_frame}, Z={cur_z})")

        if st.session_state['autoplay']:
            time.sleep(0.1)
            st.session_state['t_idx'] = (st.session_state['t_idx'] + 1) % len(frames)
            safe_rerun()
        if st.session_state['autoplay_z']:
            time.sleep(0.1)
            st.session_state['z_idx'] = (st.session_state['z_idx'] + 1) % num_slices
            safe_rerun()

        # --- FRAP SECTION ---
        if st.session_state.get('exposure_data'):
            st.markdown("---")
            _frap_has_results = bool(st.session_state.get('frap_conv_paths'))
            with st.expander("FRAP Analysis", expanded=_frap_has_results):
                # ── Controls (compact top row) ──────────────────────────────
                _frap_c1, _frap_c2, _frap_c3, _frap_c4, _frap_c5 = st.columns([2, 2, 2, 2, 2])
                _frap_frame_opts = list(frames)
                _frap_bf_default = st.session_state.get('frap_bleach_frame')
                if _frap_bf_default not in _frap_frame_opts:
                    _frap_bf_default = _frap_frame_opts[0]
                _frap_bf = _frap_c1.selectbox("Bleach frame (t₀)", options=_frap_frame_opts,
                    index=_frap_frame_opts.index(_frap_bf_default), key="frap_bf_sel")

                _crop_frap = st.session_state.get('crop_extent')
                _z_list_frap = st.session_state.get('sorted_z_positions')
                _n_z_frap_sl = len(_z_list_frap) if _z_list_frap else 1
                _frap_bz_def = min(st.session_state.get('frap_bleach_z', _n_z_frap_sl // 2),
                                   max(0, _n_z_frap_sl - 1))
                if _n_z_frap_sl > 1:
                    _frap_bz = _frap_c2.slider("Bleach Z-slice", 0, _n_z_frap_sl - 1,
                                               _frap_bz_def, key="frap_bz_sl")
                else:
                    _frap_bz = 0
                    _frap_c2.caption("Z: center slice")
                _nm_ext_frap = (_crop_frap[1] - _crop_frap[0]) if _crop_frap else 5500.0
                _r_nm_default = max(50.0, round(_nm_ext_frap * 0.1 / 50) * 50)

                _roi_cx_pct = _frap_c3.number_input("Center X (%)", 0.0, 100.0,
                    st.session_state.get('_frap_cx_pct', 50.0), 1.0, key="frap_cx_pct",
                    help="0% = left, 100% = right")
                _roi_cy_pct = _frap_c4.number_input("Center Y (%)", 0.0, 100.0,
                    st.session_state.get('_frap_cy_pct', 50.0), 1.0, key="frap_cy_pct",
                    help="0% = top, 100% = bottom")
                _roi_r_nm = _frap_c5.number_input("Radius (nm)", 10.0, float(_nm_ext_frap / 2),
                    float(st.session_state.get('_frap_r_nm', _r_nm_default)), 50.0, key="frap_r_nm")

                # ── Three-column display: preview | FRAP movie | curve ──────
                _prev_col, _movie_col, _curve_col = st.columns(3)

                # Left: live ROI preview (small)
                _conv_paths_frap = st.session_state.get('conv_paths', {})
                with _prev_col:
                    st.caption("ROI preview")
                    if _frap_bf in _conv_paths_frap and _frap_bz in _conv_paths_frap[_frap_bf]:
                        _bg_pil = Image.open(_conv_paths_frap[_frap_bf][_frap_bz][0]).convert('RGB')
                        _pw, _ph = _bg_pil.size
                        if _crop_frap:
                            from PIL import ImageDraw
                            _nm_per_px = (_crop_frap[1] - _crop_frap[0]) / _pw
                            _pcx = int(_roi_cx_pct / 100.0 * _pw)
                            _pcy = int(_roi_cy_pct / 100.0 * _ph)
                            _pr  = max(1, int(_roi_r_nm / _nm_per_px))
                            _prev = _bg_pil.copy()
                            _dd = ImageDraw.Draw(_prev)
                            _dd.ellipse([_pcx-_pr, _pcy-_pr, _pcx+_pr, _pcy+_pr],
                                        outline=(255,68,68), width=max(2, _pr//20))
                            _dd.ellipse([_pcx-3, _pcy-3, _pcx+3, _pcy+3], fill=(255,68,68))
                            st.image(_prev, use_container_width=True)
                        else:
                            st.image(_bg_pil, use_container_width=True)
                    _apply_clicked = st.button("Apply FRAP", key="apply_frap", type="primary",
                                               use_container_width=True)

                # Middle: FRAP convolved image with its own T/Z controls below
                with _movie_col:
                    _frap_cp = st.session_state.get('frap_conv_paths', {})
                    if _frap_cp:
                        _frap_bleach_t = st.session_state.get('frap_bleach_frame', frames[0])
                        _frap_n_ble = len(st.session_state.get('frap_bleached_ids') or [])
                        _frap_frames = sorted(_frap_cp.keys())
                        _frap_n_z = len(_frap_cp[_frap_frames[0]]) if _frap_frames else 1

                        if st.session_state['frap_t_idx'] >= len(_frap_frames):
                            st.session_state['frap_t_idx'] = 0
                        if st.session_state['frap_z_idx'] >= _frap_n_z:
                            st.session_state['frap_z_idx'] = _frap_n_z // 2

                        # Prefer slider key as canonical position — it captures
                        # manual drags immediately; buttons/autoplay write both keys.
                        _ft = min(st.session_state.get('frap_t_sl',
                                                        st.session_state['frap_t_idx']),
                                  len(_frap_frames) - 1)
                        _fz = min(st.session_state.get('frap_z_sl',
                                                        st.session_state['frap_z_idx']),
                                  _frap_n_z - 1)
                        _frap_frame_cur = _frap_frames[_ft]

                        st.caption(f"FRAP (T={_frap_frame_cur}, Z={_fz}) — bleach T={_frap_bleach_t}, {_frap_n_ble} bleached")
                        if _frap_frame_cur in _frap_cp and _fz in _frap_cp[_frap_frame_cur]:
                            _frap_pil = Image.open(_frap_cp[_frap_frame_cur][_fz][0]).convert('RGB')
                            st.image(_frap_pil, use_container_width=True)
                        else:
                            st.info("No image for current T/Z.")

                        # T controls below image
                        _fbt1, _fbt2, _fbt3 = st.columns([1, 2, 1])
                        if _fbt1.button("◀", key="frap_prev_t"):
                            _nv = max(0, _ft - 1)
                            st.session_state['frap_t_idx'] = _nv
                            st.session_state['frap_t_sl']  = _nv
                            safe_rerun()
                        _fat = st.session_state.get('frap_autoplay_t', False)
                        if _fbt2.button("■ T" if _fat else "▶ T", key="frap_play_t",
                                        use_container_width=True):
                            st.session_state['frap_autoplay_t'] = not _fat
                            st.session_state['frap_autoplay_z'] = False
                            safe_rerun()
                        if _fbt3.button("▶", key="frap_next_t"):
                            _nv = min(len(_frap_frames) - 1, _ft + 1)
                            st.session_state['frap_t_idx'] = _nv
                            st.session_state['frap_t_sl']  = _nv
                            safe_rerun()
                        _ft_new = st.slider("T (frame)", 0, len(_frap_frames) - 1, _ft,
                                            key="frap_t_sl")
                        st.session_state['frap_t_idx'] = _ft_new

                        # Z controls (only shown when more than one Z slice)
                        if _frap_n_z > 1:
                            _fbz1, _fbz2, _fbz3 = st.columns([1, 2, 1])
                            if _fbz1.button("◀", key="frap_prev_z"):
                                _nv = max(0, _fz - 1)
                                st.session_state['frap_z_idx'] = _nv
                                st.session_state['frap_z_sl']  = _nv
                                safe_rerun()
                            _faz = st.session_state.get('frap_autoplay_z', False)
                            if _fbz2.button("■ Z" if _faz else "▶ Z", key="frap_play_z",
                                            use_container_width=True):
                                st.session_state['frap_autoplay_z'] = not _faz
                                st.session_state['frap_autoplay_t'] = False
                                safe_rerun()
                            if _fbz3.button("▶", key="frap_next_z"):
                                _nv = min(_frap_n_z - 1, _fz + 1)
                                st.session_state['frap_z_idx'] = _nv
                                st.session_state['frap_z_sl']  = _nv
                                safe_rerun()
                            _fz_new = st.slider("Z (slice)", 0, _frap_n_z - 1, _fz,
                                                key="frap_z_sl")
                        else:
                            _fz_new = 0
                        st.session_state['frap_z_idx'] = _fz_new

                        # Autoplay
                        if st.session_state.get('frap_autoplay_t'):
                            time.sleep(0.1)
                            _nv = (_ft_new + 1) % len(_frap_frames)
                            st.session_state['frap_t_idx'] = _nv
                            st.session_state['frap_t_sl']  = _nv
                            safe_rerun()
                        if st.session_state.get('frap_autoplay_z'):
                            time.sleep(0.1)
                            _nv = (_fz_new + 1) % _frap_n_z
                            st.session_state['frap_z_idx'] = _nv
                            st.session_state['frap_z_sl']  = _nv
                            safe_rerun()

                        # Download FRAP frames as ZIP of TIFF z-stacks (built once, cached)
                        if st.session_state.get('frap_img_zip_bytes') is None:
                            import zipfile as _zflib, io as _zio
                            _zbuf = _zio.BytesIO()
                            with _zflib.ZipFile(_zbuf, 'w', _zflib.ZIP_DEFLATED) as _zf:
                                for _ff in sorted(_frap_cp.keys()):
                                    _zz_sorted = sorted(_frap_cp[_ff].keys())
                                    _z_imgs = [Image.open(_frap_cp[_ff][_zz][0]).convert('L')
                                               for _zz in _zz_sorted]
                                    _tbuf = _zio.BytesIO()
                                    _z_imgs[0].save(_tbuf, format='TIFF', save_all=True,
                                                    append_images=_z_imgs[1:],
                                                    compression='tiff_deflate')
                                    _tbuf.seek(0)
                                    _zf.writestr(f'frap/frame_{_ff:04d}.tiff', _tbuf.read())
                            st.session_state['frap_img_zip_bytes'] = _zbuf.getvalue()
                        st.download_button(
                            "⬇ Download FRAP images (.zip)",
                            data=st.session_state['frap_img_zip_bytes'],
                            file_name="frap_images.zip",
                            mime="application/zip",
                            key="dl_frap_imgs",
                            use_container_width=True,
                        )
                    else:
                        st.caption("FRAP convolved image")
                        st.info("Apply FRAP to see result here.")

                # Right: recovery curve (x-axis in seconds)
                with _curve_col:
                    _df_c = st.session_state.get('frap_curve')
                    _saved_curves = st.session_state.get('frap_saved_curves', [])
                    _frap_disp_dt = (st.session_state.get('frap_display_dt')
                                     or st.session_state.get('_hpc_display_dt')
                                     or st.session_state.get('_sim_display_dt', display_dt))

                    if _df_c is not None or _saved_curves:
                        _n_saved = len(_saved_curves)
                        _title = f"FRAP Recovery{f' — {_n_saved} saved' if _n_saved else ''}"
                        _fig_frap, _ax_frap = plt.subplots(figsize=(3.5, 3))
                        _ax_frap.axhline(1.0, color='green', linestyle=':', linewidth=0.8, label='Pre-bleach')
                        _ax_frap.axhline(0.0, color='red',   linestyle=':', linewidth=0.8)

                        # Collect all I_norm values for dynamic y limits
                        _all_vals = []

                        # Saved curves (gray, thin, no per-curve legend entry)
                        for _sc in _saved_curves:
                            _sc_t = _sc['curve'].index * _frap_disp_dt
                            _sc_col = 'I_prebleach_norm' if 'I_prebleach_norm' in _sc['curve'].columns else 'I_norm'
                            _ax_frap.plot(_sc_t, _sc['curve'][_sc_col], '-',
                                          color='gray', linewidth=0.8, alpha=0.6)
                            _all_vals.extend(_sc['curve'][_sc_col].dropna().tolist())

                        # Mean ± std across saved curves (≥2 required)
                        if _n_saved >= 2:
                            _pb_col = 'I_prebleach_norm' if all('I_prebleach_norm' in _sc['curve'].columns for _sc in _saved_curves) else 'I_norm'
                            _all_norm = pd.concat([_sc['curve'][_pb_col] for _sc in _saved_curves], axis=1)
                            _mn = _all_norm.mean(axis=1)
                            _sd = _all_norm.std(axis=1)
                            _mt = _mn.index * _frap_disp_dt
                            _ax_frap.plot(_mt, _mn, 'k-', linewidth=2, label=f'Mean (N={_n_saved})')
                            _ax_frap.fill_between(_mt, _mn - _sd, _mn + _sd, color='k', alpha=0.12)
                            _all_vals.extend((_mn + _sd).dropna().tolist())
                            _all_vals.extend((_mn - _sd).dropna().tolist())

                        # Current (unsaved) curve (blue)
                        if _df_c is not None:
                            _frap_bleach_t = st.session_state.get('frap_bleach_frame', frames[0])
                            _t_sec = _df_c.index * _frap_disp_dt
                            _bleach_sec = _frap_bleach_t * _frap_disp_dt
                            _ax_frap.axvline(_bleach_sec, color='gray', linestyle='--',
                                             linewidth=0.8, label='Bleach')
                            _pb_c = 'I_prebleach_norm' if 'I_prebleach_norm' in _df_c.columns else 'I_norm'
                            _ax_frap.plot(_t_sec, _df_c[_pb_c], 'b-o',
                                          markersize=4, linewidth=1.5, label='Current')
                            _all_vals.extend(_df_c[_pb_c].dropna().tolist())

                        # Dynamic y limits with padding
                        if _all_vals:
                            _ylo = min(_all_vals + [0.0])
                            _yhi = max(_all_vals + [1.0])
                            _pad = (_yhi - _ylo) * 0.15 or 0.15
                            _ax_frap.set_ylim(_ylo - _pad, _yhi + _pad)

                        _ax_frap.set_xlabel("Time (s)", fontsize=8)
                        _ax_frap.set_ylabel("Norm. Intensity", fontsize=8)
                        _ax_frap.tick_params(labelsize=7)
                        _ax_frap.set_title(_title, fontsize=9)
                        _ax_frap.legend(fontsize=6)
                        plt.tight_layout()
                        st.pyplot(_fig_frap)
                        plt.close(_fig_frap)
                    else:
                        st.caption("Recovery curve")
                        st.info("Apply FRAP to see curve here.")

                    # Save / Clear buttons below the plot
                    _sv_col, _cl_col = st.columns(2)
                    if _sv_col.button("Save Curve", key="frap_save_curve",
                                      disabled=_df_c is None, use_container_width=True):
                        _saved_curves = _saved_curves + [{
                            'label': f"Run {len(_saved_curves) + 1}",
                            'curve': _df_c.copy(),
                            'bleach_frame': st.session_state.get('frap_bleach_frame'),
                        }]
                        st.session_state['frap_saved_curves'] = _saved_curves
                        safe_rerun()
                    if _cl_col.button("Clear Saved", key="frap_clear_curves",
                                      disabled=len(_saved_curves) == 0, use_container_width=True):
                        st.session_state['frap_saved_curves'] = []
                        safe_rerun()

                    # CSV download — current curve + all saved curves
                    if _df_c is not None or _saved_curves:
                        _all_idx = sorted(set(
                            (list(_df_c.index) if _df_c is not None else []) +
                            [i for _sc in _saved_curves for i in _sc['curve'].index]
                        ))
                        _csv_out = pd.DataFrame(index=pd.Index(_all_idx, name='frame'))
                        _csv_out['time_s'] = _csv_out.index * _frap_disp_dt

                        # ROI geometry metadata (same value every row)
                        _roi_meta = st.session_state.get('frap_roi')
                        if _roi_meta:
                            _csv_out['roi_cx_nm'] = _roi_meta[0]
                            _csv_out['roi_cy_nm'] = _roi_meta[1]
                            _csv_out['roi_cz_nm'] = _roi_meta[2]
                            _csv_out['roi_r_nm']  = _roi_meta[3]

                        _frap_bf_csv = st.session_state.get('frap_bleach_frame', 0)

                        if _df_c is not None:
                            if 'intensity' in _df_c.columns:
                                _csv_out['intensity_raw_current'] = _df_c['intensity']
                            if 'I_prebleach_norm' in _df_c.columns:
                                _csv_out['I_prebleach_norm_current'] = _df_c['I_prebleach_norm']
                            _csv_out['I_norm_current'] = _df_c['I_norm']

                        for _ci, _sc in enumerate(_saved_curves):
                            if 'intensity' in _sc['curve'].columns:
                                _csv_out[f'intensity_raw_run_{_ci+1}'] = _sc['curve']['intensity']
                            if 'I_prebleach_norm' in _sc['curve'].columns:
                                _csv_out[f'I_prebleach_norm_run_{_ci+1}'] = _sc['curve']['I_prebleach_norm']
                            _csv_out[f'I_norm_run_{_ci+1}'] = _sc['curve']['I_norm']

                        if len(_saved_curves) >= 2:
                            _all_n = pd.concat([_sc['curve']['I_norm'] for _sc in _saved_curves], axis=1)
                            _csv_out['I_norm_mean'] = _all_n.mean(axis=1)
                            _csv_out['I_norm_std']  = _all_n.std(axis=1)
                            if all('I_prebleach_norm' in _sc['curve'].columns for _sc in _saved_curves):
                                _all_pb = pd.concat([_sc['curve']['I_prebleach_norm'] for _sc in _saved_curves], axis=1)
                                _csv_out['I_prebleach_norm_mean'] = _all_pb.mean(axis=1)
                                _csv_out['I_prebleach_norm_std']  = _all_pb.std(axis=1)
                        _csv_bytes = _csv_out.reset_index().to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇ Download FRAP curve (.csv)",
                            data=_csv_bytes,
                            file_name="frap_curve.csv",
                            mime="text/csv",
                            key="dl_frap_csv",
                            use_container_width=True,
                        )

                # ── Apply FRAP logic ────────────────────────────────────────
                if _apply_clicked:
                    if _crop_frap is None or _z_list_frap is None:
                        st.error("Missing geometry — please re-run the simulation first.")
                    else:
                        _cx_nm = _crop_frap[0] + (_roi_cx_pct / 100.0) * _nm_ext_frap
                        _cy_nm = _crop_frap[1] - (_roi_cy_pct / 100.0) * _nm_ext_frap
                        _cz_nm = _z_list_frap[_frap_bz] if _frap_bz < len(_z_list_frap) else 0.0
                        _r_nm  = _roi_r_nm

                        _raw_exp = st.session_state.get('exposure_data', {})
                        _b_exp = _raw_exp.get(_frap_bf)
                        if _b_exp is None:
                            st.error("No exposure data for the selected bleach frame.")
                        else:
                            _b_exp = _ensure_compact(_b_exp)
                            _pos_steps_nm = _b_exp['pos_steps'].astype(float) * 1000.0  # (steps, N, 3)
                            _ctr = np.array([_cx_nm, _cy_nm, _cz_nm])
                            _d2 = np.sum((_pos_steps_nm - _ctr) ** 2, axis=2)  # (steps, N)
                            _in_sphere = (_d2 <= _r_nm ** 2).any(axis=0)       # (N,)
                            _bleached_ids = np.where(_in_sphere)[0]

                            _g_info_store = st.session_state.get('grid_info', {})
                            _viz_g = st.session_state['viz_grids']
                            _lat_fwhm = float(st.session_state.get('_hpc_lateral_fwhm_nm', lateral_fwhm_nm))
                            _ax_fwhm  = float(st.session_state.get('_hpc_axial_fwhm_nm',  axial_fwhm_nm))
                            # Use HPC-stored geometry if a PKL was loaded, so FRAP grids match grid_info
                            _vox   = float(st.session_state.get('voxel_size_nm_hpc') or voxel_size_nm)
                            _nuc_d = float(st.session_state.get('nucleus_diameter_nm_hpc') or nucleus_diameter_nm)
                            _au_c  = st.session_state.get('au_calibration_dict')
                            _vmax_frap = [st.session_state.get('calib_max', 1.0)] if _au_c is None else BRIGHTNESS_VMAX

                            if st.session_state.get('frap_temp_dir') and os.path.exists(st.session_state['frap_temp_dir']):
                                shutil.rmtree(st.session_state['frap_temp_dir'], ignore_errors=True)
                            _tmp_frap_dir = tempfile.mkdtemp(prefix="sim_frap_")
                            st.session_state['frap_temp_dir'] = _tmp_frap_dir

                            _frap_paths_tmp = {}
                            _roi_rows = []

                            _gi_b = _g_info_store.get(_frap_bf)
                            _hs_b = _gi_b['half_span'] if _gi_b else 0
                            _gs_b = _gi_b['grid_size'] if _gi_b else 1
                            _gz_b = np.clip(int(np.floor((_cz_nm + _hs_b) / _vox)), 0, _gs_b - 1)
                            _vc_b = np.arange(_gs_b) * _vox - _hs_b + _vox / 2
                            _XX_b, _YY_b = np.meshgrid(_vc_b, _vc_b, indexing='ij')
                            _roi_m = ((_XX_b - _cx_nm) ** 2 + (_YY_b - _cy_nm) ** 2) <= _r_nm ** 2
                            _i_pre = float(_viz_g[_frap_bf][:, :, _gz_b][_roi_m].sum()) if _frap_bf in _viz_g else 1.0

                            with st.spinner(f"Computing FRAP — {len(_bleached_ids)} molecules bleached..."):
                                for _ff in frames:
                                    if _ff not in _g_info_store:
                                        continue
                                    _gi = _g_info_store[_ff]
                                    _hs, _gs = _gi['half_span'], _gi['grid_size']

                                    if _ff < _frap_bf:
                                        _fg = _viz_g[_ff]
                                    else:
                                        _surv = _make_frap_exposure(_raw_exp, _ff, _bleached_ids)
                                        _fg, _, _ = calculate_exposure_grid_fast(
                                            _surv, _ff, _nuc_d, _vox, _lat_fwhm, _ax_fwhm)
                                        if _fg is None:
                                            _fg = np.zeros_like(_viz_g.get(_ff, np.zeros((3, 3, 3))))

                                    _frap_spc = float(
                                        st.session_state.get('slice_spacing_nm_hpc')
                                        or st.session_state.get('_sim_slice_spacing_nm', slice_spacing_nm)
                                    )
                                    _frap_ns = (
                                        st.session_state.get('_sim_num_slices') or int(num_slices)
                                    )
                                    _fp = render_convolved_slices_fast(
                                        _fg, _nuc_d, _vox, _hs, _gs,
                                        _frap_spc, _vmax_frap,
                                        upscale_factor=int(upscale_factor),
                                        save_to_disk=True, output_dir=_tmp_frap_dir,
                                        frame_idx=int(_ff), save_prefix="frap_",
                                        bit_depth=8, au_calibration=_au_c,
                                        noise_floor_frac=display_noise_floor_frac,
                                        noise_std_frac=display_noise_std_frac,
                                        num_slices=_frap_ns
                                    )
                                    _frap_paths_tmp[_ff] = _fp

                                    _gz_f = np.clip(int(np.floor((_cz_nm + _hs) / _vox)), 0, _gs - 1)
                                    _i_roi = float(_fg[:, :, _gz_f][_roi_m].sum())
                                    _roi_rows.append({'frame': _ff, 'intensity': _i_roi})

                            _df_curve = pd.DataFrame(_roi_rows).set_index('frame')
                            # 0-1 normalization (kept for CSV)
                            _post_df  = _df_curve[_df_curve.index >= _frap_bf]
                            _i_0   = float(_post_df['intensity'].iloc[0]) if len(_post_df) > 0 else 0.0
                            _denom = _i_pre - _i_0
                            if _denom > 1e-12:
                                _df_curve['I_norm'] = (_df_curve['intensity'] - _i_0) / _denom
                            else:
                                _df_curve['I_norm'] = 0.0
                            # Pre-bleach normalization: I / mean(pre-bleach intensity) — used for plot
                            _pre_int_rows = _df_curve[_df_curve.index < _frap_bf]['intensity']
                            _i_pre_mean = float(_pre_int_rows.mean()) if len(_pre_int_rows) > 0 else _i_pre
                            _df_curve['I_prebleach_norm'] = (
                                _df_curve['intensity'] / _i_pre_mean if _i_pre_mean > 1e-12 else 0.0
                            )

                            st.session_state['frap_conv_paths'] = _frap_paths_tmp
                            st.session_state['frap_curve']      = _df_curve
                            st.session_state['frap_roi']        = (_cx_nm, _cy_nm, _cz_nm, _r_nm)
                            st.session_state['frap_bleach_frame']  = _frap_bf
                            st.session_state['frap_bleach_z']      = _frap_bz
                            st.session_state['frap_bleached_ids']  = _bleached_ids.tolist()
                            st.session_state['_frap_cx_pct'] = _roi_cx_pct
                            st.session_state['_frap_cy_pct'] = _roi_cy_pct
                            st.session_state['_frap_r_nm']   = _roi_r_nm
                            st.session_state['frap_img_zip_bytes'] = None
                            safe_rerun()

        if False:  # TIFF FRAP import removed — requires molecule tracking data, not available from images
         with st.expander("📂 TIFF FRAP Analysis"):
            # ── Step 1: Load TIFFs ──────────────────────────────────────
            st.caption("Upload one TIFF hyperstack **or** multiple TIFFs (one per timepoint). "
                       "Set **Z slices** to the number of Z planes per timepoint — pages are "
                       "grouped in that order (Z-fast within each T).")
            _ti_up_c1, _ti_up_c2, _ti_up_c3, _ti_up_c4 = st.columns([5, 1, 1, 1])
            _ti_uploads = _ti_up_c1.file_uploader(
                "TIFF file(s)",
                type=["tif", "tiff"], accept_multiple_files=True,
                key="ti_tiff_upload", label_visibility='collapsed')
            _ti_dt_in = _ti_up_c2.number_input(
                "s / frame", 0.001, 100000.0,
                float(st.session_state.get('ti_dt', 10.0)),
                key="ti_dt_in")
            _ti_nz_in = int(_ti_up_c3.number_input(
                "Z slices", 1, 9999, 1, key="ti_nz_in",
                help="Z planes per timepoint. For a hyperstack, set this to match your Z depth."))
            _ti_load_clicked = _ti_up_c4.button(
                "Load", key="ti_load_btn", type="primary", use_container_width=True)

            if _ti_load_clicked and _ti_uploads:
                _sorted_ti = sorted(_ti_uploads, key=lambda f: f.name)
                if st.session_state.get('ti_temp_dir') and os.path.exists(st.session_state['ti_temp_dir']):
                    shutil.rmtree(st.session_state['ti_temp_dir'], ignore_errors=True)
                _ti_tmp = tempfile.mkdtemp(prefix="ti_frap_")
                st.session_state['ti_temp_dir'] = _ti_tmp

                # Collect all pages from all uploaded TIFFs into one flat list
                _all_pages = []
                with st.spinner(f"Loading {len(_sorted_ti)} TIFF(s)..."):
                    for _tf in _sorted_ti:
                        _timg = Image.open(_tf)
                        try:
                            _pi = 0
                            while True:
                                _timg.seek(_pi)
                                _all_pages.append(_timg.copy().convert('RGB'))
                                _pi += 1
                        except EOFError:
                            pass

                # Group pages into timepoints: each group of _ti_nz_in pages = one T
                _ti_new_paths = {}
                _n_z_grp = max(1, _ti_nz_in)
                _total_pages = len(_all_pages)
                _n_t_grp = max(1, _total_pages // _n_z_grp)
                for _fi in range(_n_t_grp):
                    _ti_new_paths[_fi] = {}
                    for _zi in range(_n_z_grp):
                        _pg_idx = _fi * _n_z_grp + _zi
                        if _pg_idx >= _total_pages:
                            break
                        _op = os.path.join(_ti_tmp, f"ti_f{_fi}_z{_zi}.png")
                        _all_pages[_pg_idx].save(_op)
                        _ti_new_paths[_fi][_zi] = _op

                st.session_state['ti_paths'] = _ti_new_paths
                st.session_state['ti_dt']    = _ti_dt_in
                st.session_state['ti_t_idx'] = 0
                st.session_state['ti_z_idx'] = 0
                st.session_state['ti_autoplay_t'] = False
                st.session_state['ti_autoplay_z'] = False
                st.session_state['ti_curve'] = None
                safe_rerun()

            # ── Step 2: viewer + ROI controls (only when images loaded) ─
            _ti_paths = st.session_state.get('ti_paths', {})
            if _ti_paths:
                _ti_frames = sorted(_ti_paths.keys())
                _ti_n_t    = len(_ti_frames)
                _ti_n_z    = len(_ti_paths[_ti_frames[0]])
                if st.session_state['ti_t_idx'] >= _ti_n_t:
                    st.session_state['ti_t_idx'] = 0
                if st.session_state['ti_z_idx'] >= _ti_n_z:
                    st.session_state['ti_z_idx'] = _ti_n_z // 2

                _ti_vw_col, _ti_roi_col = st.columns([3, 2])

                with _ti_vw_col:
                    _ti_ft = st.session_state['ti_t_idx']
                    _ti_fz = st.session_state['ti_z_idx']
                    _ti_img_path = _ti_paths[_ti_frames[_ti_ft]].get(_ti_fz)
                    if _ti_img_path and os.path.exists(_ti_img_path):
                        _ti_disp = Image.open(_ti_img_path).convert('RGB')
                        from PIL import ImageDraw as _IDD
                        _ti_pw, _ti_ph = _ti_disp.size
                        _ti_rcx = int(st.session_state['ti_cx_pct'] / 100.0 * _ti_pw)
                        _ti_rcy = int(st.session_state['ti_cy_pct'] / 100.0 * _ti_ph)
                        _ti_rr  = st.session_state['ti_r_px']
                        _ti_draw_img = _ti_disp.copy()
                        _ti_dd = _IDD.Draw(_ti_draw_img)
                        _ti_dd.ellipse([_ti_rcx-_ti_rr, _ti_rcy-_ti_rr,
                                        _ti_rcx+_ti_rr, _ti_rcy+_ti_rr],
                                       outline=(255, 68, 68), width=max(2, _ti_rr // 20))
                        _ti_dd.ellipse([_ti_rcx-3, _ti_rcy-3, _ti_rcx+3, _ti_rcy+3],
                                       fill=(255, 68, 68))
                        st.image(_ti_draw_img,
                                 caption=f"T={_ti_frames[_ti_ft]}  Z={_ti_fz}")

                    # T controls (only show slider when more than 1 frame)
                    _ti_bt1, _ti_bt2, _ti_bt3 = st.columns([1, 2, 1])
                    if _ti_bt1.button("◀", key="ti_prev_t"):
                        st.session_state['ti_t_idx'] = max(0, _ti_ft - 1)
                        safe_rerun()
                    _ti_at = st.session_state.get('ti_autoplay_t', False)
                    if _ti_bt2.button("■ T" if _ti_at else "▶ T", key="ti_play_t",
                                      use_container_width=True):
                        st.session_state['ti_autoplay_t'] = not _ti_at
                        st.session_state['ti_autoplay_z'] = False
                        safe_rerun()
                    if _ti_bt3.button("▶", key="ti_next_t"):
                        st.session_state['ti_t_idx'] = min(_ti_n_t - 1, _ti_ft + 1)
                        safe_rerun()
                    if _ti_n_t > 1:
                        _ti_ft_new = st.slider("T (frame)", 0, _ti_n_t - 1, _ti_ft,
                                               key="ti_t_sl")
                        st.session_state['ti_t_idx'] = _ti_ft_new
                    else:
                        _ti_ft_new = _ti_ft

                    # Z controls
                    if _ti_n_z > 1:
                        _ti_bz1, _ti_bz2, _ti_bz3 = st.columns([1, 2, 1])
                        if _ti_bz1.button("◀", key="ti_prev_z"):
                            st.session_state['ti_z_idx'] = max(0, _ti_fz - 1)
                            safe_rerun()
                        _ti_az = st.session_state.get('ti_autoplay_z', False)
                        if _ti_bz2.button("■ Z" if _ti_az else "▶ Z", key="ti_play_z",
                                          use_container_width=True):
                            st.session_state['ti_autoplay_z'] = not _ti_az
                            st.session_state['ti_autoplay_t'] = False
                            safe_rerun()
                        if _ti_bz3.button("▶", key="ti_next_z"):
                            st.session_state['ti_z_idx'] = min(_ti_n_z - 1, _ti_fz + 1)
                            safe_rerun()
                        _ti_fz_new = st.slider("Z (slice)", 0, _ti_n_z - 1, _ti_fz,
                                               key="ti_z_sl")
                        st.session_state['ti_z_idx'] = _ti_fz_new
                    else:
                        _ti_fz_new = _ti_fz

                    # Autoplay
                    if st.session_state.get('ti_autoplay_t') and _ti_n_t > 1:
                        time.sleep(0.1)
                        st.session_state['ti_t_idx'] = (_ti_ft_new + 1) % _ti_n_t
                        safe_rerun()
                    if st.session_state.get('ti_autoplay_z') and _ti_n_z > 1:
                        time.sleep(0.1)
                        st.session_state['ti_z_idx'] = (_ti_fz_new + 1) % _ti_n_z
                        safe_rerun()

                with _ti_roi_col:
                    st.markdown("**ROI & FRAP parameters**")
                    _ti_bf_in = int(st.number_input(
                        "Bleach frame #", 0, _ti_n_t - 1,
                        min(st.session_state.get('ti_bleach_frame', 0), _ti_n_t - 1),
                        key="ti_bf_in"))
                    _ti_cx_in = st.number_input(
                        "Center X (%)", 0.0, 100.0,
                        float(st.session_state.get('ti_cx_pct', 50.0)), 1.0,
                        key="ti_cx_in",
                        help="0% = left, 100% = right")
                    _ti_cy_in = st.number_input(
                        "Center Y (%)", 0.0, 100.0,
                        float(st.session_state.get('ti_cy_pct', 50.0)), 1.0,
                        key="ti_cy_in",
                        help="0% = top, 100% = bottom")
                    _ti_r_in = int(st.number_input(
                        "Radius (px)", 1, 9999,
                        int(st.session_state.get('ti_r_px', 20)),
                        key="ti_r_in"))
                    # Save ROI state so viewer circle updates live
                    st.session_state['ti_bleach_frame'] = _ti_bf_in
                    st.session_state['ti_cx_pct'] = _ti_cx_in
                    st.session_state['ti_cy_pct'] = _ti_cy_in
                    st.session_state['ti_r_px']   = _ti_r_in

                    st.markdown("---")
                    _ti_compute = st.button("Compute FRAP Curve", key="ti_compute_btn",
                                            type="primary", use_container_width=True)
                    if _ti_compute:
                        _ti_rows = []
                        _ti_pre_vals = []
                        _ti_bz_for_roi = st.session_state['ti_z_idx']
                        with st.spinner("Computing FRAP curve..."):
                            for _fi in _ti_frames:
                                _zpath = _ti_paths[_fi].get(_ti_bz_for_roi)
                                if _zpath and os.path.exists(_zpath):
                                    _rim = np.array(Image.open(_zpath).convert('L'), dtype=float)
                                else:
                                    _rim = np.zeros((1, 1))
                                _rh2, _rw2 = _rim.shape
                                _rcx2 = int(_ti_cx_in / 100.0 * _rw2)
                                _rcy2 = int(_ti_cy_in / 100.0 * _rh2)
                                _yyr, _xxr = np.ogrid[:_rh2, :_rw2]
                                _rmask = (_xxr - _rcx2)**2 + (_yyr - _rcy2)**2 <= _ti_r_in**2
                                _ti_rows.append({'frame': _fi, 'intensity': float(_rim[_rmask].sum())})
                                if _fi < _ti_bf_in:
                                    _ti_pre_vals.append(float(_rim[_rmask].sum()))
                        _df_ti = pd.DataFrame(_ti_rows).set_index('frame')
                        _i_pre_ti = np.mean(_ti_pre_vals) if _ti_pre_vals else float(_df_ti['intensity'].iloc[0])
                        _post_ti = _df_ti[_df_ti.index >= _ti_bf_in]
                        _i0_ti   = float(_post_ti['intensity'].iloc[0]) if len(_post_ti) > 0 else 0.0
                        _den_ti  = _i_pre_ti - _i0_ti
                        if _den_ti > 1e-12:
                            _df_ti['I_norm'] = (_df_ti['intensity'] - _i0_ti) / _den_ti
                        else:
                            _df_ti['I_norm'] = 0.0
                        _df_ti.loc[_df_ti.index < _ti_bf_in, 'I_norm'] = 1.0
                        st.session_state['ti_curve'] = _df_ti
                        st.session_state['ti_dt']    = _ti_dt_in
                        safe_rerun()

                    # Show curve in ROI column if available
                    _ti_curve = st.session_state.get('ti_curve')
                    if _ti_curve is not None:
                        _ti_t_sec = _ti_curve.index * st.session_state.get('ti_dt', 10.0)
                        _ti_bl_sec = st.session_state.get('ti_bleach_frame', 0) * st.session_state.get('ti_dt', 10.0)
                        _fig_ti, _ax_ti = plt.subplots(figsize=(3.5, 3))
                        _ax_ti.axvline(_ti_bl_sec, color='gray', linestyle='--',
                                       linewidth=0.8, label='Bleach')
                        _ax_ti.plot(_ti_t_sec, _ti_curve['I_norm'], 'b-o',
                                    markersize=4, linewidth=1.5)
                        _ax_ti.axhline(1.0, color='green', linestyle=':', linewidth=0.8,
                                       label='Pre-bleach')
                        _ax_ti.axhline(0.0, color='red', linestyle=':', linewidth=0.8)
                        _ax_ti.set_xlabel("Time (s)", fontsize=8)
                        _ax_ti.set_ylabel("Norm. Intensity", fontsize=8)
                        _ax_ti.set_ylim(-0.2, 1.3)
                        _ax_ti.tick_params(labelsize=7)
                        _ax_ti.set_title("FRAP Recovery", fontsize=9)
                        _ax_ti.legend(fontsize=7)
                        plt.tight_layout()
                        st.pyplot(_fig_ti)
                        plt.close(_fig_ti)

        # --- EXPORT SECTION ---
        st.markdown("---")
        with st.expander("Export as TIFF Z-stacks"):
            st.caption("Each TIFF file contains all Z-slices for one timepoint.")
            
            # --- BIT DEPTH SELECTOR ---
            export_bit_depth = st.radio(
                "TIFF Bit Depth",
                [8, 16],
                index=0,
                format_func=lambda x: f"{x}-bit ({'0–255' if x == 8 else '0–65535'})",
                horizontal=True,
                key="interactive_bit_depth",
                help="16-bit preserves more dynamic range and avoids quantization of dim voxels. File size ~1.5–2× larger."
            )
            
            st.info(
                f"**Pixel Scaling:** Each {voxel_size_nm}nm voxel is upscaled to exactly **{upscale_factor}×{upscale_factor} pixels** using nearest-neighbor interpolation.\n\n"
                f"**Fiji Calibration:**\n"
                f"- Set pixel size to **{voxel_size_nm} nm**\n"
                f"- Each physical voxel = {upscale_factor}×{upscale_factor} image pixels\n"
                f"- Both ground truth and convolved images use identical scaling"
            )
            
            is_au_mode = st.session_state.get('au_calibration_enabled', False)
            stored_au_cal = st.session_state.get('au_calibration_dict', None)
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("Export Ground Truth", key="export_gt"):
                    with st.spinner("Creating Ground Truth TIFF z-stacks..."):
                        zip_buffer = io.BytesIO()
                        gt_paths_export = st.session_state.get('gt_paths', {})
                        
                        _mol_data_exp = st.session_state.get('exposure_mol_data', {})
                        _site_data_exp = st.session_state.get('exposure_site_data', {})
                        _strength_exp = st.session_state.get('site_strengths', None)
                        _synth_exp = st.session_state.get('is_synthetic', None)
                        _smults_exp = st.session_state.get('strength_multipliers', STRENGTH_MULTIPLIERS_DEFAULT)
                        _srange_exp = (min(_smults_exp), max(_smults_exp))

                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for f in frames:
                                if gt_paths_export and f in gt_paths_export:
                                    gt_images = [Image.open(p).convert('RGB') for p in gt_paths_export[f]]
                                else:
                                    _sp, _so = _site_data_exp.get(f, (None, None))
                                    gt_images = render_ground_truth_for_frame(
                                        _mol_data_exp[f], None, int(f),
                                        float(nucleus_diameter_nm), float(molecule_diameter_nm),
                                        int(image_dpi), float(slice_spacing_nm), _sp, _so,
                                        site_strengths=_strength_exp, strength_range=_srange_exp,
                                        is_synthetic=_synth_exp, save_to_disk=False, draw_boundary=False,
                                        upscale_factor=int(upscale_factor), voxel_size_nm=float(voxel_size_nm),
                                        num_slices=int(num_slices)
                                    )
                                tiff_buffer = io.BytesIO()
                                gt_images[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=gt_images[1:], compression='tiff_deflate')
                                tiff_buffer.seek(0)
                                zf.writestr(f"ground_truth/frame_{f:04d}.tiff", tiff_buffer.read())
                        zip_buffer.seek(0)
                        st.download_button(label="⬇️ Download Ground Truth ZIP", data=zip_buffer.getvalue(), file_name="ground_truth_zstacks.zip", mime="application/zip", key="dl_gt")
            
            with export_col2:
                if is_au_mode and stored_au_cal is not None:
                    st.caption("AU Calibration active — images exported in absolute AU units")
                    if st.button("Export Convolved (AU)", key="export_conv_au"):
                        with st.spinner("Creating AU-calibrated Convolved TIFF z-stacks..."):
                            zip_buffer = io.BytesIO()
                            stored_grid_info = st.session_state.get('grid_info', {})
                            
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for f in frames:
                                    grid_3d = st.session_state['viz_grids'][f]
                                    g_info = stored_grid_info[f]
                                    conv_au = render_convolved_slices_fast(
                                        grid_3d, float(nucleus_diameter_nm), float(voxel_size_nm),
                                        g_info['half_span'], g_info['grid_size'],
                                        float(slice_spacing_nm),
                                        [1.0],  # dummy, ignored in AU mode
                                        upscale_factor=int(upscale_factor),
                                        bit_depth=export_bit_depth,
                                        au_calibration=stored_au_cal
                                    )
                                    z_indices = sorted(conv_au.keys())
                                    z_images = [conv_au[z][0] for z in z_indices]
                                    
                                    tiff_buffer = io.BytesIO()
                                    z_images[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=z_images[1:], compression='tiff_deflate')
                                    tiff_buffer.seek(0)
                                    depth_str = f"_{export_bit_depth}bit"
                                    zf.writestr(f"convolved_AU{depth_str}/frame_{f:04d}.tiff", tiff_buffer.read())
                            zip_buffer.seek(0)
                            st.download_button(label="⬇️ Download Convolved AU ZIP", data=zip_buffer.getvalue(), file_name=f"convolved_AU_{export_bit_depth}bit_zstacks.zip", mime="application/zip", key="dl_conv_au")
                else:
                    st.caption("Normalized (min-max) — single auto-scaled image per z-slice")
                    if st.button("Export Convolved", key="export_conv"):
                        with st.spinner("Creating Convolved TIFF z-stacks..."):
                            zip_buffer = io.BytesIO()
                            stored_grid_info = st.session_state.get('grid_info', {})
                            norm_vmax = st.session_state.get('calib_max', 1.0)

                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for f in frames:
                                    if export_bit_depth == 16:
                                        # Re-render from stored grids at 16-bit with upscale
                                        grid_3d = st.session_state['viz_grids'][f]
                                        g_info = stored_grid_info[f]
                                        conv_16 = render_convolved_slices_fast(
                                            grid_3d, float(nucleus_diameter_nm), float(voxel_size_nm),
                                            g_info['half_span'], g_info['grid_size'],
                                            float(slice_spacing_nm),
                                            [norm_vmax],
                                            upscale_factor=int(upscale_factor),
                                            bit_depth=16,
                                            noise_floor_frac=display_noise_floor_frac,
                                            noise_std_frac=display_noise_std_frac
                                        )
                                        z_indices = sorted(conv_16.keys())
                                        z_images = [conv_16[z][0] for z in z_indices]
                                    else:
                                        # Use existing 8-bit files from disk
                                        conv_paths_f = st.session_state['conv_paths'][f]
                                        z_indices = sorted(conv_paths_f.keys())
                                        z_images = [Image.open(conv_paths_f[z][0]).convert('L') for z in z_indices]
                                    
                                    tiff_buffer = io.BytesIO()
                                    z_images[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=z_images[1:], compression='tiff_deflate')
                                    tiff_buffer.seek(0)
                                    depth_str = f"_{export_bit_depth}bit"
                                    zf.writestr(f"convolved_normalized{depth_str}/frame_{f:04d}.tiff", tiff_buffer.read())
                            zip_buffer.seek(0)
                            st.download_button(label="⬇️ Download Convolved ZIP", data=zip_buffer.getvalue(), file_name=f"convolved_normalized_{export_bit_depth}bit_zstacks.zip", mime="application/zip", key="dl_conv")
            
            # --- Molecule Count Map Export ---
            if st.session_state.get('count_paths'):
                st.markdown("---")
                _stored_cnt_mode = st.session_state.get('count_mode', 'Full exposure')
                _stored_z_proj = st.session_state.get('count_z_proj', 'slab')
                _is_snap = "Snapshot" in str(_stored_cnt_mode)
                _mode_label = "snapshot" if _is_snap else "full exposure"
                _z_label = "z-slab" if _stored_z_proj == "slab" else "single voxel"
                st.markdown(f"**Molecule Count Maps (16-bit — {_mode_label}, {_z_label})**")
                cnt_max_val = st.session_state.get('count_max', 0)
                st.caption(f"Per-voxel molecule counts ({_mode_label}, {_z_label}). Max count = {cnt_max_val}. Same geometry as convolved images — overlay in FIJI.")
                if st.button("Export Count Maps (16-bit TIFF)", key="export_count"):
                    with st.spinner("Creating count map TIFF z-stacks..."):
                        zip_buffer = io.BytesIO()
                        stored_count_paths = st.session_state['count_paths']
                        _folder_name = "molecule_counts_16bit_snapshot" if _is_snap else "molecule_counts_16bit"
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for f in frames:
                                if f not in stored_count_paths:
                                    continue
                                z_indices = sorted(stored_count_paths[f].keys())
                                z_images = []
                                for z in z_indices:
                                    entry = stored_count_paths[f][z]
                                    if isinstance(entry['raw'], str):
                                        z_images.append(Image.open(entry['raw']))
                                    else:
                                        z_images.append(entry['raw'])
                                if z_images:
                                    tiff_buffer = io.BytesIO()
                                    z_images[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=z_images[1:], compression='tiff_deflate')
                                    tiff_buffer.seek(0)
                                    zf.writestr(f"{_folder_name}/frame_{f:04d}.tiff", tiff_buffer.read())
                        zip_buffer.seek(0)
                        _zip_name = f"{_folder_name}.zip"
                        st.download_button(label="⬇️ Download Count Maps ZIP", data=zip_buffer.getvalue(), file_name=_zip_name, mime="application/zip", key="dl_count")
            
            is_synthetic_export = st.session_state.get('is_synthetic', None)
            num_synthetic = st.session_state.get('num_synthetic_sites', 0)
            if is_synthetic_export is not None and np.any(is_synthetic_export) and num_synthetic > 0:
                st.markdown("---")
                st.markdown("**Synthetic Binding Site Cluster Centroid**")
                st.caption(f"Centroid of {num_synthetic} synthetic sites in TZYX format (pixels in native grid, before {upscale_factor}× upscaling)")
                if st.button("Export Synthetic Centroid CSV", key="export_synth_centroid"):
                    exposure_site_data_export = st.session_state.get('exposure_site_data', {})
                    if exposure_site_data_export:
                        # --- FIX: compute centroid per frame, not from a single frame ---
                        centroid_data = []
                        for frame_idx in frames:
                            if frame_idx in exposure_site_data_export:
                                sites_pos, _ = exposure_site_data_export[frame_idx]
                                result = calculate_synthetic_centroid_pixels(
                                    sites_pos, is_synthetic_export, [frame_idx],
                                    float(nucleus_diameter_nm), float(voxel_size_nm)
                                )
                                centroid_data.extend(result)
                        # --- build CSV once, after all frames are collected ---
                        if centroid_data:
                            df_centroids = pd.DataFrame(centroid_data)
                            for col in ['Z', 'Y', 'X']:
                                df_centroids[col] = df_centroids[col].round(3)
                            csv_buffer = io.StringIO()
                            df_centroids.to_csv(csv_buffer, index=False)
                            st.download_button(label="⬇️ Download Centroid CSV", data=csv_buffer.getvalue(), file_name="synthetic_site_centroid_TZYX.csv", mime="text/csv", key="dl_synth_csv")
                            st.dataframe(df_centroids.head(), use_container_width=True)
                        else:
                            st.warning("No synthetic sites found to export.")
        



# ==========================================
# TAB 2: BATCH PROCESSING
# ==========================================
with tab_batch:
    st.header("Batch Processing Mode")
    st.info("Upload a CSV file where each row is a separate simulation run. The script will generate images for every row and zip them into individual folders.")

    template_data = SIDEBAR_DEFAULTS.copy()
    template_data['label'] = 'experiment_1'
    # Reproducibility columns — all empty by default. Format matches the SLURM
    # side (sim_repro.py) so a CSV can be moved between the app and the HPC
    # batch script with no edits. See "Reproducibility" UI section below.
    template_data['sim_seed'] = ''
    template_data['sim_group'] = ''
    template_data['site_group'] = ''
    for i in range(5):
        template_data[f'cluster_size_{i}'] = i + 1
        template_data[f'cluster_pct_{i}'] = 20.0
    template_df = pd.DataFrame([template_data])
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    st.download_button(label="📄 Download Parameter Template CSV", data=csv_buffer.getvalue(), file_name="simulation_parameters_template.csv", mime="text/csv")

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload filled CSV", type=['csv'])

    st.subheader("Reproducibility")
    use_batch_seed = st.checkbox(
        "Use seeds (reproducible batch)",
        value=False,
        help="When checked, every row runs with a deterministic seed. The CSV format matches the "
             "SLURM side (sim_repro.py), so the same file can drive both.\n\n"
             "**CSV columns (all optional, all blank-friendly):**\n"
             "• `sim_seed` — integer RNG seed for that row.\n"
             "• `sim_group` — label tying rows that should share **everything** (sites + motion + "
             "binding events). Only render parameters may vary across the group.\n"
             "• `site_group` — label tying rows that should share **only the site layout**. Motion "
             "physics may vary. Auto-derived from `sim_group` if blank.\n\n"
             "**Per-row seed resolution:**\n"
             "1. The row's `sim_seed` cell (if filled with an integer).\n"
             "2. Otherwise: `base seed + row_index` (matches the SLURM `--seed` convention in "
             "batch_hpc_fast.py).\n\n"
             "**How groups affect locking (mirrors batch_hpc_fast.py):**\n"
             "• Row in a `sim_group` → single-seed full lock. Sites + motion both seeded from "
             "the row's `sim_seed`. Two rows in the same `sim_group` with the same seed produce "
             "byte-identical sims (only render params should differ).\n"
             "• Row in a `site_group` only → site lock with deterministic motion seed derived "
             "via `numpy.random.SeedSequence(sim_seed, spawn_key=(1,))` — same convention as "
             "the HPC `mol_rng` stream. Sites and `cluster_D` are byte-identical across rows "
             "with matching site_physics; motion is decoupled from site-setup RNG state.\n"
             "• Row in neither → existing single-seed behavior (or OS entropy if no seed set).\n\n"
             "When unchecked, all rows use OS entropy. The seeds used for each row (sim_seed + "
             "motion_seed) are written to `seeds_used.csv` inside the output ZIP for replay.\n\n"
             "**Note on caching:** the SLURM script caches simulation output and skips re-running "
             "rows that share a group. The streamlit app does **not** cache — it re-runs each "
             "row — but because both seeds are deterministic, rows that share a `sim_group` and "
             "seed produce identical output, and rows that share a `site_group` and seed produce "
             "identical sites with reproducible (but TF-conc-dependent) motion. Same physics, "
             "no speedup.\n\n"
             "**Cross-tool replay:** app2 uses the legacy `np.random.seed()` (MT19937); the HPC "
             "script uses `Generator` (PCG64). Same seed → same numbers WITHIN each tool, but "
             "**different bytes between** them."
    )
    if use_batch_seed:
        batch_base_seed = st.number_input(
            "Base seed (integer)",
            value=42, min_value=0, max_value=2**31 - 1, step=1,
            help="Used when a row's `sim_seed` cell is blank. Row seed = base + row_index."
        )
    else:
        batch_base_seed = None

    st.subheader("Batch Settings")
    save_gt = st.checkbox("Save Ground Truth Images", value=True)
    save_conv = st.checkbox("Save Convolved Images", value=True)
    save_count = st.checkbox("Save Molecule Count Maps", value=False,
        help="Save per-voxel molecule count grids as 16-bit TIFF z-stacks. Same geometry as convolved images — overlay in FIJI to see how many molecules are in each PSF-convolved cluster.")
    batch_count_mode = "Full exposure"
    batch_count_z_proj = "slab"
    if save_count:
        batch_count_mode = st.radio(
            "Count mode",
            ["Full exposure", "Snapshot (last step)", "Centroid (mean position)"],
            index=0,
            horizontal=True,
            key="batch_count_mode",
            help="**Full exposure:** count every molecule at every physics sub-step during the exposure. "
                 "**Snapshot:** count each molecule exactly once using only the last physics step. "
                 "**Centroid:** count each molecule once at its mean position over the exposure (matches positions.txt)."
        )
        batch_count_z_proj = st.radio(
            "Z-slice projection",
            ["Z-slab (±half slice spacing)", "Single voxel"],
            index=0,
            horizontal=True,
            key="batch_count_z_proj",
            help="**Z-slab:** sum all voxels within ±(slice spacing / 2) of each z-plane. "
                 "**Single voxel:** count only the single voxel-thick layer at the exact z-plane."
        )
        batch_count_z_proj = "slab" if "slab" in batch_count_z_proj.lower() else "single"
    
    # --- BIT DEPTH SELECTOR (BATCH) ---
    batch_bit_depth = st.radio(
        "TIFF Bit Depth (Convolved)",
        [8, 16],
        index=0,
        format_func=lambda x: f"{x}-bit ({'0–255' if x == 8 else '0–65535'})",
        horizontal=True,
        key="batch_bit_depth",
        help="16-bit preserves more dynamic range and avoids quantization of dim voxels. File size ~1.5–2× larger."
    )
    
    st.caption("Batch convolved images are saved at **native resolution** (no upscaling) for analysis. Ground truth images still use the upscale factor for visualization.")
    
    # --- BRIGHTNESS MODE ---
    st.info(
        "Convolved images are auto-normalized: all rows are simulated first, then the "
        "global max voxel intensity is used as `vmax` so every image is directly comparable.\n\n"
        "Rows with `au_calibration_enabled=True` in the CSV use AU calibration instead."
    )
    headroom_factor = st.number_input(
        "Headroom factor",
        value=1.0, min_value=0.1, max_value=5.0, step=0.1,
        help="Multiplier on the global max intensity. 1.0 = use raw max as vmax; <1.0 = brighter (clips peaks); >1.0 = darker."
    )
    
    if st.button("Run Batch Processing") and uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        for key, val in SIDEBAR_DEFAULTS.items():
            if key not in batch_df.columns:
                batch_df[key] = val

        zip_buffer = io.BytesIO()
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_rows = len(batch_df)

        # Per-row seed log: written to seeds_used.csv inside the output ZIP so a
        # batch can be replayed bit-for-bit later (or its rows checked against
        # the SLURM scripts' output for parity).
        seeds_used = []
        
        # =============================================
        # AUTO-NORMALIZE MODE (two-phase)
        # =============================================
        if save_conv:
            
            # === PHASE 1: Simulate all rows, compute grids, find global max ===
            status_text.text("Phase 1: Running simulations and computing grids...")
            all_row_results = []
            global_max_intensity = 0.0
            
            for idx, row in batch_df.iterrows():
                p = extract_batch_row_params(row)
                row_folder = f"Row_{idx}_{p['row_label']}" if p['row_label'] else f"Row_{idx}"
                row_seed = resolve_row_seed(row, idx, batch_base_seed)
                row_motion_seed = resolve_row_motion_seed(row, row_seed)
                seeds_used.append({
                    'row_index': idx,
                    'label': p['row_label'],
                    'sim_seed': row_seed,
                    'motion_seed': row_motion_seed,
                    'sim_group': _row_group_str(row, 'sim_group'),
                    'site_group': _row_group_str(row, 'site_group'),
                })
                if row_seed is not None:
                    if row_motion_seed is None:
                        seed_tag = f" [seed={row_seed}, full lock]"
                    else:
                        seed_tag = f" [seed={row_seed}, sites locked / motion seed={row_motion_seed}]"
                else:
                    seed_tag = ""
                status_text.text(f"Phase 1 — Simulating Row {idx+1}/{total_rows}: {p['row_label']}{seed_tag}...")
                
                def _make_sim_cb(r_idx, r_total, r_label, st_ref):
                    def _cb(frame_done, frame_total):
                        st_ref.text(f"Phase 1 — Row {r_idx}/{r_total} ({r_label}): simulating frame {frame_done}/{frame_total}")
                    return _cb
                
                exposure_data_res, _, site_strengths_res, is_synthetic_res, steps_per_exposure_res = run_full_simulation(
                    int(row['num_frames']), int(row['num_binding_sites']),
                    float(row['nucleus_diameter_nm']), float(row['tf_concentration_nM']),
                    float(row['sim_dt']), float(row['display_dt']),
                    float(row['pct_free_init']), float(row['pct_inter_init']), float(row['pct_ns_init']),
                    float(row['d_inter']), float(row['d_free']), float(row['d_ns']),
                    float(row['d_free_std']), float(row['d_inter_std']), float(row['d_ns_std']),
                    float(row['bs_diff_mean']), float(row['bs_diff_std']), float(row['bs_jiggle_std']), float(p['bs_hurst']),
                    float(row['k_fi']), float(row['k_if']), float(row['k_fns']), float(row['k_ins']), float(row['ns_res']),
                    p['cluster_dist'], float(row['res_time_mean']), float(row['res_time_std']), 
                    float(row['capture_rad_nm']), float(row['k_on_prob']),
                    p['strength_multipliers'], p['strength_percentages'], p['cluster_correlated'],
                    num_synthetic_sites=p['num_synthetic'],
                    synthetic_strength=p['synthetic_strength'],
                    synthetic_cluster_radius_nm=p['synthetic_radius'],
                    cluster_radius_nm=p['cluster_radius_nm'],
                    exposure_time_s=p['exposure_time_s'],
                    enable_cooperativity=p['enable_coop'],
                    coop_strength=p['coop_strength'],
                    coop_hill=p['coop_hill'],
                    two_state_mode=p['two_state_mode'],
                    show_progress=False,
                    progress_callback=_make_sim_cb(idx+1, total_rows, p['row_label'], status_text),
                    sim_seed=row_seed,
                    motion_seed=row_motion_seed,
                )
                
                exposure_mol_data, exposure_site_data = compute_exposure_frames(exposure_data_res, site_strengths_res)
                display_frames = sorted(exposure_mol_data.keys())
                _batch_spacing, _batch_num_slices = _resolve_batch_slice_params(row)

                # Compute grids and track global max (only for non-AU rows)
                row_grids = {}
                row_count_grids = {}
                n_disp = len(display_frames)
                for f_idx, disp_frame in enumerate(display_frames):
                    status_text.text(f"Phase 1 — Row {idx+1}/{total_rows} ({p['row_label']}): computing grid {f_idx+1}/{n_disp}")
                    grid, hs, gs = calculate_exposure_grid_fast(
                        exposure_data_res, disp_frame,
                        float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                        p['lateral_fwhm'], p['axial_fwhm']
                    )
                    if grid is not None:
                        row_grids[disp_frame] = (grid, hs, gs)
                        if not p['au_calibration_enabled']:
                            frame_max = grid.max()
                            if frame_max > global_max_intensity:
                                global_max_intensity = frame_max
                    if save_count:
                        _batch_mode = _count_mode_str(batch_count_mode)
                        cnt_g, cnt_hs, cnt_gs, cnt_slabs = calculate_count_grid(
                            exposure_data_res, disp_frame,
                            float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                            _batch_spacing,
                            mode=_batch_mode
                        )
                        if cnt_g is not None:
                            row_count_grids[disp_frame] = (cnt_g, cnt_hs, cnt_gs, cnt_slabs)
                
                all_row_results.append({
                    'row': row, 'row_folder': row_folder, 'params': p,
                    'exposure_mol_data': exposure_mol_data,
                    'exposure_site_data': exposure_site_data,
                    'site_strengths': site_strengths_res,
                    'is_synthetic': is_synthetic_res,
                    'grids': row_grids,
                    'count_grids': row_count_grids,
                    'display_frames': display_frames,
                    'sim_seed': row_seed,
                    'batch_spacing': _batch_spacing,
                    'batch_num_slices': _batch_num_slices,
                })
                progress_bar.progress((idx + 1) / (total_rows * 2))
            
            auto_vmax = global_max_intensity * headroom_factor if global_max_intensity > 0 else 1.0
            status_text.text(f"Global max intensity: {global_max_intensity:.4f} → vmax = {auto_vmax:.4f} (×{headroom_factor}). Rendering...")
            
            # === PHASE 2: Render all rows with global vmax or AU calibration ===
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                cal_info = f"global_max_intensity,{global_max_intensity:.6f}\nheadroom_factor,{headroom_factor:.4f}\nauto_vmax,{auto_vmax:.6f}\nbit_depth,{batch_bit_depth}\n"
                zf.writestr("brightness_calibration.csv", cal_info)
                
                for phase2_idx, rd in enumerate(all_row_results):
                    row = rd['row']
                    p = rd['params']
                    n_disp_p2 = len(rd['display_frames'])
                    _batch_spacing    = rd['batch_spacing']
                    _batch_num_slices = rd['batch_num_slices']

                    # Mirror batch_hpc_fast.py:386-387 — re-seed numpy's global
                    # RNG before rendering this row so AU calibration noise is
                    # reproducible and decoupled from the simulation phase.
                    if rd.get('sim_seed') is not None:
                        np.random.seed(int(rd['sim_seed']))
                    
                    # Build AU calibration dict for this row if enabled
                    row_au_cal = None
                    if p['au_calibration_enabled']:
                        row_au_cal = {
                            'ref_concentration': p['au_ref_concentration_nM'],
                            'ref_signal': p['au_ref_signal'],
                            'background': p['au_background'],
                            'noise_std': p['au_noise_std'],
                            'sim_concentration': float(row['tf_concentration_nM']),
                        }
                    
                    for f_idx_p2, disp_frame in enumerate(rd['display_frames']):
                        status_text.text(f"Phase 2 — Row {phase2_idx+1}/{total_rows} ({p['row_label']}): rendering frame {f_idx_p2+1}/{n_disp_p2}")
                        mol_data = rd['exposure_mol_data'][disp_frame]
                        
                        # Ground Truth (with upscale factor for visualization)
                        if save_gt:
                            if disp_frame in rd['exposure_site_data']:
                                sites_pos, site_occ = rd['exposure_site_data'][disp_frame]
                            else:
                                sites_pos, site_occ = None, None
                            gt_images = render_ground_truth_for_frame(
                                mol_data, None, int(disp_frame),
                                float(row['nucleus_diameter_nm']), float(row['molecule_diameter_nm']),
                                int(row['image_dpi']), _batch_spacing,
                                sites_pos, site_occ,
                                site_strengths=rd['site_strengths'], strength_range=p['strength_range'],
                                is_synthetic=rd['is_synthetic'], save_to_disk=False, draw_boundary=False,
                                upscale_factor=p['upscale_factor'], voxel_size_nm=float(row['voxel_size_nm']),
                                num_slices=_batch_num_slices
                            )
                            tiff_buffer = io.BytesIO()
                            gt_images[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=gt_images[1:], compression='tiff_deflate')
                            tiff_buffer.seek(0)
                            zf.writestr(f"{rd['row_folder']}/Ground_Truth/frame_{disp_frame:04d}.tiff", tiff_buffer.read())
                        
                        # Convolved — AU or auto-normalize
                        if disp_frame in rd['grids']:
                            grid, half_span, grid_size = rd['grids'][disp_frame]
                            
                            if row_au_cal is not None:
                                # AU calibration mode — native resolution (no upscale)
                                conv_images_dict = render_convolved_slices_fast(
                                    grid, float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                                    half_span, grid_size, _batch_spacing,
                                    [1.0],  # dummy, ignored in AU mode
                                    upscale_factor=1, bit_depth=batch_bit_depth,
                                    au_calibration=row_au_cal,
                                    num_slices=_batch_num_slices
                                )
                                z_stack = [conv_images_dict[z][0] for z in range(len(conv_images_dict))]
                                tiff_buffer = io.BytesIO()
                                z_stack[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=z_stack[1:], compression='tiff_deflate')
                                tiff_buffer.seek(0)
                                depth_str = f"_{batch_bit_depth}bit"
                                zf.writestr(f"{rd['row_folder']}/Convolved_AU{depth_str}/frame_{disp_frame:04d}.tiff", tiff_buffer.read())
                            else:
                                # Auto-normalize with global vmax — native resolution (no upscale)
                                conv_images_dict = render_convolved_slices_fast(
                                    grid, float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                                    half_span, grid_size, _batch_spacing,
                                    [auto_vmax], upscale_factor=1, bit_depth=batch_bit_depth,
                                    num_slices=_batch_num_slices
                                )
                                z_stack = [conv_images_dict[z][0] for z in range(len(conv_images_dict))]
                                tiff_buffer = io.BytesIO()
                                z_stack[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=z_stack[1:], compression='tiff_deflate')
                                tiff_buffer.seek(0)
                                depth_str = f"_{batch_bit_depth}bit"
                                zf.writestr(f"{rd['row_folder']}/Convolved_auto{depth_str}/frame_{disp_frame:04d}.tiff", tiff_buffer.read())
                        
                        # Molecule Count Map
                        if save_count and disp_frame in rd.get('count_grids', {}):
                            cnt_g, cnt_hs, cnt_gs, cnt_slabs = rd['count_grids'][disp_frame]
                            _batch_mode_p2 = _count_mode_str(batch_count_mode)
                            _suffix = {"full": "", "snapshot": "_snapshot", "centroid": "_centroid"}[_batch_mode_p2]
                            _cnt_folder = "Molecule_Counts_16bit" + _suffix
                            if batch_count_z_proj == "single":
                                _cnt_folder += "_singlevoxel"
                            cnt_result, _ = render_count_slices(
                                cnt_g, float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                                cnt_hs, cnt_gs, _batch_spacing,
                                upscale_factor=1, z_projection=batch_count_z_proj
                            )
                            z_indices_cnt = sorted(cnt_result.keys())
                            z_stack_cnt = [cnt_result[z]['raw'] for z in z_indices_cnt]
                            if z_stack_cnt:
                                tiff_buffer = io.BytesIO()
                                z_stack_cnt[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=z_stack_cnt[1:], compression='tiff_deflate')
                                tiff_buffer.seek(0)
                                zf.writestr(f"{rd['row_folder']}/{_cnt_folder}/frame_{disp_frame:04d}.tiff", tiff_buffer.read())
                    
                    # Synthetic centroid CSV — auto mode
                    # --- FIX: compute centroid per frame using rd's per-frame site data ---
                    if rd['is_synthetic'] is not None and np.any(rd['is_synthetic']) and p['num_synthetic'] > 0:
                        centroid_data_batch = []
                        for frame_idx in rd['display_frames']:
                            if frame_idx in rd['exposure_site_data']:
                                sites_pos_batch, _ = rd['exposure_site_data'][frame_idx]
                                result = calculate_synthetic_centroid_pixels(
                                    sites_pos_batch, rd['is_synthetic'], [frame_idx],
                                    float(row['nucleus_diameter_nm']), float(row['voxel_size_nm'])
                                )
                                centroid_data_batch.extend(result)
                        if centroid_data_batch:
                            df_c = pd.DataFrame(centroid_data_batch)
                            for col in ['Z', 'Y', 'X']:
                                df_c[col] = df_c[col].round(3)
                            csv_buf = io.StringIO()
                            df_c.to_csv(csv_buf, index=False)
                            zf.writestr(f"{rd['row_folder']}/synthetic_site_centroid_TZYX.csv", csv_buf.getvalue())
                    
                    progress_bar.progress(0.5 + (phase2_idx + 1) / (total_rows * 2))

                # Write per-row seed log inside the ZIP for replay/auditing.
                if seeds_used:
                    seeds_buf = io.StringIO()
                    pd.DataFrame(seeds_used).to_csv(seeds_buf, index=False)
                    zf.writestr("seeds_used.csv", seeds_buf.getvalue())

            status_text.success(f"Batch complete! Auto vmax = {auto_vmax:.4f} | {batch_bit_depth}-bit TIFFs (AU rows use per-row calibration)")
            st.download_button(label="⬇️ Download Batch Results (ZIP)", data=zip_buffer.getvalue(), file_name="batch_simulation_results.zip", mime="application/zip")
        
        # =============================================
        # GT / COUNT MAPS ONLY (save_conv is False)
        # =============================================
        else:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for idx, row in batch_df.iterrows():
                    p = extract_batch_row_params(row)
                    row_folder = f"Row_{idx}_{p['row_label']}" if p['row_label'] else f"Row_{idx}"
                    row_seed = resolve_row_seed(row, idx, batch_base_seed)
                    row_motion_seed = resolve_row_motion_seed(row, row_seed)
                    seeds_used.append({
                        'row_index': idx,
                        'label': p['row_label'],
                        'sim_seed': row_seed,
                        'motion_seed': row_motion_seed,
                        'sim_group': _row_group_str(row, 'sim_group'),
                        'site_group': _row_group_str(row, 'site_group'),
                    })
                    if row_seed is not None:
                        if row_motion_seed is None:
                            seed_tag = f" [seed={row_seed}, full lock]"
                        else:
                            seed_tag = f" [seed={row_seed}, sites locked / motion seed={row_motion_seed}]"
                    else:
                        seed_tag = ""
                    status_text.text(f"Processing Row {idx+1}/{total_rows}: {p['row_label']}{seed_tag}...")
                    
                    def _make_sim_cb_m(r_idx, r_total, r_label, st_ref):
                        def _cb(frame_done, frame_total):
                            st_ref.text(f"Row {r_idx}/{r_total} ({r_label}): simulating frame {frame_done}/{frame_total}")
                        return _cb
                    
                    exposure_data_res, _, site_strengths_res, is_synthetic_res, steps_per_exposure_res = run_full_simulation(
                        int(row['num_frames']), int(row['num_binding_sites']),
                        float(row['nucleus_diameter_nm']), float(row['tf_concentration_nM']),
                        float(row['sim_dt']), float(row['display_dt']),
                        float(row['pct_free_init']), float(row['pct_inter_init']), float(row['pct_ns_init']),
                        float(row['d_inter']), float(row['d_free']), float(row['d_ns']),
                        float(row['d_free_std']), float(row['d_inter_std']), float(row['d_ns_std']),
                        float(row['bs_diff_mean']), float(row['bs_diff_std']), float(row['bs_jiggle_std']), float(p['bs_hurst']),
                        float(row['k_fi']), float(row['k_if']), float(row['k_fns']), float(row['k_ins']), float(row['ns_res']),
                        p['cluster_dist'], float(row['res_time_mean']), float(row['res_time_std']),
                        float(row['capture_rad_nm']), float(row['k_on_prob']),
                        p['strength_multipliers'], p['strength_percentages'], p['cluster_correlated'],
                        num_synthetic_sites=p['num_synthetic'],
                        synthetic_strength=p['synthetic_strength'],
                        synthetic_cluster_radius_nm=p['synthetic_radius'],
                        cluster_radius_nm=p['cluster_radius_nm'],
                        exposure_time_s=p['exposure_time_s'],
                        enable_cooperativity=p['enable_coop'],
                        coop_strength=p['coop_strength'],
                        coop_hill=p['coop_hill'],
                        two_state_mode=p['two_state_mode'],
                        show_progress=False,
                        progress_callback=_make_sim_cb_m(idx+1, total_rows, p['row_label'], status_text),
                        sim_seed=row_seed,
                        motion_seed=row_motion_seed,
                    )

                    exposure_mol_data, exposure_site_data = compute_exposure_frames(exposure_data_res, site_strengths_res)
                    display_frames = sorted(exposure_mol_data.keys())
                    n_disp_manual = len(display_frames)
                    _batch_spacing, _batch_num_slices = _resolve_batch_slice_params(row)

                    # Mirror batch_hpc_fast.py:386-387 — re-seed numpy's global
                    # RNG before rendering this row so AU calibration noise is
                    # reproducible.
                    if row_seed is not None:
                        np.random.seed(int(row_seed))
                    
                    for f_idx_m, disp_frame in enumerate(display_frames):
                        status_text.text(f"Row {idx+1}/{total_rows} ({p['row_label']}): rendering frame {f_idx_m+1}/{n_disp_manual}")
                        mol_data = exposure_mol_data[disp_frame]
                        
                        # Ground Truth (with upscale factor)
                        if save_gt:
                            if disp_frame in exposure_site_data:
                                sites_pos, site_occ = exposure_site_data[disp_frame]
                            else:
                                sites_pos, site_occ = None, None
                            gt_images = render_ground_truth_for_frame(
                                mol_data, None, int(disp_frame),
                                float(row['nucleus_diameter_nm']), float(row['molecule_diameter_nm']),
                                int(row['image_dpi']), _batch_spacing,
                                sites_pos, site_occ,
                                site_strengths=site_strengths_res, strength_range=p['strength_range'],
                                is_synthetic=is_synthetic_res, save_to_disk=False, draw_boundary=False,
                                upscale_factor=p['upscale_factor'], voxel_size_nm=float(row['voxel_size_nm']),
                                num_slices=_batch_num_slices
                            )
                            tiff_buffer = io.BytesIO()
                            gt_images[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=gt_images[1:], compression='tiff_deflate')
                            tiff_buffer.seek(0)
                            zf.writestr(f"{row_folder}/Ground_Truth/frame_{disp_frame:04d}.tiff", tiff_buffer.read())
                        
                        # Molecule Count Map
                        if save_count:
                            _batch_mode = _count_mode_str(batch_count_mode)
                            cnt_g, cnt_hs, cnt_gs, cnt_slabs = calculate_count_grid(
                                exposure_data_res, disp_frame,
                                float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                                _batch_spacing,
                                mode=_batch_mode
                            )
                            if cnt_g is not None:
                                _suffix = {"full": "", "snapshot": "_snapshot", "centroid": "_centroid"}[_batch_mode]
                                _cnt_folder = "Molecule_Counts_16bit" + _suffix
                                if batch_count_z_proj == "single":
                                    _cnt_folder += "_singlevoxel"
                                cnt_result, _ = render_count_slices(
                                    cnt_g, float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                                    cnt_hs, cnt_gs, _batch_spacing,
                                    upscale_factor=1, z_projection=batch_count_z_proj
                                )
                                z_indices_cnt = sorted(cnt_result.keys())
                                z_stack_cnt = [cnt_result[z]['raw'] for z in z_indices_cnt]
                                if z_stack_cnt:
                                    tiff_buffer = io.BytesIO()
                                    z_stack_cnt[0].save(tiff_buffer, format='TIFF', save_all=True, append_images=z_stack_cnt[1:], compression='tiff_deflate')
                                    tiff_buffer.seek(0)
                                    zf.writestr(f"{row_folder}/{_cnt_folder}/frame_{disp_frame:04d}.tiff", tiff_buffer.read())
                    
                    # Synthetic centroid CSV — manual mode
                    # --- FIX: use local display_frames/exposure_site_data/is_synthetic_res, per frame ---
                    if is_synthetic_res is not None and np.any(is_synthetic_res) and p['num_synthetic'] > 0:
                        centroid_data_batch = []
                        for frame_idx in display_frames:
                            if frame_idx in exposure_site_data:
                                sites_pos_batch, _ = exposure_site_data[frame_idx]
                                result = calculate_synthetic_centroid_pixels(
                                    sites_pos_batch, is_synthetic_res, [frame_idx],
                                    float(row['nucleus_diameter_nm']), float(row['voxel_size_nm'])
                                )
                                centroid_data_batch.extend(result)
                        if centroid_data_batch:
                            df_c = pd.DataFrame(centroid_data_batch)
                            for col in ['Z', 'Y', 'X']:
                                df_c[col] = df_c[col].round(3)
                            csv_buf = io.StringIO()
                            df_c.to_csv(csv_buf, index=False)
                            zf.writestr(f"{row_folder}/synthetic_site_centroid_TZYX.csv", csv_buf.getvalue())
                    
                    progress_bar.progress((idx + 1) / total_rows)

                # Write per-row seed log inside the ZIP for replay/auditing.
                if seeds_used:
                    seeds_buf = io.StringIO()
                    pd.DataFrame(seeds_used).to_csv(seeds_buf, index=False)
                    zf.writestr("seeds_used.csv", seeds_buf.getvalue())

            status_text.success(f"Batch Processing Complete! ({batch_bit_depth}-bit TIFFs)")
            st.download_button(label="⬇️ Download Batch Results (ZIP)", data=zip_buffer.getvalue(), file_name="batch_simulation_results.zip", mime="application/zip")

# Deferred rerun — called after both tabs have properly closed so the
# rerun never fires from inside a tab context (which causes ghost elements).
if st.session_state.pop('_rerun_pending', False):
    safe_rerun()