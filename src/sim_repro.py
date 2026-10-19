"""
sim_repro.py — Reproducibility & ground-truth caching helper.

PURPOSE
-------
Lets multiple CSV rows share ground-truth data while varying only selected
parameters. Two tiers of sharing are supported:

  sim_group   — rows share EVERYTHING (sites + motion + binding events).
                Only render parameters (PSF, voxel, bit depth, …) may vary.
  site_group  — rows share ONLY the site layout (positions, cluster IDs,
                strengths, per-site diffusion coefficients). Motion is
                re-simulated per row, so TF concentration, diffusion,
                kinetics, and timing are all free to vary.

A row in a `sim_group` implicitly belongs to a `site_group` of the same
name (auto-derived), so a single sim_leader writes BOTH caches on its way
through. Downstream rows can then attach to either tier without the user
having to fill in both columns manually.

CSV COLUMNS
-----------
  sim_group   (string)  — full-ground-truth cache key. Optional.
  site_group  (string)  — site-layout cache key. Optional. Auto-derived
                          from sim_group when missing.
  sim_seed    (int)     — RNG seed. Same seed + same physics ⇒ byte-
                          identical output. Split internally via
                          numpy.random.SeedSequence with spawn_key labels
                          "site" and "motion" so followers that skip site
                          generation still have a motion RNG stream
                          aligned with the leader's.

PHYSICS vs RENDER vs SITE-PHYSICS COLUMNS
-----------------------------------------
SITE_PHYSICS (locked per site_group):
    nucleus_diameter_nm, num_binding_sites, num_synthetic_sites,
    synthetic_strength, synthetic_cluster_radius_nm,
    bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
    cluster_size_0..4, cluster_pct_0..4, cluster_radius_nm,
    strength_mult_0..4, strength_pct_0..4,
    cluster_correlated_strength

MOTION_PHYSICS (locked per sim_group, free per site_group):
    tf_concentration_nM, sim_dt,
    pct_free_init, pct_inter_init, pct_ns_init,
    d_free, d_free_std, d_inter, d_inter_std, d_ns, d_ns_std,
    k_fi, k_if, k_fns, k_ins, ns_res,
    k_on_prob, res_time_mean, res_time_std, capture_rad_nm,
    enable_cooperativity, coop_strength, coop_hill,
    two_state_mode

RENDER (free in all cases):
    voxel_size_nm, lateral_fwhm_nm, axial_fwhm_nm, molecule_diameter_nm,
    image_dpi, upscale_factor, slice_spacing_nm,
    au_calibration_enabled, au_ref_concentration_nM, au_ref_signal,
    au_background, au_noise_std, label

RENDER_TIMING (free in all cases when cache has step_history):
    num_frames, display_dt, exposure_time_ms
    (see rebin_to_exposures() and validate_rebin_timing())
"""

import os
import hashlib
import pickle
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Level 2: render-timing columns. These can vary across rows in a sim_group
# IF the cache contains a step_history. Constraint: display_dt and
# exposure_time_s must both be integer multiples of the cached sim_dt_s, and
# (steps_per_display * num_display_frames) must not exceed total cached steps.
# ---------------------------------------------------------------------------
RENDER_TIMING_COLUMNS = ('display_dt', 'exposure_time_ms', 'num_frames')


def needs_rebin(row, cached_payload):
    """True if this row's display_dt / exposure / num_frames differ from cache."""
    if cached_payload.get('step_history') is None:
        return False  # cache is Level 1 only — can't rebin
    cached_meta = cached_payload.get('timing_meta', {})
    if not cached_meta:
        return False
    return (
        float(row['display_dt']) != float(cached_meta['display_dt_s'])
        or float(row['exposure_time_ms']) / 1000.0
            != float(cached_meta['exposure_time_s'])
        or int(row['num_frames']) != int(cached_meta['num_display_frames'])
    )


def validate_rebin_timing(sim_dt_s, display_dt_s, exposure_time_s,
                          num_display_frames, total_steps_available):
    """Raise ValueError if requested timing isn't compatible with the cache."""
    eps = 1e-9
    spd_f = display_dt_s / sim_dt_s
    spe_f = exposure_time_s / sim_dt_s
    if abs(spd_f - round(spd_f)) > eps:
        raise ValueError(
            "display_dt (%g s) is not an integer multiple of cached sim_dt "
            "(%g s)." % (display_dt_s, sim_dt_s))
    if abs(spe_f - round(spe_f)) > eps:
        raise ValueError(
            "exposure_time (%g s) is not an integer multiple of cached sim_dt "
            "(%g s)." % (exposure_time_s, sim_dt_s))
    spd = int(round(spd_f))
    spe = int(round(spe_f))
    if spe > spd:
        raise ValueError(
            "exposure_time (%g s, %d steps) exceeds display_dt (%g s, %d "
            "steps). Exposure must fit within one display interval."
            % (exposure_time_s, spe, display_dt_s, spd))
    needed = (num_display_frames - 1) * spd + spe
    if needed > total_steps_available:
        raise ValueError(
            "Re-binning needs %d sim steps but cache only has %d. Reduce "
            "num_frames or shorten display_dt/exposure_time, or extend the "
            "ground-truth simulation." % (needed, total_steps_available))
    return spd, spe


def rebin_to_exposures(step_history, site_strengths,
                       sim_dt_s, display_dt_s, exposure_time_s,
                       num_display_frames):
    """
    Re-bin a flat per-sim-step history into (exposure_mol_data,
    exposure_site_data, exposure_data, steps_per_exposure) using a NEW
    display_dt / exposure_time, without re-running physics.
    """
    spd, spe = validate_rebin_timing(
        sim_dt_s, display_dt_s, exposure_time_s,
        num_display_frames, len(step_history))

    exposure_data = {f: [] for f in range(num_display_frames)}
    exposure_mol_data = {}
    exposure_site_data = {}

    for df in range(num_display_frames):
        start = df * spd
        end = start + spe
        window = step_history[start:end]
        if not window:
            continue
        exposure_data[df] = window
        num_molecules = len(window[0]['pos'])
        all_pos = np.array([s['pos'] for s in window])
        all_states = np.array([s['states'] for s in window])
        all_sites = np.array([s['sites'] for s in window])
        all_occ = np.array([s['occupancy'] for s in window])
        centroid_pos = np.mean(all_pos, axis=0)
        modal_states = stats.mode(all_states, axis=0, keepdims=False)[0]
        exposure_mol_data[df] = pd.DataFrame({
            'trajectory': range(num_molecules),
            'x': centroid_pos[:, 0], 'y': centroid_pos[:, 1],
            'z': centroid_pos[:, 2],
            'state': modal_states, 'frame': df})
        site_centroid = np.mean(all_sites, axis=0)
        any_occupied = np.any(all_occ, axis=0)
        exposure_site_data[df] = (site_centroid, any_occupied)

    return exposure_mol_data, exposure_site_data, exposure_data, spe


# ---------------------------------------------------------------------------
# Column partitions
# ---------------------------------------------------------------------------

# Columns that affect ONLY site layout / strengths / per-site diffusion.
# A row that shares a site_group with another row must agree on every one of
# these. Motion kinetics, TF concentration, timing — NOT in this list.
SITE_PHYSICS_COLUMNS = (
    "nucleus_diameter_nm num_binding_sites "
    "num_synthetic_sites synthetic_strength synthetic_cluster_radius_nm "
    "bs_diff_mean bs_diff_std bs_jiggle_std bs_hurst "
    "cluster_size_0 cluster_size_1 cluster_size_2 cluster_size_3 cluster_size_4 "
    "cluster_pct_0 cluster_pct_1 cluster_pct_2 cluster_pct_3 cluster_pct_4 "
    "cluster_radius_nm "
    "strength_mult_0 strength_mult_1 strength_mult_2 strength_mult_3 strength_mult_4 "
    "strength_pct_0 strength_pct_1 strength_pct_2 strength_pct_3 strength_pct_4 "
    "cluster_correlated_strength"
).split()

# Motion-only physics columns. Locked per sim_group but free per site_group.
MOTION_PHYSICS_COLUMNS = (
    "tf_concentration_nM sim_dt "
    "pct_free_init pct_inter_init pct_ns_init "
    "d_free d_free_std d_inter d_inter_std d_ns d_ns_std "
    "k_fi k_if k_fns k_ins ns_res "
    "k_on_prob res_time_mean res_time_std capture_rad_nm "
    "enable_cooperativity coop_strength coop_hill "
    "two_state_mode"
).split()

# Full physics partition (site + motion). Locked per sim_group.
# NOTE: num_frames, display_dt, exposure_time_ms are Level-2 render-timing
# columns and deliberately omitted — rebin_to_exposures() handles them.
PHYSICS_COLUMNS = SITE_PHYSICS_COLUMNS + MOTION_PHYSICS_COLUMNS


# ---------------------------------------------------------------------------
# Group / seed accessors
# ---------------------------------------------------------------------------

def _nonempty_str(v):
    """True iff v is a meaningful (non-blank, non-NaN) string."""
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    return str(v).strip() not in ('', 'nan', 'None')


def has_repro_columns(row):
    """True if this row opts in to the full (sim_group) caching feature."""
    return _nonempty_str(row.get('sim_group', None)) if hasattr(row, 'get') else False


def has_site_repro_columns(row):
    """
    True if this row opts in to site-layout caching. A row participates if
    it has an explicit site_group OR a sim_group (which auto-derives one).
    """
    if not hasattr(row, 'get'):
        return False
    return (_nonempty_str(row.get('site_group', None))
            or _nonempty_str(row.get('sim_group', None)))


def get_sim_group(row):
    return str(row['sim_group']).strip()


def get_site_group(row):
    """
    Explicit site_group wins; otherwise derive from sim_group. Rows inside
    a sim_group therefore implicitly share a site_group of the same name,
    which lets the sim-leader double as a site-leader.
    """
    if 'site_group' in row and _nonempty_str(row['site_group']):
        return str(row['site_group']).strip()
    if 'sim_group' in row and _nonempty_str(row['sim_group']):
        return str(row['sim_group']).strip()
    raise KeyError("row has neither site_group nor sim_group")


def get_sim_seed(row, fallback_seed=None):
    """
    Read sim_seed from row; fall back to provided default if missing or
    unparseable. Handles the awkward cases a pandas-parsed CSV can produce
    in a mixed int/blank column:
      - float NaN         → fallback
      - empty string ""   → fallback
      - the string "nan"  → fallback
      - "42" / "42.0"     → 42
      - 42   /  42.0      → 42
    Anything else that int() can't coerce → fallback (silently; the caller
    can still seed from seed_base + row_idx).
    """
    if 'sim_seed' not in row:
        return fallback_seed
    v = row['sim_seed']
    # Catch every flavour of missing.
    if v is None:
        return fallback_seed
    if isinstance(v, float) and np.isnan(v):
        return fallback_seed
    if isinstance(v, str):
        s = v.strip()
        if s == '' or s.lower() == 'nan' or s.lower() == 'none':
            return fallback_seed
        try:
            return int(float(s))  # accepts "42" and "42.0"
        except (TypeError, ValueError):
            return fallback_seed
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback_seed


def make_rngs(seed):
    """
    Split a single integer seed into two independent RNG streams, one for
    site setup and one for molecule motion, using numpy's SeedSequence with
    distinct spawn_key values.

    spawn_key convention (numpy requires a tuple of ints, not strings):
        (0,) → site_rng    — site positions, strengths, cluster_D, fBm
        (1,) → mol_rng     — initial positions, state transitions, binding
    If you later add a third stream, use (2,) — never reorder existing keys
    or followers will desynchronize from leaders.

    Returns
    -------
    (site_rng, mol_rng) : tuple of np.random.Generator
        Independent streams. A follower that skips site generation can still
        construct site_rng (and simply not use it) without perturbing its
        mol_rng, because the two streams are derived independently from the
        same root seed.
    """
    if seed is None:
        # Non-reproducible mode: fresh entropy from the OS.
        site_ss = np.random.SeedSequence(pool_size=4, spawn_key=(0,))
        mol_ss  = np.random.SeedSequence(pool_size=4, spawn_key=(1,))
    else:
        # Defensive coercion: pandas/csv can hand us a numpy scalar, a float
        # like 42.0, or even a clean numeric string. SeedSequence only
        # accepts int-like input; coerce here so any type error surfaces at
        # the call site with the actual offending value.
        try:
            seed_int = int(seed)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "make_rngs: could not coerce seed=%r (type %s) to int. "
                "Check the sim_seed column in your CSV for this row — "
                "expected an integer or blank, got something else." % (
                    seed, type(seed).__name__)) from e
        site_ss = np.random.SeedSequence(seed_int, spawn_key=(0,))
        mol_ss  = np.random.SeedSequence(seed_int, spawn_key=(1,))
    return np.random.default_rng(site_ss), np.random.default_rng(mol_ss)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _hash_columns(row, columns):
    """Stable hash over a subset of row columns. NaN/bool normalized."""
    h = hashlib.sha256()
    for col in columns:
        if col in row:
            v = row[col]
            if isinstance(v, float) and np.isnan(v):
                s = 'nan'
            elif isinstance(v, (bool, np.bool_)):
                s = str(int(v))
            else:
                s = repr(v)
            h.update(col.encode())
            h.update(b'=')
            h.update(s.encode())
            h.update(b';')
    return h.hexdigest()[:16]


def physics_hash(row):
    """Hash of all physics columns (site + motion). Used by sim_group."""
    return _hash_columns(row, PHYSICS_COLUMNS)


def site_physics_hash(row):
    """Hash of site-only physics columns. Used by site_group."""
    return _hash_columns(row, SITE_PHYSICS_COLUMNS)


# ---------------------------------------------------------------------------
# Cache I/O — full (sim_group) tier
# ---------------------------------------------------------------------------

def cache_path_for(outdir, sim_group):
    """Absolute path of the full-sim cache pickle for a sim_group."""
    cache_dir = os.path.join(outdir, '_sim_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "%s.pkl" % sim_group)


def cache_exists(outdir, sim_group):
    return os.path.exists(cache_path_for(outdir, sim_group))


def save_sim_cache(outdir, sim_group, payload):
    """
    Save a full ground-truth payload. `payload` must include at minimum:
        exposure_mol_data, exposure_site_data, site_strengths,
        is_synthetic, num_molecules, steps_per_exposure,
        physics_hash, sim_seed
    Atomic via tmp + os.replace — safe under SLURM array races.
    """
    path = cache_path_for(outdir, sim_group)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_sim_cache(outdir, sim_group, expected_physics_hash=None):
    """Load a cached full ground truth, verifying the physics hash."""
    path = cache_path_for(outdir, sim_group)
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if expected_physics_hash is not None:
        cached = payload.get('physics_hash')
        if cached != expected_physics_hash:
            raise ValueError(
                "Physics hash mismatch for sim_group=%r: cache=%s, row=%s. "
                "Two rows in the same sim_group disagree on a physics column. "
                "Either fix the CSV so physics columns match, or give them "
                "different sim_group values." % (sim_group, cached, expected_physics_hash))
    return payload


# ---------------------------------------------------------------------------
# Cache I/O — site (site_group) tier
# ---------------------------------------------------------------------------

def site_cache_path_for(outdir, site_group):
    """Absolute path of the site-layout cache pickle for a site_group."""
    cache_dir = os.path.join(outdir, '_site_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "%s.pkl" % site_group)


def site_cache_exists(outdir, site_group):
    return os.path.exists(site_cache_path_for(outdir, site_group))


def save_site_cache(outdir, site_group, payload):
    """
    Save a site-layout payload. Expected keys:
        sites             — (N, 3) array of site positions
        cluster_ids       — (N,) int array of cluster assignment
        is_synthetic      — (N,) bool array
        site_strengths    — (N,) float array
        cluster_D         — (N,) float per-site diffusion coefficient
        site_physics_hash — str, hash of SITE_PHYSICS_COLUMNS
        sim_seed          — int, root seed used
    Atomic via tmp + os.replace.
    """
    required = {'sites', 'cluster_ids', 'is_synthetic',
                'site_strengths', 'cluster_D',
                'site_physics_hash', 'sim_seed'}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError("site cache payload missing keys: %s" % sorted(missing))
    path = site_cache_path_for(outdir, site_group)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_site_cache(outdir, site_group, expected_site_hash=None):
    """
    Load a cached site layout, verifying the site-physics hash.
    Mismatch ⇒ two rows claim the same site_group but disagree on a
    SITE_PHYSICS column — fix the CSV.
    """
    path = site_cache_path_for(outdir, site_group)
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if expected_site_hash is not None:
        cached = payload.get('site_physics_hash')
        if cached != expected_site_hash:
            raise ValueError(
                "Site-physics hash mismatch for site_group=%r: cache=%s, "
                "row=%s. Two rows in the same site_group disagree on a "
                "site-physics column. Either fix the CSV so those columns "
                "match, or give them different site_group values."
                % (site_group, cached, expected_site_hash))
    return payload


def site_payload_to_precomputed(payload):
    """
    Convert a loaded site-cache payload into the `precomputed_sites` dict
    expected by run_full_simulation (Pass 1). Keys: sites, cluster_ids,
    is_synthetic, site_strengths, cluster_D.
    """
    return {
        'sites': payload['sites'],
        'cluster_ids': payload['cluster_ids'],
        'is_synthetic': payload['is_synthetic'],
        'site_strengths': payload['site_strengths'],
        'cluster_D': payload['cluster_D'],
    }


# ---------------------------------------------------------------------------
# Site-payload construction (leader path)
# ---------------------------------------------------------------------------

def build_site_payload(row, extracted_params, site_rng,
                       sim_seed, site_hash):
    """
    Generate a fresh site-layout payload from physics columns + site_rng.

    Mirrors the site-setup block of run_full_simulation so a site-leader row
    can produce a cacheable payload BEFORE the motion sim runs. The resulting
    payload is then also passed back into run_full_simulation via the
    `precomputed_sites` kwarg — so leader and follower rows both enter the
    motion sim via the precomputed-sites branch and see identical code paths.

    NOTE on RNG consumption ordering: generate_clustered_sites,
    assign_site_strengths, and the cluster_D draw are invoked in the SAME
    order run_full_simulation would invoke them internally. Any future change
    to that ordering must update this function in lockstep.

    KNOWN LIMITATION: site FBM / jiggle dynamics inside the sim loop also
    consume site_rng. A follower that loads from cache has an unconsumed
    site_rng when it enters the motion loop, while a leader has one that's
    already been advanced by setup. Site dynamics therefore DIVERGE between
    leader and follower. Molecule motion (mol_rng) stays aligned. A future
    pass can split site_rng further (setup_rng vs dynamics_rng) to close
    this gap if bit-exact site dynamics become a requirement.

    Parameters
    ----------
    row : pd.Series
        The CSV row (for direct physics columns like bs_diff_mean).
    extracted_params : dict
        Output of sim_engine_fast.extract_batch_row_params(row).
    site_rng : np.random.Generator
        The "site" child RNG from make_rngs(). Consumed in-place.
    sim_seed : int or None
        The root seed (stored in the payload for provenance).
    site_hash : str
        The site_physics_hash string; stored in the payload so followers
        can verify physics agreement.

    Returns
    -------
    dict
        A payload suitable for save_site_cache() / site_payload_to_precomputed().
    """
    # Lazy import: avoids a top-level sim_engine_fast dependency in modules
    # that only want the caching API (e.g. a future CSV validator).
    from sim_engine_fast import generate_clustered_sites, assign_site_strengths

    p = extracted_params
    nucleus_diam_nm = float(row['nucleus_diameter_nm'])
    nucleus_radius_um = (nucleus_diam_nm / 2.0) / 1000.0

    sites, cluster_ids, is_synthetic = generate_clustered_sites(
        int(row['num_binding_sites']), nucleus_radius_um,
        p['cluster_dist'],
        cluster_radius_nm=p['cluster_radius_nm'],
        num_synthetic=p['num_synthetic'],
        synthetic_radius_nm=p['synthetic_radius'],
        site_rng=site_rng)

    site_strengths = assign_site_strengths(
        len(sites), cluster_ids,
        p['strength_multipliers'], p['strength_percentages'],
        is_synthetic, p['synthetic_strength'], p['cluster_correlated'],
        site_rng=site_rng)

    # cluster_D: mirrors the draw inside run_full_simulation exactly.
    # Must come AFTER assign_site_strengths to preserve site_rng ordering.
    unique_clusters = np.unique(cluster_ids)
    num_clusters = len(unique_clusters)
    cluster_D = np.maximum(
        0.005,
        site_rng.normal(float(row['bs_diff_mean']),
                        float(row['bs_diff_std']),
                        num_clusters))

    return {
        'sites': sites,
        'cluster_ids': cluster_ids,
        'is_synthetic': is_synthetic,
        'site_strengths': site_strengths,
        'cluster_D': cluster_D,
        'site_physics_hash': site_hash,
        'sim_seed': sim_seed,
    }