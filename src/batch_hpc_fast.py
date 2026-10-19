#!/usr/bin/env python3
"""
batch_hpc_fast.py - Memory-optimised batch processing (drop-in replacement).

Uses sim_engine_fast.py instead of sim_engine.py for ~35 GB memory savings
on large runs. All options and behaviour are identical to batch_hpc.py.

Two parallelism modes:
  1. MULTIPROCESSING (single node):  python batch_hpc.py params.csv -o results/ -j 16
  2. SLURM ARRAY (multi-node):       sbatch submit_batch.sh   (runs one row per node)

Usage:
  python batch_hpc.py <csv_file> [options]

Options:
  -o, --outdir DIR        Output directory (default: ./batch_output)
  -j, --jobs N            Parallel workers (default: all CPUs)
  --rows RANGE            Process these rows only, e.g. "0-9" or "5" (default: all)
  --save-gt               Save ground truth TIFF stacks
  --save-conv             Save convolved TIFF stacks (default: True)
  --no-conv               Disable convolved output
  --bit-depth {8,16}      TIFF bit depth (default: 8)
  --auto-normalize        Two-phase global brightness normalization
  --headroom FACTOR       Headroom factor for auto-normalize (default: 1.0)
  --seed SEED             Base random seed (row seed = base + row_index,
                          unless the CSV has a 'sim_seed' column which wins)
  --upscale-factor N      Upscale factor for convolved export (default: 1)
  --phase {1,2}           SLURM array phase: 1=simulate+cache, 2=render from cache

REPRODUCIBILITY (optional CSV columns):
  sim_group   string  Rows with the same value share one cached ground truth.
                      First row simulates; later rows in the group skip the
                      sim and re-render with their own PSF/voxel/slice params.
  sim_seed    int     RNG seed for that group. Required for full determinism.
                      Cache lives at <outdir>/_sim_cache/<sim_group>.pkl.
"""

import argparse
import os
import sys
import io
import time
import json
import glob
import pickle
import resource
import numpy as np
import pandas as pd
from PIL import Image
from tifffile import imread as tiff_imread, imwrite as tiff_imwrite
from multiprocessing import Pool, cpu_count


def _mem_gb():
    """Current process RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def _log(msg):
    """Print with timestamp and memory."""
    ts = time.strftime("%H:%M:%S")
    print("[%s | %.1f GB] %s" % (ts, _mem_gb(), msg), flush=True)

from sim_engine_fast import (
    SIDEBAR_DEFAULTS, STRENGTH_MULTIPLIERS_DEFAULT, BRIGHTNESS_VMAX,
    extract_batch_row_params, run_full_simulation, compute_exposure_frames,
    calculate_exposure_grid_fast, render_ground_truth_for_frame,
    render_convolved_slices_fast, calculate_synthetic_centroid_pixels,
    calculate_count_grid, render_count_slices,
    build_site_layout,
)
import sim_repro                       # <-- ADD THIS LINE
import smt_tracking

# ============================================================================
# SINGLE-ROW WORKER
# ============================================================================

def process_single_row(args_tuple):
    """
    Process one CSV row end-to-end: simulate -> grid -> render -> save TIFFs.
    Called by Pool.map(). Returns (row_idx, row_folder, max_intensity, msg).
    """
    row_idx, row_dict, config = args_tuple

    outdir     = config['outdir']
    save_gt    = config['save_gt']
    save_conv  = config['save_conv']
    bit_depth  = config['bit_depth']
    upscale    = config['upscale_factor']
    seed_base  = config['seed']
    auto_norm  = config['auto_normalize']
    save_count     = config.get('save_count', False)
    count_mode     = config.get('count_mode', 'full')
    count_z_proj   = config.get('count_z_proj', 'slab')

    row = pd.Series(row_dict)
    p = extract_batch_row_params(row)

    # ---- Binding tracking / SMT settings --------------------------------
    # Decide whether this row should record binding events at all.
    # Tracking is needed if the user passed --record-binding (tables only)
    # or --smt (tables + single-molecule images). --smt implies tracking,
    # so either flag switches it on

    # Look for a per-row label choice in the CSV column "smt_label"
    # (e.g. "synthetic:1", "random:N", "ids:4", "ids:4,17"). random:N indicates to choose random N 
    # molecules for SMT and then follow them, i.e. random:1 indicates label 1 moelcule randomly
    # and follow. Synthethic:1 means label a molecule that is binding surely. The picked molecule 
    # can also be specified via the ids. 
    want_tracking = bool(config.get('record_binding') or config.get('smt'))
    _row_spec = row.get('smt_label', None)
    row_label_spec = (str(_row_spec).strip()
                      if isinstance(_row_spec, str) and _row_spec.strip()
                      else config.get('smt_label', 'random:1'))
    tracking_result = None

    # === REPRODUCIBILITY / GROUND-TRUTH CACHING ===========================
    # If this row opts in via a sim_group column, check whether another row
    # already simulated the same ground truth. If so, skip the sim entirely
    # and reuse the cached trajectory.
    use_repro = sim_repro.has_repro_columns(row)
    cached_payload = None
    row_phys_hash = None
    row_sim_seed = None
    if use_repro:
        sim_group = sim_repro.get_sim_group(row)
        # Fall back to the existing per-row seed scheme if sim_seed column missing.
        # config['seed'] is the --seed CLI arg (base seed); per existing convention
        # the row-derived seed is base + row_idx.
        fallback = (config['seed'] + row_idx) if config.get('seed') is not None else None
        row_sim_seed = sim_repro.get_sim_seed(row, fallback_seed=fallback)
        row_phys_hash = sim_repro.physics_hash(row)
        if sim_repro.cache_exists(config['outdir'], sim_group):
            _log("[Row %d] sim_group=%s: loading cached ground truth" % (row_idx, sim_group))
            cached_payload = sim_repro.load_sim_cache(
                config['outdir'], sim_group, expected_physics_hash=row_phys_hash)
    # ======================================================================

    row_folder = "Row_%d_%s" % (row_idx, p['row_label']) if p['row_label'] else "Row_%d" % row_idx
    row_outdir = os.path.join(outdir, row_folder)
    os.makedirs(row_outdir, exist_ok=True)

    t0 = time.time()

    def _progress(frame_done, frame_total):
        if frame_done % max(1, frame_total // 5) == 0 or frame_done == frame_total:
            _log("  [Row %d] Frame %d/%d" % (row_idx, frame_done, frame_total))

    _log("[Row %d] Simulating: %s  (n_frames=%s, n_sites=%s, conc=%s nM)" % (
        row_idx, p['row_label'], row['num_frames'], row['num_binding_sites'],
        row['tf_concentration_nM']))

    if cached_payload is not None:
        # ---- CACHE HIT: skip the simulation entirely ----
        site_strengths = cached_payload['site_strengths']
        is_synthetic = cached_payload['is_synthetic']
        n_mol = cached_payload['num_molecules']
        tracking_result = cached_payload.get('tracking')
        if want_tracking and tracking_result is None:
            _log("[Row %d] WARNING: sim_group=%s cache was written without "
                 "binding tracking, so no tracking/SMT output for this row. "
                 "Delete %s/_sim_cache/ and re-run with --record-binding or "
                 "--smt." % (row_idx, sim_group, config['outdir']))

        if sim_repro.needs_rebin(row, cached_payload):
            # Level 2: row's display_dt / exposure_time_ms / num_frames
            # differs from cache. Re-bin per-step history to new timing.
            _log("[Row %d] sim_group=%s: re-binning cached step_history "
                 "(display_dt=%g, exposure_ms=%g, num_frames=%d)" % (
                     row_idx, sim_group, float(row['display_dt']),
                     float(row['exposure_time_ms']), int(row['num_frames'])))
            cached_meta = cached_payload['timing_meta']
            (exposure_mol_data, exposure_site_data,
             exposure_data, spe) = sim_repro.rebin_to_exposures(
                cached_payload['step_history'], site_strengths,
                sim_dt_s=float(cached_meta['sim_dt_s']),
                display_dt_s=float(row['display_dt']),
                exposure_time_s=float(row['exposure_time_ms']) / 1000.0,
                num_display_frames=int(row['num_frames']))
        else:
            exposure_data = cached_payload['exposure_data']
            exposure_mol_data = cached_payload['exposure_mol_data']
            exposure_site_data = cached_payload['exposure_site_data']
            spe = cached_payload['steps_per_exposure']

    else:
        # -------------------------------------------------------------------
        # SITE-LAYOUT CACHING (site_group tier)
        # When the full sim_group cache missed (or this row has no sim_group
        # at all), check whether a site_group cache applies. This is the
        # "same sites, different TF conc / motion params" use case.
        # Explicit site_group wins; otherwise sim_group auto-derives one so a
        # single sim-leader row doubles as site-leader and writes both caches.
        # -------------------------------------------------------------------
        precomputed_sites = None
        use_site_tier = sim_repro.has_site_repro_columns(row)
        if use_site_tier:
            site_group = sim_repro.get_site_group(row)
            site_hash = sim_repro.site_physics_hash(row)
            # If we didn't enter the sim-tier branch we may not have a seed yet.
            if row_sim_seed is None:
                fallback = (config['seed'] + row_idx) if config.get('seed') is not None else None
                row_sim_seed = sim_repro.get_sim_seed(row, fallback_seed=fallback)

            if sim_repro.site_cache_exists(config['outdir'], site_group):
                _log("[Row %d] site_group=%s: loading cached site layout"
                     % (row_idx, site_group))
                site_payload = sim_repro.load_site_cache(
                    config['outdir'], site_group, expected_site_hash=site_hash)
                precomputed_sites = sim_repro.site_payload_to_precomputed(site_payload)
            else:
                # Site-leader. Build the layout deterministically from
                # sim_seed so followers with a different row_idx (which
                # would otherwise shift np.random state) still produce an
                # identical layout on replay.
                # NOTE: this rng is consumed by build_site_layout and must
                # NOT be reused for run_full_simulation — the simulator
                # needs a FRESH site_rng at state S0 so that fBm/jiggle
                # draws match what a follower (which never calls
                # build_site_layout) will produce. See the sim-rng
                # construction below.
                site_rng_setup, _ = sim_repro.make_rngs(row_sim_seed)
                _log("[Row %d] site_group=%s: building site layout (leader)"
                     % (row_idx, site_group))
                precomputed_sites = build_site_layout(
                    num_binding_sites=int(row['num_binding_sites']),
                    nucleus_diameter_nm=float(row['nucleus_diameter_nm']),
                    cluster_distribution=p['cluster_dist'],
                    cluster_radius_nm=p['cluster_radius_nm'],
                    num_synthetic_sites=p['num_synthetic'],
                    synthetic_strength=p['synthetic_strength'],
                    synthetic_cluster_radius_nm=p['synthetic_radius'],
                    strength_multipliers=p['strength_multipliers'],
                    strength_percentages=p['strength_percentages'],
                    cluster_correlated_strength=p['cluster_correlated'],
                    bs_diff_mean=float(row['bs_diff_mean']),
                    bs_diff_std=float(row['bs_diff_std']),
                    capture_radius_nm=float(row['capture_rad_nm']),
                    site_rng=site_rng_setup,
                )
                sim_repro.save_site_cache(config['outdir'], site_group, {
                    **precomputed_sites,
                    'site_physics_hash': site_hash,
                    'sim_seed': row_sim_seed,
                })
                _log("[Row %d] site_group=%s: site-cache SAVED (%d sites)" % (
                    row_idx, site_group, len(precomputed_sites['sites'])))
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # RNGs for the simulator itself. Freshly constructed from sim_seed
        # (not reusing site_rng_setup above), so that:
        #   - the leader's fBm / jiggle draws start at S0, which is exactly
        #     where a follower's site_rng will start.
        #   - site_rng state is INDEPENDENT of mol_rng state, so molecule
        #     count (which varies with TF concentration) can't perturb site
        #     evolution. Without this split, all rows with the same seed
        #     would share a single np.random state, and different TF values
        #     would consume different numbers of draws before the site
        #     jiggle fires, giving divergent site trajectories.
        # For non-repro rows we stay on the legacy np.random path (seeded
        # at the top of process_single_row) by passing both rngs as None.
        # -------------------------------------------------------------------
        if use_repro or use_site_tier:
            sim_site_rng, sim_mol_rng = sim_repro.make_rngs(row_sim_seed)
            sim_seed_arg = None   # engine ignores it when rngs are provided
        else:
            sim_site_rng, sim_mol_rng = None, None
            sim_seed_arg = row_sim_seed
            if seed_base is not None:
                np.random.seed(seed_base + row_idx)

        # ---- Binding / SMT tracker (read-only; does not consume RNG) ----
        tracker = None
        label_seed = (row_sim_seed if row_sim_seed is not None else
                      (seed_base + row_idx if seed_base is not None else None))
        if want_tracking:
            tracker = smt_tracking.BindingTracker(
                label_spec=row_label_spec,
                trajectory_stride=config.get('trajectory_stride', 0),
                label_seed=label_seed)

        exposure_data, n_mol, site_strengths, is_synthetic, spe, step_history = run_full_simulation(
            int(row['num_frames']), int(row['num_binding_sites']),
            float(row['nucleus_diameter_nm']), float(row['tf_concentration_nM']),
            float(row['sim_dt']), float(row['display_dt']),
            float(row['pct_free_init']), float(row['pct_inter_init']), float(row['pct_ns_init']),
            float(row['d_inter']), float(row['d_free']), float(row['d_ns']),
            float(row['d_free_std']), float(row['d_inter_std']), float(row['d_ns_std']),
            float(row['bs_diff_mean']), float(row['bs_diff_std']),
            float(row['bs_jiggle_std']), float(p['bs_hurst']),
            float(row['k_fi']), float(row['k_if']),
            float(row['k_fns']), float(row['k_ins']), float(row['ns_res']),
            p['cluster_dist'],
            float(row['res_time_mean']), float(row['res_time_std']),
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
            progress_callback=_progress,
            sim_seed=sim_seed_arg,
            site_rng=sim_site_rng, mol_rng=sim_mol_rng,
            record_step_history=use_repro,
            precomputed_sites=precomputed_sites,
            tracker=tracker,
        )
        tracking_result = tracker.result() if tracker is not None else None

        _log("[Row %d] Simulation complete: %d molecules" % (row_idx, n_mol))

        exposure_mol_data, exposure_site_data = compute_exposure_frames(exposure_data, site_strengths)

        if use_repro:
            _log("[Row %d] sim_group=%s: saving ground-truth cache (with "
                 "step_history, %d steps)" % (
                     row_idx, sim_group,
                     len(step_history) if step_history is not None else 0))
            sim_repro.save_sim_cache(config['outdir'], sim_group, {
                'exposure_data': exposure_data,
                'exposure_mol_data': exposure_mol_data,
                'exposure_site_data': exposure_site_data,
                'site_strengths': site_strengths,
                'is_synthetic': is_synthetic,
                'num_molecules': n_mol,
                'steps_per_exposure': spe,
                'physics_hash': row_phys_hash,
                'sim_seed': row_sim_seed,
                'tracking': tracking_result,
                # --- Level 2 additions ---
                'step_history': step_history,
                'timing_meta': {
                    'sim_dt_s': float(row['sim_dt']),
                    'display_dt_s': float(row['display_dt']),
                    'exposure_time_s': float(row['exposure_time_ms']) / 1000.0,
                    'num_display_frames': int(row['num_frames']),
                },
            })

    if config.get('save_exposure'):
        _exp_pkl_path = os.path.join(row_outdir, 'exposure_data.pkl')
        with open(_exp_pkl_path, 'wb') as _ef:
            pickle.dump({
                'exposure_data': exposure_data,
                'num_molecules': n_mol,
                'site_strengths': site_strengths,
                'is_synthetic': is_synthetic,
                'params': {
                    'nucleus_diameter_nm': float(row['nucleus_diameter_nm']),
                    'voxel_size_nm':       float(row['voxel_size_nm']),
                    'lateral_fwhm_nm':     p['lateral_fwhm'],
                    'axial_fwhm_nm':       p['axial_fwhm'],
                    'slice_spacing_nm':    float(row.get('slice_spacing_nm', 2499.0)),
                    'num_slices':          p['num_slices'],
                    'upscale_factor':      int(row.get('upscale_factor', 1)),
                    'display_dt':          float(row['display_dt']),
                    'sim_dt':              float(row['sim_dt']),
                    'num_frames':          int(row['num_frames']),
                    'exposure_time_ms':    float(row.get('exposure_time_ms', 50.0)),
                },
            }, _ef, protocol=pickle.HIGHEST_PROTOCOL)
        _log("[Row %d] exposure_data.pkl saved → %s" % (row_idx, _exp_pkl_path))

    display_frames = sorted(exposure_mol_data.keys())
    _log("[Row %d] Exposure frames computed: %d display frames" % (row_idx, len(display_frames)))

    # ---- Binding tables, labelled tracks, SMT images --------------------
    if want_tracking and tracking_result is not None:
        _spd = max(1, int(float(row['display_dt']) / float(row['sim_dt'])))
        _geom = smt_tracking.image_geometry(
            float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
            float(row['slice_spacing_nm']), p.get('num_slices'))
        _lseed = (row_sim_seed if row_sim_seed is not None else
                  (seed_base + row_idx if seed_base is not None else None))
        smt_tracking.write_tracking_outputs(
            row_outdir, tracking_result, exposure_data, row_label_spec,
            smt_tracking.make_label_rng(_lseed), _geom,
            num_frames=int(row['num_frames']), spd=_spd, spe=int(spe),
            lateral_fwhm_nm=p['lateral_fwhm'], axial_fwhm_nm=p['axial_fwhm'],
            near_radius_nm=config.get('near_radius_nm', 250.0),
            write_smt=bool(config.get('smt')),
            smt_params=config.get('smt_params'),
            log=lambda m: _log("[Row %d]%s" % (row_idx, m)))

    # Compute grids and track max intensity
    row_grids = {}
    row_max = 0.0
    for disp_frame in display_frames:
        grid, hs, gs = calculate_exposure_grid_fast(
            exposure_data, disp_frame,
            float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
            p['lateral_fwhm'], p['axial_fwhm'])
        if grid is not None:
            row_grids[disp_frame] = (grid, hs, gs)
            fmax = grid.max()
            if fmax > row_max:
                row_max = fmax

    _log("[Row %d] Grids computed: %d grids, max_intensity=%.6f" % (row_idx, len(row_grids), row_max))

    # Compute count grids if requested
    row_count_grids = {}
    if save_count:
        for disp_frame in display_frames:
            cnt_g, cnt_hs, cnt_gs, cnt_slabs = calculate_count_grid(
                exposure_data, disp_frame,
                float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                float(row['slice_spacing_nm']),
                mode=count_mode)
            if cnt_g is not None:
                row_count_grids[disp_frame] = (cnt_g, cnt_hs, cnt_gs, cnt_slabs)

    # AU calibration dict
    row_au_cal = None
    if p['au_calibration_enabled']:
        row_au_cal = {
            'ref_concentration': p['au_ref_concentration_nM'],
            'ref_signal': p['au_ref_signal'],
            'background': p['au_background'],
            'noise_std': p['au_noise_std'],
            'sim_concentration': float(row['tf_concentration_nM']),
        }
        _log("[Row %d] AU calibration enabled (ref_conc=%.1f, ref_signal=%.1f)" % (
            row_idx, p['au_ref_concentration_nM'], p['au_ref_signal']))

    # Auto-normalize: save cache for Phase 2 rendering
    if auto_norm and not p['au_calibration_enabled']:
        cache_path = os.path.join(row_outdir, '_cache.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'row': row_dict, 'params': p, 'row_folder': row_folder,
                'exposure_mol_data': exposure_mol_data,
                'exposure_site_data': exposure_site_data,
                'grids': row_grids, 'display_frames': display_frames,
                'site_strengths': site_strengths, 'is_synthetic': is_synthetic,
                'row_au_cal': row_au_cal,
                'count_grids': row_count_grids,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = time.time() - t0
        msg = "[Row %d] Phase 1 cached in %.1fs (%d mol, max=%.4f)" % (row_idx, elapsed, n_mol, row_max)
        _log(msg)
        return row_idx, row_folder, row_max, msg

    # Otherwise render immediately
    _log("[Row %d] Rendering TIFFs..." % row_idx)
    # === REPRODUCIBILITY: re-seed RNG before rendering ====================
    # The AU noise draw inside render_convolved_slices_fast (sim_engine_fast.py
    # line ~957) calls np.random.normal, which consumes the global RNG state.
    # For sim_group / site_group rows, re-seed here so all rows sharing a
    # cached ground truth (or site layout) draw identical noise on top of
    # identical input — giving bit-identical convolved TIFFs across rows
    # that share render params. Non-repro rows are unaffected: their seed
    # was already set at the top of process_single_row.
    if (use_repro or sim_repro.has_site_repro_columns(row)) and row_sim_seed is not None:
        np.random.seed(int(row_sim_seed))
    # ======================================================================
    _noise_floor = (p['display_noise_floor_pct'] / 100.0) if p.get('display_noise_enabled') else 0.0
    _noise_std   = (p['display_noise_std_pct']   / 100.0) if p.get('display_noise_enabled') else 0.0
    _render_and_save(row_idx, row, p, row_outdir, display_frames,
                     exposure_mol_data, exposure_site_data, row_grids,
                     site_strengths, is_synthetic, row_au_cal,
                     save_gt, save_conv, bit_depth, upscale,
                     noise_floor_frac=_noise_floor, noise_std_frac=_noise_std,
                     save_count=save_count, row_count_grids=row_count_grids,
                     count_mode=count_mode, count_z_proj=count_z_proj)
    _save_synthetic_centroids(row_idx, row, p, row_outdir, display_frames,
                              exposure_site_data, is_synthetic)
    if config.get('save_positions'):
        _save_molecule_positions(row_idx, row_outdir, display_frames,
                                 exposure_mol_data,
                                 filename=config.get('positions_filename',
                                                     'positions.csv'))
    _build_hyperstacks(row_idx, row_outdir)

    elapsed = time.time() - t0
    msg = "[Row %d] Complete in %.1fs (%d molecules, %d frames -> %s)" % (
        row_idx, elapsed, n_mol, len(display_frames), row_outdir)
    _log(msg)
    return row_idx, row_folder, row_max, msg


# ============================================================================
# RENDERING HELPERS
# ============================================================================

def _render_and_save(row_idx, row, p, row_outdir, display_frames,
                     exposure_mol_data, exposure_site_data, row_grids,
                     site_strengths, is_synthetic, row_au_cal,
                     save_gt, save_conv, bit_depth, upscale,
                     auto_vmax=None,
                     noise_floor_frac=0.0, noise_std_frac=0.0,
                     save_count=False, row_count_grids=None,
                     count_mode='full', count_z_proj='slab'):
    """Render ground truth and/or convolved images, save as TIFF z-stacks."""
    _ns = p.get('num_slices')
    for disp_frame in display_frames:
        mol_data = exposure_mol_data[disp_frame]

        # Ground Truth
        if save_gt:
            sites_pos, site_occ = exposure_site_data.get(disp_frame, (None, None))
            gt_images = render_ground_truth_for_frame(
                mol_data, None, int(disp_frame),
                float(row['nucleus_diameter_nm']), float(row['molecule_diameter_nm']),
                int(row['image_dpi']), float(row['slice_spacing_nm']),
                sites_pos, site_occ,
                site_strengths=site_strengths, strength_range=p['strength_range'],
                is_synthetic=is_synthetic, save_to_disk=False, draw_boundary=False,
                upscale_factor=p['upscale_factor'],
                voxel_size_nm=float(row['voxel_size_nm']),
                num_slices=_ns)
            gt_dir = os.path.join(row_outdir, 'Ground_Truth')
            os.makedirs(gt_dir, exist_ok=True)
            tiff_buf = io.BytesIO()
            gt_images[0].save(tiff_buf, format='TIFF', save_all=True,
                              append_images=gt_images[1:], compression='tiff_deflate')
            tiff_buf.seek(0)
            with open(os.path.join(gt_dir, 'frame_%04d.tiff' % disp_frame), 'wb') as f:
                f.write(tiff_buf.read())

        # Convolved
        if save_conv and disp_frame in row_grids:
            grid, half_span, grid_size = row_grids[disp_frame]
            conv_dict = None

            if row_au_cal is not None:
                conv_dict = render_convolved_slices_fast(
                    grid, float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                    half_span, grid_size, float(row['slice_spacing_nm']),
                    [1.0], upscale_factor=upscale, bit_depth=bit_depth,
                    au_calibration=row_au_cal, num_slices=_ns)
                tag = 'Convolved_AU_%dbit' % bit_depth
            elif auto_vmax is not None:
                conv_dict = render_convolved_slices_fast(
                    grid, float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                    half_span, grid_size, float(row['slice_spacing_nm']),
                    [auto_vmax], upscale_factor=upscale, bit_depth=bit_depth,
                    noise_floor_frac=noise_floor_frac, noise_std_frac=noise_std_frac,
                    num_slices=_ns)
                tag = 'Convolved_auto_%dbit' % bit_depth

            if conv_dict is not None:
                conv_dir = os.path.join(row_outdir, tag)
                os.makedirs(conv_dir, exist_ok=True)
                z_stack = [conv_dict[z][0] for z in sorted(conv_dict.keys())]
                tiff_buf = io.BytesIO()
                z_stack[0].save(tiff_buf, format='TIFF', save_all=True,
                                append_images=z_stack[1:], compression='tiff_deflate')
                tiff_buf.seek(0)
                with open(os.path.join(conv_dir, 'frame_%04d.tiff' % disp_frame), 'wb') as f:
                    f.write(tiff_buf.read())

        # Molecule Count Map
        if save_count and row_count_grids and disp_frame in row_count_grids:
            cnt_g, cnt_hs, cnt_gs, cnt_slabs = row_count_grids[disp_frame]
            _suffix = {'full': '', 'snapshot': '_snapshot', 'centroid': '_centroid'}[count_mode]
            cnt_folder = 'Molecule_Counts_16bit' + _suffix
            if count_z_proj == 'single':
                cnt_folder += '_singlevoxel'
            cnt_dir = os.path.join(row_outdir, cnt_folder)
            os.makedirs(cnt_dir, exist_ok=True)
            cnt_result, _ = render_count_slices(
                cnt_g, float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']),
                cnt_hs, cnt_gs, float(row['slice_spacing_nm']),
                upscale_factor=1, z_projection=count_z_proj)
            z_stack_cnt = [cnt_result[z]['raw'] for z in sorted(cnt_result.keys())]
            if z_stack_cnt:
                tiff_buf = io.BytesIO()
                z_stack_cnt[0].save(tiff_buf, format='TIFF', save_all=True,
                                    append_images=z_stack_cnt[1:], compression='tiff_deflate')
                tiff_buf.seek(0)
                with open(os.path.join(cnt_dir, 'frame_%04d.tiff' % disp_frame), 'wb') as f:
                    f.write(tiff_buf.read())


#  State-code -> readable label mapping. Matches the engine's state encoding:
#    0 = free diffusing
#    1 = intermediate  (only reachable when two_state_mode is False; the
#                       two-state kinetic scheme skips this state entirely)
#    2 = bound to a specific site (REACHABLE IN EITHER MODE — the binding
#        mechanism in _run_physics_step sets state=2 regardless of
#        two_state_mode; only the initial-state draw and the
#        post-unbinding destination are restricted to {0, 3})
#    3 = non-specifically bound
#  `compute_exposure_frames` writes the MODAL state across exposure sub-steps
#  into the 'state' column of exposure_mol_data[frame], so a single label per
#  (molecule, frame) is well-defined.
_STATE_LABELS = {0: 'free', 1: 'intermediate', 2: 'bound', 3: 'nonspecific'}


def _save_molecule_positions(row_idx, row_outdir, display_frames,
                             exposure_mol_data, filename='positions.csv'):
    """
    Write per-molecule positions across all frames to a ViSP-compatible file,
    with an extra column tagging each molecule's binding state.

    Format (tab-delimited, no header, one localization per line):
        x[nm]\ty[nm]\tz[nm]\tintensity\tframe\tstate

    State values are string labels: 'free', 'intermediate', 'bound',
    'nonspecific' (see _STATE_LABELS). ViSP only reads the first five
    columns, so the trailing state column is ignored by ViSP but available
    for downstream analysis (pandas: `pd.read_csv(f, sep='\\t', header=None,
    names=['x','y','z','intensity','frame','state'])`).

    - Coordinates are converted µm -> nm (simulator stores µm internally).
    - Intensity is a placeholder of 1 (this is ground truth, not fit data);
      ViSP only uses it for display weighting.
    - Frames are 1-indexed per SMLM / ViSP convention (internal frames are
      0-indexed).

    The filename's extension is preserved (default .csv with tab delimiter,
    matching the user's 'csv of positions' request; pass 'positions.3d' to
    use the canonical ViSP extension). One file per row, all frames
    concatenated in frame order.
    """
    out_path = os.path.join(row_outdir, filename)
    n_rows_written = 0
    with open(out_path, 'w') as f:
        for disp_frame in display_frames:
            mol_df = exposure_mol_data.get(disp_frame)
            if mol_df is None or len(mol_df) == 0:
                continue
            # µm -> nm
            xs = mol_df['x'].values * 1000.0
            ys = mol_df['y'].values * 1000.0
            zs = mol_df['z'].values * 1000.0
            states = mol_df['state'].values.astype(int)
            visp_frame = int(disp_frame) + 1   # 1-indexed for ViSP
            for x_nm, y_nm, z_nm, st in zip(xs, ys, zs, states):
                label = _STATE_LABELS.get(int(st), 'unknown')
                f.write("%.3f\t%.3f\t%.3f\t%d\t%d\t%s\n" % (
                    x_nm, y_nm, z_nm, 1, visp_frame, label))
                n_rows_written += 1
    _log("[Row %d] Positions written: %s (%d localizations, %d frames)" % (
        row_idx, filename, n_rows_written, len(display_frames)))


def _save_synthetic_centroids(row_idx, row, p, row_outdir, display_frames,
                               exposure_site_data, is_synthetic):
    if is_synthetic is None or not np.any(is_synthetic) or p['num_synthetic'] <= 0:
        return
    centroid_data = []
    for frame_idx in display_frames:
        if frame_idx in exposure_site_data:
            sites_pos, _ = exposure_site_data[frame_idx]
            result = calculate_synthetic_centroid_pixels(
                sites_pos, is_synthetic, [frame_idx],
                float(row['nucleus_diameter_nm']), float(row['voxel_size_nm']))
            centroid_data.extend(result)
    if centroid_data:
        df_c = pd.DataFrame(centroid_data)
        for col in ['Z', 'Y', 'X']:
            df_c[col] = df_c[col].round(3)
        df_c.to_csv(os.path.join(row_outdir, 'synthetic_site_centroid_TZYX.csv'), index=False)


def _build_hyperstacks(row_idx, row_outdir):
    """
    Combine per-frame z-stack TIFFs into a single hyperstack.

    Frame files can be either:
      - grayscale        (Z, Y, X)           → hyperstack (T, Z, Y, X), axes TZYX
      - RGB/RGBA ground  (Z, Y, X, C)        → hyperstack (T, Z, C, Y, X), axes TZCYX
    ImageJ hyperstack convention requires the channel axis between Z and Y,
    which is why we transpose for the 5-D case.

    Each subdir is handled independently — if one fails (e.g. an unexpected
    shape), others still get built and the row finishes.
    """
    for subdir in sorted(os.listdir(row_outdir)):
        sub_path = os.path.join(row_outdir, subdir)
        if not os.path.isdir(sub_path):
            continue
        frame_files = sorted(glob.glob(os.path.join(sub_path, 'frame_*.tiff')))
        if not frame_files:
            continue
        try:
            stacks = []
            for ff in frame_files:
                zstack = tiff_imread(ff)
                if zstack.ndim == 2:
                    zstack = zstack[np.newaxis, ...]   # promote (Y,X) → (1,Y,X)
                stacks.append(zstack)
            hyperstack = np.stack(stacks, axis=0)

            if hyperstack.ndim == 4:
                # (T, Z, Y, X) — grayscale convolved / count maps.
                axes = 'TZYX'
                out_arr = hyperstack
            elif hyperstack.ndim == 5:
                # (T, Z, Y, X, C) — RGB(A) ground truth. ImageJ wants
                # channels between Z and Y, so transpose to (T, Z, C, Y, X).
                out_arr = np.transpose(hyperstack, (0, 1, 4, 2, 3))
                axes = 'TZCYX'
            else:
                _log("[Row %d] Skipping hyperstack for %s: unexpected ndim=%d "
                     "(shape=%s)" % (row_idx, subdir, hyperstack.ndim,
                                     hyperstack.shape))
                continue

            out_name = '%s_Hyperstack.tif' % subdir
            out_path = os.path.join(row_outdir, out_name)
            _meta = {'axes': axes}
            if out_arr.dtype == np.uint16 and axes == 'TZYX':
                # 16-bit data rarely spans 0-65535; without a stored display
                # range most viewers show it as black. Data is unchanged.
                _lo = float(np.percentile(out_arr, 0.5))
                _hi = float(out_arr.max())
                _meta.update(min=_lo, max=_hi if _hi > _lo else _lo + 1.0)
            tiff_imwrite(out_path, out_arr, imagej=True, metadata=_meta)
            _log("[Row %d] Hyperstack: %s (%s, axes=%s)" % (
                row_idx, out_name,
                'x'.join(str(s) for s in out_arr.shape), axes))
        except Exception as e:
            # Don't let one malformed subdir kill the others — in particular,
            # a failed ground-truth stack should not prevent the molecule-
            # count hyperstack from being written.
            _log("[Row %d] Hyperstack FAILED for %s: %s" % (row_idx, subdir, e))
            continue


# ============================================================================
# AUTO-NORMALIZE PHASE 2
# ============================================================================

def render_cached_row(args_tuple):
    """Phase 2: re-render a cached row using the global auto_vmax."""
    row_idx, cache_path, config = args_tuple
    _log("[Row %d] Phase 2: loading cache (%s)" % (row_idx, cache_path))
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    row = pd.Series(cache['row'])
    p = cache['params']
    row_outdir = os.path.dirname(cache_path)
    auto_vmax = config['auto_vmax']

    _log("[Row %d] Phase 2: rendering TIFFs (vmax=%.4f)" % (row_idx, auto_vmax))
    _p_noise_floor = (p['display_noise_floor_pct'] / 100.0) if p.get('display_noise_enabled') else 0.0
    _p_noise_std   = (p['display_noise_std_pct']   / 100.0) if p.get('display_noise_enabled') else 0.0
    _render_and_save(row_idx, row, p, row_outdir, cache['display_frames'],
                     cache['exposure_mol_data'], cache['exposure_site_data'],
                     cache['grids'], cache['site_strengths'], cache['is_synthetic'],
                     cache['row_au_cal'],
                     config['save_gt'], config['save_conv'], config['bit_depth'],
                     config['upscale_factor'],
                     auto_vmax=auto_vmax,
                     noise_floor_frac=_p_noise_floor, noise_std_frac=_p_noise_std,
                     save_count=config.get('save_count', False),
                     row_count_grids=cache.get('count_grids', {}),
                     count_mode=config.get('count_mode', 'full'),
                     count_z_proj=config.get('count_z_proj', 'slab'))
    _save_synthetic_centroids(row_idx, row, p, row_outdir, cache['display_frames'],
                              cache['exposure_site_data'], cache['is_synthetic'])
    if config.get('save_positions'):
        _save_molecule_positions(row_idx, row_outdir, cache['display_frames'],
                                 cache['exposure_mol_data'],
                                 filename=config.get('positions_filename',
                                                     'positions.csv'))
    _build_hyperstacks(row_idx, row_outdir)
    os.remove(cache_path)
    _log("[Row %d] Phase 2 complete (vmax=%.4f, output=%s)" % (row_idx, auto_vmax, row_outdir))
    return row_idx


# ============================================================================
# MAIN
# ============================================================================

def parse_rows(rows_str, total_rows):
    """Parse '5' -> [5], '0-9' -> [0..9], None -> all."""
    if rows_str is None:
        return list(range(total_rows))
    if '-' in rows_str:
        start, end = rows_str.split('-', 1)
        return list(range(int(start), min(int(end) + 1, total_rows)))
    return [int(rows_str)]


def main():
    parser = argparse.ArgumentParser(
        description='Parallel batch simulation for HPC',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_file', help='Input CSV (one row per simulation)')
    parser.add_argument('-o', '--outdir', default='./batch_output')
    parser.add_argument('-j', '--jobs', type=int, default=None,
                        help='Parallel workers (default: all CPUs)')
    parser.add_argument('--rows', default=None, help='Row range: "5" or "0-9"')
    parser.add_argument('--save-gt', action='store_true')
    parser.add_argument('--save-conv', action='store_true', default=True)
    parser.add_argument('--no-conv', action='store_true')
    parser.add_argument('--bit-depth', type=int, choices=[8, 16], default=8)
    parser.add_argument('--auto-normalize', action='store_true',
                        help='Two-phase global brightness normalization')
    parser.add_argument('--headroom', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--upscale-factor', type=int, default=1)
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='SLURM array phase: 1=simulate+cache, 2=render from cache')
    parser.add_argument('--save-count', action='store_true',
                        help='Save per-voxel molecule count grids as 16-bit TIFF z-stacks')
    parser.add_argument('--count-mode', choices=['full', 'snapshot', 'centroid'],
                        default='full',
                        help='Count mode for molecule count maps. '
                             '"full" (default): accumulate all sub-steps in the exposure window '
                             '(a molecule present for N steps is counted N times). '
                             '"snapshot": use only the last physics sub-step (one count per molecule). '
                             '"centroid": use the mean position across all sub-steps '
                             '(one count per molecule; matches positions.txt).')
    parser.add_argument('--count-z-proj', choices=['slab', 'single'], default='slab',
                        help='Z-slice projection for count maps: '
                             '"slab" sums voxels within ±(slice_spacing/2) of each z-plane (default); '
                             '"single" uses only the single voxel at the exact z-plane.')
    parser.add_argument('--save-positions', action='store_true',
                        help='Save per-molecule ground-truth positions per row '
                             'as a tab-delimited ViSP-compatible file '
                             '(columns: x[nm] y[nm] z[nm] intensity frame). '
                             'One file per row, all frames concatenated.')
    parser.add_argument('--positions-filename', default='positions.csv',
                        help='Output filename for --save-positions '
                             '(default: positions.csv). Use positions.3d for '
                             'the canonical ViSP extension. Content is '
                             'tab-delimited regardless of extension.')
    parser.add_argument('--save-exposure', action='store_true',
                        help='Save exposure_data + simulation params to '
                             'exposure_data.pkl per row. Upload this file to '
                             'the Streamlit GUI to run FRAP on HPC results '
                             'without re-simulating.')
    parser.add_argument('--no-au-calibration', action='store_true',
                        help='Disable AU calibration for all rows, overriding '
                             'the CSV and SIDEBAR_DEFAULTS (which default to '
                             'enabled). Use this to render with vmax '
                             'normalization instead of absolute AU units.')
    # ---- Binding tracking / single-molecule tracking (SMT) ----
    g = parser.add_argument_group('binding tracking / single-molecule tracking')
    g.add_argument('--record-binding', action='store_true',
                   help='Record every binding / state change of every molecule '
                        '(Tracking/ folder: state_events, site_dwells, '
                        'synthetic_dwells, synthetic_visits, labelled tracks).')
    g.add_argument('--smt', action='store_true',
                   help='Also render SMT images (SMT_16bit/) showing ONLY the '
                        'labelled molecule(s). Implies --record-binding.')
    g.add_argument('--smt-label', default='random:1',
                   help='Which molecules are labelled: "random:N", "ids:3,17" '
                        'or "synthetic:N" (N molecules that bound the '
                        'synthetic array). A per-row CSV column "smt_label" '
                        'overrides this. Default: random:1')
    g.add_argument('--smt-photons', type=float, default=300.0,
                   help='Photons per labelled molecule per exposure, in focus '
                        '(default 300)')
    g.add_argument('--smt-background', type=float, default=10.0,
                   help='Background photons per pixel per exposure (default 10)')
    g.add_argument('--smt-gain', type=float, default=1.0,
                   help='Camera gain, ADU per photoelectron (default 1)')
    g.add_argument('--smt-offset', type=float, default=100.0,
                   help='Camera baseline in ADU (default 100)')
    g.add_argument('--smt-read-noise', type=float, default=1.5,
                   help='Read noise, e- rms (default 1.5)')
    g.add_argument('--smt-no-noise', action='store_true',
                   help='Noise-free SMT images (expected values only)')
    g.add_argument('--smt-focus', choices=['slices', 'synthetic'],
                   default='slices',
                   help='"slices" (default): SMT z-planes follow the row\'s '
                        'slice settings. "synthetic": one focal plane at the '
                        'depth of the synthetic array (at t = 0).')
    g.add_argument('--near-radius-nm', type=float, default=250.0,
                   help='Radius around synthetic sites for '
                        'synthetic_neighborhood.csv (default 250 nm)')
    g.add_argument('--trajectory-stride', type=int, default=0,
                   help='Also save positions/states/bound site of ALL '
                        'molecules every N physics steps (0 = off; large!)')

    args = parser.parse_args()

    if args.no_conv:
        args.save_conv = False

    batch_df = pd.read_csv(args.csv_file)
    for key, val in SIDEBAR_DEFAULTS.items():
        if key not in batch_df.columns:
            batch_df[key] = val
    if args.no_au_calibration:
        batch_df['au_calibration_enabled'] = False

    row_indices = parse_rows(args.rows, len(batch_df))
    os.makedirs(args.outdir, exist_ok=True)

    config = {
        'outdir': args.outdir, 'save_gt': args.save_gt, 'save_conv': args.save_conv,
        'bit_depth': args.bit_depth,
        'upscale_factor': args.upscale_factor, 'seed': args.seed,
        'auto_normalize': args.auto_normalize,
        'save_count': args.save_count,
        'count_mode': args.count_mode,
        'count_z_proj': args.count_z_proj,
        'save_positions': args.save_positions,
        'positions_filename': args.positions_filename,
        'save_exposure': args.save_exposure,
        'record_binding': args.record_binding or args.smt,
        'smt': args.smt,
        'smt_label': args.smt_label,
        'smt_params': {'photons': args.smt_photons,
                       'background': args.smt_background,
                       'gain': args.smt_gain, 'offset': args.smt_offset,
                       'read_noise': args.smt_read_noise,
                       'add_noise': not args.smt_no_noise,
                       'focus': args.smt_focus},
        'near_radius_nm': args.near_radius_nm,
        'trajectory_stride': args.trajectory_stride,
    }

    # ==================================================================
    # SLURM ARRAY MODE: --phase 1 or --phase 2 (one row per task)
    # ==================================================================
    if args.phase == 1:
        assert len(row_indices) == 1, \
            "--phase 1 expects a single --rows value (got %d)" % len(row_indices)
        idx = row_indices[0]

        print("=" * 60)
        print("Phase 1 - Row %d / %d" % (idx, len(batch_df)))
        print("=" * 60)

        result = process_single_row((idx, batch_df.iloc[idx].to_dict(), config))
        row_idx, row_folder, row_max, msg = result

        # Write max intensity for Phase 2 aggregation
        max_path = os.path.join(args.outdir, row_folder, '_max.txt')
        with open(max_path, 'w') as f:
            f.write('%.10f\n' % row_max)

        print(msg)
        return

    if args.phase == 2:
        assert args.auto_normalize, "--phase 2 requires --auto-normalize"
        assert len(row_indices) == 1, \
            "--phase 2 expects a single --rows value (got %d)" % len(row_indices)
        idx = row_indices[0]

        # Compute global max from all _max.txt files
        max_files = glob.glob(os.path.join(args.outdir, 'Row_*', '_max.txt'))
        if not max_files:
            print("ERROR: No _max.txt files found in %s" % args.outdir,
                  file=sys.stderr)
            sys.exit(1)
        global_max = 0.0
        for mf in max_files:
            with open(mf) as f:
                val = float(f.read().strip())
                if val > global_max:
                    global_max = val
        auto_vmax = global_max * args.headroom

        print("=" * 60)
        print("Phase 2 - Row %d / %d" % (idx, len(batch_df)))
        print("  Global max: %.6f   Auto vmax: %.6f" % (global_max, auto_vmax))
        print("=" * 60)

        # Save calibration JSON atomically (multiple SLURM tasks write identical
        # content; os.replace is atomic on POSIX so no task sees a partial file).
        cal_path = os.path.join(args.outdir, 'brightness_calibration.json')
        cal_tmp = cal_path + '.tmp.%d' % os.getpid()
        with open(cal_tmp, 'w') as f:
            json.dump({'global_max': global_max, 'headroom': args.headroom,
                       'auto_vmax': auto_vmax, 'bit_depth': args.bit_depth},
                      f, indent=2)
        os.replace(cal_tmp, cal_path)

        # Find this row's cache
        p = extract_batch_row_params(batch_df.iloc[idx])
        row_folder = ("Row_%d_%s" % (idx, p['row_label'])
                      if p['row_label'] else "Row_%d" % idx)
        cache_path = os.path.join(args.outdir, row_folder, '_cache.pkl')
        if not os.path.exists(cache_path):
            # AU-calibrated rows already rendered during Phase 1 (no cache)
            print("[Row %d] Already rendered (AU calibration), skipping Phase 2" % idx)
            return

        config2 = dict(config)
        config2['auto_vmax'] = auto_vmax
        render_cached_row((idx, cache_path, config2))

        # Clean up _max.txt
        max_path = os.path.join(args.outdir, row_folder, '_max.txt')
        if os.path.exists(max_path):
            os.remove(max_path)

        print("Phase 2 complete for row %d" % idx)
        return

    # ==================================================================
    # LEGACY MODE: no --phase flag (multiprocessing on a single node)
    # ==================================================================
    n_workers = args.jobs if args.jobs else min(cpu_count(), len(row_indices))

    print("=" * 60)
    print("Batch HPC Simulation")
    print("  CSV:        %s (%d total rows)" % (args.csv_file, len(batch_df)))
    print("  Rows:       %d-%d (%d rows)" % (row_indices[0], row_indices[-1], len(row_indices)))
    print("  Workers:    %d" % n_workers)
    print("  Output:     %s" % args.outdir)
    print("  GT:         %s" % args.save_gt)
    print("  Convolved:  %s" % args.save_conv)
    print("  Bit depth:  %d" % args.bit_depth)
    print("  Auto-norm:  %s (headroom=%.1f)" % (args.auto_normalize, args.headroom))
    print("  Seed:       %s" % args.seed)
    print("  Upscale:    %d" % args.upscale_factor)
    print("  Mol counts: %s (mode=%s, z_proj=%s)" % (args.save_count, args.count_mode, args.count_z_proj))
    print("  Positions:  %s%s" % (args.save_positions,
                                  (" -> %s" % args.positions_filename) if args.save_positions else ""))
    print("  Exposure pkl: %s" % args.save_exposure)
    print("  Tracking:   %s%s" % (config['record_binding'],
          ("  (SMT images, default label=%s; CSV smt_label overrides)" % args.smt_label) if args.smt else ""))
    print("=" * 60)

    work_items = [(idx, batch_df.iloc[idx].to_dict(), config) for idx in row_indices]
    t_start = time.time()

    if args.auto_normalize and args.save_conv:
        # ---- TWO-PHASE AUTO-NORMALIZE ----
        print("\n--- Phase 1: Simulate all rows & compute max intensity ---")
        if n_workers > 1:
            with Pool(n_workers) as pool:
                results = pool.map(process_single_row, work_items)
        else:
            results = [process_single_row(item) for item in work_items]

        global_max = max(r[2] for r in results)
        auto_vmax = global_max * args.headroom
        print("\nGlobal max: %.6f  Auto vmax: %.6f" % (global_max, auto_vmax))

        with open(os.path.join(args.outdir, 'brightness_calibration.json'), 'w') as f:
            json.dump({'global_max': global_max, 'headroom': args.headroom,
                       'auto_vmax': auto_vmax, 'bit_depth': args.bit_depth}, f, indent=2)

        print("\n--- Phase 2: Render all rows with global vmax ---")
        config2 = dict(config)
        config2['auto_vmax'] = auto_vmax
        render_items = []
        for idx in row_indices:
            p = extract_batch_row_params(batch_df.iloc[idx])
            rf = "Row_%d_%s" % (idx, p['row_label']) if p['row_label'] else "Row_%d" % idx
            cp = os.path.join(args.outdir, rf, '_cache.pkl')
            if os.path.exists(cp):
                render_items.append((idx, cp, config2))
        if render_items:
            if n_workers > 1:
                with Pool(n_workers) as pool:
                    pool.map(render_cached_row, render_items)
            else:
                for item in render_items:
                    render_cached_row(item)
    else:
        # ---- SINGLE-PHASE (manual brightness) ----
        if n_workers > 1:
            with Pool(n_workers) as pool:
                results = pool.map(process_single_row, work_items)
        else:
            results = [process_single_row(item) for item in work_items]

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("All done! %d rows in %.1fs" % (len(row_indices), elapsed))
    print("Output: %s" % os.path.abspath(args.outdir))
    print("=" * 60)


if __name__ == '__main__':
    main()
