#!/usr/bin/python3
"""
Fruit Fly Optimisation (FOA) – N-dimensional parameter search.

The algorithm moves a swarm of sample points ("flies") through an N-dimensional
parameter space guided by two forces, computed in normalised [0, 1] coordinates
along each axis:

  Attraction  – toward evaluated points with a higher cost score.
  Repulsion   – away from evaluated points with a lower cost score.

Forces are accumulated as Cartesian vectors (not angles), which generalises
cleanly beyond 2-D.  Both are weighted by score difference and distance
(attraction saturates via mod_sigmoid; repulsion decays exponentially).
An adaptive random noise term is added at each step — largest when the directed
move is small — which helps flies escape local optima.

Typical usage
-------------
    Op = OptimiseN(results, current_round, optimiser_settings)
    Op.go()   # updates Op.next_run with next-round input points
"""

import os
import math
import pickle

import numpy as np

try:
    import matplotlib as _mpl  # noqa: F401 – presence check only
    del _mpl
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────────────────────
# Dict helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_subset_dict(small, large):
    """Return True if every key/value in *small* also appears in *large*."""
    return all(key in large and large[key] == small[key] for key in small)


def _find_subset_dict(needle, haystack):
    """Return the first dict in *haystack* that contains all of *needle*'s key/values, or None."""
    for d in haystack:
        if _is_subset_dict(needle, d):
            return d
    return None


def _fill_in_dict_list(sparse_list, full_list):
    """
    Enrich each dict in *sparse_list* in-place by merging the first matching
    dict from *full_list* (matched via subset equality).  Returns *sparse_list*.
    """
    for item in sparse_list:
        match = _find_subset_dict(item, full_list)
        if match is not None:
            item.update(match)
    return sparse_list


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mod_sigmoid(x, limit):
    """
    Sigmoid scaled so the output ∈ (-limit, limit) and f(0) = 0.
    Saturates cumulative force magnitudes so no single interaction dominates.
    Works on scalars or numpy arrays (via np.exp).
    """
    return 2 * limit / (1 + np.exp(-2 * x / limit)) - limit


def _mod_flipped_sigmoid(x, steepness):
    """
    Decreasing sigmoid: f(0) = 1, f(∞) → 0.
    Slows flies that already sit near the optimum (high cost score).
    Works on scalars or numpy arrays (via np.exp).
    """
    return (-2 / (1 + np.exp(-steepness * x))) + 2


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_normalisation(all_points, params, bounds):
    """
    Return ``(origins, scales)`` dicts mapping each parameter name to its
    origin and scale so that coordinates map to ~[0, 1] along every axis.

    Prefer *bounds* when provided; otherwise derives the range from *all_points*.
    Normalising prevents axes with very different physical ranges from
    distorting the direction of the net force vector.
    """
    origins, scales = {}, {}
    for p in params:
        if bounds and p in bounds:
            lo, hi = bounds[p]
        else:
            vals = [pt[p] for pt in all_points if p in pt]
            lo, hi = (min(vals), max(vals)) if len(vals) > 1 else (0.0, 1.0)
        origins[p] = lo
        scales[p] = max(hi - lo, 1e-9)
    return origins, scales


def _clamp_to_bounds(val, param, bounds):
    """Clamp *val* to [min, max] for *param* in *bounds*, or to ≥ 0.01 if no bounds given."""
    if bounds and param in bounds:
        lo, hi = bounds[param]
        return max(lo, min(hi, val))
    return max(0.01, val)


def _round_sig(val, sig=5):
    """Round *val* to *sig* significant digits (0 is returned unchanged)."""
    if val == 0:
        return 0.0
    return round(val, -int(math.floor(math.log10(abs(val)))) + (sig - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation step
# ─────────────────────────────────────────────────────────────────────────────

def crawl_optimise(current_points, all_points, crawl_speed, input_params,
                   z, bounds=None, noise_scale=0.05, repulsion_scale=0.5,
                   diversity_scale=0.05, diversity_radius=0.02, rng=None):
    """
    Compute one FOA step for every fly in *current_points* (N-dimensional).

    For each fly, accumulates attraction vectors (toward higher-scoring points),
    repulsion vectors (away from lower-scoring points), and a small always-on
    "diversity" repulsion (independent of cost, guarding against (near-)duplicate
    proposals — equal-cost points otherwise exert zero force on each other and
    can converge onto the same coordinates), then moves each fly one step along
    the net resultant direction plus adaptive random noise. All pairwise force
    terms are computed as vectorised numpy operations across every
    (fly, evaluated-point) pair at once, rather than a per-pair Python loop.
    All geometry is computed in normalised [0, 1] space. Results are
    denormalised and clamped to *bounds* before being returned.

    Args:
        current_points:   list of dicts – flies to move this step.
        all_points:       list of dicts – all evaluated points (including current).
        crawl_speed:      base step size in normalised space (0–1).
        input_params:     list of N parameter names to optimise over.
        z:                key name of the cost/efficiency field.
        bounds:           dict mapping param → [min, max] in physical units.
        noise_scale:      random exploration amplitude in normalised space.
        repulsion_scale:  fraction of attraction strength applied as repulsion.
        diversity_scale:  strength of a small always-on, cost-independent
                          repulsion between any two points, guarding against
                          (near-)duplicate proposals that cost-based
                          attraction/repulsion can't prevent (equal-cost
                          points exert zero force on each other otherwise).
        diversity_radius: normalised distance beyond which diversity_scale
                          decays to ~0 — keep this small so it only affects
                          points that are nearly coincident.
        rng:              numpy Generator used for all randomness (falls back
                          to a fresh, unseeded ``np.random.default_rng()`` if
                          not given). Pass a seeded Generator for reproducible
                          runs.

    Returns:
        List of dicts with updated parameter values for each fly, or [] on error.
    """
    if not all_points or not current_points:
        if not all_points:
            print("crawl_optimise: all_points is empty")
        return []

    rng = rng if rng is not None else np.random.default_rng()
    params = input_params
    n_dims = len(params)
    origins, scales = _compute_normalisation(all_points, params, bounds)

    def _coords(points):
        return np.array([[(pt[p] - origins[p]) / scales[p] for p in params] for pt in points], dtype=float)

    all_coords = _coords(all_points)                                    # (P, D)
    ref_coords = _coords(current_points)                                # (F, D)
    all_cost = np.array([p[z] for p in all_points], dtype=float)        # (P,)
    ref_cost = np.array([p[z] for p in current_points], dtype=float)    # (F,)

    # Identity (not value) match, mirroring the original "cmp_orig is ref_orig" skip.
    all_ids = np.array([id(p) for p in all_points])
    ref_ids = np.array([id(p) for p in current_points])
    self_mask = ref_ids[:, None] == all_ids[None, :]                    # (F, P)

    diff = all_coords[None, :, :] - ref_coords[:, None, :]              # (F, P, D): cmp - ref
    dist = np.linalg.norm(diff, axis=2)                                 # (F, P)
    safe_dist = np.where(dist > 1e-12, dist, 1.0)
    direction = np.where(dist[..., None] > 1e-12, diff / safe_dist[..., None], 0.0)

    cost_diff = all_cost[None, :] - ref_cost[:, None]                   # (F, P): cmp - ref
    attract_w = np.clip(cost_diff, 0.0, 1.0)
    repulse_w = np.clip(-cost_diff, 0.0, 1.0)

    attract_mag = crawl_speed * _mod_sigmoid(dist, crawl_speed) * attract_w
    repel_mag = repulsion_scale * crawl_speed * np.exp(-dist / max(crawl_speed, 1e-9)) * repulse_w
    diversity_mag = diversity_scale * np.exp(-(dist / max(diversity_radius, 1e-9)) ** 2)

    # Diversity pushes away from cmp (opposite of `direction`); pairs that are
    # exactly coincident (and aren't the same point) get a random push
    # instead, since cost-based direction is undefined at zero distance.
    div_dir = -direction
    dup_idx = np.argwhere((dist < 1e-9) & ~self_mask)
    if len(dup_idx):
        rand_raw = rng.standard_normal((len(dup_idx), n_dims))
        rand_norm = np.linalg.norm(rand_raw, axis=1, keepdims=True)
        div_dir[dup_idx[:, 0], dup_idx[:, 1]] = np.where(
            rand_norm > 1e-12, rand_raw / np.where(rand_norm > 1e-12, rand_norm, 1.0), 0.0)

    pair_force = (attract_mag - repel_mag)[..., None] * direction + diversity_mag[..., None] * div_dir
    pair_force = np.where(self_mask[..., None], 0.0, pair_force)
    net_force = pair_force.sum(axis=1)                                  # (F, D)

    net_mag = np.linalg.norm(net_force, axis=1)                         # (F,)
    safe_net_mag = np.where(net_mag > 1e-12, net_mag, 1.0)
    net_dir = np.where(net_mag[:, None] > 1e-12, net_force / safe_net_mag[:, None], 0.0)

    # Saturate step size and slow flies already near the optimum.
    step = _mod_sigmoid(net_mag, 4 * crawl_speed) * _mod_flipped_sigmoid(ref_cost, 4 / crawl_speed)

    # Adaptive random noise: largest when the directed step is near zero.
    noise_mag = noise_scale * np.exp(-step / max(noise_scale, 1e-9))
    noise_raw = rng.standard_normal((len(current_points), n_dims))
    noise_norm = np.linalg.norm(noise_raw, axis=1, keepdims=True)
    noise_dir = np.where(noise_norm > 1e-12, noise_raw / np.where(noise_norm > 1e-12, noise_norm, 1.0), 0.0)

    new_coords = ref_coords + step[:, None] * net_dir + noise_mag[:, None] * noise_dir  # (F, D)

    return [
        {p: _round_sig(_clamp_to_bounds(new_coords[i, j] * scales[p] + origins[p], p, bounds))
         for j, p in enumerate(params)}
        for i in range(len(current_points))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_pckl(filename, obj):
    """Pickle *obj* to *filename* relative to the current working directory."""
    path = os.path.join(os.getcwd(), filename)
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as error:
        print(f"Could not save state to {filename}: {error}")


def _load_pckl(filename):
    """Unpickle and return the object at *filename* (relative to cwd), or None if unavailable."""
    path = os.path.join(os.getcwd(), filename)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_search_history(z_param, history_file=None, save_path=None, show=True):
    """
    Plot the cost score of every fly over rounds.

    X-axis: round number.
    Y-axis: cost (z_param) for each fly.
    Each fly is drawn as a separate line with circular markers.

    Args:
        z_param:      cost/efficiency key to plot on the y-axis.
        history_file: path to the history pickle (default: search_history.pckl).
        save_path:    output PNG path (default: search_history.png in cwd).
        show:         call plt.show() after saving.

    Returns:
        (fig, ax) on success, or None if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed – cannot visualize. Run: pip install matplotlib")
        return None

    import matplotlib.pyplot as plt

    history = _load_pckl(history_file or "search_history.pckl")
    if not history:
        print("No search history found at", history_file)
        return None

    n_flies = max(len(entry['points']) for entry in history)

    # Use tab10 for up to 10 flies, cycling beyond that.
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_flies)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for fly_i in range(n_flies):
        rounds, zvals = [], []
        for entry in history:
            if fly_i < len(entry['points']):
                pt = entry['points'][fly_i]
                if z_param in pt:
                    rounds.append(entry['round'])
                    zvals.append(pt[z_param])
        if rounds:
            ax.plot(rounds, zvals,
                    marker='o', color=colors[fly_i],
                    linewidth=1.5, markersize=5,
                    label=f'Fly {fly_i + 1}')

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel(z_param, fontsize=12)
    ax.set_title(f'FOA Progress: {z_param} per Round', fontsize=13)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # type: ignore[attr-defined]
    ax.legend(title='Fly', loc='best', framealpha=0.8,
              ncol=max(1, n_flies // 10))
    plt.tight_layout()

    out = save_path or os.path.join(os.getcwd(), 'search_history.png')
    plt.savefig(out, dpi=150)
    print("Saved to", out)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def visualize_search_trajectories_2d(x_param, y_param, z_param,
                                      history_file=None, save_path=None, show=True):
    """
    Plot how optimiser flies moved through the search space across rounds.

    Only meaningful for a 2-D search (one plot axis per parameter) — callers
    with more than two ``input_parameters`` should pick which two to project
    onto *x_param*/*y_param*.

    Background squares show every evaluated point coloured by cost (viridis).
    Coloured dots (plasma: early rounds blue → late rounds red) mark each fly's
    position per round; black lines and red arrows trace the trajectory.

    Args:
        x_param, y_param: parameter names for the plot axes.
        z_param:          cost/efficiency key used for the background colormap.
        history_file:     path to the history pickle (default: search_history.pckl).
        save_path:        output PNG path (default: search_history.png in cwd).
        show:             call plt.show() after saving.

    Returns:
        (fig, ax) on success, or None if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed – cannot visualize. Run: pip install matplotlib")
        return None

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch

    history = _load_pckl(history_file or "search_history.pckl")
    if not history:
        print("No search history found at", history_file)
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Background: all evaluated points coloured by cost ────────────────────
    bg_xs, bg_ys, bg_zs = [], [], []
    for entry in history:
        for pt in entry['points']:
            if x_param in pt and y_param in pt and z_param in pt:
                bg_xs.append(pt[x_param])
                bg_ys.append(pt[y_param])
                bg_zs.append(pt[z_param])
    if bg_xs:
        sc = ax.scatter(bg_xs, bg_ys, c=bg_zs, cmap='viridis',
                        alpha=0.6, s=180, marker='s', zorder=1)  # type: ignore[arg-type]
        plt.colorbar(sc, ax=ax, label=z_param)

    # ── Per-round colours for trajectory dots (early=blue, late=red) ─────────
    n_rounds = len(history)
    sm = ScalarMappable(cmap='plasma', norm=Normalize(0.1, 0.9))
    round_colors = [sm.to_rgba(v) for v in np.linspace(0.1, 0.9, n_rounds)]

    # ── Collect per-fly position trails, indexed by order in the points list ─
    n_flies = max(len(entry['points']) for entry in history)
    fly_trails = {i: {'xs': [], 'ys': []} for i in range(n_flies)}
    for entry in history:
        for fly_i, pt in enumerate(entry['points']):
            if x_param in pt and y_param in pt:
                fly_trails[fly_i]['xs'].append(pt[x_param])
                fly_trails[fly_i]['ys'].append(pt[y_param])

    # ── Draw trails, per-round dots, and final-step arrows ────────────────────
    for trail in fly_trails.values():
        xs, ys = trail['xs'], trail['ys']
        if not xs:
            continue
        if len(xs) > 1:
            ax.plot(xs, ys, 'k-', alpha=0.25, linewidth=0.8, zorder=2)
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="->", color='red', lw=1.2), zorder=4)
        for r_idx, (px, py) in enumerate(zip(xs, ys)):
            ax.scatter(px, py, color=round_colors[r_idx], s=40,
                       edgecolors='k', linewidths=0.4, zorder=3)

    # ── Round legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=round_colors[r_idx], edgecolor='k', linewidth=0.5,  # type: ignore[arg-type]
              label=f'Round {entry["round"]}')
        for r_idx, entry in enumerate(history)
    ]
    ax.legend(handles=legend_handles, title='Round', loc='best', framealpha=0.8)

    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)
    ax.set_title(f'FOA Search Trajectories: {x_param} vs {y_param}', fontsize=13)
    plt.tight_layout()

    out = save_path or os.path.join(os.getcwd(), 'search_history.png')
    plt.savefig(out, dpi=150)
    print("Saved to", out)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser class
# ─────────────────────────────────────────────────────────────────────────────

class OptimiseN:
    """
    One round of a multi-round N-dimensional Fruit Fly Optimisation.

    Each call to ``go()`` reads the previous round's crawled positions from a
    pickle file, matches them against the new simulation results, moves each fly
    one step, saves the updated history, and writes a progress plot.

    The history pickle (``{name}_history.pckl``) stores a list of dicts, one
    per completed round::

        {'round': int, 'points': [...scored dicts...], 'next_inputs': [...param dicts...]}

    Settings keys (all passed via *optimiser_settings* dict)
    ---------------------------------------------------------
    Required:
        cost               Key name of the efficiency/cost field in results.
        input_parameters   List of N parameter names to optimise over.

    Optional:
        name               Base name for output files (default: ``'optimisation'``).
        optimisation_speed Base step size in normalised [0, 1] space (default 0.25).
        valid_cost_range   [min, max] – results outside this range have their cost
                           zeroed rather than being discarded (default [0, 1]).
        param_bounds       Dict mapping param → [min, max] in physical units.
                           Without this, flies can wander outside the search region.
        noise_scale        Random exploration amplitude in normalised space (default
                           0.05).  Decays with step size so stuck flies explore more.
        repulsion_scale    Fraction of attraction strength applied as repulsion from
                           lower-scoring points (default 0.5).
        elitism            If True (default), each round the worst-performing fly
                           (excluding any just reseeded) is replaced with a small
                           perturbation of the best point found so far, so the
                           swarm never drifts away from its best discovery.
        diversity_scale    Strength of a small always-on, cost-independent repulsion
                           between any two points (default 0.05), guarding against
                           (near-)duplicate proposals — equal-cost points otherwise
                           exert zero force on each other and can converge onto the
                           same coordinates, wasting a simulation re-evaluating it.
        diversity_radius   Normalised distance beyond which diversity_scale decays to
                           ~0 (default 0.02) — kept small so it only affects points
                           that are nearly coincident, not general exploration.
        anneal             If True (default), crawl_speed and noise_scale decay
                           linearly from their configured value (round 1) down to
                           anneal_floor × value (the final round), so early rounds
                           explore broadly and later rounds fine-tune around what's
                           been found. Requires optimisation_rounds (read from this
                           same settings dict) to know the total round budget —
                           skipped (rates stay constant) if that's absent.
        anneal_floor       Fraction of crawl_speed/noise_scale retained at the final
                           round (default 0.2).
        seed               Optional int to make a run reproducible. Each round gets
                           its own Generator derived from (seed, round number), so
                           re-running the whole multi-round optimisation with the
                           same seed reproduces the same search trajectory every
                           time. Without it, each round draws from fresh entropy.
    """

    def __init__(self, all_points, current_round, optimiser_settings):
        self.all_points       = all_points
        self.round            = current_round

        self.cost             = optimiser_settings['cost']
        self.input_parameters = optimiser_settings['input_parameters']
        self.valid_cost_range = optimiser_settings.get('valid_cost_range', [0, 1])
        self.param_bounds     = optimiser_settings.get('param_bounds', None)
        self.crawl_speed      = optimiser_settings.get('optimisation_speed', 0.25)
        self.noise_scale      = optimiser_settings.get('noise_scale', 0.05)
        self.repulsion_scale  = optimiser_settings.get('repulsion_scale', 0.5)
        self.elitism          = optimiser_settings.get('elitism', True)
        self.seed             = optimiser_settings.get('seed', None)
        self.rng              = (np.random.default_rng([self.seed, self.round])
                                  if self.seed is not None else np.random.default_rng())
        self.diversity_scale  = optimiser_settings.get('diversity_scale', 0.05)
        self.diversity_radius = optimiser_settings.get('diversity_radius', 0.02)
        self.anneal           = optimiser_settings.get('anneal', True)
        self.anneal_floor     = optimiser_settings.get('anneal_floor', 0.2)
        self.total_rounds     = optimiser_settings.get('optimisation_rounds', None)

        name = optimiser_settings.get('name', 'optimisation')
        self.history_file    = f"{name}_history.pckl"
        self.progress_plot   = f"{name}_progress.png"
        self.trajectory_plot = f"{name}_trajectory.png"

        self.points_to_crawl  = []
        self.reseeded_flags   = []
        self.new_inputs       = []
        self.next_run         = {'next_run_type': 'points'}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _annealed_params(self):
        """
        Apply an explore→exploit schedule to crawl_speed and noise_scale: both
        decay linearly from their configured value (round 1) down to
        ``anneal_floor`` × value (the final round), so early rounds explore
        broadly and later rounds fine-tune around what's been found so far.

        Returns ``(crawl_speed, noise_scale)`` unchanged if ``anneal`` is
        False or ``optimisation_rounds`` wasn't provided (nothing to anneal
        towards).
        """
        if not self.anneal or not self.total_rounds or self.total_rounds <= 1:
            return self.crawl_speed, self.noise_scale
        progress = min(max((self.round - 1) / (self.total_rounds - 1), 0.0), 1.0)
        multiplier = 1 - (1 - self.anneal_floor) * progress
        return self.crawl_speed * multiplier, self.noise_scale * multiplier

    def _sanitise_points(self):
        """
        Zero out the cost of results outside *valid_cost_range* and deduplicate.
        Points are kept (not removed) so they still contribute to the landscape.
        """
        seen = []
        for pt in self.all_points:
            cost_val = pt.get(self.cost, 0)
            if not (self.valid_cost_range[0] <= cost_val <= self.valid_cost_range[1]):
                pt[self.cost] = 0
            if pt not in seen:
                seen.append(pt)
        self.all_points = seen

    def _reseed_points(self, n):
        """
        Generate *n* replacement flies to replenish the population after some
        went unmatched this round (e.g. a failed simulation).

        Samples uniformly within ``param_bounds`` when it covers every axis in
        ``input_parameters``; otherwise falls back to duplicating a random
        already-evaluated point (the normal crawl/noise step will diversify it
        from there). Cost is set to 0 (the same "no information" convention
        ``_sanitise_points`` uses for out-of-range results) since these points
        haven't been evaluated yet — ``crawl_optimise`` requires every current
        point to carry a cost value.
        """
        has_full_bounds = self.param_bounds and all(p in self.param_bounds for p in self.input_parameters)
        points = []
        for _ in range(n):
            if has_full_bounds:
                pt = {p: _round_sig(float(self.rng.uniform(*self.param_bounds[p])))
                      for p in self.input_parameters}
            else:
                base = self.all_points[self.rng.integers(len(self.all_points))]
                pt = {p: base[p] for p in self.input_parameters if p in base}
            pt[self.cost] = 0
            points.append(pt)
        return points

    def _elite_point(self):
        """Return the highest-cost point seen so far, or None if all_points is empty."""
        if not self.all_points:
            return None
        return max(self.all_points, key=lambda p: p.get(self.cost, 0))

    def _perturb_point(self, point, noise_scale):
        """
        Return a copy of *point* nudged by small gaussian noise (scaled by
        *noise_scale*) on each axis, clamped to bounds.

        Used by elitism to place a fly near the best point found so far
        without proposing the exact same coordinates again (which would waste
        a simulation re-confirming an already-known cost).
        """
        result = {}
        for p in self.input_parameters:
            if self.param_bounds and p in self.param_bounds:
                lo, hi = self.param_bounds[p]
                span = hi - lo
            else:
                span = 1.0
            jitter = float(self.rng.normal(0.0, noise_scale * span))
            result[p] = _round_sig(_clamp_to_bounds(point[p] + jitter, p, self.param_bounds))
        return result

    def _load_current_points(self):
        """
        Determine which flies to crawl this round.

        Round 1 – all evaluated points.
        Round N – the crawled positions saved at the end of round N-1, enriched
                  with the newly evaluated scores from *all_points*. Any fly
                  that couldn't be matched to a scored result (e.g. a failed
                  simulation) is replaced with a fresh point via
                  ``_reseed_points`` so the population size doesn't silently
                  shrink round over round. ``self.reseeded_flags`` marks which
                  entries of ``points_to_crawl`` were reseeded this round, so
                  elitism (in ``go()``) can avoid immediately overwriting a
                  fresh exploration point.
        """
        self._sanitise_points()
        if self.round > 1:
            history = _load_pckl(self.history_file) or []
            prev_inputs = history[-1].get('next_inputs', []) if history else []
            candidates = _fill_in_dict_list(prev_inputs, self.all_points)
            matched = [p for p in candidates if self.cost in p]
            reseeded_flags = [False] * len(matched)

            missing = len(prev_inputs) - len(matched)
            if missing > 0 and self.all_points:
                print(f"OptimiseN round {self.round}: {missing}/{len(prev_inputs)} flies had no "
                      f"matching result (e.g. failed simulation) - reseeding {missing} replacement(s) "
                      f"to keep the population size stable.")
                matched.extend(self._reseed_points(missing))
                reseeded_flags.extend([True] * missing)

            self.points_to_crawl = matched
            self.reseeded_flags = reseeded_flags
        else:
            self.points_to_crawl = self.all_points
            self.reseeded_flags = [False] * len(self.all_points)

    def _append_history(self, scored_points, next_inputs):
        """Append this round's scored points and next-round inputs to the history pickle."""
        history = _load_pckl(self.history_file) or []
        history.append({
            'round':       self.round,
            'points':      list(scored_points),
            'next_inputs': list(next_inputs),
        })
        _save_pckl(self.history_file, history)

    # ── Public API ────────────────────────────────────────────────────────────

    def go(self):
        """
        Run one optimisation step.

        Loads current fly positions, computes the next positions via FOA
        (with crawl_speed/noise_scale annealed per ``_annealed_params``),
        applies elitism, saves the history, updates ``self.next_run``, and
        writes a progress plot.
        """
        self._load_current_points()
        crawl_speed, noise_scale = self._annealed_params()
        self.new_inputs = crawl_optimise(
            self.points_to_crawl, self.all_points,
            crawl_speed, self.input_parameters,
            self.cost,
            self.param_bounds, noise_scale, self.repulsion_scale,
            self.diversity_scale, self.diversity_radius, self.rng,
        )

        if self.elitism:
            elite = self._elite_point()
            candidates = [i for i in range(len(self.points_to_crawl)) if not self.reseeded_flags[i]]
            if elite is not None and candidates:
                worst_idx = min(candidates, key=lambda i: self.points_to_crawl[i].get(self.cost, 0))
                self.new_inputs[worst_idx] = self._perturb_point(elite, noise_scale)

        self._append_history(self.points_to_crawl, self.new_inputs)
        self.next_run['input_points'] = self.new_inputs  # type: ignore[assignment]
        visualize_search_history(
            z_param=self.cost,
            history_file=self.history_file,
            save_path=self.progress_plot,
            show=False,
        )
        if len(self.input_parameters) == 2:
            visualize_search_trajectories_2d(
                x_param=self.input_parameters[0],
                y_param=self.input_parameters[1],
                z_param=self.cost,
                history_file=self.history_file,
                save_path=self.trajectory_plot,
                show=False,
            )

    def cleanup(self):
        """Delete the history pickle after the final optimisation round."""
        path = os.path.join(os.getcwd(), self.history_file)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as error:
            print(f"Could not remove {self.history_file}: {error}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: run the optimiser over synthetic test landscapes
# ─────────────────────────────────────────────────────────────────────────────

def _test_landscape(coords, peaks):
    """Sum of gaussian bumps at *coords* (peak = (center_tuple, height, width)), clipped to [0, 1]."""
    total = 0.0
    for center, height, width in peaks:
        d2 = sum((c - c0) ** 2 for c, c0 in zip(coords, center))
        total += height * math.exp(-d2 / (2 * width ** 2))
    return max(0.0, min(1.0, total))


def _uniform_grid_points(params, points_per_axis):
    """
    Deterministic grid of points evenly spaced over [0, 1] on every axis in
    *params* — ``points_per_axis`` values per axis, full factorial (so
    ``points_per_axis ** len(params)`` points in total). No randomness.
    """
    axis_vals = np.linspace(0.0, 1.0, points_per_axis)
    mesh = np.meshgrid(*([axis_vals] * len(params)), indexing='ij')
    coords = np.stack([m.ravel() for m in mesh], axis=-1)
    return [{p: round(float(v), 5) for p, v in zip(params, row)} for row in coords]


def _run_test_case(params, peaks, optimiser_settings, initial_points, n_rounds, output_dir):
    """
    Run *n_rounds* rounds of OptimiseN against a synthetic gaussian-peak
    landscape over *params*, starting from *initial_points*, writing the
    progress/trajectory plots (and history pickle) into *output_dir* so they
    can be inspected afterwards.
    Returns the best cost found so far, one entry per round.
    """
    cwd = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    try:
        points: list = [dict(pt) for pt in initial_points]
        all_points = []
        best_costs = []
        best_so_far = 0.0

        for round_i in range(1, n_rounds + 1):
            for pt in points:
                pt['cost'] = _test_landscape([pt[p] for p in params], peaks)
            all_points.extend(points)
            best_so_far = max(best_so_far, max(pt['cost'] for pt in points))
            best_costs.append(best_so_far)

            op = OptimiseN(list(all_points), round_i, optimiser_settings)
            op.go()
            points = op.next_run['input_points']  # type: ignore[assignment]

        return best_costs
    finally:
        os.chdir(cwd)


def main():
    """
    Sanity-check OptimiseN by running it over a handful of synthetic
    multi-peak test landscapes with different dimensionalities and settings,
    printing the best cost found per round for each case.
    """
    test_cases = [
        {
            'label': '2-D, default settings',
            'params': ['x', 'y'],
            'peaks': [((0.7, 0.7), 1.0, 0.12), ((0.2, 0.3), 0.6, 0.15)],
            'settings': {'optimisation_speed': 0.25, 'noise_scale': 0.05, 'repulsion_scale': 0.5},
        },
        {
            'label': '2-D, fast/aggressive crawl',
            'params': ['x', 'y'],
            'peaks': [((0.7, 0.7), 1.0, 0.12), ((0.2, 0.3), 0.6, 0.15)],
            'settings': {'optimisation_speed': 0.5, 'noise_scale': 0.1, 'repulsion_scale': 0.3},
        },
        {
            'label': '2-D, slow/cautious crawl',
            'params': ['x', 'y'],
            'peaks': [((0.7, 0.7), 1.0, 0.12), ((0.2, 0.3), 0.6, 0.15)],
            'settings': {'optimisation_speed': 0.1, 'noise_scale': 0.02, 'repulsion_scale': 0.8},
        },
        {
            'label': '3-D, default settings',
            'params': ['x', 'y', 'z'],
            'peaks': [((0.7, 0.7, 0.5), 1.0, 0.15), ((0.2, 0.3, 0.8), 0.6, 0.18)],
            'settings': {'optimisation_speed': 0.25, 'noise_scale': 0.05, 'repulsion_scale': 0.5},
        },
    ]

    points_per_axis, n_rounds = 4, 15
    output_root = os.path.join(os.getcwd(), 'optimiseN_selftest_output')

    # Generate one shared starting layout per distinct parameter set: a fixed
    # grid evenly spaced over the search space (no randomness), so every case
    # using those params starts from exactly the same fly positions —
    # differences in the results are then attributable only to the settings
    # being varied, not to the starting layout.
    initial_points_by_params = {}
    for case in test_cases:
        key = tuple(case['params'])
        if key not in initial_points_by_params:
            initial_points_by_params[key] = _uniform_grid_points(case['params'], points_per_axis)

    print(f"OptimiseN self-test: {points_per_axis} points per axis x {n_rounds} rounds per case")
    print(f"Plots will be saved under {output_root}/\n")
    for case in test_cases:
        params = case['params']
        slug = case['label'].lower().replace(',', '').replace('/', '_').replace(' ', '_')
        case_dir = os.path.join(output_root, slug)
        settings = {
            'cost':                'cost',
            'input_parameters':    params,
            'valid_cost_range':    [0, 1],
            'param_bounds':        {p: [0.0, 1.0] for p in params},
            'name':                'selftest',
            'optimisation_rounds': n_rounds,
            'seed':                0,
            **case['settings'],
        }
        initial_points = initial_points_by_params[tuple(params)]
        best_costs = _run_test_case(params, case['peaks'], settings, initial_points, n_rounds, case_dir)
        trail = " ".join(f"{c:.3f}" for c in best_costs)
        print(f"{case['label']}")
        print(f"  best cost by round: {trail}")
        print(f"  final best:         {best_costs[-1]:.4f}")
        print(f"  plots saved to:     {case_dir}/\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
