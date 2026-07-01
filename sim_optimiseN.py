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

def _dist_nd(p1, p2, params):
    """Euclidean distance between two parameter-dicts over all axes in *params*."""
    return math.sqrt(sum((p1[p] - p2[p]) ** 2 for p in params))


def _unit_vector_nd(from_pt, to_pt, params):
    """
    Unit vector from *from_pt* toward *to_pt* as a numpy array indexed by
    the order of *params*.  Returns a zero vector when the points coincide.
    """
    diff = np.array([to_pt[p] - from_pt[p] for p in params], dtype=float)
    norm = np.linalg.norm(diff)
    return diff / norm if norm > 1e-12 else np.zeros(len(params))


def _mod_sigmoid(x, limit):
    """
    Sigmoid scaled so the output ∈ (-limit, limit) and f(0) = 0.
    Saturates cumulative force magnitudes so no single interaction dominates.
    """
    return 2 * limit / (1 + math.exp(-2 * x / limit)) - limit


def _mod_flipped_sigmoid(x, steepness):
    """
    Decreasing sigmoid: f(0) = 1, f(∞) → 0.
    Slows flies that already sit near the optimum (high cost score).
    """
    return (-2 / (1 + math.exp(-steepness * x))) + 2


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair force magnitudes
# ─────────────────────────────────────────────────────────────────────────────

def _attract_weight_z(ref_pt, cmp_pt, z):
    """
    Fractional cost improvement cmp_pt[z] - ref_pt[z], clamped to [0, 1].
    Zero when *cmp_pt* is equal or worse than *ref_pt*.
    """
    return min(max(cmp_pt[z] - ref_pt[z], 0.0), 1.0)


def _repulse_weight_z(ref_pt, cmp_pt, z):
    """
    Fractional cost deficit ref_pt[z] - cmp_pt[z], clamped to [0, 1].
    Zero when *cmp_pt* is equal or better than *ref_pt*.
    """
    return min(max(ref_pt[z] - cmp_pt[z], 0.0), 1.0)


def _attraction_magnitude(ref_pt, cmp_pt, crawl_speed, params, z):
    """
    Attraction step magnitude toward *cmp_pt*.

    = crawl_speed × mod_sigmoid(dist, crawl_speed) × z_gain_weight.

    Non-zero only when *cmp_pt* scores higher than *ref_pt*.
    """
    w_z = _attract_weight_z(ref_pt, cmp_pt, z)
    if w_z == 0.0:
        return 0.0
    d = _dist_nd(ref_pt, cmp_pt, params)
    w_dist = _mod_sigmoid(d, crawl_speed)
    return crawl_speed * w_dist * w_z


def _repulsion_magnitude(ref_pt, cmp_pt, crawl_speed, params, z):
    """
    Repulsion step magnitude away from *cmp_pt*.

    = crawl_speed × exp(-dist / crawl_speed) × z_deficit_weight.

    Decays exponentially with distance so only nearby bad points contribute.
    Non-zero only when *cmp_pt* scores lower than *ref_pt*.
    """
    w_z = _repulse_weight_z(ref_pt, cmp_pt, z)
    if w_z == 0.0:
        return 0.0
    d = _dist_nd(ref_pt, cmp_pt, params)
    w_dist = math.exp(-d / max(crawl_speed, 1e-9))
    return crawl_speed * w_dist * w_z


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


# ─────────────────────────────────────────────────────────────────────────────
# Core crawl step (N-dimensional)
# ─────────────────────────────────────────────────────────────────────────────

def _do_crawl_nd(ref_pt, step_mag, step_dir, params, noise_scale=0.05):
    """
    Displace *ref_pt* by a directed step plus an adaptive random noise term in N-D.

    *step_dir* is a unit numpy array of length ``len(params)``.

    Noise magnitude = ``noise_scale × exp(-step_mag / noise_scale)``:
    maximum when the directed step is near zero (fly near a local optimum),
    decaying to ~0 for large directed steps.  Noise direction is a uniformly
    random unit vector in the N-D space.

    Operates in normalised [0, 1] space.  The caller denormalises and clamps to
    parameter bounds.  Returns a dict with updated parameter values.
    """
    noise_mag = noise_scale * math.exp(-step_mag / max(noise_scale, 1e-9))
    noise_raw = np.random.randn(len(params))
    noise_norm = np.linalg.norm(noise_raw)
    noise_dir = noise_raw / noise_norm if noise_norm > 1e-12 else np.zeros(len(params))

    total_vec = step_mag * step_dir + noise_mag * noise_dir
    result = dict(ref_pt)
    for i, p in enumerate(params):
        result[p] = ref_pt[p] + total_vec[i]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation step
# ─────────────────────────────────────────────────────────────────────────────

def crawl_optimise(current_points, all_points, crawl_speed, input_params,
                   z, bounds=None, noise_scale=0.05, repulsion_scale=0.5):
    """
    Compute one FOA step for every fly in *current_points* (N-dimensional).

    For each fly, accumulates attraction vectors (toward higher-scoring points)
    and repulsion vectors (away from lower-scoring points) as Cartesian force
    vectors, then moves the fly one step along the net resultant direction.
    All geometry is computed in normalised [0, 1] space.  Results are
    denormalised and clamped to *bounds* before being returned.

    Args:
        current_points:  list of dicts – flies to move this step.
        all_points:      list of dicts – all evaluated points (including current).
        crawl_speed:     base step size in normalised space (0–1).
        input_params:    list of N parameter names to optimise over.
        z:               key name of the cost/efficiency field.
        bounds:          dict mapping param → [min, max] in physical units.
        noise_scale:     random exploration amplitude in normalised space.
        repulsion_scale: fraction of attraction strength applied as repulsion.

    Returns:
        List of dicts with updated parameter values for each fly, or [] on error.
    """
    if not all_points:
        print("crawl_optimise: all_points is empty")
        return []

    params = input_params
    origins, scales = _compute_normalisation(all_points, params, bounds)

    def _norm(pt):
        n = dict(pt)
        for p in params:
            n[p] = (pt[p] - origins[p]) / scales[p]
        return n

    all_pts_norm = [_norm(p) for p in all_points]
    cur_pts_norm = [_norm(p) for p in current_points]

    new_pts = []
    for ref_orig, ref_n in zip(current_points, cur_pts_norm):
        net_force = np.zeros(len(params))

        for cmp_orig, cmp_n in zip(all_points, all_pts_norm):
            if cmp_orig is ref_orig:
                continue
            direction = _unit_vector_nd(ref_n, cmp_n, params)
            attract = _attraction_magnitude(ref_n, cmp_n, crawl_speed, params, z)
            net_force += attract * direction
            repel = repulsion_scale * _repulsion_magnitude(ref_n, cmp_n, crawl_speed, params, z)
            net_force -= repel * direction  # opposite direction

        net_mag = float(np.linalg.norm(net_force))
        net_dir = net_force / net_mag if net_mag > 1e-12 else np.zeros(len(params))

        # Saturate step size and slow flies already near the optimum.
        ref_z = ref_orig.get(z, 0)
        step = (_mod_sigmoid(net_mag, 4 * crawl_speed)
                * _mod_flipped_sigmoid(ref_z, 4 / crawl_speed))

        crawled = _do_crawl_nd(ref_n, step, net_dir, params, noise_scale)
        new_pt = {p: _clamp_to_bounds(crawled[p] * scales[p] + origins[p], p, bounds)
                  for p in params}
        new_pts.append(new_pt)

    return new_pts


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

        name = optimiser_settings.get('name', 'optimisation')
        self.history_file  = f"{name}_history.pckl"
        self.progress_plot = f"{name}_progress.png"

        self.points_to_crawl = []
        self.new_inputs      = []
        self.next_run        = {'next_run_type': 'points'}

    # ── Internal helpers ──────────────────────────────────────────────────────

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

    def _load_current_points(self):
        """
        Determine which flies to crawl this round.

        Round 1 – all evaluated points.
        Round N – the crawled positions saved at the end of round N-1, enriched
                  with the newly evaluated scores from *all_points*.
        """
        self._sanitise_points()
        if self.round > 1:
            history = _load_pckl(self.history_file) or []
            prev_inputs = history[-1].get('next_inputs', []) if history else []
            candidates = _fill_in_dict_list(prev_inputs, self.all_points)
            self.points_to_crawl = [p for p in candidates if self.cost in p]
        else:
            self.points_to_crawl = self.all_points

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

        Loads current fly positions, computes the next positions via FOA, saves
        the history, updates ``self.next_run``, and writes a progress plot.
        """
        self._load_current_points()
        self.new_inputs = crawl_optimise(
            self.points_to_crawl, self.all_points,
            self.crawl_speed, self.input_parameters,
            self.cost,
            self.param_bounds, self.noise_scale, self.repulsion_scale,
        )
        self._append_history(self.points_to_crawl, self.new_inputs)
        self.next_run['input_points'] = self.new_inputs  # type: ignore[assignment]
        visualize_search_history(
            z_param=self.cost,
            history_file=self.history_file,
            save_path=self.progress_plot,
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

if __name__ == "__main__":
    print("This module is not intended to be run directly. "
          "Use optimise_1DGEM-EIT.py instead.")
