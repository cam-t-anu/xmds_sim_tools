#!/usr/bin/python3
"""
Fruit Fly Optimisation (FOA) – 2-D parameter search.

The algorithm moves a swarm of sample points ("flies") through a 2-D parameter
space guided by two forces computed in normalised [0, 1] coordinates:

  Attraction  – toward evaluated points with a higher cost score.
  Repulsion   – away from evaluated points with a lower cost score.

Both forces are weighted by the magnitude of the score difference and by
distance (attraction saturates via mod_sigmoid; repulsion decays exponentially).
An adaptive random noise term is added at each step — largest when the directed
move is small — which helps flies escape local optima.

Typical usage
-------------
    Op = Optimise3(results, current_round, optimiser_settings)
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

def _dist(p1, p2, x, y):
    """Euclidean distance between two parameter-dicts along the *x* and *y* axes."""
    dx = p1[x] - p2[x]
    dy = p1[y] - p2[y]
    return math.sqrt(dx * dx + dy * dy)


def _angle(from_pt, to_pt, x, y):
    """Angle in radians from *from_pt* toward *to_pt* in the *x*-*y* plane."""
    dx = to_pt[x] - from_pt[x]
    dy = to_pt[y] - from_pt[y]
    return math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0.0


def _add_polar(r1, a1, r2, a2):
    """
    Add two 2-D vectors in polar form and return ``{"length": r, "angle": a}``.
    """
    nx = r1 * math.cos(a1) + r2 * math.cos(a2)
    ny = r1 * math.sin(a1) + r2 * math.sin(a2)
    r = math.sqrt(nx * nx + ny * ny)
    a = math.atan2(ny, nx) if (nx != 0 or ny != 0) else 0.0
    return {"length": r, "angle": a}


def _mod_sigmoid(x, limit):
    """
    Sigmoid scaled so the output ∈ (-limit, limit) and f(0) = 0.
    Used to saturate cumulative crawl distances so no single interaction
    dominates the net step direction.
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


def _attract_weight_dist(ref_pt, cmp_pt, crawl_speed, x, y):
    """
    Distance weight for attraction: mod_sigmoid of Euclidean separation.
    Rises from 0 at zero separation and saturates around *crawl_speed*.
    """
    return _mod_sigmoid(_dist(ref_pt, cmp_pt, x, y), crawl_speed)


def _attraction(ref_pt, cmp_pt, crawl_speed, x, y, z):
    """
    Attraction step magnitude toward *cmp_pt*.

    = crawl_speed × dist_weight × z_gain_weight.

    Non-zero only when *cmp_pt* scores higher than *ref_pt*.
    """
    return (crawl_speed
            * _attract_weight_dist(ref_pt, cmp_pt, crawl_speed, x, y)
            * _attract_weight_z(ref_pt, cmp_pt, z))


def _repulsion(ref_pt, cmp_pt, crawl_speed, x, y, z):
    """
    Repulsion step magnitude away from *cmp_pt*.

    = crawl_speed × exp(-dist / crawl_speed) × z_deficit_weight.

    Decays exponentially with distance so only nearby bad points contribute.
    Non-zero only when *cmp_pt* scores lower than *ref_pt*.
    """
    w_z = _repulse_weight_z(ref_pt, cmp_pt, z)
    if w_z == 0.0:
        return 0.0
    d = _dist(ref_pt, cmp_pt, x, y)
    w_dist = math.exp(-d / max(crawl_speed, 1e-9))
    return crawl_speed * w_dist * w_z


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_normalisation(all_points, x, y, bounds):
    """
    Return ``(x_origin, x_scale, y_origin, y_scale)`` that maps physical
    coordinates to ~[0, 1].

    Prefer *bounds* when provided; otherwise derives the range from *all_points*.
    Normalising before computing distances and angles prevents axes with very
    different physical ranges from distorting the crawl direction.
    """
    if bounds and x in bounds and y in bounds:
        x_lo, x_hi = bounds[x]
        y_lo, y_hi = bounds[y]
    else:
        xs = [p[x] for p in all_points if x in p]
        ys = [p[y] for p in all_points if y in p]
        x_lo, x_hi = (min(xs), max(xs)) if len(xs) > 1 else (0.0, 1.0)
        y_lo, y_hi = (min(ys), max(ys)) if len(ys) > 1 else (0.0, 1.0)
    x_scale = max(x_hi - x_lo, 1e-9)
    y_scale = max(y_hi - y_lo, 1e-9)
    return x_lo, x_scale, y_lo, y_scale


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
# Core crawl step
# ─────────────────────────────────────────────────────────────────────────────

def _do_crawl(ref_pt, step_dist, step_angle, x, y, noise_scale=0.05):
    """
    Displace *ref_pt* by a directed step plus an adaptive random noise term.

    Noise magnitude = ``noise_scale × exp(-step_dist / noise_scale)``:
    maximum when the directed step is near zero (fly near a local optimum),
    decaying to ~0 for large directed steps, so exploration is concentrated
    where it is needed most.

    Operates in normalised [0, 1] space.  The caller denormalises and clamps to
    parameter bounds.

    Returns a dict with updated *x* and *y* values.
    """
    noise_mag = noise_scale * math.exp(-step_dist / max(noise_scale, 1e-9))
    noise_angle = np.random.uniform(-np.pi, np.pi)
    total = _add_polar(step_dist, step_angle, noise_mag, noise_angle)
    return {
        x: ref_pt[x] + total["length"] * math.cos(total["angle"]),
        y: ref_pt[y] + total["length"] * math.sin(total["angle"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation step
# ─────────────────────────────────────────────────────────────────────────────

def crawl_optimise(current_points, all_points, crawl_speed, input_params, input_dims,
                   z, bounds=None, noise_scale=0.05, repulsion_scale=0.5):
    """
    Compute one FOA step for every fly in *current_points*.

    For each fly, sums attraction vectors toward higher-scoring evaluated
    points and repulsion vectors away from lower-scoring ones, then moves the
    fly one step along the net resultant.  All geometry is computed in
    normalised [0, 1] space so axes with different physical scales contribute
    equally.  Results are denormalised and clamped to *bounds* before being
    returned.

    Args:
        current_points:  list of dicts – flies to move this step.
        all_points:      list of dicts – all evaluated points (including current).
        crawl_speed:     base step size in normalised space (0–1).
        input_params:    [x_param, y_param] – names of the two axes.
        input_dims:      must be 2.
        z:               key name of the cost/efficiency field.
        bounds:          dict mapping param → [min, max] in physical units.
        noise_scale:     random exploration amplitude in normalised space.
        repulsion_scale: fraction of attraction strength applied as repulsion.

    Returns:
        List of dicts with new x/y values for each fly, or [] on error.
    """
    if input_dims != 2:
        print("crawl_optimise requires exactly 2 input dimensions")
        return []
    if not all_points:
        print("crawl_optimise: all_points is empty")
        return []

    x, y = input_params[0], input_params[1]
    x_origin, x_scale, y_origin, y_scale = _compute_normalisation(all_points, x, y, bounds)

    def _norm(pt):
        n = dict(pt)
        n[x] = (pt[x] - x_origin) / x_scale
        n[y] = (pt[y] - y_origin) / y_scale
        return n

    all_pts_norm = [_norm(p) for p in all_points]
    cur_pts_norm = [_norm(p) for p in current_points]

    new_pts = []
    for ref_orig, ref_n in zip(current_points, cur_pts_norm):
        net = {"length": 0.0, "angle": 0.0}

        for cmp_orig, cmp_n in zip(all_points, all_pts_norm):
            if cmp_orig is ref_orig:
                continue
            angle = _angle(ref_n, cmp_n, x, y)
            attract = _attraction(ref_n, cmp_n, crawl_speed, x, y, z)
            net = _add_polar(net["length"], net["angle"], attract, angle)
            repel = repulsion_scale * _repulsion(ref_n, cmp_n, crawl_speed, x, y, z)
            net = _add_polar(net["length"], net["angle"], repel, angle + math.pi)

        # Saturate step size and slow flies already near the optimum.
        ref_z = ref_orig.get(z, 0)
        step = (_mod_sigmoid(net["length"], 4 * crawl_speed)
                * _mod_flipped_sigmoid(ref_z, 4 / crawl_speed))

        crawled = _do_crawl(ref_n, step, net["angle"], x, y, noise_scale)
        new_x = _round_sig(_clamp_to_bounds(crawled[x] * x_scale + x_origin, x, bounds))
        new_y = _round_sig(_clamp_to_bounds(crawled[y] * y_scale + y_origin, y, bounds))
        new_pts.append({x: new_x, y: new_y})

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

def visualize_search_history(x_param, y_param, z_param,
                              history_file=None, save_path=None, show=True):
    """
    Plot how optimiser flies moved through the search space across rounds.

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

class Optimise3:
    """
    One round of a multi-round Fruit Fly Optimisation.

    Each call to ``go()`` reads the previous round's crawled positions from a
    pickle file, matches them against the new simulation results, moves each fly
    one step, saves the updated history, and writes a progress plot.

    The history pickle (``{name}_history.pckl``) stores a list of dicts, one
    per completed round::

        {'round': int, 'points': [...scored dicts...], 'next_inputs': [...x/y dicts...]}

    Settings keys (all passed via *optimiser_settings* dict)
    ---------------------------------------------------------
    Required:
        cost               Key name of the efficiency/cost field in results.
        input_parameters   List of two parameter names to optimise over.

    Optional:
        name               Base name for output files (default: ``'optimisation'``).
        optimisation_speed Base step size in normalised [0, 1] space (default 0.25).
        input_dimensions   Must be 2 (default 2).
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
        self.input_dimensions = optimiser_settings.get('input_dimensions', 2)
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
            self.input_dimensions, self.cost,
            self.param_bounds, self.noise_scale, self.repulsion_scale,
        )
        self._append_history(self.points_to_crawl, self.new_inputs)
        self.next_run['input_points'] = self.new_inputs  # type: ignore[assignment]
        if self.input_dimensions == 2:
            visualize_search_history(
                x_param=self.input_parameters[0],
                y_param=self.input_parameters[1],
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
