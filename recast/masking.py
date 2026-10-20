"""Observation masks, causal input filling and synthetic hiding of inputs.

Four Boolean arrays describe each input window (shape [..., P, N]):
  O  original validity: the raw reading is finite and strictly positive.
  S  synthetic retention: False where an originally valid reading is hidden.
  M = O & S   availability; supplied to the model.
  H = O & ~S  hidden set; the only reconstruction targets.
Forecast targets are scored where they are originally valid (V = O).
"""
import numpy as np


def original_validity(raw):
    """O: finite and strictly positive readings (benchmark convention)."""
    raw = np.asarray(raw, dtype=np.float64)
    return np.isfinite(raw) & (raw > 0)


def fit_standardization(raw, valid, train_end):
    """Mean and population SD of all originally valid training readings."""
    selected = np.asarray(raw[:train_end], dtype=np.float64)[valid[:train_end]]
    return float(selected.mean()), float(selected.std())


def causal_fill(values, available):
    """Forward-fill unavailable entries along the time axis (axis -2) of each window.

    values, available: arrays of shape (..., P, N) with standardized values and M.
    Entries with no earlier available value in the window are set to 0, which is the
    training mean after standardization. Only past values of the same sensor are used.
    """
    values = np.asarray(values, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    filled = np.zeros_like(values)
    last = np.zeros_like(values[..., 0, :])
    for step in range(values.shape[-2]):
        last = np.where(available[..., step, :], values[..., step, :], last)
        filled[..., step, :] = last
    return filled


def window_masks(observed, retained):
    """Return M = O & S and H = O & ~S for input windows."""
    observed = np.asarray(observed, dtype=bool)
    retained = np.asarray(retained, dtype=bool)
    return observed & retained, observed & ~retained


def hide_point(observed, rate, rng):
    """Retention mask S that hides each originally valid reading with probability `rate`.

    observed: (T, N) Boolean O on a timeline. rng: numpy.random.Generator.
    """
    observed = np.asarray(observed, dtype=bool)
    retained = np.ones_like(observed)
    retained[(rng.random(observed.shape) < rate) & observed] = False
    return retained


def hide_blocks(observed, rate, rng, min_steps=2, max_steps=6, segment=None):
    """Retention mask S with single-sensor gaps until a fraction `rate` of valid readings is hidden.

    Each event hides one sensor for a uniformly drawn duration of `min_steps` to
    `max_steps` steps (10 to 30 min at 5-min resolution). Events are sampled with
    replacement and overlapping events are merged. Events are clipped at the end of
    the timeline and, if `segment` (an integer label per time step) is given, at
    changes of segment; durations are never shortened to meet the budget.
    Cells that are already invalid may also be marked as not retained; this has no
    effect because M = O & S and H = O & ~S. rng: numpy.random.Generator.
    """
    observed = np.asarray(observed, dtype=bool)
    t, n = observed.shape
    retained = np.ones_like(observed)
    target = int(round(rate * observed.sum()))
    removed = 0
    while removed < target:
        start = int(rng.integers(0, t))
        sensor = int(rng.integers(0, n))
        duration = int(rng.integers(min_steps, max_steps + 1))
        end = min(start + duration, t)
        if segment is not None:
            same = np.flatnonzero(np.asarray(segment[start:end]) != segment[start])
            if same.size:
                end = start + int(same[0])
        cells = slice(start, end)
        removed += int(np.count_nonzero(observed[cells, sensor] & retained[cells, sensor]))
        retained[cells, sensor] = False
    return retained
