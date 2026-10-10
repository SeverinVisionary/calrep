"""Fitting the laws in ``dcscaling.laws`` to training runs.

Two phases, as in the paper:

1. ``fit_chinchilla`` fits E, A, alpha, B, beta on single-epoch runs.
2. ``fit_repetition`` locks that base and fits one repetition form on all runs.

Both minimize a summed Huber loss on log-loss (delta = 1e-3) with L-BFGS from
a grid of starting points, and keep the best. ``bootstrap`` refits a form on
resampled runs to estimate parameter uncertainty.

Runs are passed as a DataFrame with columns ``N`` (parameters), ``U_D``
(unique tokens), ``epochs`` and ``val_loss`` -- the shape
``dcscaling.runs.load_arm`` returns.
"""

import multiprocessing as mp
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from dcscaling.laws import FORMS, Chinchilla, RepetitionLaw, astuple_floats, predict_loss

# Starting points for the Chinchilla fit: a 4 x 3 x 3 x 3 x 3 grid over
# (E, A, alpha, B, beta).
CHINCHILLA_GRID = [(E, A, al, B, be)
                   for E in [1.5, 2.0, 2.5, 3.0]
                   for A in [1.0, 10.0, 100.0]
                   for al in [0.3, 0.5, 0.7]
                   for B in [1.0, 10.0, 100.0]
                   for be in [0.3, 0.5, 0.7]]


# ============================================================
# Objective and metrics
# ============================================================

def huber_log_loss(y_pred, y_true, delta=1e-3):
    """Huber loss between log predictions and log targets, summed over runs."""
    return nn.functional.huber_loss(torch.log(y_pred + 1e-8), torch.log(y_true + 1e-8),
                                    reduction="sum", delta=delta)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def adj_r_squared(r2, n_data, n_params):
    return 1 - (1 - r2) * (n_data - 1) / (n_data - n_params - 1)


# ============================================================
# Multi-start L-BFGS
# ============================================================

def is_step_overflow(error):
    """Whether ``error`` is L-BFGS stepping a float32 parameter past its range.

    Older torch lets such a step overflow to inf, so the start ends non-finite;
    newer torch raises instead. Either way the start has diverged.
    """
    return "cannot be converted to type float without overflow" in str(error)


def multistart(make_model, inits, X, y, max_iter=500):
    """Fit ``make_model(*init)`` from every init; return (best model, its loss).

    A start that diverges -- ends at a non-finite loss, or overflows float32
    in the line search -- is skipped. Any other exception is a bug and
    propagates. Returns (None, inf) if no start ends finite.
    """
    best_model, best_loss = None, float("inf")
    for init in inits:
        model = make_model(*init)
        opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = huber_log_loss(model(X), y)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except RuntimeError as e:
            if not is_step_overflow(e):
                raise
            continue
        final = closure().item()
        if np.isfinite(final) and final < best_loss:
            best_model, best_loss = model, final
    return best_model, best_loss


def _columns(df):
    """(N, U_D, R_D, val_loss) as float arrays; R_D derived from epochs."""
    R_D = (df["epochs"].to_numpy() - 1).astype(float)
    return (df["N"].to_numpy(dtype=float), df["U_D"].to_numpy(dtype=float), R_D,
            df["val_loss"].to_numpy(dtype=float))


# ============================================================
# Phase 1: Chinchilla
# ============================================================

class _ChinchillaModule(nn.Module):
    def __init__(self, E, A, alpha, B, beta):
        super().__init__()
        for name, v in (("E", E), ("A", A), ("alpha", alpha), ("B", B), ("beta", beta)):
            setattr(self, f"log_{name}", nn.Parameter(torch.tensor(np.log(v), dtype=torch.float32)))

    def base(self):
        return Chinchilla(*(torch.exp(getattr(self, f"log_{k}"))
                            for k in ("E", "A", "alpha", "B", "beta")))

    def forward(self, X):
        return self.base().predict(X[:, 0], X[:, 1])


@dataclass
class ChinchillaFit:
    base: Chinchilla
    huber_loss: float
    r_squared: float
    adj_r_squared: float
    n_data: int

    def to_dict(self):
        return {"params": {k: getattr(self.base, k) for k in ("E", "A", "alpha", "B", "beta")},
                "r_squared": self.r_squared, "adj_r_squared": self.adj_r_squared,
                "huber_loss": self.huber_loss, "n_data": self.n_data}


def fit_chinchilla(df):
    """Fit the Chinchilla law to single-epoch runs (D = U_D * epochs)."""
    N, U_D, R_D, loss = _columns(df)
    D = U_D * (1.0 + R_D)
    X = torch.tensor(np.column_stack([N, D]), dtype=torch.float32)
    y = torch.tensor(loss, dtype=torch.float32)
    model, huber = multistart(_ChinchillaModule, CHINCHILLA_GRID, X, y, max_iter=200)
    if model is None:
        raise RuntimeError("Chinchilla fit ended non-finite from every starting point")
    base = Chinchilla(*(v.item() for v in astuple_floats(model.base())))
    r2 = r_squared(loss, base.predict(N, D))
    return ChinchillaFit(base, huber, r2, adj_r_squared(r2, len(loss), 5), len(loss))


# ============================================================
# Phase 2: a repetition form on a locked base
# ============================================================

class _FormModule(nn.Module):
    """One repetition form with its base locked. Parameters in ``log_params``
    are optimized in log space to keep them positive."""

    def __init__(self, form, base, *init):
        super().__init__()
        self.form = form
        spec = FORMS[form]
        for name, v in zip(spec.param_names, init):
            if name in spec.log_params:
                setattr(self, f"log_{name}", nn.Parameter(torch.tensor(np.log(v), dtype=torch.float32)))
            else:
                setattr(self, name, nn.Parameter(torch.tensor(v, dtype=torch.float32)))
        for name, v in zip(("E", "A", "alpha", "B", "beta"), astuple_floats(base)):
            self.register_buffer(name, torch.tensor(v, dtype=torch.float32))

    def values(self):
        return _Values(self)

    def forward(self, X):
        base = Chinchilla(self.E, self.A, self.alpha, self.B, self.beta)
        return predict_loss(self.form, self.values(), base, X[:, 0], X[:, 1], X[:, 2])


class _Values(Mapping):
    """A form's parameter values, recomputed from the raw parameters on every
    access rather than once per forward pass.

    This keeps the autograd graph identical to the fitter behind the committed
    fits, so on the same machine and library versions a refit follows the same
    optimization path. It matters for muennighoff_dn:
    its objective is nearly flat in R*_N, and last-bit differences in the
    gradient (from summing two uses of one exp rather than two exps) steer
    L-BFGS to a different point along that valley.
    """

    def __init__(self, module):
        self._module = module
        self._spec = FORMS[module.form]

    def __getitem__(self, k):
        if k not in self._spec.param_names:
            raise KeyError(k)
        if k in self._spec.log_params:
            return torch.exp(getattr(self._module, f"log_{k}"))
        return getattr(self._module, k)

    def __iter__(self):
        return iter(self._spec.param_names)

    def __len__(self):
        return len(self._spec.param_names)


@dataclass
class RepetitionFit:
    law: RepetitionLaw
    huber_loss: float
    r_squared: float
    adj_r_squared: float
    r_squared_multi: float
    n_data: int
    n_multi: int

    @property
    def n_free_params(self):
        return len(self.law.params)

    def to_dict(self):
        return {"form": self.law.form, "params": dict(self.law.params),
                "huber_loss": self.huber_loss, "r_squared": self.r_squared,
                "adj_r_squared": self.adj_r_squared, "r_squared_multi": self.r_squared_multi,
                "n_data": self.n_data, "n_multi": self.n_multi,
                "n_free_params": self.n_free_params}


def _fit_params(form, base, N, U_D, R_D, loss):
    """Multi-start fit of one form: (params, huber loss), or None."""
    X = torch.tensor(np.column_stack([N, U_D, R_D]), dtype=torch.float32)
    y = torch.tensor(loss, dtype=torch.float32)
    model, huber = multistart(lambda *init: _FormModule(form, base, *init),
                              FORMS[form].init_grid, X, y)
    if model is None:
        return None
    return {k: v.item() for k, v in model.values().items()}, huber


def fit_repetition(form, df, base):
    """Fit one repetition form on all runs with the Chinchilla ``base`` locked."""
    N, U_D, R_D, loss = _columns(df)
    fitted = _fit_params(form, base, N, U_D, R_D, loss)
    if fitted is None:
        raise RuntimeError(f"{form}: fit ended non-finite from every starting point")
    params, huber = fitted
    law = RepetitionLaw(form, base, params)
    pred = predict_loss(form, params, base, N, U_D, R_D)
    multi = R_D > 0
    r2 = r_squared(loss, pred)
    return RepetitionFit(
        law=law, huber_loss=huber, r_squared=r2,
        adj_r_squared=adj_r_squared(r2, len(loss), len(params)),
        r_squared_multi=r_squared(loss[multi], pred[multi]) if multi.sum() > 1 else float("nan"),
        n_data=len(loss), n_multi=int(multi.sum()))


# ============================================================
# Bootstrap
# ============================================================

_BOOT = {}


def stratified_resamples(N, n_bootstrap, seed):
    """Row indices of each resample, drawn with replacement within each model
    size so that every size is represented in every resample."""
    rng = np.random.RandomState(seed)
    strata = [np.where(N == n)[0] for n in np.unique(N)]
    return [np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            for _ in range(n_bootstrap)]


def _boot_init(state):
    torch.set_num_threads(1)
    _BOOT.update(state)


def _boot_one(task):
    form, idx = task
    s = _BOOT
    fitted = _fit_params(form, s["base"], s["N"][idx], s["U_D"][idx], s["R_D"][idx], s["loss"][idx])
    return None if fitted is None else fitted[0]


def scaled_mad(vals):
    """1.4826 * the median absolute deviation: the paper's ± (it equals the
    standard deviation for normally distributed values)."""
    vals = np.asarray(vals, dtype=float)
    return float(1.4826 * np.median(np.abs(vals - np.median(vals))))


def bootstrap(forms, df, base, n_bootstrap=250, seed=42, workers=1):
    """Stratified pairs bootstrap of the Phase 2 fit, with the base held fixed.

    Every resample is refit from the form's full starting grid, exactly as the
    point estimate is.

    Returns {form: {"n_bootstrap", "n_success", "seed", "mad": {name: value},
    "draws": {name: [...]}}}: the scaled MAD of each parameter, and the refit
    parameters of every successful resample for any other statistic.
    """
    N, U_D, R_D, loss = _columns(df)
    resamples = stratified_resamples(N, n_bootstrap, seed)
    tasks = [(form, idx) for form in forms for idx in resamples]
    state = dict(base=base, N=N, U_D=U_D, R_D=R_D, loss=loss)
    if workers > 1:
        # spawn, not fork: torch refuses to fork once the parent has run
        # autograd with CUDA available, and spawn also works on macOS/Windows.
        with mp.get_context("spawn").Pool(workers, initializer=_boot_init,
                                          initargs=(state,)) as pool:
            fits = pool.map(_boot_one, tasks, chunksize=1)
    else:
        _BOOT.update(state)
        fits = [_boot_one(t) for t in tasks]

    out = {}
    for i, form in enumerate(forms):
        draws = fits[i * n_bootstrap:(i + 1) * n_bootstrap]
        ok = [d for d in draws if d is not None and all(np.isfinite(list(d.values())))]
        if len(ok) < 2:
            raise RuntimeError(f"{form}: fewer than two bootstrap resamples fit")
        names = FORMS[form].param_names
        out[form] = {"n_bootstrap": n_bootstrap, "n_success": len(ok), "seed": seed,
                     "mad": {k: scaled_mad([d[k] for d in ok]) for k in names},
                     "draws": {k: [d[k] for d in ok] for k in names}}
    return out
