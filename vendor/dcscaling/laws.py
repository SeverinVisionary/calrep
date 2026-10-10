"""Scaling laws for repeated data.

The Chinchilla base law

    L(N, D) = E + A / N^alpha + B / D^beta

and the repetition forms the paper compares, each a function of model size N,
unique tokens U_D and repetitions R_D = epochs - 1:

    chinchilla      base law with every repeated token counted as fresh,
                    D = U_D (1 + R_D)
    muennighoff_d   effective data  D-hat = U_D (1 + R* (1 - exp(-R_D / R*)))
    muennighoff_dn  D-hat plus effective parameters N-hat (Muennighoff et al., 2023)
    additive_1p     chinchilla + P R_D N / U_D                        (paper eq. 6)
    additive_2p     chinchilla + P R_D (N / U_D)^kappa                (eq. 7)
    additive_4p     chinchilla + P R_D^delta (N / U_D^gamma)^kappa    (eq. 8)

Every form is written once, in ``predict_loss``, and that definition serves
fitting (torch tensors whose parameters carry gradients) as well as prediction
and allocation (numpy arrays or floats).
"""

from dataclasses import asdict, dataclass, field

import numpy as np

# The additive_4p penalty is masked to exactly zero at R_D = 0. R_D is also
# floored inside R_D^delta: at R_D = 0 the derivative in delta is 0 * log 0,
# which is NaN, and a NaN in the masked-out branch would still poison the
# gradient through the mask.
_R_D_FLOOR = 1e-12


@dataclass(frozen=True)
class Chinchilla:
    """L(N, D) = E + A / N^alpha + B / D^beta. Fields may be floats or tensors."""
    E: float
    A: float
    alpha: float
    B: float
    beta: float

    def predict(self, N, D):
        return self.E + self.A * N ** (-self.alpha) + self.B * D ** (-self.beta)

    def optimal(self, C):
        """Compute-optimal (N*, D*, L*) at training compute C = 6 N D, ignoring
        any limit on unique data."""
        A, alpha, B, beta = self.A, self.alpha, self.B, self.beta
        D = (beta * B * C ** alpha / (alpha * A * 6 ** alpha)) ** (1.0 / (alpha + beta))
        N = C / (6 * D)
        return N, D, self.E + A / N ** alpha + B / D ** beta

    def optimal_model_size(self, U_D):
        """Size at which D = U_D is compute-optimal: the U_N of Muennighoff et al."""
        return ((self.alpha * self.A / (self.beta * self.B)) ** (1.0 / self.alpha)
                * U_D ** (self.beta / self.alpha))

    def as_torch(self, dtype):
        import torch
        return Chinchilla(*(torch.tensor(v, dtype=dtype) for v in astuple_floats(self)))


def astuple_floats(base):
    return (base.E, base.A, base.alpha, base.B, base.beta)


@dataclass(frozen=True)
class Form:
    name: str
    label: str
    param_names: list
    log_params: frozenset = field(default_factory=frozenset)  # fit in log space (positive)
    init_grid: list = field(default_factory=list)  # multi-start inits, in param_names order


FORMS = {f.name: f for f in [
    Form("chinchilla", "Chinchilla (repeats as fresh data)", [], frozenset(), [()]),
    Form("muennighoff_d", "Muennighoff D-hat", ["R_star"], frozenset({"R_star"}),
         [(r,) for r in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]]),
    Form("muennighoff_dn", "Muennighoff D-hat, N-hat", ["R_star_D", "R_star_N"],
         frozenset({"R_star_D", "R_star_N"}),
         [(rd, rn) for rd in [0.5, 2, 5, 10, 20, 50] for rn in [0.5, 2, 5, 10, 20, 50]]),
    Form("additive_1p", "Additive penalty (1p)", ["P"], frozenset({"P"}),
         [(p,) for p in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]]),
    Form("additive_2p", "Additive penalty (2p)", ["P", "kappa"], frozenset({"P"}),
         [(p, k) for p in [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]
          for k in [0.2, 0.3, 0.5, 0.7, 1.0, 1.3]]),
    Form("additive_4p", "Additive penalty (4p)", ["P", "delta", "kappa", "gamma"],
         frozenset({"P"}),
         [(p, d, k, g) for p in [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
          for d in [0.5, 0.75, 1.0, 1.5]
          for k in [0.3, 0.5, 0.7, 1.0]
          for g in [0.3, 0.5, 0.7, 1.0]]),
]}

# The repetition forms, in the order the paper presents them.
FORM_ORDER = ["muennighoff_d", "muennighoff_dn", "additive_1p", "additive_2p", "additive_4p"]


def _xp(*xs):
    """torch if any argument is a tensor, else numpy."""
    for x in xs:
        if type(x).__module__.split(".")[0] == "torch":
            import torch
            return torch
    return np


def _floor(xp, x, lo):
    return x.clamp(min=lo) if xp is not np else np.maximum(x, lo)


def predict_loss(form, p, base, N, U_D, R_D):
    """Predicted loss of ``form`` with parameters ``p`` on top of ``base``.

    Works on floats, numpy arrays or torch tensors (then ``p`` and ``base`` may
    hold tensors too, which is how the fitter differentiates through it).
    """
    xp = _xp(N, U_D, R_D, *p.values(), base.E)
    if form == "muennighoff_d":
        return base.predict(N, _effective_data(xp, U_D, R_D, p["R_star"]))
    if form == "muennighoff_dn":
        D_eff = _effective_data(xp, U_D, R_D, p["R_star_D"])
        U_N = xp.minimum(N, base.optimal_model_size(U_D))
        R_N = _floor(xp, N / U_N - 1.0, 0.0)
        N_eff = U_N * (1.0 + p["R_star_N"] * (1.0 - xp.exp(-R_N / p["R_star_N"])))
        return base.predict(N_eff, D_eff)

    fresh = base.predict(N, U_D * (1.0 + R_D))
    if form == "chinchilla":
        return fresh
    if form == "additive_1p":
        return fresh + p["P"] * R_D * (N / U_D)
    if form == "additive_2p":
        return fresh + p["P"] * R_D * (N / U_D) ** p["kappa"]
    if form == "additive_4p":
        penalty = (p["P"] * _floor(xp, R_D, _R_D_FLOOR) ** p["delta"]
                   * (N / U_D ** p["gamma"]) ** p["kappa"])
        return fresh + xp.where(xp.asarray(R_D) > 0, penalty, 0.0)
    raise ValueError(f"Unknown form {form!r}; expected one of {sorted(FORMS)}")


def _effective_data(xp, U_D, R_D, R_star):
    return U_D * (1.0 + R_star * (1.0 - xp.exp(-R_D / R_star)))


def _as_input(x):
    if np.isscalar(x) or _xp(x) is not np:
        return x
    return np.asarray(x, dtype=float)


@dataclass(frozen=True)
class RepetitionLaw:
    """A fitted law: a form, its Chinchilla base, and the form's parameters."""
    form: str
    base: Chinchilla
    params: dict

    def __post_init__(self):
        if self.form not in FORMS:
            raise ValueError(f"Unknown form {self.form!r}; expected one of {sorted(FORMS)}")
        want = FORMS[self.form].param_names
        if sorted(self.params) != sorted(want):
            raise ValueError(f"{self.form} takes parameters {want}, got {sorted(self.params)}")

    def predict(self, N, U_D, epochs):
        """Predicted final loss of an N-parameter model trained for ``epochs``
        passes over U_D unique tokens."""
        N, U_D, epochs = (_as_input(x) for x in (N, U_D, epochs))
        return predict_loss(self.form, self.params, self.base, N, U_D, epochs - 1)

    def to_dict(self):
        return {"form": self.form, "base": asdict(self.base), "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["form"], Chinchilla(**d["base"]), dict(d["params"]))
