"""Every setting for an optimization run, in one place.

Start from a preset and override single fields:

    cfg = preset("quick")                    # a real but short run
    cfg = preset("quick", seeds=(0, 1, 2))   # same, with three GA seeds

Presets:
    smoke   "does it run?" -- tiny, about a minute; used by CI. Not for scoring.
    quick   a real but short run while experimenting (minutes).
    full    serious runs for the final submission (up to overnight+).
"""

from dataclasses import asdict, dataclass, fields, replace

from LINKS.CP import MAX_JOINTS, N_PROBLEMS

MIN_JOINTS = 5  # smallest size we've checked MechanismRandomizer can generate


@dataclass(frozen=True)
class Config:
    # These defaults ARE the "quick" preset. "smoke" and "full" (PRESETS, below) also
    # start from them, so changing a default here changes every preset that doesn't
    # override that field. To tune just one preset, edit its entry in PRESETS.

    # What to optimize. One GA job runs per (target, n_joints, seed) combination.
    targets: tuple[int, ...] = (0, 1, 2)  # kangaroos: 0 = Kangaroo 1 ... 2 = Kangaroo 3
    n_joints: tuple[int, ...] = (7,)  # mechanism sizes to try (at most 20 joints)
    seeds: tuple[int, ...] = (0,)  # more seeds = more independent runs = more designs

    # Starting population: random valid mechanisms from MechanismRandomizer.
    n_start: int = 50

    # GA (NSGA-II over connectivity, positions, fixed joints and target joint).
    pop_size: int = 50  # designs per generation
    n_gen: int = 30  # generations
    # Chance that a design is mutated: higher explores more, lower keeps children
    # closer to their parents. None = pymoo's mixed-variable defaults (0.9 for positions/target, 1.0 for the
    # yes/no switches), which is what the advanced notebook actually runs: its
    # PolynomialMutation(prob=0.5) is ignored because it also passes its own mating.
    mutation_prob: float | None = None

    # Gradient refinement (DifferentiableTools), applied to the GA's designs.
    grad_steps: int = 200  # maximum number of steps
    # Step sizes to try. Each design is refined once per size and keeps whichever
    # gave it the lowest distance: no single size suits every design (4e-4, the
    # notebook's, does best on GA designs; the baseline's Kangaroo 1 designs only
    # improve with smaller steps, 0.43 -> 1.07 hypervolume at 3e-5).
    step_sizes: tuple[float, ...] = (4e-4, 1e-4, 3e-5)

    # Execution.
    n_workers: int = 3  # parallel worker processes

    def __post_init__(self):
        for name in ("targets", "n_joints", "seeds", "step_sizes"):  # lists -> tuples
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.targets or any(not 0 <= t < N_PROBLEMS for t in self.targets):
            raise ValueError(
                f"targets must be in 0..{N_PROBLEMS - 1}, got {self.targets}"
            )
        if not self.n_joints or any(
            not MIN_JOINTS <= n <= MAX_JOINTS for n in self.n_joints
        ):
            raise ValueError(
                f"n_joints must be in {MIN_JOINTS}..{MAX_JOINTS} (the notebook allows "
                f"at most {MAX_JOINTS} joints), got {self.n_joints}"
            )
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        for name in ("n_start", "pop_size", "n_gen", "n_workers"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )
        if self.grad_steps < 0:
            raise ValueError(f"grad_steps must be >= 0, got {self.grad_steps}")
        if not self.step_sizes or any(s <= 0 for s in self.step_sizes):
            raise ValueError(f"step_sizes must be positive, got {self.step_sizes}")
        if self.mutation_prob is not None and not 0 <= self.mutation_prob <= 1:
            raise ValueError(
                f"mutation_prob must be in [0, 1], got {self.mutation_prob}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


# Each preset is Config's defaults plus the fields listed; anything not listed keeps
# its default. "quick" lists nothing, so it is exactly the defaults.
PRESETS = {
    "smoke": Config(n_start=16, pop_size=16, n_gen=3, grad_steps=10, n_workers=1),
    "quick": Config(),
    "full": Config(
        n_joints=(6, 7, 8),
        seeds=(0, 1, 2, 3, 4),
        n_start=200,
        pop_size=200,
        n_gen=150,
        grad_steps=1000,
    ),
}


def preset(name: str, **overrides) -> Config:
    """A preset with some fields overridden, e.g. preset("quick", n_gen=50)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    unknown = set(overrides) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}")
    return replace(PRESETS[name], **overrides)
