"""Safety package: frozen PathSafe encoder, ellipse generation, convex regions."""

__all__ = [
    "PathSafeEllipseNet",
    "load_pathsafe",
    "SafetyConditionGenerator",
    "build_local_constraint_pack",
]


def __getattr__(name):  # lazy: importing torch-heavy modules on demand
    if name in ("PathSafeEllipseNet", "load_pathsafe", "SafetyConditionGenerator"):
        from . import safety_features
        return getattr(safety_features, name)
    if name == "build_local_constraint_pack":
        from .local_regions import build_local_constraint_pack
        return build_local_constraint_pack
    raise AttributeError(name)
