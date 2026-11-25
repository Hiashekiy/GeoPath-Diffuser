"""GeoPath-Diffuser: safe conditional diffusion trajectory planning.

The package is organised around the data flow of the design document
(``docs/安全条件扩散轨迹规划方案_Conditional-DiT网络修正版.md``):

    map_encoder  -> frozen PathSafe ResUNets, bottleneck + full features
    guidance     -> Astar-Diffusion weighted guidance-field search
    safety       -> PathSafe-EllipseNet ellipses + local convex corridors
    trajectory   -> B-spline control-point representation
    diffusion    -> DDIM schedule, control-space ALM, conditional-DiT sampler
    models       -> the new Conditional DiT-S backbone
    training     -> dual-branch joint training (one backward per iteration)
    evaluation   -> metrics and visualisation

Modules copied from ``references/`` keep their upstream algorithm bodies; every
adaptation is recorded in ``docs/module_provenance.md``.
"""

__version__ = "0.1.0"
