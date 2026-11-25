"""Training package: synthetic dataset, offline safety cache, joint trainer."""
from .dataset import SyntheticTrajectoryDataset, collate_scenes, make_scene
from .dual_branch_trainer import DualBranchTrainer

__all__ = ["SyntheticTrajectoryDataset", "collate_scenes", "make_scene",
           "DualBranchTrainer"]
