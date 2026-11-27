"""Map-feature acceptance tests (design document section 13.1)."""
from __future__ import annotations

import torch

from gp_diffuser.vendor.ellipsenet.models.resunet import ResUNet


def test_bottleneck_and_full_shapes(pipeline, sample):
    item, _ = sample
    occ = item["occupancy"][None]
    sdf = item["sdf"][None]
    feats = pipeline["map_encoder"].encode(occ, sdf)
    assert tuple(feats["occ_bottleneck"].shape) == (1, 192, 32, 32)
    assert tuple(feats["sdf_bottleneck"].shape) == (1, 192, 32, 32)
    assert tuple(feats["occ_full"].shape) == (1, 128, 256, 256)
    assert tuple(feats["sdf_full"].shape) == (1, 128, 256, 256)


def test_bottleneck_is_enc3_not_decoder_output(pipeline, sample):
    """``bottleneck`` must be the pre-upsampling ``enc3`` output."""
    item, _ = sample
    net = pipeline["net"]
    occ = item["occupancy"][None]
    from gp_diffuser.map_encoder.frozen_encoder import _resunet_features

    with torch.no_grad():
        out = _resunet_features(net.occ_unet, occ)
        manual = net.occ_unet.enc3(net.occ_unet.down3(
            net.occ_unet.enc2(net.occ_unet.down2(net.occ_unet.enc1(
                net.occ_unet.down1(net.occ_unet.stem(occ)))))))
    assert torch.allclose(out["bottleneck"], manual)
    assert out["bottleneck"].shape[-2:] == (32, 32)
    assert out["full"].shape[-2:] == (256, 256)


def test_encoder_runs_exactly_once_per_call(pipeline, sample, monkeypatch):
    item, _ = sample
    calls = {"occ": 0, "sdf": 0}
    original = ResUNet.forward

    def counting(self, x):
        calls["occ" if self is pipeline["net"].occ_unet else "sdf"] += 1
        return original(self, x)

    monkeypatch.setattr(ResUNet, "forward", counting)
    pipeline["map_encoder"].encode(item["occupancy"][None], item["sdf"][None])
    assert calls == {"occ": 1, "sdf": 1}


def test_encoder_is_frozen(pipeline):
    net = pipeline["net"]
    assert all(not p.requires_grad for p in net.parameters())
    assert pipeline["map_encoder"].trainable_parameter_count() == 0
    assert not net.training


def test_weights_come_from_the_pretrained_checkpoint(pipeline):
    net = pipeline["net"]
    meta = net.checkpoint_meta
    assert meta["params"] == 5041093
    assert meta["path"].endswith("v3_best.pt")
    assert net.occ_unet.stem[0].weight.shape == (24, 1, 3, 3)
    assert net.occ_unet.head.weight.shape == (128, 24, 1, 1)


def test_encoder_weights_unchanged_by_a_training_step(pipeline, cfg, sample):
    """A full joint iteration must not touch the frozen encoders."""
    import numpy as np
    from gp_diffuser.training.dataset import SyntheticTrajectoryDataset
    from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer
    from gp_diffuser.training.safety_cache import build_safety_cache, load_safety_cache

    item, _ = sample
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    before = {k: v.clone() for k, v in pipeline["net"].occ_unet.state_dict().items()}

    cache_name = "train_pytest"
    cfg_local = dict(cfg)
    cfg_local["dataset"] = dict(cfg["dataset"])
    cfg_local["dataset"]["cache_dir"] = "outputs/cache"
    build_safety_cache(dataset, pipeline["net"], pipeline["safety_generator"],
                       pipeline["guidance"], cfg_local, split=cache_name)
    cache = load_safety_cache(cfg_local, cache_name)
    assert cache is not None

    from gp_diffuser.factory import (build_codec, build_map_condition_encoder,
                                     build_model, build_schedule)
    torch.manual_seed(0)
    trainer = DualBranchTrainer(cfg_local, build_model(cfg_local),
                                build_map_condition_encoder(cfg_local),
                                pipeline["map_encoder"], build_codec(cfg_local),
                                build_schedule(cfg_local), device="cpu",
                                scene_scale=pipeline["transform"].scale)
    items = [dataset[i] for i in range(len(dataset))]
    trainer.train_step(items, cache, np.arange(len(items)))
    after = pipeline["net"].occ_unet.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), "frozen encoder weight changed: %s" % key


def test_diffusion_only_sees_bottleneck_channels(pipeline):
    """The map encoder consumes the two 192-channel bottlenecks, nothing else."""
    encoder = pipeline["map_condition_encoder"]
    assert encoder.occ_adapter.conv.in_channels == 192
    assert encoder.sdf_adapter.conv.in_channels == 192
    assert encoder.fusion.depthwise.in_channels == 384      # 192 + 192 after concat
    mem = encoder(torch.zeros(1, 192, 32, 32), torch.zeros(1, 192, 32, 32))
    assert tuple(mem.shape) == (1, 1024, 384)


def test_no_second_map_encoder_in_the_diffusion_model(pipeline):
    model = pipeline["model"]
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs == [], "the DiT must not contain a convolutional map encoder"
    names = {name for name, _ in model.named_modules()}
    assert not any("scene_cnn" in n or "map_encoder" in n for n in names)
