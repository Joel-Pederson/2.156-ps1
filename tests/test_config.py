import pytest

from linkopt.config import PRESETS, Config, preset


def test_presets_are_valid_and_cover_all_three_kangaroos():
    assert set(PRESETS) == {"smoke", "quick", "full"}
    for cfg in PRESETS.values():
        assert cfg.targets == (0, 1, 2)
        assert all(n <= 20 for n in cfg.n_joints)


def test_preset_overrides():
    cfg = preset("quick", seeds=[0, 1, 2], n_gen=5)
    assert cfg.seeds == (0, 1, 2) and cfg.n_gen == 5
    assert PRESETS["quick"].seeds == (0,)  # the preset itself is unchanged


@pytest.mark.parametrize(
    "bad",
    [
        {"n_joints": (21,)},  # notebook: no more than 20 joints
        {"n_joints": ()},
        {"targets": (3,)},  # only Kangaroo 1..3 (indices 0..2)
        {"seeds": ()},
        {"pop_size": 0},
        {"mutation_prob": 1.5},
        {"step_sizes": ()},
        {"step_sizes": (4e-4, 0)},
    ],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        Config(**bad)


def test_unknown_preset_or_setting_is_rejected():
    with pytest.raises(ValueError):
        preset("medium")
    with pytest.raises(ValueError):
        preset("quick", n_generations=5)  # typo for n_gen
