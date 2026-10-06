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
        {"seeds": (0, 1, 2, 3, 4, 3)},  # e.g. --seeds 0-4 3: seed 3 would run twice
        {"targets": (0, 0)},
        {"n_joints": (7, 7)},
        {"pop_size": 0},
        {"mutation_prob": 1.5},
        {"refine_method": "newton"},  # only plain / adam / basin
        {"refine_method": ""},
        {"position_shapes": 0},  # --position-ga must fine-tune at least one shape
        {"step_sizes": ()},
        {"step_sizes": (4e-4, 0)},
    ],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        Config(**bad)


def test_refine_method_defaults_to_todays_behavior():
    """New descent rules must be opt-in so existing scores are reproducible."""
    assert Config().refine_method == "plain"
    for cfg in PRESETS.values():
        assert cfg.refine_method == "plain"
    assert preset("quick", refine_method="adam").refine_method == "adam"


def test_position_ga_settings_default_to_todays_behavior():
    """The positions-only GA is opt-in: a run that doesn't ask for it is unchanged."""
    assert Config().position_shapes == 15
    assert Config().position_dedup is False
    for cfg in PRESETS.values():
        assert cfg.position_shapes == 15 and cfg.position_dedup is False
    assert preset("quick", position_shapes=3).position_shapes == 3
    assert preset("quick", position_dedup=True).position_dedup is True


def test_unknown_preset_or_setting_is_rejected():
    with pytest.raises(ValueError):
        preset("medium")
    with pytest.raises(ValueError):
        preset("quick", n_generations=5)  # typo for n_gen
