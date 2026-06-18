from pathlib import Path


def test_run_training_has_clean_cache_gate_for_extra_cache_mixing():
    source = Path("navsim/planning/script/run_training.py").read_text()

    assert 'cfg.get("clean_cache_only", False)' in source
    clean_gate_idx = source.index('cfg.get("clean_cache_only", False)')
    concat_idx = source.index("ConcatDataset([train_data, train_data_perturbed, train_data_others])")

    assert clean_gate_idx < concat_idx
