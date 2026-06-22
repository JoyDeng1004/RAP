from navsim.agents.rap_dino.recovery.experiment_matrix import (
    EXPERIMENT_CELLS,
    build_training_command,
    resolve_cells,
)


def test_experiment_matrix_has_required_six_cells():
    assert [cell.name for cell in EXPERIMENT_CELLS] == [
        "baseline",
        "shift_only_log",
        "recovery_only_log",
        "offset_recovery_log",
        "recovery_aux_only_log_l03",
        "offset_recovery_aux_log_l03",
    ]
    assert [cell.agent for cell in EXPERIMENT_CELLS] == [
        "rap_ref2d_baseline",
        "rap_ref2d_shift_only_log",
        "rap_ref2d_recovery_only_log",
        "rap_ref2d_offset_recovery_log",
        "rap_ref2d_recovery_aux_only_log_l03",
        "rap_ref2d_offset_recovery_aux_log_l03",
    ]


def test_smoke_training_command_defaults_to_wandb_disabled():
    command = build_training_command(
        cell=EXPERIMENT_CELLS[0],
        checkpoint_path="/ckpt/base.ckpt",
        output_dir="/tmp/out/baseline",
        cache_path="/cache/rap_ego",
        metric_cache_path="/cache/train_metric_cache",
        smoke=True,
    )

    joined = " ".join(command)
    assert command[:2] == ["env", "WANDB_MODE=disabled"]
    assert "WANDB_INIT_TIMEOUT=300" in command
    assert "python" in command
    assert "navsim/planning/script/run_training.py" in command
    assert "agent=rap_ref2d_baseline" in command
    assert "agent.checkpoint_path=/ckpt/base.ckpt" in command
    assert "++agent.config.train_metric_cache_path=/cache/train_metric_cache" in command
    assert "trainer.params.limit_train_batches=1" in joined
    assert "trainer.params.limit_val_batches=1" in joined
    assert "trainer.params.max_epochs=1" in joined
    assert "dataloader.params.batch_size=1" in joined
    assert "dataloader.params.num_workers=0" in joined
    assert "dataloader.params.prefetch_factor=null" in joined


def test_training_command_accepts_explicit_wandb_mode():
    online_command = build_training_command(
        cell=EXPERIMENT_CELLS[0],
        checkpoint_path="/ckpt/base.ckpt",
        output_dir="/tmp/out/baseline",
        cache_path="/cache/rap_ego",
        metric_cache_path="/cache/train_metric_cache",
        wandb_mode="online",
    )
    offline_command = build_training_command(
        cell=EXPERIMENT_CELLS[0],
        checkpoint_path="/ckpt/base.ckpt",
        output_dir="/tmp/out/baseline",
        cache_path="/cache/rap_ego",
        metric_cache_path="/cache/train_metric_cache",
        wandb_mode="offline",
    )

    assert online_command[:2] == ["env", "WANDB_MODE=online"]
    assert offline_command[:2] == ["env", "WANDB_MODE=offline"]
    assert "WANDB_INIT_TIMEOUT=300" in online_command
    assert "WANDB_INIT_TIMEOUT=300" in offline_command


def test_clean_cache_command_skips_extra_cache_overrides():
    command = build_training_command(
        cell=EXPERIMENT_CELLS[0],
        checkpoint_path="/ckpt/base.ckpt",
        output_dir="/tmp/out/baseline",
        cache_path="/cache/rap_ego",
        metric_cache_path="/cache/train_metric_cache",
        clean_cache=True,
    )
    joined = " ".join(command)

    assert "clean_cache_only=true" in command
    assert "cache_path=/cache/rap_ego" in command
    assert "cache_path_perturbed=" not in joined
    assert "cache_path_others=" not in joined


def test_training_command_accepts_experiment_prefix():
    command = build_training_command(
        cell=EXPERIMENT_CELLS[0],
        checkpoint_path="/ckpt/base.ckpt",
        output_dir="/tmp/out/baseline",
        cache_path="/cache/rap_ego",
        metric_cache_path="/cache/train_metric_cache",
        experiment_prefix="ref2d_e5",
    )

    assert "experiment_name=ref2d_e5_baseline" in command


def test_log_cells_enable_log_hash_shift_overrides():
    command = build_training_command(
        cell=EXPERIMENT_CELLS[1],
        checkpoint_path="/ckpt/base.ckpt",
        output_dir="/tmp/out/shift_only",
        cache_path="/cache/rap_ego",
        metric_cache_path="/cache/train_metric_cache",
    )

    assert "++agent.config.ref2d_shift_sampling_mode=log_hash" in command


def test_auxiliary_cells_set_recovery_aux_weight():
    command = build_training_command(
        cell=EXPERIMENT_CELLS[4],
        checkpoint_path="/ckpt/base.ckpt",
        output_dir="/tmp/out/recovery_aux",
        cache_path="/cache/rap_ego",
        metric_cache_path="/cache/train_metric_cache",
    )

    assert "++agent.config.ref2d_shift_sampling_mode=log_hash" in command
    assert "++agent.config.recovery_aux_enabled=true" in command
    assert "++agent.config.recovery_aux_weight=0.3" in command


def test_legacy_cell_names_resolve_to_log_cells():
    assert [cell.name for cell in resolve_cells("shift_only")] == ["shift_only_log"]
    assert [cell.name for cell in resolve_cells("recovery_only")] == ["recovery_only_log"]
    assert [cell.name for cell in resolve_cells("offset_recovery")] == ["offset_recovery_log"]


def test_ep10_launcher_covers_all_six_cells():
    script = open("scripts/ref2d_cells_ep10.sh", encoding="utf-8").read()

    for cell in [
        "baseline",
        "shift_only",
        "recovery_only",
        "offset_recovery",
        "recovery_aux_only_log_l03",
        "offset_recovery_aux_log_l03",
    ]:
        assert f"--cell {cell}" in script
