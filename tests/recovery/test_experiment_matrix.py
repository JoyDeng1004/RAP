from navsim.agents.rap_dino.recovery.experiment_matrix import (
    EXPERIMENT_CELLS,
    build_training_command,
)


def test_experiment_matrix_has_required_four_cells():
    assert [cell.name for cell in EXPERIMENT_CELLS] == [
        "baseline",
        "recovery_only",
        "shift_only",
        "offset_recovery",
    ]
    assert [cell.agent for cell in EXPERIMENT_CELLS] == [
        "rap_ref2d_baseline",
        "rap_ref2d_recovery_only",
        "rap_ref2d_shift_only",
        "rap_ref2d_offset_recovery",
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
