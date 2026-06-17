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


def test_smoke_training_command_is_single_batch_and_offline():
    command = build_training_command(
        cell=EXPERIMENT_CELLS[0],
        checkpoint_path="/ckpt/base.ckpt",
        output_dir="/tmp/out/baseline",
        cache_path="/cache/rap_ego",
        metric_cache_path="/cache/train_metric_cache",
        smoke=True,
    )

    joined = " ".join(command)
    assert command[:2] == ["env", "WANDB_MODE=offline"]
    assert "python" in command
    assert "navsim/planning/script/run_training.py" in command
    assert "agent=rap_ref2d_baseline" in command
    assert "agent.checkpoint_path=/ckpt/base.ckpt" in command
    assert "agent.config.train_metric_cache_path=/cache/train_metric_cache" in command
    assert "trainer.params.limit_train_batches=1" in joined
    assert "trainer.params.limit_val_batches=1" in joined
    assert "trainer.params.max_epochs=1" in joined
    assert "dataloader.params.batch_size=1" in joined
    assert "dataloader.params.num_workers=0" in joined
    assert "dataloader.params.prefetch_factor=null" in joined
