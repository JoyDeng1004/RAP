from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ExperimentCell:
    name: str
    agent: str


EXPERIMENT_CELLS = [
    ExperimentCell("baseline", "rap_ref2d_baseline"),
    ExperimentCell("recovery_only", "rap_ref2d_recovery_only"),
    ExperimentCell("shift_only", "rap_ref2d_shift_only"),
    ExperimentCell("offset_recovery", "rap_ref2d_offset_recovery"),
]


def build_training_command(
    *,
    cell: ExperimentCell,
    checkpoint_path: str,
    output_dir: str,
    cache_path: str,
    metric_cache_path: str,
    cache_path_perturbed: Optional[str] = None,
    cache_path_others: Optional[str] = None,
    seed: int = 0,
    smoke: bool = False,
    train_batches: int = 1,
    val_batches: int = 1,
    batch_size: int = 1,
    max_epochs: int = 1,
    python: str = "python",
) -> List[str]:
    checkpoint_dir = str(Path(output_dir) / "checkpoints")
    command = [
        "env",
        "WANDB_MODE=disabled",
        "RAP_DINO_OFFLINE_INIT=1",
        "MPLCONFIGDIR=/gs/bs/tga-RLA/qdeng/RAP/tmp",
        "RAP_CHECKPOINT_MONITOR=train/loss",
        "RAP_CHECKPOINT_MODE=min",
        f"RAP_CHECKPOINT_DIR={checkpoint_dir}",
        python,
        "navsim/planning/script/run_training.py",
        f"agent={cell.agent}",
        "dataset=navsim_dataset",
        "train_test_split=navtrain",
        "use_cache_without_dataset=true",
        "force_cache_computation=false",
        f"cache_path={cache_path}",
        f"cache_path_perturbed={cache_path_perturbed or cache_path}",
        f"cache_path_others={cache_path_others or cache_path}",
        f"agent.checkpoint_path={checkpoint_path}",
        f"++agent.config.train_metric_cache_path={metric_cache_path}",
        "agent.config.trajectory_sampling.time_horizon=5",
        f"seed={seed}",
        f"experiment_name=ref2d_{cell.name}_smoke" if smoke else f"experiment_name=ref2d_{cell.name}",
        f"output_dir={output_dir}",
        f"trainer.params.max_epochs={max_epochs}",
        f"trainer.params.limit_train_batches={train_batches}",
        f"trainer.params.limit_val_batches={val_batches}",
        "trainer.params.num_sanity_val_steps=0",
        "trainer.params.check_val_every_n_epoch=1",
        "trainer.params.accelerator=auto",
        "trainer.params.strategy=auto",
        "+trainer.params.devices=1",
        "trainer.params.precision=32",
        "trainer.params.fast_dev_run=false",
        f"dataloader.params.batch_size={batch_size}",
        "dataloader.params.num_workers=0",
        "dataloader.params.pin_memory=false",
        "dataloader.params.prefetch_factor=null",
        "dataloader.params.drop_last=false",
    ]
    if smoke:
        command.insert(4, "RAP_DISABLE_CHECKPOINT=1")
    return command


def iter_commands(args: argparse.Namespace) -> Iterable[List[str]]:
    cells = EXPERIMENT_CELLS
    if args.cell != "all":
        cells = [cell for cell in EXPERIMENT_CELLS if cell.name == args.cell]

    for cell in cells:
        yield build_training_command(
            cell=cell,
            checkpoint_path=args.checkpoint_path,
            output_dir=str(Path(args.output_dir) / cell.name),
            cache_path=args.cache_path,
            cache_path_perturbed=args.cache_path_perturbed,
            cache_path_others=args.cache_path_others,
            metric_cache_path=args.metric_cache_path,
            seed=args.seed,
            smoke=args.smoke,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            python=args.python,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run the ref2d recovery experiment matrix.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--metric-cache-path", required=True)
    parser.add_argument("--cache-path-perturbed", default=None)
    parser.add_argument("--cache-path-others", default=None)
    parser.add_argument("--output-dir", default="outputs/ref2d_matrix")
    parser.add_argument("--cell", choices=["all"] + [cell.name for cell in EXPERIMENT_CELLS], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-batches", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--python", default="python")
    parser.add_argument("--run", action="store_true", help="Run commands instead of printing them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for command in iter_commands(args):
        printable = " ".join(shlex.quote(part) for part in command)
        print(printable, flush=True)
        if args.run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
