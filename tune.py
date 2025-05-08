# tune.py

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from torch.utils.data import DataLoader

from datasets.dataloader import get_dataloaders
from models.gaussian_model import GaussianRadioFieldModel
from render import render_channel
from utils.general_utils import set_random_seed
from utils.loss import calculate_snr

TUNING_ITERATIONS = 15_000
CKPT_FREQ = 1000


def tune_logging(
    log_dir: Path, use_rich: bool = True
) -> Tuple[logging.Logger, Console | None]:
    """Sets up logging with file handler and optional RichHandler for a trial."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "trial.log"

    logger = logging.getLogger(f"TrialLogger_{log_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console = None
    if use_rich:
        console = Console(log_path=False, force_terminal=True, record=True)
        console_handler = RichHandler(
            console=console, rich_tracebacks=True, markup=True, show_path=False
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    logger.info(f"Trial logging initialized. Log file: {log_file}")
    return logger, console


def evaluate_trial(
    model: GaussianRadioFieldModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tx_position: torch.Tensor,
    nt: int,
    nr: int,
    cfg: DictConfig,
) -> Dict[str, float]:
    """Evaluates the model on the validation set for a tuning trial."""
    model.eval()
    total_loss = 0.0
    total_snr = 0.0
    count = 0

    snr_eps = cfg.evaluation.snr_eps

    with torch.no_grad():
        for batch in val_loader:
            rx_pos_batch = batch["rx_position"].to(device)

            channel_gt_batch = batch["channel"].to(device)
            batch_size = rx_pos_batch.shape[0]

            channel_pred_batch = render_channel(
                rx_positions=rx_pos_batch,
                model=model,
                tx_position=tx_position,
                nt=nt,
                nr=nr,
                eps=snr_eps,
            )

            pred_mag = torch.abs(channel_pred_batch)
            gt_mag = torch.abs(channel_gt_batch)
            loss = criterion(pred_mag, gt_mag)

            snr = calculate_snr(loss, gt_mag, eps=snr_eps)

            total_loss += loss.item() * batch_size
            if not torch.isinf(snr) and not torch.isnan(snr):
                total_snr += snr.item() * batch_size

            count += batch_size

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_snr = total_snr / count if count > 0 else float("-inf")

    return {"val_mse_loss_mag": avg_loss, "val_snr_db_mag": avg_snr}


def objective(
    trial: optuna.trial.Trial,
    base_cfg: DictConfig,
    data_path: str,
    console: Console,
) -> float:
    """Optuna objective function for hyperparameter tuning."""

    trial_cfg = base_cfg.copy()
    with open_dict(trial_cfg):

        trial_cfg.training.iterations = TUNING_ITERATIONS
        trial_cfg.experiment.checkpoint_freq = CKPT_FREQ
        trial_cfg.experiment.tensorboard = False
        trial_cfg.experiment.log_grad_stats = False

        trial_cfg.initialization.num_gaussians = trial.suggest_int(
            "init.num_gaussians", 100, 2000, step=100
        )
        trial_cfg.initialization.opacity_value = trial.suggest_float(
            "init.opacity_value", 0.05, 0.2
        )
        trial_cfg.initialization.scale_value = trial.suggest_float(
            "init.scale_value", 0.005, 0.03
        )

        trial_cfg.model.latent_dim = trial.suggest_categorical(
            "model.latent_dim", [32, 64, 96, 128]
        )
        trial_cfg.model.attribute_network.hidden_dim = trial.suggest_categorical(
            "model.attr_net.hidden_dim", [64, 128, 192, 256]
        )
        trial_cfg.model.attribute_network.num_layers = trial.suggest_int(
            "model.attr_net.num_layers", 3, 8
        )
        trial_cfg.model.attribute_network.pos_enc_freqs = trial.suggest_int(
            "model.attr_net.pos_enc_freqs", 24, 96, step=4
        )
        trial_cfg.model.attribute_network.dropout_p = trial.suggest_float(
            "model.attr_net.dropout_p", 0.0, 0.3
        )
        trial_cfg.model.contribution_decoder.hidden_dim = trial.suggest_categorical(
            "model.decoder.hidden_dim", [32, 64, 96, 128]
        )
        trial_cfg.model.contribution_decoder.num_layers = trial.suggest_categorical(
            "model.decoder.num_layers", [2, 3, 4, 5]
        )
        trial_cfg.model.contribution_decoder.dropout_p = trial.suggest_float(
            "model.decoder.dropout_p", 0.0, 0.3
        )

        trial_cfg.training.batch_size = trial.suggest_categorical(
            "train.batch_size", [16, 32, 64]
        )
        trial_cfg.training.stop_xyz_iter_ratio = trial.suggest_float(
            "train.stop_xyz_ratio", 0.4, 1.0
        )
        trial_cfg.training.rx_noise_std = trial.suggest_float(
            "train.rx_noise_std", 0.0, 0.04
        )
        trial_cfg.training.lambda_activation_l1 = trial.suggest_float(
            "train.lambda_l1", 1e-3, 0.1, log=True
        )

        trial_cfg.training.optimizer.eps = trial.suggest_float(
            "train.opt.eps", 1e-9, 1e-6, log=True
        )
        trial_cfg.training.optimizer.weight_decay = trial.suggest_float(
            "train.opt.weight_decay", 1e-8, 1e-5, log=True
        )

        trial_cfg.training.learning_rate.position_init = trial.suggest_float(
            "train.lr.pos_init", 1e-4, 5e-3, log=True
        )
        trial_cfg.training.learning_rate.position_final = trial.suggest_float(
            "train.lr.pos_final", 1e-6, 5e-5, log=True
        )
        trial_cfg.training.learning_rate.rotation = trial.suggest_float(
            "train.lr.rotation", 5e-4, 5e-3, log=True
        )
        trial_cfg.training.learning_rate.scaling = trial.suggest_float(
            "train.lr.scaling", 5e-4, 8e-3, log=True
        )
        trial_cfg.training.learning_rate.attribute_net = trial.suggest_float(
            "train.lr.attr_net", 1e-4, 2e-3, log=True
        )
        trial_cfg.training.learning_rate.decoder = trial.suggest_float(
            "train.lr.decoder", 1e-4, 2e-3, log=True
        )

        trial_cfg.data.path = data_path

    trial_num = trial.number
    study_name = trial.study.study_name
    trial_log_dir = (
        Path(base_cfg.experiment.log_dir) / study_name / f"trial_{trial_num:04d}"
    )
    trial_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = trial_log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    logger, _ = tune_logging(trial_log_dir, use_rich=False)

    console.rule(f"[bold blue]Starting Optuna Trial {trial_num}[/bold blue]")
    console.print(f"Log directory: {trial_log_dir}")

    config_save_path = trial_log_dir / "trial_config.yaml"
    with open(config_save_path, "w") as f:
        OmegaConf.save(trial_cfg, f)
    logger.info(f"Trial config saved to {config_save_path}")

    logger.info(f"Trial {trial_num} Hyperparameters:")
    param_table = Table(
        title="Hyperparameters", show_header=True, header_style="bold magenta"
    )
    param_table.add_column("Parameter", style="dim", width=30)
    param_table.add_column("Value")
    for key, value in trial.params.items():
        logger.info(f"  {key}: {value}")
        param_table.add_row(key, str(value))
    console.print(param_table)

    set_random_seed(base_cfg.experiment.seed + trial_num)
    device = torch.device(
        base_cfg.experiment.device
        if torch.cuda.is_available() and base_cfg.experiment.device == "cuda"
        else "cpu"
    )
    console.print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    console.print("Loading data...")
    logger.info("Loading data...")
    try:
        train_loader, val_loader, metadata = get_dataloaders(cfg=trial_cfg)
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        tx_position = metadata["tx_position"].to(device)
        env_dims = metadata.get("env_dims")
        point_cloud = metadata.get("point_cloud")

        if env_dims is not None:
            env_dims = env_dims.to(device)
        if point_cloud is not None:
            point_cloud = point_cloud.to(device)

        console.print(f"[green]Dataset loaded:[/green] Nt={nt}, Nr={nr}")
        console.print(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )
        logger.info(
            f"Dataset loaded: Nt={nt}, Nr={nr}, Train batches={len(train_loader)}, Val batches={len(val_loader)}"
        )

        if len(train_loader) == 0:
            logger.error("Training dataloader is empty! Aborting trial.")
            console.print(
                "[bold red]Training dataloader is empty! Aborting trial.[/bold red]"
            )

            return -float("inf")
        if len(val_loader) == 0:
            logger.error(
                "Validation dataloader is empty! Cannot optimize. Aborting trial."
            )
            console.print(
                "[bold red]Validation dataloader is empty! Cannot optimize. Aborting trial.[/bold red]"
            )
            return -float("inf")

    except Exception as e:
        logger.exception(f"Failed to load data: {e}. Aborting trial.")
        console.print(f"[bold red]Failed to load data: {e}. Aborting trial.[/bold red]")
        return -float("inf")

    console.print("Initializing model...")
    logger.info("Initializing model...")
    try:
        model = GaussianRadioFieldModel(
            num_tx_ant=nt,
            num_rx_ant=nr,
            latent_dim=trial_cfg.model.latent_dim,
            attribute_hidden_dim=trial_cfg.model.attribute_network.hidden_dim,
            attribute_num_layers=trial_cfg.model.attribute_network.num_layers,
            attribute_pos_enc_freqs=trial_cfg.model.attribute_network.pos_enc_freqs,
            attribute_dropout_p=trial_cfg.model.attribute_network.dropout_p,
            decoder_hidden_dim=trial_cfg.model.contribution_decoder.hidden_dim,
            decoder_num_layers=trial_cfg.model.contribution_decoder.num_layers,
            decoder_dropout_p=trial_cfg.model.contribution_decoder.dropout_p,
            initial_gaussians=trial_cfg.initialization.num_gaussians,
            init_opacity_value=trial_cfg.initialization.opacity_value,
            init_scale_value=trial_cfg.initialization.scale_value,
            device=device,
        )
        logger.info(f"Model base initialized.")

        init_pc_arg = None
        init_method = trial_cfg.initialization.method
        if init_method == "point_cloud":
            if point_cloud is not None:
                logger.info("Using point cloud for Gaussian initialization")
                console.print("Using point cloud for Gaussian initialization")
                init_pc_arg = point_cloud
            else:
                logger.warning(
                    "Point cloud init requested but no data found. Using random."
                )
                console.print(
                    "[yellow]Point cloud init requested but no data found. Using random.[/yellow]"
                )
                init_method = "random"

        if init_method == "random":
            logger.info("Using random initialization for Gaussians")
            console.print("Using random initialization for Gaussians")
            model.init_gaussians(
                env_dims=env_dims if env_dims is not None else None,
                point_cloud=None,
                num_points=trial_cfg.initialization.num_gaussians,
            )
        elif init_method == "point_cloud":
            model.init_gaussians(
                env_dims=None,
                point_cloud=init_pc_arg,
                num_points=trial_cfg.initialization.num_gaussians,
            )
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        model.training_setup(trial_cfg)
        model = model.to(device)
        logger.info(f"Model initialized with {model.get_xyz.shape[0]} Gaussians")
        console.print(
            f"[green]Model initialized with {model.get_xyz.shape[0]:,} Gaussians[/green]"
        )

    except Exception as e:
        logger.exception(f"Failed to initialize model: {e}. Aborting trial.")
        console.print(
            f"[bold red]Failed to initialize model: {e}. Aborting trial.[/bold red]"
        )
        return -float("inf")

    criterion = nn.MSELoss().to(device)
    best_trial_val_snr = -float("inf")
    train_iter = iter(train_loader)

    console.print(
        f"Starting training loop for {trial_cfg.training.iterations} iterations..."
    )
    logger.info(
        f"Starting training loop for {trial_cfg.training.iterations} iterations..."
    )
    for iteration in range(trial_cfg.training.iterations):
        iter_start_time = time.time()
        model.train()
        model.update_learning_rate(iteration, trial_cfg)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        rx_pos_batch = batch["rx_position"].to(device)

        channel_gt_batch = batch["channel"].to(device)

        if trial_cfg.training.rx_noise_std > 0:
            noise = torch.randn_like(rx_pos_batch) * trial_cfg.training.rx_noise_std
            rx_pos_batch = rx_pos_batch + noise

        channel_pred_batch = render_channel(
            rx_positions=rx_pos_batch,
            model=model,
            tx_position=tx_position,
            nt=nt,
            nr=nr,
            eps=trial_cfg.training.snr_eps,
        )

        pred_mag = torch.abs(channel_pred_batch)
        gt_mag = torch.abs(channel_gt_batch)
        mse_loss = criterion(pred_mag, gt_mag)

        total_loss = mse_loss
        l1_activation_loss = torch.tensor(0.0, device=device)

        if trial_cfg.training.lambda_activation_l1 > 0 and model.get_xyz.shape[0] > 0:
            base_activations_logits = model.get_base_activation_logits(tx_position)
            l1_activation_loss = torch.mean(torch.abs(base_activations_logits))
            total_loss = (
                total_loss
                + trial_cfg.training.lambda_activation_l1 * l1_activation_loss
            )

        model.optimizer.zero_grad()
        total_loss.backward()

        found_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                logger.warning(
                    f"NaN/Inf gradient detected at iter {iteration} for '{name}'. Skipping step."
                )
                found_nan_grad = True
                break

        if not found_nan_grad:
            model.optimizer.step()
        else:
            model.optimizer.zero_grad()

        if (
            iteration % (trial_cfg.experiment.log_freq * 5) == 0
        ) or iteration == trial_cfg.training.iterations - 1:
            with torch.no_grad():
                current_loss = mse_loss.item()

                snr = calculate_snr(
                    mse_loss, gt_mag, eps=trial_cfg.training.snr_eps
                ).item()
                iter_time = time.time() - iter_start_time
                log_msg = (
                    f"T{trial_num} [{iteration}/{trial_cfg.training.iterations}] "
                    f"Loss={current_loss:.4e} | SNR={snr:.2f} dB | Time={iter_time:.3f}s"
                )

                console.print(log_msg)
                logger.info(log_msg)

        if (
            iteration % trial_cfg.experiment.eval_freq == 0 and iteration > 0
        ) or iteration == trial_cfg.training.iterations - 1:
            eval_start_time = time.time()
            eval_metrics = evaluate_trial(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                tx_position=tx_position,
                nt=nt,
                nr=nr,
                cfg=trial_cfg,
            )
            eval_time = time.time() - eval_start_time
            current_val_snr = eval_metrics["val_snr_db_mag"]
            val_log_msg = (
                f"T{trial_num} Validation @ {iteration} | Loss={eval_metrics['val_mse_loss_mag']:.4e} | "
                f"SNR={current_val_snr:.2f} dB | Time={eval_time:.2f}s"
            )
            console.print(f"[yellow]{val_log_msg}[/yellow]")
            logger.info(val_log_msg)

            if not np.isnan(current_val_snr) and not np.isinf(current_val_snr):
                best_trial_val_snr = max(best_trial_val_snr, current_val_snr)
                console.print(
                    f"  -> [Trial {trial_num}] Best SNR so far: [bold green]{best_trial_val_snr:.2f} dB[/bold green]"
                )
                logger.info(
                    f"  -> [Trial {trial_num}] Best SNR so far: {best_trial_val_snr:.2f} dB"
                )

                trial.report(best_trial_val_snr, iteration)

                if trial.should_prune():
                    logger.warning(
                        f"Trial {trial_num} pruned at iteration {iteration}."
                    )
                    console.print(
                        f"[bold red]Trial {trial_num} pruned at iteration {iteration}.[/bold red]"
                    )

                    del model, train_loader, val_loader, metadata, criterion
                    if device == torch.device("cuda"):
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned()

        if (
            iteration % trial_cfg.experiment.checkpoint_freq == 0 and iteration > 0
        ) or iteration == trial_cfg.training.iterations - 1:
            checkpoint_path = checkpoints_dir / f"checkpoint_{iteration:07d}.pt"

            model.save(checkpoint_path, iteration=iteration, cfg=trial_cfg)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    console.print(
        f"Trial {trial_num} finished. Final validation SNR: [bold green]{best_trial_val_snr:.4f} dB[/bold green]"
    )
    logger.info(f"Trial {trial_num} finished.")
    logger.info(f"Final validation SNR achieved: {best_trial_val_snr:.4f} dB")

    del model, train_loader, val_loader, metadata, criterion
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()

    return best_trial_val_snr


def log_best_trial(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """Callback to log the best trial results to a summary file."""

    if "log_dir" not in study.user_attrs:
        print("Warning: log_dir not found in study user_attrs. Cannot log summary.")
        return

    summary_log_path = Path(study.user_attrs["log_dir"]) / "tuning_summary.log"
    best_trial = study.best_trial

    with open(summary_log_path, "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{now}\n")
        f.write(f"Trial {trial.number} finished with value: {trial.value:.4f}\n")
        if best_trial:
            f.write(f"Current Best Trial: {best_trial.number}\n")
            f.write(f"  Best Value (SNR dB): {best_trial.value:.4f}\n")
            f.write("  Best Parameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"    {key}: {value}\n")
        f.write("-" * 20 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for GRF")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the base configuration file (non-tuned parameters)",
    )
    parser.add_argument(
        "--num_trials", type=int, default=50, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=f"grf_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Root directory for tuning logs"
    )

    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.config)

    default_cfg = OmegaConf.load("configs/default.yaml")
    base_cfg = OmegaConf.merge(default_cfg, base_cfg)

    base_cfg.experiment.log_dir = args.log_dir

    tuning_root_dir = Path(base_cfg.experiment.log_dir) / args.study_name
    tuning_root_dir.mkdir(parents=True, exist_ok=True)
    summary_log_path = tuning_root_dir / "tuning_summary.log"
    db_path = f"sqlite:///{tuning_root_dir}/optuna_study.db"

    console = Console()
    console.print(
        f"[bold blue]Starting Optuna Tuning Study: {args.study_name}[/bold blue]"
    )
    console.print(f"Database: {db_path}")
    console.print(f"Number of trials: {args.num_trials}")
    console.print(f"Tuning Log Directory: {tuning_root_dir}")
    console.print(f"Base Config File: {args.config}")
    console.print(f"Data Path: {args.data_path}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=db_path,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.set_user_attr("log_dir", str(tuning_root_dir))

    console.print(f"Existing trials in study: {len(study.trials)}")
    if len(study.trials) >= args.num_trials:
        console.print(
            "[yellow]Study already has the target number of trials or more. Exiting.[/yellow]"
        )
    else:
        remaining_trials = args.num_trials - len(study.trials)
        console.print(f"Running {remaining_trials} new trials...")

        objective_wrapper = lambda trial: objective(
            trial, base_cfg, args.data_path, console
        )

        try:
            study.optimize(
                objective_wrapper,
                n_trials=remaining_trials,
                callbacks=[log_best_trial],
                catch=(Exception,),
            )
        except Exception as e:
            console.print(
                f"[bold red]An critical error occurred during optimization: {e}[/bold red]"
            )
            logging.exception("Critical error during study optimization")

    console.rule("[bold blue]Tuning finished[/bold blue]")
    try:
        if study.best_trial:
            console.print(f"Best Trial Number: {study.best_trial.number}")
            console.print(f"Best value (SNR dB): {study.best_trial.value:.4f}")
            best_params_table = Table(title="Best Hyperparameters")
            best_params_table.add_column("Parameter", style="cyan")
            best_params_table.add_column("Value", style="green")
            for key, value in study.best_params.items():
                best_params_table.add_row(key, str(value))
            console.print(best_params_table)

            best_params_path = tuning_root_dir / "best_params.yaml"
            best_params_dict = study.best_params

            with open(best_params_path, "w") as f:
                import yaml

                yaml.dump(best_params_dict, f, default_flow_style=False)
            console.print(f"Best parameters saved to: {best_params_path}")
        else:
            console.print("[yellow]No completed trials found in the study.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error displaying final results: {e}[/bold red]")
        logging.exception("Error displaying final tuning results")
