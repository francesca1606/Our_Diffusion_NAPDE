from training.trainer import TrainerDualBranch
# from model.model import Model
from processing.load_data import DataModule
# from processing.augmented_dataset import AugmentedDataModule
from processing.new_dataset_loader import AugmentedDataModule
from extra.parsing_db import parse_args, STEAD_config_db
from accelerate.logging import get_logger
import torch
import logging
from accelerate import Accelerator
import wandb
import os
import random
import numpy as np
import sys
import logging

# os.environ["WANDB__SERVICE_WAIT"]="1000"

# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s — %(levelname)s — %(message)s'))
logger.addHandler(handler)

# logger.setLevel(logging.WARNING)


def project_name(config):
    if config.predict_pga:
        return "predict_pga"
    if config.conditional:
        return "conditional_sismic_diffusion_good_pga"
    else:
        return "uncondtional_sismic_diffusion"


def train(config):
    accelerator = Accelerator(
        log_with="wandb" if config.wandb else None,
        # project_config=project_config,
        project_dir="./logs",
        # mixed_precision=config.mixed_precision if torch.cuda.is_available() else "no",
    )
    if accelerator.is_main_process:
        run = project_name(config=config)
        accelerator.init_trackers(run, config=config)

    gpus = torch.cuda.device_count()
    total_batch_size = config.batch_size * gpus * config.gradient_accumulation_steps
    total_batch_size = total_batch_size if total_batch_size != 0 else config.batch_size * \
        1 * config.gradient_accumulation_steps

    """
    dataset = DataModule(
                conditional = config.conditional,
                batch_size =  config.batch_size,
                shuffle= config.shuffle
    )
    """
    if config.dataset == "AugmentedDataset":
        dataset = AugmentedDataModule(
            path="data/nsy51200/temporary/",
            batch_size=config.batch_size,
            shuffle=config.shuffle
        )
    elif config.dataset == "6000_data":
        dataset = AugmentedDataModule(
            path="data/6000_data/",
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            predict_pga=config.predict_pga
        )
    elif config.dataset == "AugmentedDatasetSTEAD":
        dataset = AugmentedDataModule(
            path="STEAD_data/chunk2/",
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            predict_pga=config.predict_pga,
            accelerator=accelerator
        )
    elif config.dataset == "NormalDataset":
        dataset = DataModule(
            conditional=config.conditional,
            batch_size=config.batch_size,
            shuffle=config.shuffle
        )
    else:
        raise ValueError("Dataset not found")

    dataset.setup()
    if accelerator.is_local_main_process:
        logger.info(f" Num Epochs : {config.nb_epochs} ")
        logger.info(f" Num of GPUs available : {gpus} ")
        logger.info(f" Original batch_size : {config.batch_size}")
        logger.info(
            f" Number of accumulation steps : {config.gradient_accumulation_steps}")
        logger.info(
            f" Total batch (w. parallel, distributed & accumulation) : {total_batch_size} ")
        logger.info(f" Model : {config.model}")
        logger.info(f" Data set Loaded / starting training")
        logger.info(f" Diffusion mode : {config.diffusion_mode}")
        logger.info(f" Conditional : {config.conditional}")
        logger.info(f" Dataset used : {config.dataset}")
        logger.info(f" Learning rate : {config.lr}")
        logger.info(f" Gradient clipping : {config.clip}")

    if config.one_batch_training:
        if config.conditional:
            dataset.train_loader = dataset.one_batch
            dataset.test_loader = dataset.one_batch
        else:
            dataset.combined_loader = dataset.one_batch

    trainer = TrainerDualBranch(
        dataloader=dataset.combined_loader if not config.conditional else None,
        train_loader=dataset.train_loader if (
            config.conditional or config.predict_pga) else None,
        test_loader=dataset.test_loader if (
            config.conditional or config.predict_pga) else None,
        lr=config.lr,
        lambda_corr=config.lambda_corr,
        conditional=config.conditional,
        nb_epochs=config.nb_epochs,
        saving=config.save,
        saving_path=config.saving_path,
        clipping_gradient=config.clip,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        accelerator=accelerator,
        diffusion_mode=config.diffusion_mode,
        prediction_type=config.prediction_type,
        checkpoint_path=config.load_from_checkpoint,
    )
    trainer.fit()
    trainer.save_model_MIO()  # dopo il training


def main():
    print("Cuda support:", torch.cuda.is_available(),
          ":", torch.cuda.device_count(), "devices")
    torch.cuda.empty_cache()
    parse_args()
    # default_config.dataset = "6000_data"
    train(STEAD_config)


if __name__ == "__main__":
    print("Cuda support:", torch.cuda.is_available(),
          ":", torch.cuda.device_count(), "devices")
    torch.cuda.empty_cache()
    parse_args()
    # default_config.dataset = "6000_data"
    train(STEAD_config_db)
