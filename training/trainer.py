import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import logging
import os
from common.common_nn import patch
from diffusion.diffusion_model import DiffusionAttnUnet1DCond, DiffusionAttnUnet1D
from diffusion.diff import DiffusionUncond
from datetime import datetime
import sys
import logging
from diffusion.utils import butterworth_decompose_batch

# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s — %(levelname)s — %(message)s'))
logger.addHandler(handler)


class Trainer:
    def __init__(self,
                 dataloader=None,
                 train_loader=None,
                 test_loader=None,
                 accelerator=Accelerator(),
                 saving_path: str = "model/saved_models/",
                 saving: bool = True,
                 lr: float = 3e-4,
                 lambda_corr: float = 10.,
                 nb_epochs: int = 100,
                 conditional: bool = True,
                 betas=(0.95, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-6,
                 lr_warmup_steps: int = 0,
                 schedule: str = "constant",
                 gradient_accumulation_steps: int = 1,
                 clipping_gradient: bool = True,
                 one_batch: bool = True,
                 test_step: int = 10,
                 diffusion_mode: str = "audio_torch",
                 checkpoint_path=None,
                 *args, **kwargs
                 ) -> None:

        self.diffusion_mode = diffusion_mode
        self.conditional = conditional

        if self.conditional:
            self.model = DiffusionUncond(
                DiffusionAttnUnet1DCond(
                    io_channels=6,
                    n_attn_layers=4,
                    depth=4,
                    c_mults=[128, 128, 256, 256] + [512]
                )
            )
        else:
            self.model = DiffusionUncond(
                DiffusionAttnUnet1DCond(
                    io_channels=3,
                    n_attn_layers=4,
                    depth=4,
                    c_mults=[128, 128, 256, 256] + [512]
                )
            )

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        else:
            if checkpoint_path:
                logger.warning(
                    f"Checkpoint path '{checkpoint_path}' not found. Starting from scratch.")

        self.lambda_corr = lambda_corr
        self.nb_epochs = nb_epochs
        self.saving_path = saving_path
        self.dataloader = dataloader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.saving = saving
        self.global_step = 0
        self.epoch_loss = 0
        self.test_step = test_step

        self.accelerate = accelerator
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )

        num_training_steps = (len(
            self.train_loader) * self.nb_epochs) if conditional else (len(self.dataloader) * self.nb_epochs)
        self.lr_scheduler = get_scheduler(
            schedule,
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=num_training_steps
        )
        if not self.conditional:
            self.dataloader, self.model, self.optimizer, self.lr_scheduler = self.accelerate.prepare(
                self.dataloader, self.model, self.optimizer, self.lr_scheduler
            )
        else:
            self.train_loader, self.test_loader, self.model, self.optimizer, self.lr_scheduler = self.accelerate.prepare(
                self.train_loader, self.test_loader, self.model, self.optimizer, self.lr_scheduler
            )

        self.device = self.accelerate.device

        self.best_loss = float("inf")
        self.clipping_gradient = clipping_gradient
        self.one_batch = one_batch

    def train_one_epoch_unconditional(self, epoch):

        self.model.train()

        progress_bar = tqdm(total=len(self.dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for idx, batch in enumerate(self.dataloader):
            y, fc, *other = batch
            y, _ = patch(y, x=None)
            if y.shape[0] == 0:
                print(f"Empty batch at epoch {epoch}")
                continue

            y = y.to(self.device)
            fc = fc.to(self.device)

            if torch.cuda.device_count() < 2:
                loss = self.model.training_step(
                    y, fc, self.lambda_corr, device=self.device)
            else:
                loss = self.model.module.training_step(
                    y, fc, self.lambda_corr, device=self.device)

            self.accelerate.backward(loss)

            if self.clipping_gradient:
                self.accelerate.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            # torch.cuda.empty_cache()
            logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[
                0], "step": self.global_step}
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)

            if idx == len(self.dataloader) - 1:
                progress_bar.close()

            self.epoch_loss += loss.detach().item()
            self.accelerate.log(logs, step=self.global_step)
            self.global_step += 1
        torch.cuda.empty_cache()

        self.epoch_loss = self.epoch_loss / len(self.dataloader)
        self.val_loss = self.epoch_loss
        logger.info(f"Epoch loss : {self.epoch_loss}")
        epoch_logs = {"epoch_loss": self.epoch_loss, "epoch": epoch}
        self.accelerate.log(epoch_logs, step=self.global_step)

    def train_one_epoch_conditional(self, epoch):
        # Train Loop
        self.model.train()
        if self.accelerate.is_local_main_process:
            progress_bar = tqdm(total=len(self.train_loader))
            progress_bar.set_description(f"Epoch {epoch}")

        for idx, batch in enumerate(self.train_loader):
            y, fc, *other = batch
            y, _ = patch(y, x=None)
            if y.shape[0] == 0:
                print(f"Empty batch at epoch {epoch}")
                continue

            y = y.to(self.device)  # not needed with accelerate
            cond_low = butterworth_decompose_batch(y)

            if torch.cuda.device_count() < 2:
                loss = self.model.training_step(
                    y, fc, self.lambda_corr, cond=cond_low, device=self.device)
            else:
                loss = self.model.module.training_step(
                    y, fc, self.lambda_corr, device=self.device)

            self.accelerate.backward(loss)

            if self.clipping_gradient:
                self.accelerate.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            # Add this line for more clarity
            self.accelerate.wait_for_everyone()
            if self.accelerate.is_local_main_process:
                batch_logs = {"step_loss": loss.detach().item(
                ), "lr": self.lr_scheduler.get_last_lr()[0]}
                self.accelerate.log(batch_logs, step=self.global_step)
                progress_bar.set_postfix(**batch_logs)
                progress_bar.update(1)
            # end
            if idx == len(self.train_loader) - 1:
                if self.accelerate.is_local_main_process:
                    progress_bar.close()
            self.epoch_loss += loss.detach().item()
            self.global_step += 1

        self.epoch_loss = self.epoch_loss / len(self.train_loader)
        epoch_logs = {"epoch_loss": self.epoch_loss, "epoch": epoch}
        if self.accelerate.is_local_main_process:
            logger.info(f"Epoch loss : {self.epoch_loss}")
        self.accelerate.log(epoch_logs, step=self.global_step)

        # Test Loop
        self.accelerate.wait_for_everyone()
        self.model.eval()
        with torch.no_grad():
            metrics_dict = {
                "test_MSE": 0,
                "test_SNR": 0,
                "train_loss": 0,
                # "train_SNR" : 0,
            }

            for idx, batch in enumerate(self.test_loader):
                y, fc, *other = batch
                y, _ = patch(y, x=None)
                if y.shape[0] == 0:
                    print(f"Empty batch at epoch {epoch}")
                    continue

                y = y.to(self.device)
                cond_low = butterworth_decompose_batch(y)

                if torch.cuda.device_count() < 2:
                    batch_dict = self.model.test_step(
                        y, x=cond_low, num_steps=self.test_step, device=self.device)
                else:
                    batch_dict = self.model.module.test_step(
                        y, x=cond_low, num_steps=self.test_step, device=self.device)

                metrics_dict["test_MSE"] += batch_dict["MSE"]
                metrics_dict["test_SNR"] += batch_dict["SNR"]

            metrics_dict["test_MSE"] = metrics_dict["test_MSE"] / \
                len(self.test_loader)
            metrics_dict["test_SNR"] = metrics_dict["test_SNR"] / \
                len(self.test_loader)
            # metrics_dict["train_MSE"] = metrics_dict["train_MSE"] / len(self.train_loader)
            # metrics_dict["train_SNR"] = metrics_dict["train_SNR"] / len(self.train_loader)

        self.val_loss = metrics_dict["test_MSE"]
        logger.info(f"test MSE loss : {self.val_loss}")
        logger.info(f"train loss : {self.epoch_loss}")
        merged_dict = {**epoch_logs, **metrics_dict}
        self.accelerate.log(merged_dict)

    def save_model(self, epoch):
        if self.accelerate.is_local_main_process:
            # logger.info("Saving model")
            try:
                run = self.accelerate.get_tracker("wandb").run.name
                run_name = run.name if run and run.name is not None else "ep"
            except:
                run_name = ""
            model = self.accelerate.unwrap_model(self.model)
            folder_path = self.saving_path + "/" + run_name
            filename = folder_path + f"/training_1.pth"
            state = model.state_dict()
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(state, filename)

    def fit(self):
        for epoch in range(self.nb_epochs):
            if self.conditional:
                self.train_one_epoch_conditional(epoch)
            else:
                self.train_one_epoch_unconditional(epoch)
            if self.saving:
                print(f"Best loss : {self.best_loss}")
                print(f"Mean epoch loss : {self.epoch_loss}")
                if self.best_loss > self.val_loss:
                    self.best_loss = self.val_loss
                    self.save_model(epoch)
            self.epoch_loss = 0

        self.accelerate.end_training()
