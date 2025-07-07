import argparse, os
from types import SimpleNamespace


# defaults
default_config = SimpleNamespace(
    sample_rate = 30, # frequence
    sample_size = 4096, # number of sample
    batch_size = 8, #128, # batch_size   
    model="Dance_diffusion", # model to train
    nb_epochs=2, #50, # number of epochs to train on
    lr=2e-5, # learning rate for the training
    seed=42, # random seed
    wandb=False, # use wandb to log
    save=True, # save the model or not 
    saving_path="model/saved_models/", # saving path for the model
    gpus=3, # number of gpus to use
    #mixed_precision="no",  # use automatic mixed precision -> fasten training
    run_name="training_run", # run name for wandb
    one_batch_training=False, # train on one epoch for wandb 
    load_from_checkpoint= "trained_models/my_trained_model_20_04.pth",    #False, # path to a model checkpoint
    save_weights_only=False, # save weights or whole model
    conditional = False, # conditional or unconditional training
    adam_beta1 = 0.95,
    adam_beta2 = 0.999,
    gradient_accumulation_steps = 1,
    clip = False,
    prediction_type = "sample",
    shuffle = True,
    diffusion_mode = "classic_mode",
    dataset = "AugmentedDataset",
    predict_pga  = False,
)

STEAD_config = SimpleNamespace(
    sample_rate = 100, # frequence
    sample_size = 6000, # number of sample
    batch_size = 16, #128, # batch_size   
    model="PrimaProvaSTEAD", # model to train
    nb_epochs=4, #50, # number of epochs to train on
    lr=8e-5, # learning rate for the training
    lambda_corr=2, # hyperparameter weight for Loss_f
    seed=42, # random seed 
    wandb=True, # use wandb to log
    save=True, # save the model or not 
    saving_path="model/saved_models/", # saving path for the model 
    gpus=3, # number of gpus to use
    #mixed_precision="no",  # use automatic mixed precision -> fasten training
    run_name="GPU_run", # run name for wandb
    one_batch_training=False, # train on one epoch for wandb 
    load_from_checkpoint= "trained_models/8e_cond.pth", # path to a model checkpoint
    save_weHights_only=False, # save weights or whole model        
    conditional = True, # conditional or unconditional training
    adam_beta1 = 0.95,
    adam_beta2 = 0.999,
    gradient_accumulation_steps = 1,
    clip = False,
    prediction_type = "sample",
    shuffle = True,
    diffusion_mode = "classic_mode",
    dataset = "AugmentedDatasetSTEAD",
    predict_pga  = False,
)


STEAD_config_db = SimpleNamespace(
    sample_rate = 100, # frequence
    sample_size = 6000, # number of sample
    batch_size = 2, #128, # batch_size   
    model="PrimaProvaSTEAD", # model to train
    nb_epochs=1, #50, # number of epochs to train on
    lr=1e-3, # learning rate for the training
    lr_warmup_steps = 1500, # number of warmup steps (not accounting for grad accumulation)
    schedule = "constant", # type of schedule (https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
    lambda_corr=10, # hyperparameter weight for Loss_f
    seed=42, # random seed 
    wandb=False, # use wandb to log
    save=True, # save the model or not 
    saving_path="model/saved_models/", # saving path for the model 
    gpus=2, # number of gpus to use
    #mixed_precision="no",  # use automatic mixed precision -> fasten training
    run_name="training_run", # run name for wandb
    one_batch_training=False, # train on one epoch for wandb 
    load_from_checkpoint= False, # path to a model checkpoint
    save_weights_only=False, # save weights or whole model
    conditional = False, # conditional or unconditional training
    adam_beta1 = 0.95,
    adam_beta2 = 0.999,
    gradient_accumulation_steps = 4,
    clip = False,
    prediction_type = "sample",
    shuffle = True,
    diffusion_mode = "classic_mode",
    dataset = "AugmentedDatasetSTEAD",
    predict_pga  = False,
)



def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description="Process hyper-parameters")
    argparser.add_argument(
        "--sample_rate", type=int, default=default_config.sample_rate, help="traing batch_size"
    )
    argparser.add_argument(
        "--sample_size", type=int, default=default_config.sample_size, help="traing batch_size"
    )

    argparser.add_argument(
        "--batch_size", type=int, default=default_config.batch_size, help="batch size"
    )
    argparser.add_argument(
        "--nb_epochs",
        type=int,
        default=default_config.nb_epochs,
        help="number of training epochs",
    )
    argparser.add_argument(
        "--lr",
        type=float,
        default=default_config.lr,
        help="learning rate",
    )
    argparser.add_argument(
        "--lambda_corr",
        type=float,
        default=STEAD_config.lambda_corr,
        help="hyperparameter for loss_f",
    )
    argparser.add_argument(
        "--model",
        type=str,
        default=default_config.model,
        help="model to train",
    )
    argparser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=default_config.wandb,
        help="Use wandb",
    )
    argparser.add_argument(
        "--seed", type=int, default=default_config.seed, help="random seed"
    )
    argparser.add_argument(
        "--gpus",
        type=int,
        default=default_config.gpus,
        help="number of gpus to use",
    )
    """
    argparser.add_argument(
        "--mixed_precision",
        type=str,
        default=default_config.mixed_precision,
        help="use fp16",
    )
    """
    argparser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=default_config.save,
        help="Save model",
    )
    argparser.add_argument(
        "--run_name",
        default=default_config.run_name,
        help="Name of the run",
    )
    argparser.add_argument(
        "--saving_path",
        default=default_config.saving_path,
        help="Path to save model",
    )
    argparser.add_argument(
        "--one_batch_training",
        action=argparse.BooleanOptionalAction,
        default=default_config.one_batch_training,
        help="Train on one batch",
    )
    argparser.add_argument(
        "--load_from_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=default_config.load_from_checkpoint,
        help="Retrain a model from a cpkt file",
    )
    argparser.add_argument(
        "--save_weights_only",
        action=argparse.BooleanOptionalAction,
        default=default_config.save_weights_only,
        help="Save weights",
    )
    argparser.add_argument(
        "--gradient_accumulation_steps",
        default= default_config.gradient_accumulation_steps,
        help="Number of accumulation steps before gradient updates"

    )
    argparser.add_argument(
        "--clip",
        action=argparse.BooleanOptionalAction,
        default = default_config.clip,
        help="Clip gradient to 5"
    )
    argparser.add_argument(
        "--prediction_type",
        default = default_config.prediction_type,
        help = "Prediction type between epsilon and sample"
    )
    argparser.add_argument(
        "--conditional",
        action = argparse.BooleanOptionalAction,
        default = default_config.conditional,
        help = "Conditional or Unconditional training" 
    )
    argparser.add_argument(
        "--diffusion_mode",
        default = default_config.diffusion_mode,
        help = "Choice between the different diffusion implementation" 
    )
    argparser.add_argument(
        "--dataset",
        default = default_config.dataset,
        help = "Choice between the different dataset implementation" 
    )
    argparser.add_argument(
        "--predict_pga",
        action = argparse.BooleanOptionalAction,
        default = default_config.predict_pga,
        help = "Predict pga or not" 
    )
    args = argparser.parse_args()
    vars(default_config).update(vars(args))

