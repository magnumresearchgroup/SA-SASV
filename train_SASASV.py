import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from models.SA_SASV import SA_SASV
from datasets.SASASV_dataset import get_dataset
from speechbrain.utils.parameter_transfer import Pretrainer

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)
    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    pretrain = Pretrainer(loadables={'model': hparams['modules']['fbanks_encoder']},
                          paths={'model': "speechbrain/spkrec-ecapa-voxceleb/embedding_model.ckpt"})
    pretrain.collect_files()
    pretrain.load_collected()
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Create dataset objects "train", "valid", and "test".
    datasets = get_dataset(hparams)
    # Initialize the Brain object to prepare for mask training.
    sasasv_model = SA_SASV(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    sasasv_model.fit(
        epoch_counter=sasasv_model.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

