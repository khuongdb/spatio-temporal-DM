# Copyright (c) 2025 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
#
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
#
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.progress_bar import MyProgressBar
from src.data.datamodules import StarmenDataModule
import os
import resource
import torch


class MyCLI(LightningCLI):

    def before_instantiate_classes(self):
        """
        Handle the problem with selecting devices if running on cpu (cannot take the list [0] as running on gpu)
        """
        # check if the config.yaml in format stage: Namespace
        stage = self.config.subcommand
        stage_conf = self.config.get(stage, None)
        if stage_conf is None: 
            # trainer = self.config["trainer"]
            stage_conf = self.config

        # Fix for CPU: devices must be an int > 0
        if not torch.cuda.is_available():
            stage_conf['trainer']['devices'] = 1

        # access fully resolved config after CLI overrides
        # only apply for test stage
        if stage == "test":
            ddim_style = stage_conf["model"]["init_args"]["test_ddim_style"]
            split = stage_conf["data"]["test_ds"]["split"]
            noise_level = stage_conf["model"]["init_args"]["test_noise_level"]
            stage_conf["trainer"]["logger"]["init_args"]["name"] = f"{split}_{ddim_style}_noise{noise_level}"


    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MyProgressBar, nested_key='progress_bar')
        parser.add_lightning_class_args(EarlyStopping, nested_key='early_stopping')
        parser.add_lightning_class_args(ModelCheckpoint, nested_key='model_checkpoint')



def cli_main():
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
    # torch.set_float32_matmul_precision("high")
    cli = MyCLI(
        datamodule_class=StarmenDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
    ) 
    return cli


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())

    cli_main()


