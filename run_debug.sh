#!/bin/bash 


source .venv/bin/activate

# # Longitudinal_learning in conjuntion with leaspy
# python3 main_diffae.py --config configs/starmen_diffae_dim4.yaml fit \
#                         --model.init_args.mode longitudinal_learning \
#                         --trainer.max_epochs 10 \
#                         --trainer.logger.init_args.save_dir workdir/debug \
#                         --trainer.limit_train_batches 10 \
#                         --model_checkpoint.dirpath workdir/debug/longitudinal_learning/checkpoints

# TADM debug
python3 main_diffae.py --config configs/starmen_diffae.yaml test \
                        --trainer.logger.init_args.save_dir workdir/debug \
                        --trainer.limit_test_batches 2 \
                        --data.test_ds.split combine_debug \
                        --trainer.limit_train_batches 2 \
                        --model_checkpoint.dirpath workdir/debug/tadm/checkpoints \
                        --model.init_args.test_ddim_style ddim1 \
                        --trainer.max_epochs 12
                        # --ckpt_path workdir/ldae_starmen/representation-learning/checkpoints/last.ckpt \
