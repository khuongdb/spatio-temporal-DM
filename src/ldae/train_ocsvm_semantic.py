import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np 
from src.ldae import CondDDPM
from src.ldae.modules import OCSVM, OCSVMguidedAutoencoder
from src.data import StarmenDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import rearrange
import random
import argparse

from torch.utils.tensorboard import SummaryWriter


def train_ocsvm(
        device,
        args):
        
    # Load dataset
    batch_size = int(args.train_size / 10)
    train_ds = StarmenDataset(
        data_dir=args.data_path,
        split="train",
        nb_subject=args.nb_sample,
        save_data=False
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    val_ds = StarmenDataset(
        data_dir=args.data_path,
        split="val",
        nb_subject=args.nb_sample,
        save_data=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    ## Load SemanticEncoder model
    use_semantic_enc = args.use_semantic
    work_dir = args.workdir
    if use_semantic_enc:
        ckpt_path = os.path.join(work_dir, "representation-learning", "checkpoints", "best.ckpt")

        diffae_litmodel = CondDDPM.load_from_checkpoint(ckpt_path, map_location=device)
        ema_encoder = diffae_litmodel.ema_encoder
        ema_encoder.eval()
        for param in ema_encoder.parameters():
            param.requires_grad = False

    # make workdir folder
    ckpt_dir = f"{work_dir}/ocsvm/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # OCSVM model
    if use_semantic_enc: 
        model = OCSVM(input_dim=512,
                    latent_dim=args.n,
                    l=args.l,
                    ocsvm_coeff=0.1,
                    nu_ocsvm_coeff=0.03,
                    gamma_rbf_coeff="scale",
                    batch_size_train=args.train_size, 
                    batch_size_valid=args.valid_size,
                    jz_mode="StopGradLoss")
    else: 
        model = OCSVMguidedAutoencoder(
            batch_size_train=args.train_size, 
            batch_size_valid=args.valid_size,
            latent_dim=args.n, 
            ocsvm_coeff=0.1,
            nu_ocsvm_coeff=0.03, 
            gamma_rbf_coeff="scale", 
            jz_mode="StopGradLoss",
            jean_zad_linear=False)

    model.to(device)

    print("Model:")
    print(model)
    print(
        "#model_params:", np.sum([len(p.flatten()) for p in model.parameters()])
    )

    # setup log
    import time
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(work_dir, f"ocsvm/ts_logs_{run_id}")
    writer = SummaryWriter(log_dir=log_dir)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_val_epoch = -1

    # Main training loop 
    nb_epochs = args.epochs
    ysems = []
    for epoch in range(nb_epochs):
        epoch_loss = 0
        epoch_loss_ocsvm = 0

        model.train()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        pbar.set_description(f"Epoch {epoch}")
        for step, batch in pbar:
            x = batch["x_origin"]
            x = rearrange(x, "b t ... -> (b t) ...").to(device)
            optimizer.zero_grad(set_to_none=True)

            # 1. Get the semantic encoded y_sem
            if use_semantic_enc:
                ysem = ema_encoder.forward(x)
            else: 
                ysem = x

            # 1.a. concat all semantic encoded ysem of training dataset to use in validation set
            if epoch == 0: 
                ysems.append(ysem)

            # 2. Train OCSVM model
            loss, mse, ocsvm_obj = model.compute_loss(ysem, training=True)

            # Optimize
            loss.backward()          
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_ocsvm += ocsvm_obj.item()
            pbar.set_postfix({"loss": epoch_loss / (step + 1)})

        avg_epoch_loss = epoch_loss / (step + 1)
        avg_epoch_loss_ocsvm = epoch_loss_ocsvm / (step + 1)
        writer.add_scalar("train_loss", avg_epoch_loss, epoch)
        writer.add_scalar("train_loss_ocsvm", avg_epoch_loss_ocsvm, epoch)

        # Validation step
        if epoch == 0: 
            if len(ysems) > 0:
                ysems = torch.cat(ysems)

        if epoch % 10 == 0: 
            model.eval()
            val_epoch_loss = 0
            val_ocsvm_loss = 0
            val_mis_clf = 0
            for step, batch in enumerate(val_loader):
                x_val = batch["x_origin"]
                x_val = rearrange(x_val, "b t ... -> (b t) ...").to(device)
                with torch.no_grad():
                    if use_semantic_enc:
                        ysem_val = ema_encoder.forward(x_val)
                    else: 
                        ysem_val = x_val
                    val_loss, mse, ocsvm_obj = model.compute_loss(ysem_val, training=True)

                    val_epoch_loss += val_loss.item()
                    val_ocsvm_loss += ocsvm_obj.item()

                    # classify using sklearn
                    pred, mis_clf = model.sklearn_clf_ocsvm(ysems, ysem_val)
                    val_mis_clf += mis_clf
        
        avg_val_loss = val_epoch_loss / (step + 1)
        avg_val_ocsvm_loss = val_ocsvm_loss / (step + 1)
        avg_val_miss_clf = val_mis_clf / len(val_loader) * x_val.shape[0]

        writer.add_scalar("val_loss", avg_val_loss, epoch)
        writer.add_scalar("val_loss_ocsvm", avg_val_ocsvm_loss, epoch)
        writer.add_scalar("val_miss_clf", avg_val_miss_clf, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_epoch = epoch

            # Save the best model
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "info": {
                        "epoch": best_val_epoch,
                        "best_val_loss": best_val_loss,
                        "best_val_ocsvm_loss": avg_val_ocsvm_loss
                    },
                },
                os.path.join(ckpt_dir, "best.pth"),
            )
        
        # Save latest.pth model after each epoch.
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "info": {
                    "epoch": epoch,
                },
            },
            os.path.join(ckpt_dir, "latest.pth"),
        )

    writer.close()
    print("Finish training.")

@torch.no_grad()
def final_decision_function(
    args,
    device=torch.device("cpu"),
):
    """
    Run through the whole train dataset to find the final decision function. 
    """

    work_dir = args.workdir

    # Load dataset
    batch_size = int(args.train_size / 10)
    train_ds = StarmenDataset(
        data_dir=args.data_path,
        split="train",
        nb_subject=args.nb_sample,
        save_data=False
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    ## Load diffusion model
    if args.use_semantic: 
        ckpt_path = os.path.join(work_dir, "representation-learning", "checkpoints", "best.ckpt")

        diffae_litmodel = CondDDPM.load_from_checkpoint(ckpt_path, map_location=device)
        ema_encoder = diffae_litmodel.ema_encoder
        ema_encoder.eval()
        for param in ema_encoder.parameters():
            param.requires_grad = False
        print(f"Load trained diffusion model from {ckpt_path}. Model set to eval mode.")


    # OCSVM model
    if args.use_semantic: 
        model = OCSVM(input_dim=512,
                    latent_dim=args.n,
                    l=args.l,
                    ocsvm_coeff=0.1,
                    nu_ocsvm_coeff=0.03,
                    gamma_rbf_coeff="scale",
                    batch_size_train=args.train_size, 
                    batch_size_valid=args.valid_size,
                    jz_mode="StopGradLoss")
    else: 
        model = OCSVMguidedAutoencoder(
            batch_size_train=args.train_size, 
            batch_size_valid=args.valid_size,
            latent_dim=args.n, 
            ocsvm_coeff=0.1,
            nu_ocsvm_coeff=0.03, 
            gamma_rbf_coeff="scale", 
            jz_mode="StopGradLoss",
            jean_zad_linear=False)
        
    ckpt_path = f"{work_dir}/ocsvm/checkpoints/best.pth"
    if not os.path.exists(ckpt_path):
        raise ValueError(f"checkpoint does not exist at {ckpt_path}")
    
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model_state"])

    model.eval()
    model.to(device)
    print(f"Load trained OCSVM model from {ckpt_path}. Model set to eval mode. ")

    # compute semantic ysem 
    zsems = [] 
    for i, batch in enumerate(train_loader):
        x = batch["x_origin"]
        x = rearrange(x, "b t ... -> (b t) ...").to(device)

        if args.use_semantic: 
            ysem = ema_encoder.forward(x)
        else:
            ysem = x

        _, zsem = model.forward(ysem)
        zsems.append(zsem)
    zsems = torch.cat(zsems)

    # using sklearn OCSVM
    from sklearn import svm

    clf = svm.OneClassSVM(kernel="rbf", gamma=model.gamma_mode, nu=model.nu)
    clf.fit(zsems.detach().cpu().numpy())

    # Save model
    import pickle
    clf_fit_path = os.path.join(work_dir, "ocsvm", "ocsvm_fit.pkl")
    with open(clf_fit_path, "wb") as f:
        pickle.dump(clf, f)
        print(f"Fitted OneClassSVM is saved at {clf_fit_path}")


parser = argparse.ArgumentParser(description="Training Autoencoders")
parser.add_argument('-n', type=int, help='latent dimension', default=16)
parser.add_argument('-l', type=int, help='implicit layer', default=0)
parser.add_argument('--train-size', type=int, default=100)
parser.add_argument('--valid-size', type=int, default=100)
parser.add_argument('--epochs', type=int, help='#epochs', default=100)
parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--dataset', type=str, default="mnist")
# parser.add_argument('--model-name', type=str)
parser.add_argument('--workdir', type=str, default="./workdir/diffae_starmen/")
parser.add_argument('--data-path', type=str, default="./data/starmen/output_random_noacc/")
parser.add_argument('--use-semantic', action="store_true", help="Whether to use semantic encoder from DiffModel. If not, input to \
                    the model will be the whole image")
parser.add_argument('--semantic-checkpoint', type=str, default="workdir/diffae_starmen/representation-learning/checkpoints", help="path to trained semantic encoder")
parser.add_argument('--debug', action="store_true", help="debug mode")



if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.debug: 
        args.train_size = 10
        args.valid_size = 10
        args.epochs = 5
        args.workdir = "workdir/debug"
        args.nb_sample = 10
    else:
        args.nb_sample = None

    train_ocsvm(device=device, args=args)
    final_decision_function(args, device=device)