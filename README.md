# Spatio-Temporal Diffusion Model 

This is source code for my intern project, titled "Longitudinal Diffusion Models for Unsupervised Anomaly Detection (UAD)". The internship was carried out with the Statity team at INRIA Grenoble.

As the project is still ongoing, the codebase is not yet fully cleaned, but it will be updated progressively.

## Requirements

The project was developed using Python `3.10` and PyTorch `2.7.1`.  
Dependencies can be installed in two ways:  

### Using `pip`  

```bash
pip install -r requirements.txt
```

### Using `uv`

I use `uv` to manage my project environment and dependencies. 

```bash
# Install uv if not already installed
pip install uv

# Sync dependencies from pyproject.toml
uv sync
```

To activate the environment: 

```bash
source .venv/bin/activate  # Linux/macOS
```

## Data preparation

This project is currently tested with the synthetic dataset Starmen \href{https://zenodo.org/records/5081988}{available here}. Please download the dataset and place it in the `data/` folder. Here, we will use the no acceleration samples, which is stored at `data/starmen/output_random_noacc`. 

The original dataset does not contain anomaly samples, so we add anomaly using `cv2` package. We provide script to generate 3 types of anomaly: `growing_circle`, `darker_circle` and `darker_line`. The first one demonstrates anomaly that grows in size over time, while the later two mimic anomalies that grow in intensity. To generate anomaly dataset, run the script:

```bash
python3 src/data/generate_anomaly.py -n 20 \
                                    -a growing_cirlce \
                                    -set test
                                    --name anomaly_data
```

The arguments are: 

- `n`: number of samples to generate. 

- `a`: anomaly type. Choose from `growing_circle`, `darker_circle` or `darker_line`.

- `set`: choose which set ('train', 'test', 'val') to consider when generating anomalous images.

- `--name`: name of dataset. Default to `<type of anomaly>_<number of sample>`. For example: `growing_circle20`.

This will generate anomaly samples in `data/output_random_noacc/anomaly_images` and place a `.csv` file in `data/output_random_noacc/`. Our data class will load information from `csv` file of the form: `starmen_<split>.csv`, e.g. `starmen_train.csv`, `starmen_darker_circle20.csv`. 

## Spatial Diffusion Model (SDM)

Our main source code is placed inside `src/ldae` for spatial diffusion model and anomaly detection module.

Our main SDM model is condition diffusion model (`condDDPM.py`), with configuration file `configs/starmen_diffae.yaml`. 

- To train the model, run: 

```bash
python3 main_diffae.py --config configs/starmen_diffae.yaml fit 
```

- To run inference: 

```bash
python3 main_diffae.py --config configs/starmen_diffae.yaml test --data.test_ds.split test
```

We can run inference for different dataset by subtitute the split, e.g. `growing_circle20`, `darker_line20`, etc...

**Representation learning with Diffusion Autoencoder**: as part of our experiment, we also provide script to train Latent Diffusion Autoencoder (following \href{https://github.com/GabrieleLozupone/LDAE}{LDAE}). 

- To train LDAE: 

```bash
python3 main_diffae.py --config configs/starmen_ldae.yaml fit
```

- To inference LDAE: 

```bash
python3 main_diffae.py --config configs/starmen_ldae.yaml test
```

## Temporal Diffusion Model (TDM)

Our main source code for TDM is placed insdie `src/tadm`. 

- To train TDM: 

```bash
python3 main_diffae.py --config configs/starmen_tadm.yaml fit
```

- To inference TDM: 

```bash
python3 main_diffae.py --config configs/starmen_tadm.yaml test \
                        --data.test_ds.split test
```

TDM infernce process includes: impute missing image from preceeding image, impute missing image from previous image with random interval time step, and oversampling. 