import os

import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
from leaspy.models import LogisticModel, LinearModel
from leaspy.datasets import load_dataset
from leaspy.io.data import Data, Dataset
from leaspy.algo import AlgorithmSettings

alzheimer_df = load_dataset("alzheimer")
alzheimer_df = alzheimer_df[["MMSE", "RAVLT", "FAQ", "FDG PET"]]
print(alzheimer_df.head())

data = Data.from_dataframe(alzheimer_df)
dataset = Dataset(data)


# %%
# model = LogisticModel(name="test-model", source_dimension=2)
model = LinearModel(name="linear", source_dimension=3)

algo_settings = AlgorithmSettings('mcmc_saem', 
                                  seed=42, 
                                  n_iter=100,           # n_iter defines the number of iterations
                                  progress_bar=True)     # To display a nice progression bar during calibration

model.fit(
    dataset,
    algorithm_settings=algo_settings,
)

# %%
print(model)


# %%
settings_personalization = AlgorithmSettings('scipy_minimize', progress_bar=True, use_jacobian=True)

test_df = alzheimer_df.loc[:"GS-010"]
test_data = Data.from_dataframe(test_df)
test_ds = Dataset(test_data)

ips = model.personalize(test_ds, algorithm_settings=settings_personalization)
