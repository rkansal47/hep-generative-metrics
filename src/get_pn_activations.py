from jetnet.datasets import JetNet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pnet.particlenet import ParticleNet

jets = JetNet(
    "g",
    data_dir="/graphganvol/MPGAN/datasets/",
    split_fraction=[1, 0, 0],
    jet_features=None,
    particle_normalisation=JetNet.fpnd_norm,
    particle_features=["etarel", "phirel", "ptrel"],
)

jets_loaded = DataLoader(jets, shuffle=False, batch_size=256, pin_memory=True)

pnet = ParticleNet(30, 3).to("cuda")
pnet.load_state_dict(torch.load(f"pnet/pnet_state_dict.pt", map_location="cuda"))
pnet.eval()

activations = []
for i, (jets_batch, _) in tqdm(
    enumerate(jets_loaded), total=len(jets_loaded), desc="Running ParticleNet"
):
    activations.append(pnet(jets_batch.to("cuda"), ret_activations=True).cpu().detach().numpy())

activations = np.concatenate(activations, axis=0)
np.save("/graphganvol/hep-generative-metrics/pnet_activations.npy", activations)
