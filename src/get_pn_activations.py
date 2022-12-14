from jetnet.datasets import JetNet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pnet.particlenet import ParticleNet
import pickle

jets = JetNet(
    "g",
    data_dir="/graphganvol/MPGAN/datasets/",
    split_fraction=[1, 0, 0],
    jet_features=None,
    particle_normalisation=JetNet.fpnd_norm,
    particle_features=["etarel", "phirel", "ptrel"],
)

print(f"Real Maxes: {np.max(np.abs(jets.reshape(-1, jets.shape[-1])), axis=0)}")

# jets_loaded = DataLoader(jets, shuffle=False, batch_size=512, pin_memory=True)


with open("pf_dists.pkl", "rb") as f:
    pf_dists = pickle.load(f)

pnet_activations = {}

pnet = ParticleNet(30, 3).to("cuda")
pnet.load_state_dict(torch.load(f"pnet/pnet_state_dict.pt", map_location="cuda"))
pnet.eval()

for key, (jets, _) in pf_dists.items():
    print(key)

    jets = JetNet.fpnd_norm(jets.astype(np.float32))
    print(f"Maxes: {np.max(np.abs(jets.reshape(-1, jets.shape[-1])), axis=0)}")
    jets_loaded = DataLoader(jets, shuffle=False, batch_size=512, pin_memory=True)

    activations = []
    # for i, (jets_batch, _) in tqdm(
    for i, jets_batch in tqdm(
        enumerate(jets_loaded), total=len(jets_loaded), desc="Running ParticleNet"
    ):
        activations.append(pnet(jets_batch.to("cuda"), ret_activations=True).cpu().detach().numpy())

    activations = np.concatenate(activations, axis=0)
    np.save(f"/graphganvol/hep-generative-metrics/pnet_activations_{key}.npy", activations)
