import os
import numpy as np

dists = [
    "truth",
    "smeared",
    "shifted",
    "tailcut",
    "all_smeared",
    "eta_smeared",
    "pt_smeared",
    "pt_shifted",
]
vals = []

for dist in dists:
    for i in range(9, 17):
        if os.path.exists(f"/graphganvol/hep-generative-models/classifier_trainings/{i}_{dist}"):
            with open(
                f"/graphganvol/hep-generative-models/classifier_trainings/{i}_{dist}/losses/aucs.txt",
                "r",
            ) as f:
                aucs = eval(f.read())

            vals.append(np.max(aucs))

print(vals)