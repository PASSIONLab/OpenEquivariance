"""
This script parse the repository of
Nequip input files at
https://github.com/mir-group/nequip-input-files.
We extract the node / edge hidden features representations.
"""

import os
import yaml


def process_nequip_configs():
    nequip_files = []
    for root, dirs, files in os.walk("../data/nequip-input-files"):
        for file in files:
            if file.endswith(".yaml"):
                nequip_files.append(os.path.join(root, file))

    irrep_pairs = []
    configs = []
    for file in nequip_files:
        with open(file, "r") as f:
            data = yaml.unsafe_load(f)
            filename = os.path.splitext(os.path.basename(file))[0]
            feature_irreps_hidden = data["feature_irreps_hidden"]
            irreps_edge_sh = data["irreps_edge_sh"]
            if (feature_irreps_hidden, irreps_edge_sh) not in irrep_pairs:
                irrep_pairs.append((feature_irreps_hidden, irreps_edge_sh))
                configs.append((feature_irreps_hidden, irreps_edge_sh, filename))

    for config in configs:
        print(config)


if __name__ == "__main__":
    process_nequip_configs()
