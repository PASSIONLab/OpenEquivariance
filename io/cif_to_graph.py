import pickle
import numpy as np
from sklearn.neighbors import radius_neighbors_graph


def cif_to_molecular_graph(cif_file, cp, radii):
    with open(f"../data/cif_files/{cif_file}", "r") as f:
        print("Started reading file...")
        lines = f.readlines()
        print("Finished reading file!")

        coords = []
        for line in lines:
            if line.startswith("ATOM"):
                parts = line.split()
                coords.append(
                    [float(parts[cp[0]]), float(parts[cp[1]]), float(parts[cp[2]])]
                )

        coords = np.array(coords)

        for radius in radii:
            print(f"Starting radius neighbors calculation, r={radius}")
            A = radius_neighbors_graph(
                coords, radius, mode="connectivity", include_self=False
            )
            print(f"Finished radius neighbors calculation, found {A.nnz} nonzeros.")

            # mmwrite(f'../data/molecular_structures/{cif_file.split(".")[0]}.mtx', A)

            coo_mat = A.tocoo()
            result = {"row": coo_mat.row, "col": coo_mat.col, "coords": coords}

            with open(
                f"../data/molecular_structures/{cif_file.split('.')[0]}_radius{radius}.pickle",
                "wb",
            ) as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # cif_to_molecular_graph('hiv_capsid.cif', (10, 11, 12), radii=[2.0, 2.5, 3.0, 3.5])
    cif_to_molecular_graph("covid_spike.cif", (10, 11, 12), radii=[2.0, 2.5, 3.0, 3.5])
    cif_to_molecular_graph("1drf.cif", (10, 11, 12), radii=[6.0])
