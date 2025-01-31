Here is where we preprocessed our data for the different regression methods.
First we pull our data from our database with `pull_data/gaspy/pull.py` using the [GASpy](https://github.com/ulissigroup/GASpy) API, which creates the `pull_data/gaspy/docs.pkl` file.

Then to use [CGCNN](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.9b01428), we need to convert the data to a graph format and then to a matrix format.
This was done with the `sdt/create_sdt.py` file, which requires the `sdt/atom_init.json` file that we provided as a seed.
The `sdt/create_sdt.py` creates the `sdt/sdt.pkl` file of preprocessed data meant for use by CGCNN and the `sdt/feature_dimensions.pkl` file also meant for use by CGCNN.
All of these files contain all of our adsorption energy data for CO.

To use our GP model, we applied the fingerprinting method outlined in our seminal GASpy [paper](https://www.nature.com/articles/s41929-018-0142-1).
This is done by the `fingerprint/fingerprint.ipynb`, which saves the `fingerprent/fingerprints.pkl` file.

Lastly, we do a train/validate/test split using the `split_data.ipynb` notebook, which creates the `splits.pkl` cache.
We use this cache in our ML experiments.
