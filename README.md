# CHAMPS Predicting Molecular Properties

This repository contains the code for my approach to the [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/overview) Kaggle competition (Top 5% Finish). 

This repo is organized into two directories:

- **champs**: a local python package containing general code for the competition
- **scripts**: a collection of different approaches to the problem along with their training scripts

To install the **champs** package locally, run the following:

```
pip install .
```

The scripts directory contains three different approaches to the problem.

### Approach 1: Molecule GCN

This was the most successful approach out of the three. It involves modeling each molecule as a graph with the atoms as features and the chemical bonds between the atoms as the edges. Each node (atom) in the graph is represented with a feature vector containing several chemical descriptors including:

- **ACSF** (Atom-Centered Symmetry Functions)
- **LMBTR** (Local Many-Body Tensor Representations)
- **SOAP** (Smooth Overlap of Atomic Representations)

All of these descriptor vectors are calculated using the **dscribe** python library.

To predict the magnetic interaction between two atoms in the molecule, the graph goes through several iterations of message passing using a suite of GCNs (graph convolutional networks). Once the GCN is finished running and the feature vectors for each atoms have been learned, the vectors for the two target atoms are concatenated and passed through fully-connected neural network to predict the interaction value. The GCN which yielded the best performance was [EdgeConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv).

### Approach 2: Pairwise Random Forest

The second best approach involves the following steps:

- Identify the two target atoms in the molecule
- Find the shortest path between the two atoms using the chemical bonds as edges in the graph
- Collect features as you traverse the path
- Train a random forest on the extracted features

Some of the features extracted along the path include one-hot encoding of the atoms along the path, the chemical properties of the atoms along the path, and the dihedral angles along the path.

Note depending on the number of atoms along the path, the extracted feature vectors will be of different dimensions. Therefore, a separate random forest is trained on each of the eight different coupling types.

### Approach 3: Convolutional Neural Network

The third best approach involves the following steps:

- Represent each molecule as a three channel raster by stacking its matrix chemical descriptors. These descriptors include the Coulomb matrix, the adjacency matrix, and the CEP matrix.
- Train a fully convolutional neural network on the raster representation of the molecules to output a matrix containing the coupling constants between all of the target atom pairs.

The reason why this method did not do as well is because the matrix representations of the molecules were not as descriptive as other approaches (e.g. graph representation).