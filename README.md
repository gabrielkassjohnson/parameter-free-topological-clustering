<img  src="varied100frames.gif" />

<img  src="aniso60frames.gif" />

<p>
    Filtrations of simplicial complexes, and their representations, encode topological information about a set of points, or a network graph, represented as a distance matrix. The 0’th dimensional persistent homology includes only points and edges, forming connected components, i.e. clusters. The bottleneck and wasserstein distances on persistence diagrams, have been proven to be stable, which gives a theoretical support to using these tools in noisy real-world scenarios. Vectorizations like persistence landscapes, and persistent entropy can be used as inputs to machine learning models, extending usefuleness even further. These representations are a powerful tool for feature detection, and the characterization of complex data with numerical invariants. Bi-filtrations extend these capabilites by introducing a second parameter to the filtration, such as a function accounting for density, or another known property of the data. Work from Bubenik and others has shown the fundamental ability of persistence diagrams to separate noise from truly persistent topological features, and a basic connection between persistence diagrams and statistics, with an ability to apply hypothesis testing to a set of barcodes. All of this taken together implies that persistence diagrams contain inherently agnostic, yet rich information about the clustering (and higher dimensional) behavior of datasets across scales and dimensions, similar to the structure of hierarchical clustering. Here, we propose a parameter agnostic clustering algorithm, based on the statistical partitioning of persistence diagrams into noise and features, called Proportional Gap Ordering, It transforms a filtration of simplicial complexes into a probability vector, and chooses cutoffs based on the behavior of connected components. returning cluster assignments for a data set, along with a statistical confidence for each possible clustering.
</p>