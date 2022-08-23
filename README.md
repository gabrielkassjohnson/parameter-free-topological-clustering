# Proportional Gap Ordering: Parameter Agnostic Topological Clustering in Metric Spaces

This is an illustration of how Pogo proceeds through a dataset. You can see the clusters merging and stabilizing
into their most persisent arrangement. The goal of Pogo is to find the best cutoff within the sequence, and assign a confidence.

<img  src="varied140frames.gif" />

## Abstract

   Filtrations of simplicial complexes, and representations like barcodes and persistence diagrams, encode topological information about a set of points, or a network graph. The 0’th dimensional persistent homology includes only points and edges, forming connected components, i.e. clusters. The bottleneck and wasserstein distances on persistence diagrams have been proven to be stable, which gives a theoretical support to using these tools in noisy real-world scenarios. Vectorizations like persistence landscapes and persistent entropy can be used as inputs to machine learning models, extending usefuleness even further. These representations are a powerful tool for feature detection in predictive models. Bi-filtrations multiply these capabilites by introducing a second parameter to the filtration, such as a function accounting for density, or another known property of the data. Work from Bubenik and others has shown the fundamental ability of persistence diagrams to separate noise from topological features, and a fundamental connection between persistence diagrams and statistics, with the possibility of applying hypothesis testing within the framework of persistent homology. All of this taken together implies that filtrations contain inherently agnostic, yet rich statistical information about the clustering (and topological) behavior of datasets across scales and dimensions. Here, we propose a parameter agnostic clustering algorithm, based on the statistical partitioning of filtrations into noise and features, called Proportional Gap Ordering, It transforms a filtration of simplicial complexes into a probability vector, and chooses optimal cutoffs based on the behavior of connected components, producing cluster assignments for a data set, along with a measure of likeliness for each possible clustering.

## Introduction
Pogo is a clustering algorithm that works by building a filtration of simplicial complexes from a dataset,
either a point cloud, or a distance matrix, which can represent a network graph.
The algorithm calculates the gaps between persistent features, normalizes the gaps,
and scales them by their position within the filtration, giving the algorithm it's name,
Proportional Gap Ordering. This weights the output towards the beginning of the filtration.
The gaps are then transformed into a probability vector, harnessing the power of statistics
to make the decisions about cluster merging. The algorithm then merges the dataset hierarchically,
based on the assignment of clusters in 0th-dimension persistent homology, returning a cleaner probability vector.
The index of the maximum value is taken as the cutoff point within the filtration, and the simplicial complex
located at that cutoff is considered to be the most prominent clustering.
If the dataset has overlapping clusters, or is especially noisy, an additional step can be performed which moves
the cutoff back in time through the filtration by analyzing the silhouette values of other candidate indices
occuring earlier than the first choice. In noisy or overlapping datasets, this allows discovery of finer-grained
sub-clustering behavior, which is often what people want to know about a dataset, in addition to it's
global properties.   

## Background Concepts


## Procedure


## Test Sets
These are the exact test sets used in the scikitlearn tutorial on clustering. The only change is that the results of Pogo have
been added. It performs comparably well to the other algorithms, exhbiting behavior expected of a topological algorithm, i.e.
discerning shapes with intertwining features. Pogo is also capable of outputting outliers, which are shown as black data points.
It's currently not optimized for speed, as it's still in an experimental phase, but several easy improvements for speed and reducing algorithmic complexity are in the works.
<img  src="plot-cluster-comparison.png" />

## Conclusion and Next Steps


## References

Robert Ghrist. Barcodes: The persistent topology of data. Bulletin of the American Mathematical
Society, 45(1):61–75, 2008.

Bubenik, Peter. “Statistical topological data analysis using persistence landscapes.” J. Mach. Learn. Res. 16 (2015): 77-102.

Chazal, Frédéric, Leonidas J. Guibas, Steve Oudot and Primoz Skraba. “Persistence-Based Clustering in Riemannian Manifolds.” J. ACM 60 (2013): 41:1-41:38.

Cohen-Steiner, David, Herbert Edelsbrunner and John Harer. “Stability of Persistence Diagrams.” Discrete & Computational Geometry 37 (2007): 103-120.

Blumberg, Andrew J., and Michael Lesnick. "Stability of 2-Parameter Persistent Homology." (2020). 

Songdechakraiwut, Tananun, Bryan M. Krause, Matthew I. Banks, Kirill V. Nourski and Barry D. Van Veen. “Fast Topological Clustering with Wasserstein Distance.” ArXiv abs/2112.00101 (2021).

Chowdhury, Samir and Facundo Mémoli. “Convergence of Hierarchical Clustering and Persistent Homology Methods on Directed Networks.” ArXiv abs/1711.04211 (2017): n. pag.

Fasy, Brittany Terese, Fabrizio Lecci, Alessandro Rinaldo, Larry A. Wasserman, Sivaraman Balakrishnan and Aarti Singh. “Confidence sets for persistence diagrams.” The Annals of Statistics 42 (2014): n. pag.

Schweinhart, Benjamin. “Fractal dimension and the persistent homology of random geometric complexes.” Advances in Mathematics (2020): n. pag.

Atienza, Nieves, Rocío González-Díaz and Matteo Rucco. “Persistent entropy for separating topological features from noise in vietoris-rips complexes.” Journal of Intelligent Information Systems 52 (2017): 637-655.

Adler, Robert J., Omer Bobrowski and Shmuel Weinberger. “Crackle: The Persistent Homology of Noise.” arXiv: Probability (2013): n. pag.

Myers, Audun D., Firas A. Khasawneh and Brittany Terese Fasy. “ANAPT: Additive noise analysis for persistence thresholding.” Foundations of Data Science (2022): n. pag.

Wang, Bei. "Separating Features from Noise with Persistence and Statistics." Department of Computer Science, Duke University, Ph.D. Dissertation, 2010.