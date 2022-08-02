import numpy as np
import gudhi
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Pogo:
    """
    Clustering algorithm that begins by building a filtration of simplicial complexes from a dataset,
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
        
    :Requires: Numpy, SciPy, Scikit-learn, Gudhi
    Attributes
    ----------
    n_clusters_: int
        The number of clusters.

    labels_: ndarray of shape (n_samples,)
        cluster labels for each point, taken from the candidates list
        
    candidates_: list of ints
        list of the top indices likely to be a cutoff
    
    
    """
    def __init__(
        self,
        overlapping = False
    ):
        """
        Args:
            overlapping (bool): Set to True if the dataset contains overlapping clusters,
            or if the dataset contains a lot of noise,
            or if the researcher wishes to see the finer sub-clusters.
            Set to False for relatively non-overlapping datasets.
            Default is False. 
        """
        self.overlapping_ = overlapping
        

    def fit(self, X, y=None):
        """
        Args:
            X ((n,d)-array of float|(n,n)-array of float|Sequence[Iterable[int]]): coordinates of the points,
                or distance matrix (full, not just a triangle) if metric is "precomputed", or list of neighbors
                for each point (points are represented by their index, starting from 0) if graph_type is "manual".
                The number of points is currently limited to about 2 billion.
            weights (ndarray of shape (n_samples)): if density_type is 'manual', a density estimate at each point
            y: Not used, present here for API consistency with scikit-learn by convention.
        """
        #check for data format, if data is a valid distance matrix, or a point cloud
        #scipy.spatial.distance.is_valid_dm(D, tol=0.0, throw=False, name='D', warning=False)
        #Main steps here
        self.X = X
        rips_complex = gudhi.RipsComplex(points=X)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
        self.num_vertices_ = simplex_tree.num_vertices()
        self.simplex_tree_ = simplex_tree
        #move through list and assign clusters to conected components
        point_dict={i:-1 for i in range(simplex_tree.num_vertices())}
        counter=0
        cluster_dict_list = []
        distance_list = []
        for simplex in simplex_tree.get_filtration():
            if len(simplex[0])>1:
                if all(value > 0 for value in list(point_dict.values())):
                    if len(np.unique(np.array(list(point_dict.values())))) == 1:
                        #print('break')
                        #print(simplex)
                        #print(simplex[1])
                        simplex_tree.prune_above_filtration(simplex[1])
                        break


                #if both points are still in cluster 0, assign both to a new cluster
                if point_dict[simplex[0][0]] == -1 and point_dict[simplex[0][1]] == -1:
                    counter += 1
                    point_dict[simplex[0][0]] = counter
                    point_dict[simplex[0][1]] = counter
                    #if one point is in cluster 0 and one is not, assign the one in cluster 0 to the existing cluster
                elif point_dict[simplex[0][0]] == -1 and point_dict[simplex[0][1]] != -1:
                    point_dict[simplex[0][0]] = point_dict[simplex[0][1]]

                    #and vice versa
                elif point_dict[simplex[0][0]] != -1 and point_dict[simplex[0][1]] == -1:
                    point_dict[simplex[0][1]] = point_dict[simplex[0][0]]

                    #if both points are not in cluster 0 and not in the same cluster, merge clusters to the lower number cluster
                elif point_dict[simplex[0][0]] != -1 and point_dict[simplex[0][1]] != -1 and point_dict[simplex[0][0]] != point_dict[simplex[0][1]]:
                    larger_cluster_number = max(point_dict[simplex[0][0]], point_dict[simplex[0][1]])
                    smaller_cluster_number = min(point_dict[simplex[0][0]], point_dict[simplex[0][1]])
                    for key, value in point_dict.items():
                        if value == larger_cluster_number:
                            point_dict[key] = smaller_cluster_number

                distance_list.append(simplex[1])
                cluster_dict_list.append(point_dict.copy())

        length = len(cluster_dict_list)
        distance_array = np.array(distance_list)

        #find the gaps between birth/death pairs
        gaps = np.diff(distance_array)
        
        #add a zero back to the beginning of the gaps
        gaps = np.concatenate([np.zeros(1),gaps])

        #find normalized distance
        scaler = MinMaxScaler()
        normed_distance = scaler.fit_transform(distance_array.reshape(-1,1)).T.reshape(length)
        
        inverted_normed_distance = 1 - normed_distance

        #and square it to increase the weighting
        inverted_normed_distance = np.power(inverted_normed_distance,2)
        normed_gaps = np.multiply(gaps, inverted_normed_distance)

        #normalize to create a probability vector
        gap_vector = normed_gaps / np.sum(normed_gaps)
        
        marker = 0
        for i in range(1,length-1):
            if cluster_dict_list[marker] == cluster_dict_list[i]:

                gap_vector[marker] += gap_vector[i]
                gap_vector[i] = 0
                #print(marker)


            else:
                marker = i

        candidates = np.flip(np.argsort(gap_vector))
        counter = 0
        idx = candidates[counter]
        if idx < simplex_tree.num_vertices():
            counter += 1
            idx = candidates[counter]
        self.initial_idx_ = idx
            
            
        

        
        

        if self.overlapping_ == True:
            idx_list = []
            for i in range(len(candidates)):
                if candidates[i] > simplex_tree.num_vertices():
                    idx_list.append(candidates[i])
                else:
                    break

            idx_list = candidates.copy()[:100]
            idx_list.sort()
            idx_array = np.asarray(idx_list)
            silhouette_list = []
            for i in idx_list:

                silhouette = metrics.silhouette_score(X, np.array(list(cluster_dict_list[i].values())), metric="euclidean")
                silhouette_list.append(silhouette)
            
            silhouette_array = np.asarray(silhouette_list)
            new_scaler = np.arange(len(gap_vector))
            scaler = MinMaxScaler()
            new_scaler = scaler.fit_transform(new_scaler.reshape(-1,1))
            new_scaler = 1 - new_scaler
            new_scaler = np.power(new_scaler,2)
            new_scaler = new_scaler.reshape(len(gap_vector))

            inverted_normed_silhouette_array = np.multiply(silhouette_array,new_scaler[idx_array])
            
            idx = idx_array[inverted_normed_silhouette_array.argmax()]
            self.idx_array_ = idx_array
            self.silhouette_array_ = inverted_normed_silhouette_array
        else:
            pass
                
            
        pred = np.array(list(cluster_dict_list[idx].values()))
        number_of_clusters = np.count_nonzero(np.unique(np.array(list(cluster_dict_list[idx].values()))))

        
        self.n_clusters_ = number_of_clusters
        self.confidence_ = '{:.1%}'.format(gap_vector[idx])

        self.idx_ = idx
        self.gap_vector_ = gap_vector       
        self.cluster_dict_list_ = cluster_dict_list
        self.candidates_ = candidates
        self.labels_ = pred


        return self


    def fit_predict(self, X, y=None, weights=None):
        """
        Equivalent to fit(), and returns the `labels_`.
        """
        return self.fit(X, y, weights).labels_


    def plot_diagram(self):
        """
        """
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("prism_r").copy()
        #cmap.set_bad(cmap(0))

        cmap.set_under('white')
        cmap.set_over('black')
        cmap.set_bad("black")
        #cmap(number_of_clusters)
        
        size=8
        plt.figure(figsize=(size,size))
        plt.scatter(self.X[:, 0], self.X[:, 1],
                    s=40, 
                    c=self.labels_,
                    marker="o",
                    cmap=cmap,
                    norm=None,
                    alpha=.7,
                    edgecolor="k",
                    vmin = 0)

        plt.show()
