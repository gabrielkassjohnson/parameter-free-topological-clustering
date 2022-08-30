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
        
    :Requires: Numpy, Scikit-learn, Gudhi
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
                or distance matrix.
            y: Not used, present here for API consistency with scikit-learn by convention.
        """
        #check for data format, if data is a valid distance matrix, or a point cloud
        #scipy.spatial.distance.is_valid_dm(D, tol=0.0, throw=False, name='D', warning=False)
        #Main steps here
        self.X = X
        rips_complex = gudhi.RipsComplex(points=X)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
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
        inverted_normed_distance = np.power(inverted_normed_distance,3)
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
        candidates = [x for x in candidates if x > 2*simplex_tree.num_vertices()]
        idx = candidates[0]
        self.initial_idx_ = idx


        if self.overlapping_ == True:
            #increase weighting even more
            gap_vector = np.multiply(gap_vector, inverted_normed_distance)

            #renormalize
            gap_vector = gap_vector / np.sum(gap_vector)

            '''for i in range(20):
                if candidates[i] < simplex_tree.num_vertices() or \
                metrics.silhouette_score(self.X, np.array(list(cluster_dict_list[candidates[i+1]].values())), metric="euclidean") > \
                0.9 * metrics.silhouette_score(self.X, np.array(list(cluster_dict_list[self.initial_idx_].values())), metric="euclidean") and \
                candidates[i+1] < self.initial_idx_:
                    idx = candidates[i+1]'''
            
            idx = candidates[1]
            new_scaler = np.arange(len(gap_vector))
            scaler = MinMaxScaler()
            new_scaler = scaler.fit_transform(new_scaler.reshape(-1,1))
            new_scaler = 1 - new_scaler
            new_scaler = np.power(new_scaler,2)
            new_scaler = new_scaler.reshape(len(gap_vector))

            #inverted_normed_silhouette_array = np.multiply(silhouette_array,new_scaler[idx_array])

            for i in range(1,30):
                if candidates[i+1] < candidates[i]:
                    if idx>candidates[0]:
                        idx = candidates[i+1]
                        break

                    current_silhouette = metrics.silhouette_score(self.X,np.array(list(cluster_dict_list[idx].values())), metric="euclidean")
                    current_normed_silhouette = (current_silhouette + 1)/2
                    
                    current_score = np.multiply(current_normed_silhouette,gap_vector[idx])

                    current_scaled_score = np.multiply(current_score,new_scaler[idx])
                    
                    
                    new_silhouette = metrics.silhouette_score(self.X,np.array(list(cluster_dict_list[i+1].values())), metric="euclidean")
                    new_normed_silhouette = (new_silhouette + 1)/2

                    new_score = np.multiply(new_normed_silhouette,gap_vector[i+1])

                    new_scaled_score = np.multiply(current_score,new_scaler[i+1])
                    


                    if  new_score >  current_score :
                        idx = candidates[i+1]
                    else:
                        break




            #self.idx_array_ = idx_array
            #self.silhouette_array_ = silhouette_array   
        def replace_groups(data):
            a,b,c, = np.unique(data, True, True)
            _, ret = np.unique(b[c], False, True)
            return ret
                
        labels = np.array(list(cluster_dict_list[idx].values()))
        neg_idxs = np.where(labels==-1)
        labels = replace_groups(labels)
        labels[neg_idxs] = -1
            
        self.simplex_tree_ = simplex_tree
        self.n_clusters_ = np.count_nonzero(np.unique(np.array(list(cluster_dict_list[idx].values()))))
        self.confidence_ = '{:.1%}'.format(gap_vector[idx])
        self.gap_vector_ = gap_vector       
        self.cluster_dict_list_ = cluster_dict_list
        self.candidates_ = candidates
        self.idx_ = idx
        self.labels_ = labels


        return self


    def fit_predict(self, X, y=None, weights=None):
        """
        Equivalent to fit(), and returns the `labels_`.
        """
        return self.fit(X, y, weights).labels_


    def plot(self,plot_idx=None):
        """
        Creates and displays a matplotlib scatterplot of the dataset colored with predicted labels.
        """
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("prism_r").copy()
        #cmap.set_bad(cmap(0))

        cmap.set_under('black')
        #cmap.set_over('black')
        #cmap.set_bad("black")
        #cmap(number_of_clusters)
        
        c = self.labels_
        if plot_idx is not None:
            c = np.array(list(self.cluster_dict_list_[plot_idx].values()))

        plt.figure(figsize=(8,8))
        plt.scatter(self.X[:, 0], self.X[:, 1],
                    s=40, 
                    c=c,
                    marker="o",
                    cmap=cmap,
                    norm=None,
                    alpha=.8,
                    edgecolor="k",
                    vmin = 0)

        plt.show()
        
    def plot_silhouette_score(self, number_of_indices):
        import matplotlib.pyplot as plt

        idx_list = self.candidates_.copy()[:number_of_indices]
        idx_list.sort()
        silhouette_indices = np.asarray(idx_list)
        silhouette_list = []
        for i in idx_list:
            silhouette = metrics.silhouette_score(self.X, np.array(list(self.cluster_dict_list_[i].values())), metric="euclidean")
            silhouette_list.append(silhouette)
        silhouette_array = np.asarray(silhouette_list)
        plt.plot(silhouette_indices,silhouette_array)
        return silhouette_indices, silhouette_array
        
    def plot_rand_score(self, number_of_indices, ground_truth_labels):
        import matplotlib.pyplot as plt
        idx_list = self.candidates_.copy()[:number_of_indices]
        idx_list.sort()
        rand_indices = np.asarray(idx_list)
        rand_score_list = []
        for i in idx_list:
            pred = np.array(list(self.cluster_dict_list_[i].values()))
            rand_score = metrics.adjusted_rand_score( ground_truth_labels, pred)
            rand_score_list.append(rand_score)
        rand_score_array = np.asarray(rand_score_list)
        plt.plot(rand_indices,rand_score_array)
        return rand_indices, rand_score_array


    def animate_pogo(self,number_of_frames=10,fps=15,save=False,filename='pogo.gif'):
        """
        Creates and saves a matplotlib animation of the dataset colored with predicted labels.
        """
        import os.path
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        cmap = plt.cm.get_cmap("prism_r").copy()
        #cmap.set_bad(cmap(0))

        cmap.set_under('white')
        num_frames = number_of_frames
        interval = 1000/fps
        frames_array = self.candidates_.copy()
        #print(len(frames_array))
        #frames_array = frames_array[frames_array < pogo.idx_]
        #print(len(frames_array))

        frames_array = frames_array[:num_frames]
        #print(len(frames_array))

        frames_array.sort()
        vmax = max(np.array(list(self.cluster_dict_list_[frames_array[-1]].values())))
        num_frames = len(frames_array)
        #print(frames_array)

        fig, ax = plt.subplots(dpi=200)
        ax.set_axis_off()

        outfile = filename + str(num_frames) + 'frames.gif'
        #print(outfile)
        if not os.path.isfile(outfile):
            def init():
                scatter = ax.scatter(self.X[:, 0], self.X[:, 1],
                                s=25, 
                                c=np.array(list(self.cluster_dict_list_[0].values())),
                                marker="o",
                                cmap=cmap,
                                norm=None,
                                alpha=1,
                                edgecolor="k",
                                vmax=vmax,
                                vmin = 0)
                #ax.set(xlim=(-1, 35), ylim=(-1, 35))

                return scatter,

            #collection = PatchCollection(X, animated=True)

            #ax.add_collection(collection)
            #ax.autoscale_view(True)

            def animate(i):

                scatter = ax.scatter(X[:, 0], X[:, 1],
                            s=25, 
                            c=np.array(list(self.cluster_dict_list_[frames_array[i]].values())),
                            marker="o",
                            cmap=cmap,
                            norm=None,
                            alpha=1,
                            edgecolor="k",
                            vmax=vmax,
                            vmin=0)
                return scatter,



            ani = FuncAnimation(fig, animate,interval=interval,init_func=init,frames=num_frames,repeat=False, blit=True)

            #ani.save('animation.gif')



            #writer=animation.PillowWriter()

            #writer = animation.FFMpegWriter(fps=2,bitrate=1000)
            if save == True:
                ani.save(outfile)
            #fig.show()

