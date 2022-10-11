def get_first_max(array):
    for i in range(len(array)):
        if idx_array[i] > 2*simplex_tree.num_vertices():

            if array[i] > array[i+1] and array[i+2]and array[i+3]and array[i+4]:

                return float(array[i])
              
first_max = get_first_max(silhouette_array)
idx_array[np.where(silhouette_array == first_max)]

idx = int(idx_array[np.where(silhouette_array == first_max)])
idx