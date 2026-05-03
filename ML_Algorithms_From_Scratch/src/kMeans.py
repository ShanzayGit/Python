import numpy as np
import random
import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def kmeans(data, k, max_iters=100):

    data = np.array(data)
    n_samples, n_features = data.shape

    np.random.seed(42)
    random.seed(42)
    centroids = np.array(random.sample(list(data), k))

    cluster = np.zeros(n_samples, dtype=int)


    for l in range(max_iters):

        # Assign clusters
        for i in range(n_samples):
            min_dist = distance(data[i], centroids[0])
            cluster[i] = 0

            for c in range(1, k):
                d = distance(data[i], centroids[c])
                if d < min_dist:
                    min_dist = d
                    cluster[i] = c

        # Prepare new centroids
        new_centroids = np.zeros((k, n_features))

        x = [0] * k
        y = [0] * k
        count = [0] * k

        # Calculate sums
        for j in range(n_samples):
            c = cluster[j]
            x[c] += data[j][0]
            y[c] += data[j][1]
            count[c] += 1


        # Update centroids
        for c in range(k):
            if count[c] > 0:
                new_centroids[c][0] = x[c] / count[c]
                new_centroids[c][1] = y[c] / count[c]
            else:
                # Keep old centroid if centroid is not assigned to any point
                new_centroids[c] = centroids[c]

        # stop if old and new centroids are same
        if np.allclose(centroids, new_centroids):
           break

        print("Iteration #", l+1)
        print("Data Points:\n", data)
        print("Cluster Assignments:", cluster)
        print("Centroids:\n", centroids)
        print()
        centroids = new_centroids

    return centroids, cluster