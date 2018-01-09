import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

X = np.array([[1,2],
              [2,3],
              [12,12],
              [8,9],
              [10,11],
              [10,10],
              [1,3]])

#plt.scatter(X[:,0],X[:,1],s=150)
#plt.show()

class K_means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [(np.linalg.norm(featureset-self.centroids[centroid]))for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100) > self.tol:
                    optimized =False

            if optimized:
                break

    def predict(self):
        pass

colors = 10*["r","b","k","g","c"]
clf = K_means()
clf.fit(X)
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker="o",color="k",s=150,linewidths=5)


for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:

        print(featureset)
        plt.scatter(featureset[0],featureset[1],marker="x",color=color,s=150,linewidths=5)

plt.show()
