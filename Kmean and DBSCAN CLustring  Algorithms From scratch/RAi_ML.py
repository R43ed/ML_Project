import numpy as np
import numpy.random as npr
import time
import pandas as pd
import matplotlib.pyplot as plt
from statistics import median
import random


class KMean:

    def __init__(self, cluster):
        self.cluster = cluster
        self.last_centroids = []
        self.new_centroids = []
        self.new_target = []
        self.colors = []
        self.__X = []

    # ======================================
    def eucldis(self, x):
        n = x.shape[0]
        dic = np.zeros((n, self.cluster))
        for i in range(self.cluster):
            dic[:, i] = ((x - self.last_centroids[i]) ** 2).sum(axis=1) ** 0.5
        return dic

    # ======================================
    def target(self, x):
        k = self.cluster
        n = x.shape[0]
        cap = n // k
        colors = np.zeros((x.shape[0],))

        for i in range(k):
            colors[i * cap: cap * (i + 1)] = i

        return colors

    # ======================================
    def fit(self, x):

        self.last_centroids = []
        k = self.cluster
        x_n = x.shape[0]

        # Generate random centroides(centers)
        self.last_centroids = x[npr.choice(x_n, k, replace=False)]
        # -----------------------------------------------------------------------#
        # Euclidean Distance between all points to each centroid
        clusters_dis = self.eucldis(x)
        # All points close to which cluster (0 = first cluster , 1 = second cluster ......)
        closest = np.argmin(clusters_dis, axis=1)

        # x[closest == 0] all points close(belong) to this cluster
        self.new_centroids = np.zeros((self.cluster, x.shape[1]))

        # updating Centroids
        for i in range(self.cluster):
            self.new_centroids[i, :] = x[closest == i].mean(axis=0)

        while not np.array_equal(self.last_centroids, self.new_centroids):
            self.last_centroids = np.copy(self.new_centroids)
            # Euclidean Distance between all points to each centroid
            clusters_dis = self.eucldis(x)

            # All points close to which cluster (0 = first cluster , 1 = second cluster ......)
            closest = np.argmin(clusters_dis, axis=1)

            # x[closest == 0] all points close(belong) to this cluster
            self.new_centroids = np.zeros((self.cluster, x.shape[1]))

            # updating Centroids
            for i in range(self.cluster):
                self.new_centroids[i, :] = x[closest == i].mean(axis=0)
        self.colors = closest
        self.__X = x
        print("KMean Algorithem completed")

    # ======================================
    def model_plot(self, centroids=None, color_set=None):
        colors = self.colors * 2
        x = self.__X
        if centroids == True:
            plt.scatter(x[:, 0], x[:, 1], s=7, c=colors, cmap=color_set)
            for pair in self.new_centroids:
                plt.plot(pair[0], pair[1], "kx", markersize=5, mew=2)
        else:
            plt.scatter(x[:, 0], x[:, 1], c=self.colors)


# ==========================================================================
# ==========================================================================
# ==========================================================================

class DBSCAN:
    # Classify point to 1:Core , 2:border , 0:Noies
    def __init__(self, data, eps, minPts):
        self.eps = eps
        self.minPts = minPts
        self.__data = data
        self.dataSet = pd.DataFrame(self.__data, columns=['x', 'y'])
        self.dataSet["state"] = 0
        self.dataSet["Clusters"] = 0
        self.core = []
        self.border = []
        self.noise = []
#----------------------------------------------------------------------------------------------------        
    def distance_matrix(self):
        """ Create a distance matrix sampler to Euclidean Distance matrix """
        s = np.array([[complex(i[0], i[1]) for i in self.__data]])
        dist_matrix = abs(s.T - s)

        return dist_matrix
#----------------------------------------------------------------------------------------------------
    def best_eps(self):
        """ Help to chose the best epsilon value """
        n = len(self.__data)
        dist = self.distance_matrix()
        df = pd.DataFrame(dist)

        dis_Kth_neighbor = [df.iloc[:, i].sort_values()[1:self.minPts + 1].mean() for i in range(n)]
        new_y = sorted(dis_Kth_neighbor)
        
        
        new_x = pd.Series(new_y).index
        plt.grid()
        plt.xlabel(f"Points Sorted According to Distance of {self.minPts}TH Nearest Neighbor")
        plt.ylabel(f"{self.minPts}TH Nearest Neighbor Distance")
        
        plt.plot(new_x, new_y, "b-")
#----------------------------------------------------------------------------------------------------        
    def fit(self):
        """ Apply the DBSCAN Algorithm """
        #     Classify point to 1:Core , 2:border , 0:Noise
        dist = self.distance_matrix()
        n_points = self.__data.shape[0]
        
        #check for core points
        for i in range(n_points):
            
            Kth_NN = len(self.naighbors(i))
            if Kth_NN >= self.minPts:
                self.dataSet.loc[i,"state"] = 1
                
        #check for non-core points        
        for p in range(n_points):
            if self.is_core(p):
                continue
            else:
                for i in self.naighbors(p):
                    if self.is_core(i):
                        self.dataSet.loc[p,"state"] = 2
                        break

        self.core = self.dataSet.query("state == 1 ")
        self.border = self.dataSet.query("state == 2 ")
        self.noise = self.dataSet.query("state == 0 ")
#----------------------------------------------------------------------------------------------------        
    def naighbors(self,x):
        """ presents the point naighbors """
        dist = self.distance_matrix() 
        Connected_points = pd.Series(dist[:,x]).sort_values()
        x_naighbors = Connected_points[Connected_points<=self.eps][1:].index
        return x_naighbors
#----------------------------------------------------------------------------------------------------        
    def is_core(self,x) -> bool:
        if self.dataSet.state.iloc[x] == 1:
            return True
        else:
            return False
        
#----------------------------------------------------------------------------------------------------        
    def is_border(self,x) -> bool:
        if self.dataSet.state.iloc[x] == 2:
            return True
        else:
            return False 
        
#----------------------------------------------------------------------------------------------------        
    def is_noise(self,x) -> bool:
        if self.dataSet.state.iloc[x] == 0:
            return True
        else:
            return False       
        
#----------------------------------------------------------------------------------------------------    
    def predict(self):
        Ci = 0
        for x in self.dataSet.index:
            
            if self.dataSet.Clusters.iloc[x] != 0:
                continue
                
            if self.is_core(x):
                Ci+=1
                self.__vist_naighbors(x,Ci)
        
        return self.dataSet["Clusters"]   
    
#----------------------------------------------------------------------------------------------------    
    def __vist_naighbors(self,x,Ci):
            self.dataSet.loc[x,["Clusters"]] = Ci
            for y in self.naighbors(x):
                #if has a cluster continue
                if self.dataSet.Clusters.iloc[y] != 0:
                    continue
                #mark x     
                self.dataSet.loc[y,["Clusters"]] = Ci        
                if self.is_core(y):
                    self.__vist_naighbors(y,Ci)
