from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import statistics as stats

class GMM:
    def __init__(self, n_components, n_iter=10):
        self.n_components = n_components
        self.n_iter = n_iter
        
    def fit(self, X):
        n_samples, n_features = X.shape
        n_target = self.n_components

        # Initialize parameters
        KM = KMeans(n_clusters = n_target, random_state=0, n_init="auto").fit(X)
        Clustered_X = []
        for count, value in enumerate(KM.labels_):
          Sub_Cluster_X = []
          for i in range(n_features):
            Sub_Cluster_X.append(X[count,i])
          Sub_Cluster_X.append(value)
          Clustered_X.append(Sub_Cluster_X) 
        
        Clustered_X = np.array(Clustered_X)
        Covariance_matrix = []
        Mixing = []
        for target in range(n_target):
          Cluster_target = Clustered_X[Clustered_X[:,n_features]==target,0:n_features] #Need to modify this for n_features
          N_i = len(Cluster_target)/n_samples
          Covariance_matrix.append(np.corrcoef(Cluster_target.T))
          Mixing.append(N_i)

        self.mn = KM.cluster_centers_
        self.cov = Covariance_matrix
        self.mc = Mixing
        
        # Expectation-Maximization algorithm
        for _ in range(self.n_iter):
            # E-step
            self.phi = self.E_step(X)
            
            # M-step
            self.mn, self.cov, self.mc = self.M_step(X)
            
    def E_step(self, X):
        n_samples, n_features = X.shape
        n_target = self.n_components

        total_phi = []
        for i in range(n_target):
          phi = []
          for j in range(n_samples):
            gaussean_sum = []
            for k in range(n_target):
              if np.isnan(self.cov[k]).any() == False:
                init_mn = multivariate_normal(mean = self.mn[k],cov = self.cov[k], allow_singular = True)
                gaussean_sum.append(self.mc[k]*init_mn.pdf(X[j])) 
            gaussean_sum = sum(gaussean_sum)
            init_gauss = multivariate_normal(mean = self.mn[i],cov = self.cov[i])
            gauss = init_gauss.pdf(X[j])
            gaussean = self.mc[i]*gauss/gaussean_sum
            phi.append(gaussean)
          total_phi.append(phi)
        return total_phi
    
    def M_step(self, X):
        n_samples, n_features = X.shape
        n_target = self.n_components

        means = []
        covariances = []
        mixing_coefficients = []
        N_k = []
        #N_k
        for i in range(n_target):
          N_k.append(sum(self.phi[i]))

        #Means
        for i in range(n_target):
          mean_gauss_sum = []
          for j in range(n_samples):
            mean_gauss = self.phi[i][j]*X[j]
            mean_gauss_sum.append(mean_gauss)
          if np.isnan(sum(mean_gauss_sum)/N_k[i]).any() == False:
            means.append(sum(mean_gauss_sum)/N_k[i])
          else:
            means.append(self.mn[i])
        
        #Covariances
        for i in range(n_target):
          cov_sum = []
          for j in range(n_samples):
            cov_sum.append(self.phi[i][j]*((X[j]-means[i]).reshape(n_features,1) * (X[j]-means[i])))
          if np.isnan(sum(cov_sum)/N_k[i] + 0.001*np.identity(n_features)).any() == False:
            covariances.append(sum(cov_sum)/N_k[i] + 0.001*np.identity(n_features))
          else:
            covariances.append(self.cov[i])

        #Mixing coefficients
        for i in range(n_target):
          if np.isnan(N_k[i]/n_samples).any() == False:
            mixing_coefficients.append(N_k[i]/n_samples)
          else:
            mixing_coefficients.append(self.mc[i])

        return means, covariances, mixing_coefficients


    def predict(self, X, y):
        self.clusters = self.cluster(X)
        prediction = self.group(X, y)

        return prediction

    def cluster(self, X):
        n_samples, n_features = X.shape
        n_target = self.n_components

        Clusters = []
        for i in range(n_samples):
          Cluster_p = np.array(self.phi)[:,i]
          Clusters.append(np.where(Cluster_p==max(Cluster_p))[0][0])

        return Clusters

    def group(self, X, y):
        new_Cluster = self.clusters
        n_samples, n_features = X.shape
        n_target = self.n_components

        for i in range(n_target):
          Cluster_loc = []
          Cluster_target = []
          
          for j in range(n_samples):
            if new_Cluster[j] == i:
              Cluster_loc.append(j)
          

          for k in Cluster_loc:
            Cluster_target.append(y[k])

          if len(Cluster_target) > 0:
            Cluster_class = stats.mode(Cluster_target)

            for new_class in Cluster_loc:
              new_Cluster[new_class] = Cluster_class

        return new_Cluster
