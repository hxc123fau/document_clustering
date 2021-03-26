from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from  scipy.spatial.distance import euclidean
import numpy as np
import scipy
from sklearn.metrics import pairwise_distances
import spacy
import pandas as pd


class k_mean():
    def __init__(self):
        pass

    def elbow_method(self,input_data,input_title,K = range(1, 15)):
        # K = range(1, 10)
        distortions = []
        for k in K:
            kmean_model = KMeans(n_clusters=k).fit(input_data)
            distance = sum(np.min(cdist(input_data, kmean_model.cluster_centers_,
                                        'euclidean'), axis=1)) / input_data.shape[0]
            distortions.append(distance)

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        new_title=input_title+' Elbow Method optimal k'
        plt.title(new_title)
        plt.show()


    def gap_statistic(self,data,input_title, refs=None, nrefs=20, ks=range(1, 15)):
        dst = euclidean
        shape = data.shape
        if not refs:
            tops = data.max(axis=0)
            bottoms = data.min(axis=0)
            dists = (scipy.diag(tops - bottoms))
            rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
            for i in range(nrefs):
                rands[:, :, i] = np.dot(rands[:, :, i],dists) + bottoms
        else:
            rands = refs
        gaps = scipy.zeros((len(ks),))
        for (i, k) in enumerate(ks):
            # k_means_args_dict['n_clusters'] = k
            kmean_model = KMeans(k)
            kmean_model.fit(data)
            (cluster_centers, point_labels) = kmean_model.cluster_centers_, kmean_model.labels_
            disp = sum(
                [dst(data[current_row_index, :], cluster_centers[point_labels[current_row_index], :]) for
                 current_row_index
                 in range(shape[0])])
            refdisps = scipy.zeros((rands.shape[2],))
            for j in range(rands.shape[2]):
                kmean_model = KMeans(k)
                kmean_model.fit(rands[:, :, j])
                (cluster_centers, point_labels) = kmean_model.cluster_centers_, kmean_model.labels_
                refdisps[j] = sum(
                    [dst(rands[current_row_index, :, j], cluster_centers[point_labels[current_row_index], :]) for
                     current_row_index in range(shape[0])])
            # let k be the index of the array 'gaps'
            gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)

        plt.plot( gaps, 'bx-')
        plt.xlabel('k')
        plt.ylabel('gaps')
        new_title=input_title+' gap statistic optimal k'
        plt.title(new_title)
        plt.show()

    def optimalK(self,data, nrefs=20, maxClusters=15):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
        for gap_index, k in enumerate(range(1, maxClusters)):

            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)

            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):
                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)

                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)

                refDisp = km.inertia_
                refDisps[i] = refDisp

            # Fit cluster to original data and create dispersion
            km = KMeans(k)
            km.fit(data)

            origDisp = km.inertia_

            # Calculate gap statistic
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)

            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap

            resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

        # gaps.argmax() + 1
        plt.plot(resultsdf.clusterCount, resultsdf.gap, linewidth=3)
        plt.scatter(resultsdf[resultsdf.clusterCount == (gaps.argmax() + 1)].clusterCount,
                    resultsdf[resultsdf.clusterCount == (gaps.argmax() + 1)].gap, s=250, c='r')
        plt.grid(True)
        plt.xlabel('Cluster Count')
        plt.ylabel('Gap Value')
        plt.title('Gap Values by Cluster Count')
        plt.show()
        # return (gaps.argmax() + 1,
        #         resultsdf)
        # # Plus 1 because index of 0 means 1 cluster is optimal,
        # # index 2 = 3 clusters are optimal




