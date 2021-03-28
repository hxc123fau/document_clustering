import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import distance


class cosine_kmeans():

    # 欧氏距离计算
    def distance_cos(self,x, y):
        return np.sqrt(np.sum((x - y) ** 2))  # euclidean
        # return (distance.cosine(x, y))  # cosine

    # 为给定数据集构建一个包含K个随机质心的集合
    def randCent(self,dataSet, k):
        m, n = dataSet.shape
        centroids = np.zeros((k, n))
        for i in range(k):
            index = int(np.random.uniform(0, m))  #
            centroids[i, :] = dataSet[index, :]
        return centroids

    # k均值聚类
    def kmeans_open(self,dataSet, k):
        m = np.shape(dataSet)[0]  # 行的数目
        # 第一列存样本属于哪一簇
        # 第二列存样本的到簇的中心点的误差
        clusterAssment = np.mat(np.zeros((m, 2)))
        clusterChange = True

        # 第1步 初始化centroids
        centroids = self.randCent(dataSet, k)
        while clusterChange:
            clusterChange = False

            # 遍历所有的样本（行数）
            for i in range(m):
                minDist = 100000.0
                minIndex = -1

                # 遍历所有的质心
                # 第2步 找出最近的质心
                for j in range(k):
                    # 计算该样本到质心的欧式距离
                    # distance = self.distance_cos(centroids[j, :], dataSet[i, :])
                    distance = self.distance_cos(centroids[j, :], dataSet[i, :])
                    if distance < minDist:
                        minDist = distance
                        minIndex = j
                # 第 3 步：更新每一行样本所属的簇
                if clusterAssment[i, 0] != minIndex:
                    clusterChange = True
                    clusterAssment[i, :] = np.int(minIndex), minDist ** 2
            # 第 4 步：更新质心
            for j in range(k):
                pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
                centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

        # print('clusterAssment.A[:, 0]', centroids.shape, type(clusterAssment.A[:, 0][0]))
        return centroids, clusterAssment.A[:, 0].astype(np.int64)


# if __name__=="__main__":
#     ds,y=make_blobs(n_samples=100, n_features=2,centers=4)
#     print('ds',ds.shape)
#     k=4
#     result, cores = kmeans_open(ds, k)
#
#     plt.scatter(ds[:, 0], ds[:, 1], s=1, c=result.astype(np.int))
#     plt.scatter(cores[:, 0], cores[:, 1], marker='x', c=np.arange(k))
#     plt.show()