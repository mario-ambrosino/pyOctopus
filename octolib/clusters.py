"""
Cluster Module
Description: Create, generate and handle Cluster objects.
Author: Mario Ambrosino
Date: 15/12/2020

"""
# Sys
import time

import numpy as np

# Project Libraries
import octolib.shared as shared
import octolib.track as track
from fastdtw import fastdtw
import pywt
from plotly.express import imshow


class Cluster(track.Track):

    def __init__(self, uid, radius: int = shared.CLUSTER_RADIUS):
        """
        Anomaly Cluster Init module. It loads all the anomalies for each side and sensor on super attributes,
        then locally generates a dictionary for slices of index centered in anomaly detected by the algorithm.
        TODO: include

        Parameters
        ----------
        uid: the unique identifier to select a specific dataset;
        radius: the radius (in samples unit) from the center of the point of interest.
        """
        super().__init__(uid)
        self.get_all_anomalies()
        self.clusters = {
            side:
                {
                    sensor:
                        {
                            index: {
                                "Center": center,
                                "Range":  radius,
                                "Slice":  (slice(
                                    max(0, center - radius),
                                    min(self.num_samples, center + radius)
                                    ) if abs(
                                    min(self.num_samples, center + radius) - max(0, center - radius)) == 2 * radius
                                           else None)} for (index, center) in
                            enumerate(self.anomaly_cluster[side][sensor][
                                          "Avg_Index"])
                            } for sensor in self.sensors
                    } for side in self.sides
            }
        self.clean_clusters()

    def __call__(self, side: str, sensor: int, cluster):
        """
        On call, the class Cluster instance takes from self.accel the slice centered on the cluster for a given side
        and sensor index.

        Parameters
        ----------
        side: int
            The side ("N" or "S") of the train;
        sensor: int
            The sensor, indexed by advancement order of the train (first index is the first sensor which sees an
            anomaly);
        cluster: int or str
            If int, the cluster index. If str, the key of the dictionary.

        Returns
        -------
        accel_slice: np.array
            The array slice for acceleration centered around the point of interest.

        """
        if isinstance(side, str) and isinstance(sensor, int) and isinstance(cluster, int):
            if (side in self.sides) and (sensor in self.sensors) and (
                    cluster in range(len(self.anomaly_cluster[side][sensor]["Avg_Index"]))):
                return self.accel[side][sensor][self.clusters[side][sensor][cluster]["Slice"]]
            else:
                print("[{}] # ".format(time.ctime()) + "ERROR: Outside valid interval - DEBUG: {}{}-{} ".format(
                    side, sensor, cluster)
                      )

    def clean_clusters(self):
        """
        Clean the cluster from null slices.
        Returns
        -------

        """
        for side in self.sides:
            for sensor in self.sensors:
                for (index, _) in enumerate(self.anomaly_cluster[side][sensor]["Avg_Index"]):
                    if self.clusters[side][sensor][index]["Slice"] is None:
                        self.clusters[side][sensor].pop(index)

    def get_all_anomalies(self,
                          threshold: float = shared.SIGMA_TH,
                          detection_range: float = shared.GROUND_WINDOW_DER,
                          epsilon: float = shared.CLUSTERING_EPS,
                          min_samples_cluster: int = shared.CLUSTERING_MS,
                          cluster_metrics: str = shared.CLUSTERING_METRICS,
                          ):
        """
        Takes all the anomaly clusters for a given dataset

        Parameters
        ----------
        threshold : float
            multiplication factor for the decision boundary for anomaly score
        detection_range : float
            range (in meters) to assess whether a cluster correctly verifies the hypothesis to be a welding.
        epsilon : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
        min_samples_cluster : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself.
        cluster_metrics : str
            The metric chosen for clustering in the set [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]

        Returns
        -------
            Nothing, writes it into the super(Track) instance attributes

        """
        print("[{}] # ".format(time.ctime()) + "Init All Anomalies Calculation for clustering")
        for side in self.sides:
            for sensor in range(4):
                self.anomaly_cluster[side][sensor]["Error_X"] = detection_range
                # North&South Side anomaly detection & z-score
                print("[{}] # ".format(time.ctime()) + "Sensor:{}/4 - Side:{}".format(sensor + 1, side))
                self.get_anomalies(side = side, sensor = sensor, threshold = threshold)
        print("[{}] # ".format(time.ctime()) + "Anomaly Detection completed.")
        self.get_anomaly_clusters(eps = epsilon,
                                  min_samples = min_samples_cluster,
                                  metric = cluster_metrics)
        self.evaluate_prediction(error_length = detection_range)

    def get_cwt(self, side, sensor, index,
                scales = shared.SCALES,
                wavelet = shared.WAVELET,
                samp_period = 1 / shared.SAMPLING_FREQUENCY, mode="coeff"):
        """
        Returns Continuous Wavelet Transform. Wrapper for pywt function
        Parameters
        ----------
        side: str = {"N","S"}
            Side of the train
        sensor: int = {0,1,2,3}
            Index for the sensor
        index: int
            Index for number of cluster
        scales:
            Number of scales for the pywt.cwt
        wavelet:
            Wavelet selected for the pywt.cwt
        samp_period:
            Sampling Period for the pywt.cwt
        Returns
        -------
        wavelet_coeff: np.array
            2-dim array containing the complex values of the wavelet function calculated for the signal.
        wave_freq: np.array
            1-dim array containing the frequency corresponding to scales.

        """
        wavelet_coeff, wave_freq = pywt.cwt(
            data = np.nan_to_num(self(side = side, sensor = sensor, cluster = index)),
            scales = scales,
            wavelet = wavelet,
            sampling_period = samp_period
            )
        if mode == "coeff":
            return wavelet_coeff
        if mode == "wsd":
            return np.abs(wavelet_coeff**2)


    def dtw_dist(self, side, sensor, x_index, y_index):
        """
        Calculate distance between two clusters x and y  using DTW

        Parameters
        ----------
        side: str = {"N","S"}
            Side of the train
        sensor: int = {0,1,2,3}
            Index for the sensor
        x_index: int
            Index for cluster x
        y_index: int
            Index for cluster y

        Returns
        -------
        dtw_distance: float
            the DTW distance.

        """
        dtw_distance, _ = fastdtw(
            x = self(side = side, sensor = sensor, cluster = x_index),
            y = self(side = side, sensor = sensor, cluster = y_index),
            )
        return dtw_distance

    def plot_all_clusters(self):
        for side in self.sides:
            for sensor in self.sensors:
                for index in range(len(self.clusters[side][sensor])):
                    wave = self.get_cwt(side = side, sensor = sensor, index = index, mode = "wsd")
                    fig = imshow(wave)
                    fig.write_image("images/WSD_clusters/WSD_T{}_D{}_S{}_C{}_L{}_A{}_CLU{}.png".format(
                        self.train, self.direction, self.avg_speed, self.component, side, sensor, index)
                        )
                    print("Written image in 'images/WSD_clusters/WSD_T{}_D{}_S{}_C{}_L{}_A{}_CLU{}.png'".format(
                        self.train, self.direction, self.avg_speed, self.component, side, sensor, index)
                        )


    def dtw_dist_matrix(self, side, sensor):
        """
        Dynamic Time Warp Distance Matrix between elements of a clustering class.

        STATUS:
        -------
        Failure: the fastdtw routine is slow and leads to results which aren't useful at the moment. To be further
        investigated.

        TODO: talk to Norman

        Parameters
        ----------
        side: str
            train side
        sensor: int
            sensor id

        Returns
        -------
        dtw_dist_matric: np.array(shape(n_signals,n_signals))
            a matrix with the distances betwee

        """
        n_signals = len(self.clusters[side][sensor].keys())
        print(n_signals)
        dtw_dist_matrix = np.empty((n_signals, n_signals))
        for x_index in range(0, n_signals):
            for y_index in range(0, x_index):
                dtw_dist_matrix[x_index, y_index] = self.dtw_dist(side, sensor, x_index, y_index)
                dtw_dist_matrix[y_index, x_index] = dtw_dist_matrix[x_index, y_index]
        return dtw_dist_matrix
