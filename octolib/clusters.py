"""
Cluster Module
Description: Create, generate and handle Cluster objects.
Author: Mario Ambrosino
Date: 15/12/2020

"""
# Sys
import time

# Project Libraries
import octolib.shared as shared
import octolib.track as track


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

    def get_all_anomalies(self, threshold: float = shared.SIGMA_TH,
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
