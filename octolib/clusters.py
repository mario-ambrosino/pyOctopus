"""
Cluster Module
Description: Create, generate and handle Cluster objects.
Author: Mario Ambrosino
Date: 15/12/2020

"""
# Project Libraries

import octolib.track as track


class Cluster(track.Track):
    def __init__(self, uid):
        super(Cluster, self).__init__(uid)
    pass
