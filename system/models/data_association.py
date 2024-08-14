import numpy as np
from scipy.optimize import linear_sum_assignment

def associate_detections_to_tracks(image, detections, tracks, cost_metric):
    cost_matrix = np.zeros((len(detections), len(tracks)))
    for d, det in enumerate(detections):
        for t, track in enumerate(tracks):
            cost_matrix[d, t] = cost_metric(image, det, track)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    associations = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))]
    return associations