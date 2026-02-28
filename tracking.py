import cv2
import numpy as np


class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False
        self.frames_without_measurement = 0
        self.max_frames_without_measurement = 15

    def predict(self):
        if not self.initialized:
            return None
        predicted = self.kf.predict()
        return predicted[0, 0], predicted[1, 0]

    def update(self, cx, cy):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

        if not self.initialized:
            self.kf.statePost = np.array([
                [np.float32(cx)],
                [np.float32(cy)],
                [np.float32(0)],
                [np.float32(0)],
            ])
            self.initialized = True
        else:
            self.kf.correct(measurement)

        self.frames_without_measurement = 0

    def miss(self):
        self.frames_without_measurement += 1

    def is_tracking(self):
        return (self.initialized and
                self.frames_without_measurement < self.max_frames_without_measurement)
