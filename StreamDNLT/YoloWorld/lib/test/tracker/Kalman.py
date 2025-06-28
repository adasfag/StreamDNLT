from filterpy.kalman import KalmanFilter
import numpy as np

class SimpleBoxKalman:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        dt = 1.0  # 时间间隔

        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0, 0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ])

        self.kf.H = np.eye(4, 8)

        self.kf.P[4:, 4:] *= 1000.
        self.kf.P[:4, :4] *= 10.
        self.kf.Q *= 0.01
        self.base_R = np.eye(4) * 1.0

        self.initialized = False

        self.bbox_history = []  # 存储历史 x, y, w, h
        self.bbox_history_len = 10
        self.bbox_ratio_thresh = 0.6

    def initialize(self, bbox):
        self.kf.x[:4] = np.array(bbox).reshape((4, 1))
        self.kf.x[4:] = 0
        self.initialized = True
        self.bbox_history = [bbox]

    def update(self, bbox, confidence=1.0):
        confidence = np.clip(confidence, 1e-3, 1.0)
        scale = 1.0 / confidence
        self.kf.R = self.base_R * scale



        self.bbox_history.append(bbox)
        if len(self.bbox_history) > self.bbox_history_len:
            self.bbox_history.pop(0)

        # if self.is_occluded(bbox):
        #     avg_bbox = np.mean(self.bbox_history, axis=0)
        #     print("[Info] update Occlusion suspected: suppressing bbox fluctuation.")
        #     bbox = avg_bbox

        self.kf.update(np.array(bbox).reshape((4, 1)))

    def predict(self, W=None, H=None):
        self.kf.predict()
        pred = self.kf.x[:4].reshape(-1)

        if W is not None and H is not None:
            x, y, w, h = pred
            x = np.clip(x, 0, W)
            y = np.clip(y, 0, H)
            w = np.clip(w, 0, W)
            h = np.clip(h, 0, H)

            self.kf.x[0, 0] = x
            self.kf.x[1, 0] = y
            self.kf.x[2, 0] = w
            self.kf.x[3, 0] = h


            avg_bbox = np.mean(self.bbox_history, axis=0)
            if len(self.bbox_history)< self.bbox_history_len:

                avg_bbox = np.mean(self.bbox_history, axis=0)
                return avg_bbox
            
            return np.array([x,y,avg_bbox[1],avg_bbox[2],avg_bbox[3]])



        return self.kf.x[:4].reshape(-1)

    def current_state(self):
        return self.kf.x[:4].reshape(-1)

    def is_occluded(self, pred_bbox):
        if len(self.bbox_history) < self.bbox_history_len:
            return False

        avg_bbox = np.mean(self.bbox_history, axis=0)

        iou = self.compute_iou(pred_bbox, avg_bbox)
        w_ratio = abs(pred_bbox[2] - avg_bbox[2]) / (avg_bbox[2] + 1e-5)
        h_ratio = abs(pred_bbox[3] - avg_bbox[3]) / (avg_bbox[3] + 1e-5)

        if iou < 0.1 or  w_ratio > self.bbox_ratio_thresh or h_ratio > self.bbox_ratio_thresh:
            return True
        return False
    @staticmethod
    def compute_iou(boxA, boxB):
        """
        boxA, boxB: [x, y, w, h]
        返回IoU（Intersection over Union）
        """
        xA1, yA1, wA, hA = boxA
        xB1, yB1, wB, hB = boxB

        xA2, yA2 = xA1 + wA, yA1 + hA
        xB2, yB2 = xB1 + wB, yB1 + hB

        inter_x1 = max(xA1, xB1)
        inter_y1 = max(yA1, yB1)
        inter_x2 = min(xA2, xB2)
        inter_y2 = min(yA2, yB2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        areaA = wA * hA
        areaB = wB * hB

        union_area = areaA + areaB - inter_area

        return inter_area / (union_area + 1e-6)
