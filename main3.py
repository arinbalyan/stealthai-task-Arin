import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import time
import argparse
from collections import deque

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set PyTorch to use GPU if available
torch.set_grad_enabled(False)  # Disable gradient calculation for inference

class KalmanBoxTracker(object):
    """This class represents the internal state of individual tracked objects observed as bbox."""
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        bbox format: [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], 
                              [0, 1, 0, 0, 0, 1, 0], 
                              [0, 0, 1, 0, 0, 0, 1], 
                              [0, 0, 0, 1, 0, 0, 0], 
                              [0, 0, 0, 0, 1, 0, 0], 
                              [0, 0, 0, 0, 0, 1, 0], 
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], 
                              [0, 1, 0, 0, 0, 0, 0], 
                              [0, 0, 1, 0, 0, 0, 0], 
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_bbox = bbox
        self.features = deque(maxlen=10) # Store up to 10 feature vectors
        
    def update(self, bbox, feature=None):
        """
        Updates the state vector with observed bbox
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.last_bbox = bbox
        
        if feature is not None:
            self.features.append(feature)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate
        """
        return self.convert_x_to_bbox(self.kf.x)
    
    def get_features(self):
        """
        Returns the average of stored features
        """
        if not self.features:
            return None
        return np.mean(self.features, axis=0)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
        [x, y, s, r] where x,y is the center of the box, s is the scale/area, and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h > 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a bounding box in the form [x, y, s, r] and returns it in the form
        [x1, y1, x2, y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3]) if x[2] * x[3] > 0 else 0
        h = x[2] / w if w > 0 else 0
        if score is None:
            return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score]).reshape((1, 5))

class PlayerReIdentifier:
    def __init__(self, model_path, conf_threshold=0.4, match_threshold=0.5, max_age=30, min_hits=5, max_lost_age=90):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.match_threshold = match_threshold
        self.reid_match_threshold = match_threshold - 0.1
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_lost_age = max_lost_age

        self.trackers = []
        self.lost_trackers = []
        self.frame_count = 0
        
        self.color_map = {}
        self.min_player_area = 500
        
    def _extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        player_img = frame[y1:y2, x1:x2]
        if player_img.size == 0:
            return None
            
        hist_b = cv2.calcHist([player_img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([player_img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([player_img], [2], None, [256], [0, 256])
        
        feature = np.concatenate((hist_b, hist_g, hist_r)).flatten()
        cv2.normalize(feature, feature)
        return feature
    
    def _calculate_similarity(self, feature1, feature2):
        if feature1 is None or feature2 is None:
            return 0
        return np.dot(feature1, feature2)
    
    def _calculate_iou(self, bbox1, bbox2):
        if len(bbox1) > 4: bbox1 = bbox1[:4]
        if len(bbox2) > 4: bbox2 = bbox2[:4]
            
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1: return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _associate(self, detections, features, trackers, iou_weight=0.7, sim_weight=0.3, match_thresh=0.5):
        if not trackers or not detections:
            return np.empty((0, 2), dtype=int), np.arange(len(detections), dtype=int), np.arange(len(trackers), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        similarity_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._calculate_iou(det, trk.get_state()[0])

        for d, feat in enumerate(features):
            for t, trk in enumerate(trackers):
                trk_feat = trk.get_features()
                similarity_matrix[d, t] = self._calculate_similarity(feat, trk_feat)

        combined_matrix = iou_weight * iou_matrix + sim_weight * similarity_matrix
        
        row_ind, col_ind = linear_sum_assignment(-combined_matrix)
        
        matches, unmatched_detections, unmatched_trackers = [], list(range(len(detections))), list(range(len(trackers)))
        
        for r, c in zip(row_ind, col_ind):
            if combined_matrix[r, c] > match_thresh:
                matches.append([r, c])
                if r in unmatched_detections: unmatched_detections.remove(r)
                if c in unmatched_trackers: unmatched_trackers.remove(c)
        
        if len(matches) == 0:
            return np.empty((0, 2), dtype=int), np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)
        
        return np.array(matches, dtype=int), np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)

    def _get_color(self, idx):
        if idx not in self.color_map:
            h = (idx * 30) % 180
            s = 200
            v = 255
            rgb = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
            self.color_map[idx] = tuple(map(int, rgb))
        return self.color_map[idx]
    
    def process_frame(self, frame):
        self.frame_count += 1
        
        results = self.model(frame, verbose=False, device=device)[0]
        
        raw_detections = results.boxes.data.cpu().numpy()

        detections, features = [], []
        for det in raw_detections:
            x1, y1, x2, y2, conf, cls = det
            if conf > self.conf_threshold and int(cls) == 2 and (x2 - x1) * (y2 - y1) > self.min_player_area:
                feat = self._extract_features(frame, [x1, y1, x2, y2])
                if feat is not None:
                    detections.append([x1, y1, x2, y2, conf])
                    features.append(feat)
        
        detections = np.array(detections)
        features = np.array(features)

        for trk in self.trackers: trk.predict()
        for trk in self.lost_trackers: trk.predict()
        
        matched, unmatched_dets_indices, unmatched_trks_indices = self._associate(
            detections.tolist(), features.tolist(), self.trackers, iou_weight=0.6, sim_weight=0.4, match_thresh=self.match_threshold
        )

        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(detections[det_idx], features[det_idx])

        unmatched_detections = detections[unmatched_dets_indices]
        unmatched_features = features[unmatched_dets_indices]

        reid_matched, reid_unmatched_dets_indices, _ = self._associate(
            unmatched_detections.tolist(), unmatched_features.tolist(), self.lost_trackers, iou_weight=0.1, sim_weight=0.9, match_thresh=self.reid_match_threshold
        )
        
        indices_to_remove_from_lost = []
        matched_det_indices_in_unmatched = []
        for m in reid_matched:
            unmatched_det_idx, lost_trk_idx = m[0], m[1]
            original_det_idx = unmatched_dets_indices[unmatched_det_idx]
            
            tracker = self.lost_trackers[lost_trk_idx]
            tracker.update(detections[original_det_idx], features[original_det_idx])
            self.trackers.append(tracker)
            
            indices_to_remove_from_lost.append(lost_trk_idx)
            matched_det_indices_in_unmatched.append(unmatched_det_idx)
            
        if indices_to_remove_from_lost:
            self.lost_trackers = [t for i, t in enumerate(self.lost_trackers) if i not in sorted(indices_to_remove_from_lost, reverse=True)]
        
        final_unmatched_det_indices = np.delete(unmatched_dets_indices, matched_det_indices_in_unmatched)

        for i in final_unmatched_det_indices:
            trk = KalmanBoxTracker(detections[i])
            trk.update(detections[i], features[i])
            self.trackers.append(trk)
        
        next_frame_trackers = []
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                self.lost_trackers.append(trk)
            else:
                next_frame_trackers.append(trk)
        self.trackers = next_frame_trackers
        self.lost_trackers = [t for t in self.lost_trackers if t.time_since_update <= self.max_lost_age]

        output_frame = frame.copy()
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = trk.get_state()[0]
                x1, y1, x2, y2 = map(int, bbox)
                color = self._get_color(trk.id)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                text = f"Player {trk.id}"
                cv2.putText(output_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return output_frame

def main(input_path, output_path, model_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    reidentifier = PlayerReIdentifier(
        model_path=model_path,
        conf_threshold=0.5,
        match_threshold=0.6,
        max_age=30,
        min_hits=5,
        max_lost_age=120
    )
    
    frame_count = 0
    processing_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        start_time = time.time()
        output_frame = reidentifier.process_frame(frame)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        out.write(output_frame)
        
        if frame_count % int(fps) == 0:
            avg_time = np.mean(processing_times[-int(fps):]) if processing_times else 0
            print(f"Frame {frame_count}/{total_frames} | Avg FPS: {1/avg_time:.2f}" if avg_time > 0 else f"Frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    if processing_times:
        avg_time = np.mean(processing_times)
        print(f"\nProcessing complete!")
        print(f"Average processing time: {avg_time:.3f} seconds per frame ({1/avg_time:.2f} FPS)")
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Player Re-Identification in Video")
    parser.add_argument("--input", type=str, default="input/15sec_input_720p.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="output/output_video_reid.mp4", help="Path to output video")
    parser.add_argument("--model", type=str, default="model-task2/best.pt", help="Path to YOLO model")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_path = os.path.join(script_dir, args.output)
    model_path = os.path.join(script_dir, args.model)
    
    main(input_path, output_path, model_path) 