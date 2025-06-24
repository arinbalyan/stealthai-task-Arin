# Player Re-Identification Task: Technical Report

## 1. Introduction

This report details the methodology and technical approach used to solve the player re-identification (Re-ID) problem in a sports video context. The primary objective was to detect all players on the field, assign a unique and persistent ID to each one, and maintain that ID even if a player is temporarily occluded or leaves the frame.

The final solution uses a combination of state-of-the-art object detection, Kalman filters for motion tracking, and a robust, two-stage matching algorithm that leverages both spatial and appearance-based features for reliable tracking and re-identification.

## 2. Core Components

The system is built on three foundational pillars:
1.  **Player Detection:** A YOLOv8 object detection model, fine-tuned on a relevant sports dataset, is used to locate players in each frame of the video.
2.  **Motion Prediction:** A Kalman filter is assigned to each tracked player to predict their position in subsequent frames. This provides an estimate of a player's location even when the detector fails for a moment.
3.  **Data Association:** The Hungarian algorithm is used to match new detections to existing tracks. The core of this system's accuracy lies in the cost matrix, which is a weighted combination of location and appearance similarity.

## 3. The Re-Identification Challenge & Solution

The most significant challenge in this task is maintaining a consistent ID when a player is lost and later reappears. A simple Intersection over Union (IoU) based matching fails in these scenarios, as the player's new location will be far from their last known position.

To address this, we implemented a sophisticated, two-stage association strategy that manages both **active** and **lost** trackers.

### Stage 1: Active Tracker Association (High-Confidence Matching)

-   **Goal:** To accurately track players who are currently visible on screen and prevent ID swaps between players in close proximity.
-   **Method:** For active trackers, we use a weighted cost matrix that combines IoU (location similarity) and a feature-based appearance score.
    -   **Cost Function:** `cost = 0.6 * IoU + 0.4 * AppearanceSimilarity`
    -   **Appearance Features:** We extract a color histogram from the player's bounding box. This serves as a simple but effective appearance signature.
    -   **Threshold:** A strict matching threshold is used here to ensure that only high-confidence matches are made, minimizing the risk of ID swaps.

### Stage 2: Lost Tracker Association (Re-Identification)

-   **Goal:** To re-identify a player who has returned to the frame after being occluded or off-screen.
-   **Method:** When a detection cannot be matched to an active tracker, it is compared against a list of "lost" trackers. In this stage, we heavily prioritize appearance over location.
    -   **Cost Function:** `cost = 0.1 * IoU + 0.9 * AppearanceSimilarity`
    -   **Logic:** By giving appearance a 90% weight, the system can correctly re-assign an ID to a returning player, even if they reappear in a completely different part of the field.
    -   **Threshold:** A more lenient matching threshold is used for re-identification to increase the chances of finding a match for a lost player.

### Tracker Lifecycle Management

-   **From Active to Lost:** If an active track is not matched with a detection for a set number of frames (`max_age`), it is moved to the `lost_trackers` list.
-   **From Lost to Active:** If a lost tracker is re-identified, it is moved back into the `trackers` list.
-   **Deletion:** If a lost tracker remains unmatched for an extended period (`max_lost_age`), it is permanently deleted to prevent outdated information from causing incorrect matches.

## 4. Implementation Details

-   **Player Detection:** The script uses the `ultralytics` library to run the fine-tuned YOLO model. We filter detections by `class_id = 2` (player) and apply a confidence threshold to discard low-quality detections.
-   **Kalman Filter:** We use the `filterpy` library to implement a constant velocity Kalman filter for each tracker. The state vector tracks the bounding box's center coordinates, area, and aspect ratio, along with their velocities.
-   **Data Association:** The `scipy.optimize.linear_sum_assignment` function provides an efficient implementation of the Hungarian algorithm for optimal matching.

## 5. Conclusion

This two-stage, dual-threshold approach provides a robust solution to the player re-identification problem. By separating the logic for active tracking and re-identification, the system achieves both high accuracy for visible players and high consistency for players who are temporarily lost, fulfilling the core requirements of the assignment. 