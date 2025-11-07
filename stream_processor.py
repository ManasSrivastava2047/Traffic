import threading
import time
import uuid
from collections import deque
import os
import subprocess
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import torch


class StreamSession:

    def __init__(self, source: str, model_coco: YOLO, vehicle_classes: Optional[set] = None,
                 target_fps: float = 5.0, aggregation_seconds: int = 1, slope_window_seconds: int = 12) -> None:
        self.session_id: str = uuid.uuid4().hex
        self.source: str = source
        self.model_coco: YOLO = model_coco
        self.vehicle_classes: set = vehicle_classes or {2, 3, 5, 7}
        self.target_fps: float = target_fps
        self.aggregation_seconds: int = aggregation_seconds
        self.slope_window_seconds: int = slope_window_seconds

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Time-series of (timestamp_seconds, vehicles_per_second)
        self._per_second_counts: Deque[Tuple[float, float]] = deque(maxlen=600)  # keep last 10 minutes

        # Last computed metrics cache
        self._latest_frame_vehicle_count: int = 0
        self._ema_cps: Optional[float] = None  # Exponential moving average of counts per second

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"stream-{self.session_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return

        target_interval = 1.0 / max(1e-6, self.target_fps)
        last_emit_second = None
        frame_counts_accumulator = 0
        frames_in_bucket = 0
        last_tick = time.time()

        try:
            while not self._stop_event.is_set():
                start_loop = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                # Optional resize for speed
                h, w = frame.shape[:2]
                scale = 640.0 / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                # Run YOLO detection (smaller imgsz, class filter, explicit device)
                results = self.model_coco.predict(
                    frame,
                    imgsz=512,
                    conf=0.3,
                    classes=list(self.vehicle_classes),
                    device=("cuda" if torch.cuda.is_available() else "cpu"),
                    verbose=False,
                )[0]
                current_vehicle_count = int(sum(1 for cls in results.boxes.cls if int(cls) in self.vehicle_classes))

                # Accumulate per-second bucket
                now = time.time()
                second = int(now)
                frame_counts_accumulator += current_vehicle_count
                frames_in_bucket += 1

                # Store last frame count for quick read
                with self._lock:
                    self._latest_frame_vehicle_count = current_vehicle_count

                # At each new second, compute avg vehicles per frame in the last second,
                # convert to vehicles per second by multiplying by fps observed.
                if last_emit_second is None:
                    last_emit_second = second

                if second != last_emit_second:
                    elapsed = max(1, second - last_emit_second)
                    observed_fps = frames_in_bucket / max(1e-6, (now - last_tick))
                    avg_per_frame = (frame_counts_accumulator / max(1, frames_in_bucket))
                    vehicles_per_second = avg_per_frame * observed_fps

                    # EMA smoothing for cps
                    alpha = 0.3
                    if self._ema_cps is None:
                        self._ema_cps = vehicles_per_second
                    else:
                        self._ema_cps = alpha * vehicles_per_second + (1 - alpha) * self._ema_cps

                    with self._lock:
                        self._per_second_counts.append((float(second), float(self._ema_cps)))

                    # reset bucket
                    last_emit_second = second
                    frame_counts_accumulator = 0
                    frames_in_bucket = 0
                    last_tick = now

                # pacing
                elapsed_loop = time.time() - start_loop
                sleep_for = target_interval - elapsed_loop
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.02))
        finally:
            cap.release()

    def get_metrics(self) -> Dict[str, float]:
        with self._lock:
            latest_count = self._latest_frame_vehicle_count
            ema_cps = float(self._ema_cps) if self._ema_cps is not None else 0.0
            slope = self._compute_slope_locked()
            window_points = len(self._per_second_counts)

        return {
            "sessionId": self.session_id,
            "currentFrameVehicleCount": float(latest_count),
            "vehiclesPerSecond": ema_cps,
            # rateOfChange is the slope of cps over time (vehicles/sec^2)
            "rateOfChange": slope,
            "dataPoints": float(window_points),
        }

    def _compute_slope_locked(self) -> float:
        if not self._per_second_counts:
            return 0.0

        # Use only recent window
        t_now = self._per_second_counts[-1][0]
        window_start = t_now - float(self.slope_window_seconds)
        xs: List[float] = []
        ys: List[float] = []
        for t, cps in self._per_second_counts:
            if t >= window_start:
                xs.append(t)
                ys.append(cps)

        if len(xs) < 2:
            return 0.0

        # Linear regression slope: slope = cov(x,y) / var(x)
        x = np.array(xs, dtype=np.float64)
        y = np.array(ys, dtype=np.float64)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_dev = x - x_mean
        var_x = float(np.sum(x_dev * x_dev))
        if var_x <= 1e-9:
            return 0.0
        cov_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        slope = cov_xy / var_x
        return float(slope)


class StreamProcessor:

    def __init__(self, model_coco: YOLO) -> None:
        self.model_coco = model_coco
        self._sessions: Dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def start_session(self, source: str) -> str:
        session = StreamSession(source=source, model_coco=self.model_coco)
        session.start()
        with self._lock:
            self._sessions[session.session_id] = session
        return session.session_id

    def stop_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.stop()
        return True

    def get_metrics(self, session_id: str) -> Optional[Dict[str, float]]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.get_metrics()


def analyze_video_file(model_coco: YOLO, source_path: str, vehicle_classes: Optional[set] = None,
                       target_fps: float = 4.0, slope_window_seconds: int = 10,
                       annotated_output_path: Optional[str] = None,
                       max_seconds: Optional[int] = 8) -> Dict[str, float]:
    """
    Offline analysis for an uploaded video file. Returns summary metrics including
    vehiclesPerSecond (EMA) and rateOfChange (slope of cps over time).
    """
    vehicle_classes = vehicle_classes or {2, 3, 5, 7}
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return {"vehiclesPerSecond": 0.0, "rateOfChange": 0.0, "dataPoints": 0.0}

    per_second_counts: Deque[Tuple[float, float]] = deque()
    ema_cps: Optional[float] = None

    # Use deterministic time base from the video FPS; fallback to target_fps if unavailable
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 1e-3:
        video_fps = max(1.0, float(target_fps))

    frame_index = 0
    current_second_index: Optional[int] = None
    frame_counts_accumulator = 0
    frames_in_bucket = 0

    writer: Optional[cv2.VideoWriter] = None
    frames_written = 0
    writer_size: Optional[Tuple[int, int]] = None
    selected_output_path: Optional[str] = None
    # Determine stride for subsampling (process ~target_fps)
    stride = max(1, int(round(float(video_fps) / max(1.0, float(target_fps)))))

    # --- Simple IoU-based multi-frame tracker to count unique vehicles once ---
    next_track_id: int = 1
    # track: {"id": int, "bbox": (x1,y1,x2,y2), "age": int, "last_second": int}
    active_tracks: List[Dict[str, object]] = []
    counted_track_ids: set = set()

    def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(float(boxA[0]), float(boxB[0]))
        yA = max(float(boxA[1]), float(boxB[1]))
        xB = min(float(boxA[2]), float(boxB[2]))
        yB = min(float(boxA[3]), float(boxB[3]))
        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH
        if interArea <= 0.0:
            return 0.0
        boxAArea = max(0.0, float(boxA[2] - boxA[0])) * max(0.0, float(boxA[3] - boxA[1]))
        boxBArea = max(0.0, float(boxB[2] - boxB[0])) * max(0.0, float(boxB[3] - boxB[1]))
        denom = boxAArea + boxBArea - interArea
        return float(interArea / denom) if denom > 0 else 0.0

    # Require a track to be seen in at least this many processed frames before counting
    min_confirm_age: int = 2
    # If a track is not matched for this many seconds, drop it
    max_unseen_seconds: int = 3

    total_unique_vehicles: int = 0

    try:
        processed_seconds = 0
        while True:
            # Read one frame to process
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # Optional resize for speed
            h, w = frame.shape[:2]
            scale = 640.0 / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # This is a processed frame (we read exactly one per stride)
            process_this_frame = True

            detected_count = 0
            if process_this_frame:
                # Use Ultralytics built-in tracker for stable IDs to avoid double counting
                results = model_coco.track(
                    frame,
                    imgsz=512,
                    conf=0.35,
                    classes=list(vehicle_classes),
                    device=("cuda" if torch.cuda.is_available() else "cpu"),
                    persist=True,
                    verbose=False,
                )[0]
                # Filter by vehicle classes and count per-frame detections
                if results.boxes is not None and len(results.boxes) > 0:
                    classes_arr = results.boxes.cls
                    ids_arr = getattr(results.boxes, 'id', None)
                    xyxy = results.boxes.xyxy
                    for idx in range(len(xyxy)):
                        if int(classes_arr[idx]) not in vehicle_classes:
                            continue
                        detected_count += 1
                        # Update age per track id, count once after confirmation age
                        if ids_arr is not None and len(ids_arr) > idx and ids_arr[idx] is not None:
                            tid = int(ids_arr[idx])
                            # Age bookkeeping in a dict on the fly
                            # We reuse active_tracks as a small map-like store
                            found = False
                            for tr in active_tracks:
                                if int(tr["id"]) == tid:  # type: ignore[index]
                                    tr["bbox"] = (float(xyxy[idx][0]), float(xyxy[idx][1]), float(xyxy[idx][2]), float(xyxy[idx][3]))  # type: ignore[index]
                                    tr["age"] = int(tr.get("age", 0)) + 1  # type: ignore[index]
                                    tr["last_second"] = int(current_second_index if current_second_index is not None else 0)  # type: ignore[index]
                                    found = True
                                    break
                            if not found:
                                active_tracks.append({
                                    "id": tid,
                                    "bbox": (float(xyxy[idx][0]), float(xyxy[idx][1]), float(xyxy[idx][2]), float(xyxy[idx][3])),
                                    "age": 1,
                                    "last_second": int(current_second_index if current_second_index is not None else 0)
                                })

                    # Count newly confirmed tracks exactly once (IDs are stable from tracker)
                    for tr in active_tracks:
                        tr_id = int(tr["id"])  # type: ignore[index]
                        tr_age = int(tr.get("age", 0))  # type: ignore[arg-type]
                        if tr_age >= 2 and tr_id not in counted_track_ids:
                            counted_track_ids.add(tr_id)
                            total_unique_vehicles += 1

                    # Drop stale tracks that have not been seen for a while
                    now_sec = int(current_second_index if current_second_index is not None else 0)
                    pruned: List[Dict[str, object]] = []
                    for tr in active_tracks:
                        last_sec = int(tr.get("last_second", now_sec))  # type: ignore[arg-type]
                        if (now_sec - last_sec) <= 3:
                            pruned.append(tr)
                    active_tracks = pruned

            # Create/write annotated frame if requested
            if annotated_output_path is not None and process_this_frame:
                annotated_frame = results.plot()
                # Overlay track IDs on annotated frame when available
                try:
                    if results.boxes is not None and len(results.boxes) > 0:
                        ids_arr = getattr(results.boxes, 'id', None)
                        xyxy = results.boxes.xyxy
                        for idx in range(len(xyxy)):
                            if ids_arr is None or len(ids_arr) <= idx or ids_arr[idx] is None:
                                continue
                            tid = int(ids_arr[idx])
                            x1, y1, x2, y2 = [int(v) for v in xyxy[idx]]
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(annotated_frame, f"ID {tid}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                except Exception:
                    pass
                if writer is None:
                    h_out, w_out = annotated_frame.shape[:2]
                    # Ensure even dimensions for codec compatibility
                    if w_out % 2 != 0 or h_out % 2 != 0:
                        w_out_even = w_out - (w_out % 2)
                        h_out_even = h_out - (h_out % 2)
                        annotated_frame = cv2.resize(annotated_frame, (w_out_even, h_out_even))
                        h_out, w_out = annotated_frame.shape[:2]

                    # Write to a robust MJPG AVI first; convert later if needed
                    base_no_ext = annotated_output_path.rsplit('.', 1)[0]
                    out_avi = f"{base_no_ext}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    w = cv2.VideoWriter(out_avi, fourcc, float(video_fps), (w_out, h_out))
                    if w is not None and w.isOpened():
                        writer = w
                        selected_output_path = out_avi
                        writer_size = (w_out, h_out)
                    else:
                        writer = None
                if writer is not None:
                    # Ensure each frame matches the writer's size
                    if writer_size is not None and (annotated_frame.shape[1], annotated_frame.shape[0]) != writer_size:
                        annotated_frame = cv2.resize(annotated_frame, writer_size)
                    writer.write(annotated_frame)
                    frames_written += 1

            # Deterministic second index based on frame index and video fps
            second_index = int(frame_index / video_fps)
            frame_index += stride

            if current_second_index is None:
                current_second_index = second_index

            # Accumulate counts only for processed frames (unbiased)
            if process_this_frame:
                frame_counts_accumulator += detected_count
                frames_in_bucket += 1

            # If we moved to a new second, finalize the previous bucket deterministically
            if second_index != current_second_index:
                avg_per_frame = frame_counts_accumulator / max(1, frames_in_bucket)
                effective_fps = float(video_fps) / float(stride)
                vehicles_per_second = avg_per_frame * effective_fps

                # EMA smoothing for cps (deterministic)
                alpha = 0.3
                if ema_cps is None:
                    ema_cps = vehicles_per_second
                else:
                    ema_cps = alpha * vehicles_per_second + (1 - alpha) * ema_cps

                per_second_counts.append((float(current_second_index), float(ema_cps)))

                # reset for next second bucket
                processed_seconds += 1
                current_second_index = second_index
                frame_counts_accumulator = 0
                frames_in_bucket = 0

                # Early exit to reduce latency on long videos
                if max_seconds is not None and processed_seconds >= int(max_seconds):
                    break

            # Skip the next stride-1 frames quickly without decoding to keep speed high
            for _ in range(stride - 1):
                grabbed = cap.grab()
                frame_index += 1  # advance timestamp for skipped frames
                if not grabbed:
                    break
            if max_seconds is not None and processed_seconds >= int(max_seconds):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

        # If we have an AVI and the desired path is MP4, try to convert using ffmpeg
        if selected_output_path and selected_output_path.lower().endswith('.avi') and annotated_output_path:
            target_mp4 = annotated_output_path if annotated_output_path.lower().endswith('.mp4') else (selected_output_path[:-4] + '.mp4')
            ffmpeg_bins: List[str] = []
            env_bin = os.environ.get('FFMPEG_BIN')
            if env_bin:
                ffmpeg_bins.append(env_bin)
            ffmpeg_bins.append('ffmpeg')
            ffmpeg_bins.append(r"C:\\ffmpeg-master-latest-win64-gpl-shared\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe")

            for ff in ffmpeg_bins:
                try:
                    subprocess.run([ff, '-y', '-i', selected_output_path, '-vcodec', 'libx264', '-crf', '18', target_mp4], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(target_mp4) and os.path.getsize(target_mp4) > 0:
                        try:
                            os.remove(selected_output_path)
                        except Exception:
                            pass
                        selected_output_path = target_mp4
                        break
                except Exception:
                    continue

    # Compute slope over recent window
    if not per_second_counts:
        return {"vehiclesPerSecond": float(ema_cps or 0.0), "rateOfChange": 0.0, "dataPoints": 0.0}

    t_now = per_second_counts[-1][0]
    window_start = t_now - float(slope_window_seconds)
    xs: List[float] = []
    ys: List[float] = []
    for t, cps in per_second_counts:
        if t >= window_start:
            xs.append(t)
            ys.append(cps)

    if len(xs) < 2:
        slope = 0.0
    else:
        x = np.array(xs, dtype=np.float64)
        y = np.array(ys, dtype=np.float64)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_dev = x - x_mean
        var_x = float(np.sum(x_dev * x_dev))
        slope = 0.0 if var_x <= 1e-9 else float(np.sum((x - x_mean) * (y - y_mean)) / var_x)

    return {
        "vehiclesPerSecond": float(ema_cps or 0.0),
        "rateOfChange": float(slope),
        "dataPoints": float(len(per_second_counts)),
        "annotatedFrames": int(frames_written),
        "annotatedOutputPath": selected_output_path or "",
        "vehicleCount": int(total_unique_vehicles),
    }


