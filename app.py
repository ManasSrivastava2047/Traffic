from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS
import os
import uuid
from ultralytics import YOLO
import torch
from PIL import Image
import cv2
from typing import Optional
from datetime import datetime
import datetime as _dt

from stream_processor import StreamProcessor, analyze_video_file
from db import init_db, insert_result
from db import fetch_latest_result

app = Flask(__name__)
CORS(app)

# Initialize DB (creates database + table if missing)
try:
    init_db()
except Exception as e:
    print(f"[app] DB init error: {e}")

# Simple in-memory metadata store for region/intersection set by the UI's "Confirm Intersection" action.
# This is a minimal solution for a single-user/local setup. For multi-user use a proper persistent store
# or associate metadata with a session id provided by the client.
metadata_store = {
    'region_name': '',
    'intersection_id': '',
}

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8 model (use yolov8n.pt or your trained weights)
model_coco = YOLO("yolov8n.pt")        # Pretrained COCO model
model_custom = YOLO("best.pt")         # Your custom retrained model # You can replace with 'yolov8m.pt', 'yolov8s.pt' or custom weights

# Optimize models for available hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model_coco.to(DEVICE)
    model_custom.to(DEVICE)
    # Fuse layers for faster inference where supported
    try:
        model_coco.fuse()
    except Exception:
        pass
    try:
        model_custom.fuse()
    except Exception:
        pass
    # Enable half precision on CUDA for speed
    if DEVICE == "cuda":
        try:
            model_coco.model.half()
        except Exception:
            pass
        try:
            model_custom.model.half()
        except Exception:
            pass
except Exception:
    DEVICE = "cpu"

# Determine emergency class ids from custom model names (prefer labels containing 'ambulance')
def _infer_emergency_class_ids(m) -> set:
    try:
        names = getattr(m, 'names', None)
        if isinstance(names, dict):
            ids = {int(i) for i, n in names.items() if 'ambulance' in str(n).lower()}
            if ids:
                return ids
            # fallback: if single-class model, assume that class 0 is ambulance
            if len(names) == 1:
                return {0}
        # fallback default
        return {0}
    except Exception:
        return {0}

EMERGENCY_CLASS_IDS = _infer_emergency_class_ids(model_custom)

# Initialize stream processor for live metrics
stream_processor = StreamProcessor(model_coco)

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run YOLOv8 detection (smaller imgsz, filtered classes, proper device)
    vehicle_classes = {2, 3, 5, 7}
    results_coco = model_coco.predict(
        filepath,
        imgsz=512,
        conf=0.3,
        classes=list(vehicle_classes),
        device=DEVICE,
        verbose=False,
    )[0]
    results_custom = model_custom.predict(
        filepath,
        imgsz=512,
        conf=0.6,
        device=DEVICE,
        verbose=False,
    )[0]

    vehicle_count = sum(1 for cls in results_coco.boxes.cls if int(cls) in vehicle_classes)

    emergency_classes = EMERGENCY_CLASS_IDS
    emergency_detected = any(int(cls) in emergency_classes for cls in results_custom.boxes.cls)

    frame = cv2.imread(filepath)
    annotated_frame = results_coco.plot()
    
    for box in results_custom.boxes:
        cls_id = int(box.cls[0])
        if cls_id in emergency_classes:  # only for custom emergency classes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Label from model names when available
            try:
                name = model_custom.names.get(cls_id, 'ambulance') if isinstance(model_custom.names, dict) else 'ambulance'
            except Exception:
                name = 'ambulance'
            label = str(name)
            conf = float(box.conf[0])
            text = f"{label} {conf:.2f}"

            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)


            cv2.rectangle(
            annotated_frame,
            (x1, y1 - th - baseline - 3),
            (x1 + tw, y1),
            (0, 0, 255),
            -1  # filled
            )
            cv2.putText(
            annotated_frame,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # white text
            2
            )

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    result_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(result_path, annotated_frame)

    signal_time = min(max(10 + (vehicle_count * 2), 10),65)
    if emergency_detected:
        signal_time += 10  # extend green time for emergency vehicles

    # Store result in DB (use defaults if region/intersection not provided)
    try:
        region_name = (
            request.form.get('region_name')
            or request.form.get('Region_Name')
            or request.values.get('region_name')
            or request.values.get('Region_Name')
            or metadata_store.get('region_name', '')
            or ''
        )
        intersection_id = (
            request.form.get('intersection_id')
            or request.form.get('Intersection_ID')
            or request.values.get('intersection_id')
            or request.values.get('Intersection_ID')
            or metadata_store.get('intersection_id', '')
            or ''
        )
        now = datetime.now()
        insert_result(
            region_name,
            intersection_id,
            now.date(),
            now.time(),
            int(vehicle_count),
            int(signal_time),
        )
    except Exception as _e:
        print(f"[app] Failed to insert detect result: {_e}")

    return jsonify({
        'vehicleCount': vehicle_count,
        'emergencyDetected': emergency_detected,
        'signalTime': signal_time,
        'detectedImage': f'/results/{filename}'
    })

@app.route('/results/<path:filename>')
def serve_result(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    range_header = request.headers.get('Range', None)
    if not range_header:
        # Fallback full send
        return send_file(path)

    # Support HTTP Range requests for proper video streaming
    size = os.path.getsize(path)
    byte1, byte2 = 0, None
    try:
        # Example: Range: bytes=0-1023
        match = range_header.strip().lower().split('=')[-1]
        if '-' in match:
            start_str, end_str = match.split('-')
            if start_str.strip():
                byte1 = int(start_str)
            if end_str.strip():
                byte2 = int(end_str)
    except Exception:
        byte1, byte2 = 0, None

    if byte2 is None or byte2 >= size:
        byte2 = size - 1
    length = byte2 - byte1 + 1

    with open(path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv


@app.route('/api/stream/start', methods=['POST'])
def stream_start():
    data = request.get_json(silent=True) or {}
    source = data.get('source')
    if source is None:
        return jsonify({"error": "Missing 'source' (e.g., 0 for webcam or RTSP/HTTP URL)"}), 400
    session_id = stream_processor.start_session(str(source))
    return jsonify({"sessionId": session_id})


@app.route('/api/stream/metrics', methods=['GET'])
def stream_metrics():
    session_id = request.args.get('sessionId')
    if not session_id:
        return jsonify({"error": "Missing 'sessionId'"}), 400
    metrics = stream_processor.get_metrics(session_id)
    if metrics is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(metrics)


@app.route('/api/stream/stop', methods=['POST'])
def stream_stop():
    data = request.get_json(silent=True) or {}
    session_id = data.get('sessionId')
    if not session_id:
        return jsonify({"error": "Missing 'sessionId'"}), 400
    ok = stream_processor.stop_session(session_id)
    if not ok:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"stopped": True})


@app.route('/api/metadata', methods=['GET', 'POST'])
def set_metadata():
    """Set or get the current region/intersection metadata.

    POST body or form: region_name, intersection_id
    GET returns the current stored values.
    """
    if request.method == 'GET':
        return jsonify(metadata_store)

    # POST: accept JSON or form
    data = request.get_json(silent=True) or request.form or request.values
    region = data.get('region_name') or data.get('Region_Name') or ''
    intersection = data.get('intersection_id') or data.get('Intersection_ID') or ''
    metadata_store['region_name'] = region
    metadata_store['intersection_id'] = intersection
    return jsonify({'ok': True, 'region_name': region, 'intersection_id': intersection})


@app.route('/api/video/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in request'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = f"{uuid.uuid4().hex}.mp4"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Prepare annotated output path
    base_annotated = f"{os.path.splitext(filename)[0]}_annotated.mp4"
    annotated_path = os.path.join(RESULT_FOLDER, base_annotated)

    metrics = analyze_video_file(
        model_coco,
        filepath,
        annotated_output_path=annotated_path,
        target_fps=4.0,
        max_seconds=8,
        model_emergency=model_custom,
        emergency_class_ids=EMERGENCY_CLASS_IDS,
        emergency_confidence=0.6,
    )
    # Only return annotated video URL if file exists, has size, and frames were written
    try:
        path_from_writer = metrics.get("annotatedOutputPath") or annotated_path
        if (
            os.path.exists(path_from_writer)
            and os.path.getsize(path_from_writer) > 0
            and int(metrics.get("annotatedFrames", 0)) > 0
        ):
            # Map absolute path to /results route
            metrics["annotatedVideo"] = f"/results/{os.path.basename(path_from_writer)}"
    except Exception:
        pass
    # Compute signal time: 33 + 10*rate + 2*(vehicleCount-10) + time_effect, clamped 15-65
    try:
        rate = float(metrics.get('rateOfChange', 0.0))
    except Exception:
        rate = 0.0
    try:
        vcount = int(metrics.get('vehicleCount', 0))
    except Exception:
        vcount = 0
    hour = datetime.now().hour
    if (8 <= hour <= 11) or (18 <= hour <= 22):
        time_effect = 3
    elif (hour >= 22) or (hour <= 7):
        time_effect = -3
    else:
        time_effect = 0
    raw_time = 33 + (10 * rate) + 2 * (vcount - 10) + time_effect
    try:
        st = int(round(raw_time))
    except Exception:
        st = 33
    metrics['signalTime'] = max(15, min(65, st))
    # Store a DB record for this analyzed video (use defaults when metadata not provided)
    try:
        region_name = (
            request.form.get('region_name')
            or request.form.get('Region_Name')
            or request.values.get('region_name')
            or request.values.get('Region_Name')
            or metadata_store.get('region_name', '')
            or ''
        )
        intersection_id = (
            request.form.get('intersection_id')
            or request.form.get('Intersection_ID')
            or request.values.get('intersection_id')
            or request.values.get('Intersection_ID')
            or metadata_store.get('intersection_id', '')
            or ''
        )
        now = datetime.now()
        insert_result(
            region_name,
            intersection_id,
            now.date(),
            now.time(),
            int(metrics.get('vehicleCount', 0)),
            int(metrics.get('signalTime', 0)),
        )
    except Exception as _e:
        print(f"[app] Failed to insert analyze_video result: {_e}")

    return jsonify(metrics)


@app.route('/api/video/analyze-multi', methods=['POST'], endpoint='analyze_video_multi_v1')
def analyze_video_multi_v1():
    # Accept up to 4 files with compass keys; fall back to lane1..lane4 for compatibility
    compass_keys = [('north', 1), ('south', 2), ('east', 3), ('west', 4)]
    legacy_keys = ['lane1', 'lane2', 'lane3', 'lane4']

    provided: list = []  # items: (direction: str, src_path: str, annotated_path: str)

    # Prefer compass keys if present
    any_compass = any(request.files.get(k) for k, _ in compass_keys)
    if any_compass:
        for dir_key, idx in compass_keys:
            file = request.files.get(dir_key)
            if file and file.filename:
                filename = f"{uuid.uuid4().hex}_{dir_key}.mp4"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                base_annotated = f"{os.path.splitext(filename)[0]}_annotated.mp4"
                annotated_path = os.path.join(RESULT_FOLDER, base_annotated)
                provided.append((dir_key, filepath, annotated_path))
    else:
        # Fallback legacy mapping lane1..4 -> north,south,east,west
        map_legacy = {0: 'north', 1: 'south', 2: 'east', 3: 'west'}
        for i, key in enumerate(legacy_keys):
            file = request.files.get(key)
            if file and file.filename:
                dir_key = map_legacy.get(i, f'lane{i+1}')
                filename = f"{uuid.uuid4().hex}_{dir_key}.mp4"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                base_annotated = f"{os.path.splitext(filename)[0]}_annotated.mp4"
                annotated_path = os.path.join(RESULT_FOLDER, base_annotated)
                provided.append((dir_key, filepath, annotated_path))

    if not provided:
        return jsonify({'error': 'No videos provided. Send files as north/south/east/west'}), 400

    lanes_out = []
    for direction, src_path, ann_path in provided:
        metrics = analyze_video_file(
            model_coco,
            src_path,
            annotated_output_path=ann_path,
            model_emergency=model_custom,
            emergency_class_ids=EMERGENCY_CLASS_IDS,
            emergency_confidence=0.6,
        )
        try:
            path_from_writer = metrics.get("annotatedOutputPath") or ann_path
            if (
                os.path.exists(path_from_writer)
                and os.path.getsize(path_from_writer) > 0
                and int(metrics.get("annotatedFrames", 0)) > 0
            ):
                metrics["annotatedVideo"] = f"/results/{os.path.basename(path_from_writer)}"
        except Exception:
            pass

        # Compute per-direction green time using new formula
        try:
            rate = float(metrics.get('rateOfChange', 0.0))
        except Exception:
            rate = 0.0
        try:
            vcount = int(metrics.get('vehicleCount', 0))
        except Exception:
            vcount = 0
        hour = datetime.now().hour
        if (8 <= hour <= 11) or (18 <= hour <= 22):
            time_effect = 3
        elif (hour >= 22) or (hour <= 7):
            time_effect = -3
        else:
            time_effect = 0
        raw_time = 33 + (10 * rate) + 2 * (vcount - 10) + time_effect
        try:
            signal_time = int(round(raw_time))
        except Exception:
            signal_time = 33
        try:
            emergency_count = int(metrics.get('emergencyCount', 0))
        except Exception:
            emergency_count = 0
        emergency_detected = emergency_count > 0 or bool(metrics.get('emergencyDetected'))
        # Add +3s if ambulance detected in this direction
        if emergency_detected:
            signal_time += 3
        signal_time = max(15, min(65, signal_time))

        lanes_out.append({
            'direction': direction,
            'vehiclesPerSecond': metrics.get('vehiclesPerSecond', 0.0),
            'rateOfChange': metrics.get('rateOfChange', 0.0),
            'vehicleCount': metrics.get('vehicleCount', 0),
            'signalTime': signal_time,
            'annotatedVideo': metrics.get('annotatedVideo'),
            'emergencyDetected': bool(emergency_detected),
            'emergencyCount': int(emergency_count),
        })

    # Enforce same timing within pairs: NS and EW use the greater of the pair
    by_dir = {item['direction']: item for item in lanes_out}
    ns = [by_dir.get('north'), by_dir.get('south')]
    ew = [by_dir.get('east'), by_dir.get('west')]
    def max_pair(pair):
        vals = [p['signalTime'] for p in pair if p]
        return max(vals) if vals else None
    ns_max = max_pair(ns)
    ew_max = max_pair(ew)
    if ns_max is not None:
        for p in ns:
            if p is not None:
                p['signalTime'] = ns_max
    if ew_max is not None:
        for p in ew:
            if p is not None:
                p['signalTime'] = ew_max

    # Sort in compass order
    order = {'north': 0, 'west': 1, 'east': 2, 'south': 3}
    lanes_out.sort(key=lambda x: order.get(x.get('direction', ''), 99))

    # Insert a single aggregated record for this Analyze Lanes action
    try:
        region_name = (
            request.form.get('region_name')
            or request.form.get('Region_Name')
            or request.values.get('region_name')
            or request.values.get('Region_Name')
            or metadata_store.get('region_name', '')
            or ''
        )
        intersection_id = (
            request.form.get('intersection_id')
            or request.form.get('Intersection_ID')
            or request.values.get('intersection_id')
            or request.values.get('Intersection_ID')
            or metadata_store.get('intersection_id', '')
            or ''
        )
        now = datetime.now()
        # compute maxima across directions
        max_vehicle = 0
        max_green = 0
        for lane in lanes_out:
            try:
                vc = int(lane.get('vehicleCount', 0))
            except Exception:
                vc = 0
            try:
                gt = int(lane.get('signalTime', 0))
            except Exception:
                gt = 0
            if vc > max_vehicle:
                max_vehicle = vc
            if gt > max_green:
                max_green = gt

        # Insert single aggregated row
        insert_result(
            region_name,
            intersection_id,
            now.date(),
            now.time(),
            max_vehicle,
            max_green,
        )
    except Exception as _e:
        print(f"[app] Failed to insert analyze_video_multi_v1 aggregated result: {_e}")

    return jsonify({ 'lanes': lanes_out })


@app.route('/api/video/analyze-multi', methods=['POST'])
def analyze_video_multi():
    # Expect up to 4 files: lane1, lane2, lane3, lane4
    lanes = []
    for idx in range(1, 5):
        key = f'lane{idx}'
        if key in request.files and request.files[key].filename != '':
            file = request.files[key]
            filename = f"{uuid.uuid4().hex}.mp4"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            base_annotated = f"{os.path.splitext(filename)[0]}_annotated.mp4"
            annotated_path = os.path.join(RESULT_FOLDER, base_annotated)

            metrics = analyze_video_file(
                model_coco,
                filepath,
                annotated_output_path=annotated_path,
                model_emergency=model_custom,
                emergency_class_ids=EMERGENCY_CLASS_IDS,
                emergency_confidence=0.6,
            )
            try:
                path_from_writer = metrics.get("annotatedOutputPath") or annotated_path
                if (
                    os.path.exists(path_from_writer)
                    and os.path.getsize(path_from_writer) > 0
                    and int(metrics.get("annotatedFrames", 0)) > 0
                ):
                    metrics["annotatedVideo"] = f"/results/{os.path.basename(path_from_writer)}"
            except Exception:
                pass

            try:
                emergency_count = int(metrics.get("emergencyCount", 0))
            except Exception:
                emergency_count = 0
            emergency_detected = emergency_count > 0 or bool(metrics.get("emergencyDetected"))

            lanes.append({
                "lane": idx,
                "vehiclesPerSecond": metrics.get("vehiclesPerSecond", 0.0),
                "rateOfChange": metrics.get("rateOfChange", 0.0),
                "vehicleCount": metrics.get("vehicleCount", 0),
                "annotatedVideo": metrics.get("annotatedVideo", None),
                "emergencyDetected": bool(emergency_detected),
                "emergencyCount": int(emergency_count),
            })
        else:
            lanes.append({
                "lane": idx,
                "vehiclesPerSecond": 0.0,
                "rateOfChange": 0.0,
                "vehicleCount": 0,
                "annotatedVideo": None,
                "emergencyDetected": False,
                "emergencyCount": 0,
            })


    # Insert a single aggregated record for this Analyze Lanes (legacy lane1..lane4)
    try:
        region_name = (
            request.form.get('region_name')
            or request.form.get('Region_Name')
            or request.values.get('region_name')
            or request.values.get('Region_Name')
            or metadata_store.get('region_name', '')
            or ''
        )
        intersection_id = (
            request.form.get('intersection_id')
            or request.form.get('Intersection_ID')
            or request.values.get('intersection_id')
            or request.values.get('Intersection_ID')
            or metadata_store.get('intersection_id', '')
            or ''
        )
        now = datetime.now()

        max_vehicle = 0
        max_green = 0
        hour = datetime.now().hour
        for lane in lanes:
            try:
                vcount = int(lane.get('vehicleCount', 0))
            except Exception:
                vcount = 0
            try:
                rate = float(lane.get('rateOfChange', 0.0))
            except Exception:
                rate = 0.0

            if (8 <= hour <= 11) or (18 <= hour <= 22):
                time_effect = 3
            elif (hour >= 22) or (hour <= 7):
                time_effect = -3
            else:
                time_effect = 0

            raw_time = 33 + (10 * rate) + 2 * (vcount - 10) + time_effect
            try:
                st = int(round(raw_time))
            except Exception:
                st = 33
            signal_time = max(15, min(65, st))

            if vcount > max_vehicle:
                max_vehicle = vcount
            if signal_time > max_green:
                max_green = signal_time

        insert_result(
            region_name,
            intersection_id,
            now.date(),
            now.time(),
            max_vehicle,
            max_green,
        )
    except Exception as _e:
        print(f"[app] Failed to insert analyze_video_multi aggregated result: {_e}")

    return jsonify({"lanes": lanes})


@app.route('/api/db/test', methods=['GET', 'POST'])
def db_test():
    """Test endpoint to insert a sample row into the results table.

    Accepts optional form/query params: region_name, intersection_id, vehicle_count, green_time
    """
    try:
        # Accept values from form (POST) or querystring (GET)
        region_name = request.values.get('region_name') or request.values.get('Region_Name') or 'TestRegion'
        intersection_id = request.values.get('intersection_id') or request.values.get('Intersection_ID') or 'TEST-1'
        try:
            vehicle_count = int(request.values.get('vehicle_count', 5))
        except Exception:
            vehicle_count = 5
        try:
            green_time = int(request.values.get('green_time', 30))
        except Exception:
            green_time = 30

        now = datetime.now()
        last_id = insert_result(region_name, intersection_id, now.date(), now.time(), vehicle_count, green_time)
        return jsonify({
            'ok': True,
            'inserted_id': last_id,
            'region_name': region_name,
            'intersection_id': intersection_id,
            'vehicle_count': vehicle_count,
            'green_time': green_time,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/db/latest', methods=['GET'])
def db_latest():
    """Return the latest stored aggregated result for a given region/intersection.

    Query params: region_name, intersection_id
    """
    region = request.args.get('region_name') or request.args.get('Region_Name')
    intersection = request.args.get('intersection_id') or request.args.get('Intersection_ID')
    if not region or not intersection:
        return jsonify({'error': 'Missing region_name or intersection_id'}), 400
    try:
        row = fetch_latest_result(region, intersection)
        if not row:
            return jsonify({'ok': False, 'error': 'No records found for the provided region/intersection'}), 404
        # Convert any non-JSON-serializable types (datetime/date/time/timedelta) to strings
        for k, v in list(row.items()):
            try:
                if isinstance(v, (_dt.datetime, _dt.date, _dt.time, _dt.timedelta)):
                    row[k] = str(v)
                else:
                    # leave other types as-is (numbers, strings)
                    row[k] = v
            except Exception:
                row[k] = str(v)
        return jsonify({'ok': True, 'row': row})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
