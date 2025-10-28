# ===================================================================
#   main.py
#   - ī�޶� 2��(��ķ, ESP32-CAM) ����
#   - ToF �Ǵ� HC-SR04(ESP32 HTTP) ��� Ʈ���� + �ð��� ��ü Ʈ����
#   - ����ȭ �� ��� �� ��ó�� �����忡 �۾� ť��
# ===================================================================

import cv2
import time
import pickle
import threading
import queue
from collections import deque
import numpy as np
import os
from datetime import datetime
import requests
from requests.exceptions import RequestException

# ����� ���
from config import *
import utils
import vision
from tracker import ObjectTracker
from background_worker import processing_worker
import firebase_handler

# ToF ����
try:
    import qwiic_vl53l5cx
    import qwiic_i2c
except ImportError:
    pass  # config.py���� �ȳ�

# ---------- ���� ���� ----------
cap = None
is_capturing = False
frame_buffer = deque(maxlen=CAPTURE_BUFFER_SIZE)
inventory_tracker = {}
pending_post_process_tasks = []
tof_buffer = deque(maxlen=3)
active_camera_source = 0  # 0=OFF, 1=��ķ, 2=ESP32-CAM
LOCATION_MAP = {1: FIREBASE_LOCATION_1, 2: FIREBASE_LOCATION_2}

# HC-SR04(ESP32 HTTP) ����
hcsr04_baseline_dist = 30.0  # cm, ���� �Ÿ�
hcsr04_session = requests.Session()  # HTTP ���� ����

# ����� ��
debug_main_cam = False
debug_opencv_mask = False
debug_tof_heatmap = False

# ���� �ӽ�
system_state = "IDLE"  # IDLE, ARMED, STABILIZING, COOLDOWN
cooldown_end_time = 0
tracker = ObjectTracker()
stabilizing_object_id = None

# ---------- ���丮/DB ----------
for dir_path in [IMAGE_DIR, POST_PROCESS_OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
try:
    with open(DB_FILE, 'rb') as f:
        known_objects_db = pickle.load(f)
    print(f"[SUCCESS] DB �ε� �Ϸ�: {len(known_objects_db)}�� ��ü.")
except Exception:
    known_objects_db = {}
    print("[INFO] DB ���� ����. ���ο� DB�� �����˴ϴ�.")

# ����(��Ȯ��) Ư¡ DB
unconfirmed_side_views_db = {}

# ---------- Firebase ----------
firebase_initialized = firebase_handler.initialize_firebase()

# ---------- ��ó�� ������ ----------
post_process_queue = queue.Queue()
post_process_progress = {'status': 'Idle', 'percentage': 0}
processing_thread = threading.Thread(
    target=processing_worker,
    args=(post_process_queue, known_objects_db, unconfirmed_side_views_db, post_process_progress),
    daemon=True
)
processing_thread.start()

# ---------- ToF ���� ----------
tof_sensor = None
if TOF_SENSOR_ENABLED:
    try:
        print("[INFO] ToF ���� �ʱ�ȭ...")
        if not qwiic_i2c.isDeviceConnected(0x29):
            print("[ERROR] I2C �ּ� 0x29���� ToF ���� ���� ����.")
        else:
            tof_sensor = qwiic_vl53l5cx.QwiicVL53L5CX()
            if not tof_sensor.begin():
                print("[ERROR] ToF begin() ����.")
                tof_sensor = None
            else:
                tof_sensor.set_resolution(TOF_RESOLUTION)
                tof_sensor.set_ranging_frequency_hz(15)
                tof_sensor.baseline_depth_map = np.full((8, 8), 4000, dtype=np.int32)
                print("[SUCCESS] ToF ���� �ʱ�ȭ �Ϸ�.")
    except Exception as e:
        print(f"[WARN] ToF ���� �ʱ�ȭ ����: {e}")
        tof_sensor = None

# ---------- ESP32(HTTP) �Ÿ� ----------
def get_esp32_distance():
    """ESP32_DISTANCE_URL���� �Ÿ�(cm) �ؽ�Ʈ�� �޾� float�� ��ȯ. ���� �� 9999."""
    if not HCSR04_ENABLED:
        return 9999
    try:
        r = hcsr04_session.get(ESP32_DISTANCE_URL, timeout=0.5)
        r.raise_for_status()
        return float(r.text)
    except (RequestException, ValueError):
        return 9999

# ---------- ���� ----------
def close_camera():
    """ī�޶�/������/ToF ����."""
    global cap
    print("[INFO] ī�޶� ����...")
    if cap:
        cap.release()
    cap = None
    if tof_sensor:
        try:
            tof_sensor.stop_ranging()
            print("[INFO] ToF Ranging ����.")
        except Exception as e:
            print(f"[WARN] ToF stop_ranging ����: {e}")
    cv2.destroyAllWindows()

def set_camera_source(source_id):
    """ī�޶� �ҽ� ��ȯ(0=OFF, 1=��ķ, 2=ESP32-CAM) �� ���� �ʱ�ȭ."""
    global cap, active_camera_source
    global system_state, cooldown_end_time, tracker, stabilizing_object_id
    global frame_buffer, is_capturing, pending_post_process_tasks

    print(f"\n[STATE] ī�޶� ����: {source_id} (����: {active_camera_source})")

    close_camera()

    print("[STATE] ����/Ʈ��Ŀ/���� �ʱ�ȭ.")
    system_state = "IDLE"
    cooldown_end_time = time.time()
    tracker = ObjectTracker()
    stabilizing_object_id = None
    frame_buffer.clear()
    is_capturing = False

    active_camera_source = source_id

    if source_id == 1:
        print("[INFO] ī�޶� 1(��ķ) ON...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not cap.isOpened():
            print("!!! ��ķ(0) ���� ����")
            cap = None
            active_camera_source = 0
        elif tof_sensor:
            try:
                tof_sensor.start_ranging()
                print("[INFO] ToF Ranging ����.")
            except Exception as e:
                print(f"[WARN] ToF start_ranging ����: {e}")

    elif source_id == 2:
        print("[INFO] ī�޶� 2(ESP32-CAM) ON...")
        cap = cv2.VideoCapture(ESP32_CAM_URL)
        if not cap.isOpened():
            print(f"!!! ESP32-CAM ���� ����: {ESP32_CAM_URL}")
            cap = None
            active_camera_source = 0

    elif source_id == 0:
        print("[INFO] ��� ī�޶� OFF.")
        active_camera_source = 0
        if pending_post_process_tasks:
            print(f"\n[INFO] ��� �� ��ó�� {len(pending_post_process_tasks)}�� ����.")
            for task in pending_post_process_tasks:
                post_process_queue.put(task)
            pending_post_process_tasks.clear()

    print(f"[STATE] �ʱ�ȭ �Ϸ�: {system_state}, Cam: {active_camera_source}")

def main():
    global cap, is_capturing, frame_buffer, inventory_tracker, known_objects_db
    global debug_main_cam, debug_opencv_mask, debug_tof_heatmap
    global system_state, cooldown_end_time, tof_sensor, active_camera_source
    global tracker, stabilizing_object_id, hcsr04_baseline_dist

    active_label_ids = {}        # �̸��� ���� label id ����
    tracker_id_to_label_id = {}  # tracker id -> "�̸�-n" ����

    print("\n--- ���α׷� ���� ---")
    print(" '0':OFF | '1':��ķ | '2':ESP32 | 'b':��� | 'd'/'v'/'t':����� | 'q':����")

    try:
        while True:
            # ��Ʈ�� �г�
            panel_h = CAMERA_SQUARE_SIZE
            panel_w = CAMERA_SQUARE_SIZE
            control_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

            frame, final_mask = None, {}
            tracked_objects = {}
            current_tof_map_display = None
            filtered_tof_map = None
            detections_boxes = []
            current_hcsr04_dist = 9999

            # ---------- ī�޶� ������ ----------
            if cap and cap.isOpened():
                ret, frame_raw = cap.read()
                if not ret:
                    print("[WARN] ������ �б� ����.")
                    time.sleep(1)
                    continue
                frame = utils.to_square(frame_raw)
                H, W = frame.shape[:2]

                # ---------- ToF �б�(��ķ�� ��) ----------
                tof_min_dist = 9999
                current_tof_map_raw = None

                if active_camera_source == 1 and tof_sensor and tof_sensor.check_data_ready():
                    tof_data = tof_sensor.get_ranging_data()
                    if tof_data and hasattr(tof_data, "distance_mm"):
                        distances = getattr(tof_data, "distance_mm")
                        if distances and len(distances) == 64:
                            current_tof_map_raw = np.array(distances, dtype=np.float32).reshape(8, 8)

                if current_tof_map_raw is not None:
                    tof_buffer.append(current_tof_map_raw)
                    if len(tof_buffer) == tof_buffer.maxlen:
                        filtered_tof_map_float = np.mean(np.array(tof_buffer), axis=0)
                        filtered_tof_map = filtered_tof_map_float.astype(np.int32)
                        # �����ڸ� ����
                        filtered_tof_map[:, 0] = 4000
                        filtered_tof_map[:, 7] = 4000
                        current_tof_map_display = filtered_tof_map.copy()
                        try:
                            tof_min_dist = int(filtered_tof_map[:, 1:7].min())
                        except ValueError:
                            tof_min_dist = 4000
                    else:
                        filtered_tof_map = None
                        current_tof_map_display = current_tof_map_raw.astype(np.int32)
                        temp_map_for_min = current_tof_map_raw.copy()
                        temp_map_for_min[:, 0] = 4000
                        temp_map_for_min[:, 7] = 4000
                        try:
                            tof_min_dist = int(temp_map_for_min[:, 1:7].min())
                        except ValueError:
                            tof_min_dist = 4000
                else:
                    filtered_tof_map = None
                    current_tof_map_display = None
                    tof_min_dist = 9999

                num_active = 0
                if filtered_tof_map is not None:
                    core = filtered_tof_map[:, 1:7]
                    active_mask = (core >= 300) & (core < 1200)
                    num_active = int(active_mask.sum())

                # ---------- HC-SR04(ESP32) ----------
                if active_camera_source == 2 and HCSR04_ENABLED:
                    current_hcsr04_dist = get_esp32_distance()

                # ---------- �ð� ��ȭ Ž�� + ToF ----------
                detections_boxes, final_mask = vision.detect_objects_by_change_with_tof(frame, filtered_tof_map)

                # ---------- ���� �ӽ� ----------
                current_time = time.time()
                if system_state == "COOLDOWN" and current_time > cooldown_end_time:
                    system_state = "IDLE"
                    print("[STATE] Cooldown finished. Ready.")

                if USE_AUTOMATIC_CAPTURE:
                    # ���� ����
                    if system_state == "IDLE" and current_time > cooldown_end_time:
                        # ToF Ʈ����(��ķ)
                        tof_triggered = (
                            active_camera_source == 1
                            and (TOF_TRIGGER_DISTANCE_MIN_MM <= tof_min_dist < TOF_TRIGGER_DISTANCE_MAX_MM)
                            and (num_active >= TOF_PIXEL_THRESHOLD)
                        )
                        # ESP32(HC-SR04) �Ǵ� �ð� ��ü Ʈ����
                        visual_triggered_sensor = False
                        visual_triggered_visual = False

                        if active_camera_source == 2:
                            if HCSR04_ENABLED and current_hcsr04_dist < 9000:
                                lower = hcsr04_baseline_dist * HCSR04_TRIGGER_LOWER_BOUND
                                upper = hcsr04_baseline_dist * HCSR04_TRIGGER_UPPER_BOUND
                                if lower <= current_hcsr04_dist <= upper:
                                    visual_triggered_sensor = True
                            else:
                                if len(detections_boxes) > 0:
                                    visual_triggered_visual = True

                        if tof_triggered or visual_triggered_sensor or visual_triggered_visual:
                            system_state = "ARMED"
                            is_capturing = True
                            if active_camera_source == 1:
                                frame_buffer.clear()
                            trigger_source = 'ToF' if tof_triggered else ('HCSR04' if visual_triggered_sensor else 'Visual')
                            print(f"[STATE] ARMED & CAPTURING STARTED! (Trigger: {trigger_source} - Cam {active_camera_source})")

                    elif system_state == "ARMED":
                        if is_capturing and active_camera_source == 1:
                            frame_buffer.append(frame.copy())

                        tracked_objects = tracker.update(detections_boxes)

                        # ��Ż ����(��ķ)
                        tof_exit = (
                            active_camera_source == 1
                            and (tof_min_dist >= TOF_TRIGGER_DISTANCE_MAX_MM or num_active == 0)
                        )
                        # ��Ż ����(ESP32)
                        visual_exit = False
                        if active_camera_source == 2:
                            if HCSR04_ENABLED and current_hcsr04_dist < 9000:
                                upper = hcsr04_baseline_dist * HCSR04_TRIGGER_UPPER_BOUND
                                if (current_hcsr04_dist > upper) or (current_hcsr04_dist < hcsr04_baseline_dist * HCSR04_TRIGGER_LOWER_BOUND):
                                    visual_exit = True
                            else:
                                if len(tracked_objects) == 0:
                                    visual_exit = True

                        if tof_exit or visual_exit:
                            system_state = "STABILIZING"
                            is_capturing = False
                            stabilization_check_start_time = time.time()
                            stabilizing_object_id = None
                            reason = 'ToF Exit' if tof_exit else ('HCSR04 Exit' if visual_exit and HCSR04_ENABLED else 'Visual Exit')
                            print(f"[STATE] STABILIZING... ({reason}). ����ȭ ����.")

                if system_state != "ARMED":
                    tracked_objects = tracker.update(detections_boxes)

                # ---------- ����ȭ ----------
                if system_state == "STABILIZING":
                    stabilization_succeeded = False
                    stabilization_timed_out = (time.time() - stabilization_check_start_time) > STABILIZATION_TIMEOUT_SECONDS

                    unknown_objects_for_tracking = {
                        oid: data for oid, data in tracked_objects.items()
                        if data.get('name', 'Unknown') == 'Unknown'
                    }

                    if unknown_objects_for_tracking:
                        # ���� ū �ڽ� ����
                        current_target_id = max(
                            unknown_objects_for_tracking,
                            key=lambda oid: (unknown_objects_for_tracking[oid]['bbox'][2] - unknown_objects_for_tracking[oid]['bbox'][0]) *
                                            (unknown_objects_for_tracking[oid]['bbox'][3] - unknown_objects_for_tracking[oid]['bbox'][1])
                        )

                        if stabilizing_object_id != current_target_id:
                            if stabilizing_object_id is not None and stabilizing_object_id in tracker.objects:
                                tracker.objects[stabilizing_object_id]['stable_frames'] = 0
                            stabilizing_object_id = current_target_id
                            if stabilizing_object_id in tracker.objects:
                                tracker.objects[stabilizing_object_id]['stable_frames'] = 1

                        if stabilizing_object_id in tracker.objects:
                            if (
                                'prev_bbox' in tracker.objects.get(stabilizing_object_id, {})
                                and tracker.objects[stabilizing_object_id]['prev_bbox'] is not None
                                and utils.calculate_iou(
                                    tracker.objects[stabilizing_object_id]['prev_bbox'],
                                    tracked_objects[stabilizing_object_id]['bbox']
                                ) > 0.90
                            ):
                                tracker.objects[stabilizing_object_id]['stable_frames'] += 1
                            else:
                                tracker.objects[stabilizing_object_id]['stable_frames'] = 1
                            tracker.objects[stabilizing_object_id]['prev_bbox'] = tracked_objects[stabilizing_object_id]['bbox']
                            stabilization_succeeded = tracker.objects.get(stabilizing_object_id, {}).get('stable_frames', 0) >= STABILITY_FRAME_THRESHOLD
                    else:
                        stabilizing_object_id = None

                    # ����/Ÿ�Ӿƿ� ó��
                    if stabilization_succeeded or stabilization_timed_out:
                        all_unknowns_now = {
                            oid: data for oid, data in tracked_objects.items()
                            if data.get('name', 'Unknown') == 'Unknown'
                        }

                        if not all_unknowns_now:
                            print("[INFO] ����ȭ/Ÿ�Ӿƿ� ����: Unknown ��ü ����. ��� �ǳʶ�.")
                        else:
                            print(f"[INFO] ����ȭ �Ϸ�. Unknown �ĺ� {len(all_unknowns_now)}�� ���͸�...")
                            truly_new_candidates = []

                            for oid, data in all_unknowns_now.items():
                                try:
                                    x1, y1, x2, y2 = data['bbox']
                                    temp_crop = frame[max(0, y1):y2, max(0, x1):x2]
                                    if temp_crop is None or temp_crop.size == 0:
                                        continue
                                    name_guess, sim, vector = vision.recognize_object(temp_crop, known_objects_db)

                                    if name_guess == "Unknown":
                                        print(f"[INFO] �ĺ� ID {oid} �űԷ� �Ǵ�(���絵 {sim:.2f}).")
                                        truly_new_candidates.append({'id': oid, 'data': data, 'vector': vector})
                                    else:
                                        print(f"[INFO] �ĺ� ID {oid} ���� '{name_guess}'�� �ν�(���絵 {sim:.2f}). ����.")
                                        if oid in tracker.objects:
                                            tracker.objects[oid]['name'] = name_guess
                                            tracker.objects[oid]['is_new'] = False
                                except Exception as e:
                                    print(f"[WARN] �ĺ� ID {oid} ���� �ν� ����: {e}")

                            if not truly_new_candidates:
                                print("[INFO] Unknown �־����� ��� ���� ��ü�� Ȯ��. ��� �ǳʶ�.")
                            else:
                                # ���� ū �ĺ� ����
                                best_candidate = max(
                                    truly_new_candidates,
                                    key=lambda c: (c['data']['bbox'][2] - c['data']['bbox'][0]) *
                                                  (c['data']['bbox'][3] - c['data']['bbox'][1])
                                )
                                target_id_for_post = best_candidate['id']
                                stable_vector = best_candidate['vector']
                                print(f"[INFO] ���� ��� ���: ID {target_id_for_post}. ���� ����/��� ����...")

                                try:
                                    x1, y1, x2, y2 = best_candidate['data']['bbox']
                                    pad = 4
                                    x1 = max(0, min(x1 - pad, W - 1))
                                    y1 = max(0, min(y1 - pad, H - 1))
                                    x2 = max(0, min(x2 + pad, W - 1))
                                    y2 = max(0, min(y2 + pad, H - 1))
                                    if x2 <= x1: x2 = min(W - 1, x1 + 1)
                                    if y2 <= y1: y2 = min(H - 1, y1 + 1)
                                    stable_crop_simple = frame[y1:y2, x1:x2]

                                    match_name, similarity = vision.find_unconfirmed_match(stable_vector, unconfirmed_side_views_db)
                                    if match_name and similarity >= HYPOTHESIS_TEST_HIGH_CONFIDENCE:
                                        object_name = match_name
                                    elif match_name and similarity >= HYPOTHESIS_TEST_MEDIUM_CONFIDENCE:
                                        object_name = match_name if utils.get_yes_no_popup("���� Ȯ��", f"'{match_name}'�� �½��ϱ�?") \
                                                    else utils.get_name_with_timeout("�� ��ü ���", "�� �̸�:", known_objects_db)
                                    else:
                                        object_name = utils.get_name_with_timeout("�� ��ü ���", "�� �̸�:", known_objects_db)

                                    if object_name:
                                        vision.initial_register(object_name, known_objects_db, unconfirmed_side_views_db)
                                        registration_success = vision.update_object_data(object_name, stable_crop_simple, known_objects_db)
                                        if registration_success:
                                            print(f"[INFO] '{object_name}' ��� �Ϸ�.")
                                            if target_id_for_post in tracker.objects:
                                                tracker.objects[target_id_for_post]['name'] = object_name
                                                tracker.objects[target_id_for_post]['is_new'] = False
                                        else:
                                            print(f"[INFO] '{object_name}' �ߺ����� ����.")

                                        # ��ó�� ť��(��ķ) �Ǵ� ��� ���ε�(ESP32)
                                        if active_camera_source == 1:
                                            target_bbox = best_candidate['data']['bbox']
                                            pending_post_process_tasks.append({
                                                'buffer': list(frame_buffer),
                                                'name': object_name,
                                                'target_bbox': target_bbox,
                                                'source_id': active_camera_source
                                            })
                                            print(f"[INFO] '{object_name}'(Cam 1) �� �м� ��⿭ �߰�.")
                                        elif active_camera_source == 2:
                                            print(f"[INFO] '{object_name}'(Cam 2) ��� ���ε�...")
                                            timestamp_fb = int(time.time())
                                            clean_filename = f"{object_name.replace(' ', '_')}_{timestamp_fb}_best.jpg"
                                            os.makedirs(IMAGE_DIR, exist_ok=True)
                                            local_clean_path = os.path.join(IMAGE_DIR, clean_filename)
                                            cv2.imwrite(local_clean_path, stable_crop_simple)
                                            korean_payload = {
                                                "�̸�": object_name, "����": 1,
                                                "��ġ": FIREBASE_LOCATION_2,
                                                "������ �̺�Ʈ �ð�": datetime.now().strftime('%y/%m/%d-%H%M%S')
                                            }
                                            firebase_handler.upload_inventory_state(korean_payload, local_image_path=local_clean_path)
                                            print(f"[INFO] '{object_name}'(Cam 2) ���ε� �Ϸ�.")
                                    else:
                                        print("[WARN] ��� ���.")

                                except (KeyError, ValueError) as e:
                                    print(f"[WARN] ����ȭ ó�� ����({e}). ��� �ǳʶ�.")

                        system_state = "COOLDOWN"
                        cooldown_end_time = time.time() + CAPTURE_COOLDOWN_SECONDS
                        print(f"[STATE] COOLDOWN. {CAPTURE_COOLDOWN_SECONDS}�� �� ��Ȱ��ȭ.")
                        stabilizing_object_id = None

                # ---------- IDLE: ���/�� ���� ----------
                elif system_state == "IDLE":
                    # Unknown ��� �̸� ����
                    for obj_id, obj_data in tracked_objects.items():
                        if obj_data['name'] == 'Unknown':
                            x1, y1, x2, y2 = obj_data['bbox']
                            name, _, _ = vision.recognize_object(frame[y1:y2, x1:x2], known_objects_db)
                            if name != "Unknown":
                                obj_data['name'] = name
                                obj_data['is_new'] = False

                    # ����� Ʈ��Ŀ ����
                    current_ids = set(tracked_objects.keys())
                    disappeared_ids = set(tracker_id_to_label_id.keys()) - current_ids
                    for tid in disappeared_ids:
                        del tracker_id_to_label_id[tid]

                    active_label_ids.clear()
                    for label_base in tracker_id_to_label_id.values():
                        try:
                            name, lid_str = label_base.rsplit('-', 1)
                            active_label_ids.setdefault(name, set()).add(int(lid_str))
                        except ValueError:
                            continue

                    # ���� ���̴� ��ü�� label id �ο�
                    for tracker_id, obj_data in tracked_objects.items():
                        name = obj_data.get('name')
                        if name and name != 'Unknown' and tracker_id not in tracker_id_to_label_id:
                            active_label_ids.setdefault(name, set())
                            new_id = 1
                            while new_id in active_label_ids[name]:
                                new_id += 1
                            tracker_id_to_label_id[tracker_id] = f"{name}-{new_id}"
                            active_label_ids[name].add(new_id)

                    # ������ �� ��ü ���� ���� + ȭ�� ǥ�ø�
                    current_frame_objects = set()
                    for tracker_id, obj_data in tracked_objects.items():
                        name = obj_data.get('name')
                        if name and name != 'Unknown':
                            label_base = tracker_id_to_label_id.get(tracker_id)
                            if label_base and name in known_objects_db and 'creation_timestamp' in known_objects_db[name]:
                                timestamp_str = known_objects_db[name]['creation_timestamp'].strftime('%y/%m/%d-%H%M')
                                unique_name = f"cam{active_camera_source}:{label_base}-{timestamp_str}"
                                display_name = f"[{active_camera_source}] {label_base}-{timestamp_str}"
                                if obj_data.get('needs_confirmation', False):
                                    display_name += " (?)"
                                obj_data['display_name'] = display_name
                                current_frame_objects.add(unique_name)

                    # IN �̺�Ʈ + Firestore
                    now_ts = time.time()
                    for uname in current_frame_objects:
                        if uname not in inventory_tracker or inventory_tracker[uname]['status'] == 'OUT':
                            if uname not in inventory_tracker or now_ts - inventory_tracker[uname].get('exit_time', 0) > EVENT_THRESHOLD_SECONDS:
                                utils.log_event(uname, "IN")
                                try:
                                    prefix, real_name_part = uname.split(':', 1)
                                    base_label, timestamp_str = real_name_part.rsplit('-', 1)
                                    name, _ = base_label.rsplit('-', 1)
                                    stock_time = timestamp_str
                                except ValueError:
                                    name = uname
                                    stock_time = datetime.now().strftime('%y/%m/%d-%H%M')

                                if firebase_initialized:
                                    current_event_time = datetime.now().strftime('%y/%m/%d-%H%M%S')
                                    current_location = LOCATION_MAP.get(active_camera_source, FIREBASE_DEFAULT_LOCATION)
                                    item_data_korean = {
                                        "�̸�": real_name_part,
                                        "���� �԰� �ð�": stock_time,
                                        "����": 1,
                                        "��ġ": current_location,
                                        "������ �̺�Ʈ �ð�": current_event_time
                                    }
                                    print(f"[DEBUG] Firestore Upload Data: {item_data_korean}")
                                    firebase_handler.upload_inventory_state(item_data_korean)

                            inventory_tracker[uname] = {'status': 'IN', 'entry_time': now_ts}
                        inventory_tracker[uname]['last_seen'] = now_ts

                    # OUT �̺�Ʈ
                    cam_prefix = f"cam{active_camera_source}:"
                    for uname, data in list(inventory_tracker.items()):
                        if not uname.startswith(cam_prefix):
                            continue
                        if uname not in current_frame_objects and data['status'] == 'IN':
                            if time.time() - data.get('last_seen', 0) > EVENT_THRESHOLD_SECONDS:
                                if time.time() - data.get('entry_time', 0) > EVENT_THRESHOLD_SECONDS:
                                    data['status'] = 'OUT'
                                    data['exit_time'] = time.time()
                                    utils.log_event(uname, "OUT")

            # ---------- ȭ�� ǥ�� ----------
            if debug_main_cam and cap and cap.isOpened():
                display_frame = frame.copy()
                if tracked_objects:
                    for obj_id, obj_data in tracked_objects.items():
                        x1, y1, x2, y2 = obj_data['bbox']
                        color = (0, 255, 0) if obj_data.get('name', 'Unknown') != "Unknown" else (0, 0, 255)
                        display_name = obj_data.get('display_name', f"[{active_camera_source}] {obj_data.get('name', 'Unknown')}-{obj_id}")
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, display_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow('Camera View', display_frame)
            else:
                try:
                    cv2.destroyWindow('Camera View')
                except cv2.error:
                    pass

            if debug_opencv_mask and final_mask is not None:
                cv2.imshow("Change Detection Mask", final_mask)
            else:
                try:
                    cv2.destroyWindow('Change Detection Mask')
                except cv2.error:
                    pass

            if active_camera_source == 1 and debug_tof_heatmap and current_tof_map_display is not None:
                heatmap = utils.draw_tof_heatmap(current_tof_map_display)
                cv2.imshow("ToF Heatmap", heatmap)
            else:
                try:
                    cv2.destroyWindow('ToF Heatmap')
                except cv2.error:
                    pass

            # ��Ʈ�� �г� �ؽ�Ʈ
            status_text_cp = f"Camera: {active_camera_source} (State: {system_state})"
            if active_camera_source == 0 and pending_post_process_tasks:
                status_text_cp = f"Cam: 0 (Pending: {len(pending_post_process_tasks)})"

            cv2.putText(control_panel, status_text_cp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if active_camera_source == 2 and HCSR04_ENABLED:
                dist_text = f"Dist: {current_hcsr04_dist:.1f} cm (Base: {hcsr04_baseline_dist:.1f})"
                cv2.putText(control_panel, dist_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if system_state == "STABILIZING" and stabilizing_object_id is not None and stabilizing_object_id in tracked_objects:
                try:
                    timer_text = f"Stabilizing... [{tracked_objects[stabilizing_object_id]['stable_frames']}/{STABILITY_FRAME_THRESHOLD}]"
                    cv2.putText(control_panel, timer_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                except KeyError:
                    pass

            if post_process_progress['status'] != "Idle":
                progress_info = f"{post_process_progress['status']} [{post_process_progress['percentage']}%]"
                cv2.putText(control_panel, progress_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if system_state == "IDLE" and time.time() < cooldown_end_time:
                cv2.putText(control_panel, "Cooldown...", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(control_panel, "0/1/2:Cam b:BG d:View v:Msk t:ToF q:Quit",
                        (10, panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Control Panel', control_panel)

            # ---------- �Է� ----------
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('0'):
                set_camera_source(0)
            elif key == ord('1'):
                set_camera_source(1)
            elif key == ord('2'):
                set_camera_source(2)
            elif key == ord('d'):
                debug_main_cam = not debug_main_cam
                print(f"[INFO] Main View {'ON' if debug_main_cam else 'OFF'}")
            elif key == ord('v'):
                debug_opencv_mask = not debug_opencv_mask
                print(f"[INFO] Mask View {'ON' if debug_opencv_mask else 'OFF'}")
            elif key == ord('t'):
                if TOF_SENSOR_ENABLED:
                    debug_tof_heatmap = not debug_tof_heatmap
                    print(f"[INFO] ToF View {'ON' if debug_tof_heatmap else 'OFF'}")
            elif cap and cap.isOpened():
                if key == ord('b'):
                    ret, bg_frame_raw = cap.read()
                    if ret:
                        bg_frame = utils.to_square(bg_frame_raw)
                        preview = bg_frame.copy()
                        cv2.putText(preview, "Set BG? (3s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("BG Preview", preview)
                        cv2.waitKey(3000)
                        cv2.destroyWindow("BG Preview")

                        print("[INFO] ���� ȭ���� ������� �н�...")
                        for _ in range(30):
                            vision.bg_subtractor.apply(bg_frame, learningRate=0.05)

                        # ToF ���� ��(��ķ)
                        if active_camera_source == 1 and tof_sensor and tof_sensor.check_data_ready():
                            tof_data = tof_sensor.get_ranging_data()
                            if tof_data and hasattr(tof_data, "distance_mm"):
                                distances = getattr(tof_data, "distance_mm")
                                if distances and len(distances) == 64:
                                    tof_sensor.baseline_depth_map = np.array(distances, dtype=np.int32).reshape(8, 8)
                                    print("[INFO] ToF ���� ���� �� ����.")

                        # HC-SR04 ���� �Ÿ�(ESP32)
                        if active_camera_source == 2 and HCSR04_ENABLED:
                            print("[INFO] HC-SR04 ���� �Ÿ� ����...")
                            dist = get_esp32_distance()
                            if dist < 9000:
                                hcsr04_baseline_dist = dist
                                print(f"[SUCCESS] ���� �Ÿ�: {hcsr04_baseline_dist:.1f} cm")
                            else:
                                print("[WARN] �Ÿ� ���� ����. ���� ���� �Ұ�.")

                        print("[INFO] ��� �н� �Ϸ�.")

    finally:
        # ---------- ���� ó�� ----------
        print("\n--- ���� ���� ���� ---")
        set_camera_source(0)

        post_process_queue.put(None)
        print("��ó�� ������ ���� ���(�ִ� 10��)...")
        processing_thread.join(timeout=10)
        if processing_thread.is_alive():
            print("[WARN] ��ó�� ������ �ð� �ʰ�.")
        else:
            print("��ó�� ������ ���� ����.")

        cv2.destroyAllWindows()
        try:
            with open(DB_FILE, 'wb') as f:
                pickle.dump(known_objects_db, f)
            print(f"[SUCCESS] DB ���� �Ϸ�: '{DB_FILE}'")
        except Exception as e:
            print(f"[WARN] DB ���� ����: {e}")
        print("--- ���α׷� ���� ---")

if __name__ == '__main__':
    main()
