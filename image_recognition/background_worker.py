# background_worker.py
# - 후처리 워커 (멀티 카메라 지원: source_id로 위치 구분)

import cv2
import time
import os
from pyzbar.pyzbar import decode
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime

from config import *
import vision
import utils  # calculate_iou
import firebase_handler


def post_process_captured_frames(frame_buffer, object_name, target_bbox, db, unconfirmed_db, progress_dict, source_id=1):
    """
    캡처된 프레임을 분석해 텍스트/다른면 후보를 수집하고, 가장 선명한 이미지를 대표 이미지로 저장/업로드.
    source_id: 1=Main, 2=ESP32
    """
    try:
        progress_dict['status'] = f"Analyzing '{object_name}'"
        progress_dict['percentage'] = 0
        total_frames = len(frame_buffer)

        # 최소 프레임 수 확인
        if total_frames < 3:
            print(f"[WARN] 분석 프레임이 부족합니다 ({total_frames} frames). 후처리를 건너뜁니다.")
            progress_dict['status'] = "Idle"
            return

        # 1) 구간 정의
        discard_start_idx = int(total_frames * 0.05)
        air_shot_end_idx = int(total_frames * 0.15)
        stable_shot_start_idx = int(total_frames * 0.85)

        air_shots_to_analyze = frame_buffer[discard_start_idx:air_shot_end_idx]
        stable_shots_to_analyze = frame_buffer[stable_shot_start_idx:]

        progress_dict['percentage'] = 10

        # 2) 공중샷 분석: 바코드, 텍스트, 다른 면 후보
        found_barcode = None
        text_candidates = []
        side_view_vectors = []
        base_vector_for_comparison = None

        if air_shots_to_analyze:
            total_air_analysis = len(air_shots_to_analyze)
            print(f"[INFO] 공중샷 분석 시작 (대상 프레임: {total_air_analysis}개)...")
            for i, frame in enumerate(air_shots_to_analyze):
                progress_dict['percentage'] = 10 + int(((i + 1) / total_air_analysis) * 60)

                boxes, final_mask = vision.detect_objects_by_change_with_tof(frame, None)
                if not boxes or final_mask is None:
                    continue

                try:
                    best_match_box = max(boxes, key=lambda box: utils.calculate_iou(box, target_bbox))
                except ValueError:
                    continue

                if utils.calculate_iou(best_match_box, target_bbox) < 0.3:
                    continue

                object_crop = vision.get_precise_crop_from_mask(frame, final_mask)
                if object_crop is None:
                    continue

                # 바코드
                if not found_barcode:
                    barcodes = decode(object_crop)
                    if barcodes:
                        found_barcode = barcodes[0].data.decode('utf-8')

                # 텍스트 후보
                if vision.detect_text_presence_east(object_crop):
                    sharpness = cv2.Laplacian(cv2.cvtColor(object_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    text_candidates.append({'image': object_crop, 'sharpness': sharpness})

                # 다른 면 후보
                current_vector = vision.extract_features(object_crop)
                if current_vector is None:
                    continue

                if base_vector_for_comparison is None:
                    base_vector_for_comparison = current_vector
                else:
                    similarity = 1 - cosine(base_vector_for_comparison, current_vector)
                    if HYPOTHESIS_SIMILARITY_LOWER_BOUND < similarity < HYPOTHESIS_SIMILARITY_UPPER_BOUND:
                        side_view_vectors.append(current_vector)

            if side_view_vectors:
                if object_name not in unconfirmed_db:
                    unconfirmed_db[object_name] = []
                unconfirmed_db[object_name].extend(side_view_vectors)
                print(f"[INFO] 공중샷 분석: '다른 면' 후보 {len(side_view_vectors)}개 저장.")
        else:
            print("[INFO] 분석할 공중샷 프레임이 없습니다.")

        # 3) 안정화 구간 분석: 가장 선명한 이미지 선정
        best_stable_image = None
        max_sharpness = -1

        if stable_shots_to_analyze:
            total_stable_analysis = len(stable_shots_to_analyze)
            print(f"[INFO] 안정화 샷 분석 시작 (대상 프레임: {total_stable_analysis}개)...")
            for i, frame in enumerate(stable_shots_to_analyze):
                progress_dict['percentage'] = 70 + int(((i + 1) / total_stable_analysis) * 20)

                hand_results = vision.hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not hand_results.multi_hand_landmarks:
                    boxes, final_mask = vision.detect_objects_by_change_with_tof(frame, None)
                    if not boxes or final_mask is None:
                        continue

                    try:
                        best_match_box = max(boxes, key=lambda box: utils.calculate_iou(box, target_bbox))
                    except ValueError:
                        continue

                    if utils.calculate_iou(best_match_box, target_bbox) < 0.3:
                        continue

                    object_crop = vision.get_precise_crop_from_mask(frame, final_mask)
                    if object_crop is None:
                        continue

                    sharpness = cv2.Laplacian(cv2.cvtColor(object_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if sharpness > max_sharpness:
                        max_sharpness = sharpness
                        best_stable_image = object_crop
        else:
            print("[INFO] 분석할 안정화 샷 프레임이 없습니다.")

        progress_dict['percentage'] = 90

        # 4) 저장/업로드
        timestamp = int(time.time())

        # 텍스트용 이미지 (공중샷 중 가장 선명한 것)
        if text_candidates:
            text_candidates.sort(key=lambda x: x['sharpness'], reverse=True)
            best_text_image = text_candidates[0]['image']
            text_filename = f"{object_name}_text_{timestamp}.jpg"
            local_text_path = os.path.join(POST_PROCESS_OUTPUT_DIR, text_filename)
            cv2.imwrite(local_text_path, best_text_image)
            print(f"[INFO] 텍스트 인식용 이미지 저장: {local_text_path}")
            text_url = firebase_handler.upload_image_to_storage(
                local_text_path,
                destination_folder=getattr(firebase_handler, "FIREBASE_IMAGE_FOLDER", "registered_images")
            )
            if text_url:
                print(f"[INFO] 텍스트 이미지 업로드 URL: {text_url}")

        # 대표 이미지 (안정화 구간)
        if best_stable_image is not None:
            updated = vision.update_object_data(object_name, best_stable_image, db)
            if updated:
                os.makedirs(IMAGE_DIR, exist_ok=True)

                timestamp_fb = int(time.time())
                clean_filename = f"{object_name.replace(' ', '_')}_{timestamp_fb}_best.jpg"
                local_clean_path = os.path.join(IMAGE_DIR, clean_filename)
                cv2.imwrite(local_clean_path, best_stable_image)
                print(f"[INFO] 대표 이미지 저장: {local_clean_path}")

                current_location = FIREBASE_LOCATION_2 if source_id == 2 else FIREBASE_LOCATION_1
                korean_payload = {
                    "이름": object_name,
                    "개수": 1,
                    "위치": current_location,
                    "마지막 이벤트 시간": datetime.now().strftime('%y/%m/%d-%H%M%S'),
                }

                firebase_handler.upload_inventory_state(
                    korean_payload,
                    local_image_path=local_clean_path
                )
        else:
            print(f"[WARN] '{object_name}'의 안정화 이미지를 찾지 못해 DB 업데이트를 건너뜁니다.")

        # 메타데이터 업데이트
        vision.update_object_metadata(object_name, db, barcode=found_barcode)
        progress_dict['percentage'] = 100
        print(f"[SUCCESS] '{object_name}' 후처리 완료.")
        time.sleep(2)

    except Exception as e:
        print(f"[ERROR] 후처리 중 예외 발생: {e}")
    finally:
        progress_dict['status'] = "Idle"
        progress_dict['percentage'] = 0


def processing_worker(q, db, unconfirmed_db, progress_dict):
    """큐에서 작업을 가져와 후처리를 수행."""
    while True:
        try:
            task = q.get()
            if task is None:
                break

            source_id = task.get('source_id', 1)
            print(f"\n[WORKER] '{task['name']}' (Cam {source_id}) 후처리 시작.")

            post_process_captured_frames(
                task['buffer'],
                task['name'],
                task.get('target_bbox'),
                db,
                unconfirmed_db,
                progress_dict,
                source_id
            )
            print(f"[WORKER] '{task['name']}' 후처리 완료.")
            q.task_done()
        except KeyError:
            print(f"[WORKER ERROR] 작업 데이터에 'target_bbox'가 없습니다: {task}")
        except Exception as e:
            print(f"[WORKER ERROR] 작업 처리 중 오류: {e}")
