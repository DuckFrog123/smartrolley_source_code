# ===================================================================
# vision.py
# - 특징 추출(ResNet18), 손/텍스트 감지, 변화 탐지, 정밀 크롭, DB 업데이트 유틸
# ===================================================================

import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from scipy.spatial.distance import cosine
import mediapipe as mp
import os
from pyzbar.pyzbar import decode
from datetime import datetime
import time
from PIL import Image

from config import *

# --- 모델 준비 ---
device = torch.device('cpu')
try:
    feature_extractor = models.resnet18(weights=None)
    feature_extractor.load_state_dict(torch.load('resnet18-f37072fd.pth', map_location=device))
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])  # GAP 전까지
    feature_extractor.to(device).eval()
    print("[INFO] ResNet-18 특징 추출기 로딩 완료.")
except FileNotFoundError:
    print("[ERROR] ResNet 모델 파일 'resnet18-f37072fd.pth'을 찾을 수 없습니다. 프로그램을 종료합니다.")
    exit()

preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 손 검출 (정지 이미지 모드)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
print("[INFO] MediaPipe 손 탐지 모델 로딩 완료.")

# 텍스트(EAST)
try:
    east_net = cv2.dnn.readNet(EAST_MODEL_PATH)
    print("[INFO] EAST 텍스트 탐지 모델 로딩 완료.")
except cv2.error:
    east_net = None
    print(f"[WARN] EAST 모델 파일 '{EAST_MODEL_PATH}'을 찾을 수 없습니다.")

# 배경차감 & 형태학 필터
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=BG_HISTORY, varThreshold=BG_VAR_THRESHOLD, detectShadows=BG_DETECT_SHADOWS)
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ToF 객체 참조(실제 초기화는 main.py)
tof_sensor = None

# -------------------------------------------------------------------
# 특징 추출
# -------------------------------------------------------------------
def extract_features(image_crop_np):
    """ResNet18로 512-D 특징 벡터 추출."""
    with torch.no_grad():
        if image_crop_np is None or image_crop_np.size == 0:
            print("[WARN] 특징 추출 시 입력 이미지가 비어있습니다.")
            return None
        image_pil = Image.fromarray(cv2.cvtColor(image_crop_np, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        return feature_extractor(input_tensor).squeeze().cpu().numpy()

# -------------------------------------------------------------------
# 인식(기지 샘플과 비교)
# -------------------------------------------------------------------
def recognize_object(image_crop, db):
    """크롭 이미지의 벡터를 추출하고 DB와 유사도 비교."""
    if image_crop is None or image_crop.size == 0:
        return "Unknown", 0.0, None

    current_vector = extract_features(image_crop)
    if current_vector is None:
        return "Unknown", 0.0, None

    if not db:
        return "Unknown", 0.0, current_vector

    best_match_name, highest_similarity = "Unknown", 0.0
    for name, data in db.items():
        if 'vectors' in data:
            for vector in data['vectors']:
                similarity = 1 - cosine(current_vector, vector)
                if similarity > highest_similarity:
                    highest_similarity, best_match_name = similarity, name

    if highest_similarity >= SIMILARITY_THRESHOLD:
        return best_match_name, highest_similarity, current_vector
    else:
        return "Unknown", highest_similarity, current_vector

# -------------------------------------------------------------------
# 미확인 측면(사전 수집 후보) 매칭
# -------------------------------------------------------------------
def find_unconfirmed_match(vector, unconfirmed_db):
    """미확인 측면 DB에서 가장 유사한 항목 탐색."""
    if not unconfirmed_db or vector is None:
        return None, 0.0
    
    best_match_name, highest_similarity = None, 0.0
    for name, vectors in unconfirmed_db.items():
        for u_vector in vectors:
            similarity = 1 - cosine(vector, u_vector)
            if similarity > highest_similarity:
                highest_similarity, best_match_name = similarity, name
    
    return best_match_name, highest_similarity

# -------------------------------------------------------------------
# DB 등록/업데이트
# -------------------------------------------------------------------
def initial_register(object_name, db, unconfirmed_db):
    """최초 등록(메타 슬롯 준비) 및 미확인 측면 제거."""
    if object_name not in db:
        db[object_name] = {
            'vectors': [],
            'images': [],
            'barcodes': [],
            'texts': [],
            'creation_timestamp': datetime.now()
        }
    if object_name in unconfirmed_db:
        try:
            del unconfirmed_db[object_name]
        except KeyError:
            pass
    print(f"[INFO] '{object_name}' 초기 등록 완료.")

def update_object_data(object_name, image_crop, db):
    """이미지로 특징을 추가하고 대표 이미지 저장."""
    if image_crop is None or image_crop.size == 0:
        print(f"[WARN] '{object_name}' 업데이트 시 이미지가 비어있습니다.")
        return False

    feature_vector = extract_features(image_crop)
    if feature_vector is None:
        print(f"[WARN] '{object_name}' 업데이트 시 특징 추출 실패.")
        return False

    if object_name not in db:
        print(f"[WARN] DB에 '{object_name}'이 없어 업데이트할 수 없습니다.")
        return False

    # 중복(유사 샘플) 필터
    is_redundant = False
    if 'vectors' in db[object_name]:
        for existing_vector in db[object_name]['vectors']:
            if (1 - cosine(feature_vector, existing_vector)) > REDUNDANCY_SIMILARITY_THRESHOLD:
                is_redundant = True
                break
    if is_redundant:
        print(f"[INFO] '{object_name}'의 새 샘플이 기존과 유사하여 추가하지 않습니다.")
        return False

    timestamp = int(time.time())
    image_filename = f"{object_name.replace(' ', '_')}_{timestamp}_best.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    try:
        cv2.imwrite(image_path, image_crop)
        print(f"[INFO] 대표 이미지 저장 완료: {image_path}")
    except Exception as e:
        print(f"[ERROR] 대표 이미지 저장 중 오류: {e}")
        return False

    db[object_name]['vectors'].append(feature_vector)
    db[object_name]['images'].append(image_path)
    print(f"[INFO] '{object_name}'의 이미지/벡터 추가 완료.")
    return True

def update_object_metadata(object_name, db, barcode=None, text_info=None):
    """바코드·텍스트 등 부가 정보 업데이트."""
    if object_name not in db: 
        return
    if barcode and barcode not in db[object_name]['barcodes']:
        db[object_name]['barcodes'].append(barcode)
    if text_info and text_info not in db[object_name]['texts']:
        db[object_name]['texts'].append(text_info)

# -------------------------------------------------------------------
# 텍스트 존재 유무(EAST)
# -------------------------------------------------------------------
def detect_text_presence_east(image):
    """EAST로 텍스트 존재 여부만 빠르게 판정."""
    if east_net is None or image is None or image.size == 0:
        return False
    (H, W) = image.shape[:2]
    if H < 32 or W < 32:
        return False

    newW, newH = 320, 320
    try:
        blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        east_net.setInput(blob)
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        (scores, geometry) = east_net.forward(layerNames)
        (numRows, numCols) = scores.shape[2:4]
        for y in range(numRows):
            if np.any(scores[0, 0, y] > EAST_CONFIDENCE_THRESHOLD):
                return True
    except cv2.error as e:
        print(f("[WARN] EAST 처리 오류: {e}"))
        return False
    return False

# -------------------------------------------------------------------
# 변화 탐지(시각)
# -------------------------------------------------------------------
def detect_objects_by_change_visual(frame):
    """MOG2 기반 변화 영역의 바운딩 박스 추출."""
    fgmask = bg_subtractor.apply(frame, learningRate=0)
    fgmask[fgmask == 127] = 0
    fgmask = cv2.medianBlur(fgmask, 5)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_boxes = []
    padding = 5
    h, w = frame.shape[:2]
    frame_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if (ww * hh) > (frame_area * MAX_BBOX_AREA_RATIO):
                continue
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(w, x + ww + padding), min(h, y + hh + padding)
            detected_boxes.append((x1, y1, x2, y2))
    return detected_boxes, fgmask

# -------------------------------------------------------------------
# 변화 탐지(시각+ToF 게이팅)
# -------------------------------------------------------------------
def detect_objects_by_change_with_tof(frame, current_tof_map=None):
    """
    시각 변화 + ToF '높이 증가' 교집합 마스크 생성 후 박스 반환.
    """
    global tof_sensor
    _, fgmask = detect_objects_by_change_visual(frame)
    if fgmask is None:
        return [], None

    h, w = frame.shape[:2]

    if not TOF_SENSOR_ENABLED or tof_sensor is None or current_tof_map is None:
        final_mask = fgmask
    else:
        depth_delta = tof_sensor.baseline_depth_map.astype(np.int32) - current_tof_map.astype(np.int32)
        tof_height_increase_mask_8x8 = (depth_delta > TOF_DEPTH_THRESHOLD_MM)
        tof_height_increase_mask_full = cv2.resize(
            tof_height_increase_mask_8x8.astype(np.uint8) * 255,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        final_mask = cv2.bitwise_and(fgmask, tof_height_increase_mask_full)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gated_boxes = []
    padding = 5
    frame_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if (ww * hh) > (frame_area * MAX_BBOX_AREA_RATIO):
                continue
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(w, x + ww + padding), min(h, y + hh + padding)
            gated_boxes.append((x1, y1, x2, y2))
    return gated_boxes, final_mask

# -------------------------------------------------------------------
# 정밀 크롭(마스크 기반)
# -------------------------------------------------------------------
def get_precise_crop_from_mask(frame, mask):
    """마스크 최대 윤곽선 기준으로 영역을 정밀 크롭."""
    if mask is None or frame is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < MIN_CONTOUR_AREA / 2:
        return None

    x, y, w, h = cv2.boundingRect(main_contour)
    if w <= 0 or h <= 0:
        return None
    
    crop1 = frame[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]

    if crop1.shape[:2] != mask_crop.shape[:2]:
        mask_crop = cv2.resize(mask_crop, (crop1.shape[1], crop1.shape[0]), interpolation=cv2.INTER_NEAREST)
        print("[WARN] Precise crop mask resize occurred.")

    precise_crop = cv2.bitwise_and(crop1, crop1, mask=mask_crop)
    return precise_crop
