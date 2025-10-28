# ===================================================================
#   config.py
#   - v35.3 (ESP32 Sensor URL)
#   - HC-SR04 GPIO 핀 제거
#   - ESP32의 거리 센서 데이터 URL 추가
# ===================================================================

# --- 동작 관련 임계값 ---
SIMILARITY_THRESHOLD = 0.8
REDUNDANCY_SIMILARITY_THRESHOLD = 0.98
STABILITY_FRAME_THRESHOLD = 15
STABILIZATION_TIMEOUT_SECONDS = 9.0 # [추가] 안정화 최대 대기 시간 (초)
TRACKER_DISAPPEARED_THRESHOLD = 10
EVENT_THRESHOLD_SECONDS = 15
NAME_INPUT_TIMEOUT_SECONDS = 10 # 이름 입력 타임아웃 (초)

# --- [신규] 가설 수립 및 검증 임계값 ---
HYPOTHESIS_SIMILARITY_LOWER_BOUND = 0.65 # 기준 이미지와 이 값 '이상' 다르면 '다른 면' 후보
HYPOTHESIS_SIMILARITY_UPPER_BOUND = 0.85 # 기준 이미지와 이 값 '이하' 다르면 '다른 면' 후보
HYPOTHESIS_TEST_HIGH_CONFIDENCE = 0.90   # 안정화된 이미지가 '다른 면' 후보와 이 값 이상 유사하면 자동 인식
HYPOTHESIS_TEST_MEDIUM_CONFIDENCE = 0.75 # 안정화된 이미지가 '다른 면' 후보와 이 값 이상 유사하면 사용자에게 질문


# --- 이미지 및 버퍼 설정 ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_BUFFER_SIZE = 90
TOP_K_SAMPLES = 2 # 후처리 시 추가할 고품질 샘플 수
USE_SQUARE_VIEW = True
CAMERA_SQUARE_SIZE = min(CAMERA_WIDTH, CAMERA_HEIGHT)

# --- [수정] ESP32 설정 ---
ESP32_CAM_URL = "http://192.168.137.218/stream"
ESP32_DISTANCE_URL = "http://192.168.137.218/distance" # [신규]

# --- 파일 및 디렉토리 경로 ---
DB_FILE = "object_database.pkl"
IMAGE_DIR = "registered_images"
LOG_FILE = "inventory_log.csv"
BACKGROUND_MODEL_FILE = "background_model.jpg"
EAST_MODEL_PATH = "frozen_east_text_detection.pb"
POST_PROCESS_OUTPUT_DIR = "post_process_output" # 후처리 결과물 저장 폴더

# --- 배경 차감 및 윤곽선 설정 ---
MIN_CONTOUR_AREA = 2500
MAX_BBOX_AREA_RATIO = 0.8  # 화면의 80% 이상을 차지하는 객체는 무시
BG_HISTORY = 5000
BG_VAR_THRESHOLD = 270 # 야간 조명 및 빛 반사에 둔감하도록 값 상향 조정
BG_DETECT_SHADOWS = True

# --- ToF 센서 설정 ---
TOF_SENSOR_ENABLED = True # ToF 센서 사용 여부
TOF_RESOLUTION = 64
TOF_DEPTH_THRESHOLD_MM = 50
TOF_PIXEL_THRESHOLD = 5

# --- 텍스트 탐지 설정 ---
EAST_CONFIDENCE_THRESHOLD = 0.5

# --- [추가] ToF 자동 캡처 설정 ---
USE_AUTOMATIC_CAPTURE = True  # True로 설정하면 ToF 센서로 자동 캡처 시작
TOF_TRIGGER_DISTANCE_MIN_MM = 300  # 손 감지 최소 거리 (10cm)
TOF_TRIGGER_DISTANCE_MAX_MM = 1200  # 손 감지 최대 거리 (100cm)
CAPTURE_COOLDOWN_SECONDS = 5.0     # 캡처 후 재정비 시간 (초)

# --- [수정] Firebase 설정 ---
FIREBASE_ENABLED = True # Firebase 연동 사용 여부
FIREBASE_KEY_PATH = "/home/smartrolley/firebase/smartrolley-65948-firebase-adminsdk-fbsvc-f68ec976af.json"
FIREBASE_STORAGE_BUCKET = "smartrolley-65948.firebasestorage.app" # 제공해주신 URL에서 .appspot.com 부분만 사용
FIREBASE_INVENTORY_COLLECTION = "inventory"
FIREBASE_LOCATION_1 = "트롤리 1" # 기본 위치 (메인 카메라)
FIREBASE_LOCATION_2 = "트롤리 2" # ESP32 카메라 위치
FIREBASE_DEFAULT_LOCATION = FIREBASE_LOCATION_1 # 이전 버전 호환용

# --- [수정] HC-SR04 (초음파 센서) 설정 ---
HCSR04_ENABLED = True # ESP32에서 거리 값을 읽어올지 여부
# HCSR04_TRIGGER_PIN = 23 # [삭제]
# HCSR04_ECHO_PIN = 24    # [삭제]
HCSR04_TRIGGER_LOWER_BOUND = 0.2 # 기준 거리의 20%
HCSR04_TRIGGER_UPPER_BOUND = 0.7 # 기준 거리의 70%
