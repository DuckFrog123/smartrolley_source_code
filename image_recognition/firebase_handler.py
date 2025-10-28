# firebase_handler.py
# - Firebase 초기화 및 Firestore/Storage 업로드 유틸

import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import json
import cv2
from datetime import datetime

from config import *

db = None
bucket = None

import mimetypes
from datetime import timedelta

def _guess_content_type(filename):
    """파일 확장자로 MIME 타입 추정."""
    ctype, _ = mimetypes.guess_type(filename)
    return ctype or "application/octet-stream"


def _normalize_base_name(name: str) -> str:
    """이벤트명에서 객체 기본 이름만 추출(슬래시 제거). 예: 'galaxy-1-...' -> 'galaxy'"""
    if not name:
        return ""
    base = name.split('-', 1)[0]
    base = base.replace('/', '').strip()
    return base

def _find_recent_local_image_for_name(name, search_dir="registered_images"):
    """search_dir에서 name(정규화 포함) 프리픽스로 시작하는 최신 이미지 경로 탐색."""
    try:
        if not os.path.isdir(search_dir):
            return None

        prefixes = []
        prefixes.append(name.replace('/', ''))
        base = _normalize_base_name(name)
        if base:
            prefixes.append(base)
        prefixes = list(dict.fromkeys(p for p in prefixes if p))

        cand = []
        for f in os.listdir(search_dir):
            fl = f.lower()
            if fl.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                for p in prefixes:
                    if f.startswith(p + "_") or f.startswith(p):
                        path = os.path.join(search_dir, f)
                        try:
                            mtime = os.path.getmtime(path)
                        except Exception:
                            mtime = 0
                        cand.append((mtime, path))
                        break
        if not cand:
            return None
        cand.sort()
        return cand[-1][1]
    except Exception as e:
        print(f"[WARN] 최근 이미지 탐색 중 오류: {e}")
        return None


def initialize_firebase():
    """Firebase Admin SDK 초기화."""
    global db, bucket
    if not FIREBASE_ENABLED:
        print("[INFO] Firebase 연동이 비활성화되어 있습니다.")
        return False
        
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred, {
                'storageBucket': FIREBASE_STORAGE_BUCKET
            })
            db = firestore.client()
            bucket = storage.bucket()
            print("[SUCCESS] Firebase Admin SDK가 성공적으로 초기화되었습니다.")
            return True
        else:
            db = firestore.client()
            bucket = storage.bucket()
            print("[INFO] Firebase Admin SDK가 이미 초기화되어 있습니다.")
            return True
    except FileNotFoundError:
        print(f"[ERROR] Firebase 키 파일을 찾을 수 없습니다: {FIREBASE_KEY_PATH}")
    except ValueError as e:
        print(f"[ERROR] Firebase 초기화 중 오류 발생: {e}")
    except Exception as e:
        print(f"[ERROR] Firebase 초기화 중 예기치 않은 오류 발생: {e}")
    
    db = None
    bucket = None
    return False


def _map_keys(korean_dict):
    """한글 키를 영문 키로 매핑."""
    mapping = {
        "이름": "name",
        "최초 입고 시간": "stock_time",
        "개수": "quantity",
        "위치": "location",
        "마지막 이벤트 시간": "last_event_time",
    }

    english_dict = {}

    # 기본 시간 값
    if "마지막 이벤트 시간" in korean_dict and korean_dict["마지막 이벤트 시간"]:
        english_dict["last_event_time"] = str(korean_dict["마지막 이벤트 시간"])
    else:
        english_dict["last_event_time"] = datetime.now().strftime('%y/%m/%d-%H%M%S')

    # 매핑 수행
    for kor_key, eng_key in mapping.items():
        if kor_key in korean_dict:
            val = korean_dict[kor_key]
            if eng_key == "quantity":
                try:
                    english_dict[eng_key] = int(val)
                except (ValueError, TypeError):
                    print(f"[WARN] 수량 값을 숫자로 변환할 수 없습니다: {val}. 0으로 설정합니다.")
                    english_dict[eng_key] = 0
            else:
                english_dict[eng_key] = str(val).strip() if val is not None else ""
        else:
            print(f"[WARN] Firestore 업로드 데이터에 필수 키 '{kor_key}'가 누락되었습니다.")

    return english_dict


def upload_inventory_state(object_data_korean_keys, local_image_path=None, destination_folder=None):
    """인벤토리 문서 업로드(이미지 있으면 함께 업로드)."""
    if not db:
        print("[ERROR] Firestore 클라이언트가 초기화되지 않아 업로드할 수 없습니다.")
        return

    item_data_english_keys = _map_keys(object_data_korean_keys)

    # name 보정
    if not item_data_english_keys.get('name'):
        raw_name = object_data_korean_keys.get('이름')
        if raw_name:
            item_data_english_keys['name'] = str(raw_name).strip()

    if 'name' not in item_data_english_keys or not item_data_english_keys['name']:
        print("[ERROR] Firestore 업로드 데이터에 'name' 키가 없거나 비어있습니다. 업로드를 건너뜁니다.")
        return

    name = item_data_english_keys['name']

    # 대표 이미지 경로: 인자 -> 별칭 -> 자동탐색
    if local_image_path is None:
        alias_keys = ['대표 이미지 경로', 'image_local_path', 'image_path', '대표이미지']
        for k in alias_keys:
            if k in object_data_korean_keys and object_data_korean_keys[k]:
                local_image_path = str(object_data_korean_keys[k]).strip()
                break

    if local_image_path is None:
        local_image_path = _find_recent_local_image_for_name(name, "registered_images")

    # 업로드 폴더 결정
    dest_folder = destination_folder or globals().get("FIREBASE_IMAGE_FOLDER", "registered_images")

    # 이미지 업로드
    image_url = None
    if local_image_path:
        image_url = upload_image_to_storage(local_image_path, destination_folder=dest_folder)
        if image_url:
            item_data_english_keys['image_url'] = image_url
            item_data_english_keys['image_uploaded_at'] = datetime.now().strftime('%y/%m/%d-%H%M%S')
        else:
            print(f"[WARN] 이미지 URL 생성 실패. 문서만 업로드합니다. (path={local_image_path})")
    else:
        print("[INFO] 대표 이미지 경로를 찾지 못했습니다. 문서만 업로드합니다.")

    # Firestore 추가
    try:
        doc_ref = db.collection(FIREBASE_INVENTORY_COLLECTION).add(item_data_english_keys)
        print(f"[SUCCESS] Firestore '{FIREBASE_INVENTORY_COLLECTION}' 컬렉션에 데이터 추가 완료 (Doc ID: {doc_ref[1].id})")
        # 필요 시 doc id 반환:
        # return doc_ref[1].id
    except Exception as e:
        print(f"[ERROR] Firestore 데이터 추가 중 오류 발생: {e}")


def upload_image_to_storage(local_image_path, destination_folder="registered_images",
                            make_public=True, signed_url_days=3650):
    """Cloud Storage에 이미지 업로드 후 접근 URL 반환."""
    if not bucket:
        print("[ERROR] Storage 버킷이 초기화되지 않아 업로드할 수 없습니다.")
        return None

    if not os.path.exists(local_image_path):
        print(f"[ERROR] 업로드할 이미지 파일을 찾을 수 없습니다: {local_image_path}")
        return None

    try:
        file_name = os.path.basename(local_image_path)
        blob_path = f"{destination_folder}/{file_name}".replace("//", "/")
        blob = bucket.blob(blob_path)

        print(f"[DEBUG] Storage 업로드 시작: bucket={bucket.name}, path={blob_path}")
        blob.upload_from_filename(local_image_path, content_type=_guess_content_type(file_name))

        # 캐시 헤더
        try:
            blob.cache_control = "public, max-age=31536000"
            blob.patch()
        except Exception:
            pass

        image_url = None

        # 공개 버킷일 경우
        if make_public:
            try:
                blob.make_public()
                image_url = blob.public_url
            except Exception as e:
                print(f"[INFO] make_public 실패(비공개 버킷일 수 있음): {e}")

        # 비공개 버킷이면 서명 URL
        if not image_url:
            try:
                image_url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(days=signed_url_days),
                    method="GET",
                )
            except Exception as e:
                print(f"[ERROR] 서명 URL 생성 실패: {e}")
                image_url = None

        if image_url:
            print(f"[SUCCESS] Storage 업로드 완료: {blob_path}")
            print(f"  - URL: {image_url}")
        else:
            print(f"[ERROR] 업로드는 되었지만 접근 가능한 URL 생성에 실패했습니다: {blob_path}")

        return image_url

    except Exception as e:
        print(f"[ERROR] Storage 이미지 업로드 중 오류 발생: {e}")
        return None
