# firebase_handler.py
# - Firebase �ʱ�ȭ �� Firestore/Storage ���ε� ��ƿ

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
    """���� Ȯ���ڷ� MIME Ÿ�� ����."""
    ctype, _ = mimetypes.guess_type(filename)
    return ctype or "application/octet-stream"


def _normalize_base_name(name: str) -> str:
    """�̺�Ʈ���� ��ü �⺻ �̸��� ����(������ ����). ��: 'galaxy-1-...' -> 'galaxy'"""
    if not name:
        return ""
    base = name.split('-', 1)[0]
    base = base.replace('/', '').strip()
    return base

def _find_recent_local_image_for_name(name, search_dir="registered_images"):
    """search_dir���� name(����ȭ ����) �����Ƚ��� �����ϴ� �ֽ� �̹��� ��� Ž��."""
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
        print(f"[WARN] �ֱ� �̹��� Ž�� �� ����: {e}")
        return None


def initialize_firebase():
    """Firebase Admin SDK �ʱ�ȭ."""
    global db, bucket
    if not FIREBASE_ENABLED:
        print("[INFO] Firebase ������ ��Ȱ��ȭ�Ǿ� �ֽ��ϴ�.")
        return False
        
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred, {
                'storageBucket': FIREBASE_STORAGE_BUCKET
            })
            db = firestore.client()
            bucket = storage.bucket()
            print("[SUCCESS] Firebase Admin SDK�� ���������� �ʱ�ȭ�Ǿ����ϴ�.")
            return True
        else:
            db = firestore.client()
            bucket = storage.bucket()
            print("[INFO] Firebase Admin SDK�� �̹� �ʱ�ȭ�Ǿ� �ֽ��ϴ�.")
            return True
    except FileNotFoundError:
        print(f"[ERROR] Firebase Ű ������ ã�� �� �����ϴ�: {FIREBASE_KEY_PATH}")
    except ValueError as e:
        print(f"[ERROR] Firebase �ʱ�ȭ �� ���� �߻�: {e}")
    except Exception as e:
        print(f"[ERROR] Firebase �ʱ�ȭ �� ����ġ ���� ���� �߻�: {e}")
    
    db = None
    bucket = None
    return False


def _map_keys(korean_dict):
    """�ѱ� Ű�� ���� Ű�� ����."""
    mapping = {
        "�̸�": "name",
        "���� �԰� �ð�": "stock_time",
        "����": "quantity",
        "��ġ": "location",
        "������ �̺�Ʈ �ð�": "last_event_time",
    }

    english_dict = {}

    # �⺻ �ð� ��
    if "������ �̺�Ʈ �ð�" in korean_dict and korean_dict["������ �̺�Ʈ �ð�"]:
        english_dict["last_event_time"] = str(korean_dict["������ �̺�Ʈ �ð�"])
    else:
        english_dict["last_event_time"] = datetime.now().strftime('%y/%m/%d-%H%M%S')

    # ���� ����
    for kor_key, eng_key in mapping.items():
        if kor_key in korean_dict:
            val = korean_dict[kor_key]
            if eng_key == "quantity":
                try:
                    english_dict[eng_key] = int(val)
                except (ValueError, TypeError):
                    print(f"[WARN] ���� ���� ���ڷ� ��ȯ�� �� �����ϴ�: {val}. 0���� �����մϴ�.")
                    english_dict[eng_key] = 0
            else:
                english_dict[eng_key] = str(val).strip() if val is not None else ""
        else:
            print(f"[WARN] Firestore ���ε� �����Ϳ� �ʼ� Ű '{kor_key}'�� �����Ǿ����ϴ�.")

    return english_dict


def upload_inventory_state(object_data_korean_keys, local_image_path=None, destination_folder=None):
    """�κ��丮 ���� ���ε�(�̹��� ������ �Բ� ���ε�)."""
    if not db:
        print("[ERROR] Firestore Ŭ���̾�Ʈ�� �ʱ�ȭ���� �ʾ� ���ε��� �� �����ϴ�.")
        return

    item_data_english_keys = _map_keys(object_data_korean_keys)

    # name ����
    if not item_data_english_keys.get('name'):
        raw_name = object_data_korean_keys.get('�̸�')
        if raw_name:
            item_data_english_keys['name'] = str(raw_name).strip()

    if 'name' not in item_data_english_keys or not item_data_english_keys['name']:
        print("[ERROR] Firestore ���ε� �����Ϳ� 'name' Ű�� ���ų� ����ֽ��ϴ�. ���ε带 �ǳʶݴϴ�.")
        return

    name = item_data_english_keys['name']

    # ��ǥ �̹��� ���: ���� -> ��Ī -> �ڵ�Ž��
    if local_image_path is None:
        alias_keys = ['��ǥ �̹��� ���', 'image_local_path', 'image_path', '��ǥ�̹���']
        for k in alias_keys:
            if k in object_data_korean_keys and object_data_korean_keys[k]:
                local_image_path = str(object_data_korean_keys[k]).strip()
                break

    if local_image_path is None:
        local_image_path = _find_recent_local_image_for_name(name, "registered_images")

    # ���ε� ���� ����
    dest_folder = destination_folder or globals().get("FIREBASE_IMAGE_FOLDER", "registered_images")

    # �̹��� ���ε�
    image_url = None
    if local_image_path:
        image_url = upload_image_to_storage(local_image_path, destination_folder=dest_folder)
        if image_url:
            item_data_english_keys['image_url'] = image_url
            item_data_english_keys['image_uploaded_at'] = datetime.now().strftime('%y/%m/%d-%H%M%S')
        else:
            print(f"[WARN] �̹��� URL ���� ����. ������ ���ε��մϴ�. (path={local_image_path})")
    else:
        print("[INFO] ��ǥ �̹��� ��θ� ã�� ���߽��ϴ�. ������ ���ε��մϴ�.")

    # Firestore �߰�
    try:
        doc_ref = db.collection(FIREBASE_INVENTORY_COLLECTION).add(item_data_english_keys)
        print(f"[SUCCESS] Firestore '{FIREBASE_INVENTORY_COLLECTION}' �÷��ǿ� ������ �߰� �Ϸ� (Doc ID: {doc_ref[1].id})")
        # �ʿ� �� doc id ��ȯ:
        # return doc_ref[1].id
    except Exception as e:
        print(f"[ERROR] Firestore ������ �߰� �� ���� �߻�: {e}")


def upload_image_to_storage(local_image_path, destination_folder="registered_images",
                            make_public=True, signed_url_days=3650):
    """Cloud Storage�� �̹��� ���ε� �� ���� URL ��ȯ."""
    if not bucket:
        print("[ERROR] Storage ��Ŷ�� �ʱ�ȭ���� �ʾ� ���ε��� �� �����ϴ�.")
        return None

    if not os.path.exists(local_image_path):
        print(f"[ERROR] ���ε��� �̹��� ������ ã�� �� �����ϴ�: {local_image_path}")
        return None

    try:
        file_name = os.path.basename(local_image_path)
        blob_path = f"{destination_folder}/{file_name}".replace("//", "/")
        blob = bucket.blob(blob_path)

        print(f"[DEBUG] Storage ���ε� ����: bucket={bucket.name}, path={blob_path}")
        blob.upload_from_filename(local_image_path, content_type=_guess_content_type(file_name))

        # ĳ�� ���
        try:
            blob.cache_control = "public, max-age=31536000"
            blob.patch()
        except Exception:
            pass

        image_url = None

        # ���� ��Ŷ�� ���
        if make_public:
            try:
                blob.make_public()
                image_url = blob.public_url
            except Exception as e:
                print(f"[INFO] make_public ����(����� ��Ŷ�� �� ����): {e}")

        # ����� ��Ŷ�̸� ���� URL
        if not image_url:
            try:
                image_url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(days=signed_url_days),
                    method="GET",
                )
            except Exception as e:
                print(f"[ERROR] ���� URL ���� ����: {e}")
                image_url = None

        if image_url:
            print(f"[SUCCESS] Storage ���ε� �Ϸ�: {blob_path}")
            print(f"  - URL: {image_url}")
        else:
            print(f"[ERROR] ���ε�� �Ǿ����� ���� ������ URL ������ �����߽��ϴ�: {blob_path}")

        return image_url

    except Exception as e:
        print(f"[ERROR] Storage �̹��� ���ε� �� ���� �߻�: {e}")
        return None
