# tracker.py
# - 중심점(Centroid) + IoU 개념 기반 간단 추적기

import numpy as np
from scipy.spatial.distance import cdist

from config import *

class ObjectTracker:
    """중심점과 프레임 지속성으로 객체를 추적."""
    def __init__(self):
        self.objects = {}            # {obj_id: {'bbox': [x1,y1,x2,y2], 'centroid': (x,y), ...}}
        self.next_object_id = 0
        self.disappeared_frames = {}

    def register(self, bbox):
        """새 객체 등록 및 ID 부여."""
        obj_id = self.next_object_id
        centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.objects[obj_id] = {
            'bbox': bbox,
            'centroid': centroid,
            'name': 'Unknown',
            'stable_frames': 0,
            'is_new': True,
            'prev_bbox': None
        }
        self.disappeared_frames[obj_id] = 0
        self.next_object_id += 1
        return obj_id

    def deregister(self, object_id):
        """오래 사라진 객체 제거."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared_frames:
            del self.disappeared_frames[object_id]

    def update(self, detected_boxes):
        """
        현재 프레임의 탐지 박스로 추적 상태 갱신.
        detected_boxes: [[x1,y1,x2,y2], ...]
        """
        # 탐지 없음: 사라짐 카운트 증가 후 제거
        if not detected_boxes:
            for obj_id in list(self.disappeared_frames.keys()):
                self.disappeared_frames[obj_id] += 1
                if self.disappeared_frames[obj_id] > TRACKER_DISAPPEARED_THRESHOLD:
                    self.deregister(obj_id)
            return self.objects

        # 입력 중심점 계산
        input_centroids = np.zeros((len(detected_boxes), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detected_boxes):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        # 추적 중인 객체가 없으면 모두 등록
        if not self.objects:
            for box in detected_boxes:
                self.register(box)
            return self.objects

        # 기존 중심점과의 거리 행렬
        existing_ids = list(self.objects.keys())
        existing_centroids = np.array([data['centroid'] for data in self.objects.values()])
        D = cdist(existing_centroids, input_centroids)

        # 가장 가까운 쌍부터 매칭
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = existing_ids[row]
            self.objects[object_id]['prev_bbox'] = self.objects[object_id]['bbox']
            self.objects[object_id]['bbox'] = detected_boxes[col]
            self.objects[object_id]['centroid'] = tuple(input_centroids[col])
            self.disappeared_frames[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # 매칭되지 않은 기존 객체: 사라짐 증가
        unused_rows = set(range(0, D.shape[0])) - used_rows
        for row in unused_rows:
            object_id = existing_ids[row]
            self.disappeared_frames[object_id] += 1
            if self.disappeared_frames[object_id] > TRACKER_DISAPPEARED_THRESHOLD:
                self.deregister(object_id)

        # 매칭되지 않은 새 박스: 신규 등록
        unused_cols = set(range(0, D.shape[1])) - used_cols
        for col in unused_cols:
            self.register(detected_boxes[col])

        return self.objects
