# tracker.py
# - �߽���(Centroid) + IoU ���� ��� ���� ������

import numpy as np
from scipy.spatial.distance import cdist

from config import *

class ObjectTracker:
    """�߽����� ������ ���Ӽ����� ��ü�� ����."""
    def __init__(self):
        self.objects = {}            # {obj_id: {'bbox': [x1,y1,x2,y2], 'centroid': (x,y), ...}}
        self.next_object_id = 0
        self.disappeared_frames = {}

    def register(self, bbox):
        """�� ��ü ��� �� ID �ο�."""
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
        """���� ����� ��ü ����."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared_frames:
            del self.disappeared_frames[object_id]

    def update(self, detected_boxes):
        """
        ���� �������� Ž�� �ڽ��� ���� ���� ����.
        detected_boxes: [[x1,y1,x2,y2], ...]
        """
        # Ž�� ����: ����� ī��Ʈ ���� �� ����
        if not detected_boxes:
            for obj_id in list(self.disappeared_frames.keys()):
                self.disappeared_frames[obj_id] += 1
                if self.disappeared_frames[obj_id] > TRACKER_DISAPPEARED_THRESHOLD:
                    self.deregister(obj_id)
            return self.objects

        # �Է� �߽��� ���
        input_centroids = np.zeros((len(detected_boxes), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detected_boxes):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        # ���� ���� ��ü�� ������ ��� ���
        if not self.objects:
            for box in detected_boxes:
                self.register(box)
            return self.objects

        # ���� �߽������� �Ÿ� ���
        existing_ids = list(self.objects.keys())
        existing_centroids = np.array([data['centroid'] for data in self.objects.values()])
        D = cdist(existing_centroids, input_centroids)

        # ���� ����� �ֺ��� ��Ī
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

        # ��Ī���� ���� ���� ��ü: ����� ����
        unused_rows = set(range(0, D.shape[0])) - used_rows
        for row in unused_rows:
            object_id = existing_ids[row]
            self.disappeared_frames[object_id] += 1
            if self.disappeared_frames[object_id] > TRACKER_DISAPPEARED_THRESHOLD:
                self.deregister(object_id)

        # ��Ī���� ���� �� �ڽ�: �ű� ���
        unused_cols = set(range(0, D.shape[1])) - used_cols
        for col in unused_cols:
            self.register(detected_boxes[col])

        return self.objects
