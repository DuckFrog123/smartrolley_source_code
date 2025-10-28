#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# camset.py
# - Raspberry Pi + OpenCV 카메라 프리뷰
# - 's': 화면 속 작은 흰 사각형 4개를 모서리로 잡아 투시변환
# - 'd': 디버그(이진화/윤곽) 보기 토글
# - 'a': ROI만 표시 토글
# - 'q': 종료

import cv2
import numpy as np
import math
from collections import namedtuple

CornerSquares = namedtuple('CornerSquares', ['pts', 'centers', 'boxes'])

# ======== 기본 파라미터 ========
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# 흰 사각형 검출
BLUR_KSIZE = 5          # 가우시안 블러 커널(홀수)
THRESH_VAL = 210        # 밝기 임계값(0~255)
MIN_AREA = 60           # 최소 면적
MAX_AREA = 20000        # 최대 면적
ASPECT_TOL = 0.25       # 정사각형 비율 허용오차
APPROX_EPS = 0.02       # 다각형 근사 비율
MORPH_OPEN = 3          # 모폴로지 오프닝 커널
MORPH_CLOSE = 3         # 모폴로지 클로징 커널

# 투시변환 초기값(자동 계산됨)
WARP_W = 800
WARP_H = 500
# ===============================

def order_corners(pts):
    """4점(TL, TR, BR, BL) 순으로 정렬."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)           # x+y
    d = np.diff(pts, axis=1)[:,0] # x-y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def is_square(cnt, approx, area):
    """근사 다각형이 '거의 정사각형'인지 검사."""
    if len(approx) != 4:
        return False
    if area < MIN_AREA or area > MAX_AREA:
        return False

    x, y, w, h = cv2.boundingRect(approx)
    if w == 0 or h == 0:
        return False
    ratio = w / float(h)
    if abs(ratio - 1.0) > ASPECT_TOL:
        return False

    # 직각 근사 여부(코사인 값으로 체크)
    def angle_cos(p0, p1, p2):
        v1 = (p0 - p1).astype(np.float32)
        v2 = (p2 - p1).astype(np.float32)
        cosang = abs(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6))
        return cosang

    max_cos = 0
    pts = approx.reshape(-1, 2)
    for i in range(4):
        p0 = pts[(i-1) % 4]
        p1 = pts[i]
        p2 = pts[(i+1) % 4]
        max_cos = max(max_cos, angle_cos(p0, p1, p2))
    return max_cos < 0.25

def find_white_squares(frame, debug=False):
    """프레임에서 작은 흰 사각형 후보를 탐지."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if BLUR_KSIZE >= 3 and BLUR_KSIZE % 2 == 1:
        gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # 밝은 영역 이진화
    _, bin_img = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY)

    # 모폴로지 정리
    if MORPH_OPEN >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_OPEN, MORPH_OPEN))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    if MORPH_CLOSE >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE, MORPH_CLOSE))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    centers = []
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS * peri, True)
        if not cv2.isContourConvex(approx):
            continue

        if is_square(cnt, approx, area):
            M = cv2.moments(approx)
            if M['m00'] == 0:
                continue
            cx = float(M['m10'] / M['m00'])
            cy = float(M['m01'] / M['m00'])
            squares.append(approx.reshape(-1, 2))
            centers.append((cx, cy))
            boxes.append(cv2.boundingRect(approx))

    if debug:
        dbg = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        for sq in squares:
            cv2.polylines(dbg, [sq.astype(np.int32)], True, (0, 255, 0), 2)
        return squares, centers, boxes, dbg
    else:
        return squares, centers, boxes, None

def pick_four_corners(centers, squares):
    """검출된 중심점들에서 4개 모서리를 선택."""
    if len(centers) < 4:
        return None

    pts = np.array(centers, dtype=np.float32)
    hull_idx = cv2.convexHull(pts, returnPoints=False).flatten()
    hull_pts = pts[hull_idx]

    if len(hull_pts) < 4:
        # 사분면 기반 대체 선택
        cx, cy = pts.mean(axis=0)
        quadrants = { 'tl': None, 'tr': None, 'br': None, 'bl': None }
        for i, (x, y) in enumerate(pts):
            key = None
            if x < cx and y < cy: key = 'tl'
            elif x > cx and y < cy: key = 'tr'
            elif x > cx and y > cy: key = 'br'
            elif x < cx and y > cy: key = 'bl'
            if key is not None and quadrants[key] is None:
                quadrants[key] = (x, y, i)
        sel = [quadrants[k] for k in ['tl', 'tr', 'br', 'bl'] if quadrants[k] is not None]
        if len(sel) != 4:
            return None
        chosen_idx = [t[2] for t in sel]
    else:
        # hull 사각 경계와 가까운 4점 선택
        x_min, y_min = np.min(hull_pts, axis=0)
        x_max, y_max = np.max(hull_pts, axis=0)
        rect_corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ], dtype=np.float32)

        chosen_idx = []
        hull_list = hull_pts.tolist()
        for rc in rect_corners:
            dists = [np.linalg.norm(rc - h) for h in hull_list]
            j = int(np.argmin(dists))
            chosen = hull_list.pop(j)
            orig_idx = int(np.where((pts == chosen).all(axis=1))[0][0])
            chosen_idx.append(orig_idx)

        if len(chosen_idx) != 4:
            return None

    chosen_centers = np.array([centers[i] for i in chosen_idx], dtype=np.float32)
    ordered = order_corners(chosen_centers)

    # 중심점을 대표 모서리로 사용(작은 사각형이 네 모서리에 충분히 가깝다는 가정)
    return ordered

def compute_warp_size(corners):
    """모서리 4점으로 타겟 크기 산출."""
    tl, tr, br, bl = corners
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    W = int(max(w_top, w_bot))
    H = int(max(h_left, h_right))
    W = max(W, 200)
    H = max(H, 150)
    return W, H

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    auto_mode = False   # True면 ROI만 표시
    debug_view = False
    corners_cached = None

    print("사용법: s=사각형 탐색 / d=디버그 토글 / a=ROI 보기 토글 / q=종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        display = frame.copy()

        if corners_cached is not None:
            corners = corners_cached.copy()
            W, H = compute_warp_size(corners)
            dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(corners, dst)
            warped = cv2.warpPerspective(frame, M, (W, H))

            if auto_mode:
                cv2.imshow("ROI (Warped)", warped)
            else:
                pts = corners.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                cv2.imshow("ROI (Warped)", warped)

        cv2.imshow("Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_view = not debug_view
            print(f"[디버그] 보기: {debug_view}")
        elif key == ord('a'):
            auto_mode = not auto_mode
            print(f"[표시 모드] ROI만 보기: {auto_mode}")
        elif key == ord('s'):
            squares, centers, boxes, dbg = find_white_squares(frame, debug=debug_view)

            if debug_view and dbg is not None:
                cv2.imshow("Debug(binary+contours)", dbg)

            if len(centers) < 4:
                print("⚠️ 흰 사각형이 4개 미만입니다. 조명/임계값(THRESH_VAL) 조정해 보세요.")
                corners_cached = None
                continue

            ordered_corners = pick_four_corners(centers, squares)
            if ordered_corners is None:
                print("⚠️ 모서리 4점을 안정적으로 선택하지 못했습니다. 배치를 더 네 모서리 형태로 만들어 보세요.")
                corners_cached = None
                continue

            corners_cached = ordered_corners
            print("✅ 4 모서리 확정:", corners_cached.astype(int).tolist())

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
