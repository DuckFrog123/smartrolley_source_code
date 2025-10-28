# utils.py
# - 공통 유틸: 영상 처리, IoU, ToF 히트맵, 로깅, 간단 팝업

import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
import os

from config import *

def to_square(frame, size=CAMERA_SQUARE_SIZE):
    """이미지를 중앙 기준 정사각형으로 자르고 리사이즈."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    square = frame[y0:y0 + side, x0:x0 + side]
    if size and (square.shape[0] != size):
        square = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return square

def calculate_iou(boxA, boxB):
    """두 박스의 IoU 계산."""
    if not all(isinstance(c, (int, float)) for c in boxA) or not all(isinstance(c, (int, float)) for c in boxB):
        return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def draw_tof_heatmap(depth_map):
    """8x8 깊이 맵을 컬러 히트맵으로 시각화(손=적계열, 선반=Viridis, 기타=회색)."""
    if depth_map is None:
        return np.zeros((200, 200, 3), dtype=np.uint8)

    depth = np.array(depth_map, dtype=np.float32).reshape(8, 8)
    depth = np.nan_to_num(depth, nan=4000.0)

    heatmap_vis = np.zeros((8, 8, 3), dtype=np.uint8)

    # 손(300~1200mm): 빨간색 계열(가까울수록 밝게)
    hand_mask = (depth >= 300) & (depth < 1200)
    normalized_hand = np.clip((depth[hand_mask] - 300) / (1200 - 300), 0, 1)
    heatmap_vis[hand_mask] = np.column_stack((
        np.zeros_like(normalized_hand),
        np.zeros_like(normalized_hand),
        (1 - normalized_hand) * 200 + 55
    )).astype(np.uint8)

    # 선반(1200~2200mm): Viridis
    shelf_mask = (depth >= 1200) & (depth < 2200)
    normalized_shelf = np.clip((depth[shelf_mask] - 1200) / (2200 - 1200), 0, 1) * 255
    shelf_colors = cv2.applyColorMap(normalized_shelf.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap_vis[shelf_mask] = shelf_colors.reshape(-1, 3)

    # 기타: 회색
    other_mask = ~hand_mask & ~shelf_mask
    heatmap_vis[other_mask] = [100, 100, 100]

    return cv2.resize(heatmap_vis, (200, 200), interpolation=cv2.INTER_NEAREST)

def log_event(unique_object_name, event_type):
    """객체의 IN/OUT 이벤트를 CSV에 기록."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp},{unique_object_name},{event_type}\n"
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding='utf-8') as f:
            f.write("Timestamp,Object,Event\n")
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(log_entry)
    print(f"[LOG] {unique_object_name} - {event_type}")

def get_name_with_timeout(title, prompt, db, timeout=NAME_INPUT_TIMEOUT_SECONDS):
    """타임아웃 있는 이름 입력 팝업. 시간 초과 시 unknown-n 자동 부여."""
    result = {"name": None}

    def on_ok(event=None):
        result["name"] = entry.get() or None
        root.destroy()

    def on_timeout():
        i = 1
        while True:
            temp_name = f"unknown-{i}"
            if temp_name not in db:
                result["name"] = temp_name
                break
            i += 1
        root.destroy()

    def on_cancel():
        result["name"] = None
        root.destroy()

    root = tk.Tk()
    root.title(title)

    window_width, window_height = 350, 130
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    root.attributes('-topmost', True)

    tk.Label(root, text=prompt, pady=10).pack()
    entry = tk.Entry(root, width=40); entry.pack(); entry.focus_set()

    btn_frame = tk.Frame(root, pady=10)
    tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side=tk.LEFT, padx=10)
    btn_frame.pack()

    root.bind('<Return>', on_ok)
    root.after(timeout * 1000, on_timeout)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()

    return result["name"]

def get_yes_no_popup(title, prompt, timeout=NAME_INPUT_TIMEOUT_SECONDS):
    """예/아니오 팝업. 타임아웃 시 '예' 반환."""
    result = {"answer": True}

    def on_yes():
        result["answer"] = True
        root.destroy()

    def on_no():
        result["answer"] = False
        root.destroy()

    root = tk.Tk()
    root.title(title)

    window_width, window_height = 350, 100
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    root.attributes('-topmost', True)

    tk.Label(root, text=prompt, pady=10).pack()

    btn_frame = tk.Frame(root, pady=10)
    tk.Button(btn_frame, text="예 (Yes)", width=10, command=on_yes).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="아니오 (No)", width=10, command=on_no).pack(side=tk.LEFT, padx=10)
    btn_frame.pack()

    root.after(timeout * 1000, on_yes)
    root.protocol("WM_DELETE_WINDOW", on_no)  # 창 닫기 = '아니오'
    root.mainloop()

    return result["answer"]
