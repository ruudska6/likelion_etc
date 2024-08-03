import cv2
import numpy as np

# 동영상 파일 설정
filename = 'black_screen.mp4'
fps = 30
duration = 3  # 초

# 프레임 크기 설정 (640x480)
width, height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

# 검정색 화면 생성
black_frame = np.zeros((height, width, 3), dtype=np.uint8)

for _ in range(fps * duration):
    out.write(black_frame)

out.release()
print(f"동영상 파일 '{filename}' 이(가) 생성되었습니다.")
