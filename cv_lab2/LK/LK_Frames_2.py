import numpy as np
import cv2
import os

# 定义特征点检测参数
feature_params = dict(
    maxCorners=100,       # 最大角点数
    qualityLevel=0.3,     # 角点质量水平
    minDistance=7,        # 角点之间的最小距离
    blockSize=7           # 角点检测时考虑的邻域块大小
)

# 定义Lucas-Kanade光流参数
lk_params = dict(
    winSize=(15, 15),     # 光流计算时搜索窗口的大小
    maxLevel=2,           # 金字塔层数
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # 终止迭代条件
)

# 随机生成颜色，用于绘制轨迹
color = np.random.randint(0, 255, (100, 3))

# 图像路径
frame_paths = ["/home/lzy/Project2/data/Venus/frame10.png", "/home/lzy/Project2/data/Venus/frame11.png"]

# 读取两张图像
frames = []
for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"无法读取图像: {frame_path}")
        continue
    frames.append(frame)

# 确定保存图像的文件夹路径
save_folder = "/home/lzy/Project2/output/LK/Venus"

# 如果目标文件夹不存在，创建该文件夹
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 读取两张图像
old_frame = frames[0]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 检测特征点
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 创建一个掩模图像用于绘制轨迹（与第一帧大小相同）
mask = np.zeros_like(old_frame)

# 读取第二张图像
new_frame = frames[1]
new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

# 计算光流
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

# 选择好的特征点
if p1 is not None and st is not None:
    good_new = p1[st.flatten() == 1]
    good_old = p0[st.flatten() == 1]
else:
    good_new = np.array([])
    good_old = np.array([])

# 绘制轨迹
for j, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
    new_frame = cv2.circle(new_frame, (int(a), int(b)), 5, color[j].tolist(), -1)

# 合并轨迹和当前帧
img = cv2.add(new_frame, mask)

# 保存光流图像
save_path = os.path.join(save_folder, "optical_flow_frame10_to_11.png")
cv2.imwrite(save_path, img)

print(f"光流图像保存为: {save_path}")
