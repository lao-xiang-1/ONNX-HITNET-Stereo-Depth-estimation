import cv2
import numpy as np
# from imread_from_url import imread_from_url # 已经不需要了

from hitnet import HitNet, ModelType, CameraConfig

# ==========================================
# 1. 设置双目相机的真实参数 (需要替换为你标定后的数据)
# ==========================================
# 注意：焦距 f 通常是内参矩阵中的 fx。
FOCAL_LENGTH_PX =554.5482  # 焦距 (像素单位) - 请替换为你的真实焦距
BASELINE_MM = 79.6753# 基线长度 (毫米) - 请替换为你的真实基距

camera_config = CameraConfig(FOCAL_LENGTH_PX / 1000, BASELINE_MM) # rough estimate from the original calibration


# Select model type 
model_type = ModelType.middlebury

# 为了防止路径报错，我建议在路径前统一加上 'r' (Raw String)
if model_type == ModelType.middlebury:
    model_path = r"model_float32.onnx"
elif model_type == ModelType.flyingthings:
    model_path = r"models/flyingthings_finalpass_xl/saved_model_480x640/model_float32.onnx"
elif model_type == ModelType.eth3d:
    model_path = r"models/eth3d/saved_model_480x640/model_float32.onnx"

# Initialize model
depth_estimator = HitNet(model_path, model_type, camera_config=camera_config)

# Load images
left_img = cv2.imread(r"./left_rectified/1.jpg")
right_img = cv2.imread(r"./right_rectified/1.jpg")

# ==========================================
# 2. 核心计算：估算视差并转换为物理深度
# ==========================================
# 估算视差 (单位：像素)
disparity_map = depth_estimator(left_img, right_img)

# 将视差转换为深度 ( Z = f * B / d )
# 警告：为了防止除以 0 导致程序崩溃，我们需要把视差中小于等于 0 的无效值替换为一个极小数
disparity_safe = np.where(disparity_map <= 0, 0.1, disparity_map)
depth_map = (FOCAL_LENGTH_PX * BASELINE_MM) / disparity_safe

# 提取彩色视差图用于可视化展示
color_disparity = depth_estimator.draw_disparity()
combined_image = np.hstack((left_img, color_disparity))

# ==========================================
# 3. 互动功能：鼠标点击获取真实距离 (已修复坐标映射)
# ==========================================
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取显示图像（大图）和 数据矩阵（小图）的尺寸
        img_h, img_w = left_img.shape[:2]
        disp_h, disp_w = disparity_map.shape[:2]

        # 判断点击的是左边(原图)还是右边(视差图)，换算真实的 x 坐标
        actual_x = x if x < img_w else x - img_w
        
        # 越界保护：防止点击到窗口空白处报错
        if actual_x < 0 or actual_x >= img_w or y < 0 or y >= img_h:
            return

        # 🎯 核心修复：按比例映射坐标！
        # 将屏幕点击的大图坐标，映射到神经网络输出的小数据矩阵坐标上
        map_x = int(actual_x * (disp_w / img_w))
        map_y = int(y * (disp_h / img_h))
        
        # 提取数值 (现在绝对安全了)
        disp_val = disparity_map[map_y, map_x]
        dist_val = depth_map[map_y, map_x]
        
        # 打印结果
        if disp_val > 0:
            print(f"📍 图像坐标: ({actual_x}, {y}) | 映射矩阵坐标: ({map_x}, {map_y}) | 视差: {disp_val:.2f} | 距离: {dist_val:.2f} mm")
        else:
            print(f"📍 图像坐标: ({actual_x}, {y}) | 视差无效，无法测距")

# 创建窗口并绑定鼠标回调函数
cv2.namedWindow("Stereo Depth Measurement", cv2.WINDOW_NORMAL)   
cv2.setMouseCallback("Stereo Depth Measurement", mouse_callback)

# 显示画面
print("✅ 测距程序已启动！请在弹出的图像上点击鼠标左键查看距离。")
cv2.imshow("Stereo Depth Measurement", combined_image)
cv2.imwrite("out_with_depth.jpg", combined_image)

cv2.waitKey(0)
cv2.destroyAllWindows()