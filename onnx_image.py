import cv2
import os
import time
from hybridnets import HybridNets, optimized_model

envpath = '/root/miniconda3/lib/python3.9/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

model_path = "models/hybridnets_384x512/hybridnets_384x512.onnx"
anchor_path = "models/hybridnets_384x512/anchors_384x512.npy"

# Initialize road detector
optimized_model(model_path) # Remove unused nodes
roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

img = cv2.imread("data/test-demo.jpg")

# Update road detector
seg_map, filtered_boxes, filtered_scores = roadEstimator(img)
time_per = time.time()
combined_img = roadEstimator.draw_2D(img)
cv2.imwrite("out/cpu_test.jpg", combined_img)
cv2.waitKey(0)
print(time.time() - time_per)
