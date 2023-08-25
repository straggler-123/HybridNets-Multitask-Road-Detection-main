import cv2
import time
import sys
import os
sys.path.append("./common/acllite")
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
from utils import *

MODEL_WIDTH = 384
MODEL_HEIGHT = 512


class HybridNets():

	def __init__(self, model_path, anchor_path, conf_thres=0.5, iou_thres=0.5):

		# Initialize model
		self.initialize_model(model_path, anchor_path, conf_thres, iou_thres)

	def __call__(self, image):
		return self.estimate_road(image)

	def initialize_model(self, model_path, anchor_path, conf_thres=0.5, iou_thres=0.5):

		self.model = AclLiteModel(model_path)
		self.conf_thres = conf_thres
		self.iou_thres = iou_thres

		# Read the anchors from the file
		self.anchors = np.squeeze(np.load(anchor_path))

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_road(self, picPath):

		input_img = self.prepare_input(picPath)

		# Perform inference on the image
		outputs = self.inference(input_img)

		# Process output data
		self.seg_map, self.filtered_boxes, self.filtered_scores = self.process_output(outputs)

		return self.seg_map, self.filtered_boxes, self.filtered_scores

	def prepare_input(self, image):
		# get img shape
		start = time.time()
		self.img_height, self.img_width = image.shape[:2]

		rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# resize img
		img = cv2.resize(rgb_img, (self.input_width,self.input_height))
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		input_img = ((img / 255.0 - mean) / std)
		input_img = input_img.transpose(2, 0, 1)
		input_tensor = input_img[np.newaxis, :, :, :].astype(np.float16)
		print(time.time() - start)
		return input_tensor


	def inference(self, l_data):

		# start = time.time()
		outputs = self.model.execute([l_data])
		# print(time.time() - start)
		return outputs

	def process_output(self, outputs):

		# Process segmentation map
		seg_map = np.squeeze(np.argmax(outputs[self.output_names.index("segmentation")], axis=1))

		# Process detections
		scores = np.squeeze(outputs[self.output_names.index("classification")])
		boxes = np.squeeze(outputs[self.output_names.index("regression")])

		filtered_boxes, filtered_scores =  self.process_detections(scores, boxes)

		return seg_map, filtered_boxes, filtered_scores

	def process_detections(self, scores, boxes):

		transformed_boxes = transform_boxes(boxes, self.anchors)

		# Filter out low score detections
		filtered_boxes = transformed_boxes[scores>self.conf_thres]
		filtered_scores = scores[scores>self.conf_thres]

		# Resize the boxes with image size
		filtered_boxes[:,[0,2]] *= self.img_width/self.input_width
		filtered_boxes[:,[1,3]] *= self.img_height/self.input_height

		# Perform nms filtering
		filtered_boxes, filtered_scores = nms_fast(filtered_boxes, filtered_scores, self.iou_thres)

		return filtered_boxes, filtered_scores


	def draw_segmentation(self, image, alpha = 0.5):

		return util_draw_seg(self.seg_map, image, alpha)

	def draw_boxes(self, image, text=True):

		return util_draw_detections(self.filtered_boxes, self.filtered_scores, image, text)

	def draw_2D(self, image, alpha = 0.5, text=True):
		front_view = self.draw_segmentation(image, alpha)
		return self.draw_boxes(front_view, text)

	def draw_bird_eye(self, image, horizon_points):

		seg_map = self.draw_2D(image, 0.00001, text=False)
		return util_draw_bird_eye_view(seg_map, horizon_points)

	def draw_all(self, image, horizon_points, alpha = 0.5):

		front_view = self.draw_segmentation(image, alpha)
		front_view = self.draw_boxes(front_view)

		bird_eye_view = self.draw_bird_eye(image, horizon_points)

		combined_img = np.hstack((front_view, bird_eye_view))

		return combined_img

	def get_input_details(self):
		self.input_names = ['input']
		self.input_shape = [1, 3, 384, 512]
		self.input_height = int(384)
		self.input_width = int(512)

	def get_output_details(self):
		self.output_names = ['regression', 'classification', 'segmentation']



if __name__ == '__main__':
	start_total = time.time()
	# acl init
	acl_resource = AclLiteResource()
	acl_resource.init()
	model_path = 'models/om_hybird_384_512.om'
	anchor_path = "models/hybridnets_384x512/anchors_384x512.npy"
	roadEstimator = HybridNets(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)
	frame_count = 0
	#video_processed
	video_path = "data/test-demo.mp4"
	output_path = "./out"
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# create output directory
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	output_Video = os.path.basename(video_path)
	output_Video = os.path.join(output_path, output_Video)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # DIVX, XVID, MJPG, X264, WMV1, WMV2
	outVideo = cv2.VideoWriter(output_Video, fourcc, fps, (Width, Height))

	# Read until video is completed
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			seg_map, filtered_boxes, filtered_scores = roadEstimator(frame)
			time_per = time.time()
			combined_img = roadEstimator.draw_2D(frame)
			outVideo.write(combined_img)
			print("FINISH PROCESSING FRAME: ", frame_count)
			frame_count += 1
			# print(time.time() - time_per)
			print('\n')
		# Break the loop
		else:
			break
	cap.release()
	outVideo.release()
	# print(time.time() - start_total)
	print("Execute end")








