import cv2
from my_utils import get_yolo_preds
with open("model_data/coco.names","r",encoding="utf-8") as f:
    labels= f.read().strip().split("\n")
print(labels)   
yolo_config_path="model_data/yolov3.cfg"
yolo_weights_path = "model_data/yolov3.weights"
input_vid_path=0
cuda = False
show_display=True
write_output = False
output_vid_path = "output_video/yolo_output.avi"
confidence_threshold = 0.50
overlapping_threshold = 0.3
net = cv2.dnn.readNetFromDarknet(yolo_config_path,yolo_weights_path)
if __name__=="__main__":
    get_yolo_preds(net, input_vid_path, output_vid_path, confidence_threshold, overlapping_threshold,write_output, show_display, labels)
