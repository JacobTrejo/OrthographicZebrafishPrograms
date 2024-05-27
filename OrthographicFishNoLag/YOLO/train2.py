#from ultralytics import YOLO
#from ultralytics2.ultralytics.models.yolo.model import YOLO
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
#model = YOLO('grayYOLOPose.yaml')

#model.train( data = 'config2.yaml', device =(0, 1),epochs = 100, mosaic = 0.0, translate = 0, scale = 0, hsv_h =0, hsv_s = 0, hsv_v = 0, degrees = 0, shear = 0 , perspective = 0, flipud = 0)

model.train( data = 'config2.yaml', device = (0, 1), epochs = 100, mosaic = 0.0, translate = 0, scale = 0, hsv_h =0, hsv_s = 0, hsv_v = 0, degrees = 0, shear = 0 , perspective = 0, flipud = 0)

#model = YOLO( 'runs/pose/train4/weights/last.pt'  )
#model.train( resume=True, device = (0,1),epochs = 100 )
