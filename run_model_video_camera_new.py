import cv2
import time
from Project import project
from models.utils import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)

from quantization.quantization import ModelQuantizer

model_path = project.quantized_trained_model_dir / '2020-06-10-14-42.pth'
label_path = project.trained_model_dir / 'voc-model-labels.txt'

cap = cv2.VideoCapture(0)  # capture from camera

cap.set(3, 480)
cap.set(4, 640)
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)




net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
model_quantizer = ModelQuantizer()
print("Quantinzed the vanilla model to get the correct definition")
model_quantizer.quantize(net)
print("Loading the quantized weights")
# q_model.load(project.quantized_trained_model_dir / file_name)
net.load(model_path)



print(net)
print("*" * 80)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
print("*" * 80)
print(predictor)

start = time.time()
current_frame = 0
while True:
    ret, orig_image = cap.read()
    num_frames = 120
    print("Capturing {0} frames".format(num_frames))

    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    # timer.start()
    start_prediction_time = time.time()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    end_prediction_time = time.time()
    print(
        f"Time for prediction is {end_prediction_time - start_prediction_time}"
    )
    # print(
    #     'Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0))
    # )
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(
            orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4
        )

        cv2.putText(
            orig_image,
            label,
            (box[0] + 20, box[1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # font scale
            (255, 0, 255),
            2
        )  # line type
    cv2.imshow('annotated', orig_image)
    current_frame += 1
    print(current_frame)
    if current_frame == num_frames:
        end = time.time()
        seconds = end - start
        print("Time taken : {0} seconds".format(seconds))
        fps = num_frames / seconds
        print("Estimated frames per second : {0}".format(fps))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
