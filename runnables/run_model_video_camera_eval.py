from models.utils import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from Project import project
from models.utils import Timer
import cv2
import sys
import torch
from thop.profile import profile
import psutil
import os

model_path = project.pruned_model_dir / '2020-02-25-14-03.pth'
label_path = project.trained_model_dir / 'voc-model-labels.txt'

cap = cv2.VideoCapture(0)  # capture from camera

cap.set(3, 480)
cap.set(4, 640)
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)

net.load(model_path)
net = net.cpu()
print(net)
print("*" * 80)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
print("*" * 80)
print(predictor)

cnt = 0
timeperframe = 0
fps = 0
timer = Timer()

#macs

dsize = (1, 3, 300, 300)
inputs = torch.randn(dsize)
net = net.to(torch.device('cpu'))
total_ops, total_params = profile(
    net,
    (inputs, ),
)
print("%s | %s | %s" % ("Model", "Params(M)", "MACs(G)"))
print(
    "%s | %.2f | %.2f" %
    ("ssd", total_params / (1000**2), total_ops / (1000**3))
)

while True:
    timer.start()
    ret, orig_image = cap.read()
    readTime = timer.end()
    cnt += 1
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()

    print(
        'Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0))
    )
    timer.start()
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

        labelfps = f"FPS : {fps:.2f}"
        cv2.putText(
            orig_image, labelfps, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 0, 0), 2
        )

    cv2.imshow('annotated', orig_image)
    bbplotTime = timer.end()

    # fps
    timeperframe += readTime + interval + bbplotTime
    if cnt == 10:
        avgtime = timeperframe / 10
        fps = 1 / avgtime
        # print('FPS : {:.2f}s'.format(fps))
        cnt = 0
        timeperframe = 0

    # memory usage
    pid = os.getpid()
    # print(pid)
    mem = psutil.Process(pid).memory_info()
    total = mem.rss / (1024.0**2)
    print('Memory used : {:.2f} MB '.format(total))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
