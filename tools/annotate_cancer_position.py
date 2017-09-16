import cv2
import os


x1 = 0
x2 = 0
y1 = 0
y2 = 0
status = 0


def get_position(event, x, y, flags, param):
    global status, x1, x2, y1, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        if status == 0:
            x1 = x
        elif status == 1:
            x2 = x
        elif status == 2:
            y1 = y
        else:
            y2 = y
        status = (status+1) % 4

image_dir = '/home/dujiajun/thyroid nodule/benign tumour'
image_list = os.listdir(image_dir)
image_list.sort()
for image_name in image_list:
    status = 0
    cv2.namedWindow(image_name)
    cv2.setMouseCallback(image_name, get_position)
    cv2.moveWindow(image_name, 300, 300)
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

    cv2.namedWindow(image_name)
    cv2.moveWindow(image_name, 300, 300)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    print('{},{},{},{},{}'.format(image_name, x1, x2, y1, y2))
