import numpy as np
import tracker
from detector import Detector
import cv2

if __name__ == '__main__':
    mask_image_temp = np.zeros((352, 640), dtype=np.uint8)
    list_pts = [[130, 152], [440, 152], [640, 302], [640, 352], [0, 640], [0, 300], [0, 270]]  # 152为进入区域阈值
    ndarray_pts = np.array(list_pts, np.int32)
    polygon_area = cv2.fillPoly(mask_image_temp, [ndarray_pts], color=1)
    polygon_area = polygon_area[:, :, np.newaxis]
    color_plate = [123, 155, 98]
    green_image = np.array(polygon_area * color_plate, np.uint8)
    list_overlapping_green_polygon = []
    down_count = 0
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_position = (int(640 * 0.01), int(352 * 0.01))

    detector = Detector()
    draw_text_postion = (int(640 * 0.02), int(352 * 0.12))
    cap = cv2.VideoCapture('vehicles.mp4')
    while cap.isOpened():
        ret, im = cap.read()
        List = []
        count = 0
        if not ret:
            break
        list_bboxs = []
        bboxes = detector.detect(im)
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=1)
        else:
            output_image_frame = im
        output_image_frame = cv2.addWeighted(output_image_frame, 0.8, green_image, 0.2, 0)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1
                List.append(y)
            for i in List:
                if i > 152:
                    count += 1




        text_draw = 'Count: ' + str(count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 0, 0), thickness=1)
        cv2.imshow('im', output_image_frame)
        cv2.waitKey(10)
    cv2.destroyAllWindows()