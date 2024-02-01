import cv2
import os
import numpy as np
import os.path as osp
import datetime
IMAGE_EXT = ['.jpg','.jpeg']
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1
    line_thickness = min(2, int(image.shape[1] / 500.))
#     text_scale = 2
#     text_thickness = 2
#     line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d  num: %d' % (frame_id, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

result_path = r'E:\Study\Lab\Tracking\BoT-SORT\YOLOX_outputs\yolox_x_mix_mot20_ch\track_vis'
test_name = 'MOT20-06'
result_file = osp.join(result_path, test_name + '-byte.txt')
print(result_file)
save_folder =r'E:\Study\Lab\Tracking\BoT-SORT\YOLOX_outputs\yolox_x_mix_mot20_ch\track_vis'
save_folder = osp.join(save_folder, test_name)
img_path = r'D:\Dataset\MOT20\test'
img_path = osp.join(img_path, test_name)
img_path = osp.join(img_path, 'img1')

img_file = get_image_list(img_path)
img_file.sort()
# result = [] # frame_id, track_id, tlwh, score, -1, -1, -1
# with open(result_file, 'r') as f:
#     for line in f:
#         result.append(np.asarray(line.split(',')[0:7]))
result = np.loadtxt(result_file, dtype=np.float64, delimiter=',')
print('result shape:',result.shape)
for frame_id,img_path in enumerate(img_file,1):
    im = cv2.imread(img_path)
    i_box = np.where(result[:,0]==frame_id)[0]
    online_tlwhs = result[i_box, 2:6]
    online_ids = result[i_box, 1] 
    online_im = plot_tracking(
    im, online_tlwhs, online_ids, frame_id=frame_id, fps=-1)
    os.makedirs(save_folder, exist_ok=True)
    cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)
    if frame_id % 200 == 0:
        print('{}/{}'.format(frame_id, len(img_file)))
#     print(osp.join(save_folder, osp.basename(img_path)))
#     if frame_id >18:
#         break
print(datetime.datetime.now())
    