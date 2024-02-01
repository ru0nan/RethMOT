import numpy as np
import sys
import os.path as osp
import os
import cv2
sys.path.append('.')
from tracker.matching import ious


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

def calcu_stage(ious_mat, bboxes, scores):
    degree_mat =  np.zeros_like(ious_mat)
    inds =  np.where(ious_mat > 0.2) #(row_idx array, colum_idx array)
    for i, j in zip(inds[0], inds[1]):
        diff = bboxes[i,3] - bboxes[j,3]
        h1 = bboxes[i,3]-bboxes[i,1]
        h2 = bboxes[j,3]-bboxes[j,1]
        if abs(diff) > min(h1, h2)*0.1:
            if diff > 0:
                degree_mat[i, j] = 1
            else:
                degree_mat[i, j] = -1
        else:
            s_diff = scores[i]-scores[j]
            if s_diff > 0.35:
                degree_mat[i, j] = 1
            if s_diff < -0.35:
                degree_mat[i, j] = -1
    count = degree_mat == -1
    stage = count.sum(axis=0)
    return stage

def ByteTrack_sep(image, boxes_tlbr, scores, thresh=0.6):
    im = np.ascontiguousarray(np.copy(image))
    line_thickness = min(2, int(image.shape[1] / 500.))

    first_stage = np.where(scores>=thresh)
    second_stage = np.where(scores<thresh)
    first_boxes = boxes_tlbr[first_stage[0],:]
    second_boxes = boxes_tlbr[second_stage[0],:]
    print(first_boxes.shape, second_boxes.shape)
    for i, tlbr in enumerate(first_boxes):
        x1, y1, x2, y2 = tlbr
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        color =  (0, 0, 225)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    for i, tlbr in enumerate(second_boxes):
        x1, y1, x2, y2 = tlbr
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        color =  (255, 0, 0)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
       
    return im

def Salient_SORT_sep(image, boxes_tlbr, scores, thresh=0.6):
    im = np.ascontiguousarray(np.copy(image))
    line_thickness = min(2, int(image.shape[1] / 500.))
    iou_mat = ious(boxes_tlbr, boxes_tlbr) - np.eye(boxes_tlbr.shape[0])
    stage = calcu_stage(iou_mat, boxes_tlbr,scores)
    scores_fuse = 0.7 * scores + 0.3* np.exp(-stage)

    first_stage = np.where(scores_fuse>=thresh)
    second_stage = np.where(scores_fuse<thresh)
    first_boxes = boxes_tlbr[first_stage[0],:]
    second_boxes = boxes_tlbr[second_stage[0],:]
    print(first_boxes.shape, second_boxes.shape)
    for i, tlbr in enumerate(first_boxes):
        x1, y1, x2, y2 = tlbr
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        color =  (0, 0, 225)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    for i, tlbr in enumerate(second_boxes):
        x1, y1, x2, y2 = tlbr
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        color =  (255, 0, 0)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    return im

result_path = r'E:\Study\Lab\Tracking\BoT-SORT\YOLOX_outputs\yolox_x_mix_mot20_ch\track_vis'
test_name = 'MOT20-08'
result_file = osp.join(result_path, test_name + '.txt')
print(result_file)
save_folder =r'E:\Study\Lab\Tracking\BoT-SORT\YOLOX_outputs\paper_pictures'
save_folder = osp.join(save_folder, test_name)
img_path = r'D:\Dataset\MOT20\test'
img_path = osp.join(img_path, test_name)
img_path = osp.join(img_path, 'img1')

img_file = get_image_list(img_path)
img_file.sort()
result = [] # frame_id, track_id, tlwh, score, -1, -1, -1
with open(result_file, 'r') as f:
    for line in f:
        result.append(np.asarray(line.split(',')[0:7]))
result = np.array(result,dtype = np.float32)
print('result shape:',result.shape)

for frame_id,img_path in enumerate(img_file,1):
    if frame_id <10 :
        im = cv2.imread(img_path)
        i_box = np.where(result[:,0]==frame_id)[0]
        online_tlwhs = result[i_box, 2:6] #tlwh
        online_tlbrs = online_tlwhs
        online_tlbrs[:,2:4] += online_tlwhs[:,0:2] #tlbr
        online_scores = result[i_box, 6]

        online_im = ByteTrack_sep(im, online_tlbrs, online_scores,0.6)
        os.makedirs(save_folder, exist_ok=True)
        base_name  ='ByteTrack'+ osp.basename(img_path)
        cv2.imwrite(osp.join(save_folder, base_name), online_im)
        
        # online_im = Salient_SORT_sep(im,online_tlbrs, online_scores,0.6)
        # base_name  ='Salient'+ osp.basename(img_path)
        # cv2.imwrite(osp.join(save_folder, base_name), online_im)
        print('---------')

    else:
        continue
        


