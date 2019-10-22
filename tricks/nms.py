import torch
import torchvision

# torchvision.ops下（pytorch>=1.2.0， torchvision >= 0.3）

#NMS
### torchvision.ops.nms(boxes, scores, iou_threshold)
# boxes (Tensor[N, 4])) – bounding boxes坐标. 格式：(x1, y1, x2, y2)
# scores (Tensor[N]) – bounding boxes得分
# iou_threshold (float) – IoU过滤阈值
# 返回值：keep :NMS过滤后的bouding boxes索引（降序排列）

#RoIAlign ==> Mask RCNN
### torchvision.ops.roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1)

#RoIPool ==>Fast RCNN
### torchvision.ops.roi_pool(input, boxes, output_size, spatial_scale=1.0)


#1，按打分最高到最低将BBox排序 ，例如：A B C D E F
#2，A的分数最高，保留。从B-E与A分别求重叠率IoU，假设B、D与A的IoU大于阈值，那么B和D可以认为是重复标记去除
#3，余下C E F，重复前面两步。
# coding:utf-8

import numpy as np

def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    # 按照score的置信度将其排序,argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
    order = scores.argsort()[::-1]
    # 计算面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 保留最后需要保留的边框的索引
    keep = []
    while order.size > 0:
        # order[0]是目前置信度最大的，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他窗口的交叠的面积，此处的maximum是np中的广播机制
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        # 计算相交框的面积,左上右下，画图理解。注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算IOU：相交的面积/相并的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr < thresh)[0]  # np.where就可以得到索引值(3,0,8)之类的，再取第一个索引
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1（因为计算inter时是少了1的），所以要把这个1加回来
        print(inds)
        order = order[inds + 1]

    return keep


def softnms(dets,thresh):
    keep=[]
    # 将不是去掉，而是抑制
    return keep

# test
if __name__ == "__main__":
    dets = np.array([[30, 20, 230, 200, 1],
                     [50, 50, 260, 220, 0.9],
                     [210, 30, 420, 5, 0.8],
                     [430, 280, 460, 360, 0.7]])
    thresh = 0.35
    keep_dets = nms(dets, thresh)
    # print(keep_dets)
    # print(dets[keep_dets])
