"""
SSDFace Demo
========


Usage
-----

Keys
----
    SPACE: pause
    ESC/q: quit
"""

"""
Revision
---------
"""

import os
import time

import mxnet as mx
import gluoncv
import numpy as np
import cv2 as cv

import altusi.configs.config as cfg
import altusi.helper.funcs as fn
import altusi.utils.visualizer as vis
from altusi.utils import imgproc, FPS, misc
from altusi.utils.logger import *


def filter_bboxes(bboxes, scores, class_ids, thresh=0.5):
    ids = np.where(scores.asnumpy().reshape(-1) > thresh)[0]

    if len(ids):
        return bboxes[ids], scores[ids], class_ids[ids]
    else:
        return None, None, None


def ssd_predict(net, image, ctx, thresh=0.5, img_dim=300):
    x, img = gluoncv.data.transforms.presets.ssd.transform_test(mx.nd.array(image), short=img_dim)
    x = x.as_in_context(ctx)

    class_ids, scores, bboxes = net(x)

    if len(bboxes[0]) > 0:
        bboxes, scores, class_ids = filter_bboxes(bboxes[0], scores[0], class_ids[0], thresh)

        if bboxes is not None:
            classes = [net.classes[int(idx.asscalar())] for idx in class_ids]

    return class_ids, scores, bboxes, img


def rescale_bboxes(bboxes, dims, new_dims):
    H, W = dims
    _H, _W = new_dims
    
    _bboxes = []
    for bbox in bboxes:
        bbox = bbox.asnumpy()
        bbox = bbox / np.array([W, H, W, H]) * np.array([_W, _H, _W, _H])
        _bboxes.append(bbox)
        
    return _bboxes


def app(video_link, video_name, show=False, record=False, 
        flip_hor=False, flip_ver=False):
    # specify device
    ctx = mx.context.gpu(0) if mx.context.num_gpus() else context.cpu()
    LOG(INFO, 'Device in Use:', ctx)

    classes = ['person']
    # load network model
    net = gluoncv.model_zoo.get_model('ssd_300_vgg16_atrous_custom', 
                                      classes=classes,
                                      ctx=ctx)
    net.load_parameters('ssd_300_vgg16_atrous_voc_best.params')
    net.collect_params().reset_ctx(ctx)

    # initialize camera capturer
    cap = cv.VideoCapture(video_link)
    fps = FPS()

    (W, H), vFPS = imgproc.cameraCalibrate(cap)
    LOG(INFO, 'Camera info: {} - FPS: {:.2f}'.format((W, H), vFPS) )

    time_str = time.strftime(cfg.TIME_FM)

    if record:
        writer = cv.VideoWriter(video_name + time_str + '.avi', 
                    cv.VideoWriter_fourcc(*'XVID'), vFPS, (W, H) )

    fps.start()
    cnt_frm = 0
    while cap.isOpened():
        _, frm = cap.read()
        if not _:
            LOG(INFO, 'Reached the end of Video stream')
            break

        if flip_ver: frm = cv.flip(frm, 0)
        if flip_hor: frm = cv.flip(frm, 1)

        cnt_frm += 1

        # predict on frame
        cids, scores, bboxes, img = ssd_predict(net, frm, ctx, thresh=0.5)
        if bboxes is not None:
            rs_bboxes = rescale_bboxes(bboxes, img.shape[:2], frm.shape[:2])
        else:
            rs_bboxes = []
        vis_frm = vis.drawObjects(frm, rs_bboxes,
                                    color=vis.COLOR_RED_LIGHT,
                                    thickness=4)

        fps.update()
        LOG(INFO, 'Time elapses: {:.2f} - FPS: {:.2f}'.format(
            fps.elapsed(), fps.fps() ) )

        if show:
            cv.imshow(video_name, vis_frm)
            key = cv.waitKey(1)
            if key in [27, ord('q') ]:
                LOG(INFO, 'Interrupted by user')
                break

        if record:
            writer.write(vis_frm)

    fps.stop()
    LOG(INFO, 'Total processing time: {:.2f}'.format(fps.totalTime() ) )

    if record:
        writer.release()
    cap.release()
    cv.destroyAllWindows()

    LOG(DEBUG, 'cnt frame:', cnt_frm)


def main(args):
    video_link = args.video if args.video else 0 
    app(video_link, args.name, args.show, args.record, 
            args.flip_hor, args.flip_ver)


if __name__ == '__main__':
    print(__doc__)

    LOG(INFO, 'Experiment: Camera Capturer\n')

    args = fn.getArgs()
    main(args)

    LOG(INFO, 'Process done')
