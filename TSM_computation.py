import cv2
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from model import get_repnet_model

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
solve_cudnn_error()


def get_result(frames, model, batch_size,strides):
    for stride in strides:
        num_batches = int(np.ceil(len(frames)/model.num_frames/stride/batch_size))
        raw_scores_per_stride = []
        within_period_score_stride = []
        for batch_idx in range(num_batches):

            idxes = tf.range(batch_idx*batch_size*model.num_frames*stride,
                             (batch_idx+1)*batch_size*model.num_frames*stride,stride)
            idxes = tf.clip_by_value(idxes, 0, len(frames)-1)
            curr_frames = tf.gather(frames, idxes)
            curr_frames = tf.reshape(curr_frames,
                                     [batch_size, model.num_frames, model.image_size, model.image_size, 3])
            raw_scores, within_period_scores, _ , dist = model(curr_frames)
            return _


def pairwise_l2_distance(a, b):
    """Computes pairwise distances between all rows of a and all rows of b."""
    norm_a = tf.reduce_sum(tf.square(a), 1)
    norm_a = tf.reshape(norm_a, [-1, 1])
    norm_b = tf.reduce_sum(tf.square(b), 1)
    norm_b = tf.reshape(norm_b, [1, -1])
    dist = tf.maximum(norm_a - 2.0 * tf.matmul(a, b, False, True) + norm_b, 0.0)
    return dist

def get_self_dist(embs):
    batch_size = tf.shape(embs)[0]
    seq_len = tf.shape(embs)[1]
    embs = tf.reshape(embs, [batch_size, seq_len, -1])

    def _get_dist(embs):
        dist = pairwise_l2_distance(embs, embs)
        return dist
    dist = tf.map_fn(_get_dist, embs)
    return dist


PATH_TO_CKPT = './repnet_ckpt/'
model_repnet = get_repnet_model(PATH_TO_CKPT)
total_video = [i for i in glob.glob('./worker_dataset/*mp4')]

total_video.sort()



for video in range(len(total_video)):
    FILE_NAME = total_video[video].split('/')[-1].split('.')[0]
    print(FILE_NAME)
    STRIDES = [1]
    BATCH_SIZE = 1
    total_dist = np.zeros([len(fix_frames)+1-64,len(fix_frames),len(fix_frames)])
    print(len(fix_frames))
    for i in range(len(fix_frames)):
        if i+64 > len(fix_frames):
            break
        else:
            frames_input = model_repnet.preprocess(fix_frames[i:i+64])
            final_emds = get_result(frames_input, model_repnet,batch_size=BATCH_SIZE,strides=STRIDES)
            dist = get_self_dist(final_emds)


            total_dist[i,i+3:i+61,i+3:i+61] = dist.numpy()[0,3:61,3:61]
    
    np.savez_compressed('./TSM/'+FILE_NAME+'_total_dist.npz', dist=total_dist)