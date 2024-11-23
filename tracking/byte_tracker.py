#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:54
@Description: byte_tracker - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""
import numpy as np
from tracker.base_track import BaseTrack, TrackState
from tracking.tracker import matching
from tracking.tracker.kalman_filter import KalmanFilter



class BYTETracker(object):

    """用于多目标跟踪任务，基于检测框的跟踪"""
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]  # 存储当前帧中活跃的跟踪对象。
        self.lost_stracks = []  # type: list[STrack]    # 存储当前帧中丢失的跟踪对象。
        self.removed_stracks = []  # type: list[STrack]  # 存储当前帧中已删除的跟踪对象。

        self.frame_id = 0   # frame_id: 当前处理的帧ID。
        self.args = args    # 跟踪参数，通常是一个包含各种配置的命名空间对象。
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1  # 检测阈值，用于过滤检测结果。
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)   # 缓冲区大小，用于存储跟踪对象的帧数。根据帧率和跟踪缓冲参数计算。
        self.max_time_lost = self.buffer_size  # 最大丢失时间，超过这个时间的跟踪对象将被移除。
        self.kalman_filter = KalmanFilter()   # 卡尔曼滤波器，用于预测跟踪对象的下一帧位置。

    # 更新跟踪结果
    def update(self, output_results, img_info, img_size):
        """"""
        # 更新帧ID:
        self.frame_id += 1

        # 初始化当前帧的跟踪状态:
        activated_starcks = []  # 存储当前帧中新激活的跟踪对象。
        refind_stracks = []     # 存储当前帧中重新激活的跟踪对象。
        lost_stracks = []       # 存储当前帧中丢失的跟踪对象。
        removed_stracks = []    # 存储当前帧中移除的跟踪对象。

        # 处理检测结果:
        # 检查检测结果的形状，如果是5列（即包含4个边界框坐标和1个置信度分数），直接提取分数和边界框。
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
        # 如果检测结果的形状不是5列，假设是Tensor，先转换为NumPy数组，然后提取分数和边界框。
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        # 将边界框坐标从归一化坐标转换为实际图像坐标:
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale  # 将检测框从缩放后的图像坐标转换回原始图像坐标。

        # 过滤检测结果
        #  获取得分高于 track_thresh 的索引。
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1 # : 获取得分大于0.1的索引。
        inds_high = scores < self.args.track_thresh  # : 获取得分小于 track_thresh 的索引。
        # 获取得分在0.1到 track_thresh 之间的索引。
        inds_second = np.logical_and(inds_low, inds_high)   # 提取得分在0.1到 track_thresh 之间的检测框。
        dets_second = bboxes[inds_second]   # : 提取得分在0.1到 track_thresh 之间的检测框。
        dets = bboxes[remain_inds]  # : 提取得分高于 track_thresh 的检测框。
        scores_keep = scores[remain_inds]   # : 提取得分高于 track_thresh 的检测框的置信度分数。
        scores_second = scores[inds_second] # : 提取得分在0.1到 track_thresh 之间的检测框的置信度分数。

        # 如果有检测框，则创建跟踪对象:
        if len(dets) > 0: # 有得分高于 track_thresh 的检测框。
            '''将检测框和得分转换为 STrack 对象。'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            # 有得分高于 track_thresh 的检测框，detections 列表为空。
            detections = []

        '''处理未确认的跟踪对象'''
        unconfirmed = []    # 存储当前帧中未确认的跟踪对象。
        tracked_stracks = []  # type: list[STrack]  # 存储当前帧中已确认的跟踪对象。
        for track in self.tracked_stracks:  # 遍历当前帧中所有活跃的跟踪对象。
            if not track.is_activated:  # 如果跟踪对象未激活，将其添加到 unconfirmed 列表。
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)   # 如果跟踪对象已激活，将其添加到 tracked_stracks 列表。

        ''' Step 2: 第一次匹配'''
        # 将已确认的跟踪对象和丢失的跟踪对象合并。
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # 使用卡尔曼滤波器预测所有跟踪对象的位置。
        STrack.multi_predict(strack_pool)
        # 计算已确认的跟踪对象和检测框之间的IOU距离。
        dists = matching.iou_distance(strack_pool, detections)
        #  如果不是MOT20数据集，融合得分。
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # 使用线性分配算法进行匹配。
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # 更新匹配成功的跟踪对象
        for itracked, idet in matches: # 遍历匹配结果。
            track = strack_pool[itracked]   # 获取匹配的跟踪对象。
            det = detections[idet]  # 获取匹配的检测结果。
            if track.state == TrackState.Tracked:  # 如果跟踪对象是已激活状态，更新其状态。
                track.update(detections[idet], self.frame_id)   # 更新跟踪对象的状态。
                activated_starcks.append(track) # 将更新后的跟踪对象添加到 activated_starcks 列表。
            else:
                # 如果跟踪对象不是已激活状态，重新激活它。
                track.re_activate(det, self.frame_id, new_id=False) # : 重新激活跟踪对象。
                refind_stracks.append(track)    # 将重新激活的跟踪对象添加到 refind_stracks 列表。

        ''' Step 3: 第二次匹配'''
        # 如果有得分在0.1到 track_thresh 之间的检测框。
        if len(dets_second) > 0:
            '''将检测框和得分转换为 STrack 对象。'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            # 如果没有得分在0.1到 track_thresh 之间的检测框，detections_second 列表为空。
            detections_second = []
        """第二次匹配继续"""
        # 获取未匹配的已激活跟踪对象。
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 计算未匹配的已激活跟踪对象与低得分检测结果之间的IoU距离。
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 使用线性分配算法进行匹配。
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # 更新第二次匹配成功的跟踪对象
        for itracked, idet in matches:  # 遍历匹配结果。
            track = r_tracked_stracks[itracked] # 获取匹配的跟踪对象。
            det = detections_second[idet]   # 获取匹配的检测结果。
            #  如果跟踪对象是已激活状态，更新其状态。
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)    # 更新跟踪对象的状态。
                activated_starcks.append(track) # 将更新后的跟踪对象添加到 activated_starcks 列表。
            else:   # 如果跟踪对象不是已激活状态，重新激活它。
                track.re_activate(det, self.frame_id, new_id=False) # 重新激活跟踪对象。
                refind_stracks.append(track)    # 将重新激活的跟踪对象添加到 refind_stracks 列表。
        # 遍历未匹配的已激活跟踪对象。
        for it in u_track:
            track = r_tracked_stracks[it] # 获取未匹配的跟踪对象。
            if not track.state == TrackState.Lost:  # 如果跟踪对象不是丢失状态，标记为丢失。
                track.mark_lost()   #  标记跟踪对象为丢失。
                lost_stracks.append(track)  # 将丢失的跟踪对象添加到 lost_stracks 列表。

        '''处理未确认的跟踪对象'''
        detections = [detections[i] for i in u_detection]   # 获取未匹配的高得分检测结果。
        dists = matching.iou_distance(unconfirmed, detections)  # 计算未确认的跟踪对象与未匹配的高得分检测结果之间的IoU距离。
        if not self.args.mot20: # 如果不是MOT20数据集，融合得分。
            dists = matching.fuse_score(dists, detections)  # 使用线性分配算法进行匹配。
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # 更新未确认的跟踪对象
        for itracked, idet in matches: # 遍历匹配结果。
            unconfirmed[itracked].update(detections[idet], self.frame_id)   # 更新未确认的跟踪对象。
            activated_starcks.append(unconfirmed[itracked]) # 将更新后的未确认跟踪对象添加到 activated_starcks 列表。
        # 遍历未匹配的未确认跟踪对象。
        for it in u_unconfirmed:
            track = unconfirmed[it] # 获取未匹配的未确认跟踪对象。
            track.mark_removed()    #  标记未确认的跟踪对象为移除。
            removed_stracks.append(track)   #  将移除的跟踪对象添加到 removed_stracks 列表。

        """ Step 4: 初始化新的跟踪对象"""
        for inew in u_detection:    # 遍历未匹配的高得分检测结果。
            track = detections[inew]    #  获取未匹配的检测结果。
            if track.score < self.det_thresh:   # 如果检测结果的得分低于 det_thresh，跳过。
                continue
            track.activate(self.kalman_filter, self.frame_id)   # 激活新的跟踪对象
            activated_starcks.append(track) # 将激活的跟踪对象添加到 activated_starcks 列表。
        """ Step 5: 更新丢失的跟踪对象"""
        # 遍历丢失的跟踪对象。
        for track in self.lost_stracks: # 如果跟踪对象丢失的时间超过 max_time_lost，标记为移除。
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()    # 标记跟踪对象为移除。
                removed_stracks.append(track)   # 将移除的跟踪对象添加到 removed_stracks 列表。

        # print('Ramained match {} s'.format(t4-t3))
        """last: 更新跟踪状态 """
        # 更新当前帧中活跃的跟踪对象列表。
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 合并当前帧中活跃的跟踪对象和新激活的跟踪对象。
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 合并当前帧中活跃的跟踪对象和重新激活的跟踪对象。
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 从丢失的跟踪对象中移除已激活的跟踪对象。
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 将丢失的跟踪对象添加到 lost_stracks 列表。
        self.lost_stracks.extend(lost_stracks)
        # 从丢失的跟踪对象中移除已移除的跟踪对象。
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 将移除的跟踪对象添加到 removed_stracks 列表。
        self.removed_stracks.extend(removed_stracks)
        # 移除重复的跟踪对象。
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 获取当前帧中活跃的跟踪对象。 返回当前帧中活跃的跟踪对象
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks



class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb