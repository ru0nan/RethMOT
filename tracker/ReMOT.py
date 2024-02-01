import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score,  feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        
        self.last_loc = None

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat, new_score=None):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if new_score is None:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            else:
                self.smooth_feat = (np.exp(-0.75*new_score)) * self.smooth_feat +  (1 - np.exp(-0.75*new_score)) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)#, new_track.score
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

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)#, new_track.score

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class ReMOT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.history = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # 前一帧的每个目标的遮挡情况
        # 该obj被多少obj遮挡
        self.occludee = [] # type: list[int]
        # 该 obj 遮挡了多少 obj
        self.occluder = [] # type: list[int]
        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def deduplicate_tracked_stracks(self, stracks, img):
        dist = matching.iou_distance(stracks, stracks) + np.eye(len(stracks))
        indx = np.where(dist<0.4)
        dup=[]
        sim = []
        num_pair = len(indx[0])/2
        if num_pair > 0:
            for ii, s in enumerate(indx):
                if ii >= num_pair:
                    break
                p, q = s[0], s[1]
                feat_p = self.encoder.inference(img, stracks[p].tlbr.copy().reshape([-1,4]))
                feat_q = self.encoder.inference(img, stracks[q].tlbr.copy().reshape([-1,4]))
                cos_sim = matching.cosine_dist(feat_p, feat_q)/2.0
                sim.append(cos_sim)
                if cos_sim < 0.05:
                    print(cos_sim)
                    if stracks[p].score > stracks[q].score:
                        dup.append(q)
                    else:
                        dup.append(p)
        res = [t for i, t in enumerate(stracks) if not i in dup]
        return res

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            
            # Find high threshold detections
            remain_inds = scores > 0.15
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 1: Get prediction boxes from KF'''
        strack_pool_all = joint_stracks(tracked_stracks, self.lost_stracks)
        len_lost = len(self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool_all)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool_all, warp)
        STrack.multi_gmc(unconfirmed, warp)

        '''Step 2: First pre-association, all to all'''
        ious_dists = matching.iou_distance(strack_pool_all, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        # IoU cost correction & low-score supress
        det_area = (dets[:,2]-dets[:,0])*(dets[:,3]-dets[:,1])
        mean_area = np.mean(det_area) if len(det_area) else det_area
        weight_correction = np.sqrt(mean_area/det_area)
        weight_correction = np.clip(weight_correction, 1.0, 2.0)
        weight_mask = np.tile(weight_correction, [len(strack_pool_all),1])
        ious_dists_w = weight_mask*ious_dists
        ious_mat = matching.ious(dets, dets) - np.eye(len(scores_keep))
        stage = calcu_stage(ious_mat, dets, scores_keep)        
        # scores_fuse = 0.8 * scores_keep + 0.2* np.exp(-stage)
        scores_fuse = scores_keep + stage
        high_scores = np.where(scores_keep>=self.args.track_high_thresh)
        low_scores = np.where(scores_fuse<self.args.track_high_thresh)

        alpha = 1.2 #1.0
        beta = 1.2 #1.0
        ious_dists_w[:, low_scores] *= alpha
        ious_dists_w[-len_lost:,:] *= beta
        dists = ious_dists_w.copy()
        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool_all, detections) / 2.0
            # emb_dists[:,low_scores] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)

            dists[:,high_scores] =  0.8*ious_dists_w[:,high_scores] + 0.2*emb_dists[:,high_scores]
            
        dists[ious_dists_mask] = 1.0
        dists_pre = dists.copy()
        # pre-associate
        matched_tracks = []
        matched_dets = []
        matches_pre, um_t, um_d = matching.linear_assignment(dists, thresh=0.8)#thresh=self.args.match_thresh-0.2
        if len(matches_pre):
            cost_pre = dists[matches_pre[:,0],matches_pre[:,1]]
            high_confd = cost_pre < 0.5#0.3       
            matched_high_qua = matches_pre[high_confd,:]
            matched_low_qua = matches_pre[np.logical_not(high_confd),:].reshape([-1,2])    

            for itracked, idet in matched_high_qua:
                track = strack_pool_all[itracked]
                det = detections[idet]
                matched_tracks.append(itracked)
                matched_dets.append(idet)
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
        else:
            matched_high_qua = np.asarray([],dtype=np.int32).reshape([-1,2])
            matched_low_qua = np.asarray([],dtype=np.int32).reshape([-1,2])

  
        '''Step 3: Second association'''
        dists = ious_dists.copy()
        if self.args.with_reid:
            dists[:,high_scores] = np.minimum(ious_dists[:,high_scores], emb_dists[:,high_scores]) 
        if len(matched_high_qua):
            dists[:,matched_high_qua[:,1]] = 1.0 #已经匹配的det不可能再被匹配
            dists[matched_high_qua[:,0], :] = 1.0#已匹配的track不可能再被匹配

        if dists.shape[1]>2:
            response = 1.0-dists
            sorted_idx = np.argsort(-response, axis=1)
            sorted_resp =  np.take_along_axis(response, sorted_idx, axis=1)
            
            PSR = (sorted_resp[:,0]+ 1e-5)/(sorted_resp[:,1]+ 1e-5)
            PSR = np.clip(PSR-1.0, 0, 2.0)
            boost = np.exp(-PSR)
            
            row = np.arange(dists.shape[0])
            dists[row, sorted_idx[:,0]] *=boost

        dists[ious_dists_mask] = 1.0

        det_mark = [True]*len(detections)
        matched_2nd, um_t, um_d = matching.linear_assignment(dists, thresh=0.3)
        
        for itracked, idet in matched_2nd:
            track = strack_pool_all[itracked]
            det = detections[idet]
            if itracked in matched_low_qua[:,0]:
                idx = np.where(matched_low_qua[:,0]==itracked)[0]
                cost1 =  dists[matched_low_qua[idx[0],0], matched_low_qua[idx[0],1]]
                cost2 = dists[itracked,idet]
                if cost1 < cost2 and det_mark[matched_low_qua[idx[0],1]] is True:
                    print('change')
                    itracked = matched_low_qua[idx[0],0]
                    idet =matched_low_qua[idx[0],1]
                    track = strack_pool_all[itracked]
                    det = detections[idet]
                    det_mark[idet] = False
            matched_tracks.append(itracked)
            matched_dets.append(idet)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        exisit_tracks = [strack_pool_all[i] for i in matched_tracks]
        u_track = set([i for i in range(len(strack_pool_all))]) - set(matched_tracks)
        u_detection = set([i for i in range(len(detections))]) - set(matched_dets)
        for it in u_track:
            track = strack_pool_all[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        rem_detections = [detections[i] for i in u_detection ]#if detections[i].score > self.track_high_thresh-0.1
        ious_dists = matching.iou_distance(unconfirmed, rem_detections)
        dists = ious_dists.copy()
        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, rem_detections) / 2.0
            high_scores = np.array([d.score for d in rem_detections]) > self.track_high_thresh
            dists[:,high_scores] =  np.minimum(ious_dists[:,high_scores], emb_dists[:,high_scores])
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = rem_detections[idet]
            if det.score < track.score-0.3:
                track.mark_removed()
                removed_stracks.append(track)
            else:
                unconfirmed[itracked].update(rem_detections[idet], self.frame_id)
                activated_starcks.append(unconfirmed[itracked])
                exisit_tracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = rem_detections[inew]
            if track.score < self.new_track_thresh:
                continue
            if len(exisit_tracks):
                overlap = matching.iou_distance([track], exisit_tracks) 
                if self.frame_id > 1 and np.min(overlap) < 0.5:
                    over_scores = [t.score for i,t in enumerate(exisit_tracks) if overlap[0,i] < 0.9]
                    if track.score < np.mean(over_scores):
                        continue
                if  self.frame_id > 1 and np.min(overlap) > 0.9:
                    if track.score < self.new_track_thresh+0.05:
                        continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # self.tracked_stracks = self.deduplicate_tracked_stracks(self.tracked_stracks, img)


        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        

        return output_stracks


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



def calcu_stage(ious_mat, bboxes, scores):
    """bboxes: [N, 4], tlbr"""
    degree_mat =  np.zeros_like(ious_mat)
    inds =  np.where(ious_mat > 0.4) #(row_idx array, colum_idx array)
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
    # count = degree_mat == -1
    # stage = count.sum(axis=0)
    # mask = np.where(degree_mat==-1, degree_mat, 0)
    stage = np.multiply(degree_mat, ious_mat).sum(axis=0)
    return stage

  