import os
from scipy import io
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import pywt
from collections import OrderedDict
import scipy.fft as fft
from scipy.signal import stft
from .builder import DATASETS
from mmdet.datasets.pipelines import Compose
import h5py

@DATASETS.register_module()
class WifiPoseDataset(dataset):
    CLASSES = ('person', )
    def __init__(self, dataset_root, pipeline, mode, **kwargs):
        
        self.data_root = dataset_root
        self.img_dir = ''  # 你这里没有rgb图像路径，但pipeline可能会读 img_prefix
        self.debug = kwargs.get('debug', False)
        self._dbg_printed = False

        self.offline_sdp_dir = kwargs.get('offline_sdp_dir', 'csi_sdp_offline')
        self.offline_sdp_ext = kwargs.get('offline_sdp_ext', '.npy')
        self.sdp_layout = kwargs.get('sdp_layout', 'wtn')  # 'wtn' or 'nwt'

        self.pipeline = Compose(pipeline)
        self.filename_list = self.load_file_name_list(os.path.join(self.data_root, mode + '_data_list.txt'))
        self._set_group_flag()
        
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir

    def get_item_single_frame(self,index):
        data_name = self.filename_list[index]
        keypoint_path = os.path.join(self.data_root, 'keypoint', str(data_name) + '.npy')
        token_path = os.path.join(self.data_root, 'token', str(data_name) + '.npy')

        csi = self._load_csi_feature(data_name)

        keypoint_raw = torch.from_numpy(np.load(keypoint_path)).float()  # (N,14,3) or (N,14,2)

        keypoint_xy = keypoint_raw[..., :2]  # (N,14,2)
        if keypoint_raw.shape[-1] >= 3:
            keypoint_v = keypoint_raw[..., 2:3]  # (N,14,1) 保留原始可见性/置信度
        else:
            keypoint_v = torch.ones(keypoint_xy.shape[0], 14, 1, dtype=keypoint_xy.dtype)  # 没有就补1

        W, H = 640.0, 360.0
        keypoint_xy[..., 0] = keypoint_xy[..., 0] / W
        keypoint_xy[..., 1] = keypoint_xy[..., 1] / H
        keypoint_xy = keypoint_xy.clamp(0.0, 1.0)

        keypoint = torch.cat([keypoint_xy, keypoint_v], dim=2)  # (N,14,3)

        assert keypoint.ndim == 3 and keypoint.shape[1] == 14 and keypoint.shape[2] == 3, \
            f"bad keypoint shape: {keypoint.shape}"

        if index < 3:
            print("[wifi_pose] img:", csi.shape, "kpt:", keypoint.shape,
                  "kpt_minmax_xy:", keypoint[..., :2].min().item(), keypoint[..., :2].max().item())

        # if self.debug and (not self._dbg_printed):
        #     print("[wifi_pose] img:", csi.shape, "kpt:", keypoint.shape,
        #           "kpt_minmax_xy:", keypoint[..., :2].min().item(), keypoint[..., :2].max().item())

        # --- END ADD ---

        # ---- ADD: load teacher token ----
        gt_token = torch.from_numpy(np.load(token_path).astype(np.float32)).contiguous()  # (768,)
        assert gt_token.ndim == 1 and gt_token.shape[0] == 768, f"bad token shape: {gt_token.shape}"

        # gt_token = np.load(token_path).astype(np.float32)  # (768,)
        # gt_token = torch.from_numpy(gt_token)  # torch.Size([768])
        # # sanity check
        # assert gt_token.ndim == 1 and gt_token.shape[0] == 768, f"bad token shape: {gt_token.shape}"
        # ---- END ADD ----
        if index < 3:
            print("[wifi_pose] token:", gt_token.shape, gt_token.dtype, "min/max:", gt_token.min().item(),
                  gt_token.max().item())
        #dis

        # if self.debug and (not self._dbg_printed):
        #     print("[wifi_pose] token:", gt_token.shape, gt_token.dtype, "min/max:",
        #           gt_token.min().item(), gt_token.max().item())
        #     self._dbg_printed = True


        numOfPerson = keypoint.shape[0]
        # gt_labels = np.zeros(numOfPerson, dtype=np.int64) #label (N,)
        # gt_bboxes = torch.tensor([])
        # gt_areas = torch.tensor([])

        gt_labels = torch.zeros((numOfPerson,), dtype=torch.long)

        # 用 keypoints 生成一个 bbox + area（像素空间）
        xy_pix = keypoint[..., :2].clone()
        xy_pix[..., 0] = xy_pix[..., 0] * W
        xy_pix[..., 1] = xy_pix[..., 1] * H

        x1 = xy_pix[..., 0].min(dim=1).values
        y1 = xy_pix[..., 1].min(dim=1).values
        x2 = xy_pix[..., 0].max(dim=1).values
        y2 = xy_pix[..., 1].max(dim=1).values

        gt_bboxes = torch.stack([x1, y1, x2, y2], dim=1).float()  # (N,4)
        gt_areas = ((x2 - x1).clamp(min=1.0) * (y2 - y1).clamp(min=1.0)).float()  # (N,)

        result = dict(img=csi, gt_keypoints=keypoint, gt_labels = gt_labels, gt_bboxes = gt_bboxes, gt_areas = gt_areas, gt_token = gt_token, img_name = data_name, img_shape=(360, 640, 3))#dis gt_token
        #result = dict(img=csi, gt_keypoints=keypoint, gt_labels = gt_labels, gt_bboxes = gt_bboxes, gt_areas = gt_areas, gt_token = gt_token, img_name = data_name, img_shape=(360, 640, 3))
        return result
    
    # def get_item_single_frame_limit(self,index):
    #     data_name = self.filename_list[index]
    #     if self.use_offline_stft:
    #         csi_path = os.path.join(
    #             self.data_root,
    #             self.offline_stft_dir,
    #             data_name + self.offline_stft_ext
    #         )
    #
    #         if self.offline_stft_ext == '.npy':
    #             csi = np.load(csi_path).astype(np.float32)
    #         elif self.offline_stft_ext == '.npz':
    #             csi = np.load(csi_path)['csi_feature'].astype(np.float32)
    #         elif self.offline_stft_ext == '.mat':
    #             with h5py.File(csi_path, 'r') as f:
    #                 csi = np.array(f['csi_feature']).astype(np.float32)
    #         else:
    #             raise ValueError(f'Unsupported offline_stft_ext={self.offline_stft_ext}')
    #
    #         csi = torch.FloatTensor(csi)
    #
    #     else:
    #         csi_path = os.path.join(self.data_root, 'csi', data_name + '.mat')
    #         csi = h5py.File(csi_path)['csi_out']
    #         csi = np.array(csi).transpose(3, 2, 1, 0)
    #         csi = csi.astype(np.complex128)
    #
    #         if self.use_stft:
    #             csi = self.stft_amp(csi)  # (3,3,30,Tbin,F)
    #             csi = torch.FloatTensor(csi)
    #         else:
    #             csi_amp = self.dwt_amp(csi)
    #             csi_ph = self.phase_deno(csi)
    #             csi_ph = np.angle(csi_ph)
    #             csi = np.concatenate((csi_amp, csi_ph), axis=2)
    #             csi = torch.FloatTensor(csi).permute(0, 1, 3, 2)
    #             print('offline csi shape:', csi.shape)
    #
    #
    #     keypoint = np.array(np.load(keypoint_path))
    #     #keypoint = self.keypoint_process(keypoint)
    #     keypoint = torch.FloatTensor(keypoint) # keypoint tensor: (N*14*3)
    #
    #     numOfPerson = keypoint.shape[0]
    #     gt_labels = np.zeros(numOfPerson, dtype=np.int64) #label (N,)
    #     gt_bboxes = torch.tensor([])
    #     gt_areas = torch.tensor([])
    #     result = dict(img=csi, gt_keypoints=keypoint, gt_labels = gt_labels, gt_bboxes = gt_bboxes, gt_areas = gt_areas )
    #     return result

    def _load_csi_feature(self, data_name):
        csi_path = os.path.join(
            self.data_root,
            self.offline_sdp_dir,
            data_name + self.offline_sdp_ext
        )

        csi = np.load(csi_path).astype(np.float32)

        # 保存时如果是 (3,3,NΔ,WT)，这里转成 (3,3,WT,NΔ)
        if self.sdp_layout == 'nwt':
            csi = np.transpose(csi, (0, 1, 3, 2))
        elif self.sdp_layout == 'wtn':
            pass
        else:
            raise ValueError(f'Unsupported sdp_layout={self.sdp_layout}')

        return torch.FloatTensor(csi)


    def __getitem__(self, index):
        result = self.get_item_single_frame(index)
        self.pre_pipeline(result)  # <<<<<< 新增
        return self.pipeline(result)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                file_name_list.append(lines.split()[0])
        return file_name_list

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    def CSI_sanitization(self, csi_rx):
        one_csi = csi_rx[0,:,:]
        two_csi = csi_rx[1,:,:]
        three_csi = csi_rx[2,:,:]
        pi = np.pi
        M = 3  # 天线数量3
        N = 30  # 子载波数目30
        T = one_csi.shape[1]  # 总包数
        fi = 312.5 * 2  # 子载波间隔312.5 * 2
        csi_phase = np.zeros((M, N, T))
        for t in range(T):  # 遍历时间戳上的CSI包，每根天线上都有30个子载波
            csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
            csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
            csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
            ai = np.tile(2 * pi * fi * np.array(range(N)), M)
            bi = np.ones(M * N)
            ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
            A = np.dot(ai, ai)
            B = np.dot(ai, bi)
            C = np.dot(bi, bi)
            D = np.dot(ai, ci)
            E = np.dot(bi, ci)
            rho_opt = (B * E - C * D) / (A * C - B ** 2)
            beta_opt = (B * D - A * E) / (A * C - B ** 2)
            temp = np.tile(np.array(range(N)), M).reshape(M, N)
            csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
        antennaPair_One = abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
        antennaPair_Two = abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
        antennaPair_Three = abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
        antennaPair = np.concatenate((np.expand_dims(antennaPair_One,axis=0), 
                                      np.expand_dims(antennaPair_Two,axis=0), 
                                      np.expand_dims(antennaPair_Three,axis=0),))
        return antennaPair


    def phase_deno(self, csi):
        #input csi shape (3*3*30*20)
        ph_rx1 = self.CSI_sanitization(csi[0,:,:,:])
        ph_rx2 = self.CSI_sanitization(csi[1,:,:,:])
        ph_rx3 = self.CSI_sanitization(csi[2,:,:,:])
        csi_phde = np.concatenate((np.expand_dims(ph_rx1,axis=0), 
                                   np.expand_dims(ph_rx2,axis=0), 
                                   np.expand_dims(ph_rx3,axis=0),))
        #csi_phde = csi_phde.transpose(0,1,3,2)
        return csi_phde
    
    def dwt_amp(self, csi):
        #csi = csi.transpose(0,1,3,2)
        #cA, cD = pywt.dwt(abs(csi), 'db11')
        #csi_amp = np.concatenate((cA, cD), axis=2)
        #csi_amp = np.concatenate((cA, cD), axis=3)
        w = pywt.Wavelet('dB11')
        list = pywt.wavedec(abs(csi), w,'sym')
        csi_amp = pywt.waverec(list, w)
        return csi_amp

    def stft_amp(self, csi):
        """
        对 amplitude 做 STFT
        input:
            csi: complex ndarray, shape (3, 3, 30, 20)

        output:
            stft feature, shape (3, 3, 30, Tbin, F)
            最后一维 F 作为 feature dim，其余维度会在 detector 里展平成 token 维
        """
        amp = np.abs(csi).astype(np.float32)  # (3,3,30,20)

        nperseg = self.stft_cfg.get('nperseg', 8)
        noverlap = self.stft_cfg.get('noverlap', 4)
        nfft = self.stft_cfg.get('nfft', 16)

        out = []
        for rx in range(3):
            rx_list = []
            for tx in range(3):
                sc_list = []
                for sc in range(30):
                    seq = amp[rx, tx, sc]  # (20,)

                    f, t, Zxx = stft(
                        seq,
                        nperseg=nperseg,
                        noverlap=noverlap,
                        nfft=nfft,
                        boundary=None,
                        padded=False
                    )

                    spec = np.abs(Zxx).astype(np.float32)  # (F, Tbin)
                    spec = np.log1p(spec)  # 数值更稳

                    # 转成 (Tbin, F)，让最后一维 F 作为 feature dim
                    spec = spec.transpose(1, 0)  # (Tbin, F)
                    sc_list.append(spec)

                sc_list = np.stack(sc_list, axis=0)  # (30, Tbin, F)
                rx_list.append(sc_list)

            rx_list = np.stack(rx_list, axis=0)  # (3, 30, Tbin, F)
            out.append(rx_list)

        out = np.stack(out, axis=0)  # (3, 3, 30, Tbin, F)
        return out


    def keypoint_process(self, keypoints):
        next_point = np.array([[0,1], [1,2], [2,5], [3,0], [4,2], [5,7],
                               [6,3], [7,3], [8,4], [9,5], [10,6], [11,7],
                               [12,9], [13,11]])
        keypoints_list = []
        for numofperson in range(keypoints.shape[0]):
            for numofpoint in range(keypoints.shape[1]):
                point_with_next = np.concatenate((keypoints[numofperson,next_point[numofpoint,0],:],
                                                  keypoints[numofperson,next_point[numofpoint,1],:]), axis=0)
                point_class = np.zeros((15))
                keypoints_list.append(point_with_next)
        
        return np.array(keypoints_list)

    @staticmethod
    def _to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _mean_or_zero(values):
        return float(np.mean(values)) if len(values) > 0 else 0.0

    @staticmethod
    def _img_wh(info):
        img_shape = info.get('img_shape', (360, 640, 3))
        return float(img_shape[1]), float(img_shape[0])

    def _extract_person_predictions(self, det_bboxes, det_keypoints,
                                    gt_keypoints):
        if isinstance(det_keypoints, (list, tuple)):
            if len(det_keypoints) == 0:
                kpt_pred = np.zeros((0, 14, 2), dtype=np.float32)
            else:
                kpt_pred = det_keypoints[0]
        else:
            kpt_pred = det_keypoints

        if isinstance(det_bboxes, (list, tuple)):
            if len(det_bboxes) == 0:
                bbox_pred = np.zeros((0, 5), dtype=np.float32)
            else:
                bbox_pred = det_bboxes[0]
        else:
            bbox_pred = det_bboxes

        kpt_pred = torch.as_tensor(
            kpt_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)
        bbox_pred = torch.as_tensor(
            bbox_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)

        if kpt_pred.numel() == 0:
            kpt_pred = gt_keypoints.new_zeros((0, gt_keypoints.shape[1], 2))
        elif kpt_pred.dim() == 2:
            kpt_pred = kpt_pred.unsqueeze(0)

        if bbox_pred.numel() == 0:
            bbox_pred = gt_keypoints.new_zeros((0, 5))
        elif bbox_pred.dim() == 1:
            bbox_pred = bbox_pred.unsqueeze(0)

        num_pred = min(kpt_pred.shape[0], bbox_pred.shape[0])
        kpt_pred = kpt_pred[:num_pred]
        bbox_pred = bbox_pred[:num_pred]
        if bbox_pred.shape[1] >= 5:
            scores = bbox_pred[:, 4]
        else:
            scores = gt_keypoints.new_ones((num_pred,))

        return bbox_pred, kpt_pred, scores

    @staticmethod
    def _sort_indices_by_score(scores):
        if scores.numel() == 0:
            return scores.new_zeros((0,), dtype=torch.long)
        return torch.argsort(scores, descending=True)

    def _select_keypoints_by_score(self, kpt_pred, scores, topk=None):
        order = self._sort_indices_by_score(scores)
        if topk is not None:
            order = order[:min(int(topk), int(order.numel()))]
        return kpt_pred[order]

    def _bboxes_to_pixel(self, bboxes, img_wh):
        if torch.is_tensor(bboxes):
            boxes = bboxes.detach().cpu().float().numpy()
        else:
            boxes = np.asarray(bboxes, dtype=np.float32)

        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32)

        boxes = boxes[:, :4].astype(np.float32, copy=True)
        W, H = img_wh
        coord_max = np.nanmax(boxes) if boxes.size else 0.0
        coord_min = np.nanmin(boxes) if boxes.size else 0.0
        if coord_max <= 2.0 and coord_min >= -0.5:
            boxes[:, [0, 2]] *= W
            boxes[:, [1, 3]] *= H

        x1 = np.minimum(boxes[:, 0], boxes[:, 2])
        y1 = np.minimum(boxes[:, 1], boxes[:, 3])
        x2 = np.maximum(boxes[:, 0], boxes[:, 2])
        y2 = np.maximum(boxes[:, 1], boxes[:, 3])
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, W)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, H)
        return boxes.astype(np.float32)

    @staticmethod
    def _bbox_iou_matrix(pred_boxes, gt_boxes):
        if pred_boxes.size == 0 or gt_boxes.size == 0:
            return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]),
                            dtype=np.float32)

        px1, py1, px2, py2 = [pred_boxes[:, i:i + 1] for i in range(4)]
        gx1, gy1, gx2, gy2 = [gt_boxes[:, i][None, :] for i in range(4)]

        inter_x1 = np.maximum(px1, gx1)
        inter_y1 = np.maximum(py1, gy1)
        inter_x2 = np.minimum(px2, gx2)
        inter_y2 = np.minimum(py2, gy2)
        inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
        inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
        inter = inter_w * inter_h

        pred_area = np.maximum(px2 - px1, 0.0) * np.maximum(py2 - py1, 0.0)
        gt_area = np.maximum(gx2 - gx1, 0.0) * np.maximum(gy2 - gy1, 0.0)
        union = pred_area + gt_area - inter
        return inter / np.maximum(union, 1e-6)

    def _compute_bbox_ap_for_thr(self, ap_items, iou_thr, area_range=None):
        detections = []
        gt_by_img = {}
        total_gt = 0

        for item in ap_items:
            gt_boxes = item['gt_boxes']
            gt_areas = item['gt_areas']
            if area_range is not None:
                low, high = area_range
                area_mask = (gt_areas >= low) & (gt_areas < high)
                gt_boxes = gt_boxes[area_mask]

            if gt_boxes.shape[0] == 0:
                continue

            image_id = item['image_id']
            gt_by_img[image_id] = {
                'boxes': gt_boxes,
                'matched': np.zeros((gt_boxes.shape[0],), dtype=bool)
            }
            total_gt += gt_boxes.shape[0]

            for box, score in zip(item['pred_boxes'], item['scores']):
                detections.append((float(score), image_id, box))

        if total_gt == 0:
            return np.nan

        detections.sort(key=lambda x: x[0], reverse=True)
        tp = np.zeros((len(detections),), dtype=np.float32)
        fp = np.zeros((len(detections),), dtype=np.float32)

        for det_idx, (_, image_id, pred_box) in enumerate(detections):
            gt_info = gt_by_img.get(image_id)
            if gt_info is None:
                fp[det_idx] = 1.0
                continue

            gt_boxes = gt_info['boxes']
            ious = self._bbox_iou_matrix(pred_box[None, :], gt_boxes)[0]
            best_gt = int(np.argmax(ious)) if ious.size else -1
            best_iou = float(ious[best_gt]) if best_gt >= 0 else 0.0
            if (best_iou >= iou_thr and best_gt >= 0
                    and not gt_info['matched'][best_gt]):
                tp[det_idx] = 1.0
                gt_info['matched'][best_gt] = True
            else:
                fp[det_idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(float(total_gt), 1e-6)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

        recall_grid = np.linspace(0.0, 1.0, 101)
        ap = 0.0
        for recall_thr in recall_grid:
            valid = precisions[recalls >= recall_thr]
            ap += float(valid.max()) if valid.size else 0.0
        return ap / len(recall_grid)

    def _compute_bbox_ap_metrics(self, ap_items):
        iou_thrs = np.arange(0.5, 0.96, 0.05)
        area_all = (0.0, 1e10)
        area_m = (32.0 * 32.0, 96.0 * 96.0)
        area_l = (96.0 * 96.0, 1e10)

        def mean_ap(area_range):
            vals = [
                self._compute_bbox_ap_for_thr(ap_items, float(thr), area_range)
                for thr in iou_thrs
            ]
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals) * 100.0) if vals else 0.0

        ap50 = self._compute_bbox_ap_for_thr(ap_items, 0.50, area_all)
        ap75 = self._compute_bbox_ap_for_thr(ap_items, 0.75, area_all)

        return OrderedDict([
            ('AP', mean_ap(area_all)),
            ('AP50', 0.0 if np.isnan(ap50) else float(ap50 * 100.0)),
            ('AP75', 0.0 if np.isnan(ap75) else float(ap75 * 100.0)),
            ('APm', mean_ap(area_m)),
            ('APl', mean_ap(area_l)),
        ])

    def evaluate(self,
                 results,
                 metric='keypoints',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):

        metric_lists = OrderedDict([
            ('mpjpe', []),
            ('mpjpe_x', []),
            ('mpjpe_y', []),
            ('mpjpe_top1', []),
            ('mpjpe_top1_x', []),
            ('mpjpe_top1_y', []),
            ('mpjpe_top5_oracle', []),
            ('mpjpe_top5_oracle_x', []),
            ('mpjpe_top5_oracle_y', []),
            ('mpjpe_top20_oracle', []),
            ('mpjpe_top20_oracle_x', []),
            ('mpjpe_top20_oracle_y', []),
            ('mpjpe_oracle_all', []),
            ('mpjpe_oracle_all_x', []),
            ('mpjpe_oracle_all_y', []),
        ])
        ap_items = []

        for i in range(len(results)):
            info = self.get_item_single_frame(i)
            gt_keypoints = info['gt_keypoints']  # Tensor (N,14,3) or (N,14,2), normalized
            data_name = info.get('img_name', str(i))
            img_wh = self._img_wh(info)

            det_bboxes, det_keypoints = results[i]

            bbox_pred, kpt_pred, scores = self._extract_person_predictions(
                det_bboxes, det_keypoints, gt_keypoints)
            if kpt_pred.shape[0] == 0:
                continue

            # ================= DEBUG START =================
            if i == 0:
                print("\n========== DEBUG EVALUATE ==========")
                print("[DBG] GT shape:", gt_keypoints.shape)
                print("[DBG] Pred shape:", kpt_pred.shape)
                print("[DBG] Score shape:", scores.shape)

                print("[DBG] GT min/max:",
                      gt_keypoints[..., :2].min().item(),
                      gt_keypoints[..., :2].max().item())

                print("[DBG] Pred min/max:",
                      kpt_pred[..., :2].min().item(),
                      kpt_pred[..., :2].max().item())

                # GT 第一人第一个关节
                print("[DBG] First joint GT  :", gt_keypoints[0, 0, :2])

                # Pred 可能是 (M,14,2)，所以取第0个query的第0个关节仅用于观察
                if kpt_pred.dim() == 3:
                    print("[DBG] First joint Pred (query0):", kpt_pred[0, 0, :2])
                elif kpt_pred.dim() == 2:
                    print("[DBG] First joint Pred:", kpt_pred[0, :2])

                print("====================================\n")
            # ================= DEBUG END =================

            # Backward-compatible metric: keep the original all-query oracle
            # MPJPE as `mpjpe`, because previous experiments used this number.
            mpjpe_2d, mpjpex, mpjpey, _ = self.calc_mpjpe(
                gt_keypoints, kpt_pred, data_name, root=[5, 7],
                img_shape=info.get('img_shape', None)
            )
            metric_lists['mpjpe'].append(self._to_float(mpjpe_2d))
            metric_lists['mpjpe_x'].append(self._to_float(mpjpex))
            metric_lists['mpjpe_y'].append(self._to_float(mpjpey))
            metric_lists['mpjpe_oracle_all'].append(self._to_float(mpjpe_2d))
            metric_lists['mpjpe_oracle_all_x'].append(self._to_float(mpjpex))
            metric_lists['mpjpe_oracle_all_y'].append(self._to_float(mpjpey))

            for topk, prefix in [(1, 'mpjpe_top1'),
                                 (5, 'mpjpe_top5_oracle'),
                                 (20, 'mpjpe_top20_oracle')]:
                selected_kpts = self._select_keypoints_by_score(
                    kpt_pred, scores, topk=topk)
                top_mpjpe, top_x, top_y, _ = self.calc_mpjpe(
                    gt_keypoints, selected_kpts, data_name=None, root=[5, 7],
                    img_shape=info.get('img_shape', None)
                )
                metric_lists[prefix].append(self._to_float(top_mpjpe))
                metric_lists[f'{prefix}_x'].append(self._to_float(top_x))
                metric_lists[f'{prefix}_y'].append(self._to_float(top_y))

            gt_bboxes = info['gt_bboxes'].detach().cpu().float().numpy()
            gt_areas = info['gt_areas'].detach().cpu().float().numpy()
            ap_items.append(dict(
                image_id=i,
                gt_boxes=self._bboxes_to_pixel(gt_bboxes, img_wh),
                gt_areas=gt_areas,
                pred_boxes=self._bboxes_to_pixel(bbox_pred, img_wh),
                scores=scores.detach().cpu().float().numpy()
            ))

        result = OrderedDict()
        for key, values in metric_lists.items():
            result[key] = self._mean_or_zero(values)
        result.update(self._compute_bbox_ap_metrics(ap_items))
        return OrderedDict(result)

    # def evaluate(self,
    #              results,
    #              metric='keypoints',
    #              logger=None,
    #              jsonfile_prefix=None,
    #              classwise=False,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thrs=None,
    #              metric_items=None):
    #
    #     mpjpe_2d_list = []
    #     mpjpe_x_list = []
    #     mpjpe_y_list = []
    #
    #     for i in range(len(results)):
    #         info = self.get_item_single_frame(i)
    #         gt_keypoints = info['gt_keypoints']  # Tensor, shape (N,14,3) or (N,14,2)
    #         data_name = info.get('img_name', str(i))
    #
    #         det_bboxes, det_keypoints = results[i]
    #
    #         # det_keypoints: list per class/person
    #         for label in range(len(det_keypoints)):
    #             kpt_pred = det_keypoints[label]
    #             kpt_pred = torch.tensor(
    #                 kpt_pred,
    #                 dtype=gt_keypoints.dtype,
    #                 device=gt_keypoints.device
    #             )
    #
    #             if i == 0 and label == 0:
    #                 print("\n========== DEBUG EVALUATE ==========")
    #
    #                 print("[DBG] GT shape:", gt_keypoints.shape)
    #                 print("[DBG] Pred shape:", kpt_pred.shape)
    #
    #                 print("[DBG] GT min/max:",
    #                       gt_keypoints[..., :2].min().item(),
    #                       gt_keypoints[..., :2].max().item())
    #
    #                 print("[DBG] Pred min/max:",
    #                       kpt_pred[..., :2].min().item(),
    #                       kpt_pred[..., :2].max().item())
    #
    #                 print("[DBG] First joint GT  :", gt_keypoints[0, 0, :2])
    #                 print("[DBG] First joint Pred:", kpt_pred[0, 0, :2])
    #
    #                 print("====================================\n")
    #             # ================= DEBUG END =================
    #
    #             # 2D MPJPE（要求你在 calc_mpjpe 内部只算 [:,:2]）
    #             mpjpe_2d, mpjpex, mpjpey, _ = self.calc_mpjpe(
    #                 gt_keypoints, kpt_pred, data_name, root=[5, 7]
    #             )
    #
    #             # mpjpe_* 可能是 Tensor，也可能是 float；统一转成 python float
    #             if torch.is_tensor(mpjpe_2d):
    #                 mpjpe_2d_list.append(float(mpjpe_2d.detach().cpu().item()))
    #             else:
    #                 mpjpe_2d_list.append(float(mpjpe_2d))
    #
    #             if torch.is_tensor(mpjpex):
    #                 mpjpe_x_list.append(float(mpjpex.detach().cpu().item()))
    #             else:
    #                 mpjpe_x_list.append(float(mpjpex))
    #
    #             if torch.is_tensor(mpjpey):
    #                 mpjpe_y_list.append(float(mpjpey.detach().cpu().item()))
    #             else:
    #                 mpjpe_y_list.append(float(mpjpey))
    #
    #     # 防止空结果导致 mean 报错
    #     mpjpe = float(np.mean(mpjpe_2d_list)) if len(mpjpe_2d_list) > 0 else 0.0
    #     mpjpe_x = float(np.mean(mpjpe_x_list)) if len(mpjpe_x_list) > 0 else 0.0
    #     mpjpe_y = float(np.mean(mpjpe_y_list)) if len(mpjpe_y_list) > 0 else 0.0
    #
    #     result = {
    #         'mpjpe': mpjpe,  # 2D 平均欧氏距离
    #         'mpjpe_x': mpjpe_x,  # |dx| 平均
    #         'mpjpe_y': mpjpe_y  # |dy| 平均
    #     }
    #     return OrderedDict(result)
    #
    # def evaluate(self,
    #              results,
    #              metric='keypoints',
    #              logger=None,
    #              jsonfile_prefix=None,
    #              classwise=False,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thrs=None,
    #              metric_items=None):
    #     mpjpe_3d_list = []
    #     mpjpe_h_list = []
    #     mpjpe_v_list = []
    #     mpjpe_d_list = []
    #     for i in range(len(results)):
    #         info = self.get_item_single_frame(i)
    #         gt_keypoints = info['gt_keypoints']
    #         data_name = info['img_name']
    #         det_bboxes, det_keypoints = results[i]
    #         for label in range(len(det_keypoints)):
    #             kpt_pred = det_keypoints[label]
    #             kpt_pred = torch.tensor(kpt_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)
    #             #np.save('/home/yankangwei/opera-main/result/pose_o/%s.npy' %data_name, kpt_pred)
    #             mpjpe_3d,mpjpeh,mpjpev,mpjped = self.calc_mpjpe(gt_keypoints, kpt_pred, data_name, root = [5,7])
    #             mpjpe_3d_list.append(mpjpe_3d.numpy())
    #             mpjpe_h_list.append(mpjpeh.numpy())
    #             mpjpe_v_list.append(mpjpev.numpy())
    #             mpjpe_d_list.append(mpjped.numpy())
    #             #mpjpe_3d_list.append(np.array([0]))
    #
    #     mpjpe = np.array(mpjpe_3d_list).mean()
    #     mpjpeh = np.array(mpjpe_h_list).mean()
    #     mpjpev = np.array(mpjpe_v_list).mean()
    #     mpjped = np.array(mpjpe_d_list).mean()
    #     result = {'mpjpe':mpjpe, 'mpjpeh':mpjpeh, 'mpjpev':mpjpev, 'mpjped':mpjped}
    #     return OrderedDict(result)
    #
    # def calc_mpjpe(self, real, pred, no, root=0):
    #     n = real.shape[0]
    #     m = pred.shape[0]
    #     j, c = pred.shape[1:]
    #     assert j == real.shape[1] and c == real.shape[2]
    #
    #     # [2D task] only use x,y (ignore 3rd dim)
    #     real = real[..., :2]
    #     pred = pred[..., :2]
    #
    #     if isinstance(root,list):
    #         real_root = real.unsqueeze(1).expand(n, m, j, c)
    #         pred_root = pred.unsqueeze(0).expand(n, m, j, c)
    #         #n*m*j  n*j
    #         distance_array = torch.ones((n,m), dtype=torch.float) * 2 ** 24  # TODO: magic number!
    #         for i in range(n):
    #             for j in range(m):
    #                 distance_array[i][j] = torch.norm(real[i]-pred[j], p=2, dim=-1).mean()
    #
    #         # distance_array = torch.norm(real_root-pred_root, p=2, dim=-1)*vis_mask.unsqueeze(1).expand(n, m, j)
    #         # distance_array = distance_array.sum(-1) / vis_mask.sum(-1).unsqueeze(1)
    #         # print(torch.min(distance_array))
    #     else:
    #         real_root = real[:, root].unsqueeze(0).expand(n, m, c)
    #         pred_root = pred[:, root].unsqueeze(1).expand(n, m, c)
    #         distance_array = torch.pow(real_root - pred_root, 2)
    #     corres = torch.ones(n, dtype=torch.long)*-1
    #     occupied = torch.zeros(m, dtype=torch.long)
    #
    #     while torch.min(distance_array) < 50:   # threshold 30.
    #         min_idx = torch.where(distance_array == torch.min(distance_array))
    #
    #         for i in range(len(min_idx[0])):
    #             distance_array[min_idx[0][i]][min_idx[1][i]] = 50
    #             if corres[min_idx[0][i]] >= 0 or occupied[min_idx[1][i]]:
    #                 continue
    #             else:
    #                 corres[min_idx[0][i]] = min_idx[1][i]
    #                 occupied[min_idx[1][i]] = 1
    #     new_pred = pred[corres]
    #     #np.save('/home/yankangwei/opera-main/result/pose_pred/%s.npy' %no, new_pred)
    #     #np.save('/home/yankangwei/opera-main/result/pose_gt/%s.npy' %no, real)
    #     # mpjpe = torch.sqrt(torch.pow(real - new_pred, 2).sum(-1))
    #     # mpjpeh = torch.sqrt(torch.pow(real[:,:,0] - new_pred[:,:,0], 2))
    #     # mpjpev = torch.sqrt(torch.pow(real[:,:,1] - new_pred[:,:,1], 2))
    #     # mpjped = torch.sqrt(torch.pow(real[:,:,2] - new_pred[:,:,2], 2))
    #
    #     mpjpe = torch.sqrt(torch.pow(real - new_pred, 2).sum(-1))  # (n,14) 2D dist
    #     mpjpe_x = torch.abs(real[:, :, 0] - new_pred[:, :, 0])
    #     mpjpe_y = torch.abs(real[:, :, 1] - new_pred[:, :, 1])
    #     return mpjpe.mean() * 1.0, mpjpe_x.mean() * 1.0, mpjpe_y.mean() * 1.0, torch.tensor(0.0)
    #
    #     # mpjpe = torch.norm(real-new_pred, p=2, dim=-1) #n*j
    #     # mpjpe_mean = (mpjpe*vis_mask.float()).sum(-1)/vis_mask.float().sum(-1) if vis_mask is not None else mpjpe.mean(-1)
    #     # return mpjpe.mean()*1000, mpjpeh.mean()*1000, mpjpev.mean()*1000, mpjped.mean()*1000

    # def calc_mpjpe(self, real, pred, data_name=None, root=None):
    #     """
    #     2D MPJPE (pixel space)
    #     real: Tensor, shape (N, 14, 3) or (N, 14, 2)
    #     pred: Tensor, shape (M, 14, 3) or (M, 14, 2)
    #
    #     return:
    #         mpjpe_2d, mpjpe_x, mpjpe_y, dummy
    #     """
    #
    #     # ---- 只取 x,y ----
    #     real = real[..., :2]  # (N,14,2)
    #     pred = pred[..., :2]  # (M,14,2)
    #
    #     # 如果有多个 pred（query），只取第一个 / 或者你外部已筛选
    #     if pred.dim() == 3:
    #         pred = pred[0:1]  # (1,14,2)
    #
    #     # 对齐 batch 维度
    #     if real.shape[0] != pred.shape[0]:
    #         pred = pred.expand(real.shape[0], -1, -1)
    #
    #     # ---- 2D 欧氏距离 ----
    #     diff = real - pred  # (N,14,2)
    #     dist = torch.norm(diff, dim=-1)  # (N,14)
    #
    #     mpjpe_2d = dist.mean()
    #     mpjpe_x = torch.abs(diff[..., 0]).mean()
    #     mpjpe_y = torch.abs(diff[..., 1]).mean()
    #
    #     # 第四个返回值占位（保持 evaluate() 接口不炸）
    #     dummy = torch.tensor(0.0, device=real.device)
    #
    #     return mpjpe_2d, mpjpe_x, mpjpe_y, dummy

    # def calc_mpjpe(self, real, pred, data_name=None, root=None):
    #     """
    #     Pixel-space 2D MPJPE.
    #
    #     real: Tensor, shape (N,14,3) or (N,14,2)  (GT, normalized [0,1])
    #     pred: Tensor, shape (M,14,3) or (M,14,2)  (Pred, should also be normalized [0,1])
    #
    #     return:
    #         mpjpe_pix, mpjpe_x_pix, mpjpe_y_pix, dummy
    #     """
    #
    #     # ---- only x,y ----
    #     real = real[..., :2]  # (N,14,2)
    #     pred = pred[..., :2]  # (M,14,2)  or maybe already (14,2) etc.
    #
    #     # 如果 pred 是 (14,2)，补成 (1,14,2)
    #     if pred.dim() == 2:
    #         pred = pred.unsqueeze(0)
    #
    #     # 如果 pred 有多个 query（M>1），默认取第一个
    #     if pred.shape[0] > 1:
    #         pred = pred[0:1]  # (1,14,2)
    #
    #     # 对齐 batch/person 维度：GT 可能是 N 人，但你现在单人 N=1
    #     if real.shape[0] != pred.shape[0]:
    #         pred = pred.expand(real.shape[0], -1, -1)  # (N,14,2)
    #
    #     # ---- diff in normalized ----
    #     diff = real - pred  # (N,14,2)
    #
    #     # ---- convert to pixel ----
    #     # 你现在 dataset 里固定了 img_shape=(360,640,3)
    #     # 这里用同样的 W,H
    #     W, H = 640.0, 360.0
    #     dx_pix = diff[..., 0] * W
    #     dy_pix = diff[..., 1] * H
    #
    #     # ---- pixel distances ----
    #     dist_pix = torch.sqrt(dx_pix * dx_pix + dy_pix * dy_pix)  # (N,14)
    #
    #     mpjpe_pix = dist_pix.mean()
    #     mpjpe_x_pix = torch.abs(dx_pix).mean()
    #     mpjpe_y_pix = torch.abs(dy_pix).mean()
    #
    #     dummy = torch.tensor(0.0, device=real.device)
    #     return mpjpe_pix, mpjpe_x_pix, mpjpe_y_pix, dummy

    # def calc_mpjpe(self, real, pred, data_name=None, root=None):
    #     """
    #     Pixel-space 2D MPJPE with matching (GT persons vs predicted queries).
    #
    #     real: Tensor (N,14,3) or (N,14,2) normalized [0,1]
    #     pred: Tensor (M,14,2) or (M,14,3) normalized [0,1]
    #     """
    #
    #     device = real.device
    #
    #     real_xy = real[..., :2]  # (N,14,2)
    #     pred_xy = pred[..., :2]  # (M,14,2) or (14,2)
    #
    #     if pred_xy.dim() == 2:
    #         pred_xy = pred_xy.unsqueeze(0)
    #
    #     n = real_xy.shape[0]
    #     m = pred_xy.shape[0]
    #     assert real_xy.shape[1] == pred_xy.shape[1] == 14
    #     assert real_xy.shape[2] == pred_xy.shape[2] == 2
    #
    #     # cost[i,j] = mean L2 distance over joints (normalized)
    #     diff_nm = real_xy.unsqueeze(1) - pred_xy.unsqueeze(0)  # (N,M,14,2)
    #     dist_nm = torch.norm(diff_nm, dim=-1)  # (N,M,14)
    #     cost = dist_nm.mean(dim=-1)  # (N,M)
    #
    #     # greedy one-to-one matching (like original)
    #     corres = torch.full((n,), -1, dtype=torch.long, device=device)
    #     occupied = torch.zeros((m,), dtype=torch.bool, device=device)
    #
    #     cost_work = cost.clone()
    #     big = 1e9
    #
    #     while True:
    #         # mask occupied preds
    #         cost_masked = cost_work.clone()
    #         cost_masked[:, occupied] = big
    #
    #         min_val = cost_masked.min()
    #         if min_val >= big:
    #             break
    #
    #         gi, pj = torch.where(cost_masked == min_val)
    #         assigned = False
    #         for t in range(len(gi)):
    #             i = int(gi[t]);
    #             j = int(pj[t])
    #             cost_work[i, j] = big
    #             if corres[i] >= 0 or occupied[j]:
    #                 continue
    #             corres[i] = j
    #             occupied[j] = True
    #             assigned = True
    #         if not assigned:
    #             break
    #         if (corres >= 0).all():
    #             break
    #
    #     # fallback: any unmatched GT -> argmin
    #     for i in range(n):
    #         if corres[i] < 0:
    #             corres[i] = int(torch.argmin(cost[i]).item())
    #
    #     new_pred_xy = pred_xy[corres]  # (N,14,2)
    #
    #     # normalized -> pixel
    #     W, H = 640.0, 360.0
    #     diff = real_xy - new_pred_xy
    #     dx_pix = diff[..., 0] * W
    #     dy_pix = diff[..., 1] * H
    #
    #     dist_pix = torch.sqrt(dx_pix * dx_pix + dy_pix * dy_pix)  # (N,14)
    #
    #     mpjpe_pix = dist_pix.mean()
    #     mpjpe_x_pix = torch.abs(dx_pix).mean()
    #     mpjpe_y_pix = torch.abs(dy_pix).mean()
    #
    #     dummy = torch.tensor(0.0, device=device)
    #     return mpjpe_pix, mpjpe_x_pix, mpjpe_y_pix, dummy

    def calc_mpjpe(self, real, pred, data_name=None, root=None,
                   img_shape=None, use_visible=False):
        """
        Pixel-space 2D MPJPE with matching (GT persons vs predicted queries).

        real: Tensor (N,14,3) or (N,14,2)  normalized [0,1]
        pred: Tensor (M,14,3) or (M,14,2)  normalized [0,1]   (M=100 queries usually)

        return:
            mpjpe_pix, mpjpe_x_pix, mpjpe_y_pix, dummy
        """

        device = real.device

        # ---- only x,y ----
        real_xy = real[..., :2]  # (N,14,2)
        pred_xy = pred[..., :2]  # (M,14,2) or (14,2)

        # pred could be (14,2) -> (1,14,2)
        if pred_xy.dim() == 2:
            pred_xy = pred_xy.unsqueeze(0)

        # basic shapes
        n = real_xy.shape[0]  # N persons (your case N=1)
        m = pred_xy.shape[0]  # M queries  (often 100)
        j = real_xy.shape[1]  # 14

        if m == 0:
            W, H = self._img_wh({'img_shape': img_shape or (360, 640, 3)})
            miss = torch.tensor((W * W + H * H) ** 0.5, device=device,
                                dtype=real_xy.dtype)
            return miss, miss, miss, torch.tensor(0.0, device=device)

        assert real_xy.shape[1] == pred_xy.shape[1], f"J mismatch: {real_xy.shape} vs {pred_xy.shape}"
        assert real_xy.shape[2] == 2 and pred_xy.shape[2] == 2, f"need xy only"

        if use_visible and real.shape[-1] >= 3:
            valid_mask = (real[..., 2] > 0).float()
        else:
            valid_mask = torch.ones(real_xy.shape[:2], device=device,
                                    dtype=real_xy.dtype)
        valid_denom = valid_mask.sum(dim=-1).clamp(min=1.0)

        # ---- compute cost matrix: (N,M) ----
        # cost[i,k] = mean L2 distance over joints (normalized space)
        # (N,1,14,2) - (1,M,14,2) -> (N,M,14,2)
        diff_nm = real_xy.unsqueeze(1) - pred_xy.unsqueeze(0)
        dist_nm = torch.norm(diff_nm, dim=-1)  # (N,M,14)
        cost = (dist_nm * valid_mask.unsqueeze(1)).sum(dim=-1) / \
            valid_denom.unsqueeze(1)  # (N,M)

        # ---- greedy matching like original code ----
        # original used while min(distance_array) < threshold and marks occupied
        # Here: for general N>1 keep the same behavior.
        corres = torch.full((n,), -1, dtype=torch.long, device=device)  # for each GT person pick a pred index
        occupied = torch.zeros((m,), dtype=torch.bool, device=device)

        # threshold: original code used 50 (but their unit was unclear).
        # For normalized coordinates, a sensible threshold is around 0.5~1.0.
        # BUT: to mimic "always pick best" in single-person, we can just pick argmin.
        # To stay closest to original behavior, do greedy with a very large threshold (effectively always match).
        # If you want strict thresholding, set match_thr = 0.5 (normalized).
        match_thr = 1e9  # effectively no threshold

        # make a copy since we'll overwrite
        cost_work = cost.clone()

        # Greedy: repeatedly take global min
        # Stop when no valid min or all matched
        while True:
            # mask out already occupied preds
            cost_work_masked = cost_work.clone()
            # set occupied columns to huge value
            cost_work_masked[:, occupied] = match_thr

            min_val = cost_work_masked.min()
            if torch.isinf(min_val) or min_val >= match_thr:
                break

            # indices of min
            min_pos = torch.where(cost_work_masked == min_val)
            # could be multiple, iterate
            assigned_any = False
            for t in range(len(min_pos[0])):
                gi = int(min_pos[0][t])
                pj = int(min_pos[1][t])
                # mark this pair as used
                cost_work[gi, pj] = match_thr
                if corres[gi] >= 0 or occupied[pj]:
                    continue
                corres[gi] = pj
                occupied[pj] = True
                assigned_any = True

            if not assigned_any:
                break

            # if all GT matched, stop
            if (corres >= 0).all():
                break

        # If some GT not matched (rare in your setting), fallback to argmin per GT
        for gi in range(n):
            if corres[gi] < 0:
                corres[gi] = int(torch.argmin(cost[gi]).item())

        # ---- aligned predictions ----
        new_pred_xy = pred_xy[corres]  # (N,14,2)

        # ---- DEBUG: print chosen query once ----
        # 单人场景：corres 只有一个元素，对应选中的 query index
        if data_name is not None and ('DEBUG_ONCE' not in globals()):
            globals()['DEBUG_ONCE'] = True
            chosen = int(corres[0].item())  # N=1
            best_cost = float(cost[0, chosen].detach().cpu().item())  # normalized-space mean L2
            print("[DBG] query0 first joint:", pred_xy[0, 0])
            print("[DBG] chosen query first joint:", pred_xy[chosen, 0])
            print("[DBG] chosen query min/max:", pred_xy[chosen].min().item(), pred_xy[chosen].max().item())

        # ---- convert normalized diff -> pixel diff ----
        W, H = self._img_wh({'img_shape': img_shape or (360, 640, 3)})
        diff = real_xy - new_pred_xy  # (N,14,2)
        dx_pix = diff[..., 0] * W
        dy_pix = diff[..., 1] * H

        dist_pix = torch.sqrt(dx_pix * dx_pix + dy_pix * dy_pix)  # (N,14)

        valid_sum = valid_mask.sum().clamp(min=1.0)
        mpjpe_pix = (dist_pix * valid_mask).sum() / valid_sum
        mpjpe_x_pix = (torch.abs(dx_pix) * valid_mask).sum() / valid_sum
        mpjpe_y_pix = (torch.abs(dy_pix) * valid_mask).sum() / valid_sum

        dummy = torch.tensor(0.0, device=device)
        return mpjpe_pix, mpjpe_x_pix, mpjpe_y_pix, dummy
