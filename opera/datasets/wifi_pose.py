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

        self.use_offline_stft = kwargs.get('use_offline_stft', False)
        self.offline_stft_dir = kwargs.get('offline_stft_dir', 'csi_stft_offline')
        self.offline_stft_ext = kwargs.get('offline_stft_ext', '.npy')

        self.use_stft = kwargs.get('use_stft', False)
        self.stft_cfg = kwargs.get(
            'stft_cfg',
            dict(nperseg=8, noverlap=4, nfft=16)
        )
        self.pipeline = Compose(pipeline)
        self.filename_list = self.load_file_name_list(os.path.join(self.data_root, mode + '_data_list.txt'))
        self._set_group_flag()
        
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir

    def get_item_single_frame(self,index): 
        data_name = self.filename_list[index]
        csi_path = os.path.join(self.data_root,'csi',(str(data_name)+'.mat'))
        keypoint_path = os.path.join(self.data_root,'keypoint',(str(data_name)+'.npy'))
        token_path = os.path.join(self.data_root, 'token', (str(data_name) + '.npy')) #dis
        
        '''csi =  io.loadmat(csi_path)['csi_out']
        csi = np.array(csi)
        csi = csi.astype(np.complex128)'''
        
        #csi = h5py.File(csi_path)['csi_out'].value
        #csi = h5py.File(csi_path)['csi_out'][()] # by po
        with h5py.File(csi_path, 'r') as f:
            csi = f['csi_out'][()]  # structured array: real/imag
        csi = csi['real'] + csi['imag']*1j
        csi = np.array(csi).transpose(3,2,1,0)
        csi = csi.astype(np.complex128)
        
        '''csi_amp = abs(csi)
        csi_amp = torch.FloatTensor(csi_amp).permute(0,1,3,2) #csi tensor: (3*3*30*20 -> 3*3*20*30)
        
        csi_ph = np.unwrap(np.angle(csi))
        csi_ph = fft.ifft(csi_ph)
        csi_phd = csi_ph[:,:,:,1:20] - csi_ph[:,:,:,0:19]
        csi_phd = torch.FloatTensor(csi_phd).permute(0,1,3,2)'''
        
        #-------------------
        # -------------------
        if self.use_offline_stft:
            # STFT baseline: only use amplitude
            csi = self.stft_amp(csi)  # (3,3,30,Tbin,F)
            csi = torch.FloatTensor(csi)
        else:
            csi_amp = self.dwt_amp(csi)
            csi_ph = self.phase_deno(csi)
            csi_ph = np.angle(csi_ph)
            csi = np.concatenate((csi_amp, csi_ph), axis=2)
            csi = torch.FloatTensor(csi).permute(0, 1, 3, 2)
        

        #keypoint = np.array(np.load(keypoint_path))

        #keypoint = self.keypoint_process(keypoint)
        #keypoint = torch.FloatTensor(keypoint) # keypoint tensor: (N*14*3)
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



        # keypoint = torch.FloatTensor(np.load(keypoint_path))  # (N,14,3) or (N,14,2)
        # # 只取 x,y
        # keypoint_xy = keypoint[..., :2]
        #
        # W, H = 640.0, 360.0
        # keypoint_xy[..., 0] = keypoint_xy[..., 0] / W
        # keypoint_xy[..., 1] = keypoint_xy[..., 1] / H
        #
        # # clamp 防止越界
        # keypoint_xy = keypoint_xy.clamp(0.0, 1.0)
        # keypoint = keypoint_xy  # (N,14,2)
        #
        #
        # #keypoint[..., 2] = 1.0
        #
        # # --- ADD: sanity check ---
        # assert keypoint.ndim == 3 and keypoint.shape[1] == 14, f"bad keypoint shape: {keypoint.shape}"
        # assert keypoint.shape[2] >= 2, f"need at least (x,y): {keypoint.shape}"
        # # 允许你是(14,3)或(14,2)，如果是(14,2)就补一维0，保证后面代码统一
        # if keypoint.shape[2] == 2:
        #     pad = torch.zeros(keypoint.shape[0], 14, 1, dtype=keypoint.dtype)
        #     keypoint = torch.cat([keypoint, pad], dim=2)

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
    
    def get_item_single_frame_limit(self,index): 
        data_name = self.filename_list[index]
        if self.use_offline_stft:
            csi_path = os.path.join(
                self.data_root,
                self.offline_stft_dir,
                data_name + self.offline_stft_ext
            )

            if self.offline_stft_ext == '.npy':
                csi = np.load(csi_path).astype(np.float32)
            elif self.offline_stft_ext == '.npz':
                csi = np.load(csi_path)['csi_feature'].astype(np.float32)
            elif self.offline_stft_ext == '.mat':
                with h5py.File(csi_path, 'r') as f:
                    csi = np.array(f['csi_feature']).astype(np.float32)
            else:
                raise ValueError(f'Unsupported offline_stft_ext={self.offline_stft_ext}')

            csi = torch.FloatTensor(csi)

        else:
            csi_path = os.path.join(self.data_root, 'csi', data_name + '.mat')
            csi = h5py.File(csi_path)['csi_out']
            csi = np.array(csi).transpose(3, 2, 1, 0)
            csi = csi.astype(np.complex128)

            if self.use_stft:
                csi = self.stft_amp(csi)  # (3,3,30,Tbin,F)
                csi = torch.FloatTensor(csi)
            else:
                csi_amp = self.dwt_amp(csi)
                csi_ph = self.phase_deno(csi)
                csi_ph = np.angle(csi_ph)
                csi = np.concatenate((csi_amp, csi_ph), axis=2)
                csi = torch.FloatTensor(csi).permute(0, 1, 3, 2)
                print('offline csi shape:', csi.shape)
        

        keypoint = np.array(np.load(keypoint_path))
        #keypoint = self.keypoint_process(keypoint)
        keypoint = torch.FloatTensor(keypoint) # keypoint tensor: (N*14*3)

        numOfPerson = keypoint.shape[0]
        gt_labels = np.zeros(numOfPerson, dtype=np.int64) #label (N,)
        gt_bboxes = torch.tensor([])
        gt_areas = torch.tensor([])
        result = dict(img=csi, gt_keypoints=keypoint, gt_labels = gt_labels, gt_bboxes = gt_bboxes, gt_areas = gt_areas )
        return result
    
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

    def evaluate(self,
                 results,
                 metric='keypoints',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):

        mpjpe_2d_list = []
        mpjpe_x_list = []
        mpjpe_y_list = []

        for i in range(len(results)):
            info = self.get_item_single_frame(i)
            gt_keypoints = info['gt_keypoints']  # Tensor (N,14,3) or (N,14,2), normalized
            data_name = info.get('img_name', str(i))

            det_bboxes, det_keypoints = results[i]

            # 你的模型通常返回 det_keypoints 为 list（按 class），只有 1 类 person
            # 我们把它统一成 Tensor: (M,14,2/3)
            if isinstance(det_keypoints, (list, tuple)):
                # person 类一般在 index 0
                if len(det_keypoints) == 0:
                    continue
                kpt_pred = det_keypoints[0]
            else:
                kpt_pred = det_keypoints

            kpt_pred = torch.as_tensor(
                kpt_pred,
                dtype=gt_keypoints.dtype,
                device=gt_keypoints.device
            )

            # ================= DEBUG START =================
            if i == 0:
                print("\n========== DEBUG EVALUATE ==========")
                print("[DBG] GT shape:", gt_keypoints.shape)
                print("[DBG] Pred shape:", kpt_pred.shape)

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

            # 关键：让 calc_mpjpe 去做匹配（GT N=1 vs Pred M=100）
            mpjpe_2d, mpjpex, mpjpey, _ = self.calc_mpjpe(
                gt_keypoints, kpt_pred, data_name, root=[5, 7]
            )

            # 统一转 float
            mpjpe_2d_list.append(
                float(mpjpe_2d.detach().cpu().item()) if torch.is_tensor(mpjpe_2d) else float(mpjpe_2d))
            mpjpe_x_list.append(float(mpjpex.detach().cpu().item()) if torch.is_tensor(mpjpex) else float(mpjpex))
            mpjpe_y_list.append(float(mpjpey.detach().cpu().item()) if torch.is_tensor(mpjpey) else float(mpjpey))

        mpjpe = float(np.mean(mpjpe_2d_list)) if len(mpjpe_2d_list) > 0 else 0.0
        mpjpe_x = float(np.mean(mpjpe_x_list)) if len(mpjpe_x_list) > 0 else 0.0
        mpjpe_y = float(np.mean(mpjpe_y_list)) if len(mpjpe_y_list) > 0 else 0.0

        result = {
            'mpjpe': mpjpe,  # 像素版 or normalized 取决于你 calc_mpjpe
            'mpjpe_x': mpjpe_x,  # |dx| mean
            'mpjpe_y': mpjpe_y  # |dy| mean
        }
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

    def calc_mpjpe(self, real, pred, data_name=None, root=None):
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

        assert real_xy.shape[1] == pred_xy.shape[1], f"J mismatch: {real_xy.shape} vs {pred_xy.shape}"
        assert real_xy.shape[2] == 2 and pred_xy.shape[2] == 2, f"need xy only"

        # ---- compute cost matrix: (N,M) ----
        # cost[i,k] = mean L2 distance over joints (normalized space)
        # (N,1,14,2) - (1,M,14,2) -> (N,M,14,2)
        diff_nm = real_xy.unsqueeze(1) - pred_xy.unsqueeze(0)
        dist_nm = torch.norm(diff_nm, dim=-1)  # (N,M,14)
        cost = dist_nm.mean(dim=-1)  # (N,M)

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
        # NOTE: You hardcoded img_shape=(360,640,3) in dataset, so use that.
        W, H = 640.0, 360.0
        diff = real_xy - new_pred_xy  # (N,14,2)
        dx_pix = diff[..., 0] * W
        dy_pix = diff[..., 1] * H

        dist_pix = torch.sqrt(dx_pix * dx_pix + dy_pix * dy_pix)  # (N,14)

        mpjpe_pix = dist_pix.mean()
        mpjpe_x_pix = torch.abs(dx_pix).mean()
        mpjpe_y_pix = torch.abs(dy_pix).mean()

        dummy = torch.tensor(0.0, device=device)
        return mpjpe_pix, mpjpe_x_pix, mpjpe_y_pix, dummy