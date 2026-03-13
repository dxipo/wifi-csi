%% ====== Person-in-WiFi CSI Reader ======
% Author: ChatGPT
% Function: Read .dat CSI files from Person-in-WiFi dataset
% File: read_personinwifi_demo.m

clear; clc; close all;

% ==== Step 1: 指定 .dat 文件路径 ====
file = '/home/xl/Downloads/rawdata/CSI/r1/data/S11_01.dat';  % 改成你的文件路径

% ==== Step 2: 打开并读取 float32 数据 ====
fid = fopen(file, 'rb');
raw = fread(fid, 'float32');
fclose(fid);

% ==== Step 3: 根据数据格式 reshape ====
N_rx = 3;      % 接收天线数量（论文中使用 3）
N_tx = 3;      % 发送天线数量（论文中使用 3）
N_sub = 20;   % 子载波数量（论文中使用 114）

N_pkt = length(raw) / (2 * N_rx * N_tx * N_sub);  % 每帧包含实部和虚部两份float32数据
if mod(N_pkt, 1) ~= 0
    error('数据长度与预期不匹配，请检查N_rx/N_tx/N_sub设置');
end

% 结构: [real imag real imag ...]
raw = reshape(raw, [2, N_sub, N_tx, N_rx, N_pkt]);
csi_complex = squeeze(raw(1,:,:,:,:)) + 1i * squeeze(raw(2,:,:,:,:));
csi_complex = permute(csi_complex, [4 3 2 1]);  % 调整为 [frame, rx, tx, subcarrier]

fprintf('读取完成：共 %d 帧，%d×%d 天线，%d 子载波\n', N_pkt, N_rx, N_tx, N_sub);

% ==== Step 4: 可视化某一帧 ====
frame_idx = 1;  % 选择第1帧
amp = abs(squeeze(csi_complex(frame_idx, 1, 1, :))); % 取第1接收天线、第1发送天线
plot(amp, 'LineWidth', 1.2);
xlabel('Subcarrier Index');
ylabel('|CSI|');
title(sprintf('CSI Amplitude (Frame %d, Rx1-Tx1)', frame_idx));
grid on;

% ==== Step 5: 可选 - 保存为 .mat 文件 ====
save('/home/xl/Downloads/rawdata/CSI/r1/data/S11_01.mat', 'csi_complex', '-v7.3');
disp('已保存为 .mat 文件');

