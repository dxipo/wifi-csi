% 假设你的变量名是 csi_out（4-D complex double）
% 大小为 [3, 3, 30, 20]

rx = 1;        % 接收天线编号（1 ~ 3）
tx = 1;        % 发送天线编号（1 ~ 3）
pkt = 1;       % 数据包编号（1 ~ 20）

% 取出这个天线对在该数据包中的所有子载波的 CSI 值
csi_slice = csi_out(rx, tx, :, pkt);    % 结果是 1×1×30，复数形式

% 变形为一维向量
csi_slice = squeeze(csi_slice);         % 得到 30×1 向量

% 计算幅值（模）
csi_amplitude = abs(csi_slice);         % 取复数模长（幅值）

% 绘图
figure;
plot(1:30, csi_amplitude, '-o');
title(sprintf('CSI Amplitude - Rx%d to Tx%d, Packet %d', rx, tx, pkt));
xlabel('Subcarrier Index');
ylabel('Amplitude');
grid on;