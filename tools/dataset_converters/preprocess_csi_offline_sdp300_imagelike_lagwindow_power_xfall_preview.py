#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preview-only SDP300 image-like preprocessing with lag-window flattening.

This is the same image-like SDP construction as
preprocess_csi_offline_sdp300_imagelike_power_xfall_preview.py, but the image
width is arranged as:

    [lag0 window0..windowN, lag1 window0..windowN, ...]

instead of:

    [window0 lag0..lagN, window1 lag0..lagN, ...]

The core SDP computation is unchanged.
"""

from __future__ import annotations

from preprocess_csi_offline_sdp300_imagelike_power_xfall_preview import (
    CONFIG as BASE_CONFIG,
    process_preview_groups,
)


CONFIG = dict(BASE_CONFIG)
CONFIG.update(
    {
        "dst_root": "/home/xl/CSI/Person-in-WiFi-3D-repo/data/wifipose/all_single_test_data_hold_out_offline_sdp300_imagelike_lagwindow_power_xfall_preview",
        "dst_csi_dir": "csi_sdp300_imagelike_lagwindow_offline",
        "preview_dir": "preview_sdp300_imagelike_lagwindow",
        "group_list_name": "sdp300_imagelike_lagwindow_group_list.txt",
        "flatten_order": "lag_window",
    }
)


if __name__ == "__main__":
    process_preview_groups(CONFIG)
