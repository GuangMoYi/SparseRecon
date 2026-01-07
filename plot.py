#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt


            
# 该函数接受两个参数 model 和 output_im，分别表示模型和模型的输出。
def plot_cs(model,output_im):
    
    
    i = 600

    # 从 model 对象中获取真实图像，假设 model 对象有一个名为 im 的属性。
    true = model.im

    # 创建一个图形对象，设置分辨率为 300。
    plt.figure(dpi=300);

    # # 在第1个子图中绘制真实图像的第 i 个切片，使用 'seismic_r' 色图，并添加颜色条。
    plt.subplot(3,1,1);
    plt.imshow(true[i,:,:,:], cmap='seismic_r');plt.colorbar(); 

    # 在第2个子图中绘制模型输出的第 i 个切片，使用 'seismic_r' 色图，并添加颜色条。
    plt.subplot(3,1,2);
    plt.imshow(output_im[i,:,:,:],cmap='seismic_r');plt.colorbar(); 

    # 在第3个子图中绘制真实图像和模型输出之间的差异，使用 'coolwarm' 色图，并添加颜色条。
    plt.subplot(3,1,3);plt.imshow(true[i,:,:,:]-output_im[i,:,:,:], cmap='coolwarm');
    plt.colorbar();

    # 调整子图的布局，确保它们适应整个图形。
    plt.tight_layout()
    
    # 添加总标题，标题内容为 i，即当前选择的索引
    plt.suptitle(i)
