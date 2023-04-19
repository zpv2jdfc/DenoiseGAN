# DenoiseGAN：基于生成对抗网络的地震信号降噪方法
[English](./README.md)
## 1.安装
#### 下载 DenoiseGAN 代码
1.1 如果你已经安装了 [Git LFS](https://git-lfs.com/)

`git lfs clone --depth=1 https://github.com/zpv2jdfc/DenoiseGAN.git DenoiseGAN`

1.2 否则请按下面步骤

- `git clone --depth=1 -b withoutmodel https://github.com/zpv2jdfc/DenoiseGAN.git DenoiseGAN`

- 下载 [模型](http://v-ming.com/files/) 并把模型放在DenoiseGAN/gan/models目录下

(如果你没用git命令而是下载了压缩包，请下载 [模型](http://v-ming.com/files/) 并把模型放在DenoiseGAN/gan/models目录下)
#### 安装 [Anaconda](https://www.anaconda.com/products/distribution)
. 打开目录

`cd DenoiseGAN`

. 安装并激活DenoiseGAN运行环境

`conda env create -f env.yml`   

`conda activate denoisegan`
## 2.降噪
. 把地震信号文件放在目录： `gan/test_data` (目前只能处理.h5和.mseed结尾的文件)

. 运行命令 `python main.py`.  降噪后的信号会保存在 `gan/results`

. 如果在降噪时同时生成信号的频谱图, 请使用这个命令 `python main.py true`. 

