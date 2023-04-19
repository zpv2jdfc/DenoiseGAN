# DenoiseGAN A GAN-Based Method for seismic signal denoising
[中文](./README-ZH.md)
## 1.intall
#### download DenoiseGAN repository
1.1 if you have already installed [Git LFS](https://git-lfs.com/)

`git lfs clone --depth=1 https://github.com/zpv2jdfc/DenoiseGAN.git DenoiseGAN`

1.2 otherwise

- `git clone --depth=1 -b withoutmodel https://github.com/zpv2jdfc/DenoiseGAN.git DenoiseGAN`

- download <a href="http://v-ming.com/files/G.pth">model</a> and put it into `DenoiseGAN/gan/models`

#### install [Anaconda environment](https://www.anaconda.com/products/distribution)
. open directory

`cd DenoiseGAN`

. install & activate DenoiseGAN environment  
`conda env create -f env.yml`   

`conda activate denoisegan`
## 2.denoise
. Drop your files or directories to `gan/test_data`. (only support .mseed or .h5 file now)

. Run command `python main.py`.  The denoised signal files will be placed in `gan/results`

. If you also want to see the figures of spectrum, use `python main.py true`. But this will be slow.


