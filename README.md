# simplifiedUnetSR

## Referece
### Code

[Mnster00/simplifiedUnetSR](https://github.com/Mnster00/simplifiedUnetSR)

[amirgholami/ZeroQ](https://github.com/amirgholami/ZeroQ)

[xushoukai/GDFQ](https://github.com/xushoukai/GDFQ)

### Paper

Lu, Z., & Chen, Y. (2021). Single image super-resolution based on a modified U-net with mixed gradient loss. Signal, Image and Video Processing, 1-9.   

Van Der Jeught, S., Muyshondt, P. G., & Lobato, I. (2021). Optimized loss function in deep learning profilometry for improved prediction performance. Journal of Physics: Photonics, 3(2), 024014.   

Xu, S., Li, H., Zhuang, B., Liu, J., Cao, J., Liang, C., & Tan, M. (2020, August). Generative low-bitwidth data free quantization. In European Conference on Computer Vision (pp. 1-17). Springer, Cham.

### Run
python main.py -m unet -uf 2 -lr 0.001 -n 300

-b [batchSize]	
-t [testBatchSize]	
-seed [random seed]	
-m [model] 	
-uf [upscale_factor]	
-lr [learning rate]	
-n [epochs]	


-m [model] 	

SimUnet--->unet	

ESPCN--->sub	

SRCNN--->srcnn	

VDSR--->vdsr	

EDSR--->edsr	

FSRCNN--->fsrcnn	

DRCN--->drcn	

SRGAN--->srgan	

Bicubic--->bi	
