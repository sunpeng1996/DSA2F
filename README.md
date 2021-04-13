

# DSA^2 F: Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion (CVPR'2021, Oral)

This repo is the official implementation of 
["DSA^2 F: Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion"](https://arxiv.org/pdf/2103.11832.pdf)

by Peng Sun, Wenhu Zhang, Huanyu Wang, Songyuan Li, and Xi Li.

### Code will be released soon after.

## Main Results

|Dataset | E<sub>r</sub>| S<sub>λ</sub><sup>mean</sup>|F<sub>β</sub><sup>mean</sup>| M |
|:---:|:---:|:---:|:---:|:---:|
|DUT-RGBD|0.950|0.921|0.926|0.030|
|NJUD|0.923|0.903|0.901|0.039|
|NLPR|0.950|0.918|0.897|0.024|
|SSD|0.904|0.876|0.852|0.045|
|STEREO|0.933|0.904|0.898|0.036|
|LFSD|0.923|0.882|0.882|0.054|
|RGBD135|0.962|0.920|0.896|0.021|

## Saliency maps and Evaluation

 All of the saliency maps mentioned in the paper are available on [GoogleDrive](https://drive.google.com/file/d/1pqRpWgyDry3o6iKNNDx_eM2_kEOftYY3/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1Fr5PuABceE7ordJvE84PKA)(code:juc2).
   
You can use the toolbox provided by [jiwei0921](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) for evaluation.

Additionally, we also provide the saliency maps of the STERE-1000 and SIP dataset on  [BaiduYun](https://pan.baidu.com/s/1idJ_yWl3N22fafa0RgzuQw)(code:r7da) for easy comparison.

|Dataset | E<sub>r</sub>| S<sub>λ</sub><sup>mean</sup>|F<sub>β</sub><sup>mean</sup>| M |
|:---:|:---:|:---:|:---:|:---:|
|STERE-1000|0.928|0.897|0.895|0.038|
|SIP|0.908|0.861|0.868|0.057|

## Citation
```
@inproceedings{Sun2021DeepRS,
  title={Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion},
  author={P. Sun and Wenhu Zhang and Huanyu Wang and Songyuan Li and Xi Li},
  journal={IEEE Conf. Comput. Vis. Pattern Recog.},
  year={2021}
}
```


## License

The code is released under MIT License (see LICENSE file for details).
