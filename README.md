# GVGEN: Text-to-3D Generation with Volumetric Representation 🧊


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2403.12957-b31b1b.svg)](https://arxiv.org/abs/2403.12957)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://sotamak1r.github.io/gvgen/)&nbsp;

</div>


<p align="center">
<img src="assets/gvgen_teaser.gif" width=95%>
<p>


## 🔥 Update
- [2024.10.16] Code for GaussianVolume Fitting is released ! See [CPS](https://github.com/sotamak1r/CPS) !
- [2024.07.04] Code and Models for text-conditional 3D generation are released !
- [2024.07.04] GVGEN was accepted by ECCV 2024. See you in Milan!

## 🌿 Introduction

We introduce GVGEN, a novel diffusion-based framework, which is designed to efficiently generate 3D Gaussian representations from text input. We propose two innovative techniques: 
- ``Structured Volumetric Representation``. We first arrange disorganized 3D Gaussian points as a structured form GaussianVolume. This transformation allows the capture of intricate texture details within a volume composed of a fixed number of Gaussians. 
- ``Coarse-to-fine Generation Pipeline``. To simplify the generation of GaussianVolume and empower the model to generate instances with detailed 3D geometry, we propose a coarse-to-fine pipeline. It initially constructs a basic geometric structure, followed by the prediction of complete Gaussian attributes. 


## 🦄 Text-conditional 3D generation

### Environment Setup

```bash
conda create -n gvgen python=3.8
pip install -r requirements.txt
```

Then, install the `diff-gaussian-rasterization` submodule according to the instructions provided by [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)


### Pretrained Models
Please download models from [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/SOTAMak1r/GVGEN), put them in the folder `./ckpts`.



### Run

After completing all the above instructions, run 

```bash
python run_text.py --text_input YOUR_TEXT_INPUT

# for example
python run_text.py --text_input "a green truck"
```

The generated gif and 3DGS will be saved to `sample.gif` and `sample.ply`, respectively.
The text condition we used during training is derived from [Cap3D](https://huggingface.co/datasets/tiange/Cap3D). 
We recommend everyone to imitate the style of Cap3D's text and create your own prompts for better generation results.


## ⚡️ ToDo List

- [x] ~~Release Code for GaussianVolume fitting~~

- [ ] Release Code for data preprocessing

- [ ] Release Code for training


## License
The majority of this project is licensed under MIT License. Portions of the project are available under separate license of referred projects, detailed in corresponding files.


## BibTeX
```bibtex
@misc{he2024gvgentextto3dgenerationvolumetric,
      title={GVGEN: Text-to-3D Generation with Volumetric Representation}, 
      author={Xianglong He and Junyi Chen and Sida Peng and Di Huang and Yangguang Li and Xiaoshui Huang and Chun Yuan and Wanli Ouyang and Tong He},
      year={2024},
      eprint={2403.12957},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.12957}, 
}
```
