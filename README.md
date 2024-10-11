# 🌀 ControlNeXt-SVD-v2

This is our implementation of ControlNeXt based on [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1). It can be seen as an attempt to replicate the implementation of [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone) with a more concise and efficient architecture.

Compared to image generation, video generation poses significantly greater challenges. While direct training of the generation model using our method is feasible, we also employ various engineering strategies to enhance performance. Although they are irrespective of academic algorithms.


> Please refer to [Inference](#inference) for more details regarding installation and inference.\
> Please refer to [Advanced Performance](#advanced-performance) for more details to achieve a better performance.\
> Please refer to [Limitations](#limitations) for more details about the limitations of current work.


# Inference

1. Clone repository
```
git clone https://github.com/newgenai79/ControlNeXt-SVD-v2
```

2. Navigate inside cloned repo
```
cd ControlNeXt-SVD-v2
```

3. Create virtual environment
```
python -m venv venv
```

4. Activate virtual environment
```
venv\scripts\activate
```

5. Install wheel
```
pip install wheel
```

6. Install requirements
```
pip install -r requirements.txt
```

7. Download examples folder from original repo
```
https://github.com/dvlab-research/ControlNeXt/tree/main/ControlNeXt-SVD-v2
```

8. Download pretrained weights

8.1. Download the pretrained weight into `pretrained/` from [here](https://huggingface.co/Pbihao/ControlNeXt/tree/main/ControlNeXt-SVD/v2). (More details please refer to [Base Model](#base-model))

8.2. Download the DWPose weights including the [dw-ll_ucoco_384](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing) and [yolox_l](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing) into `pretrained/DWPose`. For more details, please refer to [DWPose](https://github.com/IDEA-Research/DWPose):
```
├───pretrained
    └───DWPose
    |   │───dw-ll_ucoco_384.onnx
    |   └───yolox_l.onnx
    |
    ├───unet.bin
    └───controlnet.bin
```
8.3
Clone SVD model in root folder
```
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1 stabilityai/stable-video-diffusion-img2vid-xt-1-1
```

```
ControlNeXt-SVD-v2\stabilityai
	└───stable-video-diffusion-img2vid-xt-1-1
		│   .gitattributes
		│   LICENSE.md
		│   model_index.json
		│   README.md
		│   svd11.webp
		│   svd_xt_1_1.safetensors
		│
		├───feature_extractor
		│       preprocessor_config.json
		│
		├───image_encoder
		│       config.json
		│       model.fp16.safetensors
		│
		├───scheduler
		│       scheduler_config.json
		│
		├───unet
		│       config.json
		│       diffusion_pytorch_model.fp16.safetensors
		│
		└───vae
				config.json
				diffusion_pytorch_model.fp16.safetensors
```
9. Launch gradio WebUI

```
python app.py
```

> --pretrained_model_name_or_path : pretrained base model, we pretrain and fintune models based on [SVD-XT1.1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)\
> --controlnet_model_name_or_path : the model path of controlnet (a light weight module) \
> --unet_model_name_or_path : the model path of unet \
> --ref_image_path: the path to the reference image \
> --overlap: The length of the overlapped frames for long-frame video generation. \
> --sample_stride: The length of the sampled stride for the conditional controls. You can set it to `1` to make more smooth generation wihile requires more computation.


### Reference Image

It is crucial to ensure that the reference image is clear and easily understandable, especially aligning the face of the reference with the pose.

### Continuously Finetune

To significantly enhance performance on a specific pose sequence, you can continuously fine-tune the model for just a few hundred steps. 

We will release the related fine-tuning code later.

### Pose Generation

We adopt [DWPose](https://github.com/IDEA-Research/DWPose) for the pose generation, and follow the related work ([1](https://humanaigc.github.io/animate-anyone/), [2](https://tencent.github.io/MimicMotion/)) to align the pose.

# Limitations

## IP Consistency

We did not prioritize maintaining IP consistency during the development of the generation model and now rely on a helper model for face enhancement. 

However, additional training can be implemented to ensure IP consistency moving forward.

This also leaves a possible direction for futher improvement.

## Base model

The base model plays a crucial role in generating human features, particularly hands and faces. We encourage collaboration to improve the base model for enhanced human-related video generation.
