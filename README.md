![image](https://github.com/MinusZoneAI/ComfyUI-TrainTools-MZ/assets/5035199/3bdce469-5a49-4f59-8a88-1b20e3a75c85)
![image](https://github.com/MinusZoneAI/ComfyUI-TrainTools-MZ/assets/5035199/9cd7f4eb-f971-49a8-ad57-7a56b71bf022)


# ComfyUI-TrainTools-MZ
在ComfyUI中进行lora微调的节点,依赖于kohya-ss/sd-scripts等训练工具(Nodes for fine-tuning lora in ComfyUI, dependent on training tools such as kohya-ss/sd-scripts) 

## Recent changes 
 

## Installation
1. Clone this repo into `custom_nodes` folder.
2. Restart ComfyUI.
 
## Nodes 
### MZ_KohyaSSInitWorkspace
初始化训练文件夹,文件夹位于output目录(Initialize the training folder, the folder in the output directory)
+ lora_name(LoRa名称): 用于生成训练文件夹的名称(Used to generate the name of the training folder
+ branch(分支): sd-scripts的分支,默认为当前代码调试时使用的分支(sd-scripts branch, default is the branch used when debugging the current code)
+ source(源): sd-scripts的源,默认为github,下载有问题的话可以切换加速源(sd-scripts source, default is github, if there is a problem with the download, you can switch to the accelerated source)
 

### MZ_ImagesCopyWorkspace
复制图片到训练文件夹中和一些数据集配置(Copy images to the training folder and some dataset configurations)
+ images(图片列表): 推荐使用 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite 中的上传文件夹节点 (It is recommended to use the upload folder node in https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite )
+ force_clear(强制清空): 复制图片前是否强制清空原有文件夹内容(Whether to force clear the original folder content before copying the image)
+ force_clear_only_images(仅清空图片): 仅清空图片文件夹内容,不清空其他文件夹内容(Only clear the content of the image folder, not the content of other folders)
+ same_caption_generate(生成相同标注): 是否生成相同的标注文件(Whether to generate the same annotation file)
+ same_caption(单一标签): 生成相同标签的内容(Generate content with the same label)
+ 其他字段参考: https://github.com/kohya-ss/sd-scripts

![image](https://github.com/MinusZoneAI/ComfyUI-TrainTools-MZ/assets/5035199/b2eb8f78-9314-4860-9248-8c04f87bc470)

### MZ_KohyaSSUseConfig
一些基础的训练配置(Some basic training configurations)
+ 没什么特殊的,字段参考: https://github.com/kohya-ss/sd-scripts


### MZ_KohyaSSAdvConfig
更多的训练配置(More training configurations)
+ 没什么特殊的,字段参考: https://github.com/kohya-ss/sd-scripts

### MZ_KohyaSSTrain
训练主线程(Training main thread)
+ base_lora(基础lora): 加载一个lora模型后进行训练,和sd-scripts中的`network_weights`参数一致,启用时忽略dim/alpha/dropout(Train after loading a lora model, consistent with the `network_weights` parameter in sd-scripts, ignore dim/alpha/dropout when enabled)
+ sample_generate(启用样图生成): 每次保存模型时进行一次示例图片生成,并展示训练过程中每个保存epoch时的示例图片(Enable example image generation each time the model is saved, and display the example image at each saved epoch during training)
+ sample_prompt(提示词): 生成示例图片时使用的提示词(Phrase used when generating example images)


## FAQ



## Credits
+ [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
+ [https://github.com/kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)

## Star History

<a href="https://star-history.com/#MinusZoneAI/ComfyUI-TrainTools-MZ&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-TrainTools-MZ&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-TrainTools-MZ&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MinusZoneAI/ComfyUI-TrainTools-MZ&type=Date" />
 </picture>
</a>

## Contact
- 绿泡泡: minrszone
- Bilibili: [minus_zone](https://space.bilibili.com/5950992)
- 小红书: [MinusZoneAI](https://www.xiaohongshu.com/user/profile/5f072e990000000001005472)
- 爱发电: [MinusZoneAI](https://afdian.net/@MinusZoneAI)
