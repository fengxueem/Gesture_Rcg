# Gesture_Rcg
This repo will record everything of building a DL model recognizing human hand gesture. Things are arranged chronologically, I think it's a good way to leave the trace of how to solve different problems.
<details>
<summary>中文</summary>
我会在这里记录自己开发手势识别深度学习算法的所有步骤，基本上按时间顺序记录。希望可以为未来翻看时候留下发现问题解决问题的思路流程。
</details>
<!-- <script type="text/javascript">
    var all_lang = ["eng", "cn"];
    var hideElementsByClass = function(item, index){
    	var selected_lang_elem = document.getElementsByClassName(item);
        for (var i=0, len=selected_lang_elem.length|0; i<len; i=i+1|0) {
            selected_lang_elem[i].hidden=true;
        }
    };
    var updateLang = function(t){
        // 隐藏其余class
        var to_be_hide = [];
        for( var i = 0; i < all_lang.length; i++){ 
   			if ( all_lang[i] != t.value) {
     			to_be_hide.push(all_lang[i]); 
   			}
		}
        to_be_hide.forEach(hideElementsByClass);
        // 显示选中class
        var selected_lang_elem = document.getElementsByClassName(t.value);
        for (var i=0, len=selected_lang_elem.length|0; i<len; i=i+1|0) {
            selected_lang_elem[i].hidden=false;
        }
    };
</script>
<form>
 <select onchange="updateLang(this)" id="sel_lang">
 <option value='eng' selected>English</option>
 <option value='cn'>中文</option>
 </select>
</form>  -->

## Install Anaconda & Pip
+ As is well known for many DL developers, CNN is an art of data. In other words, CNN is a data-driven technique. Python is a perfect language choice on most cases where collecting and processing image data are highly involved. Anaconda provides most popular software packages to help us fulfill various image-related tasks. Please follow this [official guide](https://docs.anaconda.com/anaconda/install/linux/) to install anaconda first.
  <details>
  <summary>中文</summary>
  众所周知深度学习是一门数据驱动的技术，我们在这个项目里选择使用 Python 进行大多数的图像处理工作。Anaconda 作为一个管理 Python 环境的工具，在未来的很多任务中，我们会很依赖它提供的帮助。请先依照官方指南 https://docs.anaconda.com/anaconda/install/linux/ 完成 Anaconda 安装工作。
  </details>
  Once anaconda is properly installed, we need to create a seperate python environment for this project. Since we will use PaddlePaddle for DL development, let's call this new env 'paddle3_6' indicating a python==3.6 env.
  <details>
  <summary>中文</summary>
  当 Anaconda 安装完成后，我们需要为该项目创建一个虚拟环境，就叫它 paddle3_6 吧。
  </details>

  ```bash
  conda create -n paddle3_6 python=3.6
  ```
  One more thing for the developers not lived in the States, you may observe a very slow internet connection while downloading packages from official conda website. If so, please change conda channel based on your region, it will make your life much eaiser. For mainland China, tsinghua provides an excellent mirror.  
  <details>
  <summary>中文</summary>
  对于居住在大陆的同学们，conda 的官方网站速度实在令人头大，可以考虑清华镜像源，但是清华源也会被关停，记得2019年就关过，现在能用就用，节约时间。
  </details>

  ```bash
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --set show_channel_urls yes
  ```

+ Though Anaconda is powerful enough for most of our daily work, some package is not perfectly maintained on it, e.g. OpenCV. As a workaround, pip is our choice when Anaconda fails.
  <details>
  <summary>中文</summary>
  虽然 Anaconda 提供了不错的 python 包管理安装环境，但是有些软件的版本目前维护的依旧不是很周全。比如我们亲爱的 OpenCV.所以需要 pip 工具作为 Anaconda 的辅助，来安装缺失的依赖。同样的我们也需要清华镜像保证稳定的下载速度。
  </details>

  ```bash
  conda install pip # 安装 pip / install pip
  pip install --upgrade pip # 升级 pip / upgrade pip
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple # 配置清华源 / setup tsinghua as default
  ```
## Install OpenCV & Access Webcam
OpenCV Python API should be installed via pip instead of Anaconda.
<details>
<summary>中文</summary>
请使用 pip 安装 OpenCV 的 contrib 版本（contrib 功能比较全）。
</details>

```
conda install pyqt # 安装 Qt 图形界面 python 接口 / install Qt GUI python api
pip install opencv-contrib-python==4.1.2.30 # 安装 opencv4 python 接口 / install opencv4 python api
```
Then our first python script rendering webcam on a Qt window should be something like data/util/webcam_render.py. Good to go from here. If you wanna learn more about OpenCV Python, check out this [link](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html). Now, it's time to save lots of images for future training purposes. And a successful model should be fed with clean and extremely adequate chunks of images, say 30k(25k for training, 5k for testing) for each class.
<details>
<summary>中文</summary>
OpenCV 的 Python 接口安装结束后需要测试一下，那就打开一个摄像头吧,示例代码可以参考 data/util/webcam_render.py。更多的 OpenCV Python 接口使用，请看https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html。下一步是获取足够多干净的图片作为模型训练用途(25k张/类)以及测试集(5k张/类)。
</details>

## Create custom dataset
In real world applications, a prerequisite for developing a working DL model is to generate enough images captured by the camera you choose for inferencing stage. There is a simple python script(data/util/gen_data.py) for generating black and white images with the size of 112 * 112. Running this script under different lighting conditions and ask as many people you can find as possible to be a 5 to 10 mins hand model. We will end up with a big enough dataset.
Before you run the script, please install libsvm and an awesome non-reference image quality repo named [BRISQUE](https://github.com/bukalapak/pybrisque).
<details>
<summary>中文</summary>
在实际场景中，由于获取任务需求数据困难，深度学习算法可能需要与某种参数的相机对应，达到单个模型过拟合且鲁棒性差的效果。当没有光学工程师帮我们调节相机成像时，我们就只能用未来执行该算法的相机采图。这里写了一个简单采图脚本，采集大小为112*112的灰度图。一共计划每类手势30k张图，因此找越多的人采图越好，并且尽可能在不同的光照条件下采集。在开始采集前，需要在 python 环境下安装 livsvm 与 pybrisque，用于判别图像质量。
</details>

```bash
pip install libsvm # 安装 libsvm / install libsvm
pip install pybrisque # 安装 pybrisque / install pybrisque
python data/util/gen_data.py 45 awesome 1500 # 采集模糊度小于45的1500张图像，文件名前缀awesome / generate and save 1500 images with max blurring rate 45, save them with a prefix "awesome"
```