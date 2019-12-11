# Gesture_Rcg
<script type="text/javascript">
    var all_lang = ["eng", "cn"];
    var hide_elements_by_class = function(item, index){
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
        to_be_hide.forEach(hide_elements_by_class);
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
</form> 
## Install Anaconda & Pip
+ <p class="eng">As is well known for many DL developers, CNN is an art of data. In other words, CNN is a data-driven technique. Python is a perfect language choice on most cases where collecting and processing image data are highly involved. Anaconda provides most popular software packages to help us fulfill various image-related tasks. Please follow this official guide to install anaconda first(https://docs.anaconda.com/anaconda/install/linux/). </p>
  <p class="cn">
  众所周知深度学习是一门数据驱动的技术，我们在这个项目里选择使用 Python 进行大多数的图像处理工作。Anaconda 作为一个管理 Python 环境的工具，在未来的很多任务中，我们会很依赖它提供的帮助。请先依照官方指南完成 Anaconda 安装工作。
  </p>
  Once anaconda is properly installed, we need to create a seperate python environment for this project. Since we will use PaddlePaddle for DL development, let's call this new env 'paddle3_6' indicating a python==3.6 env.
  <details>
  <summary>中文</summary>
  当 Anaconda 安装完成后，我们需要为该项目创建一个虚拟环境，就叫它 paddle3_6 吧。
  </details>

  > conda create -n paddle3_6 python=3.6

  One more thing for the developers not lived in the States, you may observe a very slow internet connection while downloading packages from official conda website. If so, please change conda channel based on your region, it will make your life much eaiser. For mainland China, tsinghua provides an excellent mirror.  
  <details>
  <summary>中文</summary>
  对于居住在大陆的同学们，conda 的官方网站速度实在令人头大，可以考虑清华镜像源，但是清华源也会被关停，记得2019年就关过，现在能用就用，节约时间。
  </details>

  > conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  > conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  > conda config --set show_channel_urls yes

+ Though Anaconda is powerful enough for most of our daily work, some package is not perfectly maintained on it, e.g. OpenCV. As a workaround, pip is our choice when Anaconda fails.
  <details>
  <summary>中文</summary>
  虽然 Anaconda 提供了不错的 python 包管理安装环境，但是有些软件的版本目前维护的依旧不是很周全。比如我们亲爱的 OpenCV.所以需要 pip 工具作为 Anaconda 的辅助，来安装缺失的依赖。同样的我们也需要清华镜像保证稳定的下载速度。
  </details>

  > conda install pip # 安装 pip / install pip
  > pip install --upgrade pip # 升级 pip / upgrade pip
  > pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple # 配置清华源
## Install OpenCV & Access Webcam
OpenCV Python API should be installed via pip instead of Anaconda.
<details>
<summary>中文</summary>
请使用 pip 安装 OpenCV 的 contrib 版本（contrib 功能比较全）。
</details>

> conda install pyqt # 安装 Qt 图形界面 python 接口 / install Qt GUI python api
> pip install opencv-contrib-python==4.1.2.30 # 安装 opencv4 python 接口 / install opencv4 python api

Then our first python script rendering webcam on a Qt window should be something like data/util/webcam_render.py. Good to go from here. If you wanna learn more about OpenCV Python, check out this link: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
<details>
<summary>中文</summary>
OpenCV 的 Python 接口安装结束后需要测试一下，那就打开一个摄像头吧！更多的 OpenCV Python 接口使用请看 https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
</details>