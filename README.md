# movie-auto-edit

自动识别电影主角，按音乐节拍生成混剪视频。

**工作流程**

```
电影.mp4
  ↓ PySceneDetect 场景分割
  ↓ ffmpeg 提取缩略图
  ↓ face_recognition 人脸聚类 / 参考图匹配
  ↓ librosa 节拍检测
  → 输出混剪.mp4（含配乐）
```

## 环境要求

- Python 3.9+
- ffmpeg（需加入系统 PATH）
- cmake（编译 dlib 用）

**安装 ffmpeg**

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载，解压后将 bin/ 加入 PATH
```

**安装 cmake**

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt install cmake

# Windows
# 从 https://cmake.org/download/ 下载安装包
```

## 安装

```bash
git clone https://github.com/YOUR_USERNAME/movie-auto-edit.git
cd movie-auto-edit

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **注意（Python 3.14+）**：若出现 `No module named 'pkg_resources'` 错误，
> 手动编辑 `venv/lib/pythonX.Y/site-packages/face_recognition_models/__init__.py`，
> 将 `from pkg_resources import resource_filename` 替换为 `import os as _os` 并用
> `os.path.join` 拼接路径。详见 [issue #1]。

## 使用方法

### 基本用法（自动识别主角）

```bash
python movie_edit.py 电影.mp4 音乐.mp3 输出.mp4
```

### 指定参考图片（精准匹配）

```bash
python movie_edit.py 电影.mp4 音乐.mp3 输出.mp4 --reference 演员照片.jpg
```

### 全部参数

```
positional arguments:
  movie               输入电影文件路径
  music               背景音乐文件路径
  output              输出视频文件路径

options:
  --reference, -r     主角参考图片（留空则自动识别出镜最多的主角）
  --top, -n           自动模式下识别几位主角，默认 2
  --tolerance, -t     人脸匹配容差 0.4~0.65，越小越严格，默认 0.55
```

### 示例

```bash
# 自动识别出镜最多的 2 位主角
python movie_edit.py "Harry Potter.mp4" bgm.mp3 edit.mp4

# 只保留指定演员的片段
python movie_edit.py "Harry Potter.mp4" bgm.mp3 edit.mp4 -r hermione.jpg

# 识别前 3 位主角，放宽匹配容差
python movie_edit.py movie.mp4 bgm.mp3 out.mp4 -n 3 -t 0.6
```

## 中间文件

脚本运行时会在电影同目录下创建 `_edit_<电影名>/` 工作目录：

```
_edit_电影名/
├── 01_clips/        场景分割后的片段
├── 02_thumbnails/   每段的缩略图
├── 03_matched/      含主角的片段（筛选结果）
└── 04_temp/         合成临时文件
```

处理完成后可手动删除该目录。

## 依赖说明

| 库 | 用途 |
|---|---|
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | 场景分割 |
| [face_recognition](https://github.com/ageitgey/face_recognition) | 人脸编码与比对 |
| [dlib](http://dlib.net/) | 人脸特征提取底层 |
| [ffmpy](https://github.com/Ch00k/ffmpy) | ffmpeg Python 封装 |
| [librosa](https://librosa.org/) | 音频节拍检测 |
| [scikit-learn](https://scikit-learn.org/) | DBSCAN 人脸聚类 |

## License

MIT
