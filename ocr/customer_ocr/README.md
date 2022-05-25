## paddleocr
官网连接https://github.com/PaddlePaddle/PaddleOCR
paddleocr 需要先运行.py的脚本下载相关模型
paddlepaddle-gpu 版本注意支持的cuda版本

```bash
yum install libglvnd-glx
pip install paddlepaddle==2.1.1 -i https://mirror.baidu.com/pypi/simple
pip install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.
pip install "paddleocr>=2.2"
```
## teble ocr
目前只支持table3格式输出json