#### 一、环境准备

1. 下载tesseract库

   参考https://blog.csdn.net/u010698107/article/details/121736386
   注意要设置tesseract环境变量，在pytesseract中找到对于路径进行设置
   导入接口pytesseract

2. 百度API

   

#### 二、详细方案

* **预处理**

  **颜色区分**

1. 提取蓝绿黄三种车牌的HSV

2. 形态学处理

3. 阈值分割，过滤浅色的区域

   **边缘区分**（略）

* **识别车牌**

1. 筛选轮廓（面积、长宽比）
2. 根据轮廓截取图片
3. 文字识别

