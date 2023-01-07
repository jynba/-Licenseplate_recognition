import cv2
import numpy as np


def pretreatment(img):
    pic_hight, pic_width = img.shape[:2]
    MAX_WIDTH = 2000
    if pic_width > MAX_WIDTH:
        resize_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
    # 缩小图片
    return img


def find_Color(oldimg, img_contours, threshold):
    lower_blue = np.array([100, 110, 110])
    upper_blue = np.array([130, 255, 255])
    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([50, 255, 255])
    lower_green = np.array([50, 50, 50])
    upper_green = np.array([100, 255, 255])
    hsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)  # BGR转HSV
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow('output', output)
    # 形态学处理，闭运算和开运算，进行腐蚀和膨胀操作
    Matrix = np.ones((20, 20), np.uint8)
    img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
    #cv2.namedWindow("img_edge2", cv2.WINDOW_NORMAL)
    #cv2.imshow('img_edge2', img_edge2)
    # 阈值分割，过滤浅色的区域
    _, image_binary = cv2.threshold(img_edge2, threshold, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow("image_binary", cv2.WINDOW_NORMAL)
    # cv2.imshow('image_binary', image_binary)
    return image_binary


def filter_Region(oldimg, img, threshold):
    region = []
    # 查找外框轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("contours lenth is :%s" % (len(contours)))
    # 筛选面积小的
    list_rate = []
    maxPlateRatio = 5
    minPlateRatio = 2
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算轮廓面积
        area = cv2.contourArea(cnt)
        # 面积太小的忽略
        if area < 2000:
            continue
        # 转换成对应的矩形（最小）
        rect = cv2.minAreaRect(cnt)
        # print("rect is:%s" % {rect})
        # 根据矩形转成box类型，并int化
        box = np.int32(cv2.boxPoints(rect))
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        if float(height) == 0.0:
            continue
        # 正常情况车牌长高比在2-5之间
        ratio = float(width) / float(height)

        print("area", area, "ratio:", ratio, )
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        # 符合条件，加入到轮廓集合
        region.append(box)
        list_rate.append(ratio)

    print(list_rate)
    print(threshold)
    # 找出长宽比最接近3.2的
    for index, key in enumerate(list_rate):
        list_rate[index] = abs(key - 3.2)
    # 没获取到,递归修改threshold，直到获取到
    if (len(list_rate) == 0):
        return
    index = list_rate.index(min(list_rate))
    region = region[index]
    # print(list_rate.index(min(list_rate)))
    # newimg = cv2.drawContours(oldimg.copy(), [region], 0, (0, 255, 0), 2)
    # cv2.namedWindow("newimg", cv2.WINDOW_NORMAL)
    # cv2.imshow('newimg', newimg)

    # 将车牌图块单独截取出来
    Xs = [i[0] for i in region]
    YS = [i[1] for i in region]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(YS)
    y2 = max(YS)
    height_1 = y2 - y1
    width_1 = x2 - x1
    img_crop = oldimg[y1:y1 + height_1, x1:x1 + width_1]
    return img_crop


def find_edge(oldimg, img):
    # 高斯模糊+中值滤波
    img_gaus = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯模糊
    img_med = cv2.medianBlur(img_gaus, 5)  # 中值滤波
    # 进行Sobel算子运算，直至二值化
    img_gray_s = cv2.cvtColor(img_med, cv2.COLOR_BGR2GRAY)
    # sobel算子运算
    img_sobel_x = cv2.Sobel(img_gray_s, cv2.CV_32F, 1, 0, ksize=3)  # x轴Sobel运算
    img_sobel_y = cv2.Sobel(img_gray_s, cv2.CV_32F, 0, 1, ksize=3)
    img_ab_y = np.uint8(np.absolute(img_sobel_y))
    img_ab_x = np.uint8(np.absolute(img_sobel_x))  # 像素点取绝对值
    img_ab = cv2.addWeighted(img_ab_x, 0.5, img_ab_y, 0.5, 0)  # 将两幅图像叠加在一起（按一定权值）
    # 再加一次高斯去噪
    img_gaus_1 = cv2.GaussianBlur(img_ab, (5, 5), 0)  # 高斯模糊
    ret2, img_thre_s = cv2.threshold(img_gaus_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 正二值化
    #cv2.namedWindow("img_thre_s", cv2.WINDOW_NORMAL)
    #cv2.imshow('img_thre_s', img_thre_s)
    # 对比canny边缘检测
    # img = cv2.resize(img, (600, 400))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 13, 15, 15)
    # edges = cv2.Canny(gray, 30, 200)
    # cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    # cv2.imshow('edges', edges)
    return img_thre_s


def combine_color_edge(oldimg, filter_color, filter_edge):
    # 颜色空间与边缘算子的图像互相筛选
    # 同时遍历两幅二值图片，若两者均为255，则置255
    img_1 = np.zeros(oldimg.shape, np.uint8)  # 重新拷贝图片
    height = oldimg.shape[0]  # 行数
    width = oldimg.shape[1]  # 列数
    print(filter_edge)
    for i in range(height):
        for j in range(width):
            h = filter_color[i][j]
            s = filter_edge[i][j]
            if h == 255 and s == 255:
                img_1[i][j] = 255
            else:
                img_1[i][j] = 0
    # cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
    # cv2.imshow('threshold', img_1)

    # 二值化后的图像进行闭操作
    kernel = np.ones((14, 18), np.uint8)
    img_close = cv2.morphologyEx(img_1, cv2.MORPH_CLOSE, kernel)  # 闭操作
    img_med = cv2.medianBlur(img_close, 5)
    # cv2.namedWindow("close", cv2.WINDOW_NORMAL)
    # cv2.imshow('close', img_med)
    return img_med


def check_Final(img, final, threshold):
    # 循环直到阈值满足至少出现一个满足条件的结果,避免颜色太浅，阈值太高，识别不出
    while final is None and threshold > 0:
        threshold -= 1
        filter_color = find_Color(img, img, threshold)
        final = filter_Region(img, filter_color, threshold)
    if final is not None:
        return final

def main():
    for i in range(1, 6):
        print("****************** ", i, " *****************")
        s = "./img/car_g{i}.jpg".format(i=i)
        # s = "./img/car_y{i}.jpg".format(i=i)
        img = cv2.imread(s)
        threshold = 128  # 阈值设置为128，去噪
        filter_color = find_Color(img, img, threshold)
        final = filter_Region(img, filter_color, threshold)  # 轮廓提取并筛选
        # 循环直到阈值满足至少出现一个满足条件的结果,避免颜色太浅，阈值太高，识别不出
        while final is None and threshold > 0:
            threshold -= 1
            filter_color = find_Color(img, img, threshold)
            final = filter_Region(img, filter_color, threshold)
        if final is not None:
            cv2.namedWindow("final", cv2.WINDOW_NORMAL)
            cv2.imshow('final', final)
        cv2.waitKey(0)

    # img1 = cv2.imread('./img/car24.jpg')
    # img1 = pretreatment(img1)
    # cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    # cv2.imshow('img1', img1)
    # threshold = 128  # 阈值设置为128，去噪
    # filter_color = find_Color(img1, img1, threshold)  # 根据颜色区分 ;
    #
    # # filter_edge = find_edge(img1, img1)  # 根据边缘区分
    # # combine = combine_color_edge(img1, filter_edge, filter_color)
    #
    # final = filter_Region(img1, filter_color, threshold)  # 轮廓提取并筛选
    # # 循环直到阈值满足至少出现一个满足条件的结果,避免颜色太浅，阈值太高，识别不出
    # while final is None and threshold > 0:
    #     threshold -= 1
    #     filter_color = find_Color(img1, img1, threshold)
    #     final = filter_Region(img1, filter_color, threshold)
    # if final is not None:
    #     cv2.namedWindow("final", cv2.WINDOW_NORMAL)
    #     cv2.imshow('final', final)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
