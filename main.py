import pytesseract
import cv2
import LP_cut
import LP_recognition

if __name__ == '__main__':
    print(pytesseract.get_languages(config=''))
    url='./img/car7.jpg'
    img1 = cv2.imread(url)
    cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    cv2.imshow('img1', img1)
    filter_color = LP_cut.find_Color(img1, img1)  # 根据颜色区分
    # filter_edge = find_edge(img1, img1)  # 根据边缘区分
    # combine = combine_color_edge(img1, filter_edge, filter_color)
    LP_recognition.recognition(url)
    final = LP_cut.filter_Region(img1, filter_color)  # 轮廓提取并筛选
    cv2.imwrite("final.jpg", final)
    cv2.namedWindow("final", cv2.WINDOW_NORMAL)
    cv2.imshow('final', final)
    cv2.waitKey(0)

    # pytesseract只能提取字母和数字
    # print(pytesseract.image_to_string(final,
    #                                   config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    #                                   ))
