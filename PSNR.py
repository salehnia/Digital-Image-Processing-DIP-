import glob
import cv2
import numpy as np
import time
import skimage.measure

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    p = skimage.measure.compare_psnr(imageA, imageB)
    print("MSE: %.2f, PSNR: %.2f" % (m, p))
    return (p)


result1 =[]
de = []
counter_result = 0
f_res = []
d = 0
c = 0
start = time.time()
all_result =[]
res1=[]
class_names =[]
result = []
test_filenames = glob.glob("test/" + "*.jpg")
for test_index in test_filenames:
    original = cv2.imread(test_index)
    test_class = test_index.split('.jpg')[0].split(',')[2]
    for index in range(-861,2055,15):
        yclass = "c" + str(index)
        filenames = glob.glob("data/"+ str(yclass)+ "/*.jpg")
        filenames.sort()
        if len(filenames) != 0:
            label_class = filenames[0].split('.jpg')[0].split(',')[2]
            print('Start in this Class:' + str(label_class))
        for image_path in filenames:
                img = cv2.imread(image_path)
                images = ("Original",original), ("Train", img)
                rs = compare_images(original, img)
                result1.append(rs)
                res1.append((rs,label_class,test_class))

    f_res.append(max(result1))   # max har daste
    maxm = max(result1)
    result1 = []
all_result.append(maxm)
print(maxm)
print('End of All Classes')
print('===========================================')
for ind in f_res:
    if ind >= 18.00:
        counter_result = counter_result + 1
        print(ind)
        print("=================================")
        print(counter_result)
        print(len(test_filenames))
final_res = float(counter_result/len(test_filenames))*100
print("Accuracy is:")
print(final_res)