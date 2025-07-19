"""
original code from WZMIAOMIAO:
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer
"""
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
import numpy as np


def transform_3(img,transform):
    w, h = img.size
    area = w * h
    img_numpy_from_PIL = np.array(img)
    img_2 = img_numpy_from_PIL[int(h / 4):int(h / 4) + int(h / 2), int(w / 4):int(w / 4) + int(w / 2), :]

    sl = 0.02
    sh = 0.1
    r1 = 0.3
    mean = [0.4914, 0.4822, 0.4465]
    block_4 = []
    for i in range(4):
        while True:
            target_area = np.random.uniform(sl, sh) * area
            aspect_ratio = np.random.uniform(r1, 1 / r1)
            block_h = int(round(np.math.sqrt(target_area * aspect_ratio)))
            block_w = int(round(np.math.sqrt(target_area / aspect_ratio)))
            if block_h < h and block_w < w:
                block_4.append([block_h, block_w])
                break

    img_3 = img_numpy_from_PIL.copy()
    decision_array = np.random.randint(0, 2, size=4)
    if decision_array[0] == 1:
        img_3[0:block_4[0][0], 0:block_4[0][1], :] = mean
    if decision_array[1] == 1:
        img_3[0:block_4[1][0], w - block_4[1][1]:, :] = mean
    if decision_array[2] == 1:
        img_3[h - block_4[2][0]:, 0:block_4[2][1], :] = mean
    if decision_array[3] == 1:
        img_3[h - block_4[3][0]:, w - block_4[3][1]:, :] = mean

    img_2 = Image.fromarray(img_2)
    img_3 = Image.fromarray(img_3)

    img = transform(img)
    img_2 = transform(img_2)
    img_3 = transform(img_3)
    img = torch.stack([img, img_2, img_3], dim=0)
    return img



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)


    # create model
    model = create_model(num_classes=30, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/model-99.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    img_path = "./food-30-test/NaiHuangBao/HDCP-Bai_Se_Dan_Er_Wan_20201201_141049_4_02.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    if (len(img.getbands()) == 3):
        img=transform_3(img,data_transform)

        with torch.no_grad():
            pred ,pred_add,features_4,features_add, u1, u2, u3 = model(img.to(device))

            output = torch.squeeze(pred_add).cpu()
            predict = torch.softmax(output, dim=0)#类别概率

            predict_cla = torch.argmax(predict).numpy()#返回概率最大值所在的索引
            _,predict_cla_5=torch.topk(predict,5)#返回概率前五的索引
            predict_cla_5=predict_cla_5.numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())

        plt.title(print_res)
        if class_indict[str(predict_cla)] == "NaiHuangBao":
            print("TOP-1: True")


        for i in range(len(predict_cla_5)):
            if class_indict[str(predict_cla_5[i])] == "NaiHuangBao":
                print("TOP-5: True")
                break

        for i in range(len(predict_cla_5)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(predict_cla_5[i])] ,
                                                      predict[predict_cla_5[i]].numpy()))

        plt.show()


if __name__ == '__main__':
    main()
