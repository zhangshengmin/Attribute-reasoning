from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# model_type = "vit_t"
# sam_checkpoint = "../weights/mobile_sam.pt"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# mobile_sam.to(device=device)
# mobile_sam.eval()

# predictor = SamPredictor(mobile_sam)
# img=Image.open("/home/ur/Desktop/MobileSAM/app/assets/picture1.jpg")
# predictor.set_image(img)
# masks, _, _ = predictor.predict()
 
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

image = cv2.imread(r"/home/ur/Desktop/yolov5_detect/yolov5-pytorch/VOCdevkit/VOC2007/JPEGImages/4.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model_type = "vit_t"
sam_checkpoint = "../weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(image)
# print(masks)
 
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()