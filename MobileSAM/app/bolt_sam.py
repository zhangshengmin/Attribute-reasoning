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

def mask2img(mask):
    # top = mask['bbox'][1] - 0.5 * mask['bbox'][3]
    # left = mask['bbox'][0] - 0.5 * mask['bbox'][2]
    # bottom = mask['bbox'][1] + 0.5 * mask['bbox'][3]
    # right = mask['bbox'][0] + 0.5 * mask['bbox'][2]
    top = mask['bbox'][1]
    left = mask['bbox'][0]
    bottom = mask['bbox'][1] +1 * mask['bbox'][3]
    right = mask['bbox'][0] +1 * mask['bbox'][2]
    mask = ~mask['segmentation']
    mask = mask + 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = mask.astype(np.uint8)
    res = cv2.bitwise_and(image, mask)
    res[res == 0] = 255
    img = Image.fromarray(res)
    img.show()
    
    top = max(0, np.floor(top).astype('int32'))
    left = max(0, np.floor(left).astype('int32'))
    bottom = min(img.size[1], np.floor(bottom).astype('int32'))
    right  = min(img.size[0], np.floor(right).astype('int32'))
    crop_img = img.crop([left, top, right, bottom])
    crop_img.show()

def show_points(coords, ax, marker_size=375):
    points = coords
    print(points)
    
    ax.scatter(points[0], points[1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # print(sorted_anns)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        if  ann['area']<10000:
            center_point=[ann['bbox'][0] + 0.5 * ann['bbox'][2],ann['bbox'][1] + 0.5 * ann['bbox'][3]]
            print('area:',ann['area'],'point_coords:',center_point)
            mask2img(ann)
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            show_points(center_point, ax, marker_size=375)
            ax.imshow(np.dstack((img, m*0.35)))

def global_segment(model,image):
    mask_generator = SamAutomaticMaskGenerator(model)
    masks = mask_generator.generate(image)
    # print(masks)
    
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

def point_show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def point_segment(model,image,point):
    predictor = SamPredictor(model)
    predictor.set_image(image)
    input_point = point
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)
 
    # 遍历读取每个扣出的结果
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        point_show_mask(mask, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


image = cv2.imread(r"/home/ur/Desktop/yolov5_detect/yolov5-pytorch/VOCdevkit/VOC2007/JPEGImages/4.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model_type = "vit_t"
sam_checkpoint = "../weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()


# global_segment(mobile_sam,image)
point=np.array([[1107.5, 269.5]])
point_segment(mobile_sam,image,point)