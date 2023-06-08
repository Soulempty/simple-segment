import os
import cv2
from apis.infer import SegmentModel


# example for infer api call.
def infer(model_path,image_path,save_path='work_dirs/result',config_path=None):
    os.makedirs(save_path,exist_ok=True)
    runtime = SegmentModel(config_path)
    runtime.set_model_path(model_path)
    runtime.load_model()
    paths = []
    imgs = []
    for i,f in enumerate(os.scandir(image_path)):
        path = f.path
        image = cv2.imread(path)
        paths.append(path)
        imgs.append(image)
        if i==10:
            break
    
    # infer with image path.
    results = runtime.predict(paths)
    for filename in results:
        save_p = os.path.join(save_path,filename)
        cv2.imwrite(save_p,results[filename])

    # infer with cv2 image with numpy array type,filename is index by order in list.
    results = runtime.predict(imgs)
    for filename in results:
        result = results[filename]
        # save_p = os.path.join(save_path,filename)
        # cv2.imwrite(save_p,results[filename])

if __name__ == "__main__":
    model_path = 'demo/model/unet.trt' # 'demo/model/best.pth' 
    image_path = 'demo/images'
    infer(model_path,image_path=image_path)