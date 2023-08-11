import pickle
import cv2
import numpy as np

def log_inference(uuid, image, outputs, img_dir, var_dir):

    def save_pickle(var, file_name):
        with open(var_dir + f'{file_name}.pickle', 'wb') as f:
            pickle.dump(var, f)
        return None
    
    cv2.imwrite(img_dir + str(uuid) + '.jpg', image)
    outputs_file_name = f"{str(uuid)}"
    save_pickle(outputs, outputs_file_name)
    return None

def draw_box_and_save(uuid, image, outputs, img_dir):
    for output in outputs:
        height, width = image.shape[:2]
        xMin, xMax, yMin, yMax = output["xMin"], output["xMax"], output["yMin"], output["yMax"]
        # Rescale original pixel size
        xMin, xMax, yMin, yMax = int(xMin * width), int(xMax * width), int(yMin * height), int(yMax * height)
        start_point = (xMin, yMax)
        end_point = (xMax, yMin)
        color = [int(c) for c in np.random.choice(range(256), size=3)]
        label = output["class"].capitalize()
        score = output["score"]
        text = f"{label} - {str(score)}"
        org = (xMin, yMax)
        image = cv2.rectangle(image, start_point, end_point, color)
        image = cv2.putText(image, text=text, org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=color)
    cv2.imwrite(img_dir + str(uuid) + '_out' + '.jpg', image)
    return None
