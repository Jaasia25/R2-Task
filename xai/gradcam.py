import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil, sys, copy
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import getUtils
# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients # REMOVED: Redefined later

utils = getUtils()
config = utils.load_yaml()

MODEL_PATH = config['model_path']['MODEL']


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (top, bottom, left, right)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
        elif self.model.task == 'detect':
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'classify':
            return result[0]
    
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]
        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]
        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]
        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]
        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

    def release(self):
        for handle in self.handles:
            handle.remove()

class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)

class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'segment' or self.ouput_type == 'all':
                result.append(pre_post_mask[i].mean())
        return sum(result)

class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'pose' or self.ouput_type == 'all':
                result.append(pre_post_pose[i].mean())
        return sum(result)

class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'obb' or self.ouput_type == 'all':
                result.append(pre_post_angle[i])
        return sum(result)

class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        return data.max()

class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size):
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        print(f'model class info:{model_names}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False
        
        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
        else:
            raise Exception(f"not support task({task}).")
        
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())
    
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2) # 绘制检测框
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)  # 绘制类别、置信度
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    def process(self, img_path, save_path):
        # img process
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except:
            print(f"Warning... {img_path} read failure.")
            return
        original_image = img.copy() # Keep a copy of the original for plotting clarity, though not strictly needed here
        img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=True) # 如果需要完全固定成宽高一样就把auto设置为False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float_np = np.float32(img) / 255.0 # Renamed for clarity
        tensor = torch.from_numpy(np.transpose(img_float_np, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        print(f'tensor size:{tensor.size()}')
        
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            print(f"Warning... self.method(tensor, [self.target]) failure: {e}")
            return
        
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=True)
        
        # --- MODIFICATION START: Confidence Score and Heatmap Display ---
        pred_results = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7)
        if pred_results and len(pred_results) > 0:
            pred = pred_results[0]
            
            # 1. Extract and Print Confidence Score
            if pred.boxes.conf.numel() > 0:
                max_conf = pred.boxes.conf.max().item()
                print(f"✅ Max Detection Confidence: {max_conf:.4f}")
            else:
                print("⚠️ No detections found above the confidence threshold.")
            
            if self.renormalize and self.task in ['detect', 'segment', 'pose']:
                if pred.boxes.xyxy.numel() > 0:
                    # Renormalize CAM based on detected bounding boxes
                    cam_image = self.renormalize_cam_in_bounding_boxes(pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32), img_float_np, grayscale_cam)
            
            # Plot detections (boxes and labels) on top of the CAM image
            if self.show_result:
                # pred.plot returns a numpy array (image with drawings) in BGR format
                cam_image = pred.plot(img=cam_image * 255, # Plot expects 0-255 image
                                      conf=True, 
                                      font_size=None, 
                                      line_width=None, 
                                      labels=True, # Set to True to see class names/confidence
                                     )
                # Convert back to RGB for PIL saving, if pred.plot output is BGR (typical for cv2-based plot)
                cam_image = cv2.cvtColor(cam_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        else:
             print("⚠️ No predictions were generated.")
             cam_image = (cam_image * 255).astype(np.uint8) # Convert the basic CAM image to 0-255 uint8

        # Go back to original image size (remove padding)
        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        
        # 2. Save the Image
        cam_image = Image.fromarray(cam_image.astype(np.uint8)) # Ensure it's uint8 before PIL conversion
        cam_image.save(save_path)
        print(f"📸 Heatmap image saved to: {save_path}")
        # --- MODIFICATION END ---
    
    def __call__(self, img_path, save_dir): # Renamed save_path to save_dir for clarity
        # remove dir if exist
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        # make dir if not exist
        os.makedirs(save_dir, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                full_img_path = os.path.join(img_path, img_path_)
                full_save_path = os.path.join(save_dir, img_path_)
                self.process(full_img_path, full_save_path)
        else:
            # Handle single image case explicitly
            filename = os.path.basename(img_path)
            full_save_path = os.path.join(save_dir, f'heatmap_{filename}')
            self.process(img_path, full_save_path)

    def run_from_array(self, img_np):
    # Ensure directories exist
        temp_dir = "uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_input.jpg")
        cv2.imwrite(temp_path, img_np)

        save_dir = "heatmap_result"
        os.makedirs(save_dir, exist_ok=True)

        # Run explainable AI
        self(temp_path, save_dir)

        # 🔥 FIX: Auto-detect generated heatmap file
        heatmap_files = [f for f in os.listdir(save_dir) if f.lower().endswith(".jpg")]
        if not heatmap_files:
            raise FileNotFoundError("No heatmap image generated in heatmap_result folder.")

        heatmap_path = os.path.join(save_dir, heatmap_files[0])
        heatmap = cv2.imread(heatmap_path)

        return heatmap



        
def get_params():
    params = {
        # --- MODIFICATION HERE ---
        'weight': MODEL_PATH, # Updated to your specific model path
        # -------------------------
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu', 
        'method': 'GradCAMPlusPlus', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
        'layer': [10, 12, 14, 16, 18],
        'backward_type': 'all', 
        'conf_threshold': 0.2, 
        'ratio': 0.02, 
        'show_result': True, 
        'renormalize': False, 
        'task':'detect', 
        'img_size':640, 
    }
    return params



# if __name__ == '__main__':
    # params = get_params()
    # model_weight_path = params['weight']
    
    # # Check if the model file exists before proceeding
    # if not os.path.exists(model_weight_path):
    #     print(f"🚨 Error: Model weight file '{model_weight_path}' not found. Please verify the path.")
    #     sys.exit(1)
    
    # # Your model path is now correctly loaded in params
    # model = yolo_heatmap(**params)
    
    # image_path = 'explainable-ai/00000012_jpg.rf.05461a5f799e551c095311e92c4752a7.jpg'
    # save_dir = 'result'
    
    # if os.path.exists(image_path):
    #     model(image_path, save_dir)
    # else:
    #     print(f"🚨 Warning: Input image path '{image_path}' not found. Please provide a valid path.")