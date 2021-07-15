from numpy.core.numeric import True_
import torch
import torchvision
import numpy as np
import cv2

from utils.utils import *
from utils.datasets import *

from models import *
from PIL import Image
from torchvision import transforms

SELECT_CLASS = 16
IMAGE_SIZES = [416, 512, 800]


def crop_segment(det, image):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    segments = []
    idx = 0
    for *box, conf, cls in det:

        cpd = image[int(box[1]): int(box[3]), int(
            box[0]): int(box[2]), ].copy()
        cpd = cv2.resize(cpd, (299, 299))
        im = Image.fromarray(np.uint8(cpd))
        segments.append(im)

    return segments


class InceptionClassifier:

    def __init__(self):

        self.device = torch_utils.select_device(device='cpu')

        print('loading classifier')
        self.model = torchvision.models.inception.Inception3(
            transform_input=True)
        print('..............')
        self.model.aux_logits = False

        for param in self.model.parameters():
            param.requires_grad = False

        n_classes = 120

        n_inputs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(n_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, n_classes),
            nn.LogSoftmax(dim=1)
        )

        self.model.load_state_dict(torch.load(
            'weights/dog_inception.pt', map_location=self.device)['state_dict'])

        self.transforms = transforms.Compose([
            transforms.Resize(size=299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model.eval()

    def infer(self, cropped_segments):

        tensors = []
        for segment in cropped_segments:
            tensor = self.transforms(segment).float()
            tensors.append(tensor)

        tensors = torch.stack(tensors)

        with torch.no_grad():
            logits = self.model(tensors)
            result = torch.nn.functional.softmax(logits, dim=1)

        max_vals = torch.max(result, 1)

        return max_vals


class Yolov3:

    def __init__(self, save_results=True):

        self.classes = load_classes('./weights/coco.names')
        self.image_size = (IMAGE_SIZES[0], IMAGE_SIZES[0])
        self.model = Darknet('./weights/yolov3-spp.cfg',
                             img_size=self.image_size)

        self.device = torch_utils.select_device(device='cpu')
        self.classes_dogs = load_classes('./weights/labels.txt')

        self.model.load_state_dict(torch.load(
            './weights/yolov3-spp.pt', map_location=self.device)['model'])
        self.classifier = InceptionClassifier()

        if not os.path.exists('results'):
            os.mkdir('results')

        self.model.to(self.device).eval()
        self.current_files = []
        self.originals = []

        self.save_results = save_results

    def __preprocess(self, path):

        images = LoadImages(path=path)

        dataset = []

        for path, img, im0s, vid_cap in images:

            img = torch.from_numpy(img).to(self.device)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            img = img / 255.

            img_fname = path.split("/")[-1]
            self.current_files.append(img_fname)

            dataset.append(img)
            self.originals.append(im0s)

        return dataset

    def __postprocess(self, results, img, skip_classification=True):

        for i, det in enumerate(results):

            save_path = os.path.join('results', self.current_files[i])
            im0 = self.originals[i]

            if det is not None and len(det):

                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                det2 = det.clone()
                if not skip_classification:
                    segments = crop_segment(det, im0)
                    max_vals = self.classifier.infer(segments)

                    conf, indices = max_vals.values, max_vals.indices
                    print(det, conf, indices)
                    for idx, d in enumerate(det):
                        det2[idx][4] = conf[idx]
                        det2[idx][5] = indices[idx]

                for *xyxy, conf, cls in det2:

                    print(xyxy[0], xyxy[1], xyxy[2], xyxy[3])

                    if self.save_results:

                        label = '%s %.2f' % (self.classes_dogs[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=(0, 0, 0))

            if self.save_results:
                print("coming from serce", save_path)
                cv2.imwrite(save_path, im0)

    def predict(self, path):

        dataset = self.__preprocess(path)

        predictions = []
        for img in dataset:

            result = self.model(img.float())[0].float()
            result = non_max_suppression(
                result, conf_thres=0.3, classes=SELECT_CLASS)
            predictions.append(result)

        for idx, prediction in enumerate(predictions):
            self.__postprocess(
                prediction, dataset[idx], skip_classification=False)

    def clear(self):

        self.current_files.clear()
        self.originals.clear()
