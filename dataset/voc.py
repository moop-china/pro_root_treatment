import os
import sys
import tarfile
import collections
from .vision import VisionDataset

import torch

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
    
from PIL import Image

RETINA_CLASSES = {
    'root': 0,
    'tooth': 1
}

def convert_voc_2_retina(anno, class_mapping):
    names = []
    bboxes = []
    
    if isinstance(anno['annotation']['object'], list):
        for obj in anno['annotation']['object']:
            names.append(obj['name'])

            box = obj['bndbox']
            bboxes.append([
                int(box['xmin']),
                int(box['ymin']),
                int(box['xmax']),
                int(box['ymax'])
            ])
    else:
        obj = anno['annotation']['object']
        
        names.append(obj['name'])

        box = obj['bndbox']
        bboxes.append([
            int(box['xmin']),
            int(box['ymin']),
            int(box['xmax']),
            int(box['ymax'])
        ])
        
    classes = torch.tensor([class_mapping[name] for name in names], dtype=torch.long)
    
    # TODO : convert bbox format here, if needed
    bboxes = torch.tensor(bboxes, dtype=torch.long)
    
    # return an empty dict for transform meta-data
    return classes, bboxes, {}

# customized collate() for multi-bbox sample
def collate(samples):
    imgs = []
    annos = []
    
    for sample in samples:
        imgs.append(sample[0])
        annos.append(sample[1])
    
    imgs = torch.stack(imgs)
    
    # imgs is a tensor, while annos is a list
    return imgs, annos

class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)
        self.root = root
        self.image_set = image_set

        voc_root = self.root
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid'
                'image_set from the VOC ImageSets/Main folder.')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        
        anno = convert_voc_2_retina(target, RETINA_CLASSES)
        
        # print('anno: {}'.format(anno))
        
        if self.transforms is not None:
            img, anno = self.transforms(img, anno)

        return img, anno

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
        