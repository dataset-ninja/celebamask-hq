**CelebAMask-HQ** is an extensive dataset featuring 30,000 high-resolution face images. These images were carefully chosen from the CelebA dataset by adhering to the CelebA-HQ guidelines. Each image in this dataset comes with a segmentation mask outlining facial attributes that correspond to the CelebA dataset. This initiative resulted in the creation of a substantial dataset for face attribute labeling, all based on the CelebAHQ collection consisting of 30,000 high-resolution face images from CelebA.

It has several appealing properties:

- Comprehensive Annotations. CelebAMask-HQ was precisely hand-annotated with the size of 512×512 and 19 classes including all facial components and accessories such as _skin_, _nose_, _eyes_, _eyebrows_, _ears_, _mouth_, _lip_, _hair_, _hat_, _eyeglass_, _earring_, _necklace_, _neck_, and _cloth_.

![Some samples](https://i.ibb.co/b17nrtW/sample.png)

- Label Size Selection. The size of images in CelebAHQ were 1024×1024. However, we chose the size of 512×512 because the cost of the labeling would be quite high for labeling the face at 1024×1024. Besides, we could easily extend the labels from 512×512 to 1024x1024 by nearest-neighbor interpolation without introducing noticeable artifacts.

- Quality Control. After manual labeling, we had a quality control check on every single segmentation mask. Furthermore, we asked annotators to refine all masks with several rounds of iterations.

- Amodal Handling. For occlusion handling, if the facial component was partly occluded, we asked annotators to label the occluded parts of the components by human inferring. On the other hand, we skipped the annotations for those components that are totally occluded.

CelebAMask-HQ can be used to train and evaluate algorithms of face parsing, face recognition, and GANs for face generation and editing.
