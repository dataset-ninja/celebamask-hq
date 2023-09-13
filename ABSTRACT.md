**CelebAMask-HQ** is a large-scale face image dataset that has 30,000 high-resolution face images selected from the CelebA dataset by following CelebA-HQ. Each image has segmentation mask of facial attributes corresponding to CelebA.

Authors built a large-scale face semantic label dataset, which was labeled according to CelebAHQ that contains 30,000 high-resolution face images
from CelebA. It has several appealing properties:

- Comprehensive Annotations. CelebAMask-HQ was precisely hand-annotated with the size of 512 × 512 and 19 classes including all facial components and accessories such as *skin*, *nose*, *eyes*, *eyebrows*, *ears*, *mouth*, *lip*, *hair*, *hat*, *eyeglass*, *earring*, *necklace*, *neck*, and *cloth*.

![Some samples](https://i.ibb.co/b17nrtW/sample.png)

- Label Size Selection. The size of images in CelebAHQ were 1024 × 1024. However, we chose the size of 512 × 512 because the cost of the labeling would be quite high for labeling the face at 1024 × 1024. Besides, we could easily extend the labels from 512 × 512 to 1024 × 1024 by nearest-neighbor interpolation without introducing noticeable artifacts.

- Quality Control. After manual labeling, we had a quality control check on every single segmentation mask. Furthermore, we asked annotaters to refine all masks with several rounds of iterations.

- Amodal Handling. For occlusion handling, if the facial component was partly occluded, we asked annotators to label the occluded parts of the components by human inferring. On the other hand, we skipped the annotations for those components that are totally occluded.

**CelebAMask-HQ** can be used to train and evaluate algorithms of face parsing, face recognition, and GANs for face generation and editing.
