# scene-generation-from-novel-viewpoints-gens

We investigate whether [Graph Element Networks](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1904.09019&sa=D) (a graph convolutional neural network architecture that we published in ICML 2019) can be used to organize memories spatially, in the problem of generating scene images from novel viewpoints.

We sample 3D mazes from the [DeepMind Lab game platform](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1612.03801.pdf&sa=D) ([dataset](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fdeepmind%2Fgqn-datasets&sa=D)) and each maze comes with a series of images. Each image reveals how the maze appears, from a specific 2D coordinate and given a specific (yaw, pitch, roll) triple for the camera. 

![GUI wrapper preview](https://github.com/jaks19/scene-generation-from-novel-viewpoints-gens/blob/master/gifs/scene.gif)

In the animation, we have  mazes placed in a 3x3 grid structure. The animation shows generated scenes on the left and a top-down view of the 9 mazes on the right. We first sample views from different places inside the mazes, and insert them into the GEN. We then query the GEN for the inferred view at new query coordinates, while rotating 360 degrees for each position. The red nodes (in the top-down map) are active nodes from which information is interpolated to generate a new view, for each query location.

In this problem, the GEN:
* has its nodes spread across the 2D ground plane of the mazes (see white circles in right image)
* learns a useful representation for what mazes look like and we interpolate information from its nodes to generate new images
* compartmentalizes spatial memories since it trains on mazes one by one but at test time succeeds in absorbing information from 9 mazes simultaneously

How do we decode node states to draw scene images? This work was done to improve on [Deepmind's work](https://www.google.com/url?q=https%3A%2F%2Fscience.sciencemag.org%2Fcontent%2F360%2F6394%2F1204&sa=D) (Eslami et. al.) where they have a representation-learning network and an image-generation network ressembling the standard [DRAW](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1502.04623.pdf&sa=D) architecture. They can only represent one maze at a time as their model absorbs information without spatial disentangling. We use our GENs for representation learning, and apply their standard drawing architecture to decode our hidden states.

# Running experiments:

## Running our scene generation code 
If you'd like to get the pipeline running before committing to downloading and processing huge amounts of data, we provided a few processed .pt files in the data_samples folder. These can be directly with our scene generation code:

* Run our scene generation code by running 
```
python train_scene_rendering.py
```
with arguments matching our argparse header:
```
'--dataset', type=str, default='Labyrinth', help='dataset (dafault: Shepard-Mtzler)'
'--train_data_dir', type=str, help='location of training data', 
default="/home/jaks19/mazes-torch/train")
'--test_data_dir', type=str, help='location of test data', 
default="/home/jaks19/mazes-torch/test")
'--root_log_dir', type=str, help='root location of log', default='/home/jaks19/logs/'
'--log_dir', type=str, help='log directory (default: GQN)', default='GQN'
'--workers', type=int, help='number of data loading workers', default=32
'--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0,1,2,3])', default=[0,1,2,3,4,5,6,7]
'--layers', type=int, help='number of generative layers (default: 12)', default=8
'--saved_model', type=str, help='path to model', default=None
```
**Note:**
It took about a full week of non-stop training on 4 GPUs to generate the scenes shown in the animation. But we do better than the DeepMind GQN on many mazes put adjacent to each other (they perform well on one maze at a time, and their model fails with many mazes as their representation squashes all information onto the same representation). We are confident that the quality of our images can get much better with much more compute resources.


## Downloading the full dataset of images and pre-processing the data:
If you'd like the full dataset of images we used in our experiments, the following steps are crucial.

* Download the dataset from [this google cloud bucket](https://console.cloud.google.com/storage/browser/gqn-dataset) (we used the dataset called 'mazes' and store the 'train' and 'test' folders in the same directory, assume you call this directory 'DIRNAME' for explanation purposes). 

This [guide]https://cloud.google.com/storage/docs/downloading-objects) provides standard help on downloading data from google cloud buckets.

* The images you download span several '.tfrecord' files in the 'train' and 'test' folders. Run the code from 'convert.py' from our utils folder as
```convert.py DIRNAME``` to process the downloaded data from the 'DIRNAME' directory ('train' and 'test' folders) to convert all the .tfrecord files to .pt.gz files.

* Extract the .gz files and you will be left with .pt files. (e.g. aa.pt.gz -> aa.pt)
For this standard extraction, you have several options listed [here](http://kb.winzip.com/kb/entry/124/).
