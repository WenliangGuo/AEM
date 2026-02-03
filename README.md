# Procedural Mistake Detection via Action Effect Modeling

**[[Project Page](https://wenliangguo.github.io/Mistake_Detection)] [[Paper](https://arxiv.org/abs/2512.03474)]**

[Wenliang Guo](https://wenliangguo.github.io/), [Yujiang Pu](https://www.yujiangpu.com/), [Yu Kong](https://www.egr.msu.edu/~yukong/).

<p align="center">
  <img src="assets/framework.png" alt="framework" width="50%">
</p>



**Abstract:** Mistake detection in procedural tasks is essential for building intelligent systems that support learning and task execution. Existing approaches primarily analyze how an action is performed, while overlooking what it produces, i.e., the **action effect**. Yet many errors manifest not in the execution itself but in the resulting outcome, such as an unintended object state or incorrect spatial arrangement. To address this gap, we propose **Action Effect Modeling (AEM)**, a unified framework that jointly captures action execution and its outcomes through a probabilistic formulation. AEM first identifies the outcome of an action by selecting the most informative effect frame based on semantic relevance and visual quality. It then extracts complementary cues from visual grounding and symbolic scene graphs, aligning them in a shared latent space to form robust effect-aware representations. To detect mistakes, we further design a prompt-based detector that incorporates task-specific prompts and aligns each action segment with its intended execution semantics. Our approach achieves state-of-the-art performance on the EgoPER and CaptainCook4D benchmarks under the challenging one-class classification (OCC) setting. These results demonstrate that modeling both execution and outcome yields more reliable mistake detection, and highlight the potential of effect-aware representations to benefit a broader range of downstream applications.

## Environment Setup

### Step 1. Install the conda environment using the provided .yml file

We provide two options to set up the environment:
#### Option 1: Using the YML file
```
conda env create -f environment.yml
conda activate ED
```
This environment is configured for our server with an NVIDIA RTX A6000 GPU and CUDA 12.4. 

#### Option 2: Manual configuration
If the environment built using Option 1 does not work on your machine, you can manually install the following **main packages and others as needed**:
- Python: 3.10
- PyTorch and CUDA: Refer to the [official link](https://pytorch.org/get-started/previous-versions/) to find a compatible version. We strongly recommend using PyTorch 1.x, as PyTorch 2.x may cause errors.
- Main packages: 
```
tensorboardx, pyyaml, numpy, pandas, h5py, open-clip-torch, opencv-python, scikit-learn, pillow
```

### Step 2. Compile NMS operations:
```bash
cd ./libs/utils
python setup.py install --user
cd ../..
```
**Note:** Recompile the code whenever you update PyTorch.

## Data Preparation
### Step 1: Video Data
Follow the instructions on the [EgoPER website](https://github.com/robert80203/EgoPER_official) to request video data and annotations.

### Step 2: Video Features
We provide two options for obtaining video features:  
Option 1: Follow the [EgoPER website](https://github.com/robert80203/EgoPER_official) to extract I3D features into the *data/* folder. (This process is slow on our machine.)  
Option 2: Download pre-extracted I3D features (~2.9GB) from our [Google Drive](https://drive.google.com/drive/folders/1mIkjDdfPbMiG1C5S_qa8JwD9Wf-hvsDR?usp=sharing) and extract the files into the *data/* folder. (Faster with a stable internet connection.)

### (Optional) Step 3: Effect Frames
For model evaluation only, this step can **be skipped** as effect frames are only used for training. Otherwise, download the effect frames (~4GB) from our [Google Drive](https://drive.google.com/file/d/1d3GkObtZ-DLNsXjw4i5Lm0mfBlmatmK0/view?usp=sharing) and extract the files into the *data/effect_frames* folder.

Store both the annotation file and features in the *data/* folder. Videos can be saved anywhere, as they are only used for feature extraction and visualization. The *data/* folder should be organized as follows:

```
AEM/
├──...
├──data/
│  ├── active_object.json
│  ├── annotation.json   
│  ├── coffee/
│  │   ├── test.txt
│  │   ├── training.txt
│  │   ├── validation.txt
│  │   ├── features_10fps/
│  │   │   ├──coffee_u1_a1_error_001.npy
│  │   │   ├──...
│  ├── oatmeal/...
│  ├── pinwheels/...
│  ├── quesadilla/...
│  ├── tea/...
│  ├── effect_frames/
│  │   ├── effect_frames_coffee.npz
│  │   ├── effect_frames_oatmeal.npz
│  │   ├── effect_frames_pinwheels.npz
│  │   ├── effect_frames_quesadilla.npz
│  │   ├── effect_frames_tea.npz
│  ├── detection/...                   
│  ├── scene_graph_json/...           
│  └── scene_graph_npy/...            
```

**Note:** The default data path is *data/*. If you customize the folder location, update the `root_dir` path in the configuration files (e.g., `configs/coffee.yaml`) to reflect your dataset's location. 

## Checkpoints (Optional)
We provide the checkpoints used to reproduce our results:

Step 1: Download checkpoints from our [Google Drive](https://drive.google.com/drive/folders/1O5BNeC17pse4O2T4QDmo-ntrcbR6uLed?usp=sharing).  
Step 2: Create a folder called *checkpoints* under the root *AEM* directory.  
Step 3: Place all the *.pth.tar* files into the *checkpoints* folder.

The resulting folder should be organized as follows:
```
AEM/ 
├── ...
├── checkpoints/
│   ├── coffee.pth.tar
│   ├── oatmeal.pth.tar
│   ├── pinwheels.pth.tar
│   ├── quesadilla.pth.tar  
│   ├── tea.pth.tar 
```

## Training

Train the model for a specific task (e.g., coffee):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/coffee.yaml --output exp_name --use_gcn
```

**Note:** The current code does not support multi-GPU distributed training.

## Evaluation

Evaluate a trained model for a specific task (e.g., coffee):

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py configs/coffee.yaml CHECKPOINT_PATH --use_gcn
```

**Note:** For the coffee task, replace *CHECKPOINT_PATH* with *checkpoints/coffee.pth.tar* to evaluate our checkpoint. The same applies to other tasks.

## Citation

```bibtex
@article{guo2025procedural,
  title={Procedural Mistake Detection via Action Effect Modeling},
  author={Guo, Wenliang and Pu, Yujiang and Kong, Yu},
  journal={arXiv preprint arXiv:2512.03474},
  year={2025}
}
```

## Acknowledgements

Our codebase builds upon the open-source projects: [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [EgoPER](https://github.com/robert80203/EgoPER_official).
