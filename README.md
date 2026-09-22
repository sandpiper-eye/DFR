# Selective Invariance: Degradation-Aware Feature Recalibration for Low-Light Object Detection

DFR is a degradation-aware object detection framework designed for low-light domain adaptation. The current training pipeline uses a COCO-format source dataset and adapts the detector to the ExDark low-light domain through degradation representation learning, semantic-preserving feature intervention, and degradation-aware feature recalibration.

## Repository Layout

```text
DFR-detr/
├── configs/dfr_coco/
│   ├── coco_detection.py              # Stage 1: source-domain detector pretraining
│   ├── coco_exdark_generation.py      # Stage 2: degradation representation learning
│   └── coco_exdark_finetune.py        # Stage 3: target-domain fine-tuning
├── mmdet/
│   ├── models/detectors/dfr_yolo.py          # Main DFR-YOLO detector
│   ├── models/losses/dfr_losses.py           # DFR-specific objectives
│   ├── models/losses/feature_consistency_loss.py
│   ├── utils/dfr_modules.py                  # DAE, SPFI, and DARC modules
│   ├── utils/physical_degradation.py         # Synthetic low-light degradation model
│   └── datasets/
│       ├── lowlight.py                       # Unlabeled low-light image dataset
│       └── samplers/ratio_sampler.py         # Fixed-ratio source/target sampler
├── tools/train.py                             # MMDetection training entry point
└── requirements/                             # Runtime dependency specifications
```

## Environment Setup

```bash
pip install -U openmim
mim install "mmcv>=2.0.0rc4,<2.1.0"
pip install "mmengine==0.10.7"
pip install -r requirements.txt
```

## Data Preparation

The source domain is a 12-category subset of COCO stored in COCO format under `coco12c`. Its category set is identical to that used for ExDark:

`Bicycle`, `Boat`, `Bottle`, `Bus`, `Car`, `Cat`, `Chair`, `Cup`, `Dog`, `Motorbike`, `People`, and `Table`.

```text
DFRdata/
├── coco12c/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── annotations/
│       ├── train.json
│       └── val.json
└── exdarkcoco/
    ├── train/
    │   ├── images/                 # Used as low-light target images in Stage 2
    │   └── annotations/train.json  # Used for supervised Stage 3 fine-tuning
    └── val/
        ├── images/
        └── annotations/val.json
```

Stage 2 uses `RatioSampler` with a fixed 1:1 source-to-target sampling ratio. With the configured batch size of 16, each batch contains eight COCO source images and eight ExDark target images.

## Three-Stage Training


### Stage 1: Source-Domain Detector Pretraining

Stage 1 trains a standard 12-class YOLO detector on the filtered COCO source dataset. The detector uses Darknet-53 as the backbone, YOLONeck as the neck, and YOLOHead as the detection head. The current configuration uses a batch size of 8 and trains for 273 epochs.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd):${PYTHONPATH}" \
python tools/train.py configs/dfr_coco/coco_detection.py \
  --work-dir work_dirs/dfr_coco/coco_detection
```

The backbone is initialized from `open-mmlab://darknet53`. 


### Stage 2: Degradation Representation Learning

Stage 2 initializes the detector from the Stage 1 checkpoint and optimizes the degradation-aware pathway, including the Degradation Attribute Encoder (DAE), Semantic-Preserving Feature Intervention (SPFI), Degradation-Aware Response Calibration (DARC), and the degradation-regression components.

During this stage, the detector backbone, neck, and detection head remain frozen. Training focuses on learning degradation-sensitive attribute representations while maintaining compatibility with the pretrained detector semantics. The current configuration uses a batch size of 16, gradient accumulation over two iterations, 25 training epochs, and a fixed 1:1 COCO/ExDark sampling ratio.


```bash
python tools/train.py configs/dfr_coco/coco_exdark_generation.py \
  --work-dir work_dirs/dfr_coco/coco_exdark_generation
```

### Stage 3: ExDark Target-Domain Fine-Tuning

Stage 3 initializes the model from the Stage 2 checkpoint and fine-tunes it on the 12-category ExDark dataset. SPFI and the auxiliary representation-learning objectives are disabled, while DAE and DARC remain active to provide image-dependent degradation-aware feature recalibration.

The current configuration uses a batch size of 8, gradient accumulation over two iterations, 75 training epochs.

```bash
python tools/train.py configs/dfr_coco/coco_exdark_finetune.py \
  --work-dir work_dirs/dfr_coco/coco_exdark_finetune
```


## Inference

At inference time, SPFI and the auxiliary degradation-learning objectives are removed. DAE extracts the degradation attribute representation from the current input image, and DARC uses this representation to generate channel-wise modulation parameters for intermediate detector features.

Therefore, the inference pipeline retains only the degradation-estimation and feature-recalibration pathway required for image-dependent calibration, without the training-time intervention branch.
