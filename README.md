This repository contains the core structure and code for a project for medical image retrieval. Due to licensing and file size limitations, datasets, pre-trained models, and other generated files are not included. Users must set up these resources before running the code.

---

## **Setup and Usage**

### **1. Download the MIMIC-CXR Dataset**
- Download **MIMIC-CXR-JPG** and **MIMIC-CXR** (for reports) from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/).
- Place the datasets in a directory of your choice.
- Update the src/config.yaml file with the paths to the datasets. 

### **2. Place Required CSV Files**
- Place the following CSV files in the directory `data/raw/`:
  - `data/raw/mimic-cxr-2.0.0-chexpert.csv`
  - `data/raw/mimic-cxr-2.0.0-split.csv`

### **3. Create the Mimic 5x200 Dataset**
Run the following script to preprocess the dataset:
```bash
python create_mimic5x200.py
```

### **4. Generate Embeddings**
Run the script to create embeddings for images and text:
```bash
python create_embeddings.py
```

### **5. Download Pre-Trained BiomedCLIP**
Download the following files from [Hugging Face BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224):
- `open_clip_pytorch_model.bin`
- `open_clip_config.json`

Place them in the `pretrained/biomedclip/checkpoints/` directory.

### **6. Train the Model**
Run the training script:
```bash
python train_cross_attention_encoder.py
```

### **7. Evaluate the Model**
Evaluate the trained model:
```bash
python main.py
```

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

## **Notice**
This repository is provided to comply with the requirements of the Data Use Agreement (DUA) for the MIMIC-CXR dataset. It contains the code used for the project but does not include all the dependencies or instructions for running the code. Users are responsible for setting up the required environment and libraries based on their needs.
