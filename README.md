# 🧠 DeepLabV3 Segmentation Web App

A responsive web application for binary semantic segmentation powered by a custom-trained DeepLabV3 (ResNet-50) model. The model is trained from scratch using Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss) for optimization, and evaluated using the Dice Score metric to assess segmentation quality. Users can upload images or videos to instantly visualize accurate segmentation masks alongside the original input.


#### 📌 Overview
✅ Model trained in a Jupyter notebook

📦 Inference via a Streamlit web app

🎥 Supports video and image uploads

🔍 Binary segmentation (e.g., object vs background)

⚡ Built with PyTorch, Albumentations, OpenCV, Streamlit


#### 🧠 Model Details
Architecture: DeepLabV3 + ResNet-50 (No pretrained weights)

Output: Single-channel binary mask

Training Platform: Jupyter Notebook (notebook/Semantic_Segmentation_using_deeplabv3.ipynb)

Inference Platform: Streamlit Web UI (app.py)


#### 📂 Project Structure

```bash
deeplabv3-semantic-segmentation-streamlit/
├── data/
│   └── train/
│   └── valid/
│   └── test/
├── notebook/
│   └── Semantic_Segmentation_using_deeplabv3.ipynb     # Model training code
├── model/
│   └── segmentation_model.pth     			# Trained PyTorch model                   			
├── video
│   └── demo.mp4					# Test videos/images
├── app.py						# Streamlit UI code 
├── main.py   						# Mask prediction logic
├── train.py                 				# model training code                     			
├── requirements.txt
└── README.md
```
#### 💡 Features
Upload .jpg or .mp4 files

Real-time prediction using your trained model

Side-by-side visualization of input vs. predicted mask

🔧 Setup & Usage

1. Install Dependencies

```bash
	git clone https://github.com/Akhilpm156/deeplabv3-semantic-segmentation-streamlit.git
```
```bash
	cd deeplabv3-semantic-segmentation-streamlit
```
```bash
	conda create -p venv python=3.10 -y
```	
```bash
	conda activate .\venv
```	
```bash
	pip install -r requirements.txt
```
2. Run the training script
```bash
	python train.py
```
Trained Model saved in model directory

3. Run the Web App
```bash
	streamlit run app.py
```
Then open: http://localhost:8501

📝 License

MIT License — feel free to use, modify, and share.

🙋‍♂️ Author
Akhil P M<br>
📧 akhilpm64@outlook.com<br>
🔗 [LinkedIn](https://www.linkedin.com/in/akhil-p-m-614b53295)

