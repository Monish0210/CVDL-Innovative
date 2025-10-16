# 2_Web_Application/app.py
import os
from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms
from model import SiamUnet
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# --- Load Your Trained Model ---
device = torch.device('cpu')
model = None # Initialize model as None

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DAMAGE_CLASSES = {0: 'No Damage/BG', 1: 'Minor Damage', 2: 'Moderate Damage', 3: 'Major Damage', 4: 'Destroyed'}

@app.before_request
def load_model():
    # Load model only once before the first request
    global model
    if model is None:
        print("Loading model for the first time...")
        try:
            model = torch.load('siamUnet.pt', map_location=device, weights_only=False)
            model.eval()
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("ERROR: siamUnet.pt not found! The app will not work without the trained model file.")
            model = "not_found"

@app.route('/', methods=['GET', 'POST'])
def index():
    if model == "not_found":
        return "Model file 'siamUnet.pt' not found. Please add the trained model to the application folder and restart.", 500

    if request.method == 'POST':
        pre_image_file = request.files['pre_image']
        post_image_file = request.files['post_image']

        pre_image_path = os.path.join(app.config['UPLOAD_FOLDER'], pre_image_file.filename)
        post_image_path = os.path.join(app.config['UPLOAD_FOLDER'], post_image_file.filename)
        pre_image_file.save(pre_image_path)
        post_image_file.save(post_image_path)

        pre_image = Image.open(pre_image_path).convert("RGB")
        post_image = Image.open(post_image_path).convert("RGB")
        pre_image_tensor = transform(pre_image).unsqueeze(0).to(device)
        post_image_tensor = transform(post_image).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, damage_output = model(pre_image_tensor, post_image_tensor)
            preds_cls = torch.argmax(torch.nn.functional.softmax(damage_output, dim=1), dim=1)

        # Correctly convert the numerical mask to a color-coded RGB image
        colors = torch.tensor([[0,0,0], [0,0,255], [255,255,0], [255,165,0], [255,0,0]], dtype=torch.uint8)
        output_image_tensor = colors[preds_cls.squeeze(0).cpu().long()].permute(2, 0, 1)
        output_image = transforms.ToPILImage()(output_image_tensor)
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
        output_image.save(output_image_path)

        # Calculate damage analysis for the chart
        unique_classes, counts = torch.unique(preds_cls, return_counts=True)
        total_pixels = torch.numel(preds_cls)
        damage_analysis = {DAMAGE_CLASSES.get(i.item(), "Unknown"): (v.item() / total_pixels) * 100 for i, v in zip(unique_classes, counts)}

        return render_template('index.html', 
                               pre_image=pre_image_path, 
                               post_image=post_image_path, 
                               output_image=output_image_path, 
                               damage_analysis=json.dumps(damage_analysis))

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)