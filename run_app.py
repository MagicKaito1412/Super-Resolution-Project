import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import torch
from utils import *
from PIL import Image

def visualize_sr(img):
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet(convert_image(hr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')


    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(convert_image(hr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    bicubic_img = hr_img.resize((int(hr_img.width)*4, int(hr_img.height*4)), Image.BICUBIC)
    
    return sr_img_srgan, sr_img_srresnet, bicubic_img


UPLOAD_FOLDER = './static/'
UPLOAD_FILENAME = 'upload.jpg'
DOWNLOAD_FILENAME = 'download.jpg'
DOWNLOAD_SRGAN_FILENAME = 'srgan_download.jpg'
DOWNLOAD_SRRESNET_FILENAME = 'srresnet_download.jpg'
DOWNLOAD_BICUBIC = 'bicubic_download.jpg'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FILENAME'] = UPLOAD_FILENAME
app.config['DOWNLOAD_FILENAME'] = DOWNLOAD_FILENAME
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if os.path.exists(f"./static/{DOWNLOAD_SRGAN_FILENAME}"):
            os.remove(f"./static/{DOWNLOAD_SRGAN_FILENAME}")
        if os.path.exists(f"./static/{DOWNLOAD_SRRESNET_FILENAME}"):
            os.remove(f"./static/{DOWNLOAD_SRRESNET_FILENAME}")
        if os.path.exists(f"./static/{DOWNLOAD_BICUBIC}"):
            os.remove(f"./static/{DOWNLOAD_BICUBIC}")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FILENAME']))
            return redirect(url_for('home', filename=app.config['UPLOAD_FILENAME']))
    return render_template("index.html")

@app.route("/downloadSrgan")
def downloadSrgan():
    return send_file(f"./static/{DOWNLOAD_SRGAN_FILENAME}", as_attachment=True)

@app.route("/downloadSrresnet")
def downloadSrresnet():
    return send_file(f"./static/{DOWNLOAD_SRRESNET_FILENAME}", as_attachment=True)

@app.route("/downloadBicubic")
def downloadBicubic():
    return send_file(f"./static/{DOWNLOAD_BICUBIC}", as_attachment=True)

@app.route("/enhance")
def enhance():
    srgan, srresnet, bicucic = visualize_sr(f"./static/{app.config['UPLOAD_FILENAME']}")
    srgan.save(f"./static/{DOWNLOAD_SRGAN_FILENAME}")
    srresnet.save(f"./static/{DOWNLOAD_SRRESNET_FILENAME}")
    bicucic.save(f"./static/{DOWNLOAD_BICUBIC}")
    return redirect('/')

if __name__ == "__main__":
    device = torch.device('cpu')
    srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"
    # Load models
    srresnet = torch.load(srresnet_checkpoint, map_location = device)['model'].to(device)
    srresnet.eval()

    srgan_checkpoint = "./checkpoint_srgan.pth.tar"

    srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
    srgan_generator.eval()

    if os.path.exists(f"./static/{app.config['UPLOAD_FILENAME']}"):
        os.remove(f"./static/{app.config['UPLOAD_FILENAME']}")
    if os.path.exists(f"./static/{DOWNLOAD_SRGAN_FILENAME}"):
        os.remove(f"./static/{DOWNLOAD_SRGAN_FILENAME}")
    if os.path.exists(f"./static/{DOWNLOAD_SRRESNET_FILENAME}"):
        os.remove(f"./static/{DOWNLOAD_SRRESNET_FILENAME}")
    if os.path.exists(f"./static/{DOWNLOAD_BICUBIC}"):
        os.remove(f"./static/{DOWNLOAD_BICUBIC}")
    app.run()