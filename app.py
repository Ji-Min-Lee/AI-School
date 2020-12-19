import os
from glob import glob
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from numpy import asarray
from model_loader import get_result

app = Flask(__name__)
app.add_url_rule('/instance', endpoint='instance', view_func=app.send_static_file)


def inference(image):
    image = Image.open('static/' + image)
    data = asarray(image)
    result = get_result(data)
    return result


@app.route('/')
def welcome():
    return 'Welcome to AI School!'


@app.route('/demo')
def demo():
    img_list = get_images()
    # TODO 이름 바꾸기
    return render_template("index.html", name="David", img_list=img_list, img_cnt=len(img_list), inference=inference)


def get_images():
    img_list = []
    for img_f in glob("static/*"):
        if 'ai.png' in img_f:
            continue
        img_list.append(img_f.split("\\")[-1])
    return img_list


@app.route("/redirect/image/save", methods=['POST'])
def save_image():
    if request.method == 'POST':
        f = request.files['filename']
        f.save(os.path.join(app.static_folder, secure_filename(f.filename)))
    return redirect(url_for('demo'))


if __name__ == "__main__":
    app.run()
