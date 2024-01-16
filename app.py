from flask import Flask, render_template, request, redirect, url_for
import dlib
import numpy as np
import os
app = Flask(__name__)

# Load Dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def extract_face_features(image):
    img = dlib.load_rgb_image(image)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)
    return np.zeros(128)  

def compare_faces(face1, face2):
    return np.linalg.norm(face1 - face2)

@app.context_processor
def utility_processor():
    def determine_color(match):
        return 'green' if match else 'red'
    return dict(determine_color=determine_color)


@app.route('/delete-images', methods=['POST'])
def delete_images():
    try:
        image1 = request.form.get('image1')
        image2 = request.form.get('image2')
        if image1:
            os.remove(os.path.join('static/images', image1))
        if image2:
            os.remove(os.path.join('static/images', image2))
    except Exception as e:
        print("Error deleting images:", e)
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process images uploaded
        file1 = request.files['file1']
        file2 = request.files['file2']

        # Save images
        path1 = os.path.join('static/images', file1.filename)
        path2 = os.path.join('static/images', file2.filename)
        file1.save(path1)
        file2.save(path2)

        # Extract features and compare
        face1_features = extract_face_features(path1)
        face2_features = extract_face_features(path2)
        distance = compare_faces(face1_features, face2_features)

        # Determine if it's the same person (threshold can be adjusted)
        match = distance < 0.6

        return render_template('index.html', match=match, distance=distance, image1=file1.filename, image2=file2.filename)
    return render_template('index.html', match=None)

if __name__ == '__main__':
    app.run(debug=True)
