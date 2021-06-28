from flask import Flask, request, jsonify, render_template, send_from_directory, Response
import os, cv2, math
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import keras
from PIL import Image
import pickle
from gtts import gTTS
from google_trans_new import google_translator

app = Flask(__name__)

json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("final_model.h5")
print('Loaded model from disk')
# loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    language = request.form['lang']
    folder = request.form['sequence']
    class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','del',' ']
    cat_files = sorted(os.listdir(folder))
    sentence = ''
    for i in cat_files:
        img = image.load_img(os.path.join(folder, i), target_size=(64,64))
        test_image = image.img_to_array(img)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        images = np.vstack([test_image])
        classes = loaded_model.predict_classes(images, batch_size=10)
        # print(class_list[classes[0]])
        if class_list[classes[0]] == 'del':
            sentence = sentence[:-1]
        else:
            sentence += class_list[classes[0]]

    print(sentence)
    if language!='en':
        translator = google_translator()
        sentence = translator.translate(sentence.lower(), lang_tgt=language)

    output = gTTS(text=sentence, lang=language, slow=False, tld='com')
    
    output.save('audio/speech.wav')

    return render_template('index.html', prediction_text=sentence)

@app.route('/convertToSign', methods=['POST'])
def convertToSign():
    text = request.form['sentence']
    sentence = list(text.upper())
    img=[]
    for i in range(len(sentence)):
        if sentence[i] == " ":
            img.append(cv2.imread(os.path.join('Sequence-1','space.jpg')))
        else:
            img.append(cv2.imread(os.path.join('Sequence-1',sentence[i]+'.jpg')))

    height,width,layers=img[1].shape

    video=cv2.VideoWriter('video/video.mp4',-1,1,(width,height))

    for j in range(len(sentence)):
        video.write(img[j])
    video.write(img[len(img)-1])
    cv2.destroyAllWindows()
    video.release()
    # os.system("start video.mp4")
    return render_template('toSign.html')

@app.route("/wav")
def streamwav():
    def generate():
        with open("audio/speech.wav", "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/x-wav")

@app.route("/mp4")
def streammp4():
    def generate():
        with open("video/video.mp4", "rb") as fmp4:
            data = fmp4.read(1024)
            while data:
                yield data
                data = fmp4.read(1024)
    return Response(generate(), mimetype="vidio/x-mp4")

@app.route('/toSign')
def toSign():
    return render_template('toSign.html')

@app.route('/toText')
def toText():
    return render_template('index.html')

# @app.route('/audio/<path:filename>')
# def download_file(filename):
#     return send_from_directory('/audio/', filename)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = loaded_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)