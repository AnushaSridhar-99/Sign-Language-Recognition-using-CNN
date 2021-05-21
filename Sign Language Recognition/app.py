from flask import Flask, request, jsonify, render_template
import os, cv2, math
import numpy as np

import keras
# from PIL import Image
import pickle

app = Flask(__name__)

loaded_model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    i= 10
    while True:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        x,y,w,h = 0,0,300,300
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi = frame[y:y+h, x:x+w]
        cv2.imshow("Capturing", frame)
        cv2.imwrite(filename='Images\\saved_img'+str(i)+'.jpg', img=roi)
        cv2.waitKey(3000)
        i += 1 
        
        if cv2.waitKey(1) == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()  

    cat_dir = "Images"
    cat_files = os.listdir(cat_dir)

    for file in cat_files:
        image = cv2.imread(os.path.join(cat_dir, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        src = cv2.Canny(blurred, 10, 150)
        dest = os.path.join(cat_dir, file)
        im = Image.fromarray(src)
        im.save(dest)
    

    cat_dir = "Images"
    cat_files = os.listdir(cat_dir)
    sentence = ''
    for i in cat_files:
        img = cv2.imread(os.path.join(cat_dir, i))
    #     plt.imshow(img)
        dims = (64,64)
        img1 = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
        
        image = img1/255
        test_image = np.expand_dims(image, axis = 0)
        
        result = loaded_model.predict(test_image)
    #   print(result)
        labels = np.argmax(result, axis=-1)  
        if labels[0] == 0:
            sentence += '0'
        elif labels[0] == 1:
            sentence += '1'
        elif labels[0] == 2:
            sentence += '2'
        elif labels[0] == 3:
            sentence += '3'
        elif labels[0] == 4:
            sentence += '4'
        elif labels[0] == 5:
            sentence += '5'
        elif labels[0] == 6:
            sentence += '6'
        elif labels[0] == 7:
            sentence += '7'
        elif labels[0] == 8:
            sentence += '8'
        elif labels[0] == 9:
            sentence += '9'
        elif labels[0] == 10:
            sentence += 'A'
        elif labels[0] == 11:
            sentence += 'B'
        elif labels[0] == 12:
            sentence += 'C'
        elif labels[0] == 13:
            sentence += 'D'
        elif labels[0] == 36:
            sentence = sentence[:-1]
        elif labels[0] == 14:
            sentence += 'E'
        elif labels[0] == 15:
            sentence += 'F'
        elif labels[0] == 16:
            sentence += 'G'
        elif labels[0] == 17:
            sentence += 'H'
        elif labels[0] == 18:
            sentence += 'I'
        elif labels[0] == 19:
            sentence += 'J'
        elif labels[0] == 20:
            sentence += 'K'
        elif labels[0] == 21:
            sentence += 'L'
        elif labels[0] == 22:
            sentence += 'M'
        elif labels[0] == 23:
            sentence += 'N'
        elif labels[0] == 37:
            sentence += ''
        elif labels[0] == 24:
            sentence += 'O'
        elif labels[0] == 25:
            sentence += 'P'
        elif labels[0] == 26:
            sentence += 'Q'
        elif labels[0] == 27:
            sentence += 'R'
        elif labels[0] == 28:
            sentence += 'S'
        elif labels[0] == 38:
            sentence += ' '
        elif labels[0] == 29:
            sentence += 'T'
        elif labels[0] == 30:
            sentence += 'U'
        elif labels[0] == 31:
            sentence += 'V'
        elif labels[0] == 32:
            sentence += 'W'
        elif labels[0] == 33:
            sentence += 'X'
        elif labels[0] == 34:
            sentence += 'Y'
        elif labels[0] == 35:
            sentence += 'Z'

    # print(sentence)
    return render_template('index.html', prediction_text=sentence)

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = loaded_model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)