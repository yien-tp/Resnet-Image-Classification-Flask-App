from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads,IMAGES
from tensorflow.python.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
#for matrix math
import numpy as np
#from werkzeug.utils import secure_filename
import sys
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
net = load_model('model-resnet50.h5')

app = Flask(__name__)


photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST' and 'photo' in request.files:
		filename = photos.save(request.files['photo'])
		try:
			os.rename('./'+filename,'./'+'output.jpg')
		except FileExistsError:
			os.remove('./'+'output.jpg')
			os.rename('./'+filename,'./'+'output.jpg')

	#read the image into memory
	cls_list = ['cats', 'dogs']
	img = image.load_img('./output.jpg', target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	pred = net.predict(x)[0]
	top_inds = pred.argsort()[::-1][:5]
	for i in top_inds:
		print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
		return render_template("index2.html",s1 = pred[0], s2 = cls_list[0], s3 = pred[1],s4 = cls_list[1])


if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 3000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
