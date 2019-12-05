from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from Aviad_single_image_object_counting import detect 
from flask_bootstrap import Bootstrap
import multiprocessing
import time

app = Flask(__name__)
Bootstrap(app)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/upload', methods = ['POST'])

def upload():
    global full_path
    file = request.files['inputFile']
    filename = file.filename
    full_path = os.path.join('C:\\Users\\IEUser\\Desktop\\front\\frontend\\temp', filename)
    file.save(full_path)
    return redirect(url_for('Processing'))
    

@app.route('/processing')
def Processing():

    result = detect(full_path)
    b = result.split(':')[2]
        
    os.remove(full_path)
    return send_file('C:\\Users\\IEUser\\Desktop\\front\\frontend\\temp\\output.png')
    # return render_template("pretty_results.html", data=b)


if __name__ == '__main__':
    app.run(debug=True)