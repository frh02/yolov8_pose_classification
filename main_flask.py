from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mediapipe as mp
from src.utils import calculate_angle_new

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

@app.route('/rom')  # Define route for Range of Motion
def rom():
    return render_template('rom.html')

@app.route('/sts')  # Define route for Sit to Stand
def sts():
    return render_template('sts.html')

@app.route('/tug')  # Define route for Timed up and Go
def tug():
    return render_template('tug.html')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
