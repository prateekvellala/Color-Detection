import numpy as np
import cv2
import imutils
import pandas as pd
import webcolors
from scipy.spatial import KDTree

camera = cv2.VideoCapture(0)

r_val = g_val = b_val = x_pos = y_pos = 0
window_closed = False

color_data_columns = ['color', 'color_name', 'hex', 'R', 'G', 'B']
color_data_df = pd.read_csv('color_data.csv', names=color_data_columns, header=None)

color_data_rgb = color_data_df[['R', 'G', 'B']].values
kdtree = KDTree(color_data_rgb)

def get_rgb_from_name(color_name):
    try:
        rgb = webcolors.name_to_rgb(color_name)
        return rgb
    except ValueError:
        return None

def get_color_name_from_rgb(red, green, blue):
    target_color = np.array([[red, green, blue]])
    _, index = kdtree.query(target_color)
    color_name = color_data_df.loc[index[0], 'color_name']
    return color_name

def identify_pixel_color(event, x, y, flags, param):
    global b_val, g_val, r_val, x_pos, y_pos
    x_pos = x
    y_pos = y
    b_val, g_val, r_val = frame[y, x]
    b_val = int(b_val)
    g_val = int(g_val)
    r_val = int(r_val)
    color_name = get_color_name_from_rgb(r_val, g_val, b_val)
    if color_name:
        print(f"Color: {color_name} (R={r_val}, G={g_val}, B={b_val})")
    else:
        print(f"R={r_val}, G={g_val}, B={b_val}")

cv2.namedWindow('Color Detection')
cv2.setMouseCallback('Color Detection', identify_pixel_color)


cv2.resizeWindow('Color Detection', 900, 675)

while True:
    grabbed, frame = camera.read()
    frame = imutils.resize(frame, width=900)

    color_name = get_color_name_from_rgb(r_val, g_val, b_val)
    info_frame = np.zeros_like(frame)
    info_frame[:] = (255, 255, 255) 

    if color_name:
        cv2.putText(info_frame, f"Detected Color: {color_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(info_frame, "No Color Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(info_frame, f"R={r_val}, G={g_val}, B={b_val}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    
    combined_frame = np.hstack((frame, info_frame))
    cv2.imshow('Color Detection', combined_frame)

    
    if cv2.getWindowProperty('Color Detection', cv2.WND_PROP_VISIBLE) < 1:
        window_closed = True
        break

    if cv2.waitKey(20) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()

if window_closed:
    cv2.waitKey(1)
