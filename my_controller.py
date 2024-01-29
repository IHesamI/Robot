from controller import Robot
import sys
import numpy as np
import matplotlib.pyplot as plt


# Import neccesarly libraries
import numpy as np
import pandas as pd

import cv2

# from pandas.io.formats.style_render import Axis
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Add
from tensorflow.keras.layers import Softmax
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import LayerNormalization, BatchNormalization
from keras.models import Model

from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


def sign(x):
    return (x > 0) - (x < 0)
    
    
def distance_based(src, dst, precision):
    return abs(abs(src) - abs(dst)) < precision




def is_gray(r, g, b):
    thrs = 15
    r = int(r)
    g = int(g)
    b = int(b)
    return abs(r - g) < thrs and abs(r - b) < thrs and abs(g - b) < thrs


def crop_box_image(path:str):
    image = cv2.imread(path)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i][j]
            if i == 258 and j == 148:
                print()
            if j == 258 and i == 148:
                print()
            if b > 80 and not is_gray(r, g, b):
                image[i][j] = 200

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i][j]
            r = int(r)
            g = int(g)
            b = int(b)
            avg = (r + g + b) // 3
            if avg > 200:
                image[i][j] = avg - 200

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i][j]
            r = int(r)
            g = int(g)
            b = int(b)
            avg = (r + g + b) // 3
            if avg > 70:
                image[i][j] = int(min(avg + 100, 200))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.threshold(gray, 70, 255, cv2.THRESH_OTSU)[1]
    contours, h = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = []
    m = -1

    min_x = 100000000
    min_y = 100000000
    max_x = -1
    max_y = -1
    for c in contours:
        area = cv2.contourArea(c)
        m = max(m, area)
        if 2000 < area < 60000:
            for cc in c:
                x, y = cc.T
                x = x[0]
                y = y[0]
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            cnts.append(c)
    original = cv2.imread(path)
    return original[min_y:max_y, min_x:max_x]



class Mavic (Robot):

    K_VERTICAL_THRUST = 68.5  
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        
    K_ROLL_P = 50.0           
    K_PITCH_P = 30.0          

    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1

    target_precision = 0.1
    
    def __init__(self, model):
        Robot.__init__(self)

        self.init_devices()
        
        motors = [self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor]

        motors[0].setPosition(float('inf'))
        motors[0].setVelocity(1)

        motors[1].setPosition(float('inf'))
        motors[1].setVelocity(1)

        motors[2].setPosition(float('inf'))
        motors[2].setVelocity(1)

        motors[3].setPosition(float('inf'))
        motors[3].setVelocity(1)
        
        self.current_pose = [0, 0, 0, 0, 0, 0]
        self.target_position = [0, 0, 0]
        
        self.target_index = 0
        self.target_altitude = 0
        
        self.waypoints = [[-3, -2], [2, 5], [3, -3], [5, 0], [-5, 4]]
        
        self.MAX_ERROR = 0.03
        self.cnnmodel= model
        

    def init_devices(self):

        self.time_step = int(self.getBasicTimeStep())

        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)

        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)

        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)

        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")

        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(1.62)


    def move_to_target(self):

        # Fetch the first target from the target array
        if self.target_position[0:2] == [0, 0]:
            self.target_position[0:2] = self.waypoints[0]
            print(f"First target detected {self.target_position[0:2]}")

        current_targer = self.target_position
        current_location = self.current_pose[0:2]

        # Get the current roll, pitch and yaw
        curr_roll, curr_pitch, curr_yaw = self.imu.getRollPitchYaw()

        # Check if the mavic riches the wanted target
        if abs(current_targer[0] - current_location[0]) < self.target_precision and abs(current_targer[1] - current_location[1]) < self.target_precision:
            
            # If the target riches, assign the target to the current target
            self.target_index += 1
            self.target_index  %= len(self.waypoints)
            
            self.target_position[0:2] = self.waypoints[self.target_index]
            print("Target reached! New target: ", self.target_position[0:2])

            # turn the yaw to the right value
            while not distance_based(curr_yaw , 0, 0.1): #curr_yaw > 0 + 0.1 or curr_yaw < -0.1:
                self.rotate()
                curr_roll, curr_pitch, curr_yaw = self.imu.getRollPitchYaw()
            
            cameraImage=self.take_picture()
            crop_box_image(cameraImage)

        roll_disturbance, pitch_disturbance, yaw_disturbance = self.update_disturbance(current_targer, current_location)

        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        # This is now in ]-2pi;2pi[
        angle_left = self.target_position[2] - self.current_pose[5]
        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi


        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        pitch_disturbance = clamp(np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        return yaw_disturbance, pitch_disturbance, roll_disturbance 

    def rotate(self):
        # Set values to rotate the mavic
        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = -1.3

        self.target_position[2] = np.arctan2(self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        
        angle_left = self.target_position[2] - self.current_pose[5]
        
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi
        
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        x_pos, y_pos, altitude = self.gps.getValues()
        roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
        self.current_pose = [x_pos, y_pos, altitude, roll, pitch, yaw]

        t1 = self.getTime()

        roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
        pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
        yaw_input = yaw_disturbance
        clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
        vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

        front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
        front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
        rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
        rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

        self.front_left_motor.setVelocity(front_left_motor_input)
        self.front_right_motor.setVelocity(-front_right_motor_input)
        self.rear_left_motor.setVelocity(-rear_left_motor_input)
        self.rear_right_motor.setVelocity(rear_right_motor_input)
        self.step(3*self.time_step)

    
    def take_picture(self):
        image = self.camera.getImage()
        if image:
            width, height = self.camera.getWidth(), self.camera.getHeight()
            image_array = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
            
            plt.imshow(image_array[:, :, 0:3])
            plt.title("Captured Image")
            plt.show()
        image_array = cv2.resize(image_array[:, :, 0:3], (28, 28))
        plt.imshow(image_array[:, :, 0:3])
        plt.title("Captured Image")
        plt.show()
        print(image_array[:, :, 0:3].shape)
        return image_array


    def update_disturbance(self, current_targer, current_location):

        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0
    
        if abs(current_targer[0] - current_location[0]) > self.MAX_ERROR and abs(current_targer[1] - current_location[1]) > self.MAX_ERROR : 
            if current_location[0] < current_targer[0]:
                roll_disturbance = -0.5
            if current_targer[0] > current_location[0]:
                roll_disturbance = 0.5
        if abs(current_targer[0] - current_location[0]) > self.MAX_ERROR:
    
            if current_location[0] < current_targer[0]:
                yaw_disturbance = -1.3
            if current_targer[0] > current_location[0]:
                yaw_disturbance = 1.3    
        if abs(current_targer[1] - current_location[1]) > self.MAX_ERROR:
        
            if current_location[1] < current_targer[1]:
                pitch_disturbance = -2.0
            if current_targer[1] < current_location[1]:
                pitch_disturbance = 2.0
        return roll_disturbance, pitch_disturbance, yaw_disturbance 


    def run(self):
        t1 = self.getTime()

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        # Specify the patrol coordinates
        self.waypoints = [[-3, -2], [3, -3], [5, 0], [2, 5], [-5, 4]]
        # target altitude of the robot in meters
        self.target_altitude = 3

        while self.step(3*self.time_step) != -1:
        
            roll, pitch, yaw = self.imu.getRollPitchYaw()
        
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.current_pose = [x_pos, y_pos, altitude, roll, pitch, yaw]

            if altitude > self.target_altitude - 1:
                if self.getTime() - t1 > 0.1:
                    yaw_disturbance, pitch_disturbance, roll_disturbance = self.move_to_target()
                    t1 = self.getTime()

            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)

class CNNModel:

    def __init__(self):
        pass


    def convolutional_layer(self, X, filters, kernels, strides):
          
          first_input = X
        
          f1, f2, f3 = filters
          s1, s2, s3 = strides
          k1, k2, k3 = kernels
        
          # First stage
        
          X = Conv2D(filters=f1, kernel_size=k1, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), strides=s1, padding='same')(X)
          X = BatchNormalization(axis=3)(X)
          X = Activation(relu)(X)
          # X = MaxPooling2D(pool_size=(2, 2))
        
          # Second stage
        
          X = Conv2D(filters=f2, kernel_size=k2, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), strides=s2, padding='same')(X)
          X = BatchNormalization(axis=3)(X)
          X = Activation(relu)(X)
          # X = MaxPooling2D(pool_size=(2, 2))
        
          # Third stage
        
          X = Conv2D(filters=f3, kernel_size=k3, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), strides=s3, padding='same')(X)
          X = BatchNormalization(axis=3)(X)
          # result
        
          first_input = Conv2D(filters=f3, kernel_size=k3, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), strides=s3, padding='same')(first_input)
        
          X = Add()([X, first_input])
        
          X = Activation(relu)(X)
        
          return X
          
    def resnet(self, input_shape, classes):
          X_input = Input(input_shape)
        
          X = Conv2D(filters=32, kernel_size=1, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), strides=1)(X_input)
          X = Conv2D(filters=32, kernel_size=1, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), strides=1)(X)
          X = Conv2D(filters=32, kernel_size=1, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), strides=1)(X)
        
          X = Activation(relu)(X)
          X = MaxPooling2D(pool_size=(2, 2))(X)
        
          X = self.convolutional_layer(X, filters=(32, 32, 32), kernels=(3, 3, 3), strides=(1, 1, 1))
          X = self.convolutional_layer(X, filters=(32, 32, 32), kernels=(3, 3, 3), strides=(1, 1, 1))
          # X = convolutional_layer(X, filters=(32, 32, 32), kernels=(3, 3, 3), strides=(1, 1, 1))
        
          X = self.convolutional_layer(X, filters=(32, 32, 32), kernels=(3, 3, 3), strides=(1, 1, 1))
          X = self.convolutional_layer(X, filters=(32, 32, 32), kernels=(3, 3, 3), strides=(1, 1, 1))
        
          X = Conv2D(filters=32, kernel_size=3, kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), strides=1)(X)
          X = Activation(relu)(X)
          X = MaxPooling2D(pool_size=(2, 2))(X)
        
          X = Flatten()(X)
        
          X = Dense(50, activation=relu, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))(X)
        
          X_last = X
          X = Dense(50, activation=relu, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))(X)
          X = Add()([X, X_last])
        
          X = Dense(classes, activation=softmax, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))(X)
        
          return Model(inputs = X_input, outputs = X)


    def builder(self):
        self.model = self.resnet((28, 28, 1), 5)

        self.model.compile(optimizer=Adam(learning_rate = 0.0001),
              loss=[sparse_categorical_crossentropy, lambda y_true, y_pred: center_loss(y_true, y_pred, centers)],
              metrics=['accuracy'])

        self.model.load_weights("/home/soheil/Downloads/best_weight.hdf5")
        print(self.model.summary())


    def predictor(self, x):
          y_pred_1 = self.model.predict(x)
          y_pred = []
          for record in range(len(x)):
            y_pred.append(y_pred_1[record].argmax()+1)
        
          y_pred = np.array(y_pred)
          y_pred = y_pred - 1
        
          return y_pred


if __name__ == "__main__":
    model = CNNModel()
    model.builder()
    robot = Mavic(model)
    robot.run()

