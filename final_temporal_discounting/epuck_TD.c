/*
 * Copyright 1996-2020 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// From Morris

#include <stdio.h>
#include <string.h>

#include <webots/accelerometer.h>
#include <webots/camera.h>
#include <webots/distance_sensor.h>
#include <webots/light_sensor.h>
#include <webots/motor.h>
#include <webots/position_sensor.h>
#include <webots/robot.h>

// From conditioning
#include <webots/utils/system.h>

// from morris
#include <webots/supervisor.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <webots/compass.h>

#include <time.h>

#define WHEEL_RADIUS 0.02
#define AXLE_LENGTH 0.052
#define RANGE (1024 / 2)

// from conditioning
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"



// Definitions from Morris
// #define TIME_STEP 64
#define TURN_RATE 100
#define DIRECTIONS 8
#define HD_RES 1.0
#define PLATFORM_RADIUS 0.33
#define PLATFORM_X 0.33
#define PLATFORM_Z 0.67
#define START_DIST 1.5

#define TRIALS 32
#define TIMEOUT 1000000
#define SEED 7

// Definitions from Conditioning
#define SPEED 4
#define LEARNING_RATE 0.1
#define MIN_STARING_TIME 10

enum BLOB_TYPE { RED, GREEN, BLUE, NONE };

clock_t finish, start;
double duration;



static void compute_odometry(WbDeviceTag left_position_sensor, WbDeviceTag right_position_sensor) {
  double l = wb_position_sensor_get_value(left_position_sensor);
  double r = wb_position_sensor_get_value(right_position_sensor);
  double dl = l * WHEEL_RADIUS;         // distance covered by left wheel in meter
  double dr = r * WHEEL_RADIUS;         // distance covered by right wheel in meter
  double da = (dr - dl) / AXLE_LENGTH;  // delta orientation
  // printf("estimated distance covered by left wheel: %g m.\n", dl);
  // printf("estimated distance covered by right wheel: %g m.\n", dr);
  // printf("estimated change of orientation: %g rad.\n", da);
}

double LLSS_values[2];
double Nt[2] = {0,0};
double objR[2] = {100, 50};
double startingTranslation [1][3] = {
  {0,0,-4}
};

double startingRotation [1][4] = {
  {0, 1, 0, 3}
};

//double w_LL, w_SS;

// Sigmoid function
double sigmoid(double x) {
     double exp_value;
     double return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}


// get the distance between two vectors. assuming these are 2D vectors
double distance(double x1, double x2, double y1, double y2) {
    return sqrt(pow(x1-y1,2.0) + pow(x2-y2,2.0));
}

double modulated_impulsivity(double kval, double hunger){
    return kval * hunger;
}

double discounted_value(double kval, double delay, double value) {
    double DV = 1 / (1 + (kval * delay));
    return DV * value; // discounted value
}



double UCB(double t, double *LLSS_values, double *Nt, double c, int i) {
    return LLSS_values[i] + c * sqrt(log(t+1) / (Nt[i] + 1));
}

double RPE(double alpha, double previous_value, double actual_reward) {
    return previous_value + alpha * (actual_reward - previous_value);
}

double motivation(double value, double hunger) {
    return sigmoid(value / hunger);
}


double satiety = 10.0;
double kval = 0.01;
double alpha = 0.5;
double mk;


int main(int argc, char *argv[]) {
  /* initialize Webots */
  wb_robot_init();
  start = clock();
  
  // current trial's satiety and hunger
  
  
  
  // do this once only
  WbNodeRef robot_node = wb_supervisor_node_get_from_def("epuck");
  WbFieldRef trans_field = wb_supervisor_node_get_field(robot_node, "translation");
  WbFieldRef rot_field = wb_supervisor_node_get_field(robot_node, "rotation");
  
  // Set the initial position of the robot using the supervisor
  wb_supervisor_field_set_sf_vec3f(trans_field, startingTranslation[0]);
  wb_supervisor_field_set_sf_rotation(rot_field, startingRotation[0]);
  
  printf("Temporal discounting task using E-puck\n");
      
  
  
  WbDeviceTag distance_sensor[8], left_motor, right_motor, left_position_sensor, right_position_sensor;
  int i, j;
  int time_step;
  int camera_time_step;
  int width, height; 
  double left_speed = 0.0;
  double right_speed = 0.0;
  int red, blue, green;  
  int choice;
  double TD_value;
  const char *color_names[3] = {"red", "green", "blue"};
  const char *ansi_colors[3] = {ANSI_COLOR_RED, ANSI_COLOR_GREEN, ANSI_COLOR_BLUE};
  enum BLOB_TYPE current_blob;

  double wRed, wGreen; // Weights for Rescorla Wagner learning rule.
                              // In this demonstration value equals weight so no v in equations.
  
  double delta = 2.0; // reward for staring at an object  
  
  // Different colors have different saliencies
  double redRewardRate = 1;
  double greenRewardRate = 0.1;
  //double blueRewardRate = 0.5;
  
  FILE *fpRed;
  fpRed = fopen("red.txt", "w");
  FILE *fpGrn;
  fpGrn = fopen("green.txt", "w");
  FILE *fpBlu;
  fpBlu = fopen("blue.txt", "w");  

  //wRed = wGreen = wBlue = 1.0;  // initialize weights so the robot starts by staring at all 3 colors equally.
  wGreen = wRed = 1.0;
  
  time_step = 256;
  camera_time_step = 1024;
  for (i = 0; i < 8; i++) {
    char device_name[4];

    /* get distance sensors */
    sprintf(device_name, "ps%d", i);
    distance_sensor[i] = wb_robot_get_device(device_name);
    wb_distance_sensor_enable(distance_sensor[i], time_step);
  }


  double sensors_value[8];


  /* get and enable the camera and accelerometer */
  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, camera_time_step);
  width = wb_camera_get_width(camera);
  height = wb_camera_get_height(camera);  
  
  
  WbDeviceTag accelerometer = wb_robot_get_device("accelerometer");
  wb_accelerometer_enable(accelerometer, time_step);

  /* get a handler to the motors and set target position to infinity (speed control). */
  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);

  /* get a handler to the position sensors and enable them. */
  left_position_sensor = wb_robot_get_device("left wheel sensor");
  right_position_sensor = wb_robot_get_device("right wheel sensor");
  wb_position_sensor_enable(left_position_sensor, time_step);
  wb_position_sensor_enable(right_position_sensor, time_step);

  int t = 0; 
  int trial=0;
  bool found= false;
  

  /* main loop */
  while ((wb_robot_step(time_step) != -1) && (trial < TRIALS)) {

    
    /* Get the new camera values */
    const unsigned char *image = wb_camera_get_image(camera);


    /* Reset the sums */
    red = 0;
    green = 0;
    blue = 0;
    
    
    /* camera */
    
    
    
    
    
    
    /* get sensors values */
    for (i = 0; i < 8; i++)
      sensors_value[i] = wb_distance_sensor_get_value(distance_sensor[i]);
    const double *a = wb_accelerometer_get_values(accelerometer);
    // printf("accelerometer values = %0.2f %0.2f %0.2f\n", a[0], a[1], a[2]);


    // check if the robot hit the arena wall.       
    double threshold = 0.25;       
    bool left_obstacle = sensors_value[1] < threshold || sensors_value[0] < threshold;
    bool right_obstacle = sensors_value[3] < threshold || sensors_value[4] < threshold;
    bool front_obstacle = sensors_value[2] < threshold;
    bool obstacle = left_obstacle || right_obstacle || front_obstacle;
    /*
     * Here we analyse the image from the camera. The goal is to detect a
     * blob (a spot of color) of a defined color in the middle of our
     * screen.
     * In order to achieve that we simply parse the image pixels of the
     * center of the image, and sum the color components individually
     */
    for (i = width / 3; i < 2 * width / 3; i++) {
      for (j = height / 2; j < 3 * height / 4; j++) {
        red += wb_camera_image_get_red(image, width, i, j);
        blue += wb_camera_image_get_blue(image, width, i, j);
        green += wb_camera_image_get_green(image, width, i, j);
      }
    }

    if ((red > 3 * green) && (red > 3 * blue)) {
      current_blob = RED;
      found = true;
      choice = 0;

      
      // objR = 100;
      
      
      // wRed += LEARNING_RATE*(redRewardRate*delta - wRed);
      // pause_counter = (int)(defaultStareTime * wRed);
      // fprintf(fpRed,"%i\n", pause_counter);
      // fflush(fpRed);  // flush the file buffer to save latest value.
    }
    else if ((green > 3 * red) && (green > 3 * blue)) {
      current_blob = GREEN;
      found = true;
      // objR = 50;
      choice = 1;
      // Nt[choice] += 1;
      // wGreen += LEARNING_RATE*(greenRewardRate*delta - wGreen);
      // pause_counter += (int)(defaultStareTime * wGreen);
      // fprintf(fpGrn,"%i\n", pause_counter);
      // fflush(fpGrn);  // flush the file buffer to save latest value.
   }

    else {
      current_blob = NONE;
    }    



    // init speeds
    // left_speed = 0.0;
    // right_speed = 0.0;    
    
    const double *trans_value = wb_supervisor_field_get_sf_vec3f(trans_field);    


    // left is red for robot
    left_speed = SPEED * wRed;
    right_speed = SPEED * wGreen;  
    if (obstacle) { 
    
    
    }
    if (front_obstacle || left_obstacle || right_obstacle){
      // modify speeds according to obstacles
      if (front_obstacle){
        // turn back, but slightly right to not block the robot
        left_speed = 0.0;
        right_speed= -1.0;
        
      } else if (left_obstacle){
        // turn right
        left_speed = 3.0;
        right_speed = -1.0;
      } else if (right_obstacle){
        // turn left
        left_speed = -1.0;
        right_speed = 3.0;
      
      }
    }
    // else {
      // left_speed = 2.0;
      // right_speed = 2.0;
    // }
   
    if (found){
      
      
      

  
      wb_supervisor_field_set_sf_vec3f(trans_field, startingTranslation[0]);
      wb_supervisor_field_set_sf_rotation(rot_field, startingRotation[0]);
    
      finish = clock();
      duration = (double)(finish-start)/CLOCKS_PER_SEC;
      printf("Duration: %g sec.\n", duration);
      
      Nt[choice] += 1;
      
      TD_value = discounted_value(mk, duration, objR[choice]);
      LLSS_values[choice] = RPE(alpha, LLSS_values[choice], TD_value);
      trial++;
      printf("Trial: %i .\n", trial);
      found = false;
      // fprintf(trial, "%i\n");
      
      start = clock();      


    }
    
    
    
    
    
    /* set speed values */
    wb_motor_set_velocity(left_motor, left_speed);
    wb_motor_set_velocity(right_motor, right_speed);
  }
  fclose(fpRed);
  fclose(fpGrn);
  fclose(fpBlu);
  wb_robot_cleanup();

  return 0;
}
