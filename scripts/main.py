#!/usr/bin/python3
# -- coding: utf-8 --
import numpy as np
import math
import time
import random
import rospy
from geometry_msgs.msg import PoseStamped
from olfaction_msgs.msg import gas_sensor

import SensorModel
import FilterModel
import LikelyModel
import ResampleModel
import ConstraintModel
import DrawParams
import DrawParticles
from ClassType import Source
from ClassType import Theta
from ClassType import Sensor
import matplotlib.pyplot as plt
from operator import add

def callback(data):
    global current_sensor_data
    current_sensor_data = data.raw_air

def main():

    # rospy.init_node('robot_navigator', anonymous=True)
    rospy.init_node('robot_goal_publisher', anonymous=True)
    global current_sensor_data
    current_sensor_data = None
    rospy.Subscriber('/PID/Sensor_reading', gas_sensor, callback)  # 替换为实际的话题名称

    # 创建发布者，发布机器人位置指令
    #pub = rospy.Publisher('/robot_position_command', PoseStamped, queue_size=10)
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    rospy.sleep(0.5)
    S = Source(1.5, 3, 2, 0.5, 0 * math.pi/180, 1, 8)
    m = Sensor(5e-4, 0.8, 1e-4, 0.3)

    ## the position of the robot from time 0 to time k
    pos_arr=[]
    start = [8.836,5.062]
    pos = start
    pos_arr.append(pos)

    ## the move step of the robot
    step = 0.5

    ## the sensor data array
    data_arr = []

    ## search domain
    domain_x = [0, 10]
    domain_y = [0, 6]
    domain = [domain_x, domain_y]

    ## particle num
    N = 10000

    theta = Theta(domain[0][0] + (domain[0][1]-domain[0][0]) * np.random.rand(N),
                   domain[1][0] + (domain[1][1]-domain[1][0]) * np.random.rand(N),
                   np.random.gamma(2, S.q, N),
                   S.u + 2 * np.random.randn(N),
                   S.phi * 0.9 + 10 * math.pi/180 * np.random.randn(N),
                   S.d + 2 * np.random.rand(N),
                   S.tao + 2 * np.random.rand(N) -2)

    w_par = np.ones(N) / N

    likely = lambda S, data, m: LikelyModel.likelyhood(S, data, m, pos)
    constraint = ConstraintModel.gCon

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)  # 设置x轴范围（根据你的数据进行调整）
    ax.set_ylim(0, 6)  # 设置y轴范围（根据你的数据进行调整）
    scatter = ax.scatter([], [], s=3)  # 初始为空

    rate = rospy.Rate(1)  # 1Hz

    

    for i in range(100):
        rospy.sleep(5)

        while current_sensor_data is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        #data = SensorModel.sensor(S, pos, m)
        data = current_sensor_data

        data_arr.append(data)
        # time.sleep(1)
        theta, w_par = FilterModel.filter(theta, w_par, data, likely, m, constraint)


        DrawParticles.draw_pars(theta.x, theta.y, scatter)

        # Action set
        pos_new_arr = [list(map(add, pos, [-step, 0])),
                       list(map(add, pos, [step, 0])),
                       list(map(add, pos, [0, step])),
                       list(map(add, pos, [0, -step]))]
        
        Nz = 25
        mm = 1

        _, index_z = ResampleModel.resample(w_par, Nz)
        reward = []

        for i in range(len(pos_new_arr)):
            pos_new = pos_new_arr[i]
            if pos_new[0] < domain_x[0] or pos_new[0] > domain_x[1] or \
               pos_new[1] < domain_y[0] or pos_new[1] > domain_y[1]:
                reward.append(None)
                continue
            
            infoGain = 0

            for j in range(Nz):
                s_sample = Theta(theta.x[index_z[j]],
                                 theta.y[index_z[j]],
                                 theta.q[index_z[j]],
                                 theta.u[index_z[j]],
                                 theta.phi[index_z[j]],
                                 theta.d[index_z[j]],
                                 theta.tao[index_z[j]])
                
                for k in range(mm):
                    data_sample = SensorModel.sensor(s_sample, pos_new, m)
                    likelyhoold = LikelyModel.likelyhood(theta, data_sample, m, pos_new)

                    w_par_sample = w_par * likelyhoold
                    w_par_sample = w_par_sample / np.sum(w_par_sample)
                    

                    # Entropy
                    infoGain = infoGain - (-np.sum(w_par_sample * np.log2(w_par_sample + (w_par_sample == 0))))

            reward.append(infoGain)

        max_index = reward.index(max(list(filter(lambda x: x is not None, reward))))

        pos = pos_new_arr[max_index]

        # 发布机器人位置指令
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = 'map'  
        goal.pose.position.x = pos[0]
        goal.pose.position.y = pos[1]
        
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0

        goal_pub.publish(goal)

        pos_arr.append(goal)

        print(pos)

        print(theta.x)

        # stop criteria
        _, idx_stop = ResampleModel.resample(w_par, N)
        Covar = np.cov(theta.x[idx_stop], theta.y[idx_stop])
        Spread = np.sqrt(np.trace(Covar))
        if Spread < 0.2:
            break

        rate.sleep()
        

    _, index = ResampleModel.resample(w_par, N)

    theta.x = theta.x[index]
    theta.y = theta.y[index]
    theta.q = theta.q[index]
    theta.u = theta.u[index]
    theta.phi = theta.phi[index]
    theta.d = theta.d[index]
    theta.tao = theta.tao[index]
    print(theta.x)

    DrawParams.draw_param(theta)




if __name__ == "__main__":
    main()

        
