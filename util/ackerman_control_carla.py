from simple_pid import PID
import numpy as np
import carla
from util.transforms import build_se3_transform, se3_to_components

class VehicleControl:
    def __init__(self, vehicle) -> None:
        self.vehicle = vehicle
        self.pid = PID(0.05, 0.0, 0.05)
        self.throttle = 0
        self.vel  = 0
        self.prev_vel = 0
        self.prev_acc = 0
        self.throttle_list = []
        self.steer_list = []
        self.brake_list = []

    def print_control_cost(self):
        print('Throttle Cost:', np.sum(np.abs(self.throttle_list)))
        print('Steer Cost:', np.sum(np.abs(self.steer_list)))
        print('Brake Cost:', np.sum(np.abs(self.brake_list)))
        print('Throttle + Steer = ', np.sum(np.abs(self.throttle_list)) + np.sum(np.abs(self.steer_list)))
        print('All = ', np.sum(np.abs(self.throttle_list)) + np.sum(np.abs(self.steer_list)) + np.sum(self.brake_list))

    def get_gt_state(self, tf_matrix):
        ego_tf = self.vehicle.get_transform().get_matrix()
        ego_tf_odom = np.linalg.pinv(tf_matrix) @ ego_tf
        x, y, z, roll, pitch, yaw = se3_to_components(ego_tf_odom)
        vel_ego = self.vehicle.get_velocity()
        speed = np.linalg.norm(np.array([vel_ego.x, vel_ego.y]))

        ## Acceleration 
        acc = self.vehicle.get_acceleration()
        print('acc.x, acc.y: ', acc.x, acc.y)
        acc_tf = np.array(build_se3_transform([acc.x, acc.y, acc.z, 0, 0, 0]))
        odom_tf_rot = np.copy(tf_matrix)
        odom_tf_rot[:3, 3] = 0
        acc_tf_odom = np.linalg.pinv(odom_tf_rot) @ acc_tf
        acc_x, acc_y, _, _, _, _ =  se3_to_components(acc_tf_odom)
        
        return speed, x, -y, -yaw, acc_x, -acc_y



    def apply_control(self, v, w, target_acc, best_s=1):
        print('\n[INFO] Ackerman Controller')
        physics_control = self.vehicle.get_physics_control()
        max_steer_angle_list = []
        # For each Wheel Physics Control, print maximum steer angle
        for wheel in physics_control.wheels:
            max_steer_angle_list.append(wheel.max_steer_angle)
        max_steer_angle = max(max_steer_angle_list)*np.pi/180

        throttle_lower_border = -(0.01*9.81*physics_control.mass + 0.5*0.3*2.37*1.184*self.vel**2 + \
            9.81*physics_control.mass*np.sin(self.vehicle.get_transform().rotation.pitch*2*np.pi/360))/physics_control.mass

        brake_upper_border = throttle_lower_border + -500/physics_control.mass
        self.pid.setpoint = target_acc
        

        vel = self.vehicle.get_velocity()
        angvel = self.vehicle.get_velocity()
        self.vel = (vel.x**2 + vel.y**2 + vel.z**2)**0.5
        print('Current Speed: ', self.vel)
        print('Current wdot(angular vel): ', angvel.z)
        acc = (self.vel - self.prev_vel)/0.1

        if acc>10:
            control = self.pid(0)
        else:
            self.prev_acc = (self.prev_acc*4 + acc)/5
            #acc = self.vehicle.get_acceleration()
            #acc = (acc.x**2 + acc.y**2 + acc.z**2)**0.5
            control = self.pid(self.prev_acc)

        steer = np.arctan(w*3.0/v)
        steer = -steer/max_steer_angle
        throttle = 0
        brake = 0
        self.throttle = np.clip(self.throttle + control,-4.0, 4.0)

        if self.throttle>throttle_lower_border:
            throttle = (self.throttle-throttle_lower_border)/4
            brake = 0
        elif self.throttle> brake_upper_border:
            brake = 0
            throttle = 0
        else:
            brake = (brake_upper_border-self.throttle)/4
            throttle = 0
        brake = np.clip(brake, 0.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        
        # Scaling
        if self.vel < 5:
            throttle_s = throttle * best_s
            steer_s = steer * best_s
        else: 
            throttle_s = throttle * 1
            steer_s = steer * 1


        print('Controls Given: ')
        print( "Steer = ", steer_s)
        print("Throttle = ", throttle_s)
        print('Brake = ', brake)

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_s, steer=steer_s, brake=brake))
        self.throttle_list.append(throttle)
        self.steer_list.append(steer)
        self.brake_list.append(brake)

        self.prev_vel = self.vel  
        # This is old velocity, without applying the current control
        # The currnet control will be applied after world.tick()




        
