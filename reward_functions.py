import numpy as np
from CarlaEnv.wrappers import angle_diff, vector
import math
import carla

low_speed_timer = 0
max_distance    = 3.5  # Max distance from center before terminating
target_speed    = 8.0 # kmh 20.0
max_speed   = 25.0 

def create_reward_fn(reward_fn, max_speed=-1):
    """
        Wraps input reward function in a function that adds the
        custom termination logic used in these experiments

        reward_fn (function(CarlaEnv)):
            A function that calculates the agent's reward given
            the current state of the environment. 
        max_speed:
            Optional termination criteria that will terminate the
            agent when it surpasses this speed.
            (If training with reward_kendal, set this to 20)
    """
    def func(env):
        terminal_reason = "Running..."
        #max_speed   = 15.0 

        # Stop if speed is less than 1.0 km/h after the first 5s of an episode
        global low_speed_timer
        low_speed_timer += 1.0 / env.fps
        speed = env.vehicle.get_speed()
        if low_speed_timer > 5.0 and speed < 1.0 / 3.6:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        # Stop if distance from center > max distance
        if env.distance_from_center > max_distance:
            env.terminal_state = True
            terminal_reason = "Off-track"

        # Stop if speed is too high
        #if max_speed > 0 and speed > max_speed/ 3.6:
        #    env.terminal_state = True
        #    terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            reward -= 10

        if env.terminal_state:
            env.extra_info.extend([
                terminal_reason,
                ""
            ])
        return reward
    return func

#---------------------------------------------------
# Create reward functions dict
#---------------------------------------------------

reward_functions = {}

# Kenall's (Learn to Drive in a Day) reward function
def reward_kendall(env):
    speed_kmh = 3.6 * env.vehicle.get_speed()
    return speed_kmh

reward_functions["reward_kendall"] = create_reward_fn(reward_kendall)

# Our reward function (additive)
def reward_speed_centering_angle_add(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               + centering factor (1 when centered, 0 when not)
               + angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 5.0 # km/h
    max_speed = 25.0 # km/h

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Final reward
    reward = speed_reward + centering_factor + angle_factor

    return reward

reward_functions["reward_speed_centering_angle_add"] = create_reward_fn(reward_speed_centering_angle_add)



# Our reward function (additive)
def reward_speed_centering_angle_add2(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               + centering factor (1 when centered, 0 when not)
               + angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 5.0 # km/h
    max_speed = 25.0 # km/h
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)
    
    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / (max_distance-0.5), 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor =max(1.0 - abs(angle / np.deg2rad(20)), 0.0)
    # Final reward
    
    speed_kmh = 3.6 * env.vehicle.get_speed()
    factor_speed = 1.0 if (speed_kmh)>=min_speed else 0
    reward = (speed_reward + centering_factor*2 + angle_factor + env.laps_completed)*factor_speed

    return reward

reward_functions["reward_speed_centering_angle_add2"] = create_reward_fn(reward_speed_centering_angle_add2)

def reward_speed_distance(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               + centering factor (1 when centered, 0 when not)
               + angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 5.0 # km/h
    max_speed = 15.0 # km/h
    target_speed_f = 8.0

    # Get angle difference between closest waypoint and vehicle forward vector

    speed_kmh = 3.6 * env.vehicle.get_speed()
    speed_reward=10.0
    if speed_kmh < min_speed:
        speed_reward = -10.0                   # When speed is in [0, min_speed] range
        # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > max_speed:
        #env.terminal_state = True
        speed_reward = -10.0  
    elif speed_kmh > target_speed_f:
        #env.terminal_state = True
        speed_reward = 10*(1-(target_speed_f/speed_kmh))
    # Interpolated from 1 when centered to 0 when 3 m from center
    #centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    #angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Final reward
    reward = speed_reward #* abs(env.laps_completed * 100.0)

    return reward

reward_functions["reward_speed_distance"] = create_reward_fn(reward_speed_distance)

# Our reward function (multiplicative)
def reward_speed_centering_angle_multiply(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 5.0 # km/h
    max_speed = 25.0 # km/h

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    # Get angle difference between closest waypoint and vehicle forward vector

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor

    return reward

reward_functions["reward_speed_centering_angle_multiply"] = create_reward_fn(reward_speed_centering_angle_multiply)



def reward_centering_steer1(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 10.0 # km/h
    max_speed = 20.0 # km/h
    target_speed_f    = 15.0 # kmh 20.0



    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed_f:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed_f) / (max_speed-target_speed_f)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    centering_factor = 1.0 - env.distance_from_center / max_distance

    # Get current steering angle
    steer_angle =  env.vehicle.get_control().steer

    # Calculate steering reward (encourage smooth steering)
    steer_reward = 1 - np.abs(steer_angle) 

    # Final reward
    reward = speed_reward*0.5 + centering_factor*0.3 + steer_reward*0.2

    return reward

reward_functions["reward_centering_steer1"] = create_reward_fn(reward_centering_steer1)


def reward_centering_steer(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    target_speed_f    = 20.0 # kmh 20.0
    # Assuming you have the current waypoint, the vehicle instance, and the current steering angle
    current_waypoint = env.current_waypoint
    vehicle = env.vehicle
    
    waypoint_forward_vector = current_waypoint.transform.rotation.get_forward_vector()
    vehicle_forward_vector = vehicle.get_transform().rotation.get_forward_vector()
    steering_angle = vehicle.get_control().steer
    rotated_vehicle_forward_vector = carla.Vector3D(
        math.cos(steering_angle) * vehicle_forward_vector.x - math.sin(steering_angle) * vehicle_forward_vector.y,
        math.sin(steering_angle) * vehicle_forward_vector.x + math.cos(steering_angle) * vehicle_forward_vector.y,
        vehicle_forward_vector.z
    )

    # Calculate the angle difference between the rotated vehicle forward vector and the waypoint forward vector
    dot_product = waypoint_forward_vector.x * rotated_vehicle_forward_vector.x + \
                waypoint_forward_vector.y * rotated_vehicle_forward_vector.y

    magnitude_product = math.sqrt(waypoint_forward_vector.x**2 + waypoint_forward_vector.y**2) * \
                        math.sqrt(rotated_vehicle_forward_vector.x**2 + rotated_vehicle_forward_vector.y**2)

    angle_difference = math.acos(dot_product / magnitude_product)

    # Convert the angle to degrees
    angle_difference_degrees = math.degrees(angle_difference)/180.0


    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                   
        speed_reward = 0
    elif speed_kmh > max_speed:                   
        speed_reward = max_speed/speed_kmh
    elif speed_kmh > target_speed_f:                   
        speed_reward = 1.0 - abs(speed_kmh-target_speed_f)/(max_speed-target_speed_f)   
    else:
        speed_reward = 1.0
    
    centering_reward = max(1.0 - env.distance_from_center / max_distance, 0.0)

    #env.distance_traveled

    # Get current steering angle
    steer_reward = max(1.0 - abs(angle_difference_degrees / np.deg2rad(20)), 0.0)

    # Final reward
    speed_reward=speed_reward*0.3
    centering_reward=centering_reward*0.4
    steer_reward=steer_reward*0.2
    distance_reward=abs(env.distance_traveled)*0.1

    print("Components")
    print(speed_reward)
    print(centering_reward)
    print(steer_reward)
    print(distance_reward)
    print("reward")

    reward = speed_reward+centering_reward+steer_reward+distance_reward
    print(reward)

    return reward

reward_functions["reward_centering_steer"] = create_reward_fn(reward_centering_steer)

def reward_centering_steer_multiply(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    target_speed_f    = 20.0 # kmh 20.0
    # Assuming you have the current waypoint, the vehicle instance, and the current steering angle
    current_waypoint = env.current_waypoint
    vehicle = env.vehicle
    
    waypoint_forward_vector = current_waypoint.transform.rotation.get_forward_vector()
    vehicle_forward_vector = vehicle.get_transform().rotation.get_forward_vector()
    steering_angle = vehicle.get_control().steer
    rotated_vehicle_forward_vector = carla.Vector3D(
        math.cos(steering_angle) * vehicle_forward_vector.x - math.sin(steering_angle) * vehicle_forward_vector.y,
        math.sin(steering_angle) * vehicle_forward_vector.x + math.cos(steering_angle) * vehicle_forward_vector.y,
        vehicle_forward_vector.z
    )

    # Calculate the angle difference between the rotated vehicle forward vector and the waypoint forward vector
    dot_product = waypoint_forward_vector.x * rotated_vehicle_forward_vector.x + \
                waypoint_forward_vector.y * rotated_vehicle_forward_vector.y

    magnitude_product = math.sqrt(waypoint_forward_vector.x**2 + waypoint_forward_vector.y**2) * \
                        math.sqrt(rotated_vehicle_forward_vector.x**2 + rotated_vehicle_forward_vector.y**2)

    angle_difference = math.acos(dot_product / magnitude_product)

    # Convert the angle to degrees
    angle_difference_degrees = math.degrees(angle_difference)/180.0


    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                   
        speed_reward = 0
    elif speed_kmh > max_speed:                   
        speed_reward = max_speed/speed_kmh
    elif speed_kmh > target_speed_f:                   
        speed_reward = 1.0 - abs(speed_kmh-target_speed_f)/(max_speed-target_speed_f)   
    else:
        speed_reward = 1.0
    
    centering_reward = max(1.0 - env.distance_from_center / max_distance, 0.0)

    #env.distance_traveled

    # Get current steering angle
    steer_reward = max(1.0 - abs(angle_difference_degrees / np.deg2rad(20)), 0.0)

    # Final reward
    speed_reward=speed_reward*3
    centering_reward=centering_reward*4
    steer_reward=steer_reward*2
    distance_reward=abs(env.distance_traveled)

    print("Components")
    print(speed_reward)
    print(centering_reward)
    print(steer_reward)
    print(distance_reward)
    print("reward")

    reward = speed_reward*centering_reward*steer_reward*distance_reward
    print(reward)

    return reward

reward_functions["reward_centering_steer_multiply"] = create_reward_fn(reward_centering_steer_multiply)



# def reward_speed_center_distance(env):
#     """
#         reward = Positive speed reward for being close to target speed,
#                  however, quick decline in reward beyond target speed
#                + centering factor (1 when centered, 0 when not)
#                + distance factor (1 when maintaining safe distance from car in front, 0 when too close)
#     """

#     min_speed = 15.0  # km/h
#     max_speed = 25.0  # km/h
#     safe_distance = 10.0  # meters

#     # Calculate speed reward
#     speed_kmh = 3.6 * env.vehicle.get_speed()
#     if speed_kmh < min_speed:
#         speed_reward = speed_kmh / min_speed
#     elif speed_kmh > max_speed:
#         speed_reward = 1.0 - (speed_kmh - max_speed) / (max_speed - min_speed)
#     else:
#         speed_reward = 1.0

#     # Calculate centering reward
#     centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

#     # Calculate distance reward
#     distance_to_car_in_front = env.get_distance_to_front_vehicle()
#     distance_factor = max(1.0 - distance_to_car_in_front / safe_distance, 0.0)

#     # Final reward
#     reward = speed_reward + centering_factor + distance_factor

#     return reward

# reward_functions["reward_speed_center_distance"] = create_reward_fn(reward_speed_center_distance)


# def reward_speed_centering_distance_2(env):
#     """
#         Reward function based on speed, centering, and distance from the car in front.
#     """

#     # Speed-related parameters
#     min_speed = 15.0  # Minimum desired speed in km/h
#     max_speed = 25.0  # Maximum desired speed in km/h
#     target_distance = 10.0  # Target distance from the car in front in meters

#     # Centering-related parameters
#     max_distance = 3.0  # Max distance from center before terminating

#     # Get angle difference between closest waypoint and vehicle forward vector
#     fwd = vector(env.vehicle.get_velocity())
#     wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
#     angle = angle_diff(fwd, wp_fwd)

#     # Speed reward calculation
#     speed_kmh = 3.6 * env.vehicle.get_speed()
#     if min_speed <= speed_kmh <= max_speed:
#         speed_reward = 1.0
#     else:
#         speed_reward = 0.0

#     # Centering factor calculation
#     centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

#     # Distance from the car in front reward calculation
#     distance_to_front_car = env.distance_to_front_vehicle
#     distance_reward = 1.0 - abs(distance_to_front_car - target_distance) / target_distance

#     # Final reward
#     reward = speed_reward * centering_factor * distance_reward

#     return reward

# # Create the new reward function
# reward_functions["reward_speed_centering_distance_2"] = create_reward_fn(reward_speed_centering_distance_2)