import random
import math
import numpy as np
import matplotlib.pyplot as plt 

from scipy.spatial import ConvexHull 
from scipy.interpolate import splprep, splev, BSpline 

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

# PyGame init
width = 1500
height = 1080  
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = False
draw_screen = True
class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False
        self.buildtrack = True
        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        self.velocity = 15

        # Create the car.
        

        # Record steps.
        self.num_steps = 0

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        cone_pos = self.get_cone_pos()
        self.obstacle_pos_big = []
        self.obstacle_pos_small = []  


        for i in range(len(cone_pos[0])-1): 
            self.obstacles.append(self.create_obstacle(cone_pos[0][i], cone_pos[1][i], 5))
            self.obstacles.append(self.create_obstacle(cone_pos[2][i], cone_pos[3][i], 5))
        for i in range(len(cone_pos[0])-1): 
            self.obstacle_pos_big.append([cone_pos[0][i], cone_pos[1][i]])
        for i in range(len(cone_pos[0])-1): 
            self.obstacle_pos_small.append([cone_pos[2][i], cone_pos[3][i]])
        if self.buildtrack: 
            self.build_track(self.obstacle_pos_big)
            self.build_track(self.obstacle_pos_small)
        self.spawn_index = random.randint(0, len(self.obstacle_pos_big)-1)
        self.spawn_point = self.midpoint(self.obstacle_pos_big[self.spawn_index], self.obstacle_pos_small[self.spawn_index])
        print("MIDPOINT: " + str(self.spawn_point))

        self.create_car(self.spawn_point[0], self.spawn_point[1], 5)

    def midpoint(self, p1, p2): 
        return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]

    def get_cone_pos(self): 
        pointCount = random.randint(20, 30)

        points = np.random.randint(500, size=(pointCount, 2))

        hull = ConvexHull(points)

        pointsBig = (points[hull.vertices]*1.4)+200
        pointsSmall = (points[hull.vertices]*0.9)+330


        tck_big, u_big = splprep(pointsBig.T, u=None, s=0.0, per=1)
        u_bnew = np.linspace(u_big.min(), u_big.max(), 1000)
        x_big, y_big = splev(u_bnew, tck_big, der=0)

        tck_small, u_small= splprep(pointsSmall.T, u=None, s=0.0, per=1)
        u_snew = np.linspace(u_big.min(), u_small.max(), 1000)
        x_small, y_small = splev(u_snew, tck_small, der=0)


        plt.plot(x_big[0::40], y_big[0::40], 'bo', lw=0.5)
        plt.plot(x_small[0::40], y_small[0::40], 'ro', lw=0.5)


        maxpoint = np.amax(pointsBig, axis=0)
        minpoint = np.amin(pointsBig, axis=0)

        middle = self.midpoint(maxpoint, minpoint)
        print("MIDDLE: " , middle)

        x = self.midpoint([x_big[0], y_big[0]], [x_small[0], y_small[0]])
        
        plt.plot(x[0], x[1], 'go')

        plt.show()

        return [x_big[0::30], y_big[0::30], x_small[0::30], y_small[0::30]]
    
    def build_track(self, obstacle_pos): 
        body = pymunk.Body(pymunk.inf, pymunk.inf)
        body.position = pymunk.Vec2d(100, 50)
        for i in range(len(obstacle_pos)-1):
            if i == len(obstacle_pos)-2: 
                line = pymunk.Segment(body, obstacle_pos[i], obstacle_pos[0], 2)
                line.color = THECOLORS["green"]
                self.space.add(line)
            else: 
                line = pymunk.Segment(body, obstacle_pos[i], obstacle_pos[i+1], 2)
                line.color = THECOLORS["green"]
                self.space.add(line)


    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = pymunk.Vec2d(x+100, y+50)
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, r, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = pymunk.Vec2d(x+100, y+50)
        self.car_shape = pymunk.Circle(self.car_body, r)
        self.car_shape.color = THECOLORS["purple"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        reward = 0 
        
        if action == 0:  # Turn left.
            self.car_body.angle -= .2
            reward -= 500
        elif action == 1:  # Turn right.
            self.car_body.angle += .2
            reward -= 3 
        elif action == 2: 
            if self.velocity < 30:
                self.velocity += 2
            reward += 5
        elif action == 3: 
            self.velocity -= 2 
            reward -= 3
        else: 
            reward += 5


        #print("PREV: " + str(previous_location)
        previous_location = self.car_body.position[0], self.car_body.position[1]
        
        #print(dist_from_prev)
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = self.velocity * driving_direction

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        draw(screen, self.space)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()
        
        dist_from_prev = math.hypot(previous_location[0] - self.car_body.position[0], previous_location[1] - self.car_body.position[1])

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings, walls = self.get_sonar_readings(x, y, self.car_body.angle)
        normalized_readings = [(x-20.0)/20.0 for x in readings] 
        state = np.array([normalized_readings])

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed2(walls): 
            self.crashed = True
            reward = -1000
            self.recover_from_crash(driving_direction)
            print("CRASHED 2222222")
        elif self.car_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.recover_from_crash(driving_direction)
            print("CRASHED 11111")
        elif dist_from_prev < 0.3: 
            reward -= 300
        else:
            # Higher readings are better, so return the sum.
            reward += -5 + int(self.sum_readings(readings) / 10)
            if self.velocity<0: 
                reward -= 10 

        reward += self.velocity/5
        self.num_steps += 1

        return reward, state

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        speed = random.randint(20, 200)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    def car_is_crashed2(self, walls): 
        if 1 in walls: 
            return True 
        else: 
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        self.velocity = 15
        self.spawn_index = random.randint(0, len(self.obstacle_pos_big)-1)
        self.spawn_point = self.midpoint(self.obstacle_pos_big[self.spawn_index], self.obstacle_pos_small[self.spawn_index])
        self.car_body.position = self.spawn_point[0]+100, self.spawn_point[1]+50
        self.car_body.velocity = self.velocity * (driving_direction * -1)
        draw(screen, self.space)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()
        '''
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -40 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["grey7"])  # Red is scary!
                draw(screen, self.space)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()
    '''
    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        walls = [] 
        
        # Make our arms.
        arm1 = self.make_sonar_arm(x, y)
        arm2 = arm1
        arm3 = arm1
        arm4 = arm1
        arm5 = arm1
        arm6 = arm1
        arm7 = arm1 
        arm8 = arm1 
        arm9 = arm1 
        arm10 = arm1

        # Rotate them and get readings.
        #Uncommenting the readings will give you more sensors.
        readings.append(self.get_arm_distance(arm1, x, y, angle, 0.75))
        #readings.append(self.get_arm_distance(arm2, x, y, angle, 0.583))
        #readings.append(self.get_arm_distance(arm3, x, y, angle, 0.416))
        #readings.append(self.get_arm_distance(arm4, x, y, angle, 0.25))
        readings.append(self.get_arm_distance(arm5, x, y, angle, 0.0))
        #readings.append(self.get_arm_distance(arm6, x, y, angle, -0.083))
        #readings.append(self.get_arm_distance(arm7, x, y, angle, -0.25))
        #readings.append(self.get_arm_distance(arm8, x, y, angle, -0.416))
        #readings.append(self.get_arm_distance(arm9, x, y, angle, -0.583))
        readings.append(self.get_arm_distance(arm10, x, y, angle, -0.75))
       


        walls_left = self.make_sonar_arm2(x, y)
        walls_middle = walls_left
        walls_right = walls_left
        walls_back = self.make_sonar_arm2(x, y, distance=4)
        walls_backl = walls_back
        walls_backf = walls_back


        # Rotate them and get readings.
        #detects the barrier but does not parse that information to the neural net. 
        #used for detecting a crash into a barrier without giving the neural net that info 
        walls.append(self.get_arm_distance2(walls_left, x, y, angle, 0.3))
        walls.append(self.get_arm_distance2(walls_middle, x, y, angle, 0))
        walls.append(self.get_arm_distance2(walls_right, x, y, angle, -0.3))
        walls.append(self.get_arm_distance2(walls_back, x, y, angle, 3.14))
        walls.append(self.get_arm_distance2(walls_backl, x, y, angle, 2.5))
        walls.append(self.get_arm_distance2(walls_backf, x, y, angle, 3.64))




        if show_sensors:
            pygame.display.update()

        return readings, walls

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i
    def get_arm_distance2(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not2(obs) != 0:
                    print('yikes')
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return 0

    def make_sonar_arm(self, x, y, distance=10):
        spread = 10  # Default spread.
        #Distance is Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def make_sonar_arm2(self, x, y, distance=10):
        spread = 10  # Default spread.
        # Gap before first sensor.
        arm_points = [] 
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        arm_points.append((distance+x, y))
        return arm_points


    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black'] or reading == THECOLORS['green']:
            return 0
        else:  
            return 1
    def get_track_or_not2(self, reading): 
        if reading == THECOLORS['green']: 
            return 1 
        else: 
            return 0
    

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 5)))

