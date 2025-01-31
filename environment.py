"""
An environment class that wraps a gym environment and adds some additional functionality
"""

import gym
import numpy as np

class Environment():
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self):
        raise NotImplementedError
    
    def restart(self):
        """
        Reset the environment to the initial state when reset was last called
        """
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
    
class Quadrotor(Environment):
    def __init__(
            self,
            mass: float=1.0, # kg
            arm_length: float=0.2, # m
            Ixx: float=0.005, # kg-m²
            Iyy: float=0.005, # kg-m²
            Izz: float=0.006, # kg-m²
            torque_constant: float=0.017,
            gravity: float=9.80665, # m/s²
            timestep: float=0.05, # s
            max_time: float=10.0, # s
            num_obstacles: int=0,
            spatial_bounds: tuple=((-5, 5), (-5, 5), (-5, 5)), # m
            obstacle_radius_bounds: tuple=(0.1, 0.2), # m
            goal_radius: float=0.1, # m
                 ):
        super().__init__('Quadrotor')

        # Quadrotor parameters
        self.mass = mass
        self.arm_length = arm_length
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.torque_constant = torque_constant
        self.gravity = gravity
        self.timestep = timestep

        self._hover_thrust = mass*gravity/4



        self.max_time = max_time

        # Environment parameters
        self.num_obstacles = num_obstacles
        self.spatial_bounds = spatial_bounds
        self.obstacle_radius_bounds = obstacle_radius_bounds
        self.goal_radius = goal_radius

        self._xbounds = spatial_bounds[0]
        self._ybounds = spatial_bounds[1]
        self._zbounds = spatial_bounds[2]

        self.state_dict = {
            'quadrotor': np.zeros(12),
            'obstacles': np.zeros((num_obstacles, 4)),
            'goal': np.zeros(3)
        }

        self._initial_state = None
    
    def _pack_state(self):
        """
        Pack the state of the environment into a single array

        Args: 
            None
        Returns:
            state (array): A 12x1 array of the state of the environment
        """
        pass



    def _unpack_state(self, state):
        pass


    def _spwan_quadrotor(self):
        """
        Spawns a quadrotor at a random location within the spatial bounds of the environment

        Args: None

        Returns:
            state (array): A 12x1 array of the state of the quadrotor
        """

        self.quadrotor = np.array([
            np.random.uniform(self._xbounds[0], self._xbounds[1]),
            np.random.uniform(self._ybounds[0], self._ybounds[1]),
            np.random.uniform(self._zbounds[0], self._zbounds[1]),
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ])

        return self.quadrotor

    def _spawn_obstacles(self):
        pass

    def _spawn_goal(self):
        pass
        
    def _dynamics(self, 
                  state, 
                  control):
        
        """
        Implementation of simple quadrotor dynamics using Euler angles and Euler integration with a fixed timestep
        
        Args:
            state (array): A 12x1 array of the state of the quadrotor
            control (array): A 4x1 array of the control inputs to the quadrotor

        Returns:
            state (array): A 12x1 array of the state of the quadrotor at the next time step
        """

        x, y, z, xdot, ydot, zdot, phi, theta, psi, p, q, r = state
        u1, u2, u3, u4 = control

        mass = self.mass
        arm_length = self.arm_length
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        torque_constant = self.torque_constant
        gravity = self.gravity
        timestep = self.timestep

        # Rotation matrix R (using Euler angles)
        R = np.array([
            [                                      np.cos(theta)*np.cos(psi),                                          np.cos(theta)*np.sin(psi),               -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),    np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),    np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi),    np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.sin(psi),    np.cos(phi)*np.cos(theta)]
                       ])
        
        # Translational acceleration
        a = 1/mass*(R@np.array([[0], [0], [u1 + u2 + u3 + u4]]) + np.array([[0], [0], [-mass*gravity]]))

        # Angular velocity in the inertial reference frame (This is where the gimbal lock singularity appears)
        omegadot = np.array([
            [1,    np.sin(phi)*np.tan(theta),    np.cos(phi)*np.tan(theta)],
            [0,                  np.cos(phi),                 -np.sin(phi)],
            [0,    np.sin(phi)/np.cos(theta),    np.cos(phi)/np.cos(theta)]
        ])@np.array([[p], [q], [r]])

        # Angular acceleration in the body reference frame
        alpha = np.array([[(np.sqrt(2)/2*(u1 + u3 - u2 - u4)*arm_length - (Izz - Iyy)*q*r)/Ixx],
                          [(np.sqrt(2)/2*(u3 + u4 - u1 - u2)*arm_length - (Izz - Ixx)*p*r)/Iyy],
                          [                                        (torque_constant*(u1 + u4 - u2 - u3))/Izz]
                        ])

        rates = np.vstack(
            (np.array([[xdot], [ydot], [zdot]]),
            a,
            omegadot,
            alpha)
        ).flatten()
        return state + rates*timestep

    
    def reset(self):
        return self.env.reset()
    
    def restart(self):
        return self.env.restart()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
class QuadrotorSwarm(Quadrotor):
    pass

