import numpy as np
    
from environments.env import Env
import gymnasium as gym

class Quadrotor(Env):
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
            max_steps: int=200,
            num_obstacles: int=0,
            spatial_bounds: tuple=((-5, 5), (-5, 5), (-5, 5)), # m
            obstacle_radius_bounds: tuple=(0.1, 0.2), # m
            goal_radius: float=0.1, # m
            ):

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


        self.max_steps = max_steps
        self.max_time = max_steps*timestep

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

        self._steps = 0

        # Define the observation and action spaces
        # self.observation_space = gym.spaces.Box(
        #     low=-5, high=5, shape=(12,), dtype=np.float32)
        # self.action_space = gym.spaces.Box(
        #     low=-1, high=1, shape=(4,), dtype=np.float32)
        
        
    
    def _pack_state(self):
        """
        Pack the state of the environment into a single array

        Args: 
            None
        Returns:
            state (array): A 12x1 array of the state of the environment
        """
        
        for key in self.state_dict:
            state = np.vstack(self.state_dict[key])




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



def quat_mult(q, r):
    """
    Quaternion multiplication.
    Both q and r are assumed to be 4-element arrays in [q0, q1, q2, q3] (q0 = scalar).
    """
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return np.array([
        q0*r0 - q1*r1 - q2*r2 - q3*r3,
        q0*r1 + q1*r0 + q2*r3 - q3*r2,
        q0*r2 - q1*r3 + q2*r0 + q3*r1,
        q0*r3 + q1*r2 - q2*r1 + q3*r0
    ])

def quat_conjugate(q):
    """
    Quaternion conjugate.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_to_rot_matrix(q):
    """
    Convert a unit quaternion [q0,q1,q2,q3] to a 3x3 rotation matrix.
    """
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),       2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),         1 - 2*(q1**2 + q3**2),   2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),         2*(q2*q3 + q0*q1),       1 - 2*(q1**2 + q2**2)]
    ])

def rotate_vector(q, v):
    """
    Rotate a 3-vector v by the unit quaternion q.
    """
    q_v = np.concatenate(([0], v))
    q_conj = quat_conjugate(q)
    qv_rot = quat_mult(quat_mult(q, q_v), q_conj)
    return qv_rot[1:]



from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Function to create a 3D circle
def create_circle(radius=1, center=(0, 0, 0), num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.zeros_like(x) + center[2]
    return np.vstack((x, y, z)).T

# Function to create a 3D rectangular prism
def create_rectangular_prism(center, length, width, height):
    cx, cy, cz = center  # Center coordinates
    dx = length / 2
    dy = width / 2
    dz = height / 2

    # Vertices of the rectangular prism
    vertices = np.array([
        [cx - dx, cy - dy, cz - dz],  # Bottom face
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],  # Top face
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ])
    return vertices

# Function to create a trapezoidal prism
def create_trapezoidal_prism(center, top_length, top_width, bottom_length, bottom_width, height):
    cx, cy, cz = center  # Center coordinates
    # Half dimensions for the top and bottom faces
    dx_top = top_length / 2
    dy_top = top_width / 2
    dx_bottom = bottom_length / 2
    dy_bottom = bottom_width / 2
    dz = height / 2

    # Top face vertices
    top_corners = np.array([
        [cx - dx_top, cy - dy_top, cz + dz],
        [cx + dx_top, cy - dy_top, cz + dz],
        [cx + dx_top, cy + dy_top, cz + dz],
        [cx - dx_top, cy + dy_top, cz + dz],
    ])

    # Bottom face vertices
    bottom_corners = np.array([
        [cx - dx_bottom, cy - dy_bottom, cz - dz],
        [cx + dx_bottom, cy - dy_bottom, cz - dz],
        [cx + dx_bottom, cy + dy_bottom, cz - dz],
        [cx - dx_bottom, cy + dy_bottom, cz - dz],
    ])

    # Combine all vertices into a single array
    vertices = np.vstack((top_corners, bottom_corners))
    return vertices

# Function to construct faces from vertices
def construct_faces_from_vertices(vertices):
    # Top face: vertices 0, 1, 2, 3
    # Bottom face: vertices 4, 5, 6, 7
    # Side faces connect top and bottom vertices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Top face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Bottom face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # Left face
    ]
    return faces

# Plotting function
def plot_prism(ax, faces, face_color='blue', edge_color='black'):
    # Add the faces to the plot
    for face in faces:
        poly = Poly3DCollection([face], color=face_color, edgecolor=edge_color, linewidths=1, alpha=0.7)
        ax.add_collection3d(poly)

def plot_sphere(ax, center=(0, 0, 0), radius=1, num_points=100, face_color='blue', edge_color='black', alpha=0.7):
    u, v = np.mgrid[0:2*np.pi:num_points*1j, 0:np.pi:num_points//2*1j]

    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=face_color, edgecolor=edge_color, alpha=alpha)



# Euler rotation function that follows the 123 convention
def euler_rotation_123(position, phi, theta, psi):
    R_yaw = np.array([
                [np.cos(psi), np.sin(psi), 0],
                [-np.sin(psi), np.cos(psi), 0],
                [0, 0, 1]
            ])
    R_pitch = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)]
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(phi), np.sin(phi)],
        [0, -np.sin(phi), np.cos(phi)]
    ])
    return R_yaw @ R_pitch @ R_roll @ position









class QuadPole(Env):
    def __init__(
            self,
            env_name = 'QuadPole',
            max_steps = 500,):
        
        self.env_name = env_name
        self.max_steps = max_steps

        self.mass = 1.5             # Quadrotor mass
        self.load_mass = 0.5        # Payload mass
        self.gravity = 9.80665      # Gravitational acceleration
        self.tether_length = 0.5    # Rigid tether (cable) length

        self.Ixx = 4e-1
        self.Iyy = 4e-1
        self.Izz = 2.5e-1

        self.torque_constant = 0.1
        self.arm_length = 0.5

        self.timestep = 0.02

        self.hover_force = (self.mass + self.load_mass)*self.gravity/4


        self.spatial_bounds = ((-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5))
        self._xbounds = self.spatial_bounds[0]
        self._ybounds = self.spatial_bounds[1]
        self._zbounds = self.spatial_bounds[2]

        self.state_dict = {
            'quadrotor': np.zeros(13),
            'pendulum': np.zeros(7)
        }

        self._is_3d = True
        self.detailed_rendering = False


        # Define the observation space.
        # The state consists of:
        #   Quadrotor state: 3 (position) + 3 (velocity) + 4 (quaternion) + 3 (angular velocity) = 13
        #   Payload state: 4 (quaternion) + 3 (angular velocity) = 7
        # Total state dimension = 20.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        # Define the action space.
        # The control input consists of 4 rotor thrusts.
        # Here we assume each rotor’s thrust is bounded between 0 and 20 (you may adjust these limits).
        self.action_space = gym.spaces.Box(
            low=0.0, high=20.0, shape=(4,), dtype=np.float32
        )

    def _wrap_action(self, action):
        """
        Wraps the action to ensure it is within the valid range, centered around hover thrust.
        """
        return self.hover_force + self.hover_force*np.clip(action, -1, 1)

       

    def _dynamics(self, state, control):
        """
        Dynamics for a quadrotor with a suspended load using quaternion representations.
        Uses Euler integration with a fixed timestep.
        
        Args:
            state (array): A 20x1 state vector containing:
            - Quadrotor position (3)
            - Quadrotor velocity (3)
            - Quadrotor orientation quaternion (4) [q0, q1, q2, q3]
            - Quadrotor body angular velocity (3) [p, q, r]
            - Payload orientation quaternion (4) (rotating a reference tether vector to current tether direction)
            - Payload angular velocity (3)
            control (array): A 4x1 array of control inputs [u1, u2, u3, u4]
            
        Returns:
            next_state (array): A 20x1 state vector at the next timestep.
        """
        # Unpack state
        pos = state[0:3]           # Quadrotor position: [x, y, z]
        vel = state[3:6]           # Quadrotor velocity: [xdot, ydot, zdot]
        q = state[6:10]            # Quadrotor quaternion (scalar-first)
        omega = state[10:13]       # Quadrotor angular velocity [p, q, r]
        q_p = state[13:17]         # Payload quaternion (defines tether direction)
        omega_p = state[17:20]     # Payload angular velocity

        # Unpack control inputs
        u1, u2, u3, u4 = control
        u_total = u1 + u2 + u3 + u4

        # Parameters
        m0 = self.mass                 # Quadrotor mass
        m_p = self.load_mass           # Payload mass
        L = self.tether_length         # Rigid tether length
        arm_length = self.arm_length   # Distance from center to rotor
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        torque_constant = self.torque_constant
        gravity = self.gravity         # Gravitational acceleration (positive scalar)
        timestep = self.timestep

        # Gravity vector (assuming downward is negative z)
        g_vec = np.array([0, 0, -gravity])

        # Quadrotor rotation matrix from its quaternion
        R = quat_to_rot_matrix(q)
        
        # Compute thrust force in inertial frame
        F_thrust = R @ np.array([0, 0, u_total])
        
        # Determine the tether’s unit direction vector.
        # Here we define the reference tether vector as [0,0,-1] (i.e. payload hanging down)
        u_tether = rotate_vector(q_p, np.array([0, 0, -1]))
        
        # Compute the time derivative of the tether direction (from payload angular velocity)
        u_dot = np.cross(omega_p, u_tether)
        
        # --- Coupled Dynamics: Tension Computation ---
        T = m_p/(m0 + m_p)*(np.dot(F_thrust, u_tether) - m0*L*np.linalg.norm(u_dot)**2)

        # --- Quadrotor Translational Dynamics ---
        # m0 * ddot{p} = m0*g_vec + F_thrust - T*u_tether
        acc = (1/m0) * ( m0 * g_vec + F_thrust - T * u_tether )

        # Euler integration for position and velocity
        vel_next = vel + acc * timestep
        pos_next = pos + vel_next * timestep
        
        # Compute control torques (using a typical mixing law)
        tau_x = (np.sqrt(2)/2*(u1 + u3 - u2 - u4)*arm_length - (Izz - Iyy)*omega[1]*omega[2])
        tau_y = (np.sqrt(2)/2*(u3 + u4 - u1 - u2)*arm_length - (Izz - Ixx)*omega[0]*omega[2])
        tau_z = (torque_constant*(u1 + u4 - u2 - u3))
        tau = np.array([tau_x, tau_y, tau_z])
        
        # Angular acceleration in body frame: ω̇ = J⁻¹ (τ - ω×(Jω))
        J_omega = np.array([Ixx*omega[0], Iyy*omega[1], Izz*omega[2]])
        cross_term = np.cross(omega, J_omega)
        omega_dot = np.array([
            (tau[0] - cross_term[0]) / Ixx,
            (tau[1] - cross_term[1]) / Iyy,
            (tau[2] - cross_term[2]) / Izz
        ])
        omega_next = omega + omega_dot * timestep

        # --- Quadrotor Rotational Dynamics ---
        # Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, omega]
        q_dot = 0.5 * quat_mult(q, np.concatenate(([0], omega_next)))
        q_next = q + q_dot * timestep
        q_next = q_next / np.linalg.norm(q_next)  # Normalize to avoid drift

        
        
        # Payload angular acceleration:
        omega_p_dot = np.cross(L*u_tether, T*u_tether + g_vec*m_p) / (self.load_mass * L**2)
        omega_p_next = omega_p + omega_p_dot * timestep

        # --- Payload (Pendulum) Dynamics ---
        q_p_dot = 0.5 * quat_mult(np.concatenate(([0], omega_p_next)), q_p) # Flipped because 
        q_p_next = q_p + q_p_dot * timestep
        q_p_next = q_p_next / np.linalg.norm(q_p_next)

        # Hemisphere constraint: Ensure the payload quaternion is in the upper hemisphere
        # if q_p_next[0] < 0:
        #     q_p_next = -q_p_next
        # if q_next[0] < 0:
        #     q_next = -q_next

        # Pack next state (order: quad pos, quad vel, quad quaternion, quad omega, payload quaternion, payload omega)
        next_state = np.hstack((pos_next, vel_next, q_next, omega_next, q_p_next, omega_p_next))

        return next_state
        
    def reset(self):
        """
        Resets the state of the system.
        Quadrotor state: [x, y, z, xdot, ydot, zdot, q0, q1, q2, q3, p, q, r]
        Payload state: [q0, q1, q2, q3, omega_x, omega_y, omega_z]
        
        For the payload, two angles (alpha and beta) are sampled uniformly and used to create a
        quaternion that perturbs the nominal tether direction ([0,0,-1]) away from vertical.
        """
        # Sample random angles for payload initial deviation
        # alpha = np.random.uniform(-np.pi, np.pi)
        # beta = np.random.uniform(-np.pi, np.pi)

        alpha = np.random.uniform(-1.0, 1.0)
        beta = np.random.uniform(-1.0, 1.0)

        # alpha = 0.5
        # beta = 0
        
        # Quadrotor: initialize at origin with zero velocity, identity quaternion, and zero angular velocity.
        self.state_dict['quadrotor'] = np.array([
            0, 0, 0,        # position: x, y, z
            0, 0, 0,        # velocity: xdot, ydot, zdot
            1, 0, 0, 0,     # orientation quaternion (identity)
            0, 0, 0         # body angular velocity: p, q, r
        ])
        
        # Convert (alpha, beta) into a payload quaternion.
        # For instance, interpret alpha as a rotation about the x-axis and beta about the y-axis.
        q_x = np.array([np.cos(alpha/2), np.sin(alpha/2), 0, 0])
        q_y = np.array([np.cos(beta/2), 0, np.sin(beta/2), 0])
        q_p = quat_mult(q_y, q_x)
        q_p = q_p / np.linalg.norm(q_p)  # Ensure unit norm
        
        # Payload: initialize with the computed quaternion and zero angular velocity.
        self.state_dict['pendulum'] = np.hstack((q_p, np.array([0, 0, 0])))
        
        # Save the initial state for restarting later
        self._initial_state = {key: self.state_dict[key].copy() for key in self.state_dict}
        
        self._steps = 0
        self._time = 0
        self._time_balanced = 0

        # breakpoint()
    
        return self._get_obs(), self._get_info()

    def _propegate(self, action):
        """
        Propagates the state of the system using the dynamics function.
        """
        state = np.hstack((self.state_dict['quadrotor'], self.state_dict['pendulum']))
        state = self._dynamics(state, action)
        self.state_dict['quadrotor'] = state[:13]
        self.state_dict['pendulum'] = state[13:]

    def restart(self):
        """
        Resets the state to the initial state (as stored at the last reset).
        """
        self.state_dict = {key: self._initial_state[key].copy() for key in self._initial_state}
        
        self._steps = 0
        self._time = 0
        self._time_balanced = 0
        
        return self._get_obs(), self._get_info()


    def _get_obs(self):
        """
        Returns the full state observation by concatenating the quadrotor and payload states.
        """
        return np.hstack((self.state_dict['quadrotor'], self.state_dict['pendulum']))


    def _get_info(self):
        """
        Returns an info dictionary (for instance, tracking balanced time).
        """
        return {'time_balanced': self._time_balanced}


    def _out_of_bounds(self):
        """
        Checks whether the quadrotor's position is outside preset bounds.
        Note: Only the position (first 3 states) is checked.
        """
        x, y, z = self.state_dict['quadrotor'][0:3]
        return (x < self._xbounds[0] or x > self._xbounds[1] or
                y < self._ybounds[0] or y > self._ybounds[1] or
                z < self._zbounds[0] or z > self._zbounds[1])


    def step(self, action):
        """
        Advances the system one timestep using the provided action.
        
        Args:
            action (array): Control inputs for the quadrotor.
        
        Returns:
            observation (array): The new full state.
            reward (float): The timestep-scaled reward.
            terminated (bool): Whether the episode terminated (always False here).
            truncated (bool): Whether the episode was truncated (e.g. out-of-bounds or max steps reached).
            info (dict): Additional information (e.g., time balanced).
        """
        reward = 0

        # action = np.zeros(4)
        # action = np.array([-0.01, -0.01, -0.01, -0.01])

        action = self._wrap_action(action)
        
        # Propagate the state using the dynamics function (which uses quaternions)
        self._propegate(action)
        
        # Increment time and step count
        self._steps += 1
        self._time += self.timestep

        observation = self._get_obs()
        info = self._get_info()
        
        # Extract quadrotor states
        quad = self.state_dict['quadrotor']
        pos = quad[0:3]
        vel = quad[3:6]
        quat = quad[6:10]
        omega = quad[10:13]
        
        # Extract payload states
        pend = self.state_dict['pendulum']
        q_p = pend[0:4]
        omega_p = pend[4:7]
        
        # Compute a measure of deviation for the orientations.
        theta_quad = 1 - np.abs(np.dot(quat, np.array([1, 0, 0, 0])))  # Reference: upright quad
        theta_payload = 1 - np.abs(np.dot(q_p, np.array([1, 0, 0, 0])))  # Reference: downward payload
        # theta_payload = 1 - np.abs(np.dot(q_p, np.array([0, 0, 1, 0]))) # Reference upward payload

        # Compute cost terms (using similar weights as before)
        cost_pos = (pos[0]**2 + pos[1]**2 + pos[2]**2)
        cost_vel = np.sum(vel**2)
        cost_quad_orient = (theta_quad**2)
        cost_quad_rate = np.sum(omega**2)
        cost_payload_orient = (theta_payload**2)
        cost_payload_rate = np.sum(omega_p**2)

        # reward += self.timestep * np.sum([
        #     1,                # Reward for staying alive
        # - cost_pos,        # Penalty for deviation from origin
        # - cost_vel,        # Penalty for high velocity
        # - cost_quad_orient,  # Penalty for quadrotor orientation error
        # - cost_quad_rate,    # Penalty for high angular velocity
        # - cost_payload_orient,  # Penalty for payload orientation deviation
        # - cost_payload_rate     # Penalty for payload angular speed
        # ])

        reward += self.timestep * np.sum([
            1,                # Reward for staying alive
            5/(1 + 10*cost_pos),        # reward for staying near origin
            10/(1 + 10*cost_vel),        # reward for low velocity
            0.1/(1 + cost_quad_orient),  # reward for quadrotor orientation error
            5.0/(1 + cost_quad_rate),    # reward for low angular velocity
            10/(1 + 10*cost_payload_orient),  # reward for payload orientation deviation
            1/(1 + 10*cost_payload_rate)     # reward for low payload angular speed
        ])
        
        # breakpoint()

        oob = self._out_of_bounds()

        if oob:
            reward -= 10_000*self.timestep

        

        truncated = self._steps >= self.max_steps or oob
        terminated = False

        return observation, reward, terminated, truncated, info

    def render(self, ax, observation=None, color='black', alpha=1.0):
        """
        Render the detailed 3D state of the quadrotor with a suspended payload.

        This function draws the quadrotor body as a trapezoidal prism, the arms as lines
        with rotor circles at their ends, and a dashed tether connecting the quadrotor to the
        payload. The quadrotor is oriented using its quaternion state, and the payload's
        position is determined by the pendulum's quaternion.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): A 3D matplotlib axis.
            observation (array, optional): The full state observation.
                If None, uses the internal self.state_dict.
            color (str): Color for drawing the vehicle, arms, rotors, tether, and payload.
            alpha (float): Transparency for drawing.
        """

        detailed = self.detailed_rendering

        # --- State Extraction ---
        if observation is None:
            quad_state = self.state_dict['quadrotor']  # [x, y, z, xdot, ydot, zdot, q0, q1, q2, q3, p, q, r]
            pend_state = self.state_dict['pendulum']    # [q0, q1, q2, q3, omega_x, omega_y, omega_z]
        else:
            quad_state = observation[:13]
            pend_state = observation[13:]
        
        # --- Quadrotor Body Rendering ---
        # Extract position and orientation
        pos = np.array(quad_state[0:3])
        q = quad_state[6:10]
        # Compute the rotation matrix from the quaternion (body-to-inertial)
        R = quat_to_rot_matrix(q)  # Assumed to return a 3x3 rotation matrix

        if detailed:
            # Define body geometry as a trapezoidal prism
            top_length = 0.4
            top_width = 0.15
            bottom_length = 0.25
            bottom_width = 0.1
            body_height = 0.1
            # Create body vertices centered at the origin
            body_vertices = create_trapezoidal_prism([0, 0, 0],
                                                    top_length=top_length,
                                                    top_width=top_width,
                                                    bottom_length=bottom_length,
                                                    bottom_width=bottom_width,
                                                    height=body_height)
            # Rotate and translate the body vertices to the quadrotor's pose
            rotated_vertices = (R @ body_vertices.T).T + pos
            # Construct the faces from the rotated vertices
            body_faces = construct_faces_from_vertices(rotated_vertices)
            # Draw the body prism
            plot_prism(ax, body_faces, face_color=color, edge_color=color)

        # --- Quadrotor Arms and Rotor Circles ---
        # Use arm_length (defined in self) to set the scale
        l = self.arm_length
        # Define arm endpoints in the body frame (each column is one rotor offset)
        arms_body = np.array([
            [4, -4,  4, -4],
            [4,  4, -4, -4],
            [1.5, 1.5, 1.5, 1.5]
        ]) * l / np.linalg.norm([4, 4, 1.5])
        # Rotate arms into the inertial frame
        arms_inertial = R @ arms_body
        
        # Define rotor parameters
        rotor_radius = 0.2

        if detailed:
            rotor_circle = create_circle(radius=rotor_radius)
        
        # For each rotor arm, draw the arm and rotor
        for i in range(arms_inertial.shape[1]):
            # Compute rotor center position in the inertial frame
            rotor_center = pos + arms_inertial[:, i]
            # Plot the arm as a line from the body center to the rotor center
            ax.plot([pos[0], rotor_center[0]],
                    [pos[1], rotor_center[1]],
                    [pos[2], rotor_center[2]],
                    color=color, lw=3, alpha=alpha)
            if detailed:
                # Rotate the rotor circle and translate it to the rotor center
                rotor_circle_rotated = (R @ rotor_circle.T).T + rotor_center
                rotor_collection = Poly3DCollection([rotor_circle_rotated],
                                                    color=color, edgecolor=color, linewidths=3, alpha=alpha)
                ax.add_collection3d(rotor_collection)
        
        # --- Payload (Suspended Load) Rendering ---
        # Extract the payload (pendulum) quaternion which determines the tether direction
        q_p = pend_state[0:4]
        # Rotate the reference vector [0, 0, -1] to obtain the tether's unit vector
        u_tether = rotate_vector(q_p, np.array([0, 0, -1]))
        # Compute payload position by extending along the tether direction
        pos_payload = pos + self.tether_length * u_tether
        
        # Draw the tether as a dashed line connecting the quadrotor and payload
        ax.plot([pos[0], pos_payload[0]],
                [pos[1], pos_payload[1]],
                [pos[2], pos_payload[2]],
                color=color, lw=1.5, alpha=alpha)
        # Render the payload as a scatter point (or you could use a sphere via plot_sphere)
        ax.scatter([pos_payload[0]], [pos_payload[1]], [pos_payload[2]],
                color=color, s=25, zorder=3, alpha=alpha)
        

        

        # --- Axis and Aesthetic Formatting ---
        # Set plot limits based on environment bounds if available, else use default limits
        if hasattr(self, '_xbounds'):
            ax.set_xlim(self._xbounds)
        else:
            ax.set_xlim([-2, 2])
        if hasattr(self, '_ybounds'):
            ax.set_ylim(self._ybounds)
        else:
            ax.set_ylim([-2, 2])
        if hasattr(self, '_zbounds'):
            ax.set_zlim(self._zbounds)
        else:
            ax.set_zlim([-2, 2])


        # Plot shadow of payload on the ground
        z = ax.get_zlim()[0]
        shadow_radius = 0.08
        pos_shadow = np.array([pos_payload[0], pos_payload[1], z])
        shadow = create_circle(radius=shadow_radius, center=pos_shadow)
        shadow_collection = Poly3DCollection([shadow], color=color, alpha=alpha/2)
        ax.add_collection3d(shadow_collection)
        
        # Remove tick labels for a cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Hide the background panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Set equal aspect ratio if supported
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass


            

class QuadPole2D(Env):
    def __init__(
            self,
            env_name = 'QuadPole2D',
            max_steps = 500,
            timestep = 0.02):
        
        # Quadrotor parameters
        self.mq = 1.5             # Quadrotor mass                 (kg)
        self.mp = 0.5             # Payload mass                   (kg)
        self.I   = 4e-1            # Quadrotor moment of inertia   (kg-m²)
        self.Lq = 0.5             # Quadrotor arm length           (m)
        self.Lp = 0.75            # Rigid tether length            (m)

        # Simulation parameters
        self.gravity = 9.80665     # Gravitational acceleration    (m/s²)
        self.timestep = timestep   # Simulation timestep           (s)
        self.max_steps = max_steps # Maximum number of steps

        # Environment Parameters
        self.spatial_bounds = ((-2.0, 2.0), (-2.0, 2.0)) # Spatial bounds (x, y)
        self.balance_radius = 0.25                       # Radius around origin for considering the system balanced (m)
        self.env_name = env_name                         # Environment name
        self._is_3d = False                              # 2D environment
        self._xbounds = self.spatial_bounds[0]
        self._zbounds = self.spatial_bounds[1]

        # Hover force per rotor
        self.hover_force = (self.mq + self.mp)*self.gravity/2

        # Initial state dictionary
        self.state_dict = {
            'quadrotor': np.zeros(8),
            'pendulum': np.zeros(4)
        }

        # OpenAI Gym API attributes (not really used here)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=0.0, high=20.0, shape=(2,), dtype=np.float32
        )

    def _wrap_action(self, action):
        """
        Wrap the input action by computing an adjusted force based on the hover force.
        This function clips the provided action to ensure it remains within the range [-1, 1],
        then scales the clipped value by the hover_force, and finally adds it to the hover_force.
        This effectively produces a new action that is a deviation from the hover force baseline,
        bounded appropriately by the clip operation.
        Parameters:
            action (float or array-like): The input action(s) to be modulated. Expected to be within a range that,
                                          when scaled by hover_force, provides a meaningful deviation from the nominal hover force.
        Returns:
            float or array-like: The adjusted force computed as hover_force + (hover_force * clipped_action).
        Note:
            The function assumes that hover_force is defined and properly represents the baseline force required
            to hover. The clipping ensures that the adjustment does not exceed the predefined safe limits.
        """
        return self.hover_force + self.hover_force*np.clip(action, -1, 1)

    def reset(self):
        """
        Resets the environment to its initial state.
        This method resets the internal state variables and counters of the environment, including the step counter,
        time, and balanced time. The quadrotor and pendulum states are initialized:
            - Quadrotor: Set to a fixed state [0, 0, 0, 0, 0, 1, 0].
            - Pendulum: Set based on a randomly sampled angle phi in [-π, π], where the x-component is sin(phi)
                        and the y-component is cos(phi).
        The initial state of the environment is stored separately for potential future reference.
        Returns:
            tuple:
                A tuple containing:
                    - The observation state retrieved from the _get_obs() method.
                    - An info dictionary retrieved from the _get_info() method.
        """
        # Reset the counter variables
        self._steps = 0
        self._time = 0
        self._time_balanced = 0

        # Sample a random angle phi for the pendulum
        phi = np.random.uniform(-np.pi, np.pi)

        # Set the initial state of the quadrotor and pendulum
        self.state_dict['quadrotor'] = np.array([0, 0, 0, 0, 0, 1, 0])
        self.state_dict['pendulum'] = np.array([np.sin(phi), np.cos(phi), 0])

        # Save the initial state for potential restarts
        self._initial_state = self.state_dict.copy()

        # Return the initial observation and info
        return self._get_obs(), self._get_info()
    
    def restart(self):
        """
        Resets the environment to its initial state.
        This method reinitializes the state dictionary, step counter, simulation time, 
        and time balanced to their starting values. It then returns the current observation 
        and additional info by calling the _get_obs() and _get_info() methods respectively.
        Returns:
            tuple: A tuple containing the observation and ancillary information.
        """
        # Reset the state dictionary to the initial state
        self.state_dict = self._initial_state.copy()

        # Reset the counter variables
        self._steps = 0
        self._time = 0
        self._time_balanced = 0

        # Return the initial observation and info
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        """
        Retrieve the combined observation from the quadrotor and pendulum states.
        This method concatenates the state arrays for the quadrotor and pendulum into a single
        observation vector using numpy's hstack function. The resulting observation can be used
        as the input for further processing or control within the environment.
        Returns:
            np.ndarray: A one-dimensional array combining the quadrotor state followed by the 
            pendulum state.
        Note:
            The function assumes that self.state_dict contains valid 'quadrotor' and 'pendulum'
            keys, each associated with a numpy array representing the respective state.
        """
        return np.hstack((self.state_dict['quadrotor'], self.state_dict['pendulum']))
    
    def _get_info(self):
        """
        Return a dictionary containing metrics for the quadrotor environment.
        Returns:
            dict: A dictionary with the following key-value pair:
                "time_balanced" (bool): Indicates whether the environment has achieved balanced time.
        Function Note:
            This method is used to encapsulate and return relevant state metrics for time balance in the simulation.
        """
        return {'time_balanced': self._time_balanced}
    
    def out_of_bounds(self):
        """
        Check if the quadrotor is out of the defined bounds.
        This method retrieves the x and z positions from the quadrotor's state and
        compares them against the preset x bounds (self._xbounds) and z bounds 
        (self._zbounds). It returns True if either the x position is outside the 
        x bounds or the z position is outside the z bounds, indicating that the 
        quadrotor is out of the permitted area.
        Returns:
            bool: True if the quadrotor is out-of-bounds, False otherwise.
        """
        x, z = self.state_dict['quadrotor'][0:2]
        return (x < self._xbounds[0] or x > self._xbounds[1] or
                z < self._zbounds[0] or z > self._zbounds[1])
    
    def _propogate(self, action):
        """
        Propagates the system state by integrating the dynamics for a single time step.
        This method concatenates the current states of the quadrotor and pendulum into a single state vector,
        applies the dynamics function (_dynamics) using the provided action, and then updates the internal state dictionary
        by splitting the resulting state back into the quadrotor and pendulum components.
        Parameters:
            action (np.ndarray): The action to apply during propagation. The expected structure and type of the action
                                 depend on the specific dynamics model.
        Returns:
            None
        Note:
            This method updates the internal state in-place and does not return a new state.
        """

        state = np.hstack((self.state_dict['quadrotor'], self.state_dict['pendulum']))
        state = self._dynamics(state, action)
        self.state_dict['quadrotor'] = state[:8]
        self.state_dict['pendulum'] = state[8:]
    
    def _dynamics(self, state, control):
        """
        Perform one integration step for the quadrotor-payload dynamics using semi-implicit Euler integration.
        This method integrates the state of a coupled quadrotor-payload system. The input state vector is expected
        to be in the form:
            x, z            : Cartesian positions,
            vx, vz          : Linear velocities,
            s_theta, c_theta: Sin and cos of the quadrotor's pitch angle (theta),
            theta_dot       : Angular velocity of the quadrotor,
            s_phi, c_phi    : Sin and cos of the payload's angle (phi),
            phi_dot         : Angular velocity of the payload.
        Parameters:
            state (list or array-like): The current state vector.
            control (list or array-like): Control inputs [u1, u2], corresponding to the forces from the two rotors.
        Returns:
            list: The updated state vector after one integration step, containing:
                  [x_new, z_new, vx_new, vz_new, s_theta_new, c_theta_new, theta_dot_new,
                   s_phi_new, c_phi_new, phi_dot_new].
        Notes:
            - The computation includes dynamics for both the translational and rotational motion of the quadrotor,
              as well as the motion of the payload modeled as a pendulum.
            - The formulation considers the effect of the combined forces and moments, including gravitational
              acceleration and coupling between the translational and angular accelerations.
            - Semi-implicit Euler integration is used: velocities are updated first based on computed accelerations,
              and then positions are updated using these new velocities.
        """
        # Unpack state variables
        x, z, vx, vz, s_theta, c_theta, theta_dot, s_phi, c_phi, phi_dot = state
        # Unpack control inputs
        u1, u2 = control

        # Parameters
        mq = self.mq             # Quadrotor mass
        mp = self.mp             # Payload mass
        Lq = self.Lq             # Distance from center to rotor
        Lp = self.Lp             # Tether length
        I = self.I               # Quadrotor moment of inertia (2D scalar)
        g = self.gravity         # Gravitational acceleration (positive scalar)
        dt = self.timestep       # Timestep
        
        # Total force from both rotors
        F = u2 + u1
        # Combined mass of quadrotor and pendulum
        M = mq + mp

        # 1. Quadrotor attitude dynamics (theta)
        ddtheta = (Lq / I) * (u2 - u1)
        
        # 2. Pendulum dynamics (phi)
        # Derived from the coupling of translational accelerations with the pendulum equation.
        ddphi = -F * (s_phi * c_theta - s_theta * c_phi) / (mq * Lp)
        
        # 3. Translational dynamics:
        # Equation for x: M*ddx + mp*Lp*( -sin(phi)*phi_dot^2 + cos(phi)*ddphi ) = -sin(theta)*F
        ddx = (-s_theta * F - mp * Lp * c_phi * ddphi + mp * Lp * s_phi * (phi_dot**2)) / M
        
        # Equation for z: M*ddz + mp*Lp*( cos(phi)*phi_dot^2 + sin(phi)*ddphi ) = cos(theta)*F - M*g
        ddz = (c_theta * F - M * g - mp * Lp * s_phi * ddphi - mp * Lp * c_phi * (phi_dot**2)) / M

        # 4. Semi-implicit Euler update:
        # First update the velocities using the computed accelerations.
        vx_new       = vx + ddx * dt
        vz_new       = vz + ddz * dt
        theta_dot_new = theta_dot + ddtheta * dt
        phi_dot_new   = phi_dot + ddphi * dt
        
        # Then update the positions with the new velocities.
        x_new = x + vx_new * dt
        z_new = z + vz_new * dt
        
        # Update sin and cos for theta using chain rule:
        # d/dt(s_theta) = c_theta * theta_dot, and d/dt(c_theta) = -s_theta * theta_dot.
        theta = np.arctan2(s_theta, c_theta)
        s_theta_new = np.sin(theta + theta_dot * dt)
        c_theta_new = np.cos(theta + theta_dot * dt)

        # Update sin and cos for phi using chain rule:
        # d/dt(s_phi) = c_phi * phi_dot, and d/dt(c_phi) = -s_phi * phi_dot.
        phi = np.arctan2(s_phi, c_phi)
        s_phi_new = np.sin(phi + phi_dot * dt)
        c_phi_new = np.cos(phi + phi_dot * dt)

        # Pack the new state vector and return
        new_state = [x_new, z_new, vx_new, vz_new, s_theta_new, c_theta_new, theta_dot_new,
                    s_phi_new, c_phi_new, phi_dot_new]
        
        return new_state

    def step(self, action):
        """
        Take a simulation step in the quadrotor environment.
        This method processes the provided action by wrapping it, propagating the state,
        and computing a reward based on various penalty terms, including position, velocity,
        orientation, and angular velocities for both the quadrotor and payload. It also
        provides a bonus reward if a balanced state is achieved and penalizes heavily for
        out-of-bounds conditions.
        Steps:
            1. Wrap the action via _wrap_action and propagate it via _propogate.
            2. Obtain the updated state, observation, and additional environment info.
            3. Compute individual cost terms:
                  - pos_cost: A combination of absolute deviation and squared deviation from the origin.
                  - vel_cost: Squared velocity penalty for the x and z components.
                  - theta_cost: Deviation penalty using the cosine of the quadrotor's orientation angle.
                  - omega_cost: Penalty for high angular velocity of the quadrotor.
                  - phi_cost: Cubic cost term for the payload orientation.
                  - phi_dot_cost: Squared cost term for the payload angular velocity.
            4. Combine these penalties with respective weights scaled by the timestep to calculate
               the overall reward.
            5. Apply a bonus reward if the vehicle is within a set balance radius, the payload
               orientation is near its target, and the payload's angular velocity is low.
            6. Apply a heavy penalty if the state is determined to be out-of-bounds.
            7. Increment the step count and simulation time.
            8. Determine if the simulation should be truncated (when the maximum number of steps is
               reached or out-of-bounds) while termination is deliberately kept False.
        Args:
            action: The action applied at the current time step. It is first wrapped to conform
                    to the expected action space.
        Returns:
            tuple: A tuple containing:
                - state: The updated state of the environment after applying the action.
                - reward: The computed reward for the step.
                - terminated: A boolean flag indicating episode termination (always False here).
                - truncated: A boolean flag indicating if the episode was truncated (True if the maximum
                             number of steps is reached or the state is out-of-bounds).
                - info: Additional information provided by the environment.
        Note:
            The reward strategy incorporates multiple penalties to ensure the quadrotor and its payload
            maintain desired states, penalizing deviations and rewarding balance. The heavy out-of-bounds
            penalty enforces safe operation within the defined limits.
        """
        
        # Wrap the action
        action = self._wrap_action(action)

        # Propagate the state using the dynamics function
        self._propogate(action)

        # Obtain the updated state, observation, and additional info
        state = self._get_obs()
        info = self._get_info()

        # Compute cost terms
        pos_cost = np.sum(np.abs(state[0:2])) + np.sum((state[0:2])**2)  # Position cost: L1 and L2 norms
        vel_cost = np.sum(state[2:4]**2)                                 # Velocity cost: L2 norm
        theta_cost = 1 - np.abs(state[5])                                # Quadrotor orientation cost (cosine of theta): 1 - cos(theta)
        omega_cost = state[6]**2                                         # Quadrotor angular velocity cost: L2 norm
        phi_cost = state[8]**3                                           # Payload orientation cost: L3 norm
        phi_dot_cost = state[9]**2                                       # Payload angular velocity cost: L2 norm

        # Compute the reward using the timestep-scaled cost terms
        reward = 0
        reward += self.timestep * np.sum([
            - 15.0*pos_cost,       
            - 0.5*vel_cost,       
            - theta_cost,    
            - 5*omega_cost,
            - (25.0*phi_cost - 25.0)*(1/(1 + 2*phi_dot_cost)) # Balancing reward
        ])

        # Apply a bonus reward if the quadrotor is balanced
        if np.sum(state[0:2]**2)**0.5 < self.balance_radius and state[8] < -0.95 and abs(state[9]) < 0.1:
            reward += 100*self.timestep
            self._time_balanced += self.timestep
        else:
            self._time_balanced = 0

        # Increment the step count and simulation time
        self._steps += 1
        self._time += self.timestep

        # Apply heavy penalty if out-of-bounds
        oob = self.out_of_bounds()
        if oob:
            reward -= 20_000 * self.timestep

        # Determine if the episode should be truncated
        truncated = self._steps >= self.max_steps or oob
        terminated = False

        return state, reward, terminated, truncated, info

    def render(self, ax, observation=None, color='black', alpha=1.0):
        """
        Renders the quadrotor and its suspended payload on the given matplotlib axis.
        It draws the quadrotor body, its arms with rotors, and the tethered payload.
        Note:
            - The state vector is expected to be of length 10 with the following elements:
              [x, z, vx, vz, s_theta, c_theta, theta_dot, s_phi, c_phi, phi_dot]
            - If an observation is provided, it is used as the state vector; otherwise, `self.state`
              is assumed to hold the current state.
            - The appearance of the render (color and transparency) can be adjusted via the `color`
              and `alpha` arguments.
        Parameters:
            ax (matplotlib.axes.Axes): The axis object on which to render the quadrotor.
            observation (array-like, optional): The state vector to be rendered. If not provided,
                                                  `self.state` is used.
            color (str or tuple, optional): The color used for drawing the quadrotor, arms, rotors,
                                            tether, and payload. Default is 'black'.
            alpha (float, optional): The transparency (opacity) level for the rendered elements.
                                     Default is 1.0.
        Returns:
            None
        """
        # --- State Extraction ---
        # Assume self.state holds the current 2D state if observation is not provided.
        if observation is None:
            state = self.state
        else:
            state = observation

        # Unpack the state vector
        x, z, vx, vz, s_theta, c_theta, theta_dot, s_phi, c_phi, phi_dot = state
        pos = np.array([x, z])

        ax.axhline(0, color=(0, 0, 0, 0.3), lw=1, linestyle='--')
        ax.axvline(0, color=(0, 0, 0, 0.3), lw=1, linestyle='--') 

        # Draw circle around origin
        radius = 0.25  # Radius of the circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        ax.plot(circle_x, circle_y, color=(0, 0, 0, 0.3), lw=1, linestyle='--')
        
        # --- Quadrotor Rendering ---
        # Draw the quadrotor body as a scatter point.
        ax.scatter(pos[0], pos[1], color=color, s=50, zorder=3, alpha=alpha)

        # Compute the rotation matrix for the quadrotor (2D)
        R = np.array([[c_theta, -s_theta],
                    [s_theta, c_theta]])

        # Arm and rotor parameters
        Lq = self.Lq  # Arm length
        rotor_line_length = 0.4 * Lq  # Length of line representing the rotor
        half_line = rotor_line_length / 2.0

        # Define rotor offsets in the body frame (one on each side)
        rotor_offset1 = np.array([Lq, 0.2])
        rotor_offset2 = np.array([-Lq, 0.2])

        # Transform rotor positions to inertial frame
        rotor1 = pos + R @ rotor_offset1
        rotor2 = pos + R @ rotor_offset2

        # Draw arms: lines from the quadrotor center to each rotor position
        ax.plot([pos[0], rotor1[0]], [pos[1], rotor1[1]], color=color, lw=2, alpha=alpha)
        ax.plot([pos[0], rotor2[0]], [pos[1], rotor2[1]], color=color, lw=2, alpha=alpha)

        # --- Rotated Rotor Representation ---
        # Create a rotor line in its local frame (centered at zero)
        rotor_line_local = np.array([[-half_line, half_line], [0, 0]])

        # Rotate the rotor line by the same rotation matrix R (to align with vehicle orientation)
        rotor_line_rotated = R @ rotor_line_local  # shape (2,2)

        # Draw the rotor line for rotor1 at its computed position
        ax.plot(rotor_line_rotated[0, :] + rotor1[0],
                rotor_line_rotated[1, :] + rotor1[1],
                color=color, lw=3, alpha=alpha)

        # Draw the rotor line for rotor2 similarly
        ax.plot(rotor_line_rotated[0, :] + rotor2[0],
                rotor_line_rotated[1, :] + rotor2[1],
                color=color, lw=3, alpha=alpha)

        # --- Payload (Suspended Load) Rendering ---
        # Compute payload position using the pendulum angle.
        # Reconstruct phi from sin(phi) and cos(phi)
        Lp = self.Lp
        payload_pos = pos + np.array([Lp * s_phi, -Lp * c_phi])

        # Draw the tether as a line from the quadrotor to the payload
        ax.plot([pos[0], payload_pos[0]],
                [pos[1], payload_pos[1]],
                color=color, lw=1.5, alpha=alpha)

        # Draw the payload as a small circle (scatter point)
        ax.scatter(payload_pos[0], payload_pos[1], color=color, s=50, zorder=3, alpha=alpha)

        # --- Aesthetic Adjustments ---
        # ax.set_aspect('equal')
        ax.set_xlim(self._xbounds)
        ax.set_ylim(self._zbounds)
        ax.set_xticks([])
        ax.set_yticks([])
