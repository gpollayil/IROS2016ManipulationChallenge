from klampt import se3,vectorops
from klampt import *
from klampt.glrobotprogram import *
from klampt.simulation import ActuatorEmulator
import numpy as np


#The hardware name
gripper_name = 'reflex'

#The Klamp't model name
klampt_model_name = 'data/robots/reflex.rob'

#the number of Klamp't model DOFs
numLinks = 19

#The number of command dimensions
numCommandDims = 4

#The names of the command dimensions
commandNames = ['finger1','finger2','finger3','preshape']

#default postures
openCommand = [1]
closeCommand = [0]

#named preset list
presets = {'open':openCommand,
           'closed':closeCommand
           }

#range of postures
commandMinimum = [0]
commandMaximum = [1]

#range of valid command velocities
commandMinimumVelocity = [-1]
commandMaximumVelocity = [1]

swivel_links = [2,7]
proximal_links = [3,8,12]
distal_links = [4,9,13]

class HandModel:
    """A kinematic model of the Reflex hand"""
    def __init__(self,robot,link_offset=0,driver_offset=0):
        """
        Arguments:
        - robot: the RobotModel instance containing the reflex_col hand.
        - link_offset: the link of the base of the hand in the robot model
        - driver_offset: the driver index of the first driver link in the robot model
        """
        global swivel_links,proximal_links,distal_links
        self.robot = robot
        self.link_offset = link_offset
        self.driver_offset = driver_offset
        qmin,qmax = self.robot.getJointLimits()
        self.swivel_driver = self.driver_offset
        self.swivel_links = [link_offset+i for i in swivel_links]
        self.proximal_links = [link_offset+i for i in proximal_links]
        self.distal_links = [link_offset+i for i in distal_links]
        self.proximal_drivers = [self.driver_offset+1,self.driver_offset+6,self.driver_offset+10]
        self.distal_drivers = [self.driver_offset+2,self.driver_offset+7,self.driver_offset+11]
        self.jointLimits = ([qmin[link_offset+proximal_links[0]],qmin[link_offset+proximal_links[1]],qmin[link_offset+proximal_links[2]],0],
                            [qmax[link_offset+proximal_links[0]],qmax[link_offset+proximal_links[1]],qmax[link_offset+proximal_links[2]],0])

class HandEmulator(ActuatorEmulator):
    """An simulation model for the SoftHand for use with SimpleSimulation"""
    def __init__(self, sim, robotindex=0, link_offset=0, driver_offset=0):
        global klampt_model_name, gripper_name, numCommandDims
        self.world = sim.world
        self.sim = sim
        self.sim.enableContactFeedbackAll()
        self.controller = self.sim.controller(robotindex)
        self.robot = self.world.robot(robotindex)
        self.model = HandModel(self.robot, link_offset, driver_offset)

        self.link_offset = link_offset
        self.driver_offset = driver_offset

        self.hand = dict()
        self.mimic = dict()
        self.n_dofs = 0

        self.n_fingers = 3
        self.u_dofs_per_finger = 2

        self.K_p = 3.0
        self.K_d = 0.03
        self.K_i = 0.01
        self.q_a_int = 0.0

        self.K_p_m = 3.0
        self.K_d_m = 0.03

        self.synergy_reduction = 7.0  # convert cable tension into motor torque

        print "Loaded robot name is:", self.robot.getName()
        print "Number of Drivers:", self.robot.numDrivers()
        if self.robot.getName() in [gripper_name, "temp"]:
            self.n_dofs = self.robot.numDrivers()
            self.a_dofs = numCommandDims - 1
            self.d_dofs = 1
            self.m_dofs = 0
            self.u_dofs = 6
        else:
            raise Exception('loaded robot is not a reflex hand, rather %s'%self.robot.getName())

        self.u_to_l = []    # will contain a map from underactuated joint id (excluding mimics) to child link id
        self.l_to_i = dict()# will contain a map from link id (global) to link index (local)
        self.u_to_n = []    # will contain a map from underactuated joint id (excluding mimics) to driver id
        self.a_to_n = []    # will contain a map from synergy actuators id to driver id
        self.d_to_n = []    # will contain a map from regular actuators id to driver id
        self.m_to_n = []    # will contain a map from mimic joints id to driver id

        self.n_to_u = np.array(self.n_dofs*[-1])
        self.n_to_m = np.array(self.n_dofs*[-1])
        self.n_to_a = np.array(self.n_dofs*[-1])
        self.n_to_d = np.array(self.n_dofs*[-1])

        self.q_to_t = []    # maps active drivers to joint ids
                            # (basically removes weld joints, counts affine joints only once, takes into account
                            #  floating base and regular joints properly)

        for i in xrange(driver_offset):
            driver = self.robot.driver(i)
            link = self.robot.link(driver.getName())
            self.q_to_t.append(link.getIndex())

        # loading previously defined maps
        for i in xrange(driver_offset, self.robot.numDrivers()):
            driver = self.robot.driver(i)
            link = self.robot.link(driver.getName())
            self.q_to_t.append(link.getIndex())

            print "Driver ", i, ": ", driver.getName()
            try:
                prefix, driver_name = driver.getName().split(':')
            except ValueError:
                driver_name = driver.getName()
            type, num = driver_name.split('_')
            num = int(num)
            if type in ['proximal', 'distal']:
                self.u_to_n.append(i)
                self.u_to_l.append(link.getID())
                u_id = len(self.u_to_n) - 1
                self.n_to_u[i] = u_id
                if not self.hand.has_key(num):
                    self.hand[num-1] = dict()
                    self.hand[num-1]['f_to_u'] = np.array(self.u_dofs_per_finger*[0.0])
                    if type is 'proximal':
                        type_to_id = 0
                    else:
                        type_to_id = 1
                self.hand[num-1]['f_to_u'][type_to_id] = u_id
            elif type == 'wire':
                self.a_to_n.append(i)
                a_id = len(self.a_to_n) - 1
                self.n_to_a[i] = a_id
            elif type == 'swivel':
                self.d_to_n.append(i)
                d_id = len(self.d_to_n) -1
                self.n_to_d[i] = d_id

            self.l_to_i[link.getID()] = link.getIndex()


        # checking load is successful
        assert len(self.u_to_n) == self.u_dofs
        self.u_dofs = len(self.u_to_n)
        assert len(self.m_to_n) == self.m_dofs
        self.m_dofs = len(self.m_to_n)
        assert len(self.a_to_n) == self.a_dofs
        self.a_dofs = len(self.a_to_n)

        self.m_to_u = self.m_dofs * [-1]

        # loading elasticity and reduction map
        self.R = np.zeros((self.a_dofs, self.u_dofs))
        self.E = np.eye(self.u_dofs)

        self.q_a_ref = np.array(self.a_dofs*[0.0])
        self.q_d_ref = np.array(self.d_dofs*[0.0])


        # params loading
        self.E[0, 0] = self.E[2, 2] = self.E[4, 4] = 0.1
        self.E[1, 1] = self.E[3, 3] = self.E[5, 5] = 1

        q = np.array(self.controller.getSensedConfig())
        q = q[self.q_to_t]
        q_u = q[self.u_to_n]
        self.updateR(q_u)

        print 'Reflex Hand loaded.'
        print 'Actuated Joint Indices:', self.d_to_n
        print 'Synergy Joint Indices:', self.a_to_n
        print 'Underactuated Joint Indices:', self.u_to_n
        print 'Joint to Driver map:', self.q_to_t
        print 'R:', self.R
        #self.E = 20 * self.E
        print 'E:', self.E

        kP, kI, kD = self.controller.getPIDGains()
        for i in self.u_to_n:
            kP[i] = 0.0
            kI[i] = 0.0
            kD[i] = 0.0

        # we could use directy a PID here...
        for i in self.m_to_n:
            kP[i] = 0.0
            kI[i] = 0.0
            kD[i] = 0.0

        # we could use directy a PID here...
        for i in self.a_to_n:
            kP[i] = 0.0
            kI[i] = 0.0
            kD[i] = 0.0

        self.controller.setPIDGains(kP, kI, kD)

    def output(self):
        torque = np.array(self.n_dofs * [0.0])

        q = np.array(self.controller.getSensedConfig())

        q = q[self.q_to_t]
        dq = np.array(self.controller.getSensedVelocity())
        dq = dq[self.q_to_t]

        dq_a = dq[self.a_to_n]
        dq_u = dq[self.u_to_n]
        dq_m = dq[self.m_to_n]

        q_a = q[self.a_to_n]
        q_u = q[self.u_to_n]
        q_m = q[self.m_to_n]

        # updates self.R
        self.updateR(q_u)

        R_E_inv_R_T_inv = np.linalg.inv(self.R.dot(np.linalg.inv(self.E)).dot(self.R.T))
        sigma = q_a # q_a goes from 0.0 to 1.0
        f_c, J_c = self.get_contact_forces_and_jacobians()
        tau_c = J_c.T.dot(f_c)

        # tendon tension
        f_a = R_E_inv_R_T_inv.dot(sigma) * self.synergy_reduction + R_E_inv_R_T_inv.dot(self.R).dot(np.linalg.inv(self.E).dot(tau_c))
        # emulate the synergy PID, notice there is no integrator ATM working on q_a_int
        torque_a = self.K_p*(self.q_a_ref - q_a) \
                   + self.K_d*(0.0 - dq_a) \
                   + self.K_i*self.q_a_int \
                   - (f_a / self.synergy_reduction)

        torque_u = self.R.T.dot(f_a) - self.E.dot(q_u)

        torque_m = self.K_p_m*(q_u[self.m_to_u] - q_m) - self.K_d_m*dq_m

        torque[self.a_to_n] = torque_a
        torque[self.u_to_n] = torque_u
        torque[self.m_to_n] = torque_m

        #print 'q_u:', q_u
        #print 'q_a_ref-q_a:',self.q_a_ref-q_a
        #print 'q_u-q_m:', q_u[self.m_to_u]-q_m
        #print 'tau_u:', torque_u

        return torque

    def updateR(self, q_u):
        da_vinci_f_i = np.ndarray((1,self.u_dofs_per_finger))

        # base pulley radius
        r0 = 0.1
        # proximal phalanx equivalent pulley radius
        r1 = 0.01
        # the distal link does not have a pulley
        r2 = 0.0

        a0 = 0.1
        b0 = 0.1
        l0 = 0.1

        a1 = 0.1
        b1 = 0.1
        l1 = 0.1

        a2 = 0.1
        b2 = 0.1
        l2 = 0.1


        # for each finger
        for i in xrange(self.n_fingers):
            # from q to theta
            theta_1_u_id = self.hand[i]['f_to_u'][0]
            theta_2_u_id = self.hand[i]['f_to_u'][1]
            theta_1 = q_u[theta_1_u_id]
            theta_2 = q_u[theta_2_u_id]

            # from Birglen et al, 2007, page 55
            r = r2 - r1
            a = l1 - a1 + a2*np.cos(theta_2) - b2*np.sin(theta_2)
            b = -b1 + a2*np.sin(theta_2) + b2*np.cos(theta_2)
            l = a**2 + b**2 - r**2
            R1 = r1 + (b1*(r*b - a*l) - (l1-a1)*(a*r + b*l))/(a**2+b**2)

            # from Birglen Transmission matrix to R
            da_vinci_f_i[0,0] = -r0
            da_vinci_f_i[0,1] = R1

            self.R[i,i*2:i*2+2] = da_vinci_f_i
        return self.R

    def get_contact_forces_and_jacobians(self):
        """
        Returns a force contact vector 1x(6*n_contacts)
        and a contact jacobian matrix 6*n_contactsxn.
        Contact forces are considered to be applied at the link origin
        """
        n_contacts = 0 # one contact per link
        if hasattr(self, 'world'):
            maxid = self.world.numIDs()
            J_l = dict()
            f_l = dict()
            t_l = dict()
            for l_id in self.u_to_l:
                l_index = self.l_to_i[l_id]
                link_in_contact = self.robot.link(l_index)
                contacts_per_link = 0
                for j in xrange(maxid): # TODO compute just one contact per link
                    contacts_l_id_j = len(self.sim.getContacts(l_id, j))
                    contacts_per_link += contacts_l_id_j
                    if contacts_l_id_j > 0:
                        if not f_l.has_key(l_id):
                            f_l[l_id] = self.sim.contactForce(l_id, j)
                            t_l[l_id] = self.sim.contactTorque(l_id, j)
                        f_l[l_id] = vectorops.add(f_l[l_id], self.sim.contactForce(l_id, j))
                        t_l[l_id] = vectorops.add(t_l[l_id], self.sim.contactTorque(l_id, j))
                        ### debugging ###
                        #print "link", link_in_contact.getName(), """
                        #      in contact with obj""", self.world.getName(j), """
                        #      (""", len(self.sim.getContacts(l_id, j)), """
                        #      contacts)\n f=""",self.sim.contactForce(l_id, j), """
                        #      t=""", self.sim.contactTorque(l_id, j)

                ### debugging ###
                """
                if contacts_per_link == 0:
                    print "link", link_in_contact.getName(), "not in contact"
                """
                if contacts_per_link > 0:
                    n_contacts += 1
                    J_l[l_id] = np.array(link_in_contact.getJacobian(
                                        (0, 0, 0)))
                    #print J_l[l_id].shape
        f_c = np.array(6 * n_contacts * [0.0])
        J_c = np.zeros((6 * n_contacts, self.u_dofs))

        for l_in_contact in xrange(len(J_l.keys())):
            f_c[l_in_contact * 6:l_in_contact * 6 + 3
                ] = f_l.values()[l_in_contact]
            f_c[l_in_contact * 6 + 3:l_in_contact * 6 + 6
                ] = t_l.values()[l_in_contact]
            J_c[l_in_contact * 6:l_in_contact * 6 + 6,
                :] = np.array(
                    J_l.values()[l_in_contact])[:, self.u_to_n]
        return (f_c, J_c)


    def setCommand(self, command):
        self.q_a_ref = [max(min(v, 1), 0) for i, v in enumerate(command) if i < 3]
        self.q_d_ref = [max(min(v, 1), 0) for i, v in enumerate(command) if i == 3]

    def getCommand(self):
        return np.hstack([self.q_a_ref, self.q_d_ref])

    def process(self, commands ,dt):
        if commands:
            if 'position' in commands:
                self.setCommand(commands['position'])
                del commands['position']
            if 'qcmd' in commands:
                self.setCommand(commands['qcmd'])
                del commands['qcmd']
            if 'speed' in commands:
                pass
            if 'force' in commands:
                pass

        qdes = np.array(self.controller.getCommandedConfig())
        qdes[[self.q_to_t[d_id] for d_id in self.d_to_n]] = self.q_d_ref
        dqdes = self.controller.getCommandedVelocity()
        self.controller.setPIDCommand(qdes, dqdes, self.output())
        
    def substep(self, dt):
        qdes = np.array(self.controller.getCommandedConfig())
        qdes[[self.q_to_t[d_id] for d_id in self.d_to_n]] = self.q_d_ref
        dqdes = self.controller.getCommandedVelocity()
        self.controller.setPIDCommand(qdes, dqdes, self.output())

    def drawGL(self):
        pass


class HandSimGLViewer(GLSimulationProgram):
    def __init__(self,world,base_link=0,base_driver=0):
        GLSimulationProgram.__init__(self,world,"Reflex simulation program")
        self.handsim = HandEmulator(self.sim,0,base_link,base_driver)
        self.sim.addEmulator(0,self.handsim)
        self.control_dt = 0.01

    def control_loop(self):
        #external control loop
        #print "Time",self.sim.getTime()
        return

    def idle(self):
        if self.simulate:
            self.control_loop()
            self.sim.simulate(self.control_dt)
            glutPostRedisplay()

    def print_help(self):
        GLSimulationProgram.print_help()
        print "y/h: raise/lower finger 1 command"
        print "u/j: raise/lower finger 2 command"
        print "i/k: raise/lower finger 3 command"
        print "o/l: raise/lower preshape command"

    def keyboardfunc(self,c,x,y):
        #Put your keyboard handler here
        #the current example toggles simulation / movie mode
        if c=='y':
            u = self.handsim.getCommand()
            u[0] += 0.1
            self.handsim.setCommand(u)
        elif c=='h':
            u = self.handsim.getCommand()
            u[0] -= 0.1
            self.handsim.setCommand(u)
        elif c=='u':
            u = self.handsim.getCommand()
            u[1] += 0.1
            self.handsim.setCommand(u)
        elif c=='j':
            u = self.handsim.getCommand()
            u[1] -= 0.1
            self.handsim.setCommand(u)
        elif c=='i':
            u = self.handsim.getCommand()
            u[2] += 0.1
            self.handsim.setCommand(u)
        elif c=='k':
            u = self.handsim.getCommand()
            u[2] -= 0.1
            self.handsim.setCommand(u)
        elif c=='o':
            u = self.handsim.getCommand()
            u[3] += 0.1
            self.handsim.setCommand(u)
        elif c=='l':
            u = self.handsim.getCommand()
            u[3] -= 0.1
            self.handsim.setCommand(u)
        else:
            GLSimulationProgram.keyboardfunc(self,c,x,y)
        glutPostRedisplay()

        
if __name__=='__main__':
    global klampt_model_name
    world = WorldModel()
    if not world.readFile(klampt_model_name):
        print "Could not load SoftHand hand from",klampt_model_name
        exit(1)
    viewer = HandSimGLViewer(world)
    viewer.run()

    
