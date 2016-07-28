from klampt import *
from klampt.glrobotprogram import *
import numpy as np
from actuators.CompliantHandEmulator import CompliantHandEmulator


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

class HandEmulator(CompliantHandEmulator):
    """An simulation model for the SoftHand for use with SimpleSimulation"""
    def __init__(self, sim, robotindex=0, link_offset=0, driver_offset=0):
        global gripper_name, numCommandDims
        CompliantHandEmulator.__init__(self, sim, robotindex, link_offset, driver_offset, a_dofs=3, d_dofs=1, u_dofs=6)

        self.n_fingers = 3
        self.u_dofs_per_finger = 2

        self.synergy_reduction = 7.0  # convert cable tension into motor torque

        print "Loaded robot name is:", self.robot.getName()
        print "Number of Drivers:", self.robot.numDrivers()
        if self.robot.getName() not in [gripper_name, "temp"]:
            raise Exception('loaded robot is not a reflex hand, rather %s'%self.robot.getName())

        # loading previously defined maps
        for i in xrange(driver_offset, self.robot.numDrivers()):
            driver = self.robot.driver(i)
            link = self.robot.link(driver.getName())

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

        # params loading
        self.E[0, 0] = self.E[2, 2] = self.E[4, 4] = 0.1
        self.E[1, 1] = self.E[3, 3] = self.E[5, 5] = 1

        self.initR()

        print 'Reflex Hand loaded.'
        self.printHandInfo()

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
        print "Could not load Reflex hand from", klampt_model_name
        exit(1)
    viewer = HandSimGLViewer(world)
    viewer.run()

    
