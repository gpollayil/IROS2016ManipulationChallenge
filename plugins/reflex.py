from klampt import *
from klampt.vis.glrobotprogram import *
from klampt.math import *
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
        CompliantHandEmulator.__init__(self, sim, robotindex, link_offset, driver_offset, a_dofs=3, d_dofs=1, u_dofs=6)

        #self.synergy_reduction = 10.5
        self.synergy_reduction = 3.6
        self.effort_scaling = 10.5
        self.model = HandModel(self.robot, link_offset, driver_offset)

        print 'Reflex Hand loaded.'

        # debug maps: OK
        """
        print self.u_to_l
        print self.l_to_i
        for i in xrange(self.driver_offset, self.robot.numDrivers()):
            print "Driver name:", self.robot.driver(i).getName()
            u_id = self.n_to_u[i]
            print "id u_id:", i, u_id
            if u_id != -1:
                link_id = self.u_to_l[u_id]
                link_index = self.l_to_i[link_id]
                print "Link name (index):", self.robot.link(link_index).getName(), "(%d)"%link_index
                print "Link name (id):", self.world.getName(link_id), "(%d)"%link_id
        """

    def loadHandParameters(self):
        global klampt_model_name, gripper_name

        self.n_fingers = 3
        self.u_dofs_per_finger = 2

        print "Loaded robot name is:", self.robot.getName()
        print "Number of Drivers:", self.robot.numDrivers()
        if self.robot.getName() not in [gripper_name, "temp"]:
            raise Exception('loaded robot is not a reflex hand, rather %s' % self.robot.getName())

        # loading previously defined maps
        for i in xrange(self.driver_offset, self.robot.numDrivers()):
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
                u_id = len(self.u_to_n) - 1
                self.n_to_u[i] = u_id
                if not self.hand.has_key(num):
                    self.hand[num - 1] = dict()
                    self.hand[num - 1]['f_to_u'] = np.array(self.u_dofs_per_finger * [0.0])
                    if type is 'proximal':
                        type_to_id = 0
                    else:
                        type_to_id = 1
                self.hand[num - 1]['f_to_u'][type_to_id] = u_id
            elif type == 'wire':
                self.a_to_n.append(i)
                a_id = len(self.a_to_n) - 1
                self.n_to_a[i] = a_id
            elif type == 'swivel':
                self.d_to_n.append(i)
                d_id = len(self.d_to_n) - 1
                self.n_to_d[i] = d_id

        # checking load is successful
        assert len(self.u_to_n) == self.u_dofs
        self.u_dofs = len(self.u_to_n)
        assert len(self.m_to_n) == self.m_dofs
        self.m_dofs = len(self.m_to_n)
        assert len(self.a_to_n) == self.a_dofs
        self.a_dofs = len(self.a_to_n)

        # params loading
        self.E[0, 0] = self.E[2, 2] = self.E[4, 4] = 1.0
        self.E[1, 1] = self.E[3, 3] = self.E[5, 5] = 2.0

        self.q_u_rest = np.array(self.n_fingers*[-0.34,0.0])
        self.sigma_offset = np.array(self.a_dofs * [0.1])
        self.initR()

    def updateR(self, q_u):
        da_vinci_f_i = np.ndarray((1,self.u_dofs_per_finger))

        # base pulley radius
        r0 = 0.015
        # proximal phalanx equivalent pulley radius
        r1 = 0.002
        # the distal link does not have a pulley
        r2 = 0.0

        a1 = 0.04
        b1 = 0.01
        l1 = 0.05

        a2 = 0.02
        b2 = 0.0


        # for each finger
        for i in xrange(self.n_fingers):
            # from q to theta
            theta_1_u_id = self.hand[i]['f_to_u'][0]
            theta_2_u_id = self.hand[i]['f_to_u'][1]
            theta_1 = 0.5*np.pi - q_u[theta_1_u_id]
            #theta_1 = q_u[theta_1_u_id]
            theta_2 = 0.5*np.pi - q_u[theta_2_u_id]
            theta_2 = q_u[theta_2_u_id]

            # from Birglen et al, 2007, page 55
            r = r2 - r1
            a = l1 - a1 + a2*np.cos(theta_2) - b2*np.sin(theta_2)
            b = -b1 + a2*np.sin(theta_2) + b2*np.cos(theta_2)
            l = (a**2 + b**2 - r**2)**0.5
            R1 = r1 + (b1*(r*b - a*l) - (l1-a1)*(a*r + b*l))/(a**2+b**2)

            # from Birglen Transmission matrix to R
            da_vinci_f_i[0,0] = 1.0
            da_vinci_f_i[0,1] = -R1/r0

            self.R[i,i*2:i*2+2] = da_vinci_f_i
        return self.R

    def setCommand(self, command):
        self.q_a_ref = np.array([1.0 - self.sigma_offset[i] - max(min(v, 1), 0) for i, v in enumerate(command) if i < self.a_dofs])
        self.q_d_ref = np.array([max(min(v, 1), 0) for i, v in enumerate(command) if
                        i >= self.a_dofs and i < self.a_dofs + self.d_dofs])
        #print command


    def getCommand(self):
        return np.hstack([1.0 - self.sigma_offset - self.q_a_ref, self.q_d_ref])

class HandSimGLViewer(GLSimulationPlugin):
    def __init__(self,world,base_link=0,base_driver=0):
        GLSimulationPlugin.__init__(self,world,"Reflex simulation program")
        self.handsim = HandEmulator(self.sim,0,base_link,base_driver)
        self.sim.addEmulator(0,self.handsim)
        self.control_dt = 0.01

    def control_loop(self):
        #external control loop
        #print "Time",self.sim.getTime()
        return

    def display(self):
        GLSimulationPlugin.display(self)

        #draw forces
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(4.0)
        glBegin(GL_LINES)
        for l_id in self.handsim.virtual_contacts:
            glColor3f(0,1,0)
            forcelen = 0.1
            l = self.handsim.robot.link(self.handsim.l_to_i[l_id])
            b = self.sim.body(l)
            com = l.getMass().getCom()
            f = self.handsim.virtual_wrenches[l_id][0:3]
            glVertex3f(*se3.apply(b.getTransform(), com))
            glVertex3f(*se3.apply(b.getTransform(), vectorops.madd(com,f,forcelen)))
            """
            # draw local link frame
            for color in {(1, 0, 0), (0, 1, 0), (0, 0, 1)}:
                glColor3f(*color)
                glVertex3f(*se3.apply(b.getTransform(), com))
                glVertex3f(*se3.apply(b.getTransform(), vectorops.madd(com, color, 0.1)))
            """
        glEnd()
        glLineWidth(1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def idle(self):
        if self.simulate:
            for l_id in self.handsim.virtual_contacts:
                glColor3f(0, 1, 0)
                l = self.handsim.robot.link(self.handsim.l_to_i[l_id])
                b = self.sim.body(l)
                f = self.handsim.virtual_wrenches[l_id][0:3]
                com = l.getMass().getCom()
                b.applyForceAtLocalPoint(se3.apply_rotation(b.getTransform(),50*f),com) # could also use applyWrench with moment=[0,0,0]
            self.control_loop()
            self.sim.simulate(self.control_dt)

    def print_help(self):
        GLSimulationPlugin.print_help()
        print "y/h: raise/lower finger 1 command"
        print "u/j: raise/lower finger 2 command"
        print "i/k: raise/lower finger 3 command"
        print "o/l: raise/lower preshape command"
        print "e/d: activate/deactivate virtual force at finger 1 distal phalanx"
        print "r/f: activate/deactivate virtual force at finger 2 distal phalanx"
        print "t/g: activate/deactivate virtual force at finger 3 distal phalanx"

    def keyboardfunc(self,c,x,y):
        #Put your keyboard handler here
        #the current example toggles simulation / movie mode
        pl = self.handsim.model.proximal_links
        l2i = self.handsim.l_to_i
        link_index_to_id = {y: x for x, y in l2i.iteritems()}
        finger1_l_id, finger2_l_id, finger3_l_id = [link_index_to_id[index] for index in pl]
        force_at_com = [0, 0, -5.0]
        wrench_at_base = dict()
        for l_id in [finger1_l_id, finger2_l_id, finger3_l_id]:
            l = self.handsim.robot.link(self.handsim.l_to_i[l_id])
            b = self.sim.body(l)
            com = np.array(l.getMass().getCom())
            # m_b = m_com + f_com x com_b
            # com_b = -b_com = -com
            wrench_at_base[l_id] = tuple(force_at_com) + vectorops.cross(-com, force_at_com)

        if c=='y':
            u = self.handsim.getCommand()
            u[0] += 0.01
            self.handsim.setCommand(u)
        elif c=='h':
            u = self.handsim.getCommand()
            u[0] -= 0.01
            self.handsim.setCommand(u)
        elif c=='u':
            u = self.handsim.getCommand()
            u[1] += 0.01
            self.handsim.setCommand(u)
        elif c=='j':
            u = self.handsim.getCommand()
            u[1] -= 0.01
            self.handsim.setCommand(u)
        elif c=='i':
            u = self.handsim.getCommand()
            u[2] += 0.01
            self.handsim.setCommand(u)
        elif c=='k':
            u = self.handsim.getCommand()
            u[2] -= 0.01
            self.handsim.setCommand(u)
        elif c=='o':
            u = self.handsim.getCommand()
            u[3] += 0.01
            self.handsim.setCommand(u)
        elif c=='l':
            u = self.handsim.getCommand()
            u[3] -= 0.01
            self.handsim.setCommand(u)
        elif c == 'e':
            self.handsim.virtual_contacts[finger1_l_id] = True
            self.handsim.virtual_wrenches[finger1_l_id] = np.array(wrench_at_base[finger1_l_id])
        elif c == 'd':
            if self.handsim.virtual_contacts.has_key(finger1_l_id):
                self.handsim.virtual_contacts.pop(finger1_l_id)
            if self.handsim.virtual_wrenches.has_key(finger1_l_id):
                self.handsim.virtual_wrenches.pop(finger1_l_id)
        elif c == 'r':
            self.handsim.virtual_contacts[finger2_l_id] = True
            self.handsim.virtual_wrenches[finger2_l_id] = np.array(wrench_at_base[finger2_l_id])
        elif c == 'f':
            if self.handsim.virtual_contacts.has_key(finger2_l_id):
                self.handsim.virtual_contacts.pop(finger2_l_id)
            if self.handsim.virtual_wrenches.has_key(finger2_l_id):
                self.handsim.virtual_wrenches.pop(finger2_l_id)
        elif c == 't':
            self.handsim.virtual_contacts[finger3_l_id] = True
            self.handsim.virtual_wrenches[finger3_l_id] = np.array(wrench_at_base[finger3_l_id])
        elif c == 'g':
            if self.handsim.virtual_contacts.has_key(finger3_l_id):
                self.handsim.virtual_contacts.pop(finger3_l_id)
            if self.handsim.virtual_wrenches.has_key(finger3_l_id):
                self.handsim.virtual_wrenches.pop(finger3_l_id)
        else:
            GLSimulationPlugin.keyboardfunc(self,c,x,y)

        
if __name__=='__main__':
    world = WorldModel()
    if len(sys.argv) == 2:
        if not world.readFile(sys.argv[1]):
            print "Could not load Reflex hand from", sys.argv[1]
            exit(1)
    else:
        if not world.readFile(klampt_model_name):
            print "Could not load Reflex hand from", klampt_model_name
            exit(1)
    viewer = HandSimGLViewer(world)
    viewer.run()

    
