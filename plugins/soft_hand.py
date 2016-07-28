from klampt import *
from klampt.glrobotprogram import *
from loaders.soft_hand_loader import SoftHandLoader
from actuators.CompliantHandEmulator import CompliantHandEmulator
import numpy as np



#The hardware name
gripper_name = 'soft_hand'

#The Klamp't model name
klampt_model_name = 'data/robots/soft_hand.urdf'

#the number of Klamp't model DOFs
numLinks = 38

#The number of command dimensions
numCommandDims = 1

#The names of the command dimensions
commandNames = ['synergy']

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

class HandEmulator(CompliantHandEmulator):
    """An simulation model for the SoftHand for use with SimpleSimulation"""
    def __init__(self, sim, robotindex=0, link_offset=0, driver_offset=0):
        global klampt_model_name, gripper_name
        CompliantHandEmulator.__init__(self, sim, robotindex, link_offset, driver_offset, a_dofs=1, d_dofs=0)

        self.paramsLoader = SoftHandLoader(klampt_model_name)


        print "Loaded robot name is:", self.robot.getName()
        print "Number of Drivers:", self.robot.numDrivers()
        if self.robot.getName() not in [gripper_name, "temp"]:
            raise Exception('loaded robot is not a soft hand, rather %s'%self.robot.getName())

        # loading previously defined maps
        for i in xrange(driver_offset, self.robot.numDrivers()):
            driver = self.robot.driver(i)
            print "Driver ", i, ": ", driver.getName()
            try:
                _,_,finger, phalanx,fake_id = driver.getName().split('_')
            except ValueError:
                prefix, name = driver.getName().split(':')
                _, _, finger, phalanx, fake_id = name.split('_')
            if phalanx == "fake":
                if not self.mimic.has_key(finger):
                    self.mimic[finger] = []
                self.mimic[finger].append(i)
                self.m_to_n.append(i)
                m_id = len(self.m_to_n)-1
                self.n_to_m[i] = m_id
            elif phalanx == "wire":
                self.a_to_n.append(i)
                a_id = len(self.a_to_n) - 1
                self.n_to_a[i] = a_id
            else:
                if not self.hand.has_key(finger):
                    self.hand[finger] = dict()
                self.hand[finger][phalanx] = i
                self.u_to_n.append(i)
                link = self.robot.link(self.robot.driver(i).getName())
                self.u_to_l.append(link.getID())
                self.l_to_i[link.getID()] = link.getIndex()
                u_id = len(self.u_to_n)-1
                self.n_to_u[i] = u_id

        self.u_dofs = len(self.u_to_n)
        self.m_dofs = len(self.m_to_n)
        # checking load is successful
        assert len(self.a_to_n) == self.a_dofs
        self.a_dofs = len(self.a_to_n)

        # will contain a map from underactuated joint to mimic joints
        # this means, for example, that joint id 1 has to be matched by mimic joint 19
        self.m_to_u = self.m_dofs*[-1]

        for finger in self.hand.keys():
            for phalanx in self.hand[finger].keys():
                joint_count = 0
                if phalanx == 'abd':
                    continue
                else:
                    m_id = self.n_to_m[self.mimic[finger][joint_count]]
                    self.m_to_u[m_id] = self.n_to_u[self.hand[finger][phalanx]]
                    joint_count = joint_count+1

        # loading elasticity and reduction map
        self.R = np.zeros((self.a_dofs, self.u_dofs))
        self.E = np.eye(self.u_dofs)

        for i in xrange(driver_offset, self.robot.numDrivers()):
            driver = self.robot.driver(i)
            try:
                _, _, finger, phalanx, fake_id = driver.getName().split('_')
            except ValueError:
                prefix, name = driver.getName().split(':')
                _, _, finger, phalanx, fake_id = name.split('_')
            u_id = self.n_to_u[i]
            if u_id != -1:
                joint_position = self.paramsLoader.phalanxToJoint(finger,phalanx)
                self.R[0, u_id] = self.paramsLoader.handParameters[finger][joint_position]['r']
                self.E[u_id,u_id] = self.paramsLoader.handParameters[finger][joint_position]['e']

        print 'Soft Hand loaded.'
        self.printHandInfo()
        print 'Mimic Joint Info:', self.mimic
        print 'Underactuated Joint Info:', self.hand
        print 'Joint parameters:', self.paramsLoader.handParameters


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
        print "o/l: increase/decrease synergy command"

    def keyboardfunc(self, c, x, y):
        # Put your keyboard handler here
        # the current example toggles simulation / movie mode
        if c == 'o':
            u = self.handsim.getCommand()
            u[0] += 0.1
            self.handsim.setCommand(u)
        elif c == 'l':
            u = self.handsim.getCommand()
            u[0] -= 0.1
            self.handsim.setCommand(u)
        else:
            GLSimulationProgram.keyboardfunc(self, c, x, y)
        glutPostRedisplay()

        
if __name__=='__main__':
    global klampt_model_name
    world = WorldModel()
    if not world.readFile(klampt_model_name):
        print "Could not load SoftHand hand from",klampt_model_name
        exit(1)
    viewer = HandSimGLViewer(world)
    viewer.run()

    
