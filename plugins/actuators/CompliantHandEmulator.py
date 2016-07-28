from klampt import vectorops
from klampt.simulation import ActuatorEmulator
import numpy as np

class CompliantHandEmulator(ActuatorEmulator):
    """An simulation model for the SoftHand for use with SimpleSimulation"""
    def __init__(self, sim, robotindex=0, link_offset=0, driver_offset=0, a_dofs=0, d_dofs=0, u_dofs=0, m_dofs=0):
        global klampt_model_name, gripper_name, numCommandDims
        self.world = sim.world
        self.sim = sim
        self.sim.enableContactFeedbackAll()
        self.controller = self.sim.controller(robotindex)
        self.robot = self.world.robot(robotindex)

        self.link_offset = link_offset
        self.driver_offset = driver_offset

        self.hand = dict()
        self.mimic = dict()
        self.n_dofs = 0

        self.K_p = 3.0
        self.K_d = 0.03
        self.K_i = 0.01
        self.q_a_int = 0.0

        self.K_p_m = 3.0
        self.K_d_m = 0.03

        self.synergy_reduction = 7.0  # convert cable tension into motor torque

        self.n_dofs = self.robot.numDrivers()
        self.a_dofs = a_dofs
        self.d_dofs = d_dofs
        self.u_dofs = u_dofs
        self.m_dofs = m_dofs

        self.u_to_l = []  # will contain a map from underactuated joint id (excluding mimics) to child link id
        self.l_to_i = dict()  # will contain a map from link id (global) to link index (local)
        self.u_to_n = []  # will contain a map from underactuated joint id (excluding mimics) to driver id
        self.a_to_n = []  # will contain a map from synergy actuators id to driver id
        self.d_to_n = []  # will contain a map from regular actuators id to driver id
        self.m_to_n = []  # will contain a map from mimic joints id to driver id

        self.n_to_u = np.array(self.n_dofs * [-1])
        self.n_to_m = np.array(self.n_dofs * [-1])
        self.n_to_a = np.array(self.n_dofs * [-1])
        self.n_to_d = np.array(self.n_dofs * [-1])

        self.q_to_t = []  # maps active drivers to joint ids
        # (basically removes weld joints, counts affine joints only once, takes into account
        #  floating base and regular joints properly)

        for i in xrange(self.robot.numDrivers()):
            driver = self.robot.driver(i)
            link = self.robot.link(driver.getName())
            self.q_to_t.append(link.getIndex())

        self.m_to_u = self.m_dofs * [-1]

        # loading elasticity and reduction map
        self.R = np.zeros((self.a_dofs, self.u_dofs))
        self.E = np.eye(self.u_dofs)

        self.q_a_ref = np.array(self.a_dofs * [0.0])
        self.q_d_ref = np.array(self.d_dofs * [0.0])

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

    def printHandInfo(self):
        print 'Actuated Joint Indices:', self.d_to_n
        print 'Synergy Joint Indices:', self.a_to_n
        print 'Underactuated Joint Indices:', self.u_to_n
        print 'Mimic Joint Indices:', self.m_to_n
        print 'Joint to Driver map:', self.q_to_t
        print 'R:', self.R
        print 'E:', self.E

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
        sigma = q_a  # q_a goes from 0.0 to 1.0
        f_c, J_c = self.get_contact_forces_and_jacobians()
        tau_c = J_c.T.dot(f_c)

        # tendon tension
        f_a = R_E_inv_R_T_inv.dot(sigma) * self.synergy_reduction + R_E_inv_R_T_inv.dot(self.R).dot(
            np.linalg.inv(self.E).dot(tau_c))
        # emulate the synergy PID, notice there is no integrator ATM working on q_a_int
        torque_a = self.K_p * (self.q_a_ref - q_a) \
                   + self.K_d * (0.0 - dq_a) \
                   + self.K_i * self.q_a_int \
                   - (f_a / self.synergy_reduction)

        torque_u = self.R.T.dot(f_a) - self.E.dot(q_u)

        torque_m = self.K_p_m * (q_u[self.m_to_u] - q_m) - self.K_d_m * dq_m

        torque[self.a_to_n] = torque_a
        torque[self.u_to_n] = torque_u
        torque[self.m_to_n] = torque_m

        # print 'q_u:', q_u
        # print 'q_a_ref-q_a:',self.q_a_ref-q_a
        # print 'q_u-q_m:', q_u[self.m_to_u]-q_m
        # print 'tau_u:', torque_u

        return torque


    def initR(self):
        q = np.array(self.controller.getSensedConfig())
        q = q[self.q_to_t]
        q_u = q[self.u_to_n]
        self.updateR(q_u)

    def updateR(self, q_u):
        return self.R


    def get_contact_forces_and_jacobians(self):
        """
        Returns a force contact vector 1x(6*n_contacts)
        and a contact jacobian matrix 6*n_contactsxn.
        Contact forces are considered to be applied at the link origin
        """
        n_contacts = 0  # one contact per link
        if hasattr(self, 'world'):
            maxid = self.world.numIDs()
            J_l = dict()
            f_l = dict()
            t_l = dict()
            for l_id in self.u_to_l:
                l_index = self.l_to_i[l_id]
                link_in_contact = self.robot.link(l_index)
                contacts_per_link = 0
                for j in xrange(maxid):  # TODO compute just one contact per link
                    contacts_l_id_j = len(self.sim.getContacts(l_id, j))
                    contacts_per_link += contacts_l_id_j
                    if contacts_l_id_j > 0:
                        if not f_l.has_key(l_id):
                            f_l[l_id] = self.sim.contactForce(l_id, j)
                            t_l[l_id] = self.sim.contactTorque(l_id, j)
                        f_l[l_id] = vectorops.add(f_l[l_id], self.sim.contactForce(l_id, j))
                        t_l[l_id] = vectorops.add(t_l[l_id], self.sim.contactTorque(l_id, j))
                        ### debugging ###
                        # print "link", link_in_contact.getName(), """
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
                    # print J_l[l_id].shape
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
        self.q_a_ref = [max(min(v, 1), 0) for i, v in enumerate(command) if i < self.a_dofs]
        self.q_d_ref = [max(min(v, 1), 0) for i, v in enumerate(command) if i >= self.a_dofs and i < self.a_dofs + self.d_dofs]


    def getCommand(self):
        return np.hstack([self.q_a_ref, self.q_d_ref])


    def process(self, commands, dt):
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