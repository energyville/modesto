#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
File with functions to load state space models in continuous form,
discretize and transform them into optimization constraints.
"""

import numpy as np
import scipy.io as sio
import buildrc as rcm
from control import ss
import pandas as pd


class StateSpace(object):
    def __init__(self):
        """
        Class that contains state space matrices.
        B is the control input matrix, E is the disturbance matrix.
        """
        self.cont = {
            'A': 0,
            'B': 0,
            'C': 0,
            'DB': 0,
            'DE': 0,
            'E': 0
        }

    def buildss(self, A, B, E):
        """
        Fill in continuous matrices of system using matrices

        :param A: State matrix
        :param B: Input to state matrix
        :param E: Disturbance to state matrix
        """
        self.cont['A'] = A
        self.cont['B'] = B
        C = np.eye(len(A))
        self.cont['C'] = C
        DB = np.zeros_like(B)
        self.cont['DB'] = DB
        DE = np.zeros_like(E)
        self.cont['DE'] = DE
        self.cont['E'] = E
        self.ssContB = ss(A, B, C, DB)
        self.ssContE = ss(A, E, C, DE)


    def discretize(self, tstep):
        """
        Discretize existing continuous state space system.

        :param tstep: time step in seconds
        :return: None
        """
        try:
            self.ssContB
        except NameError:
            print("No continuous state space object initialized yet.")
            print("First build a state space object using 'buildss' or "
                  "another function.")
        else:
            self.ssDiscB = self.ssContB.sample(tstep)
            self.ssDiscE = self.ssContE.sample(tstep)

            self.A = pd.DataFrame(self.ssDiscB.A, columns=self.sta,
                                  index=self.sta)

            self.B = pd.DataFrame(self.ssDiscB.B, columns=self.inp,
                                  index=self.sta)

            self.E = pd.DataFrame(self.ssDiscE.B, columns=self.dist,
                                  index=self.sta)

            self.disc = {}
            self.disc['A']  = self.A
            self.disc['B']  = self.B
            self.disc['E']  = self.E
            self.disc['C']  = self.ssDiscB.C
            self.disc['DB'] = self.ssDiscB.D
            self.disc['DE'] = self.ssDiscE.D



    def set_disturbance(self, dist=['Te', 'Tg', 'QsolN', 'QsolE', 'QsolS',
                                    'QsolW', 'QintD', 'QintN']):
        """
        Set disturbances in SS model

        :param dist: list of disturbance names
        """
        assert len(dist) == np.shape(self.cont['E'])[1], 'Number of ' \
                                                         'disturbances ' \
                                                  'must ' \
                                                 'match the size of E'
        self.dist = dist

    def set_input(self, inp=['QhD', 'QhN']):
        """
        set control inputs

        :param inp: list of control input names
        """
        assert len(inp) == np.shape(self.cont['B'])[1], 'Number of ' \
                                                        'disturbances ' \
                                                  'must ' \
                                                'match the size of E'
        self.inp = inp

    def set_state(self,
                  sta=['TiD', 'TwiD', 'TwD', 'TflD', 'TflDN', 'TiN', 'TiwN',
                       'TwN', 'TflND']):
        """
        Set state names for ss model

        :param sta: list of state names
        """
        assert len(sta) == len(
            self.cont['A']), 'Number of states must match the size of A'

        self.sta = sta

    def read_mat(self, filename='BuildingsFH/SFH_T5_RecVent',
                 sta=['TiD', 'TwiD', 'TwD', 'TflD', 'TflDN', 'TiN', 'TiwN',
                      'TwN', 'TflND'],
                 inp=['QhD', 'QhN'],
                 dist=['Te', 'Tg', 'QsolN', 'QsolE', 'QsolS', 'QsolW',
                       'QintD', 'QintN']):

        """
        Read state space system from matrix

        :param sta: List of states
        :param inp: List of inputs
        :param dist: List of disturbances
        :param filename: Location of input file
        """
        # Read MatLab .mat file. Zeros are needed to access correct matrix.
        data = sio.loadmat(filename)['model']['systemCont'][0][0][0][0]
        A = data[0]
        B = data[1]
        E = data[2]
        self.buildss(A, B, E)
        self.set_disturbance(dist)
        self.set_input(inp)
        self.set_state(sta)

    def read_rc(self, rc):
        """
        Construct state space model from RCmodel
        This constructs the continuous-time state-space representation.
        Please discretize before building optimization problem.

        :param rc: RCModel instance
        """
        assert isinstance(rc, rcm.RCmodel), 'rc must be an instance of RCmodel'
        A, sta = rc.buildA()
        B, _, inp = rc.buildB()
        E, _, dist = rc.buildE()

        self.buildss(A,B,E)
        self.set_state(sta)
        self.set_input(inp)
        self.set_disturbance(dist)
