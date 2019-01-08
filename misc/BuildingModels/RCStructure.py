#!/usr/bin/env python
"""
Description
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

G = nx.Graph()

params = pd.read_csv('buildParamSummary.csv', sep=';', index_col=0)

bp = params['SFH_SD_3_2zone_TAB']

print(bp.index)

# Day zone
G.add_node('TiD', C=bp['CiD'], T_fix=None,
           Q={'Q_sol_N': bp['abs3ND'], 'Q_sol_E': bp['abs3ED'], 'Q_sol_S': bp['abs3SD'], 'Q_sol_W': bp['abs3WD'],
              'Q_int_D': bp['f3D'], 'Q_hea_D': bp['f3D']}, state_type='day')
G.add_node('TflD', C=bp['CflD'], T_fix=None,
           Q={'Q_sol_N': bp['abs4ND'], 'Q_sol_E': bp['abs4ED'], 'Q_sol_S': bp['abs4SD'], 'Q_sol_W': bp['abs4WD'],
              'Q_int_D': bp['f4D'], 'Q_hea_D': bp['f4D']}, state_type='day')
G.add_node('TwiD', C=bp['CwiD'], T_fix=None,
           Q={'Q_sol_N': bp['abs2ND'], 'Q_sol_E': bp['abs2ED'], 'Q_sol_S': bp['abs2SD'], 'Q_sol_W': bp['abs2WD'],
              'Q_int_D': bp['f2D'], 'Q_hea_D': bp['f2D']}, state_type='day')
G.add_node('TwD', C=bp['CwD'], T_fix=None,
           Q={'Q_sol_N': bp['abs1ND'], 'Q_sol_E': bp['abs1ED'], 'Q_sol_S': bp['abs1SD'], 'Q_sol_W': bp['abs1WD'],
              'Q_int_D': bp['f1D'], 'Q_hea_D': bp['f1D']}, state_type='day')

# Internal floor
G.add_node('TfiD', C=bp['CfiD'], T_fix=None,
           Q={'Q_sol_N': bp['abs5ND'], 'Q_sol_E': bp['abs5ED'], 'Q_sol_S': bp['abs5SD'], 'Q_sol_W': bp['abs5WD'],
              'Q_int_D': bp['f5D'], 'Q_hea_D': bp['f5D']}, state_type='floor')
G.add_node('TfiN', C=bp['CfiD'], T_fix=None,
           Q={'Q_sol_N': bp['abs5NN'], 'Q_sol_E': bp['abs5EN'], 'Q_sol_S': bp['abs5SN'], 'Q_sol_W': bp['abs5WN'],
              'Q_int_N': bp['f5N'], 'Q_heaN': bp['f5N']}, state_type='floor')

# Night zone
G.add_node('TiN', C=bp['CiN'], T_fix=None,
           Q={'Q_sol_N': bp['abs3NN'], 'Q_sol_E': bp['abs3EN'], 'Q_sol_S': bp['abs3SN'], 'Q_sol_W': bp['abs3WN'],
              'Q_int_N': bp['f3N'], 'Q_heaN': bp['f3N']}, state_type='night')
G.add_node('TwiN', C=bp['CwiN'], T_fix=None,
           Q={'Q_sol_N': bp['abs2NN'], 'Q_sol_E': bp['abs2EN'], 'Q_sol_S': bp['abs2SN'], 'Q_sol_W': bp['abs2WN'],
              'Q_int_N': bp['f2N'], 'Q_heaN': bp['f2N']}, state_type='night')
G.add_node('TwN', C=bp['CwN'], T_fix=None,
           Q={'Q_sol_N': bp['abs1NN'], 'Q_sol_E': bp['abs1EN'], 'Q_sol_S': bp['abs1SN'], 'Q_sol_W': bp['abs1WN'],
              'Q_int_N': bp['f1N'], 'Q_heaN': bp['f1N']}, state_type='night')

# External temperatures
G.add_node('Te', T_fix='T_e', Q=None, state_type=None)
G.add_node('Tg', T_fix='T_g', Q=None, state_type=None)

# Connections
G.add_edge('Te', 'TwD', U=bp['UwD'])
G.add_edge('Te', 'TiD', U=bp['infD'])
G.add_edge('TwD', 'TiD', U=bp['hwD'])
G.add_edge('TiD', 'TflD', U=bp['hflD'])
G.add_edge('TflD', 'Tg', U=bp['UflD'])

G.add_edge('TiD', 'TwiD', U=bp['hwiD'])
G.add_edge('TiD', 'TfiD', U=bp['UfDN'])
G.add_edge('TfiD', 'TfiN', U=bp['UfND'])

G.add_edge('TfiN', 'TiN', U=bp['UfND'])
G.add_edge('TiN', 'TwiN', U=bp['hwiN'])
G.add_edge('TiN', 'TwN', U=bp['hwN'])
G.add_edge('TwN', 'Te', U=bp['UwN'])
G.add_edge('TiN', 'Te', U=bp['infN'])

nx.draw_networkx(G)
plt.show()

# [u'abs1ED', u'abs1ND', u'abs1SD', u'abs1WD', u'abs2ED', u'abs2ND',
#        u'abs2SD', u'abs2WD', u'abs3ED', u'abs3ND', u'abs3SD', u'abs3WD',
#        u'abs4ED', u'abs4ND', u'abs4SD', u'abs4WD', u'abs5ED', u'abs5ND',
#        u'abs5SD', u'abs5WD', u'CflD', u'CiD', u'CwD', u'CwiD', u'hwD', u'hflD',
#        u'hwiD', u'infD', u'UflD', u'UwD', u'f1D', u'f2D', u'f3D', u'f4D',
#        u'f5D', u'abs1EN', u'abs1NN', u'abs1SN', u'abs1WN', u'abs2EN',
#        u'abs2NN', u'abs2SN', u'abs2WN', u'abs3EN', u'abs3NN', u'abs3SN',
#        u'abs3WN', u'abs5EN', u'abs5NN', u'abs5SN', u'abs5WN', u'CiN', u'CwN',
#        u'CwiN', u'f1N', u'f2N', u'f3N', u'f5N', u'hwN', u'hwiN', u'infN',
#        u'UwN', u'CfiD', u'CfiN', u'UfDN', u'Ufi', u'UfND']
