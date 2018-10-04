from __future__ import division

import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from modesto.main import Modesto
from modesto.mass_flow_calculation import MfCalculation
import modesto.utils as ut
import math
import numpy as np
import os
import pickle
import copy
import collections

# TODO Put this elsewhere?

old_building = 'SFH_D_3_2zone_TAB'
mixed_building = 'SFH_D_3_2zone_REF1'
new_building = 'SFH_D_5_ins_TAB'


max_heat = {'SFH_D_5_ins_TAB': 8900,
            'SFH_D_3_2zone_REF1': 8659,
            'SFH_D_3_2zone_TAB': 20000}

supply_temp = 333.15
return_temp = 303.15
delta_T_prod = 10  # Producer temperature difference
delta_T = supply_temp - return_temp  # substation temperature difference


class ResponseFunctionGenerator:

    def __init__(self, name, horizon, time_step, start_time):
        self.logger = logging.getLogger('Main.py')

        self.name = name
        self.graph = None

        self.time_index = None
        self.original_horizon = horizon
        self.horizon = None
        self.time_step = None
        self.start_time = None
        self.time_settings(horizon, time_step, start_time)

        self.dr = DataReader(horizon + 10*time_step, time_step, start_time)

        self.district_cases = []
        self.model_cases = []
        self.flex_cases = []
        self.sens_cases = []

        self.model = None
        self.results = {}

    def time_settings(self, horizon=None, time_step=None, start_time=None):
        if horizon is not None:
            self.horizon = horizon
        if time_step is not None:
            self.time_step = time_step
        if start_time is not None:
            self.start_time = start_time

        self.create_time_index(self.start_time, self.horizon, self.time_step)

    def create_district_cases(self, cases):
        available_cases = {'Old street': OldStreetCase,
                           'Mixed street': MixedStreetCase,
                           'New street': NewStreetCase,
                           'Series district': SeriesCase,
                           'Parallel district': ParallelCase,
                           'Genk': GenkCase}

        for case in cases:
            if case not in available_cases:
                raise Exception('{} is an invalid class type, choose among {} instead'.
                                format(case, available_cases.keys()))
            self.district_cases.append(available_cases[case]())

    def create_model_cases(self, cases):
        available_cases = {'Buildings - ideal network': BuildingsIdealCase,
                           'Buildings': BuildingCase,
                           'Network': NetworkCase,
                           'Combined - LP': CombinedLPCase}
        for case in cases:
            if case not in available_cases:
                raise Exception('{} is an invalid model cases, choose among {} instead'
                                .format(cases, available_cases.keys()))
            self.model_cases.append(available_cases[case]())

    def create_flex_cases(self):
        available_cases = {'Reference': ReferenceCase,
                           'Flexibility': FlexibilityCase}

        for case in available_cases:
            self.flex_cases.append(available_cases[case]())

    def create_sens_cases(self, cases):
        available_cases = {'Network size': NetworkSize,
                           'Pipe length': PipeLength,
                           'Pipe diameter': PipeDiameter,
                           'Heat demand': HeatDemand,
                           'Supply temperature level': SupplyTempLevel,
                           'Supply temperature reach': SupplyTempReach,
                           'Substation temperature difference': SubstationTempDiff}

        for case in cases:
            self.sens_cases.append(available_cases[case]())

    def create_time_index(self, start_time, horizon, time_step):
        self.time_index = pd.date_range(start=start_time, periods=int(horizon /time_step) + 1, freq=str(time_step) + 'S')

    def check_time_settings(self):
        if self.horizon is None:
            raise Exception('A value has not been assigned to the horizon yet')
        if self.start_time is None:
            raise Exception('A value has not been assigned to the start time yet')
        if self.time_step is None:
            raise Exception('A value has not been assigned to the time step yet')
        if self.time_index is None:
            raise Exception('A value has not been assigned to the time index yet')

    def collect_data(self):
        self.check_time_settings()
        self.dr.read_data([mixed_building, old_building, new_building]
                          )

    def find_node_mf_rate(self, district_case, model_case, node_nr, sens_case=None, value=None):
        heat_profile_type = model_case.get_heat_profile()
        if heat_profile_type:
            mult = district_case.get_node_mult(node_nr)
            heat_profile = \
                self.find_heat_profile(district_case, heat_profile_type, node_nr, sens_case, value)
        else:
            mult = 0
            heat_profile = pd.Series(0, self.time_index)

        return heat_profile/4186/delta_T

    def add_mass_flow_rates(self, district_case, model_case, sens_case=None, value=None):
        self.check_time_settings()
        mfcalc = MfCalculation(district_case.graph,
                               self.get_model_case('Buildings').get_time_step(),
                               self.horizon)

        for i in range(district_case.get_nr_of_nodes()):
            node_name = district_case.get_building_name(i)

            mfcalc.add_mf(node=node_name, name='building',
                          mf_df=self.find_node_mf_rate(district_case, model_case, i, sens_case, value))

            mfcalc.add_mf(node=node_name, name='DHW',
                          mf_df=self.dr.get_dhw_use(district_case.aggregated,
                                                    district_case.get_node_mult(i),
                                                    building_number=i, return_heat_profile=False) *
                          district_case.get_node_mult(i))


        mfcalc.set_producer_node('Producer')
        mfcalc.set_producer_component('plant')
        mfcalc.calculate_mf()

        for pipe in district_case.get_pipe_names():
            self.model.change_param(node=None, comp=pipe, param='mass_flow', val=mfcalc.get_edge_mf(pipe))

        self.model.change_param(node='Producer', comp='plant', param='mass_flow',
                                val=mfcalc.get_comp_mf(node='Producer', comp='plant'))

    def get_general_parameters(self):
        return {'Te': self.dr.t_amb,
                'Tg': self.dr.t_g,
                'Q_sol_E': self.dr.QsolE,
                'Q_sol_N': self.dr.QsolE,
                'Q_sol_S': self.dr.QsolE,
                'Q_sol_W': self.dr.QsolE,
                'time_step': self.time_step,
                'horizon': self.horizon}

    def select_parameters(self, params, key_list):
        return {name: params[name] for name in key_list}

    def get_building_params(self, district_case, model_case, node_nr, sens_case=None, value=None):
        b_params = {'model_type': district_case.get_building_type(node_nr),
                    'max_heat': max_heat[district_case.get_building_type(node_nr)],
                    'mult': district_case.get_node_mult(node_nr),
                    'delta_T': delta_T,
                    'temperature_return': return_temp,
                    'temperature_supply': supply_temp,
                    'temperature_max': 373.15,
                    'temperature_min': 273.15,
                    }

        b_params.update(self.dr.get_user_behaviour(district_case.aggregated,
                                                   district_case.get_node_mult(node_nr),
                                                   node_nr,
                                                   district_case.get_building_type(node_nr)))

        heat_profile_type = model_case.get_heat_profile()
        if heat_profile_type:
            b_params['heat_profile'] = self.find_heat_profile(district_case,
                                                              heat_profile_type, node_nr, sens_case, value)
            b_params['mult'] = 1

        return self.select_parameters(b_params, model_case.get_building_params())

    def find_heat_profile(self, dist_case, hp_type, node_nr, sens_case=None, value=None):

        mult = dist_case.get_node_mult(node_nr)
        if sens_case is None:
            heat_profile = self.results[dist_case.name]['Buildings'][hp_type] \
                ['building_heat_use'][dist_case.get_building_name(node_nr)]
        else:
            heat_profile = self.results[dist_case.name]['Buildings'][sens_case.name][value][hp_type] \
                                       ['building_heat_use'][dist_case.get_building_name(node_nr)]

        # Introducing bypass to increase robustness
        for j, val in enumerate(heat_profile):
            if val <= 0.1:
                heat_profile[j] = 10 * mult

        return heat_profile

    def get_producer_params(self, flex_case, model_case, sens_case=None):
        self.check_time_settings()
        prod_params = {'fuel_cost': flex_case.get_price_profile(self.time_index),
                       'efficiency': 1,
                        'PEF': 1,
                        'CO2': 0.178,  # based on HHV of CH4 (kg/KWh CH4)
                        'Qmax': 1.5e12,
                        'ramp_cost': 0.01,
                        'ramp': 1.5e12 / 3600,
                        'temperature_supply': supply_temp,
                        'temperature_return': return_temp,
                        'temperature_max': supply_temp + delta_T_prod,
                        'temperature_min': supply_temp}

        return self.select_parameters(prod_params, model_case.get_producer_params())

    def get_pipe_params(self, district_case, model_case, pipe_name, sens_case=None):
        pipe_params = {'temperature_supply': supply_temp,
                       'temperature_return': return_temp,
                       'temperature_history_return': pd.Series(return_temp, index=range(10)),
                       'temperature_history_supply': pd.Series(supply_temp, index=range(10)),
                       'mass_flow_history': pd.Series(0.1, index=range(10)),
                       'wall_temperature_supply': supply_temp,
                       'wall_temperature_return': return_temp,
                       'temperature_out_supply': supply_temp,
                       'temperature_out_return': return_temp,
                       'diameter': district_case.get_pipe_diameter(pipe_name)}

        return self.select_parameters(pipe_params, model_case.get_pipe_params())

    def get_dhw_params(self, district_case, model_case, node_nr, sens_case=None):

        dhw_params = {
            'delta_T': delta_T,
            'mult': district_case.get_node_mult(node_nr),
            'temperature_return': return_temp,
            'temperature_supply': supply_temp,
            'temperature_max': supply_temp + 20,
            'temperature_min': return_temp - 20,
            'heat_profile': self.dr.get_dhw_use(district_case.aggregated,
                                                district_case.get_node_mult(node_nr),
                                                node_nr)
        }

        return self.select_parameters(dhw_params, model_case.get_dhw_params())

    def set_up_modesto(self, district_case, model_case, flex_case, sens_case=None, value=None):

        self.time_settings(horizon=self.original_horizon + model_case.extra_horizon(),
                           time_step=model_case.get_time_step())

        district_case.renew_graph(model_case.get_building_model())

        if self.model is not None:
            del self.model
        self.model = Modesto(model_case.get_pipe_model(), district_case.graph)
        self.model.opt_settings(allow_flow_reversal=False)

        general, build_params, prod_params, dhw_params, pipe_params = \
            self.collect_parameters(district_case, model_case, flex_case, sens_case, value)

        self.model.change_params(general)

        for i in district_case.get_building_names():
            self.model.change_params(build_params[i],
                                     node=i,
                                     comp='building')

            self.model.change_params(dhw_params[i],
                                     node=i,
                                     comp='DHW')

        for pipe in district_case.get_pipe_names():
            self.model.change_params(pipe_params[pipe],
                                     node=None,
                                     comp=pipe)

        self.model.change_params(prod_params,
                                 node='Producer',
                                 comp='plant')

        if model_case.is_node_method():
            self.add_mass_flow_rates(district_case, model_case, sens_case, value)


    def collect_parameters(self, district_case, model_case, flex_case, sens_case=None, value=None):

        if sens_case.heat_demand:  # TODO prettier
            for i in range(district_case.get_nr_of_nodes()):
                district_case.change_mult(value, i)

        general = self.get_general_parameters()
        build_params = {name: self.get_building_params(district_case, model_case, i, sens_case, value)
                        for i, name in enumerate(district_case.get_building_names())}
        dhw_params = {name: self.get_dhw_params(district_case, model_case, i)
                        for i, name in enumerate(district_case.get_building_names())}
        pipe_params = {i: self.get_pipe_params(district_case, model_case, i)
                        for i in district_case.get_pipe_names()}
        prod_params = self.get_producer_params(flex_case, model_case)

        if sens_case is not None:
            sens_case.change_parameters(value, build_params, prod_params, dhw_params,
                                        pipe_params, model_case, district_case)

        return general, build_params, prod_params, dhw_params, pipe_params

    def solve_optimization(self, timelim=None, tee=False):
        self.model.compile(start_time=self.start_time)
        self.model.set_objective('cost')
        status = self.model.solve(timelim=timelim, tee=tee)

        if status == 0:
            print 'Slack: ', self.model.model.Slack.value
            print 'Energy:', self.model.get_objective('energy') - self.model.model.Slack.value, ' kWh'
            print 'Cost:  ', self.model.get_objective('cost') - self.model.model.Slack.value, ' euro'
            return True
        else:
            return False

    def get_building_heat_profile(self, district_case):
        node_names = district_case.get_building_names()
        building_heat_use = {}

        for i in node_names:
            building_heat_use[i] = \
                self.model.get_result('heat_flow', node=i, comp='building', state=True)

        return building_heat_use

    def get_building_temperatures(self, district_case, statename):
        node_names = district_case.get_building_names()
        result = {}

        for i in node_names:
            result[i] = self.model.get_result('StateTemperatures', node=i,
                                               comp='building', index=statename, state=True)


        return result

    def get_heat_injection(self):
        return self.model.get_result('heat_flow', node='Producer', comp='plant')

    def get_water_velocity(self, district_case):
        edge_names = district_case.get_pipe_names()
        result = {}

        for i in edge_names:
            result[i] = self.model.get_result('mass_flow', node=None, comp=i) / 1000 / (
                        3.14 * self.model.get_pipe_diameter(i) ** 2 / 4)

        return result

    def get_total_mf_rate(self):
        return self.model.get_result('mass_flow', node='Producer', comp='plant')

    def get_network_temperatures(self, district_case):
        edge_names = district_case.get_pipe_names()
        result = {}

        for pipe in edge_names:
            result[pipe] = self.model.get_result('temperature_out', node=None, comp=pipe, index='supply')

        return result

    def get_plant_temperature(self):

        result = {'supply': self.model.get_result(node='Producer', comp='plant', name='temperatures', index='supply'),
                  'return': self.model.get_result(node='Producer', comp='plant', name='temperatures', index='return')}

        return result

    def collect_results(self, district_case, model_case):
        result = {'building_heat_use': self.get_building_heat_profile(district_case)}

        if model_case.is_rc_model():
            result['day_zone_temperatures'] = self.get_building_temperatures(district_case, 'TiD')
            result['night_zone_temperatures'] = self.get_building_temperatures(district_case, 'TiN')
        if model_case.is_node_method():
            result['network_temperature'] = self.get_network_temperatures(district_case)
            result['water_velocity'] = self.get_water_velocity(district_case)
            result['plant_temperature'] = self.get_plant_temperature()
        result['heat_injection'] = self.get_heat_injection()
        result['total_mass_flow_rate'] = self.get_total_mf_rate()

        return result

    # TODO Change inputs?
    def plot_network_temperatures(self, fig, ax, pipe_name, nresults):
        ax.plot(nresults['Reference']['network_temperature'][pipe_name], label='Reference')
        ax.plot(nresults['Flexibility']['network_temperature'][pipe_name], label='Flexibility')
        ax.set_title(pipe_name)
        self.plot_price_increase_time(ax)
        fig.suptitle('Network temperature')
        ax.set_ylabel('Temperature [K]')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
        # fig.tight_layout()

        return ax

    def plot_plant_temperature(self, fig, ax, nresults, modelname):
        ax.plot(nresults['Reference']['plant_temperature']['supply'], linestyle=':', color='r')
        ax.plot(nresults['Reference']['plant_temperature']['return'], linestyle=':', color='g')
        ax.plot(nresults['Flexibility']['plant_temperature']['supply'], color='r')
        ax.plot(nresults['Flexibility']['plant_temperature']['return'], color='g')
        self.plot_price_increase_time(ax)
        fig.suptitle('Plant temperature')
        ax.set_ylabel('Temperature [K]')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
        ax.set_title(modelname)
        fig.tight_layout()

        return ax

    def plot_water_speed(self, fig, ax, pipe_name, nresults, ylabel=False):
        ax.plot(nresults['Flexibility']['water_velocity'][pipe_name])
        ax.set_title(pipe_name)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
        fig.suptitle('Water speed')
        if ylabel:
            ax.set_ylabel('Speed [m\s]')
        self.plot_price_increase_time(ax)
        # fig.tight_layout()

    def plot_building_temperatures(self, bparams, nresults):

        fig, axarr = plt.subplots(len(bparams), 2)

        for i, build in enumerate(bparams):
            axarr[i, 0].plot(bparams[build]['day_min_temperature'], linestyle=':', color='k')
            axarr[i, 1].plot(bparams[build]['night_min_temperature'], linestyle=':', color='k')
            axarr[i, 0].plot(bparams[build]['day_max_temperature'], linestyle=':', color='k')
            axarr[i, 1].plot(bparams[build]['night_max_temperature'], linestyle=':', color='k')

            axarr[i, 0].plot(nresults['day_zone_temperatures'][build])
            axarr[i, 1].plot(nresults['night_zone_temperatures'][build])
            axarr[i, 0].set_title(build)
            self.plot_price_increase_time(axarr[i, 0])
            self.plot_price_increase_time(axarr[i, 1])

        plt.show()

        return axarr

    def plot_heat_injection(self, fig1, fig2, ax1, ax2, nresults, modelcase, distcase, label=None):
        ax1.plot(nresults['Reference']['heat_injection'], label='Reference')
        ax1.plot(nresults['Flexibility']['heat_injection'], label='Flexibility')
        ax1.set_title(modelcase)
        self.plot_price_increase_time(ax1)

        if label is None:
            label = modelcase

        ax2.plot(nresults['Flexibility']['heat_injection'] -
                 nresults['Reference']['heat_injection'], label=label)
        self.plot_price_increase_time(ax2)

        fig1.suptitle('Heat injection')
        ax1.set_ylabel('Heat [W]')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=70)
        ax1.set_title(modelcase + ' ' + distcase)
        ax1.legend()
        fig1.tight_layout()

        fig2.suptitle('Step response')
        ax2.set_ylabel('Heat [W]')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=70)
        ax2.set_title(distcase)
        ax2.legend()
        fig2.tight_layout()

        return ax1, ax2

    def plot_combined_heat_injection(self, ax1, ax2, nresults_network, nresults_buildings):
        ax1.plot(nresults_buildings['Reference']['heat_injection'], label='Reference')
        ax1.plot(nresults_network['Flexibility']['heat_injection'], label='Flexibility')
        ax1.set_title('Combined - LP')
        self.plot_price_increase_time(ax1)

        # Resampling price profile to correct frequency
        resampled_b_data = nresults_buildings['Reference']['heat_injection'].resample(
            nresults_network['Flexibility']['heat_injection'].index.freq).pad()
        resampled_b_data = resampled_b_data.ix[
            ~(resampled_b_data.index > nresults_network['Flexibility']['heat_injection'].index[-1])]

        ax2.plot(nresults_network['Flexibility']['heat_injection'] -
                 resampled_b_data, label='Combined - LP')
        self.plot_price_increase_time(ax2)

        return ax1, ax2

    def plot_price_increase_time(self, ax):
        ax.axvline(x=self.find_price_increase_time(),
                   color='k', linestyle=':', linewidth=2)

    def find_price_increase_time(self):
        return self.time_index[int(self.get_flex_case('Flexibility').get_pos() * len(self.time_index))]

    def plot_response_functions(self, zoom=None):

        fig1, axarr1 = plt.subplots(len(self.district_cases), 1, sharex=True)
        fig2, axarr2 = plt.subplots(len(self.model_cases), len(self.district_cases), sharex=True)

        # Zoom case
        dist = self.get_district_case(zoom)
        pipe_names = dist.get_pipe_names()

        fig4, axarr4 = plt.subplots(1, 2, sharex=True)

        # n_pipes_1 = int(len(pipe_names)/2)
        # n_pipes_2 = len(pipe_names) - n_pipes_1
        #
        # fig5a, axarr5a = plt.subplots(n_pipes_1, 2, sharex=True)
        # fig5b, axarr5b = plt.subplots(n_pipes_2, 2, sharex=True)
        # fig3a, axarr3a = plt.subplots(n_pipes_1, 2, sharex=True)
        # fig3b, axarr3b = plt.subplots(n_pipes_2, 2, sharex=True)
        #
        # j = 0
        # for m, model in enumerate(self.model_cases):
        #     if model.is_node_method():
        #         results = self.results[zoom][model.name]
        #         self.plot_plant_temperature(fig4, axarr4[j], results, model.name)
        #
        #         for p, pipe in enumerate(pipe_names):
        #             if p == n_pipes_1 - 1 or p == len(pipe_names) - 1:
        #                 ylabel = True
        #             else:
        #                 ylabel = False
        #             if p < n_pipes_1:
        #                 self.plot_water_speed(fig5a, axarr5a[p, j], pipe, results, ylabel)
        #                 self.plot_network_temperatures(fig3a, axarr3a[p, j], pipe, results)
        #             else:
        #                 self.plot_water_speed(fig5b, axarr5b[p - n_pipes_1, j], pipe, results, ylabel)
        #                 self.plot_network_temperatures(fig3b, axarr3b[p - n_pipes_1, j], pipe, results)
        #         j += 1

        for l, dist in enumerate(self.district_cases):
            for m, model in enumerate(self.model_cases):
                results = self.results[dist.name][model.name]

                if len(self.district_cases) == 1:
                    self.plot_heat_injection(fig2, fig1, axarr2[m], axarr1, results, model.name, dist.name)
                else:
                    self.plot_heat_injection(fig2, fig1, axarr2[m, l], axarr1[l], results, model.name, dist.name)

        plt.show()

    def plot_sensitivity(self, sens_case, district_case):

        sens_values = sens_case.get_values()

        fig1, axarr1 = plt.subplots(len(self.model_cases), 1, sharex=True)
        fig2, axarr2 = plt.subplots(len(self.model_cases), len(sens_values), sharex=True)

        for l, value in enumerate(sens_values):
            for m, model in enumerate(self.model_cases):

                results = self.results[district_case.name][model.name][sens_case.name][value]

                if results['Reference'] is None or results['Flexibility'] is None:
                    print 'Case skipped: {}.{}.{}'.format(district_case.name, model.name, sens_case.name + ':' + str(value))
                else:
                    self.plot_heat_injection(fig1, fig2, axarr2[m, l], axarr1[m], results, model.name,
                                             district_case.name, label=value)
                if not model.is_node_method():
                    #self.plot_building_temperatures()
                    pass
                else:
                    fig3, ax3 = plt.subplots(1, 1)
                    self.plot_plant_temperature(fig3, ax3,results, model.name)

        for ax in axarr1:
            ax.set_title(model.name)

        plt.show()

    def generate_response_functions(self, tee=False):
        self.collect_data()
        self.results = {}
        n_cases = len(self.model_cases) * len(self.district_cases) * len(self.flex_cases)
        n = 1

        for dist in self.district_cases:
            self.results[dist.name] = {}

            for model in self.model_cases:
                self.results[dist.name][model.name] = {}
                for flex in self.flex_cases:

                    string = 'CASE ' + str(n) + ' of ' + str(
                        n_cases) + ': ' + dist.name + ' - ' + model.name + ' - ' + flex.name
                    print '\n', string, '\n', '-' * len(string), '\n'

                    self.set_up_modesto(dist, model, flex)

                    self.solve_optimization(tee=tee)
                    self.results[dist.name][model.name][flex.name] = self.collect_results(dist, model)

                    n += 1

        self.plot_response_functions(zoom=self.district_cases[0].name)
        self.save_obj(self.results, self.name)

    def run_sensitivity_analysis(self, tee=False):
        self.collect_data()
        self.results = {}
        n_cases = len(self.model_cases) * len(self.district_cases) * len(self.flex_cases) * \
                  sum(len(case.get_values()) for case in self.sens_cases)
        n = 1

        for dist in self.district_cases:
            self.results[dist.name] = {}

            for model in self.model_cases:
                self.results[dist.name][model.name] = {}

                for sens in self.sens_cases:
                    self.results[dist.name][model.name][sens.name] = {}
                    for value in sens.get_values():
                        self.results[dist.name][model.name][sens.name][value] = {}
                        for flex in self.flex_cases:
                            string = 'CASE ' + str(n) + ' of ' + str(
                                n_cases) + ': ' + dist.name + ' - ' + model.name + ' - ' + flex.name + \
                                ' - ' + sens.name + '-' + str(value)
                            print '\n', string, '\n', '-' * len(string), '\n'

                            self.set_up_modesto(dist, model, flex, sens_case=sens, value=value)

                            status = self.solve_optimization(tee=tee)
                            if status:
                                self.results[dist.name][model.name][sens.name][value][flex.name] = \
                                    self.collect_results(dist, model)
                            else:
                                self.results[dist.name][model.name][sens.name][value][flex.name] = None
                                print 'WARNING: This case did not converge'

                            n += 1

        self.save_obj(self.results, self.name)

        for sens in self.sens_cases:
            self.plot_sensitivity(sens, self.district_cases[0])

    def save_obj(self, obj, name):
        with open('results/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def get_district_case(self, name):
        for district in self.district_cases:
            if name == district.name:
                return district
        raise KeyError('{} is not a valid district name'.format(name))

    def get_flex_case(self, name):
        for flex in self.flex_cases:
            if name == flex.name:
                return flex
        raise KeyError('{} is not a valid district name'.format(name))

    def get_model_case(self, name):
        for model in self.model_cases:
            if name == model.name:
                return model
        raise KeyError('{} is not a valid district name'.format(name))


class ModelCase:

    def __init__(self, name):
        available_cases = ['Buildings - ideal network', 'Buildings', 'Network', 'Combined - LP']
        if name not in available_cases:
            raise Exception('{} is not a valid Model case, choose among {}'.format(name, available_cases))

        self.name = name

        self.time_steps = {'StSt': 900,
                           'Dynamic': 300}

        self.pipe_models = {'NoPipes': 'SimplePipe',
                       'StSt': 'ExtensivePipe',
                       'Dynamic': 'NodeMethod'}

        self.building_models = {'RC': 'RCmodel',
                           'Fixed': 'BuildingFixed'}

    def get_time_step(self):
        return self.data['time_step']

    def get_pipe_model(self):
        return self.data['pipe_model']

    def get_building_model(self):
        return self.data['building_model']

    def get_heat_profile(self):
        return self.data['heat_profile']

    def get_building_params(self):
        pass

    def get_producer_params(self):
        pass

    def get_dhw_params(self):
        pass

    def get_pipe_params(self):
        pass

    def is_node_method(self):
        return False

    def is_rc_model(self):
        return False

    def extra_horizon(self):
        if self.is_rc_model():
            return 2*self.get_time_step()
        else:
            return 0


class BuildingsIdealCase(ModelCase):

    def __init__(self):
        ModelCase.__init__(self, 'Buildings - ideal network')

        self.data = {'pipe_model': self.pipe_models['NoPipes'],
                     'time_step': self.time_steps['StSt'],
                     'building_model': self.building_models['RC'],
                     'heat_profile': None}

    def get_building_params(self):
        return ['delta_T', 'mult', 'night_min_temperature', 'night_max_temperature',
                      'day_min_temperature', 'day_max_temperature', 'bathroom_min_temperature',
                      'bathroom_max_temperature', 'floor_min_temperature', 'floor_max_temperature',
                      'model_type',
                      'Q_int_D', 'Q_int_N', 'TiD0', 'TflD0', 'TwiD0', 'TwD0', 'TfiD0',
                      'TfiN0', 'TiN0', 'TwiN0', 'TwN0', 'max_heat']

    def get_producer_params(self):
        return ['efficiency', 'PEF', 'CO2',
                'fuel_cost', 'Qmax', 'ramp_cost', 'ramp']

    def get_dhw_params(self):
        return ['delta_T', 'mult', 'heat_profile']

    def get_pipe_params(self):
       return ['diameter']

    def is_rc_model(self):
        return True


class BuildingCase(ModelCase):

    def __init__(self):
        ModelCase.__init__(self, 'Buildings')

        self.data = {
                    'pipe_model': self.pipe_models['StSt'],
                    'time_step': self.time_steps['StSt'],
                    'building_model': self.building_models['RC'],
                    'heat_profile': None
                    }

    def get_building_params(self):
        return ['delta_T', 'mult', 'night_min_temperature', 'night_max_temperature',
                'day_min_temperature', 'day_max_temperature', 'bathroom_min_temperature',
                'bathroom_max_temperature', 'floor_min_temperature', 'floor_max_temperature',
                'model_type',
                'Q_int_D', 'Q_int_N', 'TiD0', 'TflD0', 'TwiD0', 'TwD0', 'TfiD0',
                'TfiN0', 'TiN0', 'TwiN0', 'TwN0', 'max_heat']

    def get_producer_params(self):
        return ['efficiency', 'PEF', 'CO2',
                'fuel_cost', 'Qmax', 'ramp_cost', 'ramp']

    def get_dhw_params(self):
        return ['delta_T', 'mult', 'heat_profile']

    def get_pipe_params(self):
        return ['diameter', 'temperature_supply', 'temperature_return']

    def is_rc_model(self):
        return True


class NetworkCase(ModelCase):
    def __init__(self):
        ModelCase.__init__(self, 'Network')

        self.data = {
                    'pipe_model': self.pipe_models['Dynamic'],
                    'time_step': self.time_steps['Dynamic'],
                    'building_model': self.building_models['Fixed'],
                    'heat_profile': 'Reference'
            }

    def get_building_params(self):
        return ['delta_T', 'mult', 'temperature_return',
                'temperature_supply', 'temperature_max',
                'temperature_min', 'heat_profile']

    def get_producer_params(self):
        return ['fuel_cost', 'efficiency', 'PEF', 'CO2', 'Qmax', 'ramp_cost',
                'ramp', 'temperature_supply', 'temperature_return', 'temperature_max', 'temperature_min']

    def get_dhw_params(self):
        return ['delta_T', 'mult', 'heat_profile', 'temperature_return',
                'temperature_supply', 'temperature_max', 'temperature_min']

    def get_pipe_params(self):
        return ['diameter', 'temperature_history_supply', 'temperature_history_return', 'mass_flow_history',
                'wall_temperature_supply', 'wall_temperature_return', 'temperature_out_supply',
                'temperature_out_return']

    def is_node_method(self):
        return True


class CombinedLPCase(ModelCase):
    def __init__(self):
        ModelCase.__init__(self, 'Combined - LP')

        self.data = {
                'pipe_model': self.pipe_models['Dynamic'],
                'time_step': self.time_steps['Dynamic'],
                'building_model': self.building_models['Fixed'],
                'heat_profile': 'Flexibility'
            }

    def get_building_params(self):
        return ['delta_T', 'mult', 'temperature_return',
                'temperature_supply', 'temperature_max',
                'temperature_min', 'heat_profile']

    def get_producer_params(self):
        return ['fuel_cost', 'efficiency', 'PEF', 'CO2', 'Qmax', 'ramp_cost',
                'ramp', 'temperature_supply', 'temperature_return', 'temperature_max', 'temperature_min']

    def get_dhw_params(self):
        return ['delta_T', 'mult', 'heat_profile', 'temperature_return',
                'temperature_supply', 'temperature_max', 'temperature_min']

    def get_pipe_params(self):
        return ['diameter', 'temperature_history_supply', 'temperature_history_return', 'mass_flow_history',
                'wall_temperature_supply', 'wall_temperature_return', 'temperature_out_supply',
                'temperature_out_return']

    def is_node_method(self):
        return True

class FlexCase:

    def __init__(self, name):
        available_cases = ['Reference', 'Flexibility']
        if name not in available_cases:
            raise Exception('{} is not a valid Flexibility case, choose among {}'.format(name, available_cases))

        self.name = name

    def get_price_profile(self, time_index):
        pass


class ReferenceCase(FlexCase):

    def __init__(self):
        FlexCase.__init__(self, 'Reference')

    def get_price_profile(self, time_index):
        return pd.Series(1, index=time_index)


class FlexibilityCase(FlexCase):

    def __init__(self):
        FlexCase.__init__(self, 'Flexibility')
        self.pos = 3.5/7

    def get_price_profile(self, time_index):
        return pd.Series([1] * int(len(time_index) * self.pos) + [2] *
                         (len(time_index) - int(len(time_index) * self.pos)),
                         index=time_index)

    def change_pos(self, new_pos):
        self.pos = new_pos

    def get_pos(self):
        return self.pos


class DistrictCase:

    def __init__(self, name, building_names, building_mult, building_types, pipe_names,
                 pipe_diameters, topology, aggregated):
        self.name = name
        self.building_names = building_names
        self.n_buildings = len(building_names)
        self.building_types = building_types
        self.building_mult = building_mult

        self.changed_mult = copy.copy(building_mult)  #Used in case of sensitivity analyses that change the mult parameter

        self.pipe_names = pipe_names
        self.pipe_diameters = pipe_diameters

        self.aggregated = aggregated

        if topology not in ['street', 'parallel_district', 'series_district', 'Genk']:
            raise Exception('{} is not a valid district topology type'.format(topology))
        self.topology = topology

        self.graph = self.create_graph('RCmodel', False)

    def change_building_model(self, building_model):
        for node, data in self.graph.nodes(data=True):
            if 'building' in data['comps']:
                data['comps']['building'] = building_model

    def renew_graph(self, building_model, draw=False):
        self.graph = self.create_graph(building_model, draw)

    def create_graph(self, building_model, draw=False):
        pass

    def get_node_mult(self, node_nr):
        return self.changed_mult[node_nr]

    def get_building_type(self, node_nr):
        return self.building_types[node_nr]

    def get_pipe_diameter(self, pipe_name):
        return self.pipe_diameters[self.pipe_names.index(pipe_name)]

    def get_nr_of_nodes(self):
        return self.n_buildings

    def get_building_name(self, node_nr):
        return self.building_names[node_nr]

    def get_building_names(self):
        return self.building_names

    def get_pipe_names(self):
        return self.pipe_names

    def change_mult(self, factor, node_nr):
        self.changed_mult[node_nr] = int(factor*self.building_mult[node_nr])


class ParallelCase(DistrictCase):

    def __init__(self):

        n_streets = 3
        n_buildings = 10

        building_types = [old_building, mixed_building, new_building]
        building_names = ['Street' + str(i) for i in range(n_streets)]
        building_mult = [n_buildings] * n_streets
        pipe_names = ['dist_pipe' + str(i) for i in range(n_streets)]
        pipe_diameters = [50, 50, 50]

        DistrictCase.__init__(self, 'Parallel district', building_names, building_mult,
                              building_types, pipe_names, pipe_diameters, 'street', True)

    def create_graph(self, building_model, draw=True):
        """

        :param street_names:
        :param pipe_names:
        :param building_model:
        :param draw:
        :return:
        """

        dist_pipe_length = 150

        g = nx.DiGraph()

        g.add_node('Producer', x=0, y=0, z=0,
                   comps={'plant': 'ProducerVariable'})

        nr_streets = len(self.building_names)
        angle = 2*np.pi/nr_streets
        distance = dist_pipe_length

        for i in range(nr_streets):

            street_angle = i*angle
            x_coor = np.cos(street_angle)*distance
            y_coor = np.sin(street_angle)*distance
            g.add_node(self.building_names[i],  x=x_coor, y=y_coor, z=0,
                       comps={'building': building_model,
                              'DHW': 'BuildingFixed'})

            g.add_edge('Producer', self.building_names[i], name=self.pipe_names[i])

        if draw:

            coordinates = {}
            for node in g.nodes:
                coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

            fig = plt.figure()
            nx.draw(g, coordinates, with_labels=True, node_size=0)
            fig.savefig('img/parallel_district_layout.svg')
            plt.show()

        return g


class SeriesCase(DistrictCase):

    def __init__(self):

        n_streets = 3
        n_buildings = 10

        building_types = [old_building, mixed_building, new_building]
        building_names = ['Street' + str(i) for i in range(n_streets)]
        building_mult = [n_buildings] * n_streets
        pipe_names = ['dist_pipe' + str(i) for i in range(n_streets)]
        pipe_diameters = [80, 65, 50]

        DistrictCase.__init__(self, 'Series district', building_names, building_mult,
                              building_types, pipe_names, pipe_diameters, 'street', True)


    def create_graph(self, building_model, draw=True):
        """

        :param nr_streets:
        :param building_model:
        :param draw:
        :return:
        """

        dist_pipe_length = 150

        g = nx.DiGraph()

        g.add_node('Producer', x=0, y=0, z=0,
                   comps={'plant': 'ProducerVariable'})

        distance = dist_pipe_length
        nr_streets = len(self.building_names)

        for i in range(nr_streets):

            g.add_node(self.building_names[i],  x=distance*(i+1), y=0, z=0,
                       comps={'building': building_model,
                              'DHW': 'BuildingFixed'})

        g.add_edge('Producer', self.building_names[0], name=self.pipe_names[0])

        for i in range(nr_streets-1):
            g.add_edge(self.building_names[i], self.building_names[i+1], name=self.pipe_names[i+1])

        if draw:

            coordinates = {}
            for node in g.nodes:
                coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

            fig = plt.figure()
            nx.draw(g, coordinates, with_labels=True, node_size=0)
            fig.savefig('img/series_district_layout.svg')
            plt.show()

        return g


class GenkCase(DistrictCase):

    def __init__(self):

        building_types = [old_building]*9
        building_names = ['TermienWest', 'TermienOost', 'Boxbergheide', 'Winterslag', 'OudWinterslag',
                          'ZwartbergNW', 'ZwartbergZ', 'ZwartbergNE', 'WaterscheiGarden']
        building_mult = [633, 746, 2363, 1789, 414, 567, 1571, 584, 2094]
        pipe_names = ['dist_pipe' + str(i) for i in range(14)]
        pipe_diameters = [800, 250, 450, 800, 250, 400, 700, 250, 700, 450, 400, 400, 300, 300]

        DistrictCase.__init__(self, 'Genk', building_names, building_mult,
                              building_types, pipe_names, pipe_diameters, 'street', True)

    def create_graph(self, building_model, draw=True):
        """

        :param building_model:
        :param draw:
        :return:
        """

        g = nx.DiGraph()

        g.add_node('Producer', x=5000, y=5000, z=0,
                   comps={'plant': 'ProducerVariable'})
        g.add_node('WaterscheiGarden', x=3500, y=5100, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('ZwartbergNE', x=3300, y=6700, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('ZwartbergNW', x=1500, y=6600, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('ZwartbergZ', x=2000, y=6000, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('OudWinterslag', x=1700, y=4000, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('Winterslag', x=1000, y=2500, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('Boxbergheide', x=-1200, y=2100, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('TermienOost', x=800, y=880, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('TermienWest', x=0, y=0, z=0,
                   comps={'building': building_model,
                          'DHW': 'BuildingFixed'})
        g.add_node('p1', x=3500, y=6100, z=0,
                   comps={})
        g.add_node('p2', x=1700, y=6300, z=0,
                   comps={})
        g.add_node('p3', x=250, y=5200, z=0,
                   comps={})
        g.add_node('p4', x=0, y=2700, z=0,
                   comps={})
        g.add_node('p5', x=620, y=700, z=0,
                   comps={})

        g.add_edge('Producer', 'p1', name='dist_pipe0')
        g.add_edge('p1', 'ZwartbergNE', name='dist_pipe1')
        g.add_edge('p1', 'WaterscheiGarden', name='dist_pipe2')
        g.add_edge('p1', 'p2', name='dist_pipe3')
        g.add_edge('p2', 'ZwartbergNW', name='dist_pipe4')
        g.add_edge('p2', 'ZwartbergZ', name='dist_pipe5')
        g.add_edge('p2', 'p3', name='dist_pipe6')
        g.add_edge('p3', 'OudWinterslag', name='dist_pipe7')
        g.add_edge('p3', 'p4', name='dist_pipe8')
        g.add_edge('p4', 'Boxbergheide', name='dist_pipe9')
        g.add_edge('p4', 'Winterslag', name='dist_pipe10')
        g.add_edge('p4', 'p5', name='dist_pipe11')
        g.add_edge('p5', 'TermienOost', name='dist_pipe12')
        g.add_edge('p5', 'TermienWest', name='dist_pipe13')

        if draw:

            coordinates = {}
            for node in g.nodes:
                coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

            fig = plt.figure()
            nx.draw(g, coordinates, with_labels=True, node_size=0)
            fig.savefig('img/genk_layout.svg')
            plt.show()

        return g


class StreetCase(DistrictCase):

    def __init__(self, name, n_buildings, building_types):

        building_names = ['Building' + str(i) for i in range(n_buildings)]
        building_mult = [1] * n_buildings
        pipe_names = ['dist_pipe' + str(i) for i in range(int(math.ceil(n_buildings/2)))] + \
                     ['serv_pipe' + str(i) for i in range(n_buildings)]
        pipe_diameters = [50, 40, 32, 32, 25] + [20] * n_buildings

        DistrictCase.__init__(self, name, building_names, building_mult,
                              building_types, pipe_names, pipe_diameters, 'street', False)

    def create_graph(self, building_model, draw=False):
        """
        :param building_model:
        :param draw:
        :return:
        """

        street_pipe_length = 30
        service_pipe_length = 30

        g = nx.DiGraph()

        g.add_node('Producer', x=0, y=0, z=0,
                   comps={'plant': 'ProducerVariable'})

        nr_buildings = len(self.building_names)
        n_points = int(math.ceil(nr_buildings / 2))

        for i in range(n_points):
            g.add_node('p' + str(i), x=street_pipe_length * (i + 1), y=0, z=0,
                       comps={})
            g.add_node(self.building_names[i], x=street_pipe_length * (i + 1), y=service_pipe_length, z=0,
                       comps={'building': building_model,
                              'DHW': 'BuildingFixed'})

            if n_points + i + 1 <= nr_buildings:
                g.add_node(self.building_names[n_points + i], x=street_pipe_length * (i + 1), y=-service_pipe_length,
                           z=0,
                           comps={'building': building_model,
                                  'DHW': 'BuildingFixed'})

        k = 1
        g.add_edge('Producer', 'p0', name=self.pipe_names[0])
        for i in range(n_points - 1):
            g.add_edge('p' + str(i), 'p' + str(i + 1), name=self.pipe_names[k])
            k += 1

        for i in range(n_points):
            g.add_edge('p' + str(i), 'Building' + str(i), name=self.pipe_names[k])

            if n_points + i + 1 <= nr_buildings:
                g.add_edge('p' + str(i), 'Building' + str(n_points + i), name=self.pipe_names[n_points + k])

            k += 1

        if draw:

            coordinates = {}
            for node in g.nodes:
                coordinates[node] = (g.nodes[node]['x'], g.nodes[node]['y'])

            fig = plt.figure()
            nx.draw(g, coordinates, with_labels=True, node_size=0)
            fig.savefig('img/street_layout.svg')
            plt.show()

        return g


class MixedStreetCase(StreetCase):

    def __init__(self):
        n_buildings = 10
        building_types = [new_building, old_building]*int(n_buildings/2)
        StreetCase.__init__(self, 'Mixed street', n_buildings, building_types)


class OldStreetCase(StreetCase):

    def __init__(self):

        n_buildings = 10
        building_types = [old_building]*n_buildings
        StreetCase.__init__(self, 'Old street', n_buildings, building_types)


class NewStreetCase(StreetCase):

    def __init__(self):

        n_buildings = 10
        building_types = [new_building]*n_buildings
        StreetCase.__init__(self, 'New street', n_buildings, building_types)


class SensitivityCase:

    def __init__(self, name, values, pipe_length=False,
                 pipe_diameter_discrete=False, pipe_diameter_cont=False , heat_demand=False,
                 supply_temp_level=False, supply_temp_reach=False,
                 substation_temp_difference=False):

        self.name = name
        self.values = values

        self.pipe_length = pipe_length
        self.pipe_diameter_cont = pipe_diameter_cont
        self.pipe_diameter_discrete = pipe_diameter_discrete
        self.heat_demand = heat_demand
        self.supply_temp_level = supply_temp_level
        self.supply_temp_reach = supply_temp_reach
        self.substation_temp_difference = substation_temp_difference

    def get_values(self):
        return self.values

    def change_parameters(self, value, build_params, prod_params, dhw_params, pipe_params, model_case, district_case):
        if self.pipe_length:
            self.change_pipe_length(value, pipe_params)
        if self.pipe_diameter_cont:
            self.change_pipe_diameter_cont(value, pipe_params, district_case)
        if self.pipe_diameter_discrete:
            self.change_pipe_diameter_discrete(value, pipe_params)
        if self.heat_demand:
            self.change_heat_demand(value, build_params, dhw_params, model_case, district_case)
        if self.supply_temp_level:
            self.change_supply_temp_level(value, build_params, prod_params, dhw_params, pipe_params)
        if self.supply_temp_reach:
            self.change_supply_temp_reach(value, prod_params)
        if self.substation_temp_difference:
            self.change_substation_temp_difference(value, build_params, prod_params, dhw_params, pipe_params)
            self.change_substation_temp_difference(value, build_params, prod_params, dhw_params, pipe_params)

        return build_params, prod_params, dhw_params, pipe_params

    def change_pipe_diameter_cont(self, value, pipe_params, district_case):

        mult_factor = []
        nodes = district_case.get_building_names()
        for i, node in enumerate(nodes):
            mult_factor.append(district_case.get_node_mult(i) * value)

        diams = self.size(district_case.graph, nodes, mult_factor, 30000, delta_T, 'Producer')

        for p in pipe_params:
            pipe_params[p]['diameter'] = diams[p]

        return pipe_params

    def find_diameter(self, diam, new_pos):
        """
        Find the new diameter of pipe


        :param diam: Current diameter
        :param new_pos: Number of positions the diameter has to become.
                e.g. -1 means one size smaller
        :return:
        """

        diam_list = [20, 25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250,
                     300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]

        # find current position in list
        try:
            pos = diam_list.index(diam)
        except:
            raise ValueError('{} is not an existing diameter'.format(pos))

        new_pos += pos

        try:
            return diam_list[new_pos]
        except:
            print 'Warning: There is no diameter at the new position {}, ' \
                  'the original diameter is returned instead'.format(new_pos)
            return diam
        # TODO Skip case if diameter not found

    def change_pipe_length(self, value, pipe_params):
        for p in pipe_params:
            pipe_params[p]['length_scale_factor'] = value

        return pipe_params

    def change_pipe_diameter_discrete(self, value, pipe_params):
        for p in pipe_params:
            pipe_params[p]['diameter'] = self.find_diameter(pipe_params[p]['diameter'], value)

        return pipe_params

    def change_heat_demand(self, value, build_params, dhw_params, model_case, district_case):
        pass
        # if model_case.is_rc_model():
        #     for i, b in enumerate(build_params):
        #         # if bparams[b]['mult'] is not None:
        #         build_params[b]['mult'] = int(value * build_params[b]['mult'])
        #
        #     for i in range(district_case.get_nr_of_nodes()):
        #         district_case.change_mult(value, i)
        #
        # for dhw in dhw_params:
        #     dhw_params[dhw]['mult'] = int(value * dhw_params[dhw]['mult'])

        return build_params, dhw_params

    def change_supply_temp_level(self, value, build_params, prod_params, dhw_params, pipe_params):
        for b in build_params:
            build_params[b]['temperature_supply'] = value

        prod_params['temperature_supply'] = value
        prod_params['temperature_max'] = value + delta_T_prod
        prod_params['temperature_min'] = value

        for p in pipe_params:
            pipe_params[p]['temperature_supply'] = value
            pipe_params[p]['temperature_history_supply'] = pd.Series(value, index=range(10))
            pipe_params[p]['wall_temperature_supply'] = value
            pipe_params[p]['temperature_out_supply'] = value
        for d in dhw_params:
            dhw_params[d]['temperature_supply'] = value
            dhw_params[d]['temperature_max'] = value + delta_T
            dhw_params[d]['temperature_min'] = value - delta_T

        return build_params, prod_params, dhw_params, pipe_params

    def change_supply_temp_reach(self, value, prod_params):
        prod_params['temperature_max'] += -delta_T_prod + value

        return prod_params

    def change_substation_temp_difference(self, value, build_params, prod_params, dhw_params, pipe_params):
        for b in build_params:
            supply_temp = build_params[b]['temperature_supply']
            build_params[b]['delta_T'] = value
            build_params[b]['temperature_return'] = supply_temp - value

        supply_temp = prod_params['temperature_supply']
        prod_params['temperature_return'] = supply_temp - value

        for p in pipe_params:
            supply_temp = pipe_params[p]['temperature_supply']
            pipe_params[p]['temperature_return'] = supply_temp - value
            pipe_params[p]['temperature_history_return'] = pd.Series(supply_temp - value, index=range(10))
            pipe_params[p]['wall_temperature_return'] = supply_temp - value
            pipe_params[p]['temperature_out_return'] = supply_temp - value
        for d in dhw_params:
            supply_temp = dhw_params[d]['temperature_supply']
            dhw_params[d]['delta_T'] = value
            dhw_params[d]['temperature_return'] = supply_temp - value

        return build_params, prod_params, dhw_params, pipe_params

    def size(self, graph, neighs, mult, qnom, delta_t, producer_node):

        """

        :param graph:
        :param mult:
        :param qnom:
        :param delta_t:
        :return:
        """

        """
        Find mass flow rates through network
        """

        nodes = list(graph.nodes)
        tuples = list(graph.edges)
        dict = nx.get_edge_attributes(graph, 'name')
        edges = []
        for tuple in tuples:
            edges.append(dict[tuple])

        inc_matrix = -nx.incidence_matrix(graph, oriented=True).todense()

        # Remove the producer node and the corresponding row from the matrix to make the system determined

        row_nr = nodes.index(producer_node)
        del nodes[row_nr]
        matrix = np.delete(inc_matrix, row_nr, 0)

        vector = []

        # Collect known mass flow rates at nodes
        for node in nodes:
            if node in neighs:
                mf_node = mult[neighs.index(node)] * qnom / delta_t / 4186
            else:  # nodes that have no components are not included in neughs list
                mf_node = 0
            vector.append(mf_node)

        sol = np.linalg.solve(matrix, vector)

        edge_mf = {}
        for i, edge in enumerate(edges):
            edge_mf[edge] = abs(sol[i] / 1000 * 3600)

        """
        Select diameters
        """

        vflomax = collections.OrderedDict()  # Maximal volume flow rate per DN in m3/h
        # Taken from IsoPlus Double-Pipe catalog p. 7
        vflomax[20] = 1.547,
        vflomax[25] = 2.526,
        vflomax[32] = 4.695,
        vflomax[40] = 6.303,
        vflomax[50] = 11.757,
        vflomax[65] = 19.563,
        vflomax[80] = 30.791,
        vflomax[100] = 51.891,
        vflomax[125] = 89.350,
        vflomax[150] = 152.573,
        vflomax[200] = 299.541,
        vflomax[250] = 348 * 1.55,
        vflomax[300] = 547 * 1.55,
        vflomax[350] = 705 * 1.55,
        vflomax[400] = 1550,
        vflomax[450] = 1370 * 1.55,
        vflomax[500] = 1820 * 1.55,
        vflomax[600] = 2920 * 1.55,
        vflomax[700] = 4370 * 1.55,
        vflomax[800] = 6240 * 1.55,
        vflomax[900] = 9500 * 1.55,
        vflomax[1000] = 14000 * 1.55

        edge_diam = {}
        for edge in edge_mf:
            for diam, vflo in vflomax.items():
                edge_diam[edge] = diam
                if vflo > edge_mf[edge]:
                    break

        return edge_diam


class NetworkSize(SensitivityCase):

    def __init__(self):
        name = 'Network size'
        values = [0.001, 0.01]  # , 0.01
        SensitivityCase.__init__(self, name, values, pipe_length=True,
                                 pipe_diameter_cont=True, heat_demand=True)


class PipeLength(SensitivityCase):

    def __init__(self):
        name = 'Pipe length'
        values = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 10] # 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 10
        SensitivityCase.__init__(self, name, values, pipe_length=True)


class PipeDiameter(SensitivityCase):

    def __init__(self):
        name = 'Pipe diameter'
        values = [1, -1, 0,] #
        SensitivityCase.__init__(self, name, values, pipe_diameter_discrete=True)


class HeatDemand(SensitivityCase):

    def __init__(self):
        name = 'Heat demand'
        values = [0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 2]  #
        SensitivityCase.__init__(self, name, values, heat_demand=True)


class SupplyTempLevel(SensitivityCase):

    def __init__(self):
        name = 'Supply temperature level'
        values = [i + 273.15 for i in [50, 60]]  # , 70, 80, 90
        SensitivityCase.__init__(self, name, values, supply_temp_level=False)


class SupplyTempReach(SensitivityCase):

    def __init__(self):
        name = 'Supply temperature reach'
        values = [5, 7.5, 10, 12.5, 15, 17.5, 20] #
        SensitivityCase.__init__(self, name, values, supply_temp_reach=False)


class SubstationTempDiff(SensitivityCase):

    def __init__(self):
        name = 'Substation temperature difference'
        values = [15, 20, 25, 30, 35, 40] #
        SensitivityCase.__init__(self, name, values, substation_temp_difference=False)

# TODO Add fail-safe if optimization fails!


class DataReader:

    def __init__(self, horizon, time_step, start_time):
        self.day_min_df = None
        self.night_min_df = None
        self.QCon_df = None
        self.QRad_df = None
        self.m_DHW_df = None
        self.day_min = {}
        self.night_min = {}
        self.tamb = None
        self.t_g = None
        self.QsolN = None
        self.QsolE = None
        self.QsolS = None
        self.QsolW = None

        self.horizon = horizon
        self.time_step = time_step
        self.start_time = start_time
        self.create_time_index(start_time, horizon+10*time_step, time_step)

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.modesto_path = os.path.dirname(os.path.dirname(self.dir_path))
        self.data_path = os.path.join(self.modesto_path, 'modesto', 'Data')

    # TODO Take a look at these methods
    def aggregate_ISO13790(self, name, street_nr, n_buildings):
        df = pd.DataFrame(index=self.time_index, columns=range(n_buildings))
        for i in range(n_buildings):
            profile_nr = street_nr * n_buildings + i
            df[i] = ut.read_time_data(self.get_data_path('UserBehaviour\ISO1370_statistic'),
                                      name='ISO13790_stat_profile' + str(profile_nr) + '.csv')[name]
        return ut.aggregate_columns(df)

    def aggregate_StROBe(self, data, start_building, n_buildings):
        df = data.ix[:, n_buildings * start_building: n_buildings * start_building + n_buildings]
        return ut.aggregate_columns(df)

    def aggregate_min_temp(self, name, building_model, start_building, n_buildings):
        if name is 'day':
            df = self.day_min[building_model].ix[:,
                 n_buildings * start_building: n_buildings * start_building + n_buildings]
        if name is 'night':
            df = self.night_min[building_model].ix[:,
                 n_buildings * start_building: n_buildings * start_building + n_buildings]

        df.index = self.time_index[0: len(df.index)]
        return ut.aggregate_columns(df)

    def get_data_path(self, subfolder):
        return os.path.join(self.data_path, subfolder)

    def create_time_index(self, start_time, horizon, time_step):
        self.time_index = pd.date_range(start=start_time, periods=int(horizon /time_step) + 1, freq=str(time_step) + 'S')

    def read_data(self, building_types):
        self.day_min_df = ut.read_period_data(path=self.get_data_path('UserBehaviour\Strobe_profiles'),
                                              name='sh_day.csv',
                                              horizon=self.horizon,
                                              time_step=self.time_step,
                                              start_time=self.start_time) + 273.15

        self.night_min_df = ut.read_period_data(path=self.get_data_path('UserBehaviour\Strobe_profiles'),
                                                name='sh_night.csv',
                                                horizon=self.horizon,
                                                time_step=self.time_step,
                                                start_time=self.start_time) + 273.15

        self.QCon_df = ut.read_period_data(self.get_data_path('UserBehaviour\Strobe_profiles'),
                                           name='QCon.csv',
                                           horizon=self.horizon,
                                           time_step=self.time_step,
                                           start_time=self.start_time)

        self.QRad_df = ut.read_period_data(self.get_data_path('UserBehaviour\Strobe_profiles'),
                                           name='QRad.csv',
                                           horizon=self.horizon,
                                           time_step=self.time_step,
                                           start_time=self.start_time)

        self.m_DHW_df = ut.read_period_data(self.get_data_path('UserBehaviour\Strobe_profiles'),
                                            name='mDHW.csv',
                                            horizon=self.horizon,
                                            time_step=self.time_step,
                                            start_time=self.start_time)

        for building_type in building_types:
            self.day_min[building_type] = ut.read_period_data(
                os.path.join(self.modesto_path, 'misc', 'aggregation_methods'),
                name='day_t_' + building_type + '.csv',
                horizon=self.horizon,
                time_step=self.time_step,
                start_time=self.start_time)

            self.night_min[building_type] = ut.read_period_data(
                os.path.join(self.modesto_path, 'misc', 'aggregation_methods'),
                name='night_t_' + building_type + '.csv',
                horizon=self.horizon,
                time_step=self.time_step,
                start_time=self.start_time)

        self.t_amb = ut.read_period_data(self.get_data_path('Weather'), name='weatherData.csv',
                                         horizon=self.horizon, time_step=self.time_step,
                                         start_time=self.start_time)['Te']
        self.t_g = ut.read_period_data(self.get_data_path('Weather'), name='weatherData.csv',
                                       horizon=self.horizon, time_step=self.time_step,
                                       start_time=self.start_time)['Tg']
        self.QsolN = ut.read_period_data(self.get_data_path('Weather'), name='weatherData.csv',
                                         horizon=self.horizon, time_step=self.time_step,
                                         start_time=self.start_time)['QsolN']
        self.QsolE = ut.read_period_data(self.get_data_path('Weather'), name='weatherData.csv',
                                         horizon=self.horizon, time_step=self.time_step,
                                         start_time=self.start_time)['QsolS']
        self.QsolS = ut.read_period_data(self.get_data_path('Weather'), name='weatherData.csv',
                                         horizon=self.horizon, time_step=self.time_step,
                                         start_time=self.start_time)['QsolN']
        self.QsolW = ut.read_period_data(self.get_data_path('Weather'), name='weatherData.csv',
                                         horizon=self.horizon, time_step=self.time_step,
                                         start_time=self.start_time)['QsolW']

    def get_user_behaviour(self, aggregated, mult, building_number=None, building_type=None):
        output = {}

        if aggregated:

            if building_type is None:
                raise Exception('In case aggregated user behaviour is required, two extra inputs -district_number- and '
                                '-building_type- are required')

            building_number = 0
            mult = 50

            # TODO Improve city user behaviour: different profiles for different city districts

            output['day_min_temperature'] = self.aggregate_min_temp('day', building_type, building_number, mult)
            output['night_min_temperature'] = self.aggregate_min_temp('night', building_type, building_number, mult)
            QCon = self.aggregate_StROBe(self.QCon_df, building_number, mult)
            QRad = self.aggregate_StROBe(self.QRad_df, building_number, mult)

        else:
            output['day_min_temperature'] = self.day_min_df.ix[:, building_number]
            output['night_min_temperature'] = self.night_min_df.ix[:, building_number]
            QCon = self.QCon_df.ix[:, building_number]
            QRad = self.QRad_df.ix[:, building_number]

        output['Q_int_D'] = (QCon + QRad) * 0.5
        output['Q_int_N'] = (QCon + QRad) * 0.5

        output['day_max_temperature'] = pd.Series(max(output['day_min_temperature']) + 1, index=self.time_index)
        output['night_max_temperature'] = pd.Series(max(max(output['day_min_temperature']) - 3,
                                  max(output['night_min_temperature']) + 1),
                              index=self.time_index)
        output['bathroom_max_temperature'] = ut.read_time_data(self.get_data_path('UserBehaviour\ISO1370_statistic'),
                                         name='ISO13790_stat_profile' + str(building_number) + '.csv')['bathroom_max']
        output['bathroom_min_temperature'] = ut.read_time_data(self.get_data_path('UserBehaviour\ISO1370_statistic'),
                                         name='ISO13790_stat_profile' + str(building_number) + '.csv')['bathroom_min']
        output['floor_max_temperature'] = ut.read_time_data(self.get_data_path('UserBehaviour\ISO1370_statistic'),
                                      name='ISO13790_stat_profile' + str(building_number) + '.csv')['floor_max']
        output['floor_min_temperature'] = ut.read_time_data(self.get_data_path('UserBehaviour\ISO1370_statistic'),
                                      name='ISO13790_stat_profile' + str(building_number) + '.csv')['floor_min']

        day_states = ['TiD0', 'TflD0', 'TwiD0', 'TwD0', 'TfiD0']
        night_states = ['TfiN0', 'TiN0', 'TwiN0', 'TwN0']

        for state in day_states:
            output[state] = max(output['day_min_temperature'])
        for state in night_states:
            output[state] = max(output['night_min_temperature'])

        return output

    def get_dhw_use(self, aggregated, mult, building_number=None, return_heat_profile=True):
        if aggregated:
            # TODO Improve this agregation
            building_number = 0
            mult = 50

            mass_flow = self.aggregate_StROBe(self.m_DHW_df, building_number, mult) / 60 / (38-10) * delta_T
            heat_profile = mass_flow * 4186 * delta_T

        else:
            mass_flow = self.m_DHW_df.iloc[:, building_number] / 60 / (38-10) * delta_T
            heat_profile = mass_flow * 4186 * delta_T

        if return_heat_profile:
            return heat_profile
        else:
            return mass_flow












