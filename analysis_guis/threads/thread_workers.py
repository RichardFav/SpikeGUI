# module import
import gc
import os
import copy
import random
import platform
import numpy as np
import pickle as p
import pandas as pd
import multiprocessing as mp
from numpy.matlib import repmat

# scipy module imports
from scipy.stats import norm, linregress
from scipy.spatial.distance import *
from scipy.interpolate import PchipInterpolator as pchip
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import periodogram, hamming, boxcar, find_peaks

# sklearn module imports
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# statsmodels module imports
from statsmodels.nonparametric.smoothers_lowess import lowess

# pyqt5 module import
from PyQt5.QtCore import QThread, pyqtSignal

# custom module imports
import analysis_guis.common_func as cf
import analysis_guis.calc_functions as cfcn
import analysis_guis.rotational_analysis as rot
from analysis_guis.dialogs.rotation_filter import RotationFilteredData
from analysis_guis.cluster_read import ClusterRead
from probez.spike_handling import spike_io

# other parameters
dcopy = copy.deepcopy
default_dir_file = os.path.join(os.getcwd(), 'default_dir.p')
interp_arr = lambda xi, y: np.vstack([interp1d(np.linspace(0, 1, len(x)), x, kind='nearest')(xi) for x in y])
cell_perm_ind = lambda n_cell_tot, n_cell: np.sort(np.random.permutation(n_cell_tot)[:n_cell])
set_sf_cell_perm = lambda spd_sf, n_pool, n_cell: [x[:, :, cell_perm_ind(n_pool, n_cell)] for x in spd_sf]
grp_expt_indices = lambda i_expt0: [np.where(i_expt0 == i)[0] for i in np.unique(i_expt0)]

# lambda function declarations
lin_func = lambda x, a: a * x

########################################################################################################################
########################################################################################################################


class WorkerThread(QThread):
    # creates the signal object
    work_started = pyqtSignal()
    work_progress = pyqtSignal(str, float)
    work_finished = pyqtSignal(object)
    work_error = pyqtSignal(str, str)
    work_plot = pyqtSignal(object)

    def __init__(self, parent=None, main_gui=None):
        # creates the worker object
        super(WorkerThread, self).__init__(parent)

        self.update_pbar = True
        self.is_running = False
        self.forced_quit = False
        self.sub_job = None
        self.is_ok = True
        self.data = None

        # other initialisations
        self.main_gui = main_gui
        self.thread_job_primary = None
        self.thread_job_secondary = None
        self.thread_job_para = None

    def set_worker_func_type(self, thread_job_primary, thread_job_secondary=None, thread_job_para=None):
        '''

        :param func_type:
        :return:
        '''

        # updates the worker primary/secondary job type and parameters
        self.thread_job_primary = thread_job_primary
        self.thread_job_secondary = thread_job_secondary
        self.thread_job_para = thread_job_para

    def run(self):
        '''

        :return:
        '''

        # initialisations
        w_prog, w_err = self.work_progress, self.work_error

        # updates the running/forced quit flagsv
        self.is_running = True
        self.forced_quit = False
        self.is_ok = True

        # updates the running parameter and enables the progress group parameters
        self.work_started.emit()

        # runs the job based on the type
        thread_data = None
        if self.thread_job_primary == 'init_data_file':
            # case is initialising the data file
            self.init_cluster_data()

        elif self.thread_job_primary == 'init_pool_object':
            # case is initialising the pool worker object
            thread_data = self.init_pool_worker()

        ##################################
        ####    DATA I/O FUNCTIONS    ####
        ##################################

        elif self.thread_job_primary == 'load_data_files':
            # case is loading the data files
            thread_data = self.load_data_file()

        elif self.thread_job_primary == 'save_multi_expt_file':
            # retrieves the parameters
            data, out_info = self.thread_job_para[0], self.thread_job_para[1]

            # case is loading the data files
            thread_data = self.save_multi_expt_file(data, out_info)

        elif self.thread_job_primary == 'save_multi_comp_file':
            # retrieves the parameters
            data, out_info = self.thread_job_para[0], self.thread_job_para[1]

            # case is loading the data files
            thread_data = self.save_multi_comp_file(data, out_info)

        elif self.thread_job_primary == 'run_calc_func':
            # case is the calculation functions
            calc_para, plot_para = self.thread_job_para[0], self.thread_job_para[1]
            data, pool, g_para = self.thread_job_para[2], self.thread_job_para[3], self.thread_job_para[4]

            ################################################
            ####    CLUSTER CLASSIFICATION FUNCTIONS    ####
            ################################################

            if self.thread_job_secondary == 'Fixed/Free Cluster Matching':

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['clust'])

                # case is determining the cluster matches
                self.det_cluster_matches(data, calc_para, w_prog)

            elif self.thread_job_secondary == 'Cluster Cross-Correlogram':
                # case is the cc-gram type determinations
                thread_data = self.calc_ccgram_types(calc_para, data.cluster)

            ######################################
            ####    AHV ANALYSIS FUNCTIONS    ####
            ######################################

            elif ' (Fixed)' in self.thread_job_secondary or \
                                            (self.thread_job_secondary == 'Correlation Significance Overlap'):

                # ensures the smoothing window is an odd integer (if smoothing)
                if calc_para['is_smooth']:
                    if calc_para['n_smooth'] % 2 != 1:
                        # if not, then output an error message to screen
                        e_str = 'The median smoothing filter window span must be an odd integer.'
                        w_err.emit(e_str, 'Incorrect Smoothing Window Span')

                        # sets the error flag and exits the function
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                # initialises the rotation filter class object (if not already set)
                if plot_para['rot_filt'] is None:
                    plot_para['rot_filt'] = cf.init_rotation_filter_data(False)

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['vel', 'vel_sf_fix'], other_para=False)

                # calculates the shuffled kinematic spiking frequencies
                cfcn.calc_binned_kinemetic_spike_freq(data, plot_para, dcopy(calc_para), w_prog, roc_calc=False)
                cfcn.calc_shuffled_kinematic_spike_freq(data, dcopy(calc_para), w_prog)

                # runs any specific additional function
                fit_func = ['Correlation Comparison (Fixed)',
                            'Correlation Fit Parameters (Fixed)',
                            'Individual Cell Correlation (Fixed)']
                if self.thread_job_secondary in fit_func:
                    # case is the correlation fit parameters
                    self.calc_corr_fit_para(data, plot_para, dcopy(calc_para), w_prog)

            elif (' (Freely Moving)' in self.thread_job_secondary):
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['vel_sf_free'], other_para=False)

                # updates the bin velocity
                data.rotation.vel_bin_corr = calc_para['vel_bin']

            elif 'Fixed/Free Spiking Correlation' in self.thread_job_secondary:

                # determines if the freely moving data file has been loaded
                if not hasattr(data.externd, 'free_data'):
                    # if the data-file has not been loaded then output an error to screen and exit
                    e_str = 'The freely moving spiking frequency/statistics data file must be loaded ' \
                            'before being able to run this function.\n\nPlease load this data file and try again.'
                    w_err.emit(e_str, 'Freely Moving Data Missing?')

                    # exits the function with an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['ff_corr', 'vel'], other_para=False)

                # calculates the shuffled kinematic spiking frequencies
                cfcn.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para, w_prog, roc_calc=False, use_raw=True)

                # calculates the fixed/free correlations (if not already set)
                if not data.comp.ff_corr.is_set:
                    self.calc_fix_free_correlation(data, calc_para, w_prog)

            ################################################
            ####    FREELY MOVING ANALYSIS FUNCTIONS    ####
            ################################################

            elif self.thread_job_secondary == 'Freely Moving Cell Fit Residual':

                # ensures the calculation fields are
                self.calc_cell_fit_residual(data, calc_para, w_prog)

            ######################################
            ####    EYE TRACKING FUNCTIONS    ####
            ######################################

            elif self.thread_job_secondary in ['Eye Movement Event Signals']:

                # check to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['eye_track'])

                # calculates the eye-tracking metrics (if not calculated)
                if len(data.externd.eye_track.t_evnt) == 0:
                    self.calc_eye_track_metrics(data, calc_para, w_prog)

            elif 'Eye Movement Correlation' in self.thread_job_secondary:

                # check to see if any parameters have been altered/
                self.check_altered_para(data, calc_para, plot_para, g_para, ['eye_track'])

                # calculates the eye-tracking metrics (if not calculated)
                if len(data.externd.eye_track.t_evnt) == 0:
                    self.calc_eye_track_metrics(data, calc_para, w_prog)

                # calculates the eye-tracking metrics
                if len(data.externd.eye_track.t_sp_h) == 0:
                    self.calc_eye_track_corr(data, calc_para, w_prog)

            ######################################
            ####    ROC ANALYSIS FUNCTIONS    ####
            ######################################

            elif self.thread_job_secondary == 'Direction ROC Curves (Single Cell)':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['condition'])

                # case is the shuffled cluster distances
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, False, 100.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

            elif self.thread_job_secondary == 'Direction ROC Curves (Whole Experiment)':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['condition', 'phase'])

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, False, 33.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, calc_para, 66.)
                self.calc_phase_roc_significance(calc_para, g_para, data, pool, 100.)

            elif self.thread_job_secondary in ['Direction ROC AUC Histograms',
                                               'Direction ROC Spiking Rate Heatmap']:
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['condition'])

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, True, 100., True):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

            elif 'Velocity ROC Curves' in self.thread_job_secondary:
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['vel'], other_para=True)

                # calculates the binned kinematic spike frequencies
                cfcn.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para, w_prog)
                self.calc_kinematic_roc_curves(data, pool, calc_para, g_para, 50.)

            elif self.thread_job_secondary == 'Velocity ROC Significance':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['vel'], other_para=True)

                # calculates the binned kinematic spike frequencies
                cfcn.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para, w_prog)

                # calculates the kinematic roc curves and their significance
                self.calc_kinematic_roc_curves(data, pool, calc_para, g_para, 0.)
                self.calc_kinematic_roc_significance(data, calc_para, g_para)

            elif self.thread_job_secondary == 'Condition ROC Curve Comparison':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['phase'])

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, True, 33.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, calc_para, 66.)
                self.calc_phase_roc_significance(calc_para, g_para, data, pool, 100.)

            elif self.thread_job_secondary == 'Direction ROC Significance':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['condition', 'phase'])

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, calc_para, plot_para, g_para, True, 33.,
                                                 force_black_calc=True):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, calc_para, 66.)
                self.calc_phase_roc_significance(calc_para, g_para, data, pool, 100.)

                if cf.det_valid_vis_expt(data, True):
                    if not self.calc_dirsel_group_types(data, pool, calc_para, plot_para, g_para):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

            ###############################################
            ####    COMBINED ANALYSIS LDA FUNCTIONS    ####
            ###############################################

            elif self.thread_job_secondary == 'Rotation/Visual Stimuli Response Statistics':
                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, calc_para, 50.)

                # calculates the direction/selection group types
                if not self.calc_dirsel_group_types(data, pool, calc_para, plot_para, g_para):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)

            elif self.thread_job_secondary == 'Combined Direction ROC Curves (Whole Experiment)':
                # checks that the conditions are correct for running the function
                if not self.check_combined_conditions(calc_para, plot_para):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['condition', 'phase', 'visual'])

                # initisalises the rotational filter (if not initialised already)
                if plot_para['rot_filt'] is None:
                    plot_para['rot_filt'] = cf.init_rotation_filter_data(False)

                # adds motordrifting (if the visual expt type)
                _plot_para, _calc_para = dcopy(plot_para), dcopy(calc_para)
                if calc_para['vis_expt_type'] == 'MotorDrifting':
                    _plot_para['rot_filt']['t_type'].append('MotorDrifting')

                # resets the flags to use the full rotation/visual phases
                _calc_para['use_full_rot'], _calc_para['use_full_vis'] = True, True

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(data, pool, _calc_para, _plot_para, g_para, False, 33.):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(data, _calc_para, 66.)
                if (calc_para['vis_expt_type'] == 'UniformDrifting') and \
                                                (calc_para['grp_stype'] != 'Wilcoxon Paired Test'):
                    # sets up the visual rotation filter
                    r_filt_v = cf.init_rotation_filter_data(False)
                    r_filt_v['t_type'], r_filt_v['is_ud'], r_filt_v['t_cycle'] = ['UniformDrifting'], [True], ['15']

                    # retrieves the visual filter object
                    plot_exp_name, plot_all_expt = plot_para['plot_exp_name'], plot_para['plot_all_expt']
                    r_obj_vis, ind_type = cf.split_unidrift_phases(data, r_filt_v, None, plot_exp_name, plot_all_expt,
                                                                   'Whole Experiment', 2.)

                    # calculates the full uniform-drifting curves
                    self.calc_ud_roc_curves(data, r_obj_vis, ind_type, 66.)

                # calculates the direction selection types
                if not self.calc_dirsel_group_types(data, pool, _calc_para, _plot_para, g_para):
                    self.is_ok = False

                # calculates the partial roc curves
                self.calc_partial_roc_curves(data, calc_para, plot_para, 66.)

            elif self.thread_job_secondary in ['Normalised Kinematic Spiking Frequency']:
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['vel'], other_para=False)

                # calculates the binned kinematic spike frequencies
                cfcn.calc_binned_kinemetic_spike_freq(data, plot_para, calc_para, w_prog, roc_calc=False)

            ######################################################
            ####    DEPTH-BASED SPIKING ANALYSIS FUNCTIONS    ####
            ######################################################

            elif self.thread_job_secondary == 'Depth Spiking Rate Comparison':
                # make a copy of the plotting/calculation parameters
                _plot_para, _calc_para, r_data = dcopy(plot_para), dcopy(calc_para), data.depth
                _plot_para['plot_exp_name'] = None

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['condition', 'phase', 'visual'])

                # reduces the data clusters to only include the RSPd/RSPg cells
                _data = cfcn.get_rsp_reduced_clusters(data)

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(_data, pool, _calc_para, _plot_para, g_para, True,
                                                 33., r_data=r_data, force_black_calc=True):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(_data, _calc_para, 66., r_data=r_data)

                ############################################
                ####    SPIKING FREQUENCY CALCULATION   ####
                ############################################

                # initialisations
                r_filt = _plot_para['rot_filt']
                r_data.ch_depth, r_data.ch_region, r_data.ch_layer = \
                                cfcn.get_channel_depths_tt(_data._cluster, r_filt['t_type'])
                t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

                # rotation filtered object calculation
                r_obj_rot = RotationFilteredData(_data, r_filt, None, None, True, 'Whole Experiment', False,
                                                 t_ofs=t_ofs, t_phase=t_phase)

                # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
                sp_f0_rot, sp_f_rot = cf.calc_phase_spike_freq(r_obj_rot)
                s_plt, _, sf_stats, ind = cf.setup_spike_freq_plot_arrays(r_obj_rot, sp_f0_rot, sp_f_rot, None, 3)
                r_data.plt, r_data.stats, r_data.ind, r_data.r_filt = s_plt, sf_stats, ind, dcopy(r_filt)

            elif self.thread_job_secondary == 'Depth Spiking Rate Comparison (Multi-Sensory)':
                # checks that the conditions are correct for running the function
                if not self.check_combined_conditions(calc_para, plot_para):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                else:
                    # otherwise, make a copy of the plotting/calculation parameters
                    _plot_para, _calc_para, r_data = dcopy(plot_para), dcopy(calc_para), data.depth
                    _plot_para['plot_exp_name'], r_filt = None, _plot_para['rot_filt']
                    t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['condition', 'phase', 'visual'])

                # adds motordrifting (if it is the visual expt type)
                if calc_para['vis_expt_type'] == 'MotorDrifting':
                    _plot_para['rot_filt']['t_type'].append('MotorDrifting')

                # reduces the data clusters to only include the RSPd/RSPg cells
                _data = cfcn.get_rsp_reduced_clusters(data)

                # calculates the phase roc-curves for each cell
                if not self.calc_cond_roc_curves(_data, pool, _calc_para, _plot_para, g_para, False, 33., r_data=r_data):
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # calculates the phase roc curve/significance values
                self.calc_phase_roc_curves(_data, _calc_para, 66., r_data=r_data)
                if (calc_para['vis_expt_type'] == 'UniformDrifting'):
                    # sets up the visual rotation filter
                    r_filt_v = cf.init_rotation_filter_data(False)
                    r_filt_v['t_type'], r_filt_v['is_ud'], r_filt_v['t_cycle'] = ['UniformDrifting'], [True], ['15']

                    # retrieves the visual filter object
                    r_obj_vis, ind_type = cf.split_unidrift_phases(_data, r_filt_v, None, None, True,
                                                                   'Whole Experiment', 2., t_phase, t_ofs)

                    # calculates the full uniform-drifting curves
                    self.calc_ud_roc_curves(_data, r_obj_vis, ind_type, 66., r_data=r_data)

                    # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
                    sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj_vis)
                    s_plt, _, sf_stats, ind = cf.setup_spike_freq_plot_arrays(r_obj_vis, sp_f0, sp_f, ind_type, 2)
                    r_data.plt_vms, r_data.stats_vms, r_data.ind_vms = s_plt, sf_stats, ind, r_filt_v
                    r_data.r_filt_vms = dcopy(r_filt_v)
                else:
                    # resets the uniform drifting fields
                    r_data.plt_vms, r_data.stats_vms, r_data.ind_vms, r_data.r_filt_vms = None, None, None, None

                ############################################
                ####    SPIKING FREQUENCY CALCULATION   ####
                ############################################

                # rotation filtered object calculation
                r_obj_rot = RotationFilteredData(_data, r_filt, None, None, True, 'Whole Experiment', False,
                                                 t_phase=t_phase, t_ofs=t_ofs)
                r_data.ch_depth_ms, r_data.ch_region_ms, r_data.ch_layer_ms = \
                                    cfcn.get_channel_depths_tt(_data._cluster, r_filt['t_type'])

                # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
                sp_f0_rot, sp_f_rot = cf.calc_phase_spike_freq(r_obj_rot)
                s_plt, _, sf_stats, ind = cf.setup_spike_freq_plot_arrays(r_obj_rot, sp_f0_rot, sp_f_rot, None, 3)
                r_data.plt_rms, r_data.stats_rms, r_data.ind_rms = s_plt, sf_stats, ind
                r_data.r_filt_rms = dcopy(r_filt)

            ##########################################################
            ####    ROTATION DISCRIMINATION ANALYSIS FUNCTIONS    ####
            ##########################################################

            elif self.thread_job_secondary == 'Rotation Direction LDA':
                # if the solver parameter have not been set, then initalise them
                d_data = data.discrim.dir

                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=d_data)

                # sets up the lda values
                r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, d_data,
                                                                             w_prog, w_err=w_err)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                            d_data=d_data, w_prog=w_prog):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

            elif self.thread_job_secondary == 'Temporal Duration/Offset LDA':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.temp)

                # if the temporal data parameters have changed/has not been initialised then calculate the values
                if data.discrim.temp.lda is None:
                    # checks to see if any base LDA calculation parameters have been altered
                    self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.temp)

                    # sets up the important arrays for the LDA
                    r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, data.discrim.temp,
                                                                                 w_prog, w_err=w_err)
                    if status == 0:
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                    # if an update in the calculations is required, then run the temporal LDA analysis
                    if status == 2:
                        if not self.run_temporal_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                            # if there was an error in the calculations, then return an error flag
                            self.is_ok = False
                            self.work_finished.emit(thread_data)
                            return

            elif self.thread_job_secondary == 'Individual LDA':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.indiv)
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.dir)

                # sets up the important arrays for the LDA
                r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, data.discrim.dir,
                                                                             w_prog, True, w_err=w_err)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                            d_data=data.discrim.dir, w_prog=w_prog):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                # if the individual data parameters have changed/has not been initialised then calculate the values
                if data.discrim.indiv.lda is None:
                    # runs the individual LDA
                    if not self.run_individual_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

            elif self.thread_job_secondary == 'Shuffled LDA':
                # checks to see if any parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.shuffle)
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.dir)

                # sets up the important arrays for the LDA
                r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, data.discrim.dir,
                                                                             w_prog, True, w_err=w_err)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                            d_data=data.discrim.dir, w_prog=w_prog):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                # runs the shuffled LDA
                if not self.run_shuffled_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

            elif self.thread_job_secondary == 'Pooled Neuron LDA':
                # resets the minimum cell count and checks if the pooled parameters have been altered
                # calc_para['lda_para']['n_cell_min'] = calc_para['n_cell_min']
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.part)

                # if the pooled data parameters have changed/has not been initialised then calculate the values
                if data.discrim.part.lda is None:
                    # checks to see if any base LDA calculation parameters have been altered
                    self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.dir)

                    # sets up the important arrays for the LDA
                    r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, data.discrim.dir,
                                                                                 w_prog, True, w_err=w_err)
                    if not calc_para['pool_expt']:
                        if status == 0:
                            # if there was an error in the calculations, then return an error flag
                            self.is_ok = False
                            self.work_finished.emit(thread_data)
                            return
                        # elif status == 2:
                        #     # if an update in the calculations is required, then run the rotation LDA analysis
                        #     if not cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                        #                             d_data=data.discrim.dir, w_prog=w_prog):
                        #         self.is_ok = False
                        #         self.work_finished.emit(thread_data)
                        #         return

                    # runs the partial LDA
                    if not self.run_pooled_lda(pool, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

            elif self.thread_job_secondary == 'Individual Cell Accuracy Filtered LDA':
                # check to see if the individual LDA calculations have been performed
                if data.discrim.indiv.lda is None:
                    # if the individual LDA has not been run, then output an error to screen
                    e_str = 'The Individual LDA must be run first before this analysis can be performed'
                    w_err.emit(e_str, 'Missing Individual LDA Data')

                    # sets the ok flag to false and exit the function
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                #
                _calc_para = dcopy(calc_para)
                _calc_para['comp_cond'] = dcopy(data.discrim.indiv.ttype)

                #########################################
                ####    ROTATION LDA CALCULATIONS    ####
                #########################################

                # sets the min/max accuracy values
                _calc_para['lda_para']['y_acc_min'] = 0
                _calc_para['lda_para']['y_acc_max'] = 100

                # checks to see if any base LDA calculation parameters have been altered
                self.check_altered_para(data, _calc_para, g_para, ['lda'], other_para=data.discrim.dir)

                # sets up the important arrays for the LDA
                r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, _calc_para, data.discrim.dir,
                                                                             w_prog, True, w_err=w_err)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not cfcn.run_rot_lda(data, _calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                            d_data=data.discrim.dir, w_prog=w_prog, pW=50.):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                #########################################
                ####    FILTERED LDA CALCULATIONS    ####
                #########################################

                # sets the min/max accuracy values
                _calc_para['lda_para']['y_acc_min'] = _calc_para['y_acc_min']
                _calc_para['lda_para']['y_acc_max'] = _calc_para['y_acc_max']

                # checks to see if any base LDA calculation parameters have been altered
                self.check_altered_para(data, _calc_para, g_para, ['lda'], other_para=data.discrim.filt)

                # sets up the important arrays for the LDA
                r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, _calc_para, data.discrim.filt,
                                                                             w_prog, True, w_err=w_err)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not cfcn.run_rot_lda(data, _calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                             d_data=data.discrim.filt, w_prog=w_prog, pW=50., pW0=50.):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return
                    else:
                        # otherwise, update the calculation parameters
                        data.discrim.filt.yaccmn = _calc_para['y_acc_min']
                        data.discrim.filt.yaccmx = _calc_para['y_acc_max']

            elif self.thread_job_secondary == 'LDA Group Weightings':
                # checks to see if the data class as changed parameters
                d_data, w_prog = data.discrim.wght, self.work_progress
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=d_data)

                # sets up the lda values
                r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, d_data,
                                                                             w_prog, w_err=w_err)
                if status == 0:
                    # if there was an error in the calculations, then return an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return
                elif status == 2:
                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not self.run_wght_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

            #######################################################
            ####    SPEED DISCRIMINATION ANALYSIS FUNCTIONS    ####
            #######################################################

            elif self.thread_job_secondary == 'Speed LDA Accuracy':
                # checks to see if any base LDA calculation parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.spdacc)

                # if the pooled data parameters have changed/has not been initialised then calculate the values
                if data.discrim.spdc.lda is None:

                    # sets up the important arrays for the LDA
                    r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, data.discrim.spdacc,
                                                                                 w_prog, True, w_err=w_err)
                    if status == 0:
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return
                    elif status == 2:
                        if not self.run_speed_lda_accuracy(data, calc_para, r_filt, i_expt, i_cell, n_trial_max, w_prog):
                            self.is_ok = False
                            self.work_finished.emit(thread_data)
                            return

            elif self.thread_job_secondary == 'Speed LDA Comparison (Individual Experiments)':
                # checks to see if any base LDA calculation parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.spdc)

                # if the pooled data parameters have changed/has not been initialised then calculate the values
                if data.discrim.spdc.lda is None:

                    # sets up the important arrays for the LDA
                    r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, data.discrim.spdc,
                                                                                 w_prog, True, w_err=w_err)
                    if status == 0:
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return
                    elif status == 2:
                        # if an update in the calculations is required, then run the rotation LDA analysis
                        if not self.run_kinematic_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max, w_prog):
                            self.is_ok = False
                            self.work_finished.emit(thread_data)
                            return

            elif self.thread_job_secondary == 'Speed LDA Comparison (Pooled Experiments)':
                # checks to see if any base LDA calculation parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.spdcp)

                # if the pooled data parameters have changed/has not been initialised then calculate the values
                if data.discrim.spdcp.lda is None:

                    # sets up the important arrays for the LDA
                    r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, data.discrim.spdcp,
                                                                                 w_prog, True, w_err=w_err)
                    if status == 0:
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return
                    # elif status == 2:/

                    # if an update in the calculations is required, then run the rotation LDA analysis
                    if not self.run_pooled_kinematic_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max,
                                                         w_prog):
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return

                # # calculates the psychometric curves
                # w_prog.emit('Calculating Pyschometric Curves', 100.)
                # cfcn.calc_all_psychometric_curves(data.discrim.spdcp, float(calc_para['vel_bin']), calc_para['use_all'])

            elif self.thread_job_secondary == 'Velocity Direction Discrimination LDA':
                # checks to see if any base LDA calculation parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['lda'], other_para=data.discrim.spddir)

                # if the pooled data parameters have changed/has not been initialised then calculate the values
                if data.discrim.spddir.lda is None:

                    # sets up the important arrays for the LDA
                    r_filt, i_expt, i_cell, n_trial_max, status = cfcn.setup_lda(data, calc_para, data.discrim.spddir,
                                                                                 w_prog, True, w_err=w_err)
                    if status == 0:
                        # if there was an error in the calculations, then return an error flag
                        self.is_ok = False
                        self.work_finished.emit(thread_data)
                        return
                    elif status == 2:
                        if not self.run_speed_dir_lda_accuracy(data, calc_para, r_filt, i_expt, i_cell,
                                                               n_trial_max, w_prog):
                            self.is_ok = False
                            self.work_finished.emit(thread_data)
                            return

            #######################################
            ####    MISCELLANEOUS FUNCTIONS    ####
            #######################################

            elif self.thread_job_secondary == 'Velocity Multilinear Regression Dataframe Output':
                # checks to see if any base spiking frequency dataframe parameters have been altered
                self.check_altered_para(data, calc_para, plot_para, g_para, ['spikedf'], other_para=data.spikedf)

                # checks to see if the overlap duration is less than the time bin size
                if calc_para['t_over'] >= calc_para['bin_sz']:
                    # if not, then output an error to screen
                    e_str = 'Bin Overlap Duration must be less than the Time Bin Size.\n' \
                            'Reset these parameters before running this function.'
                    w_err.emit(e_str, 'Incorrect Function Parameters')

                    # exits the function with an error flag
                    self.is_ok = False
                    self.work_finished.emit(thread_data)
                    return

                # only continue if the spiking frequency dataframe has not been set up
                if not data.spikedf.is_set:
                    self.setup_spiking_freq_dataframe(data, calc_para)

            elif self.thread_job_secondary == 'Autocorrelogram Theta Index Calculations':
                # case to see if any parameters have changed
                self.check_altered_para(data, calc_para, plot_para, g_para, ['theta'], other_para=data.theta_index)

                # only continue if the theta index dataframe has not been setup
                if not data.theta_index.is_set:
                    self.calc_auto_ccgram_fft(data, calc_para)

    ###############################
    ####    OTHER FUNCTIONS    ####
    ###############################

            elif self.thread_job_secondary == 'Shuffled Cluster Distances':
                # case is the shuffled cluster distances
                thread_data = self.calc_shuffled_cluster_dist(calc_para, data.cluster)

        elif self.thread_job_primary == 'update_plot':
            pass

        # emits the finished work signal
        self.work_finished.emit(thread_data)

    ############################################
    ####    THREAD CALCULATION FUNCTIONS    ####
    ############################################

    def load_data_file(self):
        '''

        :param exp_file:
        :return:
        '''

        # retrieves the job parameters
        load_dlg, loaded_exp, is_multi = self.thread_job_para[0], self.thread_job_para[1], self.thread_job_para[2]
        if not np.any([not x in loaded_exp for x in load_dlg.exp_name]):
            # if there are no new experiments to load, then exit the function
            return None
        else:
            n_file = len(load_dlg.exp_files)
            dpw, p_rlx, data = 1.0 / n_file, 0.05, []
            _, f_extn = os.path.splitext(load_dlg.exp_files[0])

        #
        for i_file in range(n_file):
            if not self.is_running:
                # if the user cancelled, then exit
                return None
            else:
                # updates the progress bar string
                p_str, pw0 = 'Loading File {0} of {1}'.format(i_file+1, n_file), i_file / n_file
                self.work_progress.emit(p_str, 100.0 * pw0)

            # sets the experiment file and name
            if load_dlg.exp_name[i_file] not in loaded_exp:
                # loads the data from the data file
                with open(load_dlg.exp_files[i_file], 'rb') as fp:
                    data_nw = p.load(fp)

                # setting of other fields
                if isinstance(data_nw, dict):
                    data_nw['expFile'] = load_dlg.exp_files[i_file]

                # re-calculates the signal features (single experiment only)
                if f_extn == '.cdata':
                    if np.shape(data_nw['sigFeat'])[1] == 5:
                        # memory allocation for the signal features
                        xi = np.array(range(data_nw['nPts']))
                        sFeat = np.zeros((data_nw['nC'], 2))

                        for i in range(data_nw['nC']):
                            # creates the piecewise-polynomial of the mean signal
                            pp, t_max = pchip(xi, data_nw['vMu'][:, i]), data_nw['sigFeat'][i, 2]
                            t_min = np.argmin(data_nw['vMu'][int(t_max):, i]) + t_max
                            v_max_2 = data_nw['vMu'][int(t_max), i] / 2.0
                            v_min = np.min(data_nw['vMu'][int(t_max):, i])
                            v_half = data_nw['vMu'][int(data_nw['sigFeat'][i, 1]), i] / 2.0

                            ##################################################
                            ####    POST-STIMULI SPIKE HALF-WIDTH TIME    ####
                            ##################################################

                            # determines the point/voltage of the pmaximum proceding the minimum
                            bnd_1 = [(data_nw['sigFeat'][i, 0], data_nw['sigFeat'][i, 1])]
                            bnd_2 = [(data_nw['sigFeat'][i, 1], data_nw['sigFeat'][i, 2])]
                            bnd_3 = [(data_nw['sigFeat'][i, 2], t_min)]

                            # determines the location of the half-width points
                            t_hw1_lo = cfcn.opt_time_to_y0((pp, v_half), bnd_1)
                            t_hw1_hi = cfcn.opt_time_to_y0((pp, v_half), bnd_2)
                            t_hw2_lo = cfcn.opt_time_to_y0((pp, v_max_2), bnd_2)
                            t_hw2_hi = cfcn.opt_time_to_y0((pp, v_max_2), bnd_3)
                            t_rlx = cfcn.opt_time_to_y0((pp, v_min + p_rlx * (v_max_2 - v_min)), bnd_3)

                            # determine if it is feasible to find the 2nd peak half-width point
                            if (t_hw2_hi is None) or (t_rlx is None):
                                # if not, then linearly extrapolate past the end point of the signal
                                xi2 = np.array(range(2*xi[-1]))
                                ppL = IUS(xi, data_nw['vMu'][:, i], k=1)

                                # determines the half-width/relaxtion time from the extrapolated signal
                                bnd_4 = [(data_nw['sigFeat'][i, 2], xi2[-1])]
                                t_hw2_hi = cfcn.opt_time_to_y0((ppL, v_max_2), bnd_4)
                                t_rlx = cfcn.opt_time_to_y0((ppL, v_min + p_rlx * (v_max_2 - v_min)), bnd_4)

                            # calculates the new signal features
                            data_nw['sigFeat'][i, 3] = t_hw1_lo
                            data_nw['sigFeat'][i, 4] = t_hw1_hi
                            sFeat[i, 0] = t_hw2_hi - t_hw2_lo
                            sFeat[i, 1] = t_rlx - t_max

                        # concatenates the new signal feature date
                        data_nw['sigFeat'] = np.concatenate((data_nw['sigFeat'], sFeat), axis=1)

                    # sets the cell cluster include indices (if not already set)
                    if 'clInclude' not in data_nw['expInfo']:
                        data_nw['expInfo']['clInclude'] = np.ones(data_nw['nC'], dtype=bool)

                # appends the new data dictionary to the overall data list
                data.append(data_nw)

        # appends the current filename to the data dictionary and returns the object
        return data

    def save_multi_expt_file(self, data, out_info):
        '''

        :return:
        '''

        # updates the progressbar
        self.work_progress.emit('Saving Data To File...', 50.0)

        # sets the file extension (based on the data type)
        if hasattr(data.comp, 'data'):
            f_extn = 'mdata' if len(data.comp.data) == 0 else 'mcomp'
        else:
            f_extn = 'mdata'

        # sets the output file name
        out_file = os.path.join(out_info['inputDir'], '{0}.{1}'.format(out_info['dataName'], f_extn))

        # outputs the data to file
        with open(out_file, 'wb') as fw:
            p.dump(data, fw)

        # updates the progressbar
        self.work_progress.emit('Data Save Complete!', 100.0)

    def save_multi_comp_file(self, data, out_info):
        '''

        :return:
        '''

        # updates the progressbar
        self.work_progress.emit('Saving Data To File...', 50.0)

        # memory allocation
        n_file = len(out_info['exptName'])

        # sets the output file name
        out_file = os.path.join(out_info['inputDir'], '{0}.mcomp'.format(out_info['dataName']))

        # output data file
        data_out = {
            'data': np.empty((n_file, 2), dtype=object),
            'c_data': np.empty(n_file, dtype=object),
            'ff_corr': data.comp.ff_corr if hasattr(data.comp, 'ff_corr') else None,
            'f_data': data.externd.free_data if hasattr(data.externd, 'free_data') else None
        }

        for i_file in range(n_file):
            # retrieves the index of the data field corresponding to the current experiment
            fix_file = out_info['exptName'][i_file].split('/')[0]
            i_comp = cf.det_comp_dataset_index(data.comp.data, fix_file)

            # creates the multi-experiment data file based on the type
            data_out['c_data'][i_file] = data.comp.data[i_comp]
            data_out['data'][i_file, 0], data_out['data'][i_file, 1] = \
                                        cf.get_comp_datasets(data, c_data=data_out['c_data'][i_file], is_full=True)

        # outputs the data to file
        with open(out_file, 'wb') as fw:
            p.dump(data_out, fw)

        # updates the progressbar
        self.work_progress.emit('Data Save Complete!', 100.0)

    def init_pool_worker(self):
        '''

        :return:
        '''

        # creates the pool worker object
        p = mp.Pool(int(np.floor(1.5 * mp.cpu_count())))

        # returns the object
        return p

    def init_cluster_data(self):
        '''

        :return:
        '''

        def map_cluster_depths():
            '''

            :param cluster_depth:
            :return:
            '''

            # retrieves the cluster depths from the spike I/O class object
            cluster_depth = sp_io.get_cluster_depths(cluster_ids)

            # sets the mapped cluster depths based on the file type
            if (exp_info['dmapFile'] is None) or (len(exp_info['dmapFile']) == 0):
                # no map is given so return the original depth values
                return cluster_depth, None
            else:
                # otherwise, map the cluster depth values from the probe to actual values
                data = np.array(pd.read_csv(exp_info['dmapFile']))
                if np.size(data, axis=1) < 4:
                    # if the mapping file is not correct, then output an error to screen
                    e_str = 'Channel mapping file does not have the correct format.\n\n' \
                            'Re-select a valid file before attempting to initialise the combined data files.'
                    self.work_error.emit(e_str, 'Invalid Channel Mapping File')

                    # return none values indicating the error
                    return None, None
                else:
                    # otherwise, return the mapped channel depths and the other mapping values
                    return np.array([data[data[:, 1] == x, 0][0] for x in cluster_depth]), data[:, :4]

        # retrieves the job parameters
        exp_info, out_name, g_para = self.thread_job_para[0], self.thread_job_para[1], self.thread_job_para[2]

        # sets the global parameters
        n_hist = int(g_para['n_hist'])
        n_spike = int(g_para['n_spike'])
        cluster_ids = None

        # retrieves the spike I/O data and sets the cluster IDs based on the cluster type
        sp_io = spike_io.SpikeIo(exp_info['srcDir'], exp_info['traceFile'], int(exp_info['nChan']))
        if exp_info['clusterType'] == 'Good':
            # case is the good clusters
            if hasattr(sp_io, 'good_cluster_ids'):
                cluster_ids = sp_io.good_cluster_ids
        elif exp_info['clusterType'] == 'MUA':
            # case is the multi-unit clusters
            if hasattr(sp_io, 'MUA_cluster_ids'):
                cluster_ids = sp_io.MUA_cluster_ids

        if cluster_ids is None:
            e_str = 'Cluster group file is missing? Please re-run with cluster-group file in the source data directory'
            self.work_error.emit(e_str, 'Cluster Group File Missing!')
            return

        # retrieves the clusters spike data and channel depths
        self.work_progress.emit('Reshaping Cluster Data...', 0.0)
        clusters = [ClusterRead(sp_io, cid) for cid in cluster_ids]

        # determines the channel depths mapping
        depth, channel_map_data = map_cluster_depths()
        if depth is None:
            # if the file has an incorrect format, then exit the function
            return

        # determines if the mapping values were set correctly
        if channel_map_data is not None:
            # if so, then determine the region/recording layers
            y_coords = channel_map_data[:, 3]
            depthLo, depthHi = np.array(exp_info['depthLo']).astype(int), np.array(exp_info['depthHi']).astype(int)
            indD = np.array([next((i for i in range(len(depthHi)) if x <= depthHi[i]), len(depthHi)-1) for x in y_coords])
            chRegion = np.array(exp_info['regionName'])[indD][depth.astype(int)]
            chLayer = np.array(exp_info['recordLayer'])[indD][depth.astype(int)]

        else:
            # otherwise, return N/A for the region/recording layers
            chRegion, chLayer = ['N/A'] * len(clusters), ['N/A'] * len(clusters)
            depthLo, depthHi = None, None

        # sets the signal point-wise/ISI bin vectors
        xi_pts_H = np.linspace(-200, 100, n_hist + 1)
        xi_isi_H = np.linspace(0, 1000, n_hist + 1)

        # creates the recording/experimental information sub-dictionaries
        expInfo = {'name': exp_info['expName'], 'date': exp_info['expDate'], 'cond': exp_info['expCond'],
                   'type': exp_info['expType'], 'sex': exp_info['expSex'], 'age': exp_info['expAge'],
                   'probe': exp_info['expProbe'], 'lesion': exp_info['lesionType'], 'channel_map': channel_map_data,
                   'cluster_type': exp_info['clusterType'], 'other_info': exp_info['otherInfo'],
                   'record_state': exp_info['recordState'], 'record_coord': exp_info['recordCoord'],
                   'depth_lo': depthLo, 'depth_hi': depthHi}

        # memory allocation
        pW0, pW1, nFeat = 20.0, 60.0, 5
        nC, nSample = len(clusters), np.size(sp_io.traces, axis=0)
        sFreq, vGain = float(exp_info['sFreq']), float(exp_info['vGain'])

        # sets the data file dictionary object
        A = {
            'vSpike': np.empty(nC, dtype=object), 'tSpike': np.empty(nC, dtype=object),
            'vMu': None, 'vSD': None, 'ccGram': None, 'ccGramXi': None, 'sigFeat': np.zeros((nC, nFeat)),
            'clustID': cluster_ids, 'expInfo': expInfo, 'chDepth': depth, 'chRegion': chRegion, 'chLayer': chLayer,
            'sFreq': sFreq, 'nC': nC,  'nPts': None, 'tExp': nSample / sFreq, 'vGain': vGain,
            'isiHist': np.empty(nC, dtype=object), 'isiHistX': xi_isi_H,
            'ptsHist': np.empty(nC, dtype=object), 'ptsHistX': xi_pts_H,
            'rotInfo': None,
        }

        # sets up the rotation analysis data dictionary
        A['rotInfo'] = rot.load_rot_analysis_data(A, exp_info, sp_io, w_prog=self.work_progress, pW0=pW0)

        # sets up the sub-job flags
        self.sub_job = np.zeros(nC, dtype=bool)

        # retrieves the cluster data
        for i, c in enumerate(clusters):
            if not self.is_running:
                # if the user cancelled, then exit the function
                return
            else:
                # updates the main gui progressnbar
                pW = pW0 + pW1 * (i + 1) / nC
                self.work_progress.emit('Processing Cluster {0} of {1}'.format(i + 1, nC), pW)

            ###################################################
            ####    DATA RETRIEVAL & MEMORY ALLOCATIONS    ####
            ###################################################

            # retrieves the spike voltage/timing
            v_spike = c.channel_waveforms
            t_spike = 1000.0 * sp_io.get_spike_times_in_cluster(cluster_ids[i]) / sFreq

            # memory allocation (only for the first cluster)
            if i == 0:
                A['nPts'] = np.size(v_spike, axis=0)
                A['vMu'] = np.zeros((A['nPts'], nC), dtype=float)
                A['vSD'] = np.zeros((A['nPts'], nC), dtype=float)
                xi = np.array(range(A['nPts']))

            ###############################################
            ####    MAIN METRIC CALCULATION/STORAGE    ####
            ###############################################

            # sets the values into the final array
            A['vSpike'][i] = v_spike[:, :n_spike] * vGain
            A['tSpike'][i] = t_spike[:np.size(v_spike, axis=1)]

            # calculates the mean/standard deviation of the voltage spikes
            A['vMu'][:, i] = np.mean(v_spike, axis=1) * vGain
            A['vSD'][:, i] = np.std(v_spike, axis=1) * vGain

            ######################################
            ####    HISTOGRAM CALCULATIONS    ####
            ######################################

            # calculates the point-wise histograms
            A['ptsHist'][i] = np.zeros((A['nPts'], n_hist), dtype=int)
            for iPts in range(A['nPts']):
                H = np.histogram(v_spike[iPts, :], bins=xi_pts_H)
                A['ptsHist'][i][iPts, :] = H[0]

            # calculates the ISI histograms
            dT = np.diff(A['tSpike'][i])
            dT = dT[dT <= xi_isi_H[-1]]
            H_isi = np.histogram(dT, bins=xi_isi_H, range=(xi_isi_H[0], xi_isi_H[-1]))
            A['isiHist'][i] = H_isi[0]

            ###########################################
            ####    SIGNAL FEATURE CALCULATIONS    ####
            ###########################################

            # creates the piecewise-polynomial of the mean signal
            pp = pchip(xi, A['vMu'][:, i])

            # determines the point/voltage of the pmaximum proceding the minimum
            i_min = np.argmin(A['vMu'][:, i])
            i_max1 = np.argmax(A['vMu'][:i_min, i])
            i_max2 = np.argmax(A['vMu'][i_min:, i]) + i_min

            # determines the location of the half-width points
            v_half = (min(pp(i_max1), pp(i_max2)) + pp(i_min)) / 2.0
            t_lo = cfcn.opt_time_to_y0((pp, v_half), [(i_max1, i_min)])
            t_hi = cfcn.opt_time_to_y0((pp, v_half), [(i_min, i_max2)])

            # sets the signal features into the final array
            A['sigFeat'][i, :] = [i_max1, i_min, i_max2, t_lo, t_hi]

            # memory garbage collection
            gc.collect()

        ######################################################
        ####    CLUSTER CROSS-CORRELOGRAM CALCULATIONS    ####
        ######################################################

        # memory allocation
        win_size = 50

        # calculates the cross-correlation between each signal from each cluster
        for i_row in range(nC):
            if not self.is_running:
                # if the user cancelled, then exit the function
                return
            else:
                # updates the main gui progressbar
                pW = (pW0 + pW1) + (100.0 - (pW0 + pW1)) * (i_row + 1) / (nC + 1)
                self.work_progress.emit('Calculating CC-Grams...', pW)

            # calculates the cross-correlograms between each of the other clusters
            for j_row in range(nC):
                if (i_row == 0) and (j_row == 0):
                    # case is the first cluster so allocate memory and set the time bin array
                    ccGram, A['ccGramXi'] = cfcn.calc_ccgram(A['tSpike'][i_row], A['tSpike'][j_row], win_size)
                    A['ccGram'] = np.zeros((nC, nC, len(ccGram)))
                    A['ccGram'][i_row, j_row, :] = ccGram
                else:
                    # otherwise, set the new values directly into the array
                    A['ccGram'][i_row, j_row, :], _ = cfcn.calc_ccgram(A['tSpike'][i_row], A['tSpike'][j_row], win_size)

        #################################
        ####    FINAL DATA OUTPUT    ####
        #################################

        # dumps the cluster data to file
        self.work_progress.emit('Outputting Data To File...', 99.0)
        cf.save_single_file(out_name, A)

    ##########################################
    ####    CLUSTER MATCHING FUNCTIONS    ####
    ##########################################

    def det_cluster_matches(self, data, calc_para, w_prog):
        '''

        :param exp_name:
        :param comp_dlg:
        :return:
        '''

        # retrieves the comparison dataset
        i_comp = cf.det_comp_dataset_index(data.comp.data, calc_para['calc_comp'])
        c_data, data.comp.last_comp = data.comp.data[i_comp], i_comp

        # if there is no further calculation necessary, then exit the function
        if c_data.is_set:
            return

        # updates the cluster matching parameters
        c_data.is_set = True
        c_data.d_max = calc_para['d_max']
        c_data.r_max = calc_para['r_max']
        c_data.sig_corr_min = calc_para['sig_corr_min']
        c_data.isi_corr_min = calc_para['isi_corr_min']
        c_data.sig_diff_max = calc_para['sig_diff_max']
        c_data.sig_feat_min = calc_para['sig_feat_min']
        c_data.w_sig_feat = calc_para['w_sig_feat']
        c_data.w_sig_comp = calc_para['w_sig_comp']
        c_data.w_isi = calc_para['w_isi']

        # retrieves the fixed/free cluster dataframes
        data_fix, data_free = cf.get_comp_datasets(data, c_data=c_data, is_full=True)

        def det_overall_cluster_matches(is_feas, D):
            '''

            :param data_fix:
            :param data_free:
            :param D:
            :return:
            '''

            # calculates the pair-wise SS distances between each the fixed/free mean signals
            iDsort, n_rows = np.argsort(D.T, axis=None), np.size(D, axis=0)

            # memory allocation
            isFix = np.zeros(data_fix['nC'], dtype=bool)
            isFree = np.zeros(data_free['nC'], dtype=bool)
            i_match = -np.ones(data_fix['nC'], dtype=int)

            # determines the overall unique
            for i in range(len(iDsort)):
                # determines the indices of the next best match
                iR, iC = cfcn.ind2sub(n_rows, iDsort[i])
                if not (isFix[iR] or isFree[iC]) and is_feas[iR, iC]:
                    # if there is not already a match, then update the match arrays
                    i_match[iR] = iC
                    isFix[iR], isFree[iC] = True, True
                    if all(isFix) or all(isFree):
                        # if all matches are found, then exit the loop
                        break

            # returns the final match array
            return i_match

        def det_cluster_matches_old(c_data, is_feas, d_depth):
            '''

            :param data_fix:
            :param data_free:
            :return:
            '''

            # parameters
            z_max = 1.0

            # calculates the inter-signal euclidean distances
            DD = cdist(data_fix['vMu'].T, data_free['vMu'].T)

            # determines the matches based on the signal euclidean distances
            c_data.i_match_old = det_overall_cluster_matches(is_feas, DD)

            # calculates the correlation coefficients between the best matching signals
            for i in range(data_fix['nC']):
                # calculation of the z-scores
                i_match = c_data.i_match_old[i]
                if i_match >= 0:
                    # z-score calculations
                    dW = data_fix['vMu'][:, i] - data_free['vMu'][:, i_match]
                    c_data.z_score[:, i] = np.divide(dW, data_fix['vSD'][:, i])

                    # calculates the correlation coefficient
                    CC = np.corrcoef(data_fix['vMu'][:, i], data_free['vMu'][:, i_match])
                    c_data.sig_corr_old[i] = CC[0, 1]
                    c_data.sig_diff_old[i] = DD[i, i_match]
                    c_data.d_depth_old[i] = d_depth[i, i_match]

                    # sets the acceptance flag. for a cluster to be accepted, the following must be true:
                    #   * the maximum absolute z-score must be < z_max
                    #   * the correlation coefficient between the fixed/free signals must be > sig_corr_min
                    c_data.is_accept_old[i] = np.max(np.abs(c_data.z_score[:, i])) < z_max and \
                                                   c_data.sig_corr_old[i] > c_data.sig_corr_min
                else:
                    # sets NaN values for all the single value metrics
                    c_data.sig_corr[i] = np.nan
                    c_data.d_depth_old[i] = np.nan

                    # ensures the group is rejected
                    c_data.is_accept_old[i] = False

        def det_cluster_matches_new(c_data, is_feas, d_depth, r_spike, w_prog):
            '''

            :param data_fix:
            :param data_free:
            :return:
            '''

            # parameters
            pW = 100.0 / 7.0

            # memory allocation
            signal_metrics = np.zeros((data_fix['nC'], data_free['nC'], 4))
            isi_metrics = np.zeros((data_fix['nC'], data_free['nC'], 3))
            isi_metrics_norm = np.zeros((data_fix['nC'], data_free['nC'], 3))
            total_metrics = np.zeros((data_fix['nC'], data_free['nC'], 3))

            # initialises the comparison data object
            w_prog.emit('Calculating Signal DTW Indices', pW)
            c_data = cfcn.calc_dtw_indices(c_data, data_fix, data_free, is_feas)

            # calculates the signal feature metrics
            w_prog.emit('Calculating Signal Feature Metrics', 2.0 * pW)
            signal_feat = cfcn.calc_signal_feature_diff(data_fix, data_free, is_feas)

            # calculates the signal direct matching metrics
            w_prog.emit('Calculating Signal Comparison Metrics', 3.0 * pW)
            cc_dtw, dd_dtw, dtw_scale = \
                cfcn.calc_signal_corr(c_data.i_dtw, data_fix, data_free, is_feas)

            signal_metrics[:, :, 0] = cc_dtw
            signal_metrics[:, :, 1] = 1.0 - dd_dtw
            signal_metrics[:, :, 2] = dtw_scale
            signal_metrics[:, :, 3] = \
                cfcn.calc_signal_hist_metrics(data_fix, data_free, is_feas, cfcn.calc_hist_intersect, max_norm=True)

            # calculates the ISI histogram metrics
            w_prog.emit('Calculating ISI Histogram Comparison Metrics', 4.0 * pW)
            isi_metrics[:, :, 0], isi_metrics_norm[:, :, 0] = \
                cfcn.calc_isi_corr(data_fix, data_free, is_feas)
            isi_metrics[:, :, 1], isi_metrics_norm[:, :, 1] = \
                cfcn.calc_isi_hist_metrics(data_fix, data_free, is_feas, cfcn.calc_hist_intersect, max_norm=True)
            # isi_metrics[:, :, 2], isi_metrics_norm[:, :, 2] = \
            #     cfcn.calc_isi_hist_metrics(data_fix, data_free, is_feas, cfcn.calc_wasserstein, max_norm=False)
            # isi_metrics[:, :, 3], isi_metrics_norm[:, :, 3] = \
            #     cfcn.calc_isi_hist_metrics(data_fix, data_free, is_feas, cfcn.calc_bhattacharyya, max_norm=True)

            # sets the isi relative spiking rate metrics
            isi_metrics[:, :, 2] = np.nan
            for i_row in range(np.size(r_spike, axis=0)):
                isi_metrics[i_row, is_feas[i_row, :], 2] = r_spike[i_row, is_feas[i_row, :]]
            isi_metrics_norm[:, :, 2] = cfcn.norm_array_rows(isi_metrics[:, :, 2], max_norm=False)

            # calculates the array euclidean distances (over all measures/clusters)
            weight_array = [c_data.w_sig_feat, c_data.w_sig_comp, c_data.w_isi]
            total_metrics[:, :, 0] = cfcn.calc_array_euclidean(signal_feat)
            total_metrics[:, :, 1] = cfcn.calc_array_euclidean(signal_metrics)
            total_metrics[:, :, 2] = cfcn.calc_array_euclidean(isi_metrics_norm)
            total_metrics_mean = cfcn.calc_weighted_mean(total_metrics, W=weight_array)

            # determines the unique overall cluster matches
            w_prog.emit('Determining Overall Cluster Matches', 5.0 * pW)
            c_data.i_match = det_overall_cluster_matches(is_feas, -total_metrics_mean)

            # matches which are from different regions are to be removed
            ii = np.where(c_data.i_match >= 0)[0]
            same_region = data_fix['chRegion'][ii] == data_free['chRegion'][c_data.i_match[ii]]
            c_data.i_match[ii[~same_region]] = -1

            # calculates the correlation coefficients between the best matching signals
            w_prog.emit('Setting Final Match Metrics', 6.0 * pW)
            for i in range(data_fix['nC']):
                # calculation of the z-scores
                i_match = c_data.i_match[i]
                if i_match >= 0:
                    # sets the signal feature metrics
                    c_data.match_intersect[:, i] = cfcn.calc_single_hist_metric(data_fix, data_free, i, i_match,
                                                                              True, cfcn.calc_hist_intersect)
                    c_data.match_wasserstain[:, i] = cfcn.calc_single_hist_metric(data_fix, data_free, i,
                                                                                i_match, True, cfcn.calc_wasserstein)
                    c_data.match_bhattacharyya[:, i] = cfcn.calc_single_hist_metric(data_fix, data_free, i,
                                                                                  i_match, True, cfcn.calc_bhattacharyya)

                    # sets the signal difference metrics
                    c_data.d_depth[i] = d_depth[i, i_match]
                    c_data.dtw_scale[i] = dtw_scale[i, i_match]
                    c_data.sig_corr[i] = cc_dtw[i, i_match]
                    c_data.sig_diff[i] = max(0.0, 1 - dd_dtw[i, i_match])
                    c_data.sig_intersect[i] = signal_metrics[i, i_match, 2]

                    # sets the isi metrics
                    c_data.isi_corr[i] = isi_metrics[i, i_match, 0]
                    c_data.isi_intersect[i] = isi_metrics[i, i_match, 1]

                    # sets the total match metrics
                    c_data.signal_feat[i, :] = signal_feat[i, i_match, :]
                    c_data.total_metrics[i, :] = total_metrics[i, i_match, :]
                    c_data.total_metrics_mean[i] = total_metrics_mean[i, i_match]

                    # sets the acceptance flag. for a cluster to be accepted, the following must be true:
                    #   * the ISI correlation coefficient must be > isi_corr_min
                    #   * the signal correlation coefficient must be > sig_corr_min
                    #   * the inter-signal euclidean distance must be < sig_diff_max
                    #   * all signal feature metric similarity scores must be > sig_feat_min
                    c_data.is_accept[i] = (c_data.isi_corr[i] > c_data.isi_corr_min) and \
                                          (c_data.sig_corr[i] > c_data.sig_corr_min) and \
                                          (c_data.sig_diff[i] > (1 - c_data.sig_diff_max)) and \
                                          (np.all(c_data.signal_feat[i, :] > c_data.sig_feat_min))
                else:
                    # sets NaN values for all the single value metrics
                    c_data.d_depth[i] = np.nan
                    c_data.dtw_scale[i] = np.nan
                    c_data.sig_corr[i] = np.nan
                    c_data.sig_diff[i] = np.nan
                    c_data.sig_intersect[i] = np.nan
                    c_data.isi_corr[i] = np.nan
                    c_data.isi_intersect[i] = np.nan
                    c_data.signal_feat[i, :] = np.nan
                    c_data.total_metrics[i, :] = np.nan
                    c_data.total_metrics_mean[i] = np.nan

                    # ensures the group is rejected
                    c_data.is_accept[i] = False

        # determines the number of spikes
        n_spike_fix = [len(x) / data_fix['tExp'] for x in data_fix['tSpike']]
        n_spike_free = [len(x) / data_free['tExp'] for x in data_free['tSpike']]

        # calculates the relative spiking rates (note - ratios are coverted so that they are all > 1)
        r_spike = np.divide(repmat(n_spike_fix, data_free['nC'], 1).T,
                            repmat(n_spike_free, data_fix['nC'], 1))
        r_spike[r_spike < 1] = 1 / r_spike[r_spike < 1]

        # calculates the pair-wise distances between the fixed/free probe depths
        d_depth = np.abs(np.subtract(repmat(data_fix['chDepth'], data_free['nC'], 1).T,
                                     repmat(data_free['chDepth'], data_fix['nC'], 1)))

        # determines the feasible fixed/free cluster groupings such that:
        #  1) the channel depth has to be <= d_max
        #  2) the relative spiking rates between clusters is <= r_max
        is_feas = np.logical_and(r_spike <= c_data.r_max, d_depth <= c_data.d_max)

        # determines the cluster matches from the old/new methods
        det_cluster_matches_old(c_data, is_feas, d_depth)
        det_cluster_matches_new(c_data, is_feas, d_depth, r_spike, w_prog)

    def calc_ccgram_types(self, calc_para, data):
        '''

        :return:
        '''

        # determines the indices of the experiment to be analysed
        if calc_para['calc_all_expt']:
            # case is all experiments are to be analysed
            i_expt = list(range(len(data)))
        else:
            # case is a single experiment is being analysed
            i_expt = [cf.get_expt_index(calc_para['calc_exp_name'], data)]

        # memory allocation
        d_copy = copy.deepcopy
        A, B, C = np.empty(len(i_expt), dtype=object), [[] for _ in range(5)], [[] for _ in range(4)]
        c_type, t_dur, t_event, ci_lo, ci_hi, ccG_T = d_copy(A), d_copy(A), d_copy(A), d_copy(A), d_copy(A), d_copy(A)

        #
        for i_ex in i_expt:
            # sets the experiment ID info based on the number of experiments being analysed
            if len(i_expt) == 1:
                # only one experiment is being analysed
                expt_id = None
            else:
                # multiple experiments are being analysed
                expt_id = [(i_ex+1), len(i_expt)]

            # retrieves the cluster information
            t_dur[i_ex], t_event[i_ex] = d_copy(C), d_copy(C)
            c_type[i_ex], ci_lo[i_ex], ci_hi[i_ex], ccG_T[i_ex] = d_copy(B), d_copy(B), d_copy(B), d_copy(B)
            ccG, ccG_xi, t_spike = data[i_ex]['ccGram'], data[i_ex]['ccGramXi'], data[i_ex]['tSpike']

            c_id = data[i_ex]['clustID']

            # runs the cc-gram type calculation function
            c_type0, t_dur[i_ex], t_event[i_ex], ci_hi0, ci_lo0, ccG_T0 = cfcn.calc_ccgram_types(
                       ccG, ccG_xi, t_spike, calc_para=calc_para, expt_id=expt_id, w_prog=self.work_progress, c_id=c_id)

            # sets the final values into their respective groupings
            for i in range(5):
                # sets the final type values and lower/upper bound confidence interval signals
                if len(c_type0[i]):
                    #
                    c_type[i_ex][i] = np.vstack(c_type0[i])

                    # sorts the values by the reference cluster index
                    i_sort = np.lexsort((c_type[i_ex][i][:, 1], c_type[i_ex][i][:, 0]))
                    c_type[i_ex][i] = c_type[i_ex][i][i_sort, :]

                    # reorders the duration/timing of the events (if they exist)
                    if i < len(t_dur[i_ex]):
                        t_dur[i_ex][i] = np.array(t_dur[i_ex][i])[i_sort]
                        t_event[i_ex][i] = np.array(t_event[i_ex][i])[i_sort]

                        ci_lo[i_ex][i] = (np.vstack(ci_lo0[i]).T)[:, i_sort]
                        ci_hi[i_ex][i] = (np.vstack(ci_hi0[i]).T)[:, i_sort]
                        ccG_T[i_ex][i] = (np.vstack(ccG_T0[i]).T)[:, i_sort]

        # returns the data as a dictionary
        return {'c_type': c_type, 't_dur': t_dur, 't_event': t_event,
                'ci_lo': ci_lo, 'ci_hi': ci_hi, 'ccG_T': ccG_T, 'calc_para': calc_para}

    def calc_shuffled_cluster_dist(self, calc_para, data):
        '''

        :return:
        '''

        # FINISH ME!
        pass

    ##########################################
    ####    CLUSTER MATCHING FUNCTIONS    ####
    ##########################################

    def calc_fix_free_correlation(self, data, calc_para, w_prog):
        '''

        :param data:
        :param plot_para:
        :param calc_para:
        :param w_prog:
        :return:
        '''

        # initialisations
        i_bin = ['5', '10'].index(calc_para['vel_bin'])
        tt_key = {'DARK1': 'Black', 'DARK': 'Black', 'LIGHT1': 'Uniform', 'LIGHT2': 'Uniform'}
        f_data, r_data, ff_corr = data.externd.free_data, data.rotation, data.comp.ff_corr
        n_bin = 2 * int(f_data.v_max / float(calc_para['vel_bin']))

        # determines matching experiment index and fix-to-free cell index arrays
        i_expt, f2f_map = cf.det_matching_fix_free_cells(data, apply_filter=False)

        # determines the global indices for each file
        nC = [len(x) for x in r_data.r_obj_kine.clust_ind[0]]
        ind_g = [np.arange(i0, i0 + n) for i0, n in zip(np.cumsum([0] + nC)[:-1], nC)]

        # memory allocation
        n_file, t_type = len(i_expt), f_data.t_type
        nan_bin = np.nan * np.ones(n_bin)
        ff_corr.sf_fix = np.empty((n_file, len(t_type)), dtype=object)
        ff_corr.sf_free = np.empty((n_file, len(t_type)), dtype=object)
        ff_corr.sf_corr = np.empty((n_file, len(t_type)), dtype=object)
        ff_corr.sf_corr_sh = np.empty((n_file, len(t_type)), dtype=object)
        ff_corr.sf_corr_sig = np.empty((n_file, len(t_type)), dtype=object)
        ff_corr.sf_grad = np.empty((n_file, len(t_type)), dtype=object)
        ff_corr.clust_id = np.empty(n_file, dtype=object)
        ff_corr.ind_g = np.empty(n_file, dtype=object)

        # sets the velocity spiking rates (depending on calculation type)
        if r_data.is_equal_time:
            # case is resampled spiking times
            vel_sf = dcopy(r_data.vel_sf_rs)
        else:
            # case is non-resampled spiking times
            vel_sf = dcopy(r_data.vel_sf)

        # loops through each external data file retrieving the spike frequency data and calculating correlations
        n_cell_tot, i_cell_tot = np.sum(np.array(nC)[i_expt]), 0
        for i_file in range(n_file):
            # initialisations for the current external data file
            ind_nw = ind_g[i_expt[i_file]]
            i_f2f = f2f_map[i_file][:, 1]
            s_freq = dcopy(f_data.s_freq[i_file][i_bin, :])

            # retrieves the spiking frequency data between the matched fixed/free cells for the current experiment
            for i_tt, tt in enumerate(t_type):
                # sets the fixed/free spiking frequency values
                ff_corr.sf_fix[i_file, i_tt] = np.nanmean(vel_sf[tt_key[tt]][:, :, ind_nw], axis=0).T
                ff_corr.sf_free[i_file, i_tt] = np.vstack([s_freq[i_tt][ii] if ii >= 0 else nan_bin for ii in i_f2f])

            # sets the cluster ID values
            is_ok = i_f2f >= 0
            i_expt_fix = cf.get_global_expt_index(data, data.comp.data[i_expt[i_file]])
            fix_clust_id = np.array(data._cluster[i_expt_fix]['clustID'])[is_ok]
            free_clust_id = np.array(data.externd.free_data.cell_id[i_file])[f2f_map[i_file][is_ok, 1]]
            ff_corr.clust_id[i_file] = np.vstack((fix_clust_id, free_clust_id)).T
            ff_corr.ind_g[i_file] = ind_nw

            # removes any spiking frequency data for where there is no matching data
            cfcn.calc_shuffled_sf_corr(ff_corr, i_file, calc_para, [i_cell_tot, n_cell_tot], w_prog)

            # increments the progressbar counter
            i_cell_tot += len(ind_nw)

        # sets the parameter values
        ff_corr.vel_bin = int(calc_para['vel_bin'])
        ff_corr.n_shuffle_corr = calc_para['n_shuffle']
        ff_corr.split_vel = int(calc_para['split_vel'])
        ff_corr.is_set = True

    ######################################
    ####    EYE TRACKING FUNCTIONS    ####
    ######################################

    def calc_eye_track_metrics(self, data, calc_para, w_prog):
        '''

        :param data:
        :param calc_para:
        :param w_prog:
        :return:
        '''

        def calc_position_diff(p0, dt, calc_para):
            '''

            :param p:
            :param dt:
            :param calc_para:
            :return:
            '''

            # retrieves the position values and calculates the rolling difference
            is_ok, n_frm = ~p0.isna(), p0.shape[0]

            # calculates the mid-point derivative values
            dp0 = p0.rolling(window=3, center=True).apply(lambda x: (x[2] - x[0]) / 2)

            # calculates the end-point derivative values (for the first/last valid values)
            i_ok = np.where(is_ok)[0]
            i0, i1 = i_ok[0], i_ok[-1]
            dp0.iloc[i0] = sum(np.multiply([-3,  4, -1], np.array(p0.iloc[i0:i0+3]).astype(float))) / 2
            dp0.iloc[i1] = sum(np.multiply([ 3, -4,  1], np.array(p0.iloc[i1-3:i1]).astype(float))) / 2

            # calculates the rolling median
            if calc_para['use_med_filt']:
                dp0_med = dp0.rolling(window=3, center=True).median()
            else:
                dp0_med = dp0

            # converts pd dataframes to float np-arrays (sets any NaN derivative values to zero)
            p = np.array(p0).astype(float)
            dp = np.array(dp0_med).astype(float) / (1000. * dt)
            dp[~is_ok] = 0

            # removes any outliers (regions where the derivative is greater than dp_max)
            i_grp = cf.get_index_groups(np.abs(dp) > calc_para['dp_max'])
            for ig in cf.expand_index_groups(i_grp, 2, n_frm):
                dp[ig], p[ig] = 0, np.nan

            # removes the baseline component (if required)
            if calc_para['rmv_baseline']:
                w_frm = 70 / n_frm
                dp_bl = lowess(dp, np.arange(n_frm), w_frm, return_sorted=False)
                dp -= dp_bl

            # returns the derivative array
            return dp - np.nanmean(dp), p

        def det_movement_events(p_pos, dp_pos, calc_para, n_pre, n_post, t_frm):
            '''

            :param dp_pos:
            :return:
            '''

            def get_event_sig_seg(p_pos, i_grp0, n_pre, n_post, n_frm):
                '''

                :param p_pos:
                :param i_grp0:
                :param n_frm:
                :return:
                '''

                def get_sig_seg(y_sig, i_grp0, n_pp, n_frm=None):
                    '''

                    :param dp_pos:
                    :param i_grp0:
                    :param n_frm:
                    :return:
                    '''

                    if n_frm is None:
                        # case is the signal values preceding the onset point
                        return list(y_sig[max(0, (i_grp0 - n_pp)):(i_grp0 + 1)])
                    else:
                        # case is the signal values proceding the onset point
                        return list(y_sig[(i_grp0 + 1):min(n_frm - 1, i_grp0 + (1 + n_pp))])

                return np.array(get_sig_seg(p_pos, i_grp0, n_pre) + get_sig_seg(p_pos, i_grp0, n_post, n_frm))

            # initialisations
            n_frm, i_ofs = len(t_frm), 1
            t_evnt, y_evnt = [], []
            n_sd, dp_max, n_event_win = calc_para['n_sd'], calc_para['dp_max'], n_pre + n_post + 1

            # thresholds the position derivative values
            b_arr, sgn_arr = np.abs(dp_pos) >= np.nanstd(dp_pos) * n_sd, np.sign(dp_pos)
            if np.any(b_arr):
                # if there are any derivative values greater than threshold, then determine the index groups of the
                # continguous points that are greater than threshold. from this determine the max absolute amplitudes within
                # these groups and the start indices of each group
                i_grp = cf.get_index_groups(b_arr)
                grp_mx, i_grp0 = [np.max(np.abs(dp_pos[x])) for x in i_grp], np.array([(x[0] - i_ofs) for x in i_grp])

                # determines the groups that are within the event window (and have a position derivative less than the
                # maximum derivative parameter value, dp_max)
                di_grp0 = np.diff(i_grp0)
                is_ok = np.array([(x >= n_pre) and (x <= (n_frm - n_post)) for x in i_grp0])
                for ig in np.where(di_grp0 < n_event_win)[0]:
                    if sgn_arr[i_grp0[ig]] * sgn_arr[i_grp0[ig + 1]] < 0:
                        # if the thresholded groups have differing derivative signs, then ignore both groups
                        is_ok[ig:ig+2] = False
                    else:
                        # otherwise, remove the thresholded group with the lower amplitude peak
                        is_ok[1 + (grp_mx[ig] > grp_mx[ig + 1])] = False

                # memory allocation
                n_evnt = len(is_ok)
                t_evnt0, y_evnt0 = np.zeros(n_evnt), np.zeros((n_evnt, n_event_win))

                # removes the ignored contiguous groups
                for i in range(n_evnt):
                    if is_ok[i]:
                        y_evnt_nw = get_event_sig_seg(p_pos, i_grp0[i], n_pre, n_post, n_frm)
                        if not np.any(np.isnan(y_evnt_nw)):
                            y_evnt0[i, :], t_evnt0[i] = y_evnt_nw, t_frm[i_grp0[i]]
                        else:
                            is_ok[i] = False

                # removes the
                t_evnt0, y_evnt0 = t_evnt0[is_ok], y_evnt0[is_ok]

                # appends the time stamps of the events for both eye movement types
                i_sgn = np.array([int(sgn_arr[x + i_ofs] > 0) for x in i_grp0[is_ok]])
                t_evnt.append([t_evnt0[i_sgn == i] for i in range(2)])

                # sets the sub-signal/mean sub-signal values for both eye movement types
                y_evnt_tmp = [y_evnt0[i_sgn == i, :] for i in range(2)]
                y_evnt.append([np.subtract(x, x[:, n_pre][:, None]) if len(x) else [] for x in y_evnt_tmp])

            else:
                # if no event, then set empty time/signal events for both types
                t_evnt.append([[], []])
                y_evnt.append([[], []])

            # returns the event time/signal arrays
            return t_evnt, y_evnt

        # retrieves the eye-tracking class object
        et_class = data.externd.eye_track
        n_file = len(et_class.et_data)

        # sets the pre/post event duration
        n_pre, n_post = calc_para['n_pre'], calc_para['n_post']

        # memory allocation
        dt = 1 / et_class.fps
        A = np.empty(n_file, dtype=object)
        et_class.t_evnt, et_class.y_evnt = dcopy(A), dcopy(A)
        et_class.t_type = list(np.unique(cf.flat_list([x.t_type for x in et_class.et_data])))

        # loops through each of the file calculating the eye-movement events
        for i_file, et_d in enumerate(et_class.et_data):
            # updates the progress bar string
            w_str = 'Detecting Movement Events (Expt {0} of {1})'.format(i_file + 1, n_file)

            # memory allocation
            n_tt = len(et_d.t_type)
            B = np.empty(len(et_class.t_type), dtype=object)
            et_class.t_evnt[i_file], et_class.y_evnt[i_file] = dcopy(B), dcopy(B)

            # loops through each of the trial types calculate the eye-movement events
            for i_tt in range(n_tt):
                # updates the progress-bar
                w_prog.emit(w_str, 100. * ((i_file * n_tt + i_tt) / (n_tt * n_file)))

                # retrieves the position values
                p0 = dcopy(et_d.p_pos[i_tt])
                if calc_para['use_med_filt']:
                    # calculates the rolling median (if required)
                    p0 = p0.rolling(window=3, center=True).median()

                # calculates the position difference values
                dp, p = calc_position_diff(p0, dt, calc_para)

                # calculates the events/signal sub-segments for all events
                j_tt = et_class.t_type.index(et_class.et_data[i_file].t_type[i_tt])
                t_frm = np.arange(len(p)) / et_class.fps
                tt, yy = det_movement_events(p, dp, calc_para, n_pre, n_post, t_frm)
                et_class.t_evnt[i_file][j_tt], et_class.y_evnt[i_file][j_tt] = tt[0], yy[0]

        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # updates the calculation parameters
        et_class.use_med_filt = calc_para['use_med_filt']
        et_class.rmv_baseline = calc_para['rmv_baseline']
        et_class.dp_max = calc_para['dp_max']
        et_class.n_sd = calc_para['n_sd']
        et_class.n_pre = calc_para['n_pre']
        et_class.n_post = calc_para['n_post']
        et_class.is_set = True

    def calc_eye_track_corr(self, data, calc_para, w_prog):
        '''

        :param data:
        :param calc_para:
        :param w_prog:
        :return:
        '''

        def get_trial_group_start_time(r_info, tt_c0):
            '''

            :param c:
            :param tt_c:
            :return:
            '''

            def get_expt_time_span(ind0, i_type):
                '''

                :param ind0:
                :return:
                '''

                if i_type == 0:
                    # returns the first trial index
                    return ind0[0]
                else:
                    # determines the 2nd order difference in the trial start times
                    dind0 = np.zeros(len(ind0), dtype=int)
                    dind0[2:] = np.diff(ind0, 2)
    
                    #
                    i_diff = np.where(np.abs(dind0) > 1e10)[0]
                    return ind0[i_diff[0]]

            # sets the trial type (removes any extra indices at the end of the trial type string)
            i_type = int(tt_c0[-1] == '2')
            tt = tt_c0 if (i_type == 0) else tt_c0[:-1]

            # retrieves the start time of the trial grouping
            return get_expt_time_span(r_info['wfm_para'][tt]['ind0'], i_type)

        def get_grouping_spike_times(t_sp, t_exp, t0):
            '''

            :param t_sp_c:
            :param t_exp:
            :param t0:
            :return:
            '''

            # memory allocation
            n_cell = len(t_sp)
            t_sp_h = np.zeros((n_cell, len(t_exp)))

            # calculates the time spiking histograms (for each cell) downsampled to that of the eye-tracking analysis
            for i_cell in range(n_cell):
                # retrieves the spike times for the current cell
                t_sp_tmp = t_sp[i_cell] / 1000
                t_sp_grp = t_sp_tmp[np.logical_and(t_sp_tmp >= t0, t_sp_tmp <= t0 + t_exp[-1])] - t0

                # calculates the spike time histogram (time bins are set for the eye-tracking analysis)
                t_sp_h[i_cell, 1:] = np.histogram(t_sp_grp, bins=t_exp)[0]

            # returns the histogram arrays
            return t_sp_h

        def get_event_spike_times(t_sp_h, t_evnt, dt_et, calc_para):
            '''

            :param t_sp_h:
            :param t_evnt:
            :param calc_para:
            :return:
            '''

            # memory allocation
            n_cell, n_frm = np.shape(t_sp_h)
            sp_evnt = np.empty(len(t_evnt), dtype=object)

            # sets the pre/post event duration
            n_pre, n_post = calc_para['n_pre'], calc_para['n_post']
            n_pts = n_pre + n_post + 1

            # retrieves the spike time events for each eye-movement type
            for i in range(len(t_evnt)):
                # sets the indices of the events (ensures all frames are within that of the eye-tracking analysis)
                i_evnt = np.round(t_evnt[i] / dt_et).astype(int)
                i_evnt = i_evnt[np.logical_and((i_evnt - n_pre) >= 0, (i_evnt + n_post) < n_frm)]

                # memory allocation for eye-movement type
                n_evnt = len(t_evnt[i])
                sp_evnt[i] = np.zeros((n_evnt, n_pts, n_cell))

                # retrieves the spike time histogram values over each cell/eye-movement event
                for j in range(n_evnt):
                    i_rng = np.arange(i_evnt[j] - n_pre, i_evnt[j] + n_post + 1)
                    sp_evnt[i][j, :, :] = t_sp_h[:, i_rng].T

            # returns the array
            return sp_evnt

        # initialisations and memory allocation
        et_class = data.externd.eye_track
        exp_file = [cf.extract_file_name(x['expFile']) for x in data.cluster]
        n_exp, dt_et = et_class.n_file, 1. / et_class.fps

        # memory allocation
        A = np.empty(n_exp, dtype=object)
        t_sp_h, sp_evnt, y_corr, p_corr = dcopy(A), dcopy(A), dcopy(A), dcopy(A)

        # loops through each experiment calculating the spiking rate/eye movement correlations
        for i_exp, et_d in enumerate(et_class.et_data):
            # initialisations
            n_tt, pw0 = len(et_d.t_type), 1 / n_exp

            # memory allocation
            B = np.empty(n_tt, dtype=object)
            t_sp_h[i_exp], sp_evnt[i_exp], y_corr[i_exp], p_corr[i_exp] = dcopy(B), dcopy(B), dcopy(B), dcopy(B)

            # retrieves the rotation info of the corresponding expt
            c = data._cluster[cf.det_likely_filename_match(exp_file, et_class.exp_name[i_exp])]
            r_info, dt_c, t_sp_c = c['rotInfo'], 1. / c['sFreq'], c['tSpike']

            # loops through each trial type calculating the correlations
            for i_tt, tt in enumerate(et_d.t_type):
                # updates the progressbar
                tt_c = tt.capitalize()
                w_str = 'Calculating Correlations (Expt {0}/{1} - {2})'.format(i_tt + 1, n_tt, tt_c)
                w_prog.emit(w_str, 100. * (pw0 + (i_tt / n_tt)))

                # sets the time vector over the eye-tracking analysis
                j_tt = et_class.t_type.index(et_class.et_data[i_exp].t_type[i_tt])
                t_exp = np.arange(len(et_d.p_pos[j_tt])) * dt_et

                # retrieves the spike times over the duration of the eye tracking analysis
                t0 = get_trial_group_start_time(r_info, tt_c) * dt_c
                t_sp_h[i_exp][j_tt] = get_grouping_spike_times(t_sp_c, t_exp, t0)

                # retrieves the spike times traces surrounding the times of the eye movement
                t_evnt = et_class.t_evnt[i_exp][j_tt]
                sp_evnt[i_exp][j_tt] = get_event_spike_times(t_sp_h[i_exp][j_tt], t_evnt, dt_et, calc_para)

                # calculates the correlations between each cell and the eye movement events
                y_evnt = et_class.y_evnt[i_exp][j_tt]
                y_corr[i_exp][j_tt], p_corr[i_exp][j_tt] = cfcn.calc_event_correlation(y_evnt, sp_evnt[i_exp][j_tt])

        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # sets the arrays into the eye-tracking class object
        data.externd.eye_track.t_sp_h = t_sp_h
        data.externd.eye_track.sp_evnt = sp_evnt
        data.externd.eye_track.y_corr = y_corr
        data.externd.eye_track.p_corr = p_corr

        # final update of the progressbar
        w_prog.emit('Correlation Calculations Complete!', 100.)

    ######################################
    ####    AHV ANALYSIS FUNCTIONS    ####
    ######################################

    def calc_corr_fit_para(self, data, plot_para, calc_para, w_prog):
        '''

        :param data:
        :param plot_para:
        :param calc_para:
        :param w_prog:
        :return:
        '''

        def calc_sf_lin_para(xi, sf, peak_hz, err_type):
            '''

            :param sf:
            :return:
            '''

            # memory allocation
            n_cell = np.shape(sf)[0]
            sf_slope, sf_int = np.zeros(n_cell), np.zeros(n_cell)
            sf_err = np.zeros(n_cell)

            # calculates the linear parameters for each cell
            for i_cell in range(n_cell):
                # slope/intercept calculation
                sf_calc = sf[i_cell]
                l_fit = linregress(xi, sf_calc / peak_hz[i_cell])
                sf_slope[i_cell], sf_int[i_cell] = l_fit.slope, l_fit.intercept

                # error calculation
                dsf_calc = (sf_calc - sf_calc[0])
                dsf_max = np.max(np.abs(dsf_calc))

                if (dsf_max > 0) and (err_type is not None):
                    if err_type == 'Covariance':
                        _, pcov = curve_fit(lin_func, xi, dsf_calc / dsf_max)
                        sf_err[i_cell] = np.sqrt(pcov[0][0])

                    elif err_type == 'Sum-of-Squares':
                        p_fit_err = np.polyfit(xi, dsf_calc / dsf_max, 1, full=True)
                        sf_err[i_cell] = p_fit_err[1][0]

                    elif err_type == 'Standard Error':
                        l_fit_err = linregress(xi, dsf_calc / dsf_max)
                        sf_err[i_cell] = l_fit_err.stderr

            # returns the array
            return sf_slope, sf_int, sf_err

        # appends the fields to the rotation class object
        r_data = data.rotation
        if not hasattr(r_data, 'sf_fix_slope'):
            r_data.sf_fix_slope = None
            r_data.sf_fix_int = None
            r_data.sf_fix_err = None
            r_data.peak_hz_fix = None

        # applies the rotation filter to the dataset
        r_obj = RotationFilteredData(data, plot_para['rot_filt'], None, None, True, 'Whole Experiment', False)
        n_filt = r_obj.n_filt

        # determines the common cell indices for each filter types
        t_type_full = [x['t_type'][0] for x in r_obj.rot_filt_tot]
        i_cell_b, _ = cfcn.get_common_filtered_cell_indices(data, r_obj, t_type_full, True)

        # retrieves the spiking frequencies
        r_data = data.rotation
        sf = dcopy(r_data.vel_sf_mean)
        err_type = None if 'err_type' not in calc_para else calc_para['err_type']
        norm_sf = False if 'norm_sf' not in calc_para else calc_para['norm_sf']

        # sets up the velocity bin values
        v_max, v_bin = 80, r_data.vel_bin_corr
        xi_bin = np.arange(-v_max + v_bin / 2, v_max, v_bin)
        is_pos = xi_bin > 0
        n_bin = sum(is_pos)

        # memory allocation
        A = np.empty((2, n_filt), dtype=object)
        sf_slope, sf_int, sf_err, peak_hz = dcopy(A), dcopy(A), dcopy(A), np.empty(n_filt, dtype=object)

        if norm_sf:
            # for each filter type, calculate the linear fit parameters
            dsf_filt = np.empty(n_filt, dtype=object)
            peak_hz_filt = np.empty(n_filt, dtype=object)
            for i_filt, tt in enumerate(t_type_full):
                # calculates the slope/intercept values
                sf_filt = sf[tt][i_cell_b[i_filt], :]

                #
                sf_comb = [np.vstack(sf_filt[:, 0])[:, ::-1], np.vstack(sf_filt[:, 1])]
                dsf_filt[i_filt] = [sf - repmat(sf[:, 0], n_bin, 1).T for sf in sf_comb]

                # determines the peak frequency
                peak_hz_filt[i_filt] = np.max(np.abs(np.hstack((dsf_filt[i_filt][0], dsf_filt[i_filt][1]))), axis=1)

            # determines the peak spiking frequency across all conditions
            peak_hz = np.max(np.abs(np.vstack(peak_hz_filt)), axis=0)

        # for each filter type, calculate the linear fit parameters
        for i_filt, tt in enumerate(t_type_full):
            # updates the progress bar
            w_str = 'Linear Fit Calculations ({0})'.format(tt)
            w_prog.emit(w_str, 100. * i_filt / len(t_type_full))

            if norm_sf:
                # sets the positive/negative spiking frequencies
                sf_neg, sf_pos = dsf_filt[i_filt][0], dsf_filt[i_filt][1]

            else:
                # calculates the slope/intercept values
                sf_filt = sf[tt][i_cell_b[i_filt], :]

                # sets the positive/negative spiking frequencies
                sf_neg, sf_pos = np.vstack(sf_filt[:, 0])[:, ::-1], np.vstack(sf_filt[:, 1])
                peak_hz = np.ones(np.shape(sf_neg)[0])

            # calculates the spiking freuency slope, intercept and errors
            sf_slope[0, i_filt], sf_int[0, i_filt], sf_err[0, i_filt] = \
                                    calc_sf_lin_para(xi_bin[is_pos], sf_neg, peak_hz, err_type)
            sf_slope[1, i_filt], sf_int[1, i_filt], sf_err[1, i_filt] = \
                                    calc_sf_lin_para(xi_bin[is_pos], sf_pos, peak_hz, err_type)


        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # sets the class object fields
        r_data.sf_fix_slope = sf_slope
        r_data.sf_fix_int = sf_int
        r_data.sf_fix_err = sf_err
        r_data.r_obj_sf = r_obj
        r_data.peak_hz_fix = peak_hz

    #######################################
    ####    FREELY MOVING FUNCTIONS    ####
    #######################################

    def calc_cell_fit_residual(self, data, calc_para, w_prog):
        '''

        :param data:
        :param calc_para:
        :param w_prog:
        :return:
        '''

        def calc_cell_res_gain(xi, sf_split):
            '''

            :param sf_cell:
            :param xi:
            :param is_pos:
            :return:
            '''

            def calc_sf_res(xi, sf):
                '''

                :param xi:
                :param sf:
                :return:
                '''

                # fits a linear equation to the spiking frequencies
                l_fit = LinearRegression(fit_intercept=False).fit(xi, sf)
                # p_fit = np.polyfit(xi, sf, 1)

                # calculates the absolute residual values (normalising by the maximum spiking rate)
                return np.abs(l_fit.predict(xi) - sf)

            # memory allocation
            n_type = np.shape(sf_split)[1]
            sf_gain, sf_res = np.empty(n_type, dtype=object), np.empty(n_type, dtype=object)

            # calculates the overall spiking frequency maximum
            # sf_max = np.max([[np.max(y) for y in x] for x in sf_split])
            # if sf_max == 0:
            sf_max = np.max([[np.max(np.abs(y)) for y in x] for x in sf_split])

            # calculates/sets the residual/gain values for each direction/condition type
            for i_type in range(n_type):
                sf_gain[i_type] = np.array(cf.flat_list(sf_split[:, i_type]))
                sf_res[i_type] = np.array([calc_sf_res(xi, sf / np.max(np.abs(sf))) for sf in sf_split[:, i_type]]).flatten()

            # calculates the normalised absolute residuals from the linear fits to the spiking frequencies
            return sf_gain, sf_res, sf_max

        # initialisations
        f_data = data.externd.free_data

        # ensures the freely moving class calculation fields have been set (initialies them if they have not)
        if not hasattr(f_data, 'sf_gain'):
            setattr(f_data, 'sf_gain', None)
            setattr(f_data, 'sf_res', None)
            setattr(f_data, 'sf_vbin', None)
            setattr(f_data, 'sf_tt', None)
            setattr(f_data, 'sf_max', None)

        # initialisations
        t_type = ['DARK', calc_para['lcond_type']]
        v_bin, v_max = int(calc_para['vel_bin']), 80.
        i_bin = [5, 10].index(v_bin)
        i_tt = [list(f_data.t_type).index(tt) for tt in t_type]

        # sets up the velocity bin array
        xi = np.arange(-v_max + v_bin / 2, v_max, v_bin)

        # memory allocation
        n_type = len(t_type)
        A = np.empty(f_data.n_file, dtype=object)
        sf_res, sf_gain, sf_max = dcopy(A), dcopy(A), dcopy(A)

        ##########################################
        ####    GAIN/RESIDUAL CALCULATIONS    ####
        ##########################################

        # memory allocation and other initialisations
        is_pos = xi > 0
        n_bin, n_dir = int(len(xi) / 2), 2

        # retrieves the spiking frequencies for the velocity bin size
        sf_bin = [sf[i_bin] for sf in f_data.s_freq]

        # calculates the gain/residuals for each file
        for i_file in range(f_data.n_file):
            # updates the waitbar progress
            w_str = 'Gain/Residual Calculations ({0} of {1})'.format(i_file + 1, f_data.n_file)
            w_prog.emit(w_str, 100 * (i_file / f_data.n_file))

            # memory allocation
            n_cell = np.shape(sf_bin[i_file][0])[0]
            B = np.empty((n_cell, n_type), dtype=object)
            sf_res[i_file], sf_gain[i_file], sf_max[i_file] = dcopy(B), dcopy(B), np.zeros(n_cell)

            # calculates the gain/residuals for each cell/condition type
            for i_cell in range(n_cell):
                # memory allocation
                sf_split = np.empty((n_dir, n_type), dtype=object)

                # splits the spiking frequency into positive/negative velocities for each condition type
                for i_type in range(n_type):
                    # retrieves the spiking frequency for the current cell/condition type and separates
                    sf_cell = sf_bin[i_file][i_tt[i_type]][i_cell]
                    sf_split0 = [sf_cell[~is_pos][::-1], sf_cell[is_pos]]

                    # removes the first time bin from each direction
                    for i_dir in range(n_dir):
                        sf_split[i_dir, i_type] = sf_split0[i_dir] - sf_split0[i_dir][0]

                # calculates the gain/residual for condition type
                sf_gain[i_file][i_cell, :], sf_res[i_file][i_cell, :], sf_max[i_file][i_cell] = \
                                                        calc_cell_res_gain(xi[is_pos].reshape(-1, 1), sf_split)

        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # sets the class object fields
        f_data.sf_gain = sf_gain
        f_data.sf_res = sf_res
        f_data.sf_vbin = int(calc_para['vel_bin'])
        f_data.sf_tt = t_type
        f_data.sf_max = sf_max

    #########################################
    ####    ROTATION LDA CALCULATIONS    ####
    #########################################

    def run_temporal_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''

        :param data:
        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial_max:
        :return:
        '''

        # initialisations and memory allocation
        d_data, w_prog = data.discrim.temp, self.work_progress
        d_data.lda, d_data.y_acc = np.empty(2, dtype=object), np.empty(2, dtype=object)

        # retrieves the rotation phase duration
        r_obj = RotationFilteredData(data, r_filt, None, None, True, 'Whole Experiment', False)
        t_phase = r_obj.t_phase[0][0]

        ################################################
        ####    DIFFERING PHASE LDA CALCULATIONS    ####
        ################################################

        # creates a copy of the calculation parameters for the differing phase duration LDA calculations
        calc_para_phs = dcopy(calc_para)
        calc_para_phs['t_ofs_rot'] = 0

        # memory allocation
        dt_phs = np.arange(calc_para['dt_phase'], t_phase, calc_para['dt_phase'])
        d_data.lda[0], d_data.y_acc[0] = np.empty(len(dt_phs), dtype=object), np.empty(len(dt_phs), dtype=object)

        # loops through each of the phase discretisations calculating the LDA calculations
        n_phs = len(dt_phs)
        for i_phs in range(n_phs):
            # updates the progress bar
            w_str = 'Duration LDA Calculations (Group {0} of {1})'.format(i_phs + 1, n_phs)
            w_prog.emit(w_str, 50. * ((i_phs + 1)/ n_phs))

            # updates the phase duration parameter
            calc_para_phs['t_phase_rot'] = dt_phs[i_phs]

            # runs the rotation analysis for the current configuration
            result = cfcn.run_rot_lda(data, calc_para_phs, r_filt, i_expt, i_cell, n_trial_max)
            if isinstance(result, bool):
                # if there was an error, then return a false flag value
                return False
            else:
                # otherwise, store the lda/accuracy values
                d_data.lda[0][i_phs], d_data.y_acc[0][i_phs] = result[0], result[1]

        #################################################
        ####    DIFFERING OFFSET LDA CALCULATIONS    ####
        #################################################

        # creates a copy of the calculation parameters for the differing offset LDA calculations
        calc_para_ofs = dcopy(calc_para)
        calc_para_ofs['t_phase_rot'] = calc_para['t_phase_const']

        # sets the differing phase/offset value arrays
        dt_ofs = np.arange(0., t_phase - calc_para['t_phase_const'], calc_para['t_phase_const'])
        d_data.lda[1], d_data.y_acc[1] = np.empty(len(dt_ofs), dtype=object), np.empty(len(dt_ofs), dtype=object)

        # loops through each of the phase discretisations calculating the LDA calculations
        n_ofs = len(dt_ofs)
        for i_ofs in range(n_ofs):
            # updates the progress bar
            w_str = 'Offset LDA Calculations (Group {0} of {1})'.format(i_ofs + 1, n_ofs)
            w_prog.emit(w_str, 50. * (1 + ((i_ofs + 1) / n_ofs)))

            # updates the phase duration parameter
            calc_para_ofs['t_ofs_rot'] = dt_ofs[i_ofs]

            # runs the rotation analysis for the current configuration
            result = cfcn.run_rot_lda(data, calc_para_ofs, r_filt, i_expt, i_cell, n_trial_max)
            if isinstance(result, bool):
                # if there was an error, then return a false flag value
                return False
            else:
                # otherwise, store the lda/accuracy values
                d_data.lda[1][i_ofs], d_data.y_acc[1][i_ofs] = result[0], result[1]

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # retrieves the LDA solver parameter fields
        lda_para = calc_para['lda_para']

        # sets the solver parameters
        d_data.lda = 1
        d_data.exp_name = result[2]
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max)
        d_data.lda_trial_type = cfcn.get_glob_para('lda_trial_type')

        # sets the other calculation parameters
        d_data.dt_phs = calc_para['dt_phase']
        d_data.dt_ofs = calc_para['dt_ofs']
        d_data.phs_const = calc_para['t_phase_const']

        # sets the other variables/parameters of interest
        d_data.xi_phs = dt_phs
        d_data.xi_ofs = dt_ofs

        # returns a true value indicating the calculations were successful
        return True

    def run_shuffled_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''

        :param data:
        :param calc_para:
        :param r_filt:00
        :param i_expt:
        :param i_cell:
        :param n_trial_max:
        :return:
        '''

        # initialisations and memory allocation
        d_data, w_prog = data.discrim.shuffle, self.work_progress
        if d_data.lda is not None:
            return True

        # retrieves the phase duration/offset values
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)
        if t_ofs is None:
            t_ofs, t_phase = 0, 3.5346

        ###############################################
        ####    SHUFFLED TRIAL LDA CALCULATIONS    ####
        ###############################################

        # creates a reduce data object and creates the rotation filter object
        n_ex, n_sh, n_cond = len(i_expt), calc_para['n_shuffle'], len(r_filt['t_type'])
        d_data.y_acc = np.empty((n_ex, n_cond + 1, n_sh), dtype=object)
        n_sp = np.empty((n_ex, n_sh), dtype=object)

        # runs the LDA for each of the shuffles
        for i_sh in range(n_sh):
            # updates the progressbar
            w_str = 'Shuffled Trial LDA (Shuffle #{0} of {1})'.format(i_sh + 1, n_sh)
            w_prog.emit(w_str, 100. * (i_sh / n_sh))

            # runs the rotation analysis for the current configuration
            result = cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max, is_shuffle=True)
            if isinstance(result, bool):
                # if there was an error, then return a false flag value
                return False
            else:
                # otherwise, store the lda/accuracy values
                d_data.y_acc[:, :, i_sh], n_sp[:, i_sh] = result[1], result[3]
                if i_sh == 0:
                    # sets the experiment names (for the first shuffle only)
                    d_data.exp_name == result[2]

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # retrieves the LDA solver parameter fields
        lda_para = calc_para['lda_para']

        # sets the solver parameters
        d_data.lda = 1
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max)
        d_data.lda_trial_type = cfcn.get_glob_para('lda_trial_type')

        # sets the phase offset/duration parameters
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # sets the other parameters
        d_data.nshuffle = n_sh
        # d_data.bsz = calc_para['b_sz']

        # calculates the correlations
        n_sp_tot = [np.dstack(x) for x in n_sp]
        cfcn.calc_noise_correl(d_data, n_sp_tot)

        # returns a true value indicating the calculations were successful
        return True

    def run_individual_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''
        
        :param data: 
        :param calc_para: 
        :param r_filt: 
        :param i_expt: 
        :param i_cell: 
        :param n_trial_max: 
        :return: 
        '''

        # initialisations and memory allocation
        d_data, w_prog = data.discrim.indiv, self.work_progress

        # removes normalisation for the individual cell LDA calculations
        _calc_para = dcopy(calc_para)
        # _calc_para['lda_para']['is_norm'] = False

        ################################################
        ####    INDIVIDUAL CELL LDA CALCULATIONS    ####
        ################################################

        # creates a reduce data object and creates the rotation filter object
        n_ex = len(i_expt)
        A = np.empty(n_ex, dtype=object)
        d_data.y_acc, d_data.exp_name = dcopy(A), dcopy(A)
        n_cell = [len(i_c) for i_c in i_cell]

        #
        for i_ex in range(n_ex):
            # creates a copy a copy of the accepted cell array for the analysis
            _i_cell = np.zeros(n_cell[i_ex], dtype=bool)
            _n_cell = np.sum(i_cell[i_ex])
            d_data.y_acc[i_ex] = np.zeros((_n_cell, 1 + len(calc_para['lda_para']['comp_cond'])))

            # runs the LDA analysis for each of the cells
            for i, i_c in enumerate(np.where(i_cell[i_ex])[0]):
                # updates the progressbar
                w_str = 'Single Cell LDA (Cell {0}/{1}, Expt {2}/{3})'.format(i + 1, _n_cell, i_ex + 1, n_ex)
                w_prog.emit(w_str, 100. * (i_ex + i / _n_cell) / n_ex)

                # sets the cell for analysis and runs the LDA
                _i_cell[i_c] = True
                results = cfcn.run_rot_lda(data, _calc_para, r_filt, [i_expt[i_ex]], [_i_cell], n_trial_max)
                if isinstance(results, bool):
                    # if there was an error, then return a false flag value
                    return False
                else:
                    # otherwise, reset the cell boolear flag
                    _i_cell[i_c] = False

                # stores the results from the single cell LDA
                d_data.y_acc[i_ex][i, :] = results[1]
                if i == 0:
                    # if the first iteration, then store the experiment name
                    d_data.exp_name[i_ex] = results[2]

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # retrieves the LDA solver parameter fields
        lda_para = calc_para['lda_para']
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

        # sets the solver parameters
        d_data.lda = 1
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max)
        d_data.lda_trial_type = cfcn.get_glob_para('lda_trial_type')

        # sets the phase offset/duration
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # returns a true value indicating the calculations were successful
        return True

    def run_pooled_lda(self, pool, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''

        :param data:
        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial_max:
        :return:
        '''

        def run_pooled_lda_expt(data, calc_para, r_filt, i_expt0, i_cell0, n_trial_max, n_cell, n_sp0):
            '''

            :param data:
            :param calc_para:
            :param r_filt:
            :param i_expt:
            :param i_cell:
            :param n_trial_max:
            :param xi:
            :return:
            '''

            while 1:
                # sets the required number of cells for the LDA analysis
                if calc_para['pool_expt']:
                    n_sp = n_sp0[:, np.random.permutation(np.size(n_sp0, axis=1))[:n_cell]]
                    i_cell, i_expt = i_cell0, i_expt0

                else:
                    i_cell = dcopy(i_cell0)
                    is_keep = np.ones(len(i_expt0), dtype=bool)

                    for i_ex in range(len(i_expt0)):
                        # determines the original valid cells for the current experiment
                        ii = np.where(i_cell0[i_ex])[0]
                        if len(ii) < n_cell:
                            is_keep[i_ex] = False
                            continue

                        # from these cells, set n_cell cells as being valid (for analysis purposes)
                        i_cell[i_ex][:] = False
                        i_cell[i_ex][ii[np.random.permutation(len(ii))][:n_cell]] = True

                    # removes the experiments which did not have the min number of cells
                    i_expt, i_cell, n_sp = i_expt0[is_keep], i_cell[is_keep], n_sp0

                # runs the LDA
                results = cfcn.run_rot_lda(data, calc_para, r_filt, i_expt, i_cell, n_trial_max, n_sp0=n_sp)
                if not isinstance(results, bool):
                    # if successful, then exit the loop
                    break

            # returns the decoding accuracy values
            if calc_para['pool_expt']:
                return results[1]
            else:
                # retrieves the results from the LDA
                y_acc0 = results[1]

                # sets the values into
                y_acc = np.nan * np.ones((len(is_keep), np.size(y_acc0, axis=1)))
                y_acc[is_keep, :] = y_acc0
                return y_acc

        # initialisations
        d_data = data.discrim.part
        w_prog, n_sp = self.work_progress, None

        #############################################
        ####    PARTIAL CELL LDA CALCULATIONS    ####
        #############################################

        # initialisations
        if calc_para['pool_expt']:
            # case is all experiments are pooled

            # initialisations and memory allocation
            ind_t, n_sp = np.arange(n_trial_max), []
            t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

            # creates a reduce data object and creates the rotation filter object
            data_tmp = cfcn.reduce_cluster_data(data, i_expt, True)
            r_obj = RotationFilteredData(data_tmp, r_filt, None, None, True, 'Whole Experiment', False,
                                         t_ofs=t_ofs, t_phase=t_phase)

            # sets up the LDA data/group index arrays across each condition
            for i_filt in range(r_obj.n_filt):
                # retrieves the time spikes for the current filter/experiment, and then combines into a single
                # concatenated array. calculates the final spike counts over each cell/trial and appends to the
                # overall spike count array
                A = dcopy(r_obj.t_spike[i_filt])[:, ind_t, :]
                if r_obj.rot_filt['t_type'][i_filt] == 'MotorDrifting':
                    # case is motordrifting (swap phases)
                    t_sp_tmp = np.hstack((A[:, :, 2], A[:, :, 1]))
                else:
                    # case is other experiment conditions
                    t_sp_tmp = np.hstack((A[:, :, 1], A[:, :, 2]))

                # calculates the spike counts and appends them to the count array
                n_sp.append(np.vstack([np.array([len(y) for y in x]) for x in t_sp_tmp]))

            # combines the spike counts/group indices into the final combined arrays
            n_sp, n_expt, i_expt_lda = np.hstack(n_sp).T, 1, np.array([i_expt[0]])
            xi = cfcn.get_pool_cell_counts(data, calc_para['lda_para'], 1)

            # reduces the cells to the selected cell type
            _, _, i_cell0, _, _ = cfcn.setup_lda(data, {'lda_para': calc_para['lda_para']}, None)
            n_sp = n_sp[:, np.hstack(i_cell0)]
            i_cell = np.array([np.ones(np.size(n_sp, axis=1), dtype=bool)])

        else:
            # case is experiments are not pooled

            # initialisations
            # y_acc_d, n_expt = data.discrim.dir.y_acc, min([3, len(i_expt)])
            y_acc_d, n_expt, i_expt_lda = data.discrim.dir.y_acc, len(i_expt), i_expt

            # # retrieves the top n_expt experiments based on the base decoding accuracy
            # ii = np.sort(np.argsort(-np.prod(y_acc_d, axis=1))[:n_expt])
            # i_expt, i_cell = i_expt[ii], i_cell[ii]

            # determines the cell count (based on the minimum cell count over all valid experiments)
            n_cell_max = np.max([sum(x) for x in i_cell])
            xi = [x for x in cfcn.n_cell_pool1 if x <= n_cell_max]

        # memory allocation
        n_xi, n_sh, n_cond = len(xi), calc_para['n_shuffle'], len(r_filt['t_type'])
        d_data.y_acc = np.zeros((n_expt, n_cond + 1, n_xi, n_sh))

        # loops through each of the cell counts calculating the partial LDA
        for i_sh in range(n_sh):
            # updates the progressbar
            w_str = 'Pooling LDA Calculations (Shuffle {0} of {1})'.format(i_sh + 1, n_sh)
            w_prog.emit(w_str, 100. * (i_sh / n_sh))

            # # runs the analysis based on the operating system
            # if 'Windows' in platform.platform():
            #     # case is Richard's local computer
            #
            #     # initialisations and memory allocation
            #     p_data = [[] for _ in range(n_xi)]
            #     for i_xi in range(n_xi):
            #         p_data[i_xi].append(data)
            #         p_data[i_xi].append(calc_para)
            #         p_data[i_xi].append(r_filt)
            #         p_data[i_xi].append(i_expt)
            #         p_data[i_xi].append(i_cell)
            #         p_data[i_xi].append(n_trial_max)
            #         p_data[i_xi].append(xi[i_xi])
            #
            #     # runs the pool object to run the partial LDA
            #     p_results = pool.map(cfcn.run_part_lda_pool, p_data)
            #     for i_xi in range(n_xi):
            #         j_xi = xi.index(p_results[i_xi][0])
            #         d_data.y_acc[:, :, j_xi, i_sh] = p_results[i_xi][1]
            # else:

            # case is Subiculum

            # initialisations and memory allocation
            for i_xi in range(n_xi):
                d_data.y_acc[:, :, i_xi, i_sh] = run_pooled_lda_expt(
                    data, calc_para, r_filt, i_expt_lda, dcopy(i_cell), n_trial_max, xi[i_xi], dcopy(n_sp)
                )

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # retrieves the LDA solver parameter fields
        lda_para = calc_para['lda_para']
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

        # sets the solver parameters
        d_data.lda = 1
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max, ignore_list=['n_cell_min'])
        d_data.lda_trial_type = cfcn.get_glob_para('lda_trial_type')

        # sets the phase offset/duration parametrs
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # sets the other parameters/arrays
        d_data.nshuffle = n_sh
        d_data.poolexpt = calc_para['pool_expt']
        d_data.xi = xi

        # returns a true value indicating the calculations were successful
        return True

    def run_wght_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial_max):
        '''

        :param data:
        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial_max:
        :param d_data:
        :param w_prog:
        :return:
        '''

        # initialisations and memory allocation
        d_data, w_prog = data.discrim.wght, self.work_progress
        if d_data.lda is not None:
            # if no change, then exit flagging the calculations are already done
            return True
        else:
            lda_para = calc_para['lda_para']

        #######################################
        ####    LDA WEIGHT CALCULATIONS    ####
        #######################################

        # initialisations
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)
        n_ex, n_tt, n_t, _r_filt = len(i_expt), len(r_filt['t_type']), dcopy(n_trial_max), dcopy(r_filt)
        p_wt, p_wex, xi = 1 / n_tt, 1 / n_ex, np.linspace(0, 1, 101)
        p_w = p_wt * p_wex

        # memory allocation
        A, B, C = np.empty((n_ex, n_tt), dtype=object), np.empty(n_ex, dtype=object), np.empty(n_tt, dtype=object)
        c_ind, c_wght0 = dcopy(A), dcopy(A)
        c_wght, y_top, y_bot = dcopy(C), dcopy(C), dcopy(C)

        # reduces down the data cluster to the valid experiments
        data_tmp = cfcn.reduce_cluster_data(data, i_expt, True)

        # sets the LDA solver type
        lda = cfcn.setup_lda_solver(lda_para)

        # creates a reduce data object and creates the rotation filter object
        for i_tt, tt in enumerate(r_filt['t_type']):
            # retrieves the rotation filter for the current
            _r_filt['t_type'] = [tt]
            r_obj = RotationFilteredData(data_tmp, _r_filt, None, None, True, 'Whole Experiment', False,
                                         t_ofs=t_ofs, t_phase=t_phase)

            # memory allocation
            y_acc_bot, y_acc_top, c_wght_ex = dcopy(B), dcopy(B), dcopy(B)

            # calculates the cell weight scores for each experiment
            for i_ex in range(n_ex):
                # updates the progress bar
                w_str = 'Weighting LDA ({0}, Expt {1}/{2}'.format(tt, i_ex + 1, n_ex)
                p_w0 = p_wt * (i_tt + p_wex * i_ex)

                # retrieves the spike counts for the current experiment
                n_sp, i_grp = cfcn.setup_lda_spike_counts(r_obj, i_cell[i_ex], i_ex, n_t, return_all=False)

                try:
                    # normalises the spike counts and fits the lda model
                    n_sp_norm = cfcn.norm_spike_counts(n_sp, 2 * n_t, lda_para['is_norm'])
                    lda.fit(n_sp_norm, i_grp)
                except:
                    if w_prog is not None:
                        e_str = 'There was an error running the LDA analysis with the current solver parameters. ' \
                                'Either choose a different solver or alter the solver parameters before retrying'
                        w_prog.emit(e_str, 'LDA Analysis Error')
                    return False

                # retrieves the coefficients from the LDA solver
                coef0 = dcopy(lda.coef_)
                coef0 /= np.max(np.abs(coef0))

                # sets the sorting indices and re-orders the weights
                c_ind[i_ex, i_tt] = np.argsort(-np.abs(coef0))[0]
                c_wght0[i_ex, i_tt] = coef0[0, c_ind[i_ex, i_tt]]
                n_sp = n_sp[:, c_ind[i_ex, i_tt]]

                # calculates the top/bottom removed cells lda performance
                y_acc_bot[i_ex] = cfcn.run_reducing_cell_lda(w_prog, lda, lda_para, n_sp, i_grp, p_w0, p_w/2, w_str, True)
                y_acc_top[i_ex] = cfcn.run_reducing_cell_lda(w_prog, lda, lda_para, n_sp, i_grp, p_w0+p_w/2, p_w/2, w_str)

            # calculates the interpolated bottom/top removed values
            c_wght[i_tt] = interp_arr(xi, np.abs(c_wght0[:, i_tt]))
            y_bot[i_tt], y_top[i_tt] = interp_arr(xi, y_acc_bot), interp_arr(xi, y_acc_top)

        #######################################
        ####    HOUSE KEEPING EXERCISES    ####
        #######################################

        # sets the solver parameters
        d_data.lda = 1
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        cfcn.set_lda_para(d_data, lda_para, r_filt, n_trial_max)
        d_data.lda_trial_type = cfcn.get_glob_para('lda_trial_type')

        # sets the phase offset/duration parametrs
        d_data.tofs = t_ofs
        d_data.tphase = t_phase
        d_data.usefull = calc_para['use_full_rot']

        # sets the other parameters
        d_data.xi = xi
        d_data.c_ind = c_ind
        d_data.c_wght = c_wght
        d_data.c_wght0 = c_wght0
        d_data.y_acc_bot = y_bot
        d_data.y_acc_top = y_top

        # return the calculations were a success
        return True

    ##########################################
    ####    KINEMATIC LDA CALCULATIONS    ####
    ##########################################

    def run_speed_lda_accuracy(self, data, calc_para, r_filt, i_expt, i_cell, n_trial, w_prog):
        '''

        :param data:
        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial:
        :param w_prog:
        :return:
        '''

        # initialisations
        d_data = data.discrim.spdacc

        # reduces down the cluster data array
        _data = cfcn.reduce_cluster_data(data, i_expt, True)

        # sets up the kinematic LDA spiking frequency array
        w_prog.emit('Setting Up LDA Spiking Frequencies...', 0.)
        spd_sf, _r_filt = cfcn.setup_kinematic_lda_sf(_data, r_filt, calc_para, i_cell, n_trial, w_prog)

        # case is the normal kinematic LDA
        if not cfcn.run_full_kinematic_lda(_data, dcopy(spd_sf), calc_para, _r_filt, n_trial, w_prog, d_data):
            # if there was an error then exit with a false flag
            return False

        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # sets the lda values
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell

        # returns a true value indicating success
        return True

    def run_kinematic_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial, w_prog):
        '''

        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial:
        :param w_prog:
        :param d_data:
        :return:
        '''

        # initialisations
        d_data = data.discrim.spdc

        # reduces down the cluster data array
        _data = cfcn.reduce_cluster_data(data, i_expt, True)

        # sets up the kinematic LDA spiking frequency array
        w_prog.emit('Setting Up LDA Spiking Frequencies...', 0.)
        spd_sf, _r_filt = cfcn.setup_kinematic_lda_sf(_data, r_filt, calc_para, i_cell, n_trial, w_prog)

        # case is the normal kinematic LDA
        if not cfcn.run_kinematic_lda(_data, spd_sf, calc_para, _r_filt, n_trial, w_prog=w_prog, d_data=d_data):
            # if there was an error then exit with a false flag
            return False

        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # sets the lda values
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell

        # returns a true value indicating success
        return True

    def run_pooled_kinematic_lda(self, data, calc_para, r_filt, i_expt, i_cell, n_trial, w_prog, r_data_type='rotation'):
        '''

        :param data:
        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial:
        :param w_prog:
        :return:
        '''

        # initialisations
        d_data = data.discrim.spdcp
        tt, lda_para, n_shuff = r_filt['t_type'], calc_para['lda_para'], calc_para['n_shuffle']

        ###########################################
        ####    PRE-PROCESSING CALCULATIONS    ####
        ###########################################

        # reduces down the cluster data array
        _data = cfcn.reduce_cluster_data(data, i_expt, True)

        # sets up the kinematic LDA spiking frequency array
        w_prog.emit('Setting Up LDA Spiking Frequencies...', 0.)
        spd_sf, _r_filt = cfcn.setup_kinematic_lda_sf(_data, r_filt, calc_para, i_cell, n_trial,
                                                      w_prog, is_pooled=calc_para['pool_expt'])

        ##############################################
        ####    POOLED NEURON LDA CALCULATIONS    ####
        ##############################################

        # retrieves the rotation data class
        r_data = _data.rotation

        # determines the cell pool groupings
        if calc_para['pool_expt']:
            n_cell, is_keep = cfcn.get_pool_cell_counts(data, lda_para), []
        else:
            n_cell_ex = [sum(x) for x in i_cell]
            n_cell = [x for x in cfcn.n_cell_pool1 if x <= np.max(n_cell_ex)]

        # memory allocation
        n_cell_pool = n_cell[-1]
        n_ex = 1 if calc_para['pool_expt'] else len(i_cell)
        nC, n_tt, n_xi = len(n_cell), len(tt), len(r_data.spd_xi)
        y_acc = [np.nan * np.ones((n_shuff, n_xi, nC, n_ex)) for _ in range(n_tt)]

        #
        for i_c, n_c in enumerate(n_cell):
            n_shuff_nw = n_shuff if (((i_c + 1) < nC) or (not calc_para['pool_expt'])) else 1
            for i_s in range(n_shuff_nw):
                # updates the progressbar
                w_str = 'Speed LDA (G:{0}/{1}, Sh:{2}/{3}'.format(i_c + 1, nC, i_s + 1, n_shuff_nw)
                pw0 = 100. * (i_c + (i_s / n_shuff_nw)) / nC

                while 1:
                    # sets the new shuffled spiking frequency array (over all expt)
                    if calc_para['pool_expt']:
                        # case all cells are pooled over all experiments
                        spd_sf_sh = [set_sf_cell_perm(dcopy(spd_sf), n_cell_pool, n_c)]

                    else:
                        # case all cells
                        is_keep = np.array(n_cell_ex) >= n_c
                        spd_sf_sh = [set_sf_cell_perm(x, n_ex, n_c) for x, n_ex, is_k in
                                     zip(dcopy(spd_sf), n_cell_ex, is_keep) if is_k]

                    # runs the kinematic LDA on the new data
                    n_ex_sh = 1 if calc_para['pool_expt'] else sum(is_keep)
                    results = cfcn.run_kinematic_lda(_data, spd_sf_sh, calc_para, _r_filt, n_trial, w_prog=w_prog,
                                                     w_str0=w_str, pw0=pw0)
                    if not isinstance(results, bool):
                        # if successful, then retrieve the accuracy values
                        for i_tt in range(n_tt):
                            for i_ex in range(n_ex_sh):
                                y_acc[i_tt][i_s, :, i_c, i_ex] = results[0][i_ex, :, i_tt]

                        # exits the loop
                        break

        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # sets a copy of the lda parameters and updates the comparison conditions
        _lda_para = dcopy(lda_para)
        _lda_para['comp_cond'] = r_data.r_obj_kine.rot_filt['t_type']

        # sets the lda values
        d_data.lda = 1
        d_data.y_acc = y_acc
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell
        d_data.n_cell = n_cell
        d_data.exp_name = [os.path.splitext(os.path.basename(x['expFile']))[0] for x in _data.cluster]
        d_data.lda_trial_type = cfcn.get_glob_para('lda_trial_type')

        # sets the rotation values
        d_data.spd_xi = r_data.spd_xi
        d_data.i_bin_spd = r_data.i_bin_spd

        # sets the solver parameters
        cfcn.set_lda_para(d_data, _lda_para, r_filt, n_trial)

        # sets the phase duration/offset parameters
        d_data.spd_xrng = calc_para['spd_x_rng']
        d_data.vel_bin = calc_para['vel_bin']
        d_data.n_sample = calc_para['n_sample']
        d_data.equal_time = calc_para['equal_time']
        d_data.nshuffle = calc_para['n_shuffle']
        d_data.poolexpt = calc_para['pool_expt']

        # returns a true value indicating success
        return True

    def run_speed_dir_lda_accuracy(self, data, calc_para, r_filt, i_expt, i_cell, n_trial, w_prog):
        '''

        :param calc_para:
        :param r_filt:
        :param i_expt:
        :param i_cell:
        :param n_trial_max:
        :param w_prog:
        :return:
        '''

        # initialisations
        d_data = data.discrim.spddir

        # reduces down the cluster data array
        _data = cfcn.reduce_cluster_data(data, i_expt, True)

        # sets up the kinematic LDA spiking frequency array
        w_prog.emit('Setting Up LDA Spiking Frequencies...', 0.)
        vel_sf, _r_filt = cfcn.setup_kinematic_lda_sf(_data, r_filt, calc_para, i_cell, n_trial, w_prog, use_spd=False)

        # case is the normal kinematic LDA
        if not cfcn.run_vel_dir_lda(_data, dcopy(vel_sf), calc_para, _r_filt, n_trial, w_prog, d_data):
            # if there was an error then exit with a false flag
            return False

        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # sets the lda values
        d_data.i_expt = i_expt
        d_data.i_cell = i_cell

        # returns a true value indicating success
        return True

    ######################################
    ####    ROC CURVE CALCULATIONS    ####
    ######################################

    def calc_partial_roc_curves(self, data, calc_para, plot_para, pW, r_data=None):
        '''

        :param data:
        :param calc_para:
        :param plot_para:
        :param pW:
        :return:
        '''

        # initialises the RotationData class object (if not provided)
        if r_data is None:
            r_data = data.rotation

        # memory allocation
        r_data.part_roc, r_data.part_roc_xy, r_data.part_roc_auc = {}, {}, {}

        # initisalises the rotational filter (if not initialised already)
        if plot_para['rot_filt'] is None:
            plot_para['rot_filt'] = cf.init_rotation_filter_data(False)

        # calculates the partial roc curves for each of the trial conditions
        for tt in plot_para['rot_filt']['t_type']:
            # if tt not in r_data.part_roc:
            r_data.part_roc[tt], r_data.part_roc_xy[tt], r_data.part_roc_auc[tt] = \
                                        self.calc_phase_roc_curves(data, calc_para, pW, t_type=tt, r_data=None)

    def calc_phase_roc_curves(self, data, calc_para, pW, t_type=None, r_data=None):
        '''

        :param calc_para:
        :param plot_para:
        :param data:
        :param pool:
        :return:
        '''

        # parameters and initialisations
        phase_str = ['CW/BL', 'CCW/BL', 'CCW/CW']
        if r_data is None:
            r_data = data.rotation

        # if the black phase is calculated already, then exit the function
        if (r_data.phase_roc is not None) and (t_type is None):
            return

        # retrieves the offset parameters
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

        # sets up the black phase data filter and returns the time spikes
        r_filt = cf.init_rotation_filter_data(False)

        if t_type is None:
            r_data.r_obj_black = r_obj = RotationFilteredData(data, r_filt, 0, None, True, 'Whole Experiment', False,
                                                              t_phase=t_phase, t_ofs=t_ofs)
        else:
            r_filt['t_type'] = [t_type]
            r_obj = RotationFilteredData(data, r_filt, 0, None, True, 'Whole Experiment', False,
                                         t_phase=t_phase, t_ofs=t_ofs)

        # retrieves the time spikes and sets the roc class fields for update
        t_spike = r_obj.t_spike[0]

        # memory allocation
        n_cell = np.size(t_spike, axis=0)
        roc = np.empty((n_cell, len(phase_str)), dtype=object)
        roc_xy = np.empty(n_cell, dtype=object)
        roc_auc = np.ones((n_cell, len(phase_str)))

        # calculates the roc curves/integrals for all cells over each phase
        for i_phs, p_str in enumerate(phase_str):
            # updates the progress bar string
            w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
            self.work_progress.emit(w_str, pW * i_phs / len(phase_str))

            # calculates the bootstrapped confidence intervals for each cell
            ind = np.array([1 * (i_phs > 1), 1 + (i_phs > 0)])
            for i_cell in range(n_cell):
                # calculates the roc curve/auc integral
                roc[i_cell, i_phs] = cf.calc_roc_curves(t_spike[i_cell, :, :], ind=ind)
                roc_auc[i_cell, i_phs] = cf.get_roc_auc_value(roc[i_cell, i_phs])

                # if the CW/CCW phase interaction, then set the roc curve x/y coordinates
                if (i_phs + 1) == len(phase_str):
                    roc_xy[i_cell] = cf.get_roc_xy_values(roc[i_cell, i_phs])

        # case is the rotation (black) condition
        if t_type is None:
            r_data.phase_roc, r_data.phase_roc_xy, r_data.phase_roc_auc = roc, roc_xy, roc_auc
        else:
            return roc, roc_xy, roc_auc

    def calc_ud_roc_curves(self, data, r_obj_vis, ind_type, pW, r_data=None):
        '''

        :param data:
        :param r_obj_vis:
        :param calc_para:
        :param pW:
        :return:
        '''

        # initialises the RotationData class object (if not provided)
        if r_data is None:
            r_data = data.rotation

        # parameters and initialisations
        t_spike = r_obj_vis.t_spike
        phase_str, ind = ['CW/BL', 'CCW/BL', 'CCW/CW'], np.array([0, 1])

        # array indexing values
        n_filt = round(r_obj_vis.n_filt / 2)
        n_trial = min([np.shape(x)[1] for x in t_spike])
        n_cell_expt = [x['nC'] for x in np.array(data.cluster)[cf.det_valid_rotation_expt(data, is_ud=True)]]
        n_cell = sum(n_cell_expt)

        # sets up the global index arrays
        i_ofs = np.concatenate(([0], np.cumsum(n_cell_expt[:-1])))
        i_cell_g = [i0 + np.arange(nC) for i0, nC in zip(i_ofs, n_cell_expt) if nC > 0]

        # if the uniformdrifting phase is calculated already, then exit the function
        if r_data.phase_roc_ud is not None:
            return

        # memory allocation
        roc = np.empty((n_cell, len(phase_str)), dtype=object)
        roc_xy = np.empty(n_cell, dtype=object)
        roc_auc = np.ones((n_cell, len(phase_str)))

        for i_filt in range(n_filt):
            # sets the time spike array and global cell indices array
            ind_CC, ind_CCW = ind_type[0][i_filt], ind_type[1][i_filt]
            ig_cell = cf.flat_list([ig[ind] for ig, ind in zip(i_cell_g, r_obj_vis.clust_ind[i_filt])])

            # sets the number of cells to be analysed for the current filter
            n_cell_f = np.shape(t_spike[ind_CC])[0]

            # calculates the roc curves/integrals for all cells over each phase
            for i_phs, p_str in enumerate(phase_str):
                # updates the progress bar string
                w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
                self.work_progress.emit(w_str, 100 * pW * ((i_filt / n_filt) + (i_phs / len(phase_str))))

                # loops through each of the cells calculating the roc curves (and associated values)
                for i_cell in range(n_cell_f):
                    # sets the time spike arrays depending on the phase type
                    if (i_phs + 1) == len(phase_str):
                        t_spike_phs = np.vstack((t_spike[ind_CC][i_cell, :n_trial, 1],
                                                 t_spike[ind_CCW][i_cell, :n_trial, 1])).T
                    else:
                        t_spike_phs = t_spike[ind_type[i_phs][i_filt]][i_cell, :, :]

                    # calculates the roc curve/auc integral
                    ig_nw = int(ig_cell[i_cell])
                    roc[ig_nw, i_phs] = cf.calc_roc_curves(t_spike_phs, ind=np.array([0, 1]))
                    roc_auc[ig_nw, i_phs] = cf.get_roc_auc_value(roc[ig_nw, i_phs])

                    # if the CW/CCW phase interaction, then set the roc curve x/y coordinates
                    if (i_phs + 1) == len(phase_str):
                        roc_xy[ig_nw] = cf.get_roc_xy_values(roc[ig_nw, i_phs])

        # sets the final
        r_data.phase_roc_ud, r_data.phase_roc_xy_ud, r_data.phase_roc_auc_ud = roc, roc_xy, roc_auc

    def calc_cond_roc_curves(self, data, pool, calc_para, plot_para, g_para, calc_cell_grp, pW,
                             force_black_calc=False, r_data=None):
        '''

        :param calc_para:
        :param plot_para:
        :param data:
        :param pool:
        :return:
        '''

        # initialises the RotationData class object (if not provided)
        if r_data is None:
            r_data = data.rotation

        # parameters and initialisations
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)
        r_obj_sig, plot_scope, c_lvl = None, 'Whole Experiment', float(g_para['roc_clvl'])
        phase_str = ['CW/BL', 'CCW/BL', 'CCW/CW']

        # initisalises the rotational filter (if not initialised already)
        if plot_para['rot_filt'] is None:
            plot_para['rot_filt'] = cf.init_rotation_filter_data(False)

        # sets the condition types (ensures that the black phase is always included)
        t_type = dcopy(plot_para['rot_filt']['t_type'])
        if 'Black' not in t_type:
            t_type = ['Black'] + t_type

        if 'vis_expt_type' in calc_para:
            if calc_para['vis_expt_type'] == 'MotorDrifting':
                t_type += ['MotorDrifting']

        # retrieves the rotation phase offset time/duration
        if t_ofs is not None:
            # if the values are not none, and do not match previous values, then reset the stored roc array
            if (r_data.t_ofs_rot != t_ofs) or (r_data.t_phase_rot != t_phase):
                r_data.t_ofs_rot, r_data.t_phase_rot, r_data.cond_roc = t_ofs, t_phase, None
        elif 'use_full_rot' in calc_para:
            # if using the full rotation, and the previous calculations were made using non-full rotation phases,
            # the reset the stored roc array
            if (r_data.t_ofs_rot > 0):
                r_data.t_ofs_rot, r_data.t_phase_rot, r_data.cond_roc = -1, -1, None

        # sets up a base filter with only the
        r_filt_base = cf.init_rotation_filter_data(False)
        r_filt_base['t_type'] = [x for x in t_type if x != 'UniformDrifting']

        # sets up the black phase data filter and returns the time spikes
        r_obj = RotationFilteredData(data, r_filt_base, None, plot_para['plot_exp_name'], True, plot_scope, False,
                                     t_ofs=t_ofs, t_phase=t_phase)
        if not r_obj.is_ok:
            # if there was an error, then output an error to screen
            self.work_error.emit(r_obj.e_str, 'Incorrect Analysis Function Parameters')
            return False

        # memory allocation (if the conditions have not been set)
        if r_data.cond_roc is None:
            r_data.cond_roc, r_data.cond_roc_xy, r_data.cond_roc_auc = {}, {}, {}
            r_data.cond_gtype, r_data.cond_auc_sig, r_data.cond_i_expt, r_data.cond_cl_id = {}, {}, {}, {}
            r_data.cond_ci_lo, r_data.cond_ci_hi, r_data.r_obj_cond = {}, {}, {}
            r_data.phase_gtype, r_data.phase_auc_sig, r_data.phase_roc = None, None, None

        for i_rr, rr in enumerate(r_obj.rot_filt_tot):
            # sets the trial type
            tt = rr['t_type'][0]

            # updates the progress bar string
            w_str = 'ROC Curve Calculations ({0})...'.format(tt)
            self.work_progress.emit(w_str, pW * (i_rr / r_obj.n_filt))

            if tt not in r_data.cond_roc:
                # array dimensions
                t_spike = r_obj.t_spike[i_rr]
                n_cell = np.size(t_spike, axis=0)

                # memory allocation and initialisations
                r_data.cond_roc[tt] = np.empty((n_cell, 3), dtype=object)
                r_data.cond_roc_xy[tt] = np.empty(n_cell, dtype=object)
                r_data.cond_roc_auc[tt] = np.zeros((n_cell, 3))
                r_data.cond_gtype[tt] = -np.ones((n_cell, 3))
                r_data.cond_auc_sig[tt] = np.zeros((n_cell, 3), dtype=bool)
                r_data.cond_i_expt[tt] = r_obj.i_expt[i_rr]
                r_data.cond_cl_id[tt] = r_obj.cl_id[i_rr]
                r_data.cond_ci_lo[tt] = -np.ones((n_cell, 2))
                r_data.cond_ci_hi[tt] = -np.ones((n_cell, 2))
                r_data.r_obj_cond[tt] = dcopy(r_obj)

                # calculates the roc curves/integrals for all cells over each phase
                for i_phs, p_str in enumerate(phase_str):
                    # updates the progress bar string
                    self.work_progress.emit(w_str, pW * ((i_rr / r_obj.n_filt) + (i_phs / len(phase_str))))

                    # calculates the roc curve values for each phase
                    ind = np.array([1 * (i_phs > 1), 1 + (i_phs > 0)])
                    for ic in range(n_cell):
                        r_data.cond_roc[tt][ic, i_phs] = cf.calc_roc_curves(t_spike[ic, :, :], ind=ind)
                        r_data.cond_roc_auc[tt][ic, i_phs] = cf.get_roc_auc_value(r_data.cond_roc[tt][ic, i_phs])

                        if (i_phs + 1) == len(phase_str):
                            r_data.cond_roc_xy[tt][ic] = cf.get_roc_xy_values(r_data.cond_roc[tt][ic, i_phs])

            # calculates the confidence intervals for the current (only if bootstrapping count has changed or the
            # confidence intervals has not already been calculated)
            if 'auc_stype' in calc_para:
                # updates the auc statistics calculation type
                r_data.cond_auc_stats_type = calc_para['auc_stype']

                # determine if the auc confidence intervals need calculation
                is_boot = int(calc_para['auc_stype'] == 'Bootstrapping')
                if is_boot:
                    # if bootstrapping, then determine if the
                    if r_data.n_boot_cond_ci != calc_para['n_boot']:
                        # if the bootstrapping count has changed, flag that the confidence intervals needs updating
                        r_data.n_boot_cond_ci, calc_ci = calc_para['n_boot'], True
                    else:
                        # otherwise, recalculate the confidence intervals if they have not been set
                        calc_ci = np.any(r_data.cond_ci_lo[tt][:, 1] < 0)
                else:
                    # otherwise, recalculate the confidence intervals if they have not been set
                    calc_ci = np.any(r_data.cond_ci_lo[tt][:, 0] < 0)

                # calculates the confidence intervals (if required)
                if calc_ci:
                    conf_int = self.calc_roc_conf_intervals(pool, r_data.cond_roc[tt][:, 2],
                                                            calc_para['auc_stype'], calc_para['n_boot'], c_lvl)
                    r_data.cond_ci_lo[tt][:, is_boot] = conf_int[:, 0]
                    r_data.cond_ci_hi[tt][:, is_boot] = conf_int[:, 1]

            # if not calculating the cell group indices, or the condition type is Black (the phase statistics for
            # this condition are already calculated in "calc_phase_roc_significance"), then continue
            if (not calc_cell_grp) or ((tt == 'Black') and (not force_black_calc)):
                continue

            # sets the rotation object filter (if using wilcoxon paired test for the cell group stats type)
            if calc_para['grp_stype'] == 'Wilcoxon Paired Test':
                if np.all(r_data.cond_gtype[tt][:, 0] >= 0):
                    # if all the values have been calculated, then exit the function
                    continue

                # sets the rotation object for the current condition
                r_obj_sig = RotationFilteredData(data, r_obj.rot_filt_tot[i_rr], None, plot_para['plot_exp_name'],
                                                 True, plot_scope, False, t_ofs=t_ofs, t_phase=t_phase)
                if not r_obj_sig.is_ok:
                    # if there was an error, then output an error to screen
                    self.work_error.emit(r_obj_sig.e_str, 'Incorrect Analysis Function Parameters')
                    return False

            # calculates the condition cell group types
            self.calc_phase_roc_significance(calc_para, g_para, data, pool, None, c_type='cond',
                                             roc=r_data.cond_roc[tt], auc=r_data.cond_roc_auc[tt],
                                             g_type=r_data.cond_gtype[tt], auc_sig=r_data.cond_auc_sig[tt],
                                             r_obj=r_obj_sig)

        # returns a true value
        return True

    def calc_phase_roc_significance(self, calc_para, g_para, data, pool, pW, c_type='phase',
                                    roc=None, auc=None, g_type=None, auc_sig=None, r_obj=None, r_data=None):
        '''

        :param calc_data:
        :param data:
        :param pool:
        :return:
        '''

        # initialises the RotationData class object (if not provided)
        if r_data is None:
            r_data = data.rotation

        # sets the roc objects/integrals (if not provided)
        c_lvl = float(g_para['roc_clvl'])
        if c_type == 'phase':
            # case is the significance tests are being calculated for the phase
            r_data.phase_grp_stats_type = calc_para['grp_stype']
            roc, auc, r_obj = r_data.phase_roc, r_data.phase_roc_auc, r_data.r_obj_black
        else:
            # case is the significance tests are being calculated for the conditions
            r_data.cond_grp_stats_type = calc_para['grp_stype']

        # parameters and initialisations
        phase_str, i_col = ['CW/BL', 'CCW/BL', 'CCW/CW'], 0
        p_value, n_cell = 0.05, np.size(roc, axis=0)

        # allocates memory for the group-types (if not already calculated)
        if c_type == 'phase':
            # case is for the phase type
            n_boot = r_data.n_boot_phase_grp
            if r_data.phase_gtype is None:
                # group type has not been set, so initialise the array
                r_data.phase_gtype = g_type = -np.ones((n_cell, 3))
                r_data.phase_auc_sig = auc_sig = np.zeros((n_cell, 3), dtype=bool)
            else:
                # otherwise, retrieve the currently stored array
                g_type, auc_sig = r_data.phase_gtype, r_data.phase_auc_sig
        else:
            # case is for the condition type
            n_boot = r_data.n_boot_cond_grp

        #########################################
        ####    WILCOXON STATISTICAL TEST    ####
        #########################################

        if calc_para['grp_stype'] == 'Wilcoxon Paired Test':
            # if the statistics have already been calculated, then exit the function
            if np.all(g_type[:, 0] >= 0):
                return

            # updates the progress bar string
            if pW is not None:
                self.work_progress.emit('Calculating Wilcoxon Stats...', pW + 25.)

            # calculates the statistical significance between the phases
            sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj)
            _, _, sf_stats, _ = cf.setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, None)

            # determines which cells are motion/direction sensitive
            for i_phs in range(len(sf_stats)):
                auc_sig[:, i_phs] = sf_stats[i_phs] < p_value

        ##########################################
        ####    ROC-BASED STATISTICAL TEST    ####
        ##########################################

        else:
            # determines what kind of statistics are to be calculated
            is_boot = calc_para['grp_stype'] == 'Bootstrapping'
            i_col, phase_stype = 1 + is_boot, calc_para['grp_stype']

            # if the statistics have been calculated for the selected type, then exit the function
            if is_boot:
                if np.all(g_type[:, 2] >= 0) and (calc_para['n_boot'] == n_boot):
                    # if bootstrapping is selected, but all values have been calculated and the bootstrapping values
                    # has not changed, then exit the function
                    return
                else:
                    # otherwise, update the bootstrapping count
                    if c_type == 'phase':
                        r_data.n_boot_phase_grp = dcopy(calc_para['n_boot'])
                    else:
                        r_data.n_boot_cond_grp = dcopy(calc_para['n_boot'])

            elif np.all(g_type[:, 1] >= 0):
                # if delong significance is selected, and all values have been calculated, then exit the function
                return

            # calculates the significance for each phase
            for i_phs, p_str in enumerate(phase_str):
                # updates the progress bar string
                if pW is not None:
                    w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
                    self.work_progress.emit(w_str, pW * (1. + i_phs / len(phase_str)))

                # calculates the confidence intervals for the current
                conf_int = self.calc_roc_conf_intervals(pool, roc[:, i_phs], phase_stype, n_boot, c_lvl)

                # determines the significance for each cell in the phase
                auc_ci_lo = (auc[:, i_phs] + conf_int[:, 1]) < 0.5
                auc_ci_hi = (auc[:, i_phs] - conf_int[:, 0]) > 0.5
                auc_sig[:, i_phs] = np.logical_or(auc_ci_lo, auc_ci_hi)

        # calculates the cell group types
        g_type[:, i_col] = cf.calc_cell_group_types(auc_sig, calc_para['grp_stype'])

    def calc_dirsel_group_types(self, data, pool, calc_para, plot_para, g_para, r_data=None):
        '''

        :param data:
        :param plot_para:
        :return:
        '''

        def calc_combined_spiking_stats(r_data, r_obj, pool, calc_para, g_para, p_value, ind_type=None,
                                        t_type='Black'):
            '''

            :param r_obj:
            :param ind_type:
            :return:
            '''

            # calculates the individual trial/mean spiking rates and sets up the plot/stats arrays
            sp_f0, sp_f = cf.calc_phase_spike_freq(r_obj)
            s_plt, _, sf_stats, i_grp = cf.setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, ind_type)

            # calculates the CW/CCW spiking frequency ratio
            r_CCW_CW = np.array(s_plt[2][1]) / np.array(s_plt[2][0])

            #########################################
            ####    WILCOXON STATISTICAL TEST    ####
            #########################################

            if calc_para['grp_stype'] == 'Wilcoxon Paired Test':
                # case is the wilcoxon paired test
                sf_scores = cf.calc_ms_scores(s_plt, sf_stats, p_value)

            ##########################################
            ####    ROC-BASED STATISTICAL TEST    ####
            ##########################################

            else:
                # determines what kind of statistics are to be calculated
                phase_stype = calc_para['grp_stype']
                is_boot, n_boot = calc_para['grp_stype'] == 'Bootstrapping', calc_para['n_boot']
                phase_str, c_lvl, pW = ['CW/BL', 'CCW/BL', 'CCW/CW'], float(g_para['roc_clvl']), 100.

                # retrieves the roc/auc fields (depending on the type)
                if t_type == 'Black':
                    # case is the black (rotation) condition
                    roc, auc = r_data.phase_roc, r_data.phase_roc_auc
                elif t_type == 'UniformDrifting':
                    # case is the uniformdrifting (visual) condition
                    roc, auc = r_data.phase_roc_ud, r_data.phase_roc_auc_ud
                else:
                    # case is the motordrifting (visual) condition
                    roc, auc = r_data.cond_roc['MotorDrifting'], r_data.cond_roc_auc['MotorDrifting']

                # REMOVE ME LATER?
                c_lvl = 0.95

                # if the statistics have been calculated for the selected type, then exit the function
                if is_boot:
                    # otherwise, update the bootstrapping count
                    r_data.n_boot_comb_grp = dcopy(calc_para['n_boot'])

                # calculates the significance for each phase
                auc_sig = np.zeros((np.size(roc, axis=0), 3), dtype=bool)
                for i_phs, p_str in enumerate(phase_str):
                    # updates the progress bar string
                    if pW is not None:
                        w_str = 'ROC Curve Calculations ({0})...'.format(p_str)
                        self.work_progress.emit(w_str, pW * (i_phs / len(phase_str)))

                    # calculates the confidence intervals for the current
                    conf_int = self.calc_roc_conf_intervals(pool, roc[:, i_phs], phase_stype, n_boot, c_lvl)

                    # determines the significance for each cell in the phase
                    auc_ci_lo = (auc[:, i_phs] + conf_int[:, 1]) < 0.5
                    auc_ci_hi = (auc[:, i_phs] - conf_int[:, 0]) > 0.5
                    auc_sig[:, i_phs] = np.logical_or(auc_ci_lo, auc_ci_hi)

                # case is the wilcoxon paired test
                sf_scores = np.zeros((np.size(roc, axis=0), 3), dtype=int)
                for ig in i_grp:
                    sf_scores[ig, :] = cf.calc_ms_scores(auc[ig, :], auc_sig[ig, :], None)

            # returns the direction selectivity scores
            return sf_scores, i_grp, r_CCW_CW

        def det_dirsel_cells(sf_score, grp_stype):
            '''

            :param sf_score:
            :return:
            '''

            # calculates the minimum/sum scores
            if grp_stype == 'Wilcoxon Paired Test':
                score_min, score_sum = np.min(sf_score[:, :2], axis=1), np.sum(sf_score[:, :2], axis=1)

                # determines the direction selective cells, which must meet the following conditions:
                #  1) one direction only produces a significant result, OR
                #  2) both directions are significant AND the CW/CCW comparison is significant
                one_dir_sig = np.logical_and(score_min == 0, score_sum > 0)     # cells where one direction is significant
                both_dir_sig = np.min(sf_score[:, :2], axis=1) > 0              # cells where both CW/CCW is significant
                comb_dir_sig = sf_score[:, -1] > 0                              # cells where CW/CCW difference is significant

                # determines which cells are direction selective (removes non-motion sensitive cells)
                return np.logical_or(one_dir_sig, np.logical_and(both_dir_sig, comb_dir_sig)).astype(int)
            else:
                # case is the roc analysis statistics (only consider the CW/CCW comparison for ds)
                return sf_score[:, 2] > 0

        # initialises the RotationData class object (if not provided)
        if r_data is None:
            r_data = data.rotation

        # initialises the rotation filter (if not set)
        rot_filt = plot_para['rot_filt']
        if rot_filt is None:
            rot_filt = cf.init_rotation_filter_data(False)

        # sets the p-value
        if 'p_value' in calc_para:
            p_val = calc_para['p_value']
        else:
            p_val = 0.05

        # initialisations and memory allocation
        p_scope, n_grp, r_data, grp_stype = 'Whole Experiment', 4, r_data, calc_para['grp_stype']
        # r_filt_rot, r_filt_vis = dcopy(rot_filt), dcopy(rot_filt)
        plot_exp_name, plot_all_expt = plot_para['plot_exp_name'], plot_para['plot_all_expt']
        r_data.ds_p_value = dcopy(p_val)

        t_ofs_rot, t_phase_rot = cfcn.get_rot_phase_offsets(calc_para)
        t_ofs_vis, t_phase_vis = cfcn.get_rot_phase_offsets(calc_para, True)

        # determines what type of visual experiment is being used for comparison (if provided)
        if 'vis_expt_type' in calc_para:
            # case is a calculation parameter is set
            ud_rot_expt = calc_para['vis_expt_type'] == 'UniformDrifting'
        else:
            # case is no calculation parameter is set, so use uniform drifting
            ud_rot_expt = True

        # sets up the black-only rotation filter object
        r_filt_black = cf.init_rotation_filter_data(False)
        r_obj_black = RotationFilteredData(data, r_filt_black, None, plot_exp_name, plot_all_expt, p_scope, False,
                                           t_ofs=t_ofs_rot, t_phase=t_phase_rot)

        # retrieves the rotational filtered data (black conditions only)
        r_filt_rot = cf.init_rotation_filter_data(False)
        r_data.r_obj_rot_ds = RotationFilteredData(data, r_filt_rot, None, plot_exp_name, plot_all_expt,
                                                   p_scope, False)

        # retrieves the visual filtered data
        r_filt_vis = cf.init_rotation_filter_data(True)
        if ud_rot_expt:
            # sets the visual phase/offset
            if t_phase_vis is None:
                # if the phase duration is not set
                t_phase_vis, t_ofs_vis = 2., 0.
            elif (t_phase_vis + t_ofs_vis) > 2:
                # output an error to screen
                e_str = 'The entered analysis duration and offset is greater than the experimental phase duration:\n\n' \
                        '  * Analysis Duration + Offset = {0}\n s. * Experiment Phase Duration = {1} s.\n\n' \
                        'Enter a correct analysis duration/offset combination before re-running ' \
                        'the function.'.format(t_phase_vis + t_ofs_vis, 2.0)
                self.work_error.emit(e_str, 'Incorrect Analysis Function Parameters')

                # return a false value indicating the calculation is invalid
                return False

            # case is uniform-drifting experiments (split into CW/CCW phases)
            r_filt_vis['t_type'], r_filt_vis['is_ud'], r_filt_vis['t_cycle'] = ['UniformDrifting'], [True], ['15']
            r_data.r_obj_vis, ind_type = cf.split_unidrift_phases(data, r_filt_vis, None, plot_exp_name, plot_all_expt,
                                                           p_scope, t_phase_vis, t_ofs_vis)

            if (r_data.phase_roc_ud is None) and ('Wilcoxon' not in calc_para['grp_stype']):
                self.calc_ud_roc_curves(data, r_data.r_obj_vis, ind_type, 66.)

        else:
            # case is motor-drifting experiments

            # retrieves the filtered data from the loaded datasets
            r_filt_vis['t_type'], r_filt_vis['is_ud'], ind_type = ['MotorDrifting'], [False], None
            t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para, is_vis=True)

            # runs the rotation filter
            r_data.r_obj_vis = RotationFilteredData(data, r_filt_vis, None, plot_exp_name, plot_all_expt,
                                                    p_scope, False, t_ofs=t_ofs, t_phase=t_phase)
            if not r_data.r_obj_vis.is_ok:
                # if there was an error, then output an error to screen
                self.work_error.emit(r_data.r_obj_vis.e_str, 'Incorrect Analysis Function Parameters')
                return False

        # calculate the visual/rotation stats scores
        sf_score_rot, i_grp_rot, r_CCW_CW_rot = calc_combined_spiking_stats(r_data, r_data.r_obj_rot_ds, pool,
                                                                            calc_para, g_para, p_val)

        sf_score_vis, i_grp_vis, r_CCW_CW_vis = calc_combined_spiking_stats(r_data, r_data.r_obj_vis, pool,
                                                                            calc_para, g_para, p_val, ind_type,
                                                                            r_filt_vis['t_type'][0])

        # memory allocation
        ds_type_tmp, ms_type_tmp, pd_type_tmp = [], [], []
        r_data.ms_gtype_N, r_data.ds_gtype_N, r_data.pd_type_N = [], [], []
        A = np.empty(len(i_grp_rot), dtype=object)
        r_data.ds_gtype_ex, r_data.ms_gtype_ex, r_data.pd_type_ex = dcopy(A), dcopy(A), dcopy(A)
        r_data.ds_gtype_comb, r_data.ms_gtype_comb = dcopy(A), dcopy(A)

        # reduces the arrays to the matching cells
        for i in range(len(i_grp_rot)):
            if len(i_grp_rot[i]):
                # retrieves the matching rotation/visual indices
                ind_rot, ind_vis = cf.det_cell_match_indices(r_data.r_obj_rot_ds, i, r_data.r_obj_vis)

                # determines the motion sensitivity from the score phase types (append proportion/N-value arrays)
                #   0 = None
                #   1 = Rotation Only
                #   2 = Visual Only
                #   3 = Both
                _sf_score_rot = sf_score_rot[i_grp_rot[i][ind_rot]][:, :-1]
                _sf_score_vis = sf_score_vis[i_grp_vis[i][ind_vis]][:, :-1]
                ms_gtype_comb = (np.sum(_sf_score_rot, axis=1) > 0) + 2 * (np.sum(_sf_score_vis, axis=1) > 0)
                ms_type_tmp.append(cf.calc_rel_prop(ms_gtype_comb, 4))
                r_data.ms_gtype_N.append(len(ind_rot))

                # determines the direction selectivity type from the score phase types (append proportion/N-value arrays)
                #   0 = None
                #   1 = Rotation Only
                #   2 = Visual Only
                #   3 = Both
                is_ds_rot = det_dirsel_cells(sf_score_rot[i_grp_rot[i][ind_rot]], calc_para['grp_stype'])
                is_ds_vis = det_dirsel_cells(sf_score_vis[i_grp_vis[i][ind_vis]], calc_para['grp_stype'])
                ds_gtype_comb = is_ds_rot.astype(int) + 2 * is_ds_vis.astype(int)
                ds_type_tmp.append(cf.calc_rel_prop(ds_gtype_comb, 4))
                r_data.ds_gtype_N.append(len(ind_rot))

                # determines which cells have significance for both rotation/visual stimuli. from this determine the
                # preferred direction from the CW vs CCW spiking rates
                is_both_ds = ds_gtype_comb == 3
                r_CCW_CW_comb = np.vstack((r_CCW_CW_rot[i_grp_rot[i][ind_rot]][is_both_ds],
                                           r_CCW_CW_vis[i_grp_vis[i][ind_vis]][is_both_ds])).T

                # determines the preferred direction type (for clusters which have BOTH rotation and visual significance)
                #   0 = Incongruent (preferred direction is the same)
                #   1 = Congruent (preferred direction is different)
                pd_type = np.zeros(sum(is_both_ds), dtype=int)
                pd_type[np.sum(r_CCW_CW_comb > 1, axis=1) == 1] = 1

                # calculates the preferred direction type count/proportions
                r_data.pd_type_N.append(cf.calc_rel_count(pd_type, 2))
                pd_type_tmp.append(cf.calc_rel_prop(pd_type, 2))

                # sets the indices of the temporary group type into the total array
                ind_bl, ind_bl_rot = cf.det_cell_match_indices(r_obj_black, [0, i], r_data.r_obj_rot_ds)
                ind_comb = ind_bl[np.searchsorted(ind_bl_rot, ind_rot)]

                # sets the indices for each experiment
                i_expt0 = r_data.r_obj_vis.i_expt[i][ind_vis]
                i_expt, i_expt_cong = grp_expt_indices(i_expt0), grp_expt_indices(i_expt0[is_both_ds])

                # sets the final motion sensitivity, direction selectivity and congruency values
                r_data.ms_gtype_ex[i] = np.vstack([cf.calc_rel_prop(ms_gtype_comb[x], 4) for x in i_expt])
                r_data.ds_gtype_ex[i] = np.vstack([cf.calc_rel_prop(ds_gtype_comb[x], 4) for x in i_expt])

                if len(i_expt_cong):
                    r_data.pd_type_ex[i] = np.vstack([cf.calc_rel_prop(pd_type[x], 2) for x in i_expt_cong])
                else:
                    r_data.pd_type_ex[i] = np.nan * np.ones((1, 2))

                # sets the direction selective/motion sensitivity types for current experiment
                r_data.ds_gtype_comb[i] = [ds_gtype_comb[i_ex] for i_ex in i_expt]
                r_data.ms_gtype_comb[i] = [ms_gtype_comb[i_ex] for i_ex in i_expt]

            else:
                # appends the counts to the motion sensitive/direction selectivity arrays
                r_data.ms_gtype_N.append(0)
                r_data.ds_gtype_N.append(0)

                # appends NaN arrays to the temporary arrays
                ms_type_tmp.append(np.array([np.nan] * 4))
                ds_type_tmp.append(np.array([np.nan] * 4))
                pd_type_tmp.append(np.array([np.nan] * 2))

        # combines the relative proportion lists into a single array ()
        r_data.ms_gtype_pr = np.vstack(ms_type_tmp).T
        r_data.ds_gtype_pr = np.vstack(ds_type_tmp).T
        r_data.pd_type_pr = np.vstack(pd_type_tmp).T

        # return a true flag to indicate the analysis was valid
        return True

    def calc_kinematic_roc_curves(self, data, pool, calc_para, g_para, pW0, r_data=None):
        '''

        :param calc_para:
        :return:
        '''

        def resample_spike_freq(pool, sf, c_lvl, n_rs=100):
            '''

            :param data:
            :param r_data:
            :param rr:
            :param ind:
            :param n_rs:
            :return:
            '''

            # array dimensioning and other initialisations
            n_trial = len(sf)
            pz = norm.ppf(1 - (1 - c_lvl) / 2)
            n_trial_h = int(np.floor(n_trial / 2))

            # if the spiking frequency values are all identical, then return the fixed values
            if cfcn.arr_range(sf) == 0.:
                return sf[0] * np.ones(n_trial_h), sf[0] * np.ones(n_trial_h), 0.5, np.zeros(2)

            # initialisations and memory allocation
            p_data = [[] for _ in range(n_rs)]

            # returns the shuffled spike frequency arrays
            for i_rs in range(n_rs):
                ind0 = np.random.permutation(n_trial)
                p_data[i_rs].append(np.sort(sf[ind0[:n_trial_h]]))
                p_data[i_rs].append(np.sort(sf[ind0[n_trial_h:(2 * n_trial_h)]]))

            # calculates the roc curves and the x/y coordinates
            _roc = pool.map(cfcn.calc_roc_curves_pool, p_data)
            _roc_xy = cfcn.calc_avg_roc_curve([cf.get_roc_xy_values(x) for x in _roc])

            # calculate the roc auc values (ensures that they are > 0.5)
            _roc_auc = [cf.get_roc_auc_value(x) for x in _roc]
            _roc_auc = [(1. - x) if x < 0.5 else x for x in _roc_auc]

            # calculates the roc auc mean/confidence interval
            roc_auc_mn = np.mean(_roc_auc)
            roc_auc_ci = pz * np.ones(2) * (np.std(_roc_auc) / (n_rs ** 0.5))

            # returns the arrays and auc mean/confidence intervals
            return _roc_xy[:, 0], _roc_xy[:, 1], roc_auc_mn, roc_auc_ci

        # initialises the RotationData class object (if not provided)
        if r_data is None:
            r_data = data.rotation

        # initialisations
        is_boot = int(calc_para['auc_stype'] == 'Bootstrapping')
        pW1, c_lvl = 100 - pW0, float(g_para['roc_clvl'])

        # memory allocation (if the conditions have not been set)
        if r_data.vel_roc is None:
            r_data.vel_roc, r_data.vel_roc_xy, r_data.vel_roc_auc = {}, {}, {}
            r_data.spd_roc, r_data.spd_roc_xy, r_data.spd_roc_auc = {}, {}, {}
            r_data.vel_ci_lo, r_data.vel_ci_hi, r_data.spd_ci_lo, r_data.spd_ci_hi = {}, {}, {}, {}
            r_data.vel_roc_sig, r_data.spd_roc_sig = None, None

        for i_rr, rr in enumerate(r_data.r_obj_kine.rot_filt_tot):
            tt, _pW1 = rr['t_type'][0], pW1 * (i_rr / r_data.r_obj_kine.n_filt)
            init_data = tt not in r_data.vel_roc

            # array dimensions
            calc_ci = None
            if r_data.is_equal_time:
                vel_sf = dcopy(r_data.vel_sf_rs[tt])
                if not r_data.pn_comp:
                    spd_sf = dcopy(r_data.spd_sf_rs[tt])
            else:
                vel_sf = dcopy(r_data.vel_sf[tt])
                if not r_data.pn_comp:
                    spd_sf = dcopy(r_data.spd_sf[tt])

            # array indexing
            n_trial, n_bin_vel, n_cell = np.shape(vel_sf)
            if r_data.pn_comp:
                n_bin_vel = int(n_bin_vel / 2)

            if init_data:
                # velocity roc memory allocation and initialisations
                r_data.vel_roc[tt] = np.empty((n_cell, n_bin_vel), dtype=object)
                r_data.vel_roc_xy[tt] = np.empty((n_cell, n_bin_vel), dtype=object)
                r_data.vel_roc_auc[tt] = np.zeros((n_cell, n_bin_vel))
                r_data.vel_ci_lo[tt] = -np.ones((n_cell, n_bin_vel, 2))
                r_data.vel_ci_hi[tt] = -np.ones((n_cell, n_bin_vel, 2))

                # speed roc memory allocation and initialisations (non pos/neg comparison only
                if not r_data.pn_comp:
                    n_bin_spd = np.size(spd_sf, axis=1)
                    r_data.spd_roc[tt] = np.empty((n_cell, n_bin_spd), dtype=object)
                    r_data.spd_roc_xy[tt] = np.empty((n_cell, n_bin_spd), dtype=object)
                    r_data.spd_roc_auc[tt] = np.zeros((n_cell, n_bin_spd))
                    r_data.spd_ci_lo[tt] = -np.ones((n_cell, n_bin_spd, 2))
                    r_data.spd_ci_hi[tt] = -np.ones((n_cell, n_bin_spd, 2))

            # calculates the roc curves/integrals for all cells over each phase
            w_str0 = 'ROC Calculations ({0} - '.format(tt)
            for ic in range(n_cell):
                # updates the progress bar string
                w_str = '{0}{1}/{2})'.format(w_str0, ic+1, n_cell)
                self.work_progress.emit(w_str, pW0 + _pW1 + (pW1 / r_data.r_obj_kine.n_filt) * ( + (ic/ n_cell)))

                if init_data:
                    # memory allocations
                    vel_auc_ci, ii_v = [], ~np.isnan(vel_sf[:, 0, ic])

                    # calculates the velocity roc curves values for each velocity bin
                    for i_bin in range(n_bin_vel):
                        if r_data.pn_comp:
                            is_resampled = False
                            vel_sf_x = vel_sf[ii_v, n_bin_vel + i_bin, ic]
                            vel_sf_y = vel_sf[ii_v, n_bin_vel - (i_bin + 1), ic]
                        else:
                            # case is single bin comparison
                            if (i_bin == r_data.i_bin_vel[0]) or (i_bin == r_data.i_bin_vel[1]):
                                is_resampled = True
                                vel_sf_x, vel_sf_y, vel_auc_roc, _auc_ci = \
                                                    resample_spike_freq(pool, vel_sf[ii_v, i_bin, ic], c_lvl)
                                vel_auc_ci.append(_auc_ci)
                            else:
                                is_resampled = False
                                vel_sf_x = vel_sf[ii_v, i_bin, ic]
                                if r_data.vel_xi[i_bin, 0] < 0:
                                    vel_sf_y = vel_sf[ii_v, r_data.i_bin_vel[0], ic]
                                else:
                                    vel_sf_y = vel_sf[ii_v, r_data.i_bin_vel[1], ic]

                        # calculates the roc curves/coordinates from the spiking frequencies
                        r_data.vel_roc[tt][ic, i_bin] = cf.calc_roc_curves(None, None,
                                                                           x_grp=vel_sf_x, y_grp=vel_sf_y)
                        r_data.vel_roc_xy[tt][ic, i_bin] = cf.get_roc_xy_values(r_data.vel_roc[tt][ic, i_bin])

                        # sets the roc auc values
                        if is_resampled:
                            # case is the resampled frequencies
                            r_data.vel_roc_auc[tt][ic, i_bin] = vel_auc_roc
                        else:
                            # other cases
                            r_data.vel_roc_auc[tt][ic, i_bin] = cf.get_roc_auc_value(r_data.vel_roc[tt][ic, i_bin])

                    # calculates the speed roc curves values for each speed bin
                    if not r_data.pn_comp:
                        ii_s = ~np.isnan(spd_sf[:, 0, ic])
                        for i_bin in range(n_bin_spd):
                            calc_roc = True
                            if i_bin == r_data.i_bin_spd:
                                # spd_sf_x, spd_sf_y = resample_spike_freq(data, r_data, rr, [i_rr, i_bin, ic])
                                is_resampled = True
                                spd_sf_x, spd_sf_y, spd_auc_roc, spd_auc_ci = \
                                                resample_spike_freq(pool, spd_sf[ii_s, i_bin, ic], c_lvl)
                            else:
                                is_resampled = False
                                spd_sf_x, spd_sf_y = spd_sf[ii_s, r_data.i_bin_spd, ic], spd_sf[ii_s, i_bin, ic]

                            # calculates the roc curves/coordinates from the spiking frequencies
                            r_data.spd_roc[tt][ic, i_bin] = cf.calc_roc_curves(None, None, x_grp=spd_sf_x, y_grp=spd_sf_y)
                            r_data.spd_roc_xy[tt][ic, i_bin] = cf.get_roc_xy_values(r_data.spd_roc[tt][ic, i_bin])

                            # sets the roc auc values
                            if is_resampled:
                                # case is the resampled frequencies
                                r_data.spd_roc_auc[tt][ic, i_bin] = spd_auc_roc
                            else:
                                # other cases
                                r_data.spd_roc_auc[tt][ic, i_bin] = cf.get_roc_auc_value(r_data.spd_roc[tt][ic, i_bin])

                # calculates the confidence intervals for the current (only if bootstrapping count has changed or
                # the confidence intervals has not already been calculated)
                if calc_ci is None:
                    if 'auc_stype' in calc_para:
                        # updates the auc statistics calculation type
                        r_data.kine_auc_stats_type = dcopy(calc_para['auc_stype'])

                        # determine if the auc confidence intervals need calculation
                        is_boot = int(calc_para['auc_stype'] == 'Bootstrapping')
                        if is_boot:
                            # if bootstrapping, then determine if the
                            if r_data.n_boot_kine_ci != calc_para['n_boot']:
                                # if the count has changed, flag the confidence intervals needs updating
                                r_data.n_boot_kine_ci, calc_ci = dcopy(calc_para['n_boot']), True
                            else:
                                # otherwise, recalculate the confidence intervals if they have not been set
                                calc_ci = np.any(r_data.vel_ci_lo[tt][ic, :, 1] < 0)
                        else:
                            # otherwise, recalculate the confidence intervals if they have not been set
                            calc_ci = np.any(r_data.vel_ci_lo[tt][ic, :, 0] < 0)

                # calculates the confidence intervals (if required)
                if calc_ci:
                    # calculates the velocity confidence intervals
                    auc_type, n_boot = calc_para['auc_stype'], calc_para['n_boot']
                    conf_int_vel = self.calc_roc_conf_intervals(pool, r_data.vel_roc[tt][ic, :],
                                                                auc_type, n_boot, c_lvl)

                    # resets the resampled confidence interval values
                    if not r_data.pn_comp and init_data:
                        conf_int_vel[r_data.i_bin_vel[0], :] = vel_auc_ci[0]
                        conf_int_vel[r_data.i_bin_vel[1], :] = vel_auc_ci[1]

                    # sets the upper and lower velocity confidence intervals
                    r_data.vel_ci_lo[tt][ic, :, is_boot] = conf_int_vel[:, 0]
                    r_data.vel_ci_hi[tt][ic, :, is_boot] = conf_int_vel[:, 1]

                    # calculates the speed confidence intervals
                    if not r_data.pn_comp:
                        # calculates the speed confidence intervals
                        conf_int_spd = self.calc_roc_conf_intervals(pool, r_data.spd_roc[tt][ic, :],
                                                                    auc_type, n_boot, c_lvl)

                        # resets the resampled confidence interval values
                        if init_data:
                            conf_int_spd[r_data.i_bin_spd] = spd_auc_ci

                        # sets the upper and lower speed confidence intervals
                        r_data.spd_ci_lo[tt][ic, :, is_boot] = conf_int_spd[:, 0]
                        r_data.spd_ci_hi[tt][ic, :, is_boot] = conf_int_spd[:, 1]

    def calc_roc_conf_intervals(self, pool, roc, phase_stype, n_boot, c_lvl):
        '''

        :param r_data:
        :return:
        '''

        # sets the parameters for the multi-processing pool
        p_data = []
        for i_cell in range(len(roc)):
            p_data.append([roc[i_cell], phase_stype, n_boot, c_lvl])

        # returns the rotation data class object
        return np.array(pool.map(cf.calc_roc_conf_intervals, p_data))

    def calc_kinematic_roc_significance(self, data, calc_para, g_para):
        '''

        :param data:
        :param calc_para:
        :param g_para:
        :return:
        '''

        # initialisations and other array indexing
        r_data = data.rotation
        is_boot, r_obj = int(calc_para['auc_stype'] == 'Bootstrapping'), r_data.r_obj_kine
        n_filt = r_obj.n_filt

        # sets the comparison bin for the velocity/speed arrays
        for use_vel in range(2):
            #
            if use_vel:
                i_bin = np.array([r_data.i_bin_vel])
                roc_auc, ci_lo, ci_hi = dcopy(r_data.vel_roc_auc), dcopy(r_data.vel_ci_lo), dcopy(r_data.vel_ci_hi)

            else:
                i_bin = np.array([r_data.i_bin_spd])
                roc_auc, ci_lo, ci_hi = dcopy(r_data.spd_roc_auc), dcopy(r_data.spd_ci_lo), dcopy(r_data.spd_ci_hi)

            # if the significance array is not set or the correct size, then reset the array dimensions
            is_sig = np.empty((n_filt,2), dtype=object)

            # determines the indices of the cell in the overall array
            t_type_base = list(r_data.spd_sf_rs.keys()) if r_data.is_equal_time else list(r_data.spd_sf.keys())
            for i_filt in range(n_filt):
                # determines the match condition with the currently calculated roc values
                tt = r_obj.rot_filt_tot[i_filt]['t_type'][0]
                i_match = t_type_base.index(tt)
                tt_nw = t_type_base[i_match]

                # determines which errorbars are significant
                ci_lo_tmp, ci_hi_tmp = ci_lo[tt][:, :, is_boot], ci_hi[tt][:, :, is_boot]
                is_sig[i_filt, is_boot] = np.logical_or((roc_auc[tt_nw] - ci_lo_tmp) > 0.5,
                                                        (roc_auc[tt_nw] + ci_hi_tmp) < 0.5)
                is_sig[i_filt, is_boot][:, i_bin] = False

            # updates the significance arrays (based on whether calculating for speed or velocity)
            if use_vel:
                r_data.vel_roc_sig = is_sig
            else:
                r_data.spd_roc_sig = is_sig

    ###################################################
    ####    MISCELLANEOUS FUNCTION CALCULATIONS    ####
    ###################################################

    def setup_spiking_freq_dataframe(self, data, calc_para):
        '''

        :param data:
        :param calc_para:
        :return:
        '''

        def get_mlt(t_type, i_dir):
            '''

            :param t_type:
            :param i_dir:
            :return:
            '''

            if t_type == 'MotorDrifting':
                # return [-1, 1][i_dir]
                return [1, -1][i_dir]
            else:
                return [1, -1][i_dir]

        def is_valid_cell_type(ch_region):
            '''

            :param ch_region:
            :return:
            '''

            # the valid region types
            valid_type = ['RSPd', 'RSPg', 'V1', 'Hip', 'SUB']

            # returns the cells which have a valid region type
            return np.array([ch_reg in valid_type for ch_reg in ch_region])

        def setup_expt_dataframe(data, calc_para, i_expt_rot, i_ex, i_ex_c, t_phase):
            '''

            :param data:
            :param calc_para:
            :param i_expt_rot:
            :param i_ex:
            :param t_phase:
            :return:
            '''

            # dictionaries and lambda function declarations
            d_str = {-1: 'CW', 1: 'CCW'}
            stack_arr = lambda y_arr, n_trial: np.hstack([yy * np.ones(n_trial) for yy in y_arr]).reshape(-1, 1)
            ind_fcn = lambda i_dir, cond: (1 - i_dir) if cond == 'MotorDrifting' else i_dir

            # DETERMINE VALID CELLS HERE!
            j_ex = i_expt_rot[i_ex]
            w, c = np.pi / t_phase, data._cluster[j_ex]
            is_ok = is_valid_cell_type(c['chRegion'])

            # other initialisations
            mlt = [-1, 1]
            cond_key = {'Black': 'Vestibular', 'Uniform': 'Visual + Vestibular', 'MotorDrifting': 'Visual',
                        'Mismatch1': 'Mismatch Opposite', 'Mismatch2': 'Mismatch Same'}
            r_filt, exp_name = calc_para['rot_filt'], cf.extract_file_name(c['expFile'])
            t_ofs0, n_cond, n_cell = 0., len(r_filt['t_type']), c['nC']
            t_phs, dt_ofs = calc_para['bin_sz'] / 1000., (calc_para['bin_sz'] - calc_para['t_over']) / 1000.

            # memory allocation
            n_bin_tot = int(np.floor((t_phase - dt_ofs) / dt_ofs)) + 1
            A = np.zeros((n_bin_tot, 1))
            p_bin, v_bin = dcopy(A), dcopy(A)

            # calculates the spiking frequencies for all cells over the duration configuration
            for i_bin_tot in range(n_bin_tot):
                # # check to see if the current time offset will allow for a feasible number of future time bins (i.e.,
                # # the current time bin + the future time bins must fit into the phase duration). if not then exit loop
                # if (t_ofs0 + t_phs) > t_phase:
                #     break

                # retrieves the filtered time spiking data for the current phase/duration configuration
                r_obj = RotationFilteredData(data, r_filt, None, exp_name, False, 'Whole Experiment', False,
                                             t_phase=t_phs, t_ofs=t_ofs0)

                # calculates the average spiking frequency data for the current experiment
                sp_f0, _ = cf.calc_phase_spike_freq(r_obj)

                # memory allocation (first iteration only)
                if i_bin_tot == 0:
                    n_cell = np.shape(sp_f0[0])[0]
                    wvm_para = r_obj.wvm_para

                    y_dir = [x[0]['yDir'] for x in wvm_para]
                    n_trial = [sum(~np.isnan(y)) for y in y_dir]

                    B = [np.empty(nt * n_bin_tot, dtype=object) for nt in n_trial]
                    sf, s_dir0 = dcopy(B), dcopy(B)

                # retrieves the CW/CCW phases (removes BL)
                sp_f_tmp = [sp_f[:, :, 1:] for sp_f in dcopy(sp_f0)]

                # if the first bin, calculate the average speed over the bin's duration
                w_vals0 = rot.calc_waveform_values(90, w, t_ofs0)
                w_vals1 = rot.calc_waveform_values(90, w, t_ofs0 + t_phs)
                p_bin[i_bin_tot] = 0.5 * (w_vals1[0] + w_vals0[0]) + 90
                v_bin[i_bin_tot] = 0.5 * (w_vals1[1] + w_vals0[1])

                # splits/stores the spiking frequency by the condition
                for i_cond in range(n_cond):
                    i_trial = 0
                    for i in range(len(y_dir[i_cond])):
                        # if there was an error with the trial, then continue
                        if np.isnan(y_dir[i_cond][i]):
                            continue

                        # sets the spiking frequency values
                        ind_sf = i_bin_tot * n_trial[i_cond] + i_trial
                        sf[i_cond][ind_sf] = sp_f_tmp[i_cond][:, i, :]

                        # sets the direction string
                        i_dir0 = y_dir[i_cond][i]
                        s_dir0[i_cond][ind_sf] = d_str[i_dir0]

                        # increments the trial counter
                        i_trial += 1

                # increments the time offset by the time-overlap
                t_ofs0 += dt_ofs

            # initialisations
            df_tot, tt = [], r_filt['t_type']
            g_str = {'Nar': 'Narrow', 'Wid': 'Wide', 'N/A': 'N/A'}

            # sets the trial condition type column
            tt_col = np.hstack([cf.flat_list([[cond_key[_tt]] * (2 * _nt * n_bin_tot)])
                                for _tt, _nt in zip(tt, n_trial)]).reshape(-1, 1)
            bin_col = np.vstack([repmat(np.vstack([(i + 1) * np.ones((_nt, 1), dtype=int)
                                                   for i in range(n_bin_tot)]), 2, 1) for _nt in n_trial])
            trial_col = np.vstack([repmat(np.arange(_nt).reshape(-1, 1) + 1, 2 * n_bin_tot, 1) for _nt in n_trial])

            for i_cell in range(n_cell):
                # combines the information for the current cell
                sf_cell = np.vstack(
                    [np.vstack(
                        [np.hstack((stack_arr(p_bin, nt) if (mlt[i_dir] > 0) else (180 - stack_arr(p_bin, nt)),
                                    mlt[i_dir] * stack_arr(v_bin, nt), s_dir0[i_cond].reshape(-1, 1),
                                    np.array([_sf[i_cell, ind_fcn(i_dir, tt[i_cond])] for _sf in sf[i_cond]]).reshape(
                                        -1, 1)))
                         for i_dir in range(2)])
                        for i_cond, nt in enumerate(n_trial)]
                )

                # # combines the information for the current cell
                # sf_cell = np.vstack(
                #     [np.vstack(
                #         [np.hstack((stack_arr(p_bin, nt) if (get_mlt(tt[i_cond], i_dir) > 0) else (180 - stack_arr(p_bin, nt)),
                #                     get_mlt(tt[i_cond], i_dir) * stack_arr(v_bin, nt),
                #                     np.array([_sf[i_cell, i_dir] for _sf in sf[i_cond]]).reshape(-1, 1)))
                #                     for i_dir in range(2)])
                #     for i_cond, nt in enumerate(n_trial)]
                # )

                # sets the other column details
                n_row = np.size(sf_cell, axis=0)
                reg_col = np.array([c['chRegion'][i_cell]] * n_row).reshape(-1, 1)
                layer_col = np.array([c['chLayer'][i_cell]] * n_row).reshape(-1, 1)

                # sets the cell indices
                ind_col = (i_cell + 1) * np.ones((n_row, 1), dtype=int)

                # appends all the data for the given cell
                if data.classify.class_set:
                    # sets the cell classification type ('N/A' if 'SC'/'N/A', otherwise use the classification string)
                    g_str_nw = g_str[data.classify.grp_str[i_expt_rot[i_ex]][i_cell]] if is_ok[i_cell] else 'N/A'

                    # adds in the cell group type (if calculated)
                    grp_col = np.array([g_str_nw] * n_row).reshape(-1, 1)
                    df_tot.append(
                        np.hstack((ind_col, bin_col, trial_col, sf_cell, tt_col, reg_col, layer_col, grp_col)))

                else:
                    # otherwise, use the existing information only
                    df_tot.append(np.hstack((ind_col, bin_col, trial_col, sf_cell, tt_col, reg_col, layer_col)))

            # combines all data from each cell (along with the experiment index) into a final np array
            exp_col = (i_ex_c + 1) * np.ones((n_row * n_cell, 1), dtype=int)
            return np.hstack((exp_col, np.vstack(df_tot)))

        # determines the valid rotation experiments
        i_expt_rot = np.where(cf.det_valid_rotation_expt(data))[0]

        # memory allocation and initialisations
        n_ex = len(i_expt_rot)
        sf_data = np.empty(n_ex, dtype=object)
        w_prog, d_data = self.work_progress, data.spikedf

        # retrieves the rotation filter
        r_filt = calc_para['rot_filt']
        if r_filt is None:
            # if not set, then initialise
            r_filt = cf.init_rotation_filter_data(False)

        # returns the overall rotation filter class object
        r_obj = RotationFilteredData(data, r_filt, None, None, True, 'Whole Experiment', False)
        t_phase, t_type, i_ex_c = r_obj.t_phase[0][0], calc_para['rot_filt']['t_type'], 0

        # creates the spiking frequency dataframe for the each experiment
        for i_ex in range(n_ex):
            # updates the progress bar
            w_str = 'Combining Spike Freq. Data (Expt #{0}/{1})'.format(i_ex + 1, n_ex)
            w_prog.emit(w_str, 100. * (i_ex / n_ex))

            # determines if all trial types exist within the current experiment
            tt_expt = list(data._cluster[i_expt_rot[i_ex]]['rotInfo']['trial_type'])
            if np.all([tt in tt_expt for tt in t_type]):
                # if so, then set the data for the current experiment
                sf_data[i_ex] = setup_expt_dataframe(data, calc_para, i_expt_rot, i_ex, i_ex_c, t_phase)
                if sf_data[i_ex] is not None:
                    i_ex_c += 1

        ######################################
        ####    HOUSEKEEPING EXERCISES    ####
        ######################################

        # updates the progressbar
        w_prog.emit('Setting Final Dataframe...', 100.)

        # sets the calculation parameters
        d_data.rot_filt = dcopy(calc_para['rot_filt'])
        d_data.bin_sz = calc_para['bin_sz']
        d_data.t_over = calc_para['t_over']

        # creates the final dataframe
        c_str = ['Expt #', 'Cell #', 'Bin #', 'Trial #', 'Position (deg)', 'Speed (deg/s)', 'Initial Dir'] + \
                ['Firing Rate', 'Trial Condition', 'Region', 'Layer'] + \
                (['Cell Type'] if data.classify.class_set else [])
        sf_data_valid = np.vstack([x for x in sf_data if x is not None])
        d_data.sf_df = pd.DataFrame(sf_data_valid, columns=c_str)

    def calc_auto_ccgram_fft(self, data, calc_para):
        '''

        :param data:
        :param calc_para:
        :return:
        '''

        # parameters
        n_count = 0
        t_bin = calc_para['t_bin']
        n_bin = int(t_bin / calc_para['bin_sz'])    # the number of time bins
        f_theta = [5, 11]                           # theta frequency range (from Yartsev 2011)
        freq_rng = [0, 50]                          # theta index comparison frequency range (from Yartsev 2011)
        ratio_tol = 5                               # threshold ratio (from Yartsev 2011)
        n_pad = 2 ** 16

        # sets up the psd frequency
        df = (2 * t_bin) / n_pad
        f = np.arange(0, 2 * t_bin, df) / calc_para['bin_sz']
        i_theta_f0 = np.logical_and(f >= f_theta[0], f <= f_theta[1])
        i_theta_nf, i_theta_f = np.where(np.logical_and(~i_theta_f0, f <= freq_rng[1]))[0], np.where(i_theta_f0)[0]

        # calculates the number of bins for 1Hz within the freq. range
        dn = int(np.floor(1 / df))

        # # sets the array index ranges
        # i_theta = np.arange(f_theta[0], f_theta[1] + 1)
        # i_freq_rng = np.arange(freq_rng[0], freq_rng[1] + 1)

        # sets up the boolean array for the non-zero lag bins (used to set the zero-lag bin value below)
        is_ok = np.ones(2 * n_bin - 1, dtype=bool)
        is_ok[n_bin - 1] = False

        # memory allocation and other initialisations
        is_free = np.logical_not(cf.det_valid_rotation_expt(data))
        a = np.empty(np.sum(is_free), dtype=object)
        cc_gram, p_fft, th_index = dcopy(a), dcopy(a), dcopy(a)
        w_prog, th_data = self.work_progress, data.theta_index
        exp_name = [cf.extract_file_name(c['expFile']) for c in np.array(data._cluster)[is_free]]

        # retrieves the time spike arrays
        t_spike = [c['tSpike'] for c, i in zip(data._cluster, is_free) if i]
        n_cell_tot = np.sum([len(x) for x in t_spike])

        # for each free experiment, calculate the theta index for each cell
        n_expt = len(t_spike)
        for i_expt in range(n_expt):
            # memory allocation for the current expt
            n_cell = len(t_spike[i_expt])
            cc_gram[i_expt] = np.zeros((n_cell, 2 * n_bin - 1))
            p_fft[i_expt] = np.zeros((n_cell, int(n_pad / 2)))
            th_index[i_expt] = np.zeros((n_cell, 2))

            # calculates the theta index for each cell in the experiment
            for i_cell in range(n_cell):
                # updates the progress bar
                n_count += 1
                w_str = 'Theta Index (Expt={0}/{1}, Cell={2}/{3})'.format(i_expt + 1, n_expt, i_cell + 1, n_cell)
                w_prog.emit(w_str, 100. * (n_count / (n_cell_tot + 1)))

                # calculates the new autocorrelogram for the current cell
                t_sp = t_spike[i_expt][i_cell]
                cc_gram[i_expt][i_cell, :], _ = cfcn.calc_ccgram(t_sp, t_sp, t_bin, bin_size=calc_para['bin_sz'])

                # sets the zero-lag bin value to be the max non zero-lag cc-gram bin value
                cc_gram[i_expt][i_cell, n_bin - 1] = np.max(cc_gram[i_expt][i_cell, is_ok])

                # calculates the PSD estimate of the cc-gram
                cc_gram_calc = cc_gram[i_expt][i_cell, :]
                if calc_para['remove_bl']:
                    cc_gram_calc -= np.mean(cc_gram[i_expt][i_cell, :])

                if calc_para['pow_type'] == 'FFT-Squared':
                    # calculates using the square of the FFT
                    if calc_para['win_type'] == 'none':
                        # if no signal windowing, then scale the signal by its length
                        y_sig = cc_gram_calc / len(cc_gram_calc)
                    else:
                        # otherwise, set the windowing function based on the specified type
                        if calc_para['win_type'] == 'boxcar':
                            y_win = boxcar(len(cc_gram_calc))
                        else:
                            y_win = hamming(len(cc_gram_calc))

                        # applies the windowing function
                        y_sig = np.multiply(cc_gram_calc / len(cc_gram_calc), y_win)

                    # pads zero to the end of the function (increases resolution for the PSD)
                    y_sig_pad = np.pad(y_sig, (0, n_pad - (2 * n_bin - 1)), 'constant')

                    # calculates the fft of the signal and calculates the power spectrum
                    y_fft = np.fft.fft(y_sig_pad)
                    p_fft0 = np.abs(y_fft)

                    # rectangular smoothing of the PSD (2Hz in length)
                    p_fft_mn0 = pd.DataFrame(p_fft0).rolling(2 * dn, min_periods=1, center=True).mean()
                    p_fft_mn = np.array(p_fft_mn0.ix[:, 0])

                    # taking positive frequency range of PSD for visualisation
                    p_fft[i_expt][i_cell, :] = p_fft_mn[:int(n_pad / 2)]

                else:
                    # calculates using the periodgram method
                    _, p_fft[i_expt][i_cell, :] = periodogram(cc_gram_calc, window=calc_para['win_type'])

                # calculates the location of the max peak within the theta range
                i_fft_mx = find_peaks(p_fft0[i_theta_f])[0]
                if len(i_fft_mx):
                    i_mx = np.argmax(p_fft0[i_theta_f][i_fft_mx])
                    if_mx = i_theta_f[i_fft_mx[i_mx]]

                    # calculates the theta index numerator/denominator
                    th_index_num = np.mean(p_fft0[(if_mx-dn):(if_mx+dn)])       # mean power for +/- 1Hz surrounding peak within theta range
                    th_index_den = np.mean(p_fft0[i_theta_nf])                  # mean power spectrum outside of theta range

                else:
                    # if there are no peaks, then ensure the theta index value is zero
                    th_index_num, th_index_den = 0, 1

                # calculates the theta index of the signal
                #  this is calculate as the ratio of the mean of the points surrounding the max power spectrum value
                #  between the 5-11Hz freq range divided by the mean power spectrum values btwn the 1-125Hz freq range
                th_index[i_expt][i_cell, 0] = th_index_num / th_index_den
                th_index[i_expt][i_cell, 1] = th_index[i_expt][i_cell, 0] > ratio_tol

        #######################################
        ####    HOUSE-KEEPING EXERCISES    ####
        #######################################

        # sets the final values into the class object
        th_data.cc_gram = cc_gram
        th_data.p_fft = p_fft
        th_data.th_index = th_index
        th_data.f = f

        # sets the other fields
        th_data.is_set = True
        th_data.exp_name = exp_name
        th_data.t_bin = calc_para['t_bin']
        th_data.bin_sz = calc_para['bin_sz']
        th_data.vel_bin = calc_para['vel_bin']
        th_data.win_type = calc_para['win_type']
        th_data.remove_bl = calc_para['remove_bl']

    ###########################################
    ####    OTHER CALCULATION FUNCTIONS    ####
    ###########################################

    def check_combined_conditions(self, calc_para, plot_para):
        '''

        :param calc_para:
        :param plot_para:
        :return:
        '''

        if plot_para['rot_filt'] is not None:
            if 'MotorDrifting' in plot_para['rot_filt']['t_type']:
                # if the mapping file is not correct, then output an error to screen
                e_str = 'MotorDrifting is not a valid filter option when running this function.\n\n' \
                        'De-select this filter option before re-running this function.'
                self.work_error.emit(e_str, 'Invalid Filter Options')

                # returns a false value
                return False

        # if everything is correct, then return a true value
        return True

    def check_altered_para(self, data, calc_para, plot_para, g_para, chk_type, other_para=None):
        '''

        :param calc_para:
        :param g_para:
        :param chk_type:
        :return:
        '''

        def check_class_para_equal(d_data, attr, chk_value, def_val=False):
            '''

            :param d_data:
            :param attr:
            :param chk_value:
            :return:
            '''

            if hasattr(d_data, attr):
                return getattr(d_data, attr) == chk_value
            else:
                return def_val

        # initialisations
        r_data, ff_corr = data.rotation, data.comp.ff_corr if hasattr(data.comp, 'ff_corr') else None
        t_ofs, t_phase = cfcn.get_rot_phase_offsets(calc_para)

        # loops through each of the check types determining if any parameters changed
        for ct in chk_type:
            # initialises the change flag
            is_change = data.force_calc

            if ct == 'condition':
                # case is the roc condition parameters

                # retrieves the rotation phase offset time/duration
                if t_ofs is not None:
                    # if the values are not none, and do not match previous values, then reset the stored roc array
                    if (r_data.t_ofs_rot != t_ofs) or (r_data.t_phase_rot != t_phase):
                        r_data.t_ofs_rot, r_data.t_phase_rot, is_change = t_ofs, t_phase, True
                elif 'use_full_rot' in calc_para:
                    # if using the full rotation, and the previous calculations were made using non-full rotation
                    # phases, the reset the stored roc array
                    if (r_data.t_ofs_rot > 0):
                        r_data.t_ofs_rot, r_data.t_phase_rot, is_change = -1, -1, True

                # if there was a change, then re-initialise the roc condition fields
                if is_change:
                    # memory allocation (if the conditions have not been set)
                    r_data.phase_roc, r_data.phase_roc_auc, r_data.phase_roc_xy = {}, {}, {}
                    r_data.phase_ci_lo, self.phase_ci_hi, self.phase_gtype = None, None, None
                    r_data.phase_auc_sig, r_data.phase_grp_stats_type = None, None

                    r_data.cond_roc, r_data.cond_roc_xy, r_data.cond_roc_auc = {}, {}, {}
                    r_data.cond_gtype, r_data.cond_auc_sig, r_data.cond_i_expt, r_data.cond_cl_id = {}, {}, {}, {}
                    r_data.cond_ci_lo, r_data.cond_ci_hi, r_data.r_obj_cond = {}, {}, {}
                    r_data.phase_gtype, r_data.phase_auc_sig, r_data.phase_roc = None, None, None

                    r_data.part_roc, r_data.part_roc_xy, r_data.part_roc_auc = {}, {}, {}

            elif ct == 'clust':
                # case is the fixed/free cell clustering calculations
                i_expt = cf.det_comp_dataset_index(data.comp.data, calc_para['calc_comp'])
                c_data = data.comp.data[i_expt]

                # if the calculations have not been made, then exit the function
                if not c_data.is_set:
                    continue

                # determines if the global parameters have changed
                is_equal = [
                    check_class_para_equal(c_data, 'd_max', calc_para['d_max']),
                    check_class_para_equal(c_data, 'r_max', calc_para['r_max']),
                    check_class_para_equal(c_data, 'sig_corr_min', calc_para['sig_corr_min']),
                    check_class_para_equal(c_data, 'isi_corr_min', calc_para['isi_corr_min']),
                    check_class_para_equal(c_data, 'sig_diff_max', calc_para['sig_diff_max']),
                    check_class_para_equal(c_data, 'sig_feat_min', calc_para['sig_feat_min']),
                    check_class_para_equal(c_data, 'w_sig_feat', calc_para['w_sig_feat']),
                    check_class_para_equal(c_data, 'w_sig_comp', calc_para['w_sig_comp']),
                    check_class_para_equal(c_data, 'w_isi', calc_para['w_isi']),
                ]

                # determines if there was a change in parameters (and hence a recalculation required)
                c_data.is_set = np.all(is_equal)

            elif ct == 'ff_corr':

                # case is the fixed/freely moving spiking frequency correlation analysis
                is_equal = [
                    not data.force_calc,
                    check_class_para_equal(ff_corr, 'vel_bin', float(calc_para['vel_bin'])),
                    check_class_para_equal(ff_corr, 'n_shuffle_corr', float(calc_para['n_shuffle'])),
                    check_class_para_equal(ff_corr, 'split_vel', int(calc_para['split_vel'])),
                ]

                # determines if recalculation is required
                ff_corr.is_set = np.all(is_equal)
                if not ff_corr.is_set:
                    data.force_calc = True

            elif ct == 'eye_track':

                # case is the eye tracking data
                et_data = data.externd.eye_track

                # if the calculations have not been made, then exit the function
                if not et_data.is_set:
                    return

                # case is the fixed/freely moving spiking frequency correlation analysis
                is_equal = [
                    check_class_para_equal(et_data, 'dp_max', float(calc_para['dp_max'])),
                    check_class_para_equal(et_data, 'n_sd', float(calc_para['n_sd'])),
                    check_class_para_equal(et_data, 'n_pre', int(calc_para['n_pre'])),
                    check_class_para_equal(et_data, 'n_post', int(calc_para['n_post'])),
                ]

                # determines if recalculation is required
                et_data.is_set = np.all(is_equal)
                if not et_data.is_set:
                    et_data.t_evnt, et_data.y_evnt, et_data.sp_evnt = [], [], []
                    et_data.y_corr, et_data.t_sp_h = [], []

            elif ct == 'phase':
                # case is the phase ROC calculations
                pass

            elif ct == 'visual':
                # retrieves the visual phase time offset/duration
                t_ofs_vis, t_phase_vis = cfcn.get_rot_phase_offsets(calc_para, True)

                # if the values are not none, and do not match previous values, then reset the stored roc array
                if (r_data.t_ofs_vis != t_ofs_vis) or (r_data.t_phase_vis != t_phase_vis):
                    r_data.t_ofs_vis, r_data.t_phase_vis, is_change = t_ofs_vis, t_phase_vis, True

                # if there was a change, then re-initialise the fields
                if is_change:
                    r_data.phase_roc_ud, r_data.phase_roc_auc_ud, r_data.phase_roc_xy_ud = None, None, None

            elif ct == 'vel':
                # case is the kinematic calculations

                # initialisations
                roc_calc = other_para
                vel_bin = float(calc_para['vel_bin']) if ('vel_bin' in calc_para) else float(plot_para['vel_bin'])

                # checks to see if the dependent speed has changed
                if 'spd_x_rng' in calc_para:
                    # case is a single speed bin range comparison

                    # if the dependent speed range has changed then reset the roc curve calculations
                    if r_data.comp_spd != calc_para['spd_x_rng']:
                        is_change = True

                    if r_data.pn_comp is True:
                        r_data.pn_comp, is_change = False, True

                    # updates the speed comparison flag
                    r_data.comp_spd = dcopy(calc_para['spd_x_rng'])

                else:
                    # case is the positive/negative speed comparison

                    # if the positive/negative comparison flag is not set to true, then reset the roc curve calculations
                    if r_data.pn_comp is False:
                        r_data.pn_comp, is_change = True, True

                    # if using equal time bins, then check to see if the sample size has changed (if so then recalculate)
                    if calc_para['equal_time']:
                        if r_data.n_rs != calc_para['n_sample']:
                            r_data.vel_sf_rs, r_data.spd_sf_rs = None, None
                            r_data.n_rs, is_change = dcopy(calc_para['n_sample']), True

                # if the velocity bin size has changed or isn't initialised, then reset velocity roc values
                if data.force_calc:
                    r_data.vel_sf_rs, r_data.spd_sf_rs = None, None
                    r_data.vel_sf, r_data.spd_sf = None, None

                if roc_calc:
                    if (vel_bin != r_data.vel_bin) or (calc_para['freq_type'] != r_data.freq_type):
                        r_data.vel_sf_rs, r_data.spd_sf_rs = None, None
                        r_data.vel_sf, r_data.spd_sf = None, None
                        r_data.vel_bin, is_change = vel_bin, True
                        r_data.freq_type = dcopy(calc_para['freq_type'])

                    if r_data.is_equal_time != calc_para['equal_time']:
                        is_change = True

                    # if there was a change, then re-initialise the roc phase fields
                    if is_change:
                        r_data.vel_roc = None

                else:
                    if (vel_bin != r_data.vel_bin):
                        r_data.vel_sf_rs, r_data.spd_sf_rs = None, None
                        r_data.vel_sf, r_data.spd_sf = None, None

            elif ct == 'vel_sf_fix':
                # if the spiking frequency calculation field has not been set, then force an update
                if not hasattr(r_data, 'vel_shuffle_calc'):
                    data.force_calc = True

                # case is the kinematic spiking frequency calculations
                is_equal = [
                    check_class_para_equal(r_data, 'vel_sf_nsm', calc_para['n_smooth'] * calc_para['is_smooth']),
                    check_class_para_equal(r_data, 'vel_bin_corr', float(calc_para['vel_bin'])),
                    check_class_para_equal(r_data, 'n_shuffle_corr', calc_para['n_shuffle']),
                    check_class_para_equal(r_data, 'split_vel', calc_para['split_vel']),
                    check_class_para_equal(r_data, 'vel_sf_eqlt', calc_para['equal_time'])
                ]

                # if there was a change in any of the parameters, then reset the spiking frequency fields
                if not np.all(is_equal) or data.force_calc:
                    r_data.vel_shuffle_calc, r_data.vel_sf_corr = False, None
                    r_data.vel_sf, r_data.vel_sf_rs = None, None

                # determines if all trial conditions have been calculated (for calculation if not)
                if r_data.vel_shuffle_calc:
                    t_type = list(r_data.vel_sf_mean.keys())
                    r_data.vel_shuffle_calc = np.all([tt in t_type for tt in plot_para['rot_filt']['t_type']])

            elif ct == 'vel_sf_free':
                # case is the kinematic spiking frequency calculations
                is_equal = [
                    check_class_para_equal(r_data, 'vel_bin_corr', float(calc_para['vel_bin'])),
                ]

                # if there was a change in any of the parameters, then reset the spiking frequency fields
                if not np.all(is_equal) or data.force_calc:
                    r_data.vel_shuffle_calc, r_data.vel_sf_corr = False, None
                    r_data.vel_sf, r_data.vel_sf_rs = None, None

            elif ct == 'lda':
                # case is the LDA calculations

                # if initialising the LDA then continue (as nothing has been set)
                d_data, lda_para, lda_tt = other_para, calc_para['lda_para'], cfcn.get_glob_para('lda_trial_type')
                if d_data.lda is None:
                    continue

                # otherwise, determine if there are any changes in the parameters
                is_equal = [
                    check_class_para_equal(d_data, 'solver', lda_para['solver_type']),
                    check_class_para_equal(d_data, 'shrinkage', lda_para['use_shrinkage']),
                    check_class_para_equal(d_data, 'norm', lda_para['is_norm']),
                    check_class_para_equal(d_data, 'cellmin', lda_para['n_cell_min']),
                    check_class_para_equal(d_data, 'trialmin', lda_para['n_trial_min']),
                    check_class_para_equal(d_data, 'yaccmx', lda_para['y_acc_max']),
                    check_class_para_equal(d_data, 'yaccmn', lda_para['y_acc_min'], def_val=True),
                    check_class_para_equal(d_data, 'yaucmx', lda_para['y_auc_max'], def_val=True),
                    check_class_para_equal(d_data, 'yaucmn', lda_para['y_auc_min'], def_val=True),
                    check_class_para_equal(d_data, 'lda_trial_type', lda_tt, def_val=True),
                    check_class_para_equal(d_data, 'fctype', lda_para['free_ctype'], def_val='All'),
                    set(d_data.ttype) == set(lda_para['comp_cond']),
                ]

                #
                if d_data.type in ['Direction', 'Individual', 'TrialShuffle', 'Partial', 'IndivFilt', 'LDAWeight']:
                    if 'use_full_rot' in calc_para:
                        if d_data.usefull:
                            is_equal += [
                                check_class_para_equal(d_data, 'usefull', calc_para['use_full_rot']),
                            ]
                        else:
                            if 't_ofs_rot' in calc_para:
                                is_equal += [
                                    check_class_para_equal(d_data, 'tofs', calc_para['t_ofs_rot']),
                                    check_class_para_equal(d_data, 'tphase', calc_para['t_phase_rot']),
                                ]
                            else:
                                is_equal += [
                                    check_class_para_equal(d_data, 'tofs', calc_para['t_ofs']),
                                    check_class_para_equal(d_data, 'tphase', calc_para['t_phase']),
                                ]

                    if d_data.type in ['Direction']:
                        is_equal += [
                            hasattr(d_data, 'z_corr')
                        ]

                    elif d_data.type in ['TrialShuffle']:
                        is_equal += [
                            check_class_para_equal(d_data, 'nshuffle', calc_para['n_shuffle']),
                        ]

                    elif d_data.type in ['IndivFilt']:
                        is_equal += [
                            check_class_para_equal(d_data, 'yaccmn', calc_para['y_acc_min']),
                            check_class_para_equal(d_data, 'yaccmx', calc_para['y_acc_max']),
                        ]

                    elif d_data.type in ['Partial']:
                        is_equal[3] = True

                        is_equal += [
                            check_class_para_equal(d_data, 'nshuffle', calc_para['n_shuffle']),
                        ]

                elif d_data.type in ['Temporal']:
                    is_equal += [
                        check_class_para_equal(d_data, 'dt_phs', calc_para['dt_phase']),
                        check_class_para_equal(d_data, 'dt_ofs', calc_para['dt_ofs']),
                        check_class_para_equal(d_data, 'phs_const', calc_para['t_phase_const']),
                     ]

                elif d_data.type in ['SpdAcc', 'SpdComp', 'SpdCompPool', 'SpdCompDir']:
                    is_equal += [
                        check_class_para_equal(d_data, 'vel_bin', calc_para['vel_bin']),
                        check_class_para_equal(d_data, 'n_sample', calc_para['n_sample']),
                        check_class_para_equal(d_data, 'equal_time', calc_para['equal_time']),
                     ]

                    if d_data.type in ['SpdComp', 'SpdCompPool']:
                        is_equal += [
                            check_class_para_equal(d_data, 'spd_xrng', calc_para['spd_x_rng']),
                        ]

                    if d_data.type in ['SpdCompPool']:
                        is_equal += [
                            check_class_para_equal(d_data, 'nshuffle', calc_para['n_shuffle']),
                            check_class_para_equal(d_data, 'poolexpt', calc_para['pool_expt']),
                        ]

                        # if there was a change in any of the parameters, then flag recalculation is needed
                if not np.all(is_equal) or data.force_calc:
                    d_data.lda = None

            elif ct == 'spikedf':
                # initialisations
                d_data = other_para

                # if the spike frequency dataframe has not been setup, then exit the function
                if not d_data.is_set:
                    return

                # case is the spiking frequency dataframe
                is_equal = [
                    check_class_para_equal(d_data, 'rot_filt', calc_para['rot_filt']),
                    check_class_para_equal(d_data, 'bin_sz', calc_para['bin_sz']),
                    check_class_para_equal(d_data, 't_over', calc_para['t_over']),
                ]

                # if there was a change in any of the parameters, then flag recalculation is needed
                if not np.all(is_equal) or data.force_calc:
                    d_data.is_set = False

            elif ct == 'theta':
                # initialisations
                th_data = other_para

                # if the data is not calculated, then exit the function
                if not th_data.is_set:
                    return

                # determines the calculation parameter that have been altered
                is_equal = [
                    check_class_para_equal(th_data, 'vel_bin', calc_para['vel_bin']),
                    check_class_para_equal(th_data, 'bin_sz', calc_para['bin_sz']),
                    check_class_para_equal(th_data, 'win_type', calc_para['win_type']),
                    check_class_para_equal(th_data, 'remove_bl', calc_para['remove_bl']),
                ]

                # if there was a change in any of the parameters, then flag recalculation is needed
                if not np.all(is_equal) or data.force_calc:
                    th_data.is_set = False