# module import
import os
import re
import copy
import time
import platform
import pickle as p
import numpy as np

# pyqt5 module imports
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QHBoxLayout, QDialog, QFormLayout, QPushButton, QGridLayout, QGroupBox,
                             QHeaderView, QMessageBox, QTableWidget)

# custom module import
import analysis_guis.common_func as cf
import analysis_guis.calc_functions as cfcn

# font objects
txt_font = cf.create_font_obj()
txt_font_bold = cf.create_font_obj(is_bold=True, font_weight=QFont.Bold)
grp_font_sub = cf.create_font_obj(size=10, is_bold=True, font_weight=QFont.Bold)
grp_font_sub2 = cf.create_font_obj(size=9, is_bold=True, font_weight=QFont.Bold)
grp_font_main = cf.create_font_obj(size=12, is_bold=True, font_weight=QFont.Bold)

# other initialisations
dX = 10
dY = 10

# sets the style data
styleData="""
QPushButton
{
    font-size: 10;
    font-weight: bold;
}
QGroupBox
{
    font-weight: bold;
    font-size: 14;
}
# QLabel { 
#     background-color: white;
# }
"""

dcopy = copy.deepcopy

########################################################################################################################
########################################################################################################################


class InfoDialog(QDialog):
    def __init__(self, main_obj, parent=None, width=1500, height=600, rot_filt=None):
        # creates the gui object
        super(InfoDialog, self).__init__(parent)

        # field initialisations
        self.main_obj = main_obj
        self.get_data_fcn = main_obj.get_data
        self.rot_filt = rot_filt
        self.can_close = False

        #
        self.init_gui_objects(width, height)
        self.init_all_expt_groups()
        self.create_control_buttons()

        # shows and executes the dialog box
        self.show()
        self.exec()

    def init_gui_objects(self, width, height):
        '''

        :return:
        '''

        # retrieves the loaded data object from the main gui
        self.data = self.get_data_fcn()

        # width dimensions
        self.gui_width = width
        self.grp_wid_main = self.gui_width - 2 * dX
        self.grp_wid_expt = self.grp_wid_main - 0.5 * dX
        self.grp_wid_info = self.grp_wid_expt - 0.6 * dX

        # height dimensions
        self.gui_hght = height
        self.grp_hght_main = self.gui_hght - (2*dY + 55)
        self.grp_hght_expt = self.grp_hght_main - (2 * dY)
        self.grp_hght_info = self.grp_hght_expt - (2 * dY)

        # memory allocation
        self.n_expt = len(self.data._cluster)
        self.h_expt = np.empty(self.n_expt, dtype=object)
        self.h_info = np.empty((self.n_expt, 2), dtype=object)
        self.h_grpbx = np.empty(2, dtype=object)

        # main layout object
        self.mainLayout = QGridLayout()

        # sets the final window properties
        self.setWindowTitle('Experiment Information')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setStyleSheet(styleData)

    def init_all_expt_groups(self):
        '''

        :param data:
        :return:
        '''

        # creates the tab groups for each experiment
        for i_expt in range(self.n_expt):
            self.create_expt_group(i_expt)

        # creates the tab object
        self.h_grpbx[0] = cf.create_tab(None, QRect(10, 10, self.grp_wid_main, self.grp_hght_main), None,
                                        h_tabchild=[x for x in self.h_expt],
                                        child_name=['Expt #{0}'.format(i_expt+1) for i_expt in range(self.n_expt)])
        cf.set_obj_fixed_size(self.h_grpbx[0], width=self.grp_wid_main, height=self.grp_hght_main)

        # sets the main widget into the GUI
        self.mainLayout.addWidget(self.h_grpbx[0], 0, 0)
        self.setLayout(self.mainLayout)

        # sets the gui's fixed dimensions
        cf.set_obj_fixed_size(self, width=self.gui_width, height=self.gui_hght)

    def create_control_buttons(self):
        '''

        :return:
        '''

        # initialisations
        b_txt = ['Refresh', 'Output Information Data', 'Close Window']
        cb_fcn = [self.refresh_fields, self.output_info, self.close_window]
        b_name = ['refresh_fields', 'output_info', 'close_window']

        # group box object
        self.h_grpbx[1] = QGroupBox("")
        layout = QHBoxLayout()

        # creates the load config file object
        for i in range(len(b_txt)):
            # creates the button object
            hButton = QPushButton(b_txt[i])
            hButton.clicked.connect(cb_fcn[i])
            hButton.setObjectName(b_name[i])
            hButton.setAutoDefault(False)
            cf.update_obj_font(hButton, pointSize=9)

            # adds the objects to the layout
            layout.addWidget(hButton)

        # sets the box layout
        self.h_grpbx[1].setLayout(layout)
        self.mainLayout.addWidget(self.h_grpbx[1], 1, 0)

    def create_expt_group(self, i_expt):
        '''

        :param i_expt:
        :return:
        '''

        # creates the calculation/plotting parameter
        self.h_info[i_expt, 0] = cf.create_groupbox(None, QRect(10, 10, self.grp_wid_info, self.grp_hght_info),
                                                    grp_font_sub2, "", "calc_para")
        self.h_info[i_expt, 1] = cf.create_groupbox(None, QRect(10, 10, self.grp_wid_info, self.grp_hght_info),
                                                    grp_font_sub2, "", "plot_para")
        for hh in self.h_info[i_expt, :]:
            cf.set_obj_fixed_size(hh, width=self.grp_wid_info, height=self.grp_hght_info)

        # creates the tab object
        self.h_expt[i_expt]= cf.create_tab(None, QRect(5, 55, self.grp_wid_expt, self.grp_hght_expt), None,
                                           h_tabchild=[self.h_info[i_expt, 0], self.h_info[i_expt, 1]],
                                           child_name=['Experiment Info', 'Cluster Info'])
        cf.set_obj_fixed_size(self.h_expt[i_expt], width=self.grp_wid_expt, height=self.grp_hght_expt)

        # initialises the groupbox layout
        self.h_info[i_expt, 0].setLayout(QFormLayout())
        self.h_info[i_expt, 1].setLayout(QFormLayout())

        #
        self.setup_expt_info(i_expt)
        self.setup_cluster_info(i_expt)

    def setup_expt_info(self, i_expt):
        '''

        :param i_expt:
        :return:
        '''

        # retrieves the cluster data
        c_data = self.data._cluster[i_expt]

        # removes all parameters from the layout
        h_layout = self.h_info[i_expt, 0].layout()
        for i_row in range(h_layout.rowCount()):
            h_layout.removeRow(0)

        # sets the experiment information fields
        expt_info = [
            ['Experiment Name', 'name', True],
            ['Experiment Date', 'date', True],
            ['Experiment Condition', 'cond', True],
            ['Experiment Type', 'type', True],
            ['Specimen Sex', 'sex', True],
            ['Specimen Age', 'age', True],
            ['Probe Name', 'probe', True],
            ['Lesion Location', 'lesion', True],
            ['Cluster Types', 'cluster_type', True],
            ['Recording State', 'record_state', True],
            ['Recording Coordinate', 'record_coord', True],
            ['Probe Depth (um)', 'probe_depth', True],
            ['Cluster Count', 'nC', False],
            ['Experiment Duration (s)', 'tExp', False],
            ['Sampling Frequency (Hz)', 'sFreq', False],
        ]

        #
        for tt in expt_info:
            # sets the label value
            if tt[1] == 'name':
                # special case - experiment name
                lbl_str = cf.extract_file_name(c_data['expFile'])

            elif tt[2]:
                # case is the field is in the experiment information dictionary field
                if tt[1] not in c_data['expInfo']:
                    # case is the field is not in the experiment information dictionary
                    lbl_str = 'N/A'
                else:
                    # case is the field is in the experiment info dictionary
                    nw_val = eval('c_data["expInfo"]["{0}"]'.format(tt[1]))
                    if nw_val is None:
                        # case is the field is None
                        lbl_str = 'N/A'
                    else:
                        # otherwise, set the field value
                        lbl_str = '{0}'.format(nw_val)
            else:
                # case is the field is in the general information fields
                lbl_str = '{0}'.format(int(eval('c_data["{0}"]'.format(tt[1]))))

            # creates the label objects
            h_lbl = cf.create_label(None, txt_font_bold, '{0}: '.format(tt[0]), align='right')
            h_lbl_str = cf.create_label(None, txt_font, lbl_str, align='left')

            # adds the widgets to the layout
            h_layout.addRow(h_lbl, h_lbl_str)

        # sets the horizontal spacer
        h_layout.setHorizontalSpacing(250)

    def setup_cluster_info(self, i_expt):
        '''

        :param i_expt:
        :return:
        '''

        # retrieves the cluster data
        ff_dict, st_info = {}, {}
        c_data = self.data._cluster[i_expt]
        nC, is_fixed = c_data['nC'], c_data['expInfo']['cond'] == 'Fixed'
        has_free_data = hasattr(self.data.externd, 'free_data')
        f_name_nw = cf.extract_file_name(c_data['expFile'])

        # determines the indices that are excluded due to the general filter
        cl_inc = cfcn.get_inclusion_filt_indices(c_data, self.main_obj.data.exc_gen_filt)
        cl_exc = np.where(np.logical_xor(c_data['expInfo']['clInclude'], cl_inc))[0]

        # sets the experiment information fields
        cl_info = [
            ['Include?', 'special'],
            ['Cluster\nIndex', 'special'],
            ['Cluster\nID#', 'clustID'],
            ['Channel\nDepth', 'chDepth'],
            ['Channel\nDepth ({0}m)'.format(cf._mu), 'special'],
            ['Surface\nDepth ({0}m)'.format(cf._mu), 'special'],
            ['Channel\nRegion', 'chRegion'],
            ['Channel\nLayer', 'chLayer'],
            ['Spiking\nFrequency', 'special'],
            ['Matching\nCluster', 'special'],
            ['Spike\nClassification', 'special'],
            ['Action\nType', 'special'],
            ['Cell Type\n(5deg/s)', 'special'],
            ['AHV Pearson\n(Pos)', 'special'],
            ['AHV Pearson\n(Neg)', 'special'],
            ['AHV %Tile\n(Pos)', 'special'],
            ['AHV %Tile\n(Neg)', 'special'],
            ['Velocity\nPearson', 'special'],
            ['Mean Vec.\nLength', 'special'],
            ['AHV Pearson\n(-ve 1st Half)', 'special'],
            ['AHV Pearson\n(+ve 1st Half)', 'special'],
            ['AHV Pearson\n(-ve 2nd Half)', 'special'],
            ['AHV Pearson\n(+ve 2nd Half)', 'special'],
            ['AHV Null\nCorr %', 'special'],
            ['AHV Stability\nIndex', 'special'],
            ['Vel Pearson\n(1st Half)', 'special'],
            ['Vel Pearson\n(2nd Half)', 'special'],
            ['Vel Null\nCorr %', 'special'],
            ['Vel Stability\nIndex', 'special'],
            ['Theta\nIndex', 'special'],
            ['AHV Intercept\n(-ve)', 'special'],
            ['AHV Intercept\n(+ve)', 'special'],
            ['AHV Slope\n(-ve)', 'special'],
            ['AHV Slope\n(+ve)', 'special'],
            ['Velocity\nIntercept', 'special'],
            ['Velocity\nSlope', 'special'],
        ]

        # removes all parameters from the layout
        h_layout = self.h_info[i_expt, 1].layout()
        for i_row in range(h_layout.rowCount()):
            h_layout.removeRow(0)

        #
        if has_free_data:
            # retrieves the freely moving data class object
            f_data = self.data.externd.free_data

            # retrieves the free experiment index (which matches the current tab)
            i_expt_ff = f_data.exp_name.index(cf.det_closest_file_match(f_data.exp_name, f_name_nw)[0])

            # retrieves the cell information for the current experiment
            v_grp = 0                   # velocity group (0 for 5 deg/s, 1 for 10 deg/s)
            t_type = 'LIGHT1'           # trial type (default is "LIGHT1"
            c_info = f_data.c_info[i_expt_ff][v_grp][t_type]

            # sets the matching cell/external free data file cell indices (based on experiment type)
            if is_fixed:
                # case is a fixed experiment
                _, f2f_map = cf.det_matching_fix_free_cells(self.data,
                                                            exp_name=[f_data.exp_name[i_expt_ff]], apply_filter=True)

                i0_ff = np.where(f2f_map[0][:, 0] >= 0)[0]
                i_free_ff = f2f_map[0][i0_ff, 1]
            else:
                # case is a free experiment
                _, i0_ff, i_free_ff = np.intersect1d(c_data['clustID'], f_data.cell_id[i_expt_ff],
                                                            return_indices=True)

            # sets up the freely moving data table header to the column headers from the data files
            ff_dict = {
                'AHV Pearson\n(Pos)': 'ahv_pearson_r_pos',
                'AHV Pearson\n(Neg)': 'ahv_pearson_r_neg',
                'AHV %Tile\n(Pos)': 'pearson_pos_percentile',
                'AHV %Tile\n(Neg)': 'pearson_neg_percentile',
                'Velocity\nPearson': 'velocity_pearson_r',
                'Mean Vec.\nLength': 'mean_vec_length',
            }

            #
            st_dict = {
                'AHV Pearson\n(-ve 1st Half)': 'ahv_pearson_r_first_half_neg',
                'AHV Pearson\n(+ve 1st Half)': 'ahv_pearson_r_first_half_pos',
                'AHV Pearson\n(-ve 2nd Half)': 'ahv_pearson_r_second_half_neg',
                'AHV Pearson\n(+ve 2nd Half)': 'ahv_pearson_r_second_half_pos',
                'AHV Null\nCorr %': 'ahv_null_correlation_percentile',
                'AHV Stability\nIndex': 'ahv_stability_index',
                'Vel Pearson\n(1st Half)': 'velocity_pearson_r_first_half',
                'Vel Pearson\n(2nd Half)': 'velocity_pearson_r_second_half',
                'Vel Null\nCorr %': 'velocity_null_correlation_percentile',
                'Vel Stability\nIndex': 'velocity_stability_index',
                'AHV Intercept\n(-ve)': 'ahv_fit_intercept_neg',
                'AHV Intercept\n(+ve)': 'ahv_fit_intercept_pos',
                'AHV Slope\n(-ve)': 'ahv_fit_slope_neg',
                'AHV Slope\n(+ve)': 'ahv_fit_slope_pos',
                'Velocity\nIntercept': 'velocity_fit_intercept',
                'Velocity\nSlope': 'velocity_fit_slope',
            }

        # retrieves the channel map/depth values
        ch_map = c_data['expInfo']['channel_map']
        ch_depth = ch_map[c_data['chDepth'].astype(int), 3]

        # sets the data for each cell (over each metric type)
        t_data = np.empty((nC, len(cl_info)), dtype=object)
        for itt, tt in enumerate(cl_info):
            # sets the label value
            if tt[1] == 'special':
                if tt[0] == 'Include?':
                    nw_data = cl_inc

                elif tt[0] == 'Channel\nDepth ({0}m)'.format(cf._mu):
                    nw_data = ch_depth.astype(int).astype(str)

                elif tt[0] == 'Surface\nDepth ({0}m)'.format(cf._mu):
                    nw_data = np.array(c_data['expInfo']['probe_depth'] - ch_depth).astype(int).astype(str)

                elif tt[0] == 'Cluster\nIndex':
                    nw_data = (np.array(list(range(nC))) + 1).astype(str)

                elif tt[0] == 'Spiking\nFrequency':
                    nw_data = np.array(['{:5.3f}'.format(len(x) / c_data['tExp']) for x in c_data['tSpike']])

                elif tt[0] == 'Matching\nCluster':
                    if self.data.comp.is_set:
                        #
                        i_comp = cf.det_comp_dataset_index(self.data.comp.data, f_name_nw, is_fixed)
                        c_data_nw = self.data.comp.data[i_comp]

                        #
                        data_fix, data_free = cf.get_comp_datasets(self.data, c_data=c_data_nw, is_full=True)
                        i_fix = np.where(c_data_nw.is_accept)[0]
                        i_free = c_data_nw.i_match[c_data_nw.is_accept]

                        if is_fixed:
                            clustID = data_free['clustID']
                            i_ref, i_comp = i_fix, i_free
                        else:
                            clustID = data_fix['clustID']
                            i_ref, i_comp = i_free, i_fix

                        nw_data = np.array(['N/A'] * nC, dtype='U10')
                        nw_data[i_ref] = np.array([clustID[x] for x in i_comp])
                        nw_data[np.logical_not(cl_inc)] = '---'
                    else:
                        nw_data = np.array(['---'] * nC)

                elif tt[0] == 'Spike\nClassification':
                    if self.data.classify.class_set:
                        nw_data = np.array(['N/A'] * nC, dtype='U10')
                        if self.data.classify.grp_str[i_expt] is not None:
                            nw_data[cl_inc] = self.data.classify.grp_str[i_expt][cl_inc]
                            nw_data[np.logical_not(cl_inc)] = '---'
                    else:
                        nw_data = np.array(['---'] * nC)

                elif tt[0] == 'Action\nType':
                    if self.data.classify.action_set:
                        # memory allocation
                        nw_data = np.array(['N/A'] * nC, dtype='U10')
                        act_str = np.array(['---', 'Inhibitory', 'Excitatory'])

                        if self.data.classify.act_type[i_expt] is not None:
                            nw_data[cl_inc] = act_str[self.data.classify.act_type[i_expt][cl_inc]]
                            nw_data[np.logical_not(cl_inc)] = '---'
                    else:
                        nw_data = np.array(['---'] * nC)

                elif tt[0] == 'Theta\nIndex':
                    # memory allocation
                    nw_data = np.array(['---'] * nC)
                    if has_free_data:
                        if hasattr(self.data, 'theta_index'):
                            if self.data.theta_index.th_index is not None:
                                # retrieves the theta index information
                                i_expt_st = self.get_free_expt_index(f_data, c_data)
                                th_index = self.data.theta_index.th_index[i_expt_st][:, 0]

                                # sets the data values for the current column
                                for i_row, i_row_ff in zip(i0_ff, i_free_ff):
                                    nw_data[i_row] = '{:.4f}'.format(th_index[i_row_ff])

                elif 'Cell Type\n' in tt[0]:
                    if has_free_data:
                        # memory allocation
                        nw_data = np.array(['---'] * nC, dtype='U15')

                        # initialisations and memory allocation
                        c_name = np.array(['HD', 'HDMod', 'AHV', 'Spd'])
                        c_type = np.array(f_data.cell_type[i_expt_ff][int('10' in tt[0])])

                        # sets the column data
                        for i_row, i_row_ff in zip(i0_ff, i_free_ff):
                            # sets the free cell type
                            if not np.any(c_type[i_row_ff, :]):
                                nw_data[i_row] = 'None'
                            else:
                                nw_data[i_row] = '/'.join(c_name[c_type[i_row_ff, :]])

                elif (tt[0] in ff_dict) and has_free_data:
                    # memory allocation
                    nw_data = np.array(['---'] * nC, dtype='U15')
                    fr_data = c_info[ff_dict[tt[0]]]

                    # sets the column data
                    for i_row, i_row_ff in zip(i0_ff, i_free_ff):
                        nw_data[i_row] = '{:.4f}'.format(fr_data[i_row_ff])

                elif (tt[0] in st_dict) and has_free_data:
                    # memory allocation
                    nw_data = np.array(['---'] * nC, dtype='U15')
                    if hasattr(f_data, 'stability_info'):
                        # retrieves the stability information
                        i_expt_st = self.get_free_expt_index(f_data, c_data)
                        st_info = f_data.stability_info[i_expt_st]

                        if len(st_info):
                            # if there is data for the experiment, then add it to the table
                            st_data = st_info[st_dict[tt[0]]]
                            for i_row, i_row_ff in zip(i0_ff, i_free_ff):
                                nw_data[i_row] = '{:.4f}'.format(st_data[i_row_ff])

                else:
                    nw_data = np.array(['---'] * nC)

            else:
                nw_data = np.array(eval('c_data["{0}"]'.format(tt[1]))).astype(str)

            # appends the new data to the table data array
            t_data[:, itt] = nw_data

        # creates the label objects
        col_hdr = [tt[0] for tt in cl_info]
        h_table = cf.create_table(None, txt_font, data=t_data, col_hdr=col_hdr, n_row=nC, max_disprows=20,
                                  check_col=[0], check_fcn=self.includeCheck, exc_rows=cl_exc)
        h_table.verticalHeader().setVisible(False)
        # h_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        h_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # sets the table dimensions
        c_wid = 100
        nrow_table = min(20, nC)
        h0 = (57 - int(platform.system() == 'Windows')) - nrow_table
        row_hght = 22 if platform.system() == 'Windows' else 23

        # resets the table dimensions
        cf.set_obj_fixed_size(h_table, height=h0 + nrow_table * row_hght, width=self.grp_wid_info - 2*dX)

        # resets the column headers
        h_table.horizontalHeader().setSectionResizeMode(2)
        for i in range(h_table.columnCount()):
            h_table.setColumnWidth(i, c_wid)

        # adds the widgets to the layout
        h_layout.addRow(h_table)

    def get_free_expt_index(self, f_data, c_data):

        # retrieves the experiment file name
        exp_file = cf.extract_file_name(c_data['expFile'])
        if c_data['rotInfo'] is not None:
            # if a fixed experiment, then retrieve the corresponding free data file
            comp_data = self.data.comp.data
            exp_file = comp_data[[x.fix_name for x in comp_data].index(exp_file)].free_name

        # determines the index of the free experiment matching the free cluster file name
        return f_data.exp_name.index(cf.det_closest_file_match(f_data.exp_name, exp_file)[0])

    def includeCheck(self, i_row, i_col, state):
        '''

        :return:
        '''

        # retrieves the base inclusion indices
        i_expt = self.h_grpbx[0].currentIndex()
        cl_inc = self.data._cluster[i_expt]['expInfo']['clInclude']

        # determines the indices that are excluded due to the general filter
        cl_inc_full = cfcn.get_inclusion_filt_indices(self.data._cluster[i_expt], self.main_obj.data.exc_gen_filt)
        cl_inc_xor = np.logical_xor(cl_inc, cl_inc_full)

        # determines if the selected row is part of the general exclusion filter indices
        if cl_inc_xor[i_row]:
            # if so, then output a message and revert back to the original state
            a = 1
        else:
            # flag that an update is required and updates the inclusion flag for the cell
            self.main_obj.data.req_update = True
            cl_inc[i_row] = state > 0

        # force a calculation update
        self.main_obj.data.force_calc = True

    def refresh_fields(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        pass

    def output_info(self):
        '''

        :return:
        '''

        # memory allocation
        exp_type = ['Free', 'Fixed']
        t_data = [[] for _ in range(2)]
        is_rot = cf.det_valid_rotation_expt(self.data)
        exp_name = [cf.extract_file_name(x['expFile']) for x in self.data._cluster]

        # retrieves the table for each experiment (splitting the data into fixed/free experiments)
        for i_expt in range(self.n_expt):
            # retrieves the current table data
            t_data_nw = self.get_table_data(i_expt)
            if np.shape(t_data_nw)[0] <= 2:
                # if no data, then exit
                continue
            else:
                t_data_nw[2, 0] = exp_name[i_expt]

            # appends the data to the storage lists
            i = int(is_rot[i_expt])
            if len(t_data[i]):
                t_data[i].append(t_data_nw[2:, :])
            else:
                t_data[i].append(t_data_nw)

        # outputs the experiments based on the type (i.e., fixed or free)
        for i_t, ex_t in enumerate(exp_type):
            # outputs the data file (only if there were any experiments of that type)
            if len(t_data[i_t]):
                # determines if the column counts match across data sets
                n_col = np.array([np.shape(x)[1] for x in t_data[i_t]])
                n_col_mx = np.max(n_col)

                # if there are missing columns, then add them in so all column counts are equal
                miss_col = n_col != n_col_mx
                if np.any(miss_col):
                    for i_col in np.where(miss_col)[0]:
                        # creates the missing array
                        t_data_add = np.empty((np.shape(t_data[i_t][i_col])[0], n_col_mx - n_col[i_col]), dtype=object)
                        t_data_add[:] = '---'

                        # appends the data to the array
                        t_data[i_t][i_col] = np.hstack((t_data[i_t][i_col], t_data_add))

                # outputs the data to file
                f_name = 'Experiment Info ({0} Expts).csv'.format(ex_t)
                self.main_obj.output_data_file(f_name, np.vstack(t_data[i_t]))

    def close_window(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.can_close = True
        self.close()

    def closeEvent(self, evnt):

        if self.can_close:
            super(InfoDialog, self).closeEvent(evnt)
        else:
            evnt.ignore()

    def get_table_data(self, i_expt):
        '''

        :param i_expt:
        :return:
        '''

        # retrieves the table handle
        h_table = self.h_info[i_expt, 1].findChildren(QTableWidget)[0]

        # memory allocation
        n_row, n_col, i_ofs = h_table.rowCount(), h_table.columnCount(), 1
        is_inc = np.ones(n_row + (i_ofs + 1), dtype=bool)
        t_data = np.empty((n_row + (i_ofs + 1), n_col), dtype=object)

        # sets the table header
        for i_col in range(1, n_col):
            t_data[i_ofs, i_col] = h_table.horizontalHeaderItem(i_col).text().replace('\n', ' ')
            if '\u03bc' in t_data[i_ofs, i_col]:
                t_data[i_ofs, i_col] = t_data[i_ofs, i_col].replace('\u03bc', 'u')

        # loops through each of the rows/column retrieving the table data
        for i_col in range(n_col):
            for i_row in range(n_row):
                if i_col == 0:
                    # retrieves the inclusion value
                    is_inc[i_row + (i_ofs + 1)] = h_table.cellWidget(i_row, 0).isEnabled()
                else:
                    # retrieves the data from the table
                    t_data[i_row + (i_ofs + 1), i_col] = h_table.item(i_row, i_col).text()

        # removes any None values
        t_data[t_data == None] = ''

        # returns the table data
        return t_data[is_inc, :][:, ~np.all(t_data[(1 + i_ofs):, :] == '---', axis=0)]

########################################################################################################################
########################################################################################################################

class ParaFieldDialog(QDialog):
    def __init__(self, main_obj, parent=None, title=None, chk_flds=None, fld_vals=None, f_name=None, cl_ind=None):
        # creates the gui object
        super(ParaFieldDialog, self).__init__(parent)

        # field initialisations
        self.is_ok = True
        self.is_init = False
        self.is_updating = True
        self.can_close = False

        # sets the object fields
        self.main_obj = main_obj
        self.chk_flds = chk_flds
        self.fld_vals0 = dcopy(fld_vals)
        self.f_name = f_name

        # sets the cluster indices/field values (depending on whether parameters are missing or being altered)
        if cl_ind is None:
            # case is the parameters are being altered
            self.cl_ind = np.arange(len(f_name))
            self.fld_vals = dcopy(fld_vals)
        else:
            # case is the parameters are missing
            self.cl_ind = cl_ind
            self.fld_vals = cfcn.det_missing_data_fields(fld_vals, f_name, chk_flds)

        # creates the GUI objects
        self.init_data_table()
        self.create_control_buttons()
        self.setLayout(self.mainLayout)

        # resizes the sub-gui width
        n_fld = np.size(fld_vals, axis=1)
        cf.set_obj_fixed_size(self, width=350 + 100 * n_fld)

        # sets the final window properties
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.is_init = True

        # shows and executes the dialog box
        self.show()
        self.exec()

    def init_data_table(self):
        '''

        :return:
        '''

        # sets the widgets into the box layout
        self.mainLayout = QGridLayout()

        # group box object
        self.p_table = QGroupBox("")
        layout = QHBoxLayout()

        # parameters and initialisations
        n_exp, n_fld = np.shape(self.fld_vals)
        t_data = np.empty((n_exp, n_fld + 2), dtype=object)
        fld_key = {
            'probe_depth': 'Probe Depth (um)'
        }

        # sets the column names
        col_hdr = ['Expt #', 'Expt Name'] + [fld_key[c_flds] for c_flds in self.chk_flds]

        # sets the table data into a single array
        t_data[:, 2:] = self.fld_vals
        for i_exp in range(n_exp):
            t_data[i_exp, 0], t_data[i_exp, 1] = str(i_exp + 1), self.f_name[i_exp]

            for i_fld in range(n_fld):
                if self.fld_vals[i_exp, i_fld] is None:
                    t_data[i_exp, i_fld + 2] = ''
                else:
                    t_data[i_exp, i_fld + 2] = str(self.fld_vals[i_exp, i_fld])

        # creates the label objects
        self.h_table = cf.create_table(None, txt_font, data=t_data, col_hdr=col_hdr, n_row=n_exp,
                                       max_disprows=min(10, n_exp + 1), cb_fcn=self.edit_table_cell)
        self.h_table.verticalHeader().setVisible(False)

        h_hdr = self.h_table.horizontalHeader()
        h_hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h_hdr.setSectionResizeMode(1, QHeaderView.Stretch)

        # sets the column widths
        for i_main in range(2):
            for i_exp in range(n_exp):
                c_item = self.h_table.item(i_exp, i_main)
                c_item.setFlags(Qt.ItemIsEnabled)

        for i_fld in range(n_fld):
            h_hdr.setSectionResizeMode(i_fld + 2, QHeaderView.ResizeToContents)

        # adds the widgets to the layout
        layout.addWidget(self.h_table)
        self.p_table.setLayout(layout)
        self.mainLayout.addWidget(self.p_table, 0, 0)

    def create_control_buttons(self):
        '''

        :return:
        '''

        # initialisations
        b_txt = ['Update Changes', 'Cancel']
        cb_fcn = [self.user_continue, self.user_cancel]
        b_name = ['user_continue', 'user_cancel']

        # group box object
        self.p_but = QGroupBox("")
        layout = QHBoxLayout()

        # creates the load config file object
        for i in range(len(b_txt)):
            # creates the button object
            hButton = QPushButton(b_txt[i])
            hButton.clicked.connect(cb_fcn[i])
            hButton.setObjectName(b_name[i])
            hButton.setAutoDefault(False)
            cf.update_obj_font(hButton, pointSize=9)

            # adds the objects to the layout
            layout.addWidget(hButton)

        # sets the box layout
        self.p_but.setLayout(layout)
        self.mainLayout.addWidget(self.p_but, 1, 0)

        # sets the button enabled properties
        self.set_button_enabled_props()

    def set_button_enabled_props(self):
        '''

        :return:
        '''

        # disables the continue button if not all values are valid
        any_missing = np.any(np.any(self.fld_vals == None, axis=1))
        self.p_but.findChild(QPushButton, 'user_continue').setEnabled(not any_missing)

    def edit_table_cell(self, i_row, i_col):
        '''

        :return:
        '''

        if not self.is_init:
            # if initialising, then exit the function
            return

        # retrieves the current cell and its contents
        h_cell = self.h_table.item(i_row, i_col)
        nw_str = h_cell.text()

        if len(nw_str):
            # if the user enetered the string, determine if it is valid
            nw_val, e_str = cf.check_edit_num(nw_str, True, min_val=0)
            if e_str is None:
                # if so, then update the
                self.fld_vals[i_row, i_col - 2] = nw_val
            else:
                # otherwise, set the cell back to its previous value
                if self.fld_vals[i_row, i_col - 2] is None:
                    # there was no valid number, so keep the cell empty
                    h_cell.setText('')
                else:
                    # otherwise, use the previous valid value
                    h_cell.setText(str(self.fld_vals[i_row, i_col - 2]))
        else:
            # if the user entered nothing, then clear the field value
            self.fld_vals[i_row, i_col - 2] = None

        # resets the button enabled properties
        self.set_button_enabled_props()

    def user_cancel(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.can_close = True
        self.close()

    def user_continue(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.can_close = True
        self.close()

        # determines which values have been altered from their original values
        is_change = self.fld_vals != self.fld_vals0
        if np.any(is_change[:]):
            # if so, then prompt the user if they really want to update the files
            u_choice = QMessageBox.question(self, 'Update Experiment Parameters?',
                                            "Are you sure you want to update the experimental file parameters?",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if u_choice == QMessageBox.Yes:
                # if the user chose yes, then update the experiment files
                self.update_expt_files(is_change)
            else:
                # otherwise, flag that the
                self.is_ok = False

    def update_expt_files(self, is_change):
        '''

        :param is_change:
        :return:
        '''

        # determines if any change has occured for any experiment (over all parameters)
        is_multi = self.main_obj.is_multi
        any_change = np.where(np.any(is_change, axis=1))[0]
        n_exp = len(any_change)

        # sets the cluster indices, and the change/field values
        cl_ind, f_name = self.cl_ind[any_change], self.f_name[any_change]
        is_change, fld_vals = is_change[any_change, :], dcopy(self.fld_vals)[any_change, :]

        #
        for i, i_cl in enumerate(cl_ind):
            # retrieves the clusters (original and data - if available)
            _c = self.main_obj.data._cluster[i_cl]
            c = self.main_obj.data.cluster[i] if (self.main_obj.data.cluster is not None) else None

            # loops through each of the parameter fields updating values where necessary
            for i_cfld, c_fld in enumerate(self.chk_flds):
                # only update the parameters if a change has occurred
                if is_change[i, i_cfld]:
                    if c_fld == 'probe_depth':
                        # case is the probe depth

                        # updates the cluster field values
                        _c['expInfo'][c_fld] = fld_vals[i, i_cfld]
                        if c is not None:
                            c['expInfo'][c_fld] = fld_vals[i, i_cfld]

            # re-saves the cluster file with the new parameter values
            if os.path.exists(_c['expFile']):
                # updates the progressbar
                p_str = 'Updating File Parameters (File #{0} of {1})'.format(i + 1, n_exp)
                self.main_obj.update_thread_job(p_str, 100. * (i + 1) / n_exp)
                time.sleep(0.5)

                # outputs the new data to file
                cf.save_single_file(_c['expFile'], _c)

        #
        if is_multi:
            # sets the file path/name strings
            path, filename = os.path.split(self.main_obj.data.multi.names[0])

            # outputs the data to file based on the type
            if '.mdata' in self.main_obj.data.multi.names[0]:
                # case is a multi-experiment data file
                out_info = {'inputDir': path, 'dataName': filename.replace('.mdata', '')}
                cf.save_multi_data_file(self.main_obj, out_info, True)
            else:
                # case is a multi-experiment comparison data file
                out_info = {'inputDir': path, 'dataName': filename.replace('.mcomp', '')}
                cf.save_multi_data_file(self.main_obj, out_info, True)
        else:
            # otherwise, reset the main GUI progressbar
            self.main_obj.update_thread_job('Waiting For Process...', 0)

    def closeEvent(self, evnt):

        if self.can_close:
            super(ParaFieldDialog, self).closeEvent(evnt)
        else:
            evnt.ignore()

    def user_cancel(self):
        '''

        :return:
        '''

        # resets the close flag and closes the GUI
        self.is_ok = False
        self.can_close = True
        self.close()

    def get_info(self):
        '''

        :return:
        '''

        if not self.is_ok:
            # user cancelled
            return None, None
        else:
            # determines which values have been altered from their original values
            is_change = np.any(self.fld_vals != self.fld_vals0, axis=1)

            # returns the field values and the indices of the groups that changed
            return self.fld_vals, np.where(is_change)[0]
