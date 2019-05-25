# module import
import os
import re
import sys
import copy
import functools
import math as m
import numpy as np
from numpy import ndarray
from skimage import measure

# pyqt5 module import
from PyQt5.QtGui import QFont, QFontMetrics, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QPushButton, QListWidget, QComboBox, QMenuBar, QProgressBar, QHeaderView,
                             QMenu, QAction, QLabel, QWidget, QLineEdit, QCheckBox, QMessageBox, QTableWidget,
                             QTabWidget, QTableWidgetItem, QHBoxLayout)

#
from scipy.optimize import curve_fit

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import FloatVector, BoolVector, StrVector, IntVector
from rpy2.robjects.packages import importr
from rpy2.robjects.functions import SignatureTranslatedFunction
rpy2.robjects.numpy2ri.activate()
r_stats = importr("stats")
r_pROC = importr("pROC")

#
_roc = r_pROC.roc
_ci_auc = SignatureTranslatedFunction(r_pROC.ci_auc,
                                      init_prm_translate = {'boot_n': 'boot.n', 'conf_level': 'conf.level'})
_roc_test = SignatureTranslatedFunction(r_pROC.roc_test,
                                        init_prm_translate = {'boot_n': 'boot.n'})

# lambda function declarations
lin_func = lambda x, a: a * x
spike_count_fcn = lambda t_sp: np.array([len(x) for x in t_sp])
swap_array = lambda x1, x2, is_swap: np.array([x if is_sw else y for x, y, is_sw in zip(x1, x2, is_swap)])
combine_spike_freq = lambda sp_freq, i_dim: flat_list([list(sp_freq[i_filt][:, i_dim]) for i_filt in range(len(sp_freq))])
calc_rel_prop = lambda x, n: 100 * np.array([sum(x == i) for i in range(n)]) / len(x)
calc_rel_count = lambda x, n: np.array([sum(x == i) for i in range(n)])

# vectorisation function declarations
sp_freq = lambda x, t_phase: len(x) / t_phase if x is not None else 0
sp_freq_fcn = np.vectorize(sp_freq)

# unicode characters
_bullet_point = '\u2022'
_mu = '\u03bc'
_delta = '\u0394'

# other initialisations
t_wid_f = 0.99
dcopy = copy.deepcopy
is_linux = sys.platform == 'linux'


def flat_list(l):
    '''

    :param l:
    :return:
    '''

    #
    if len(l) == 0:
        return []
    elif isinstance(l[0], list) or isinstance(l[0], ndarray):
        return [item for sublist in l for item in sublist]
    else:
        return l

class CheckableComboBox(QComboBox):
    def __init__(self, parent=None, has_all=False, first_line=None):
        super(CheckableComboBox, self).__init__(parent)
        self.view().pressed.connect(self.handleItemPressed)
        self.n_item = 0
        self.has_all = has_all
        self.first_line = first_line

    def addItem(self, item, can_check):
        '''

        :param item:
        :param can_check:
        :return:
        '''

        super(CheckableComboBox, self).addItem(item)
        item = self.model().item(self.count()-1,0)
        self.n_item += 1

        if can_check:
            # item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setFlags(Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
        else:
            item.setFlags(Qt.NoItemFlags)

    # def itemChecked(self, index):
    #     item = self.model().item(index, 0)
    #     return item.checkState() == Qt.Checked

    def getSelectedItems(self):
        '''

        :return:
        '''

        # initialisations
        txt_sel = []

        # retrieves the checkbox text for selected items
        for i in range(self.count()):
            item = self.model().item(i)
            if item.checkState() == Qt.Checked:
                txt_sel.append(item.text())

        # returns the selected text
        return txt_sel

    def setState(self, index, state):
        '''

        :param index:
        :param state:
        :return:
        '''

        # retrieves the item corresponding to the current index
        if (index == 0) and (self.first_line is not None):
            return

        item = self.model().item(index)
        item.setCheckState(Qt.Checked if state else Qt.Unchecked)

    def handleItemPressed(self, index, is_checked=None):
        '''

        :param index:
        :param is_checked:
        :return:
        '''

        #
        if (index == 0) and (self.first_line is not None):
            return

        #
        if isinstance(index, int):
            item, i_sel = self.model().item(index), index
        else:
            item = self.model().itemFromIndex(index)
            i_sel = item.row()

        #
        if is_checked is None:
            is_checked = (item.checkState() == Qt.Checked)
            item.setCheckState(Qt.Unchecked if is_checked else Qt.Checked)

        #
        if (i_sel == 1) and self.has_all:
            for i_item in range(2, self.n_item):
                item_new = self.model().item(i_item)
                if is_checked:
                    item_new.setFlags(Qt.ItemIsEnabled)
                else:
                    item_new.setCheckState(Qt.Unchecked)
                    item_new.setFlags(Qt.NoItemFlags)

#########################################
####    OBJECT PROPERTY FUNCTIONS    ####
#########################################

def create_font_obj(size=8, is_bold=False, font_weight=QFont.Normal):
    '''

    :param is_bold:
    :param font_weight:
    :return:
    '''

    # creates the font object
    font = QFont()

    # sets the font properties
    font.setPointSize(size)
    font.setBold(is_bold)
    font.setWeight(font_weight)

    # returns the font object
    return font


def update_obj_font(h_obj, pointSize=8, weight=QFont.Normal):
    '''

    :param hObj:
    :param pointSize:
    :param weight:
    :return:
    '''

    mainFont = h_obj.font().family()
    qF = QFont(mainFont, pointSize=pointSize, weight=weight)
    h_obj.setFont(qF)


def set_obj_fixed_size(h_obj, width=None, height=None, fix_size=True):
    '''

    '''

    # retrieves the suggested object object size
    obj_sz = h_obj.sizeHint()

    if width is None:
        width = obj_sz.width()

    if height is None:
        height = obj_sz.height()

    # resets the object size
    if fix_size:
        h_obj.setFixedSize(width, height)
    else:
        h_obj.resize(width, height)


def set_text_colour(text, col='black'):
    '''

    :param text:
    :param col:
    :return:
    '''

    return '<span style="color:{0}">{1}</span>'.format(col, text)

#####################################################
####    PYQT5 OBJECT INITIALISATION FUNCTIONS    ####
#####################################################

def create_groupbox(parent, dim, font, title, name=None):
    '''

    :param parent:
    :param dim:
    :param font:
    :param name:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the groupbox object
    h_group = QGroupBox(parent)

    # sets the object properties
    h_group.setGeometry(dim)
    h_group.setFont(font)
    h_group.setTitle(title)

    # sets the object name (if provided)
    if name is not None:
        h_group.setObjectName(name)

    # returns the group object
    return h_group


def create_label(parent, font, text, dim=None, name=None, align='left'):
    '''

    :param parent:
    :param dim:
    :param font:
    :param text:
    :param name:
    :param align:
    :return:
    '''

    # creates the label object
    h_lbl = QLabel(parent)

    # sets the label properties
    h_lbl.setFont(font)
    h_lbl.setText(text)

    # set the object dimensions (if not None)
    if dim is not None:
        h_lbl.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_lbl.setObjectName(name)

    # sets the horizontal alignment of the label
    if align == 'centre':
        h_lbl.setAlignment(Qt.AlignCenter)
    elif align == 'left':
        h_lbl.setAlignment(Qt.AlignLeft)
    else:
        h_lbl.setAlignment(Qt.AlignRight)

    # returns the label object
    return h_lbl


def create_edit(parent, font, text, dim=None, name=None, cb_fcn=None, align='centre'):
    '''

    :param font:
    :param text:
    :param dim:
    :param parent:
    :param name:
    :param cb_fcn:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the editbox object
    h_edit = QLineEdit(parent)

    # sets the label properties
    h_edit.setFont(font)
    h_edit.setText(text)

    # sets the object name (if provided)
    if name is not None:
        h_edit.setObjectName(name)

    # set the object dimensions (if not None)
    if dim is not None:
        h_edit.setGeometry(dim)

    # sets the horizontal alignment of the label
    if align == 'centre':
        h_edit.setAlignment(Qt.AlignCenter)
    elif align == 'left':
        h_edit.setAlignment(Qt.AlignLeft)
    else:
        h_edit.setAlignment(Qt.AlignRight)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_edit.editingFinished.connect(cb_fcn)

    # returns the object
    return h_edit


def create_button(parent, dim, font, text, name=None, icon=None, tooltip=None, cb_fcn=None):
    '''

    :param dim:
    :param font:
    :param name:
    :param icon:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the button object
    h_button = QPushButton(parent)

    # sets the button properties
    h_button.setFont(font)
    h_button.setText(text)

    #
    if dim is not None:
        h_button.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_button.setObjectName(name)

    # sets the icon (if provided)
    if icon is not None:
        h_button.setIcon(icon)

    # sets the tooltip string (if provided)
    if tooltip is not None:
        h_button.setToolTip(tooltip)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_button.clicked.connect(cb_fcn)

    # returns the button object
    return h_button


def create_checkbox(parent, font, text, dim=None, name=None, state=False, cb_fcn=None):
    '''

    :param parent:
    :param dim:
    :param font:
    :param text:
    :param name:
    :param state:
    :param cb_fcn:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_chk = QCheckBox(parent)

    #
    h_chk.setText(text)
    h_chk.setFont(font)
    h_chk.setChecked(state)

    # set the object dimensions (if not None)
    if dim is not None:
        h_chk.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_chk.setObjectName(name)

    # sets the callback function
    if cb_fcn is not None:
        h_chk.stateChanged.connect(cb_fcn)

    # returns the checkbox object
    return h_chk


def create_listbox(parent, dim, font, text, name=None, cb_fcn=None):
    '''

    :param parent:
    :param dim:
    :param text:
    :param name:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_list = QListWidget(parent)

    # sets the listbox object properties
    h_list.setGeometry(dim)
    h_list.setFont(font)

    # sets the object name (if provided)
    if name is not None:
        h_list.setObjectName(name)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_list.itemSelectionChanged.connect(cb_fcn)

    # sets the listbox text (if provided)
    if text is not None:
        for t in text:
            h_list.addItem(t)

    # returns the listbox object
    return h_list


def create_progressbar(parent, dim, font, text=None, init_val=0, name=None, max_val=100.0):
    '''

    :param parent:
    :param font:
    :param text:
    :param dim:
    :param init_val:
    :param name:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_pbar = QProgressBar(parent)

    # sets the listbox object properties
    h_pbar.setGeometry(dim)
    h_pbar.setFont(font)
    h_pbar.setValue(init_val)
    h_pbar.setMaximum(max_val)

    # sets the object name (if provided)
    if name is not None:
        h_pbar.setObjectName(name)

    # removes the text if not provided
    if text is None:
        h_pbar.setTextVisible(False)

    # returns the progressbar object
    return h_pbar


def create_combobox(parent, font, text, dim=None, name=None, cb_fcn=None):
    '''

    :param parent:
    :param dim:
    :param font:
    :param list:
    :param name:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_combo = QComboBox(parent)

    # sets the combobox object properties
    h_combo.setFont(font)

    # sets the object dimensions (if provided)
    if dim is not None:
        h_combo.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_combo.setObjectName(name)

    # sets the combobox text (if provided)
    if text is not None:
        for t in text:
            h_combo.addItem(t)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_combo.currentIndexChanged.connect(cb_fcn)

    # returns the listbox object
    return h_combo

def create_checkcombo(parent, font, text, dim=None, name=None, cb_fcn=None,
                      first_line='--- Select From Options List Below ---', has_all=False):
    '''

    :param parent:
    :param font:
    :param combo_opt:
    :param dim:
    :param name:
    :param cb_fcn:
    :param combo_fcn:
    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the listbox object
    h_chkcombo = CheckableComboBox(parent, has_all, first_line)

    # sets the combobox object properties
    h_chkcombo.setFont(font)

    # sets the object dimensions (if provided)
    if dim is not None:
        h_chkcombo.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_chkcombo.setObjectName(name)

    # sets the combobox text (if provided)
    if text is not None:
        if first_line is not None:
            text = [first_line] + text

        for i, t in enumerate(text):
            h_chkcombo.addItem(t, i>0)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_chkcombo.view().pressed.connect(cb_fcn)

    # returns the listbox object
    return h_chkcombo

def create_table(parent, font, data=None, col_hdr=None, row_hdr=None, n_row=None, dim=None, name=None,
                 cb_fcn=None, combo_fcn=None, max_disprows=3, check_col=None, check_fcn=None, exc_rows=None):
    '''

    :param parent:
    :param font:
    :param col_hdr:
    :param row_hdr:
    :param n_row:
    :param dim:
    :param name:
    :param cb_fcn:
    :param combo_fcn:
    :return:
    '''

    #
    n_col = len(col_hdr)
    if n_row is None:
        n_row = 3

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the table object
    h_table = QTableWidget(parent)

    # sets the object properties
    h_table.setRowCount(n_row)
    h_table.setColumnCount(n_col)
    h_table.setFont(font)

    if col_hdr is not None:
        h_table.setHorizontalHeaderLabels(col_hdr)

    if row_hdr is not None:
        h_table.setVerticalHeaderLabels(row_hdr)

    # sets the object dimensions (if provided)
    if dim is not None:
        h_table.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_table.setObjectName(name)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_table.cellChanged.connect(cb_fcn)

    # sets the table dimensions
    h_table.setMaximumHeight(20 + min(max_disprows, n_row) * 22)
    h_table.resizeRowsToContents()

    # sets the table headers
    h_hdr = h_table.horizontalHeader()
    for i_col in range(len(col_hdr)):
        h_hdr.setSectionResizeMode(i_col, QHeaderView.Stretch)

    #
    if data is not None:
        for i_row in range(n_row):
            for i_col in range(n_col):
                if check_col is not None:
                    if i_col in check_col:
                        # creates the checkbox widget
                        h_chk = QCheckBox()
                        h_chk.setCheckState(Qt.Checked if data[i_row, i_col] else Qt.Unchecked)

                        if check_fcn is not None:
                            check_fcn_full = functools.partial(check_fcn, i_row, i_col)
                            h_chk.stateChanged.connect(check_fcn_full)

                        # creates the widget object
                        h_cell = QWidget()
                        h_layout = QHBoxLayout(h_cell)
                        h_layout.addWidget(h_chk)
                        h_layout.setAlignment(Qt.AlignCenter)
                        h_layout.setContentsMargins(0, 0, 0, 0)
                        h_cell.setLayout(h_layout)

                        # if the row is excluded
                        if exc_rows is not None:
                            if i_row in exc_rows:
                                item = QTableWidgetItem('')
                                item.setBackground(QColor(200, 200, 200))
                                h_table.setItem(i_row, i_col, item)

                                h_cell.setEnabled(False)

                        # continues to the next column
                        h_table.setCellWidget(i_row, i_col, h_cell)
                        continue

                # retrieves the current cell object and determines if is a combobox object
                item = QTableWidgetItem(data[i_row, i_col])

                # resets the background colour (if the row is excluded)
                if exc_rows is not None:
                    if i_row in exc_rows:
                        item.setBackground(QColor(200, 200, 200))

                # adds the item to the table
                item.setTextAlignment(Qt.AlignHCenter)
                h_table.setItem(i_row, i_col, item)

                # # if the column is checkable, then modify the cell item properties
                # if check_col is not None:
                #     if i_col in check_col:
                #         item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                #         item.setCheckState(Qt.Checked if data[i_row, i_col] else Qt.Unchecked)

    # returns the table object
    return h_table


def create_tablecombo(parent, font, combo_opt, col_hdr=None, row_hdr=None, n_row=None,
                      dim=None, name=None, cb_fcn=None, combo_fcn=None):
    '''

    :param parent:
    :param font:
    :param col_hdr:
    :param combo_opt:
    :param dim:
    :param name:
    :param cb_fcb:
    :return:
    '''

    #
    if n_row is None:
        n_row = 3

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the table object
    h_table = QTableWidget(parent)

    # sets the object properties
    h_table.setRowCount(n_row)
    h_table.setColumnCount(len(col_hdr))
    h_table.setFont(font)

    # sets the
    for opt_col in combo_opt:
        for i_row in range(n_row):
            # sets the combobox callback function (if provided)
            if combo_fcn is None:
                cb_fcn_combo = None
            else:
                cb_fcn_combo = functools.partial(combo_fcn[0], combo_fcn[1], h_table, i_row, opt_col)

            # creates the combo-box object
            h_combocell = create_combobox(h_table, font, combo_opt[opt_col], cb_fcn=cb_fcn_combo)

            # creates the combobox object and fills in the options
            h_table.setCellWidget(i_row, opt_col, h_combocell)

    if col_hdr is not None:
        h_table.setHorizontalHeaderLabels(col_hdr)

    if row_hdr is not None:
        h_table.setVerticalHeaderLabels(row_hdr)

    # sets the object dimensions (if provided)
    if dim is not None:
        h_table.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_table.setObjectName(name)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_table.cellChanged.connect(cb_fcn)

    #
    h_table.setMaximumHeight(20 + min(3, n_row) * 22)
    h_table.resizeRowsToContents()

    h_hdr = h_table.horizontalHeader()
    for i_col in range(len(col_hdr)):
        h_hdr.setSectionResizeMode(i_col, QHeaderView.Stretch)

    # returns the table object
    return h_table


def create_tab(parent, dim, font, h_tabchild=None, child_name=None, name=None, cb_fcn=None):
    '''

    :return:
    '''

    # creates a default font object (if not provided)
    if font is None:
        font = create_font_obj()

    # creates the tab object
    h_tab = QTabWidget(parent)

    # sets the listbox object properties
    h_tab.setGeometry(dim)
    h_tab.setFont(font)

    # adds any children widgets (if provided)
    if (h_tabchild is not None) and (child_name is not None):
        for h_tc, c_n in zip(h_tabchild, child_name):
            h_tab.addTab(h_tc, c_n)

    # sets the object name (if provided)
    if name is not None:
        h_tab.setObjectName(name)

    # sets the tab changed callback function (if provided)
    if cb_fcn is not None:
        h_tab.currentChanged.connect(cb_fcn)

    # returns the tab object
    return h_tab


def create_menubar(parent, dim, name=None):
    '''

    :param parent:
    :param dim:
    :param name:
    :return:
    '''

    # creates the menubar object
    h_menubar = QMenuBar(parent)

    # sets the menubar properties
    h_menubar.setGeometry(dim)

    # sets the object name (if provided)
    if name is not None:
        h_menubar.setObjectName(name)

    # returns the menubar object
    return h_menubar


def create_menu(parent, title, name=None):
    '''

    :param parent:
    :param title:
    :param name:
    :return:
    '''

    # creates the menu item
    h_menu = QMenu(parent)

    # sets the menu properties
    h_menu.setTitle(title)

    # sets the object name (if provided)
    if name is not None:
        h_menu.setObjectName(name)

    # returns the menu object
    return h_menu


def create_menuitem(parent, text, name=None, cb_fcn=None, s_cut=None):
    '''

    :param parent:
    :param title:
    :param name:
    :return:
    '''

    # creates the menu item object
    h_menuitem = QAction(parent)

    # sets the menuitem properties
    h_menuitem.setText(text)

    # sets the object name (if provided)
    if name is not None:
        h_menuitem.setObjectName(name)

    # sets the callback function (if provided)
    if cb_fcn is not None:
        h_menuitem.triggered.connect(cb_fcn)

    # sets the callback function (if provided)
    if s_cut is not None:
        h_menuitem.setShortcut(s_cut)

    # returns the menu item object
    return h_menuitem


def delete_widget_children(h_grp):
    '''

    :param h_grp:
    :return:
    '''

    # deletes all widgets that are children to the groupbox object
    for hh in h_grp.findChildren(QWidget):
        hh.deleteLater()


#######################################
####    MISCELLANEOUS FUNCTIONS    ####
#######################################


def set_file_name(f_name, f_type):
    '''

    :param f_name:
    :param f_type:
    :return:
    '''

    f_type_ex = re.search('\(([^)]+)', f_type).group(1)
    if f_type_ex[1:] not in f_name:
        return '{0}.{1}'.format(f_name, f_type_ex[1:])
    else:
        return f_name


def check_edit_num(nw_str, is_int=False, min_val=-1e100, max_val=1e10, show_err=True):
    '''

    :param nw_str:
    :param is_int:
    :return:
    '''

    # initialisations
    nw_val, e_str = None, None

    if is_int:
        # case is the string must be a float
        try:
            nw_val = int(nw_str)
        except:
            try:
                # if there was an error, then determine if the string was a float
                nw_val = float(nw_str)
                if nw_val % 1 == 0:
                    # if the float is actually an integer, then return the value
                    nw_val, e_str = int(nw_val), 1
                else:
                    # otherwise,
                    e_str = 'Entered value is not an integer.'
            except:
                # case is the string was not a valid number
                e_str = 'Entered value is not a valid number.'
    else:
        # case is the string must be a float
        try:
            nw_val = float(nw_str)
        except:
            # case is the string is not a valid number
            e_str = 'Entered value is not a valid number.'

    # determines if the new value meets the min/max value requirements
    if nw_val is not None:
        if nw_val < min_val:
            e_str = 'Entered value must be greater than or equal to {0}'.format(min_val)
        elif nw_val > max_val:
            e_str = 'Entered value must be less than or equal to {0}'.format(max_val)
        else:
            return nw_val, e_str

    # shows the error message (if required)
    if show_err:
        show_error(e_str, 'Error!')

    # shows the error and returns a None value
    return None, e_str


def expand_dash_number(num_str):
    '''

    :param x:
    :return:
    '''

    if '-' in num_str:
        if num_str.count('-') > 1:
            return 'NaN'
        else:
            i_dash = num_str.index('-')
            return [str(x) for x in list(range(int(num_str[0:i_dash]),int(num_str[(i_dash+1):])+1))]
    else:
        return [num_str]


def calc_text_width(font, text, w_ofs=0):
    '''

    :param font:
    :param text:
    :return:
    '''

    # creates the font metrics object
    fm = QFontMetrics(font)

    # returns the text width based on the type
    if isinstance(text, list):
        # case is a list, so return the maximum width of all text strings
        return max([fm.width(t) for t in text]) + w_ofs
    else:
        # otherwise, return the width of the text string
        return fm.width(text) + w_ofs


def det_subplot_dim(n_plot):
    '''

    :param n_plot:
    :return:
    '''

    #
    return m.ceil(0.5 * (1 + m.sqrt(1 + 4 * n_plot))) - 1, m.ceil(m.sqrt(n_plot))


def setup_index_arr(ind, n_ele):
    '''
        sets the index arrays for the unique groups
    '''

    # memory allocation
    ind_grp = np.zeros(len(ind), dtype=object)

    # sets the indices of the sub-groups
    for i in range(len(ind)):
        if i == (len(ind) - 1):
            ind_grp[i] = np.array(range(ind[i], n_ele))
        else:
            ind_grp[i] = np.array(range(ind[i], ind[i+1]))

    # returns the index array
    return ind_grp


def show_error(text, title):
    '''

    :param text:
    :param title:
    :return:
    '''

    # otherwise, create the error message
    err_dlg = QMessageBox()
    err_dlg.setText(text)
    err_dlg.setWindowTitle(title)
    err_dlg.setWindowFlags(Qt.WindowStaysOnTopHint)

    # shows the final message
    err_dlg.exec()

def get_index_groups(b_arr):
    '''

    :param b_arr:
    :return:
    '''

    if not any(b_arr):
        return []
    else:
        labels = measure.label(b_arr)
        return [np.where(labels == (i + 1))[0] for i in range(max(labels))]


def det_largest_index_group(b_arr):
    '''

    :param b_arr:
    :return:
    '''

    # determines the index groups from the binary array
    i_grp = get_index_groups(b_arr)

    # returns the largest group of all the index groups
    return i_grp[np.argmax([len(x) for x in i_grp])]

def set_binary_groups(sz, ind):
    '''

    :param sz:
    :param ind:
    :return:
    '''

    if not isinstance(ind, list):
        ind = [ind]

    b_arr = np.zeros(sz, dtype=bool)
    for i in range(len(ind)):
        b_arr[ind[i]] = True

    # returns the final binary array
    return b_arr


def extract_file_name(f_file):
    '''

    :param f_name:
    :return:
    '''

    #
    f_name = os.path.basename(f_file)
    return f_name[:f_name.rfind('.')]

def get_expt_index(exp_name, cluster, ind_arr=None):
    '''

    :param exp_name:
    :param cluster:
    :return:
    '''

    # returns the index of the experiment corresponding to the experiment with name, exp_name
    i_expt = next(i for i in range(len(cluster)) if extract_file_name(cluster[i]['expFile']) == exp_name)
    if ind_arr is None:
        return i_expt
    else:
        return np.where(np.where(ind_arr)[0] == i_expt)[0][0]

def get_para_dict(fcn_para, f_type):
    '''

    :return:
    '''

    return [p for p in fcn_para if fcn_para[p]['gtype'] == f_type]


def set_group_enabled_props(h_groupbox, is_enabled=True):
    '''

    :param h_groupbox:
    :return:
    '''

    for h_obj in h_groupbox.findChildren(QWidget):
        h_obj.setEnabled(is_enabled)
        if isinstance(h_obj, QGroupBox):
            set_group_enabled_props(h_obj, is_enabled)


def init_general_filter_data():
    '''

    :return:
    '''

    f_data = {
        'region_name': [],
        'record_layer': [],
    }

    # returns the field data
    return f_data


def init_rotation_filter_data(is_ud, is_empty=False):
    '''

    :return:
    '''

    # initialisations
    t_type0 = [['Black'], ['UniformDrifting']]

    t_key = {
        't_type': None,
        'sig_type': None,
        'match_type': None,
        'region_name': None,
        'record_layer': None,
        't_freq': {'0.5': '0.5 Hz', '2.0': '2 Hz', '4.0': '4 Hz'},
        't_freq_dir': {'-1': 'CW', '1': 'CCW'},
        't_cycle': {'15': '15 Hz', '120': '120 Hz'},
    }

    if is_empty:
        f_data = {
            't_type': [],
            'sig_type': [],
            'match_type': [],
            'region_name': [],
            'record_layer': [],
            'record_coord': [],
            't_freq': [],
            't_freq_dir': [],
            't_cycle': [],
            'is_ud': [is_ud],
            't_key': t_key,
        }
    else:
        f_data = {
            't_type': t_type0[int(is_ud)],
            'sig_type': ['All'],
            'match_type': ['All'],
            'region_name': ['All'],
            'record_layer': ['All'],
            'record_coord': ['All'],
            't_freq': ['All'],
            't_freq_dir': ['All'],
            't_cycle': ['All'],
            'is_ud': [is_ud],
            't_key': t_key,
        }

    # returns the field data
    return f_data

def get_plot_col(n_plot=1, i_ofs=0):
    '''

    :param index:
    :return:
    '''

    def get_new_colour(index):

        #
        if index < 10:
            return 'C{0}'.format(index)
        else:
            # FINISH ME!
            print('Finish Me!!')

    c = []
    for i_plot in range(n_plot):
        c.append(get_new_colour(i_plot+i_ofs))

    return c


def det_valid_rotation_expt(data, is_ud=False, t_type=None, min_count=2):
    '''

    :return: 
    '''

    # determines which experiments has rotational analysis information associated with them
    is_valid = [x['rotInfo'] is not None for x in data._cluster]
    if is_ud:
        # if experiment is uniform drifting, determine if these trials were performed
        is_valid = [('UniformDrifting' in x['rotInfo']['trial_type'])
                                                    if y else False for x, y in zip(data._cluster, is_valid)]
    elif t_type is not None:
        # if the trial types are given, then ensure that at least 2 trial types are within the experiment
        is_valid = [sum([z in x['rotInfo']['trial_type'] for z in t_type]) >= min_count
                                                    if y else False for x, y in zip(data._cluster, is_valid)]

    # returns the array
    return is_valid

def set_axis_limits(ax, x_lim, y_lim):
    '''

    :param ax:
    :param x_lim:
    :param y_lim:
    :return:
    '''

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

def set_equal_axis_limits(ax, ind):
    '''

    :param ax:
    :param ind:
    :return:
    '''

    # initialisations
    xL, yL = [1e6, -1e6], [1e6, -1e6]

    # determines the overall x/y limits over the axis indices
    for i in ind:
        # retrieves the x/y axis limits
        x_lim, y_lim = ax[i].get_xlim(), ax[i].get_ylim()

        # determines the min/max limits from the
        xL[0], xL[1] = min(xL[0], x_lim[0]), max(xL[1], x_lim[1])
        yL[0], yL[1] = min(yL[0], y_lim[0]), max(yL[1], y_lim[1])

    # resets the axis limits
    xx = [min(xL[0], yL[0]), max(xL[1], yL[1])]
    for i in ind:
        set_axis_limits(ax[i], xx, xx)

def reset_plot_axes_limits(ax, ax_lim, ax_str, is_high):
    '''

    :param ax_lim:
    :param ax_str:
    :param is_low:
    :return:
    '''

    if ax_str == 'x':
        axL = list(ax.get_xlim())
        axL[is_high] = ax_lim
        ax.set_xlim(axL)
    elif ax_str == 'y':
        axL = list(ax.get_ylim())
        axL[is_high] = ax_lim
        ax.set_ylim(axL)
    else:
        axL = list(ax.get_zlim())
        axL[is_high] = ax_lim
        ax.set_zlim(axL)


def combine_nd_arrays(A, B, dim=1, dim_append=0):
    '''

    :param A0:
    :param A:
    :return:
    '''

    # if the original array is empty, then return the new array
    if A is None:
        return B

    #
    n_szA, n_szB = np.shape(A), np.shape(B)

    # appends columns to the original/new arrays if they are not the correct size
    if n_szA[dim] > n_szB[dim]:
        # if the new
        d_dim = n_szA[dim] - n_szB[dim]
        Bnw = np.empty([x if i != dim else d_dim for i, x in enumerate(n_szB)], dtype=object)
        B = np.append(B, Bnw, axis=dim)

    elif n_szA[dim] < n_szB[dim]:
        d_dim = n_szB[dim] - n_szA[dim]
        Anw = np.empty([x if i != dim else d_dim for i, x in enumerate(n_szA)], dtype=object)
        A = np.append(A, Anw, axis=dim)

    # returns the arrays appended across the rows
    return np.append(A, B, axis=dim_append)


def create_stacked_bar(ax, Y, c=None):
    '''

    :param ax:
    :param Y:
    :return:
    '''

    # initialisations
    h_bar, xi_ind = [], np.array(range(np.size(Y, axis=1)))
    if c is None:
        c = get_plot_col(len(xi_ind))

    # creates/appends to the stacked bar graph
    for i_type in range(np.size(Y, axis=0)):
        if i_type == 0:
            # case is the first bar plot stack
            h_bar.append(ax.bar(xi_ind, Y[i_type, :], color=c[i_type]))
            bar_bottom = Y[i_type, :]
        else:
            # case is the other bar-plot stacks
            h_bar.append(ax.bar(xi_ind, Y[i_type, :], bottom=bar_bottom, color=c[i_type]))
            bar_bottom += Y[i_type, :]

    # sets the x-axis tick marks
    ax.set_xticks(xi_ind)

    # returns the bar graph handles
    return h_bar


def create_plot_table(ax, data, row_hdr, col_hdr, loc='bottom', bbox=None, rowColours=None,
                      colColours=None, f_sz=None, colWidths=None, cellColours=None):
    '''

    :param ax:
    :param data:
    :param row_hdr:
    :param col_hdr:
    :return:
    '''

    # creates the table object
    h_table = ax.table(cellText=data, rowLabels=row_hdr, colLabels=col_hdr, loc=loc, rowLoc='center',
                       cellLoc='center', bbox=bbox, rowColours=rowColours, colColours=colColours,
                       cellColours=cellColours, colWidths=colWidths)
    # h_table.auto_set_column_width(False)

    # sets the table font size (if provided)
    if f_sz is not None:
        h_table.auto_set_font_size(False)
        h_table.set_fontsize(f_sz)

    # returns the table object
    return h_table

def add_rowcol_sum(A):
    '''

    :param A:
    :return:
    '''


    A_csum = np.hstack((A, np.reshape(np.sum(A, axis=1), (-1, 1))))
    return np.vstack((A_csum, np.sum(A_csum, axis=0)))


def create_bubble_boxplot(ax, Y, wid=0.75, plot_median=True, s=60):
    '''

    :param Y:
    :return:
    '''

    # initialisations
    n_plot, col = len(Y), get_plot_col(len(Y))

    #
    for i_plot in range(n_plot):
        # creates the bubble plot
        X = (i_plot + 1) + wid * (0.5 - np.random.rand(len(Y[i_plot])))
        ax.scatter(X, Y[i_plot], s=s, facecolors='none', edgecolors=col[i_plot])

        # plots the median line (if required)
        if plot_median:
            Ymd = np.median(Y[i_plot])
            ax.plot((i_plot + 1) + (wid / 2) * np.array([-1, 1]), Ymd * np.ones(2), linewidth=2)

    # sets the x-axis limits/ticks
    ax.set_xlim(0.5, n_plot + 0.5)
    ax.set_xticks(np.array(range(n_plot)) + 1)


def create_connected_line_plot(ax, Y, s=60):
    '''

    :param ax:
    :param Y:
    :param s:
    :return:
    '''

    # initialisations
    n_plot, n_cell, col = len(Y), len(Y[0]), get_plot_col(len(Y))
    X = np.ones(n_cell)
    y_mn0, y_mn1 = np.mean(Y[0]), np.mean(Y[1])

    # plots the connecting lines
    for i_cell in range(n_cell):
        ax.plot([1, 2], [Y[0][i_cell], Y[1][i_cell]], 'k--')

    # creates the scatter plots
    ax.scatter(X, Y[0], s=s, facecolors='none', edgecolors=col[0])
    ax.scatter(2 * X, Y[1], s=s, facecolors='none', edgecolors=col[1])

    # creates the mean scatter plot points
    ax.plot([1, 2], [y_mn0, y_mn1], 'k', linewidth=4)
    ax.scatter(1, y_mn0, s=2 * s, edgecolors=col[0])
    ax.scatter(2, y_mn1, s=2 * s, edgecolors=col[1])

    # sets the x-axis limits/ticks
    ax.set_xlim(0.5, n_plot + 0.5)
    ax.set_xticks(np.array(range(n_plot)) + 1)


def det_reqd_cond_types(data, t_type):
    '''

    :param t_type:
    :return:
    '''

    is_rot_expt = det_valid_rotation_expt(data, t_type=t_type, min_count=1)
    return [[z for z in t_type if z in x['rotInfo']['trial_type']]
                                                for x, y in zip(data._cluster, is_rot_expt) if y]


def lcm(x, y):
   """This function takes two
   integers and returns the L.C.M."""

   # choose the greater number
   if x > y:
       greater = x
   else:
       greater = y

   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break

       greater += 1

   return lcm


def combine_stacks(x, y):
    '''

    :param x:
    :param y:
    :return:
    '''

    #
    n_col_x, n_col_y = np.size(x, axis=1), np.size(y, axis=1)

    #
    if n_col_x > n_col_y:
        n_row_y = np.size(y, axis=0)
        y = np.concatenate((y, np.empty((n_row_y, n_col_x - n_col_y), dtype=object)), axis=1)
    elif n_col_x < n_col_y:
        n_row_x = np.size(x, axis=0)
        x = np.concatenate((x, np.empty((n_row_x, n_col_y - n_col_x), dtype=object)), axis=1)

    #
    return np.dstack((x, y))


def calc_phase_spike_freq(r_obj):
    '''

    :param r_obj:
    :return:
    '''

    # sets the spiking frequency across all trials
    sp_f0 = [sp_freq_fcn(x, y[0]) for x, y in zip(r_obj.t_spike, r_obj.t_phase)]
    if r_obj.is_single_cell:
        sp_f = [np.squeeze(x) for x in sp_f0]
    else:
        # if not single cell, then calculate average over all trials
        sp_f = [np.mean(x, axis=1) for x in sp_f0]

    # returns the total/mean spiking frequency arrays
    return sp_f0, sp_f


def setup_spike_freq_plot_arrays(r_obj, sp_f0, sp_f, ind_type, n_sub=3, plot_trend=False, is_3d=False):
    '''

    :param sp_f0:
    :param sp_f:
    :return:
    '''

    # memory allocation
    A, i_grp = np.empty(n_sub, dtype=object), None
    s_plt, sf_trend, sf_stats = dcopy(A), dcopy(A), dcopy(A)

    # combines the all the data from each phase type
    for i_sub in range(n_sub):
        if is_3d:
            # case is a 3d scatter plot
            s_plt[i_sub] = [combine_spike_freq(sp_f, i) for i in range(3)]
        elif r_obj.is_ud:
            # case is uniform drifting
            if (i_sub + 1) == n_sub:
                # case is the CW vs CCW phase
                sp_sub = [np.vstack((sp_f[x][:, 1], sp_f[y][:, 1])).T for x, y in zip(ind_type[0], ind_type[1])]
                sp_f0_sub = [combine_stacks(sp_f0[x][:, :, 1], sp_f0[y][:, :, 1]) for x, y in
                             zip(ind_type[0], ind_type[1])]
            else:
                # case is the CW/CCW vs BL phases
                sp_sub = np.array(sp_f)[ind_type[i_sub]]
                sp_f0_sub = [sp_f0[x] for x in ind_type[i_sub]]

            # sets the plot values
            s_plt[i_sub] = [combine_spike_freq(sp_sub, i) for i in range(2)]

            # calculates the wilcoxon signed rank test between the baseline/stimuli phases
            # if not r_obj.is_single_cell:
            sf_stats[i_sub] = calc_spike_freq_stats(sp_f0_sub, [0, 1])

            # adds the trend-line (if selected)
            if plot_trend:
                sf_trend[i_sub] = calc_spike_freq_correlation(sp_sub, [0, 1])

            # sets the cell group indices (over each filter type)
            if i_sub == 0:
                ii = np.append(0, np.cumsum([np.size(x, axis=0) for x in sp_f0_sub]))
        else:
            # case is the default plot
            i1, i2 = 1 * (i_sub > 1), 1 + (i_sub > 0)
            s_plt[i_sub] = [combine_spike_freq(sp_f, i) for i in [i1, i2]]

            # calculates the wilcoxon signed rank test between the stimuli phases
            # if not r_obj.is_single_cell:
            sf_stats[i_sub] = calc_spike_freq_stats(sp_f0, [i1, i2])

            # adds the trend-line (if selected)
            if plot_trend:
                sf_trend[i_sub] = calc_spike_freq_correlation(sp_f, [i1, i2])

            # sets the cell group indices (over each filter type)
            if i_sub == 0:
                ii = np.append(0, np.cumsum([np.size(x, axis=0) for x in sp_f0]))

    # sets the indices for each filter type grouping
    if (not is_3d):
        i_grp = [np.array(range(ii[i], ii[i + 1])) for i in range(len(ii) - 1)]

    # returns the important arrays
    return s_plt, sf_trend, sf_stats, i_grp


def calc_spike_freq_stats(sp_f0, ind, concat_results=True):
    '''

    :param sp_f0:
    :param ind:
    :return:
    '''

    # memory allocation
    n_filt = len(sp_f0)
    n_row = [np.size(x, axis=0) if (x is not None) else 0 for x in sp_f0]
    sf_stats = [np.zeros(nr) for nr in n_row]

    # calculates the p-values for each of the trials
    for i_filt in range(n_filt):
        if n_row[i_filt] > 0:
            for i_row in range(n_row[i_filt]):
                x, y = sp_f0[i_filt][i_row, :, ind[0]], sp_f0[i_filt][i_row, :, ind[1]]
                ii = np.logical_and(~np.equal(x, None), ~np.equal(y, None))
                results = r_stats.wilcox_test(FloatVector(x[ii]), FloatVector(y[ii]), paired=True, exact=True)
                sf_stats[i_filt][i_row] = results[results.names.index('p.value')][0]

    # returns the stats array
    if concat_results:
        return np.concatenate(sf_stats)
    else:
        return sf_stats


def calc_spike_freq_correlation(sp_f, ind):
    '''

    :param sp_f:
    :param ind:
    :param is_single_cell:
    :return:
    '''

    # memory allocation
    n_filt = np.size(sp_f, axis=0)
    sp_corr = np.zeros((n_filt, 1))

    #
    for i_filt in range(n_filt):
        # sets the x/y points for the correlation calculation
        x, y = sp_f[i_filt][:, ind[0]], sp_f[i_filt][:, ind[1]]
        sp_corr[i_filt], _ = curve_fit(lin_func, x, y)

    # returns the correlation array
    return sp_corr

##################################################
####    ROC ANALYSIS CALCULATION FUNCTIONS    ####
##################################################

def get_roc_xy_values(roc):
    '''

    :param roc:
    :return:
    '''

    # retrieves the roc coordinates and returns them in a combined array
    roc_ss, roc_sp = roc[roc.names.index('sensitivities')], roc[roc.names.index('specificities')]
    return np.vstack((1-np.array(roc_ss), np.array(roc_sp))).T


def get_roc_auc_value(roc):
    '''

    :param roc:
    :return:
    '''

    # returns the roc curve integral
    return roc[roc.names.index('auc')][0]


def calc_inter_roc_significance(roc1, roc2, method, boot_n):
    '''

    :param roc1:
    :param roc2:
    :return:
    '''

    # runs the test and returns the p-value
    results = _roc_test(roc1, roc2, method=method[0].lower(), boot_n=boot_n, progress='none')
    return results[results.names.index('p.value')][0]

# roc.test(roc1, roc2, method=c("delong", "bootstrap",
# "venkatraman", "sensitivity", "specificity"), sensitivity = NULL,
# specificity = NULL, alternative = c("two.sided", "less", "greater"),
# paired=NULL, reuse.auc=TRUE, boot.n=2000, boot.stratified=TRUE,
# ties.method="first", progress=getOption("pROCProgress")$name,
# parallel=FALSE, ...)


def calc_roc_curves(comp_vals, roc_type='Cell Spike Times', x_grp=None, y_grp=None, ind=[1, 2]):
    '''

    :param t_spike:
    :return:
    '''

    # sets up the x/y groupings and threshold values based on type
    if (x_grp is None) or (y_grp is None):
        if roc_type == 'Cell Spike Times':
            # case is the cell spike times

            # sets the cw/ccw trial spike arrays
            n_trial = np.sum([(x is not None) for x in comp_vals[:, 0]])
            t_sp_cc = [comp_vals[i, ind[0]] for i in range(n_trial)]       # CW trial spikes
            t_sp_ccw = [comp_vals[i, ind[1]] for i in range(n_trial)]      # CCW trial spikes

            # determines the spike counts for the cc/ccw trials
            x_grp, y_grp = spike_count_fcn(t_sp_cc), spike_count_fcn(t_sp_ccw)

        elif roc_type == 'Cell Spike Counts':
            # case is the cell spike counts

            # sets the pooled neuron preferred/non-preferred trial spike counts
            x_grp, y_grp = comp_vals[:, 0], comp_vals[:, 1]

    # sets up the roc
    nn = len(x_grp)
    roc_pred, roc_class = np.hstack((np.zeros(nn), np.ones(nn))), np.hstack((x_grp, y_grp))
    return r_pROC.roc(FloatVector(roc_pred), FloatVector(roc_class), direction = ">")


def calc_cell_roc_bootstrap_wrapper(p_data):
    '''

    :param p_data:
    :return:
    '''

    # initialisations
    t_spike, n_boot, ind = p_data[0], p_data[1], p_data[2]

    # sets the cw/ccw trial spike arrays
    n_trial = np.sum([(x is not None) for x in t_spike[:, 0]])
    t_sp_p1 = [t_spike[i, ind[0]] for i in range(n_trial)]  # 1st phase trial spikes
    t_sp_p2 = [t_spike[i, ind[1]] for i in range(n_trial)]  # 2nd phase trial spikes

    # determines the spike counts for the cc/ccw trials
    n_spike = [spike_count_fcn(t_sp_p1), spike_count_fcn(t_sp_p2)]

    return calc_cell_roc_bootstrap(None, n_spike, n_boot=n_boot, ind=ind)


def calc_roc_conf_intervals(p_data):
    '''

    :param roc:
    :param type:
    :param n_boot:
    :return:
    '''

    # parameters and input arguments
    c_lvl = 0.95
    roc, grp_stype, n_boot = p_data[0],p_data[1], p_data[2]

    # calculates the roc curve integral
    results = _ci_auc(roc, method=grp_stype[0].lower(), boot_n=n_boot, conf_level=c_lvl, progress='none')
    return [results[1] - results[0], results[2] - results[1]]

def calc_cell_group_types(auc_sig, stats_type):
    '''

    :param auc_sig:
    :return:
    '''

    # memory allocation

    # Cell group type convention
    #   =0 - Both MS/DS
    #   =1 - MS but not DS
    #   =2 - Not MS
    g_type = 2 * np.ones(np.size(auc_sig, axis=0), dtype=int)

    # determines which cells are motion/direction sensitive
    #  * Motion Sensitive - either (one direction only is significant), OR (both are significant AND
    #                       the CW/CCW phase difference is significant)
    n_sig = np.sum(auc_sig[:, :2], axis=1)

    is_ms = n_sig > 0

    #
    if stats_type == 'Wilcoxon Paired Test':
        # case is if phase statistics ws calculated via Wilcoxon paried test
        is_ds = np.logical_or(n_sig == 1, np.logical_and(n_sig == 2, auc_sig[:, 2]))
    else:
        # case is if phase stats was calculated using ROC analysis
        is_ds = auc_sig[:, 2]

    # sets the MS/DS and MS/Not DS indices
    g_type[np.logical_and(is_ms, is_ds)] = 0            # case is both MS/DS
    g_type[np.logical_and(is_ms, ~is_ds)] = 1           # case is MS but not DS

    # returns the group type array
    return g_type


# # calculates the table dimensions
# bbox, t_data, cell_col, c_wids = cf.calc_table_dim(self.plot_fig, 1, table_font, n_sig_grp, row_hdr,
#                                                    row_cols, g_type, t_loc='bottom')

# bbox, t_data, cell_col, c_wids = cf.add_plot_table(self.plot_fig, 1, table_font, n_sig_grp, row_hdr,
#                                                    g_type, row_cols, col_cols, table_fsize, t_loc='bottom')


# calculates the table dimensions
def add_plot_table(fig, ax, font, data, row_hdr, col_hdr, row_cols, col_cols, t_loc,
                   cell_cols=None, n_row=1, n_col=2, pfig_sz=1.0, t_ofs=0, h_title=None, p_wid=1.0):
    '''

    :param ax:
    :param font:
    :param data:
    :param row_hdr:
    :param col_hdr:
    :return:
    '''

    # initialisations
    n_line, title_hght, pWT = 0, 0, 0.5
    y0, w_gap, h_gap, cell_wid, cell_wid_row, n_row_data = 0.01, 10 * p_wid, 2, 0, 0, np.size(data, axis=0)

    # creates the font metrics object
    fm, f_sz0 = QFontMetrics(font), font.pointSize()

    # sets the axis object (if the axis index was provided)
    if isinstance(ax, int):
        ax = fig.ax[ax]

    # objection dimensioning
    fig_wid, fig_hght = fig.width(), fig.height() * pfig_sz
    cell_hght0, ax_wid, ax_hght, ax_pos = fm.height(), ax.bbox.width, ax.bbox.height, ax.get_position()

    # if there is a title, then retrieve the title height
    if h_title is not None:
        fm_title = QFontMetrics(create_font_obj(size=h_title.get_fontsize()))
        title_hght = fm_title.height()

    if t_loc == 'bottom':
        # case is the table is located at the bottom of the axes

        # if there is an x-label then increment the line counter
        if ax.xaxis.label is not None:
            n_line += 1

        # if there is an x-ticklabels then increment the line counter depending on the number of lines
        h_xticklbl = ax.get_xticklabels()
        if h_xticklbl is not None:
            n_line += np.max([(1 + x._text.count('\n')) for x in h_xticklbl])

    elif t_loc == 'top':
        # case is the table is located at the top of the axes

        # case is the title
        if len(ax.title._text):
            n_line += 1

    # parameters and other dimensioning
    n_rowhdr_line, n_colhdr_line = row_hdr[0].count('\n') + 1, col_hdr[0].count('\n') + 1

    ############################################
    ####    CELL ROW/HEIGHT CALCULATIONS    ####
    ############################################

    # calculates the maximum column header width
    if col_hdr is not None:
        cell_wid = np.max([fm.width(x) for x in col_hdr]) * p_wid

    # calculates the maximum of the cell widths
    for i_row in range(n_row_data):
        cell_wid = max(cell_wid, np.max([fm.width(str(x)) for x in data[i_row, :]]) * p_wid)

    # calculates the maximum row header width
    if row_hdr is not None:
        cell_wid_row = np.max([fm.width(x) for x in row_hdr]) * p_wid

    # sets the row header/whole table widths and cell/whole table heights
    table_wid = ((cell_wid_row > 0) * (cell_wid_row + w_gap) + len(col_hdr) * (cell_wid + w_gap)) / ax_wid
    table_hght = (n_colhdr_line * cell_hght0 + (n_colhdr_line + 1) * h_gap + \
                  n_row_data * (cell_hght0 + (n_rowhdr_line + 1) * h_gap))

    # if the table width it too large, then rescale
    sp_width = get_subplot_width(ax, n_col)
    if table_wid > sp_width:
        ptable_wid = sp_width / table_wid
        cell_wid, cell_wid_row = ptable_wid * cell_wid, ptable_wid * cell_wid_row
        table_wid = sp_width

    if t_loc == 'bottom':
        ax_bot = np.floor(ax_pos.y1 / (1 / n_row)) / n_row
        ax_y0, ax_y1 = ax_bot * fig_hght + table_hght + cell_hght0 * (1.5 + n_line), ax.bbox.y1
    elif t_loc == 'top':
        ax_top = np.ceil(ax_pos.y1 / (1 / n_row)) / n_row
        ax_y0, ax_y1 = ax.bbox.y0, ax_top * fig_hght - (table_hght + 2 * cell_hght0)
    else:
        ax_y0 = fig_hght * np.floor(ax_pos.y0 / (1 / n_row)) / n_row
        ax_y1 = fig_hght * np.ceil(ax_pos.y1 / (1 / n_row)) / n_row

    # sets the bounding box dimensions
    ax_hght_new = pfig_sz * (ax_y1 - ax_y0) / fig_hght
    ax_fig_hght = ax_hght_new * fig_hght
    table_x0 = get_axis_scaled_xval(ax, (sp_width - table_wid) / 2, n_col)

    if t_loc == 'bottom':
        table_y0 = -(table_hght + (1 + pWT) * title_hght + cell_hght0 * (1 + n_line)) / ax_fig_hght
        bbox = [table_x0, table_y0, table_wid, table_hght / ax_fig_hght]
    elif t_loc == 'top':
        table_y0 = 1 + (cell_hght0 + (1 + pWT) * title_hght) / ax_fig_hght
        bbox = [table_x0, table_y0, table_wid, table_hght / ax_fig_hght]
    else:
        table_y0 = 1 - (t_ofs + table_hght + (1 + pWT) * title_hght + cell_hght0) / ax_fig_hght
        bbox = [table_x0, table_y0, table_wid, table_hght / ax_fig_hght]

    ####################################################
    ####    AXIS RE-POSITIONING & TABLE CREATION    ####
    ####################################################

    # resets the axis position to accomodate the table
    ax_pos_nw = [ax_pos.x0, ax_y0 / fig_hght, ax_pos.width, ax_hght_new]
    ax.set_position(ax_pos_nw)

    # resets the position of the title object
    if h_title is not None:
        x_title = get_axis_scaled_xval(ax, 0.5, n_col, False)
        h_title.set_position([x_title, table_y0 + (table_hght + pWT * title_hght) / ax_fig_hght])

    # sets the table parameters based on whether there is a row header column
    if cell_wid_row == 0:
        # case is there is no row header column
        c_wids = [cell_wid + w_gap] * len(col_hdr)
    else:
        # case is there is a row header column
        c_wids = [cell_wid_row + w_gap] + [cell_wid + w_gap] * len(col_hdr)

        #
        if cell_cols is None:
            cell_cols = np.vstack([['w'] * np.size(data, axis=1)] * np.size(data, axis=0))

        # resets the data and column header arrays
        data = np.hstack((np.array(row_hdr).reshape(-1, 1), data))
        col_hdr, col_cols = [''] + col_hdr, ['w'] + col_cols

        #
        if np.size(data, axis=0) == 1:
            cell_cols = np.array(row_cols + list(cell_cols[0]), dtype=object).reshape(-1, 1).T
        else:
            cell_cols = np.hstack((np.array(row_cols, dtype=object).reshape(-1, 1), cell_cols))

    # creates the table
    h_table = create_plot_table(ax, data, None, col_hdr, bbox=bbox, colWidths=c_wids, cellColours=cell_cols,
                                colColours=col_cols, f_sz=f_sz0)

    # removes the outline from the top-left cell
    h_table._cells[(0, 0)].set_linewidth(0)

    # returns the cell width/height
    return [h_table, table_y0, ax_fig_hght]


def get_subplot_width(ax, n_col):
    '''

    :param ax:
    :return:
    '''

    return (t_wid_f / n_col) / ax.get_position().width


def get_axis_scaled_xval(ax, x, n_col, is_scaled=True):
    '''

    :param ax:
    :param x:
    :return:
    '''

    # retrieves the axis position vector
    ax_pos = ax.get_position()
    x_col = np.floor(ax_pos.x0 / (1 / n_col)) / n_col

    # calculates the subplot axis left/right location
    sp_left = (1 - t_wid_f) / 2. + (x_col - ax_pos.x0) / ax_pos.width
    sp_right = ((x_col + t_wid_f / n_col) - ax_pos.x0) / ax_pos.width

    # returns the scaled value
    if is_scaled:
        return sp_left + (x / get_subplot_width(ax, n_col)) * (sp_right - sp_left)
    else:
        return sp_left + x * (sp_right - sp_left)

def reset_axes_dim(ax, d_type, d_val, is_prop):
    '''

    :param ax:
    :param d_type:
    :param d_val:
    :param is_prop:
    :return:
    '''

    #
    ax_pos0 = ax.get_position()
    ax_pos = [ax_pos0.x0, ax_pos0.y0, ax_pos0.width, ax_pos0.height]
    i_dim = ['left', 'bottom', 'width', 'height'].index(d_type.lower())

    if is_prop:
        ax_pos[i_dim] *= (1 + d_val)
    else:
        ax_pos[i_dim] = d_val

    #
    ax.set_position(ax_pos)


def setup_trial_condition_filter(rot_filt, plot_cond):
    '''

    :param plot_cond:
    :return:
    '''

    if not isinstance(plot_cond, list):
        plot_cond = [plot_cond]

    # determines the unique trial types within the experiment that match the required list
    if len(plot_cond) == 0:
        # if the black phase is not in any of the experiments, then output an error to screen
        e_str = 'At least one trial condition type must be selected before running this function.'
        return None, e_str, 'No Trial Conditions Selected'
    else:
        t_type_exp = ['Black'] + plot_cond

    # initialises the rotation filter (if not set)
    if rot_filt is None:
        rot_filt = init_rotation_filter_data(False)

    # sets the trial types into the rotation filter
    rot_filt['t_type'] = list(np.unique(flat_list(t_type_exp)))
    if 'Black' not in rot_filt['t_type']:
        # if the black phase is not in any of the experiments, then output an error to screen
        e_str = 'The loaded experiments do not include the "Black" trial condition. To run this function ' \
                'you will need to load an experiment with this trial condition.'
        return None, e_str, 'Invalid Data For Analysis'

    elif len(rot_filt['t_type']) == 1:
        # if there are insufficient trial types in the loaded experiments, then create the error string
        e_str = 'The loaded experiments only has the "{0}" trial condition. To run this function you will ' \
                'need to load a file with the following trial condition:\n'.format(rot_filt['t_type'][0])
        for tt in plot_cond:
            e_str = '{0}\n => {1}'.format(e_str, tt)

        # outputs the error to screen and exits the function
        return None, e_str, 'Invalid Data For Analysis'

    # otherwise, return the rotational filter
    return rot_filt, None, None


def det_matching_filters(r_obj, ind):
    '''

    :param r_obj:
    :param ind:
    :return:
    '''

    # sets the candidate rotation filter dictionary
    r_filt0 = r_obj.rot_filt_tot[ind]

    # loops through each of the filter dictionaries determining the match
    for i in range(len(r_obj.rot_filt_tot)):
        # if the current index is
        if i == ind:
            continue

        # loops through each of the field values determining if they all match
        is_match = True
        for f_key in r_filt0.keys():
            # no need to consider the trial type field
            if f_key == 't_type':
                continue

            # if the field values do not match, then update the match flag and exit the loop
            if r_filt0[f_key] != r_obj.rot_filt_tot[i][f_key]:
                is_match = False
                break

        # if all the fields match, then return the matching index
        if is_match:
            return [ind, i]

def calc_dirsel_scores(s_plt, sf_stats, p_value=0.05):
    '''

    :param s_plt:
    :param sf_stats:
    :return:
    '''

    # calculates the relative change for CW/CCW from baseline, and CCW to CW
    grad_CW = np.array(s_plt[0][1]) / np.array(s_plt[0][0])
    grad_CCW = np.array(s_plt[1][1]) / np.array(s_plt[1][0])
    grad_CCW_CW = np.array(s_plt[1][1]) / np.array(s_plt[0][1])

    # calculates the score type for the CW/CCW phases
    sf_score = np.zeros((len(grad_CW), 3), dtype=int)
    sf_score[:, 0] = (sf_stats[0] < p_value).astype(int) * (1 + (grad_CW > 1).astype(int))
    sf_score[:, 1] = (sf_stats[1] < p_value).astype(int) * (1 + (grad_CCW > 1).astype(int))

    # if both CW and CCW are significant (wrt the baseline phase), then determine from these cells which
    # cells have a significant CW/CCW difference (1 for CW, 2 for CCW, 0 otherwise):
    #   1) significant for a single direction (either CW or CCW preferred)
    #   2) significant for both, but the CCW/CW gradient is either > 1 (for CCW preferred) or < 1 (for CW preferred)
    both_sig = np.logical_and(sf_score[:, 0] > 0, sf_score[:, 1] > 0)
    sf_score[both_sig, 2] = (sf_stats[2][both_sig] < p_value).astype(int) * (1 + (grad_CCW_CW[both_sig] > 1).astype(int))

    # returns the scores array
    return sf_score

def det_cell_match_indices(r_obj, ind, r_obj2=None):
    '''

    :param r_obj:
    :param ind:
    :return:
    '''

    if r_obj2 is None:
        # determines the cell id's which overlap each other
        cell_id = [1000 * r_obj.i_expt[i] + np.array(flat_list(r_obj.clust_ind[i])) for i in ind]
        id_match = np.array(list(set(cell_id[0]).intersection(set(cell_id[1]))))

        # returns the indices of the matching
        return np.searchsorted(cell_id[0], id_match), np.searchsorted(cell_id[1], id_match)
    else:
        #
        if isinstance(ind, int):
            cell_id_1 = 1000 * r_obj.i_expt[ind] + np.array(flat_list(r_obj.clust_ind[ind]))
            cell_id_2 = 1000 * r_obj2.i_expt[ind] + np.array(flat_list(r_obj2.clust_ind[ind]))
        else:
            cell_id_1 = 1000 * r_obj.i_expt[ind[0]] + np.array(flat_list(r_obj.clust_ind[ind[0]]))
            cell_id_2 = 1000 * r_obj2.i_expt[ind[1]] + np.array(flat_list(r_obj2.clust_ind[ind[1]]))

        # returns the indices of the matching cells
        id_match = np.sort(np.array(list(set(cell_id_1).intersection(set(cell_id_2)))))
        return np.searchsorted(cell_id_1, id_match), np.searchsorted(cell_id_2, id_match)

def split_unidrift_phases(data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, dt, t0=0):
    '''

    :param rot_filt:
    :return:
    '''

    from analysis_guis.dialogs.rotation_filter import RotationFilteredData

    # parameters
    rot_filt['t_freq_dir'] = ['-1', '1']

    # splits the data by the forward/reverse directions
    r_obj = RotationFilteredData(data, rot_filt, i_cluster, plot_exp_name, plot_all_expt, plot_scope, True)
    if (t0 + dt) > r_obj.t_phase[0][0]:
        return None, None

    # shortens the stimuli phases to the last/first dt of the baseline/stimuli phases
    t_phase, n_filt = r_obj.t_phase[0][0], int(r_obj.n_filt / 2)
    ind_type = [np.where([tt in r_filt['t_freq_dir'] for r_filt in
                          r_obj.rot_filt_tot])[0] for tt in rot_filt['t_freq_dir']]

    # reduces down the fields to account for the manditory direction filtering
    r_obj.t_phase = [[dt] * len(x) for x in r_obj.t_phase]
    r_obj.phase_lbl = ['Baseline', 'Clockwise', 'Counter-Clockwise']
    r_obj.lg_str = [x.replace('Reverse\n', '') for x in r_obj.lg_str[:n_filt]]
    r_obj.i_expt, r_obj.ch_id, r_obj.cl_id = r_obj.i_expt[:n_filt], r_obj.ch_id[:n_filt], r_obj.cl_id[:n_filt]

    # loops through each stimuli direction (CW/CCW) and for each filter reducing the stimuli phases
    for i_dir in range(len(rot_filt['t_freq_dir'])):
        for i_filt in range(n_filt):

            # for each cell (in each phase) reduce the spikes to fit the shortened stimuli range
            ii = ind_type[i_dir][i_filt]
            for i_cell in range(np.size(r_obj.t_spike[ii], axis=0)):
                for i_trial in range(np.size(r_obj.t_spike[ii], axis=1)):
                    for i_phase in range(np.size(r_obj.t_spike[ii], axis=2)):
                        # reduces the spikes by the shortened stimuli phase depending on the phase
                        if r_obj.t_spike[ii][i_cell, i_trial, i_phase] is not None:
                            x = dcopy(r_obj.t_spike[ii][i_cell, i_trial, i_phase])
                            if i_phase == 0:
                                # case is the baseline phase
                                r_obj.t_spike[ii][i_cell, i_trial, i_phase] = x[x > (t_phase - dt)]
                            else:
                                # case is the stimuli phase
                                jj = np.logical_and(x >= t0, x < (t0 + dt))
                                r_obj.t_spike[ii][i_cell, i_trial, i_phase] = x[jj]

    # returns the rotational analysis object
    return r_obj, ind_type

def eval_class_func(funcName, *args):
    '''

    :param funcName:
    :param args:
    :return:
    '''

    return funcName(*args)