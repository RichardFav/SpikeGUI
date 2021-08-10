# module import
import os
import sys
import time

# change R home directory here (if using Linux)
if sys.platform == 'linux':
    os.environ['R_HOME'] = '/home/skeshav/miniconda3/envs/rotation_analysis/lib/R/'

from PyQt5.QtWidgets import QApplication
from analysis_guis import main_analysis

import seaborn as sns
#sns.set_palette(sns.color_palette("Paired", 8))
#colors = ["cobalt", "light orange", "teal", "dusty lavender", "sea blue", "maize", "dull teal", "purpley grey",
#          "turquoise blue", "sandy", "dark seafoam", "dark lilac"]

def main():
    # sets the windows style (if on Subiculum)
    if sys.platform == 'linux':
        QApplication.setStyle('windows')

    # runs the analysis GUIs
    app = QApplication([])
    time.sleep(0.5)
    h_main = main_analysis.AnalysisGUI()
    time.sleep(0.5)
    app.exec()

if __name__ == "__main__":
    main()