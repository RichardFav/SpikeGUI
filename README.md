# spikeGUI 
:email: &nbsp; **For any question, contact Richard Faville (richard.faville@gmail.com)**

**:warning: :construction: &nbsp; This is a work in progress**

## About
SpikeGUI is an extracellular spike analysis software with graphical user interface. The main aim of this software is to correlate spike times with various sensory stimuli and behavioural variables, including vestibular and visual stimulation and eye movements. Spike times are generated using [KiloSort](https://github.com/cortex-lab/KiloSort) and [KiloSort2](https://github.com/MouseLand/Kilosort), The pupil, ear, and body positions using [DeepLabCut](https://github.com/AlexEMG/DeepLabCut), and open-field analysis of spatial behaviour using [opendirection](https://github.com/adamltyson/opendirection). 

This pipeline is written by Richard Faville in close collaboration with Sepiedeh Keshavarzi and is funded by the [Margrie lab](https://www.sainsburywellcome.org/web/groups/margrie-lab) at [Sainsbury Wellcome Centre](https://www.sainsburywellcome.org/web/). 

probez and rotation-analysis packages are written by [Stephen Lenzi](https://github.com/stephenlenzi). 

vest_phys and pyphys packages are written by [Charly Rousseau](https://github.com/crousseau). 

margrie_libs package is written by [Charly Rousseau](https://github.com/crousseau) with contributions from [Stephen Lenzi](https://github.com/stephenlenzi). 

## Details

This package allows selection and comparison of data based on multiple criteria, including the recording regions (i.e. retrosplenial cortex, superior colliculus, V1, etc.) and sub-regions (i.e. anatomical layers), cell types (i.e. putative excitatory and inhibitory cells), freely-moving tuning types (i.e. head direction cells, angular head velocity cells, speed cells), experimental condition (i.e. vestibular stimulus, visual stimulus, etc.), and interventions (i.e. control group, lesion group) for the rest of the analysis. The following analyses are available in this package:

* **Cluster Matching** – This group of functions identifies the same single unit in two different (discontinuous) recording sessions based on the similarity of the average spike waveform, the probe geometry, the spiking rate, and the interspike interval (ISI) distribution. These functions have been used to track individual units between head-fixed and freely moving recordings that occurred on the same day with a maximum of 5 hours in between. Individual parameters can be adjusted using the graphical interface. These functions require two sets of data files ("fix" and "free" .cdata files) as input.

* **Cluster Classification** – This function group uses K-means clustering to classify the isolated units as wide or narrow spiking based on features of their average waveform. The "Cluster Cross-Correlogram" function identifies putative monosynaptic excitatory and inhibitory interactions between single units based on the cross-correlogram analysis and overlays the result (excitatory or inhibitory cell type) on the cluster classification plot.

* **Eye Tracking** – This function group takes the tracked pupil position and the spiking data frame (.cdata files) as inputs. It identifies and plots fast eye-movement events at temporal and nasal directions, and correlates the firing rate of each single unit with pupil position. 

* **Rotation Analysis** – This group of functions performs multiple analyses to quantify and visualise the spiking activity of individual or population of units in response to rotational vestibular and/or optic flow stimulations (horizontal). It aligns the stimulus and spike times, identifies and quantifies evoked responses, constructs heatmaps of population activity, and constructs the angular velocity tuning plots. 

* **Angular Head Velocity Analysis** – This group of functions quantifies the correlation between the angular head velocity and the spiking rate of individual units under various head-fixed and freely moving experimental conditions. It also calculates and compares various parameters of a linear fit between the two variables, including the slope and the intercept. 

* **ROC Analysis** – This group of functions performs ROC analysis to quantify speed and direction discrimination of individual neurons and to compare the results between various experimental conditions.

* **Rotation Discrimination Analysis and Kinematic Discrimination Analysis** – These function groups perform linear discriminant analysis (LDA) to quantify population (or pseudo-population) decoding of speed and direction and to compare the results between various experimental conditions. 

* **Freely Moving Analysis** – This group of functions takes both head-fixed (.cdata files) and open-field analysis (from [opendirection](https://github.com/adamltyson/opendirection)) datasets as input. It quantifies and displays the population data from the open-field analysis, and for each tracked single unit, quantifies the correlation between the "fix" and "free" angular velocity tuning functions. 
