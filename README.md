# Thermodynamic Labeling Enables CO₂ Prediction in Polymers of Intrinsic Microporosity Under Small Data Constraints 
### Authors: Johnathan W. Campbell, Ashley P. DeRegis, Jeffrey A. Laub, Konstantinos D. Vogiatzis

# Abstract
Porous polymers, particularly polymers of intrinsic microporosity (PIMs), combine high surface areas with tunable functionalities, positioning them as promising materials for post-combustion CO₂ capture technologies. A key physical parameter of these materials is the isosteric heat of adsorption (Qₛₜ), which quantifies the interaction strength between CO₂ molecules and the polymer framework. In this work, we demonstrate that data-driven models constructed from computationally determined physicochemical descriptors can accurately predict Qst at room temperature using a modestly sized dataset of 75 PIMs. By optimizing model parameters through Bayesian methods and interpreting feature importance using SHAP analysis, we identify the molecular characteristics that most strongly influence the prediction of CO2-PIMs isosteric heats of adsorption. These results demonstrate that accurate predictive workflows can be developed using relatively small datasets. Such workflows can accelerate the design of polymeric adsorbents by identifying key molecular descriptors that correlate with the isosteric heat of adsorption. Overall, this adaptable methodology can be extended to other gas‐separation challenges, helping pave the way for faster discovery of advanced polymeric membranes for capturing CO₂. 


# Dataset Availability
The original dataset used for this study can be found in: Raghuram Thyagarajan; Sholl, D. S. A Database of Porous Rigid Amorphous Materials. Chemistry of Materials 2020, 32 (18), 8020–8033. DOI:https://doi.org/10.1021/acs.chemmater.0c03057

The features and labels for the 75 PIM materials used for this study can be found in the directory: "Dataset_labels". Additionally, the individual and aggregated SHAP scores are located in the SHAP_Results folder.

# Code Availability
The scripts used to predict the isosteric heat of adsorption for PIMs at 298K can be found in the directory "Scripts". All code was written in Python version 3.9.22 using Jupyter Notebooks version 4.4.0. The AI-specific packages can be easily installed with conda-forge. It is highly recommended to create a new conda environment for replicating the study's results. This can be done by:

<pre><code>conda create -n "ENV_NAME" python=3.9.22</code></pre>

Please replace "ENV_NAME" with the desired name of the environment

Below is a list of packages used during this study for ease of replicability:

<pre><code>conda install conda-forge::scikit-learn</code></pre>

<pre><code>conda install conda-forge::scikit-optimize</code></pre>

<pre><code>conda install conda-forge::shap</pre></code>

<pre><code>conda install conda-forge::xgboost</pre></code>

<pre><code>conda install conda-forge::catboost</pre></code>

<pre><code>conda install conda-forge::pandas</pre></code>

<pre><code>conda install conda-forge::numpy</pre></code>



