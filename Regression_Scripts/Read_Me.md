This folder contains the scripts used in this study to benchmark various AI algorithms for predicting the isosteric heats of adsorption of PIMs with CO<sub>2</sub>. The code is saved as .py files and can be used by these commands:  


<pre><code>conda activate "ENV_NAME"</code></pre>

<pre><code>python "FILE_NAME.py" python=3.9.22</code></pre>

The workflows for the various machine learning models are largely identical, and follow the same procedural preprocessing and handling of the materials' features. Below the scripts imports, a new directory is created that stores the hyperparameters and model results. This helps organize each model's data for ease on the user. Next, the features are defined that will be used to represent each structure for the algorithms. The labels used for each material were determined using Grand Connonical Monte Carlo Simulations by Drs. Raghuram Thyagarajan and David Sholl in their work: https://doi.org/10.1021/acs.chemmater.0c03057. Additionally, the isosteric heats of adsorption used are specified for carbon dioxide gas at 298K. 

Scikit Learn's MinMaxScaler was used to normalize the features, so that they had comparable magnitudes. Bayesian optimization was used to determine the optimal hyperparameters for each model. This was based on the average testing partitions' coefficients of determination. More specifically, due to skopt seeking to minimize the objective function, we return the negative average of the testing R<sup>2</sup> scores. This inverts the performance metric landscape and converts the desired maximization to match skopt's format. The training/testing construction used was chosen based on the limited dataset. Varying the KFold random state or the amount of partitions resulted in a model whose results are comparable to those reported in Table XXX of this study. Section XX of this study's Supporting Information contains the performance metrics for altering the KRR (rbf)'s training/testing partitions and its random state seed.    



