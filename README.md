# Dissertation

This Project is my Thesis for the Computer Science Meng at the University of Bristol. The PDF report includes full details of the project. 

Instructions to run the code can be found below as well as in the report.

The main code file is unet.py which can be run simply with Python and will read the run settings from args.yaml. Setting options detailed below, and an example yaml file is in the repo.

## Run Options

#### log_dir

String. The directory the tensorboard logs are stored in.

#### learning_rate

Float. The (initial) learning rate of the run.

#### momentum

Float. The momentum of the run. This option is ignored if the run uses
ADAM.

#### ADAM

Boolean. Whether to use ADAM instead of the standard SGD optimiser.

#### beta1, beta2

Floats. The parameters for ADAM. Will be ignored if ADAM is False.

#### fixwind

Boolean. If True the wind is converted from a {direction, speed} format
to {northward wind, eastward wind} is carried out during data loading.

#### balancewind

Boolean. This option which ensures the selected data has an even split
in wind directions. Fixwind must be True to use this.

#### batch_size

Integer. The batch size to be used in the network.

#### norm

Boolean. Whether to use batch normalisation in the Double Convolution
operation.

#### epochs

Integer. The number of epochs to train the network for.

#### model

String. The name of the model to use. Only option is UNetSmall
currently.

#### noskip

Boolean. Whether to remove the skip connections that are typically a
part of the U-Net architecture.

#### transconv

Boolean. Whether to use a Transpose Convolutional layer instead of
standard upsampling in the architecture.

#### val_frequency

Integer. How often (in epochs) to pass the validation data through the
network.

#### log_frequency,print_frequency

Integers. How often (in steps) to log and print the Loss respectively.

#### timestep

Integer. The minimum number of hours between two samples that may both
be used in training the network.

#### samples

Integer. The number of samples to use in training and testing the
network. Must be a multiple of ten. The set is later split into 80%
training data and 20% test/validation data.

#### vars

List of strings. Which variables to include in the model training. The
full list is: Wind, PBLH, Pressure, Temperature, Sea_Level_Pressure.

## Output

The code creates a number of files for further use. These are:

-   *groundtruth.npy* The ground truth for the test data. The *.npy*
    format was chosen for its efficiency and ease of loading back into
    numpy arrays for evaluation and visualisation.

-   *predictions.npy* The final model's predictions for the test data.
    The file *show_results.py* can be run with this and the previous
    file as inputs to visualise the ground truth and predictions side by
    side for visual assessment.

-   *state_dict_model.pt*. The final trained model. The file
    *run_model.py* can be used to pass any set of input points through
    the trained model after the initial training process is done.

-   *variance.npy*. A file containing the variance of ground truth
    across the dataset used to train the model.

-   *timesused.csv*. A file containing all the dates and times used in
    the training and testing of the model.

-   *min.pkl, max.pkl*. Serialised python files containing the
    dictionary of the minimum and maximum values observed in the data
    respectively.

-   A log folder. The folder contains logs of the test and training Loss
    during the training process and can be visualised using tensorboard.
    
#Auxiliary Code

## Passing New Data Through a Trained Network

For the purposes of assessing the performance of the network, it is
important to be able to \"pass\" new points through the network. The
file \"newpointcalc.py\" is a simple piece of code to do this. It takes
two command line arguments, one flagged with \"--folder\" which defines
which subfolder of the current directory the information should be
loaded from, and one flagged with \"--num\" which determines how many
points will be loaded.

The code then loads the arguments file, the trained network, and the min
and max pickle files stored in the named folder, and calls the loading
function to load the requested number of points. The points are loaded
in order from any files in the met and footprints folders. These are
normalised based on the min and max loaded from the folder to ensure
consistency between the new data and the data the model was trained on.
Once the data has been prepared, it is passed through the trained
network, and the results and corresponding ground truth are stored into
the files \"predictions.npy\" and \"groundtruth.npy\" respectively.
Then, the time execution started, the time loading the data finished,
the time the data was done passing through the network, and the time
execution finished are printed to the console for performance
assessment.

## Plotting Pairs of Predictions and Ground Truths

The file plotresults.py plots pairs of predictions and ground truths,
reading them from a folder specified through the command line argument
\"--folder\". There are three pairs per row and three rows per plot, and
the code will produce sets of plots until it runs out of data to read.
The files in the folder must be called 'predictions.npy' and
'groundtruth.npy'. The images in Appendix [\[appendixb\]][1] were
produced with this code.

## Getting Evaluation Metric Averages and Statistics

In order to evaluate the performance of the network, a set of evaluation
metrics are utilised. It takes two command line arguments, one flagged
\"--folder\" and one flagged \"--folder2\". The first is compulsory,
while the second is optional. If only the first is provided, the code
will read the predictions and ground truth from the indicated folder,
and calculate and print the average value of each metric. If both
arguments are provided, the code will calculate the average of each
metric for each folder and print them, and will additionally perform a
Mann-Whitney test to determine whether the two sets of numbers for each
metric have a significantly different median.

## Getting and Plotting Correlation Between Variables and Error Metrics

As part of the evaluation, the correlation between each input variable
and each evaluation metric is examined. Two pieces of code were written
to achieve this:

-   *getcorr.py*, which reads the timesused.csv file of a run and,
    iterating through the reversed list of time points, it finds each
    point in the MET data, calculates the average for each variable, and
    calculates the value of each evaluation metric using the prediction
    and ground truth. This is done in reverse order as the timesused.csv
    file stores the training data, followed by the test data, but only
    the test data is of interest for evaluation. Once the desired number
    of points has been covered, the gathered data is saved in a python
    'pkl' file, which compresses it.

-   *plotcorr.py*, which reads in the pkl file, calculates the
    correlation matrix of Variables vs Metrics, and plots a heatmap. The
    code for plotting the heatmap is based on [@docs]. This file will
    automatically remove any NaN values in the pkl files before
    calculating the correlation matrix. One such value was observed
    which was due to an error in the saving of the data so this was
    added as a failsafe.

## Kruskal Test

The file reads a list of folder names from the command line and performs
a Kruskal test, the non parametric equivalent of ANOVA, on the MSE of
the predictions in each folder. The test determines whether the
differences in the provided samples depend on a variable which is
assumed to change by a step between every sample and the next.

## Plotting the Loss

The file *plotloss.py* reads a list of folder names from the command
line and two boolean variables '--train' and '--test' and plots the Test
and/or Train Loss based on CSV files stored in the named folders. The
CSVs must be named \"test.csv\" and \"train.csv\" in each folder, and
they can be downloaded through the tensorboard interface.

## Calculating and Plotting the Estimated Gas Concentration

As part of the model's evaluation, the predicted footprints are
multiplied with an emissions field described in Section 2.2.3 to
get a prediction of the amount of target gas present at the target
location. The results are then compared with the results of performing
the same operation but with the ground truth footprints instead of the
predicted ones.

The file *estimates.py* performs these calculations and prints the
average error, average estimated value, average ground truth value,
average percentage error, and the correlation coefficient between the
estimates and ground truth. It also plots a scatterplot of the two sets
of data against each other.

## Plotting Histograms of Predicted and Correct Footprint Velues

The file *plothistograms.py* reads a folder name from the command line
and creates a histogram of the values in the ground truth and the
predicted footprints to compare the two distributions. A
Kolmogorov--Smirnov test is also performed to assess the similarity of
the two distributions.
