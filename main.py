import sys
from load_images import *
from load_label import *
from extraction_preprocess import *
from training import *

# Define the dataset and expression to spot
dataset_name = 'CASME_sq' # Specify CASME_sq or SAMMLV only
expression_type = 'micro-expression' ### Specify micro-expression or macro-expression only

print('------ Spotting', dataset_name, expression_type, '-------')

# Load Images
print('\n ------ Croping Images ------')
#Can comment this out after completed on the dataset specified and intend to try on another expression_type
crop_images(dataset_name) 
print("\n ------ Loading Images ------")
images, subjects, subjectsVideos = load_images(dataset_name)

# Load Ground Truth Label
print('\n ------ Loading Excel ------')
codeFinal = load_excel(dataset_name)
print('\n ------ Loading Ground Truth From Excel ------')
final_images, final_videos, final_subjects, final_samples = load_gt(dataset_name, expression_type, images, subjectsVideos, subjects, codeFinal) 
print('\n ------ Computing k ------')
k = cal_k(dataset_name, expression_type, final_samples)

# Feature Extraction & Pre-processing
print('\n ------ Feature Extraction & Pre-processing ------')
dataset = extract_preprocess(final_images, k)

# Pseudo-labeling
print('\n ------ Pseudo-Labeling ------')
pseudo_y = pseudo_labeling(final_images, final_samples, k)

# LOSO
print('\n ------ Leave one Subject Out ------')
X, y, groupsLabel = loso(dataset, pseudo_y, final_images, final_samples, k)

# Model Training & Evaluation
print('\n ------ SOFTNet Training & Testing ------')
TP, FP, FN, metric_fn = training(X, y, groupsLabel, dataset_name, expression_type, final_samples, k, dataset)
final_evaluation(TP, FP, FN, metric_fn)


