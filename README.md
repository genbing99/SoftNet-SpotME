# Shallow Optical Flow Three-Stream CNN For Macro and Micro-Expression Spotting From Long Videos

## Framework of Proposed SOFTNet approach
Overall framework: </br></br>
<img src='images/framework.PNG' width=900 height=400>

Mainly four phases involved: 
<ul>
<li> Feature Extraction - Extract the optical flow features (u, v, ε) that represents each frame. </li>
<li> Pre-processing - Remove global head motion, eye masking, ROI selection, and image resampling. </li>
<li> SOFTNet - Three-stream shallow architecture that takes inputs (u, v, ε) and outputs a spotting confidence score. </li>
<li> Spotting - Smoothing spotting confidence score, then perform thresholding and peak detection to obtain the spotted interval for evaluation. </li>
</ul>

## Training
Tensorflow and keras are used in the experiment. Two dataset with macro- and micro-expression are used for training and testing purposes:

CAS(ME)<sup>2</sup> - http://fu.psych.ac.cn/CASME/cas(me)2-en.php

SAMM Long Videos - http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php

## Results
### Evaluation
Comparison between the proposed approaches against baseline and state-of-the-art approaches in Third Facial Micro-Expression Grand Challenge (MEGC 2020) in terms of F1-Score:
<img src='images/result.PNG' width=900 height=200>

### Visualization
Samples visual results for SOFTNet: </br></br>
<img src='images/teaser.PNG' width=500 height=200>

## Discussion
The proposed SOFTNet approach outperforms other methods on CAS(ME)<sup>2</sup> while ranked second on SAMM Long Videos. To better justify the effectiveness of SOFTNet approach, we experimented a similar framework but without SOFTNet, the results show that the framework with SOFTNet is much more efficient overall.

Visually, SOFTNet activation units shows our intuition to concatenate the optical flow features (u, v, ε) from three-stream. The spatio-temporal motion information are captured when macro and micro-expression occur. After the concatenation, the action unit 4 (Brow Lower) is triggered when a disgust emotion elicited. 

## Reproduce the results for SOFTNet approach
The complete code is shown the in Jupyter Notebook script for reader to have a better understanding.

<b>Step 1)</b> The datasets, CAS(ME)<sup>2</sup> (CASME_sq) and SAMM Long Videos (SAMMLV) obtained are required to placed in the structure as follows:
>├─Extraction_Preprocess <br>
>├─SOFTNet_Spotting <br>
>├─SOFNet_Weights <br>
>├─Utils <br>
>├─CASME_sq <br>
>>├─CAS(ME)^2code_final.xlsx <br>
>>├─cropped <br>
>>├─rawpic <br>
>>├─rawvideo <br>
>>└─selectedpic <br>

>├─SAMMLV <br>
>>├─SAMM_longvideos <br>
>>└─SAMM_LongVideos_V1_Release.xlsx <br>

<b>Step 2)</b> Feature Extraction and Pre-processing

<blockquote> Open the Extraction_Preprocess.ipynb and run the codes follow the instruction given inside. </blockquote>

<b>Step 3)</b> SOFTNet and Spotting

<blockquote> Open the SOFTNet_Spotting.ipynb and run the codes follow the instruction given inside. The evaluation for TP, FP, FN, F1-Score is returned at the last piece of code. </blockquote>

### Note for pre-trained weights
The pre-trained weights for CAS(ME)<sup>2</sup >and SAMM Long Videos with macro and micro-expression separately are located under folder SOFTNet_Weights. You may load the weights at SOFTNet_Spotting.ipynb for evaluation. However, the result is slightly different with the result given in table shown above.





