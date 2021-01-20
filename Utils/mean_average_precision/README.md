# mAP: Mean Average Precision

The package is extended from https://github.com/bes-dev/mean_average_precision for the purposes of calculate:

<ul>
  <li> TP - True Positive which measures the amounts that the model <b>correctly</b> predicts macro- or micro-expression samples.  </li>
  <li> FP - False Positive which measures the amounts that the model <b>incorrectly</b> predicts macro- or micro-expression samples  </li>
  <li> FN - False Negative which measures the amounts of macro or micro-expressions that are not predicted by the model.  </li>
  <li> AP - Average Precision for Intersection over Union (IoU) in the range of [.5:.95] is used in this project to measure the quality of predicted samples. </li>
</ul>
