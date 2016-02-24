.. _guide_classification:

==============
Classification
==============

Maps of discrete land cover classes can be created using the following steps:

1. Train example model segments using :ref:`yatsm train <yatsm_train>`.
2. Classify all of the model segments using :ref:`yatsm classify <yatsm_classify>`.
3. Create a land cover map using :ref:`yatsm map <yatsm_map>`.

Training
_________

In this step, a `scikit-learn` compatible classifier is trained on YATSM output and then saved to a file. The training data must be supplied as a raster file containing the land cover categories to be used. These must be collected over a time period where the observed land cover type remains stable. 

Classification
______________

In this step, all of the records of the YATSM saved files are classified using the file resulting from the training that was performed previously.


Mapping
_______
In this step, the information on land cover class that is now stored in the YATSM result files is used to generate a categorical map of these classes on a specific date. A probability map can also be created.



Examples
========

TODO
====

- Include ancillary information
- (add other topics to potential roadmap)

