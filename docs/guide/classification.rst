.. _guide_classification:

==============
Classification
==============

You can create maps of discrete land cover classes using the following steps:

1. Train example model segments using :ref:`yatsm train <yatsm_train>`.
2. Classify all of the model segments using :ref:`yatsm classify <yatsm_classify>`.
3. Create a land cover map using :ref:`yatsm map <yatsm_map>`.

Training
_________

In this step, you can make use of any `scikit-learn` compatible estimator to train a classifier object using the multiple coefficients stored in the YATSM results. For this you need to specify a raster file that contains the land cover categories to be used. These must be collected over a fixed time period and in areas where the land cover types remain stable. The start and end dates must be specified in the YAML model configuration file. An additional file containing the classifier configuration is required, in which you need to specify the algorithm to be used and its corresponding hyperparameters. 

Classification
______________

You can classify all of the records in the YATSM saved files using the trained algorithm resulting from the previous step. After this operation is performed, each record will be assigned a land cover type label and a prediction probability.


Mapping
_______
Finally, in this step you can generate categorical maps of the land cover classes for any given date, and you can includea prediction probability band to assess its reliability. This utility also lets you create maps of the model coefficientsand other derived information, as described in :ref:`Mapping derived information <guide_map_static>`. 



Examples
========

Consider this example workflow:

1. Train a classifier object using a RandomForest algorithm and run the K-fold diagnostics

.. code-block:: bash

    $ yatsm train --diagnostics p013r030.yaml RandomForest.yaml train.pkl

2. Use this trained algorithm to classify all the saved YATSM records. Split the process in 100 jobs::
     
    njob=100

    for job in $(seq 1 $njob); do
        yatsm classify p013r030.yaml RandomForest.yaml $job $njob
    done

3. Create a categorical land cover type map:
.. code-block:: bash

    $ yatsm map --root $ts_path/images --result $ts_path/Results --image $ts_path/images/example_img
        class 2014-01-01 Class_20014-01-01.tif


An example template of the parameter file is located within
``examples/p013r030/p013r030.yaml``:

.. literalinclude:: ../../examples/p013r030/p013r030.yaml
   :language: yaml 

An example of a classifier configuration file is located in
 ``examples/classifiers/RandomForest.yaml``:

.. literalinclude:: ../../examples/classifiers/RandomForest.yaml

TODO
====

- Include ancillary information (e.g. DEM) in order to improve the classification.
- Allow storing multiple land cover type labels in the YATSM records after using different training algorithms.

