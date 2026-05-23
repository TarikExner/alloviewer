Clusterers and Clustering
=========================

Clusterer interface and prediction result
-----------------------------------------

.. autoclass:: alloviewer.flow_cytometry.gating.clusterers.GenericPrediction
   :members:
   :show-inheritance:

.. autoclass:: alloviewer.flow_cytometry.gating.clusterers.BaseClusterer
   :members:
   :show-inheritance:


Clusterer adapters
------------------

.. autoclass:: alloviewer.flow_cytometry.gating.clusterers.FlowSOMClusterer
   :members:
   :show-inheritance:

.. autoclass:: alloviewer.flow_cytometry.gating.clusterers.PARCClusterer
   :members:
   :show-inheritance:

.. autoclass:: alloviewer.flow_cytometry.gating.clusterers.HDBSCANClusterer
   :members:
   :show-inheritance:


Factories
---------

.. autofunction:: alloviewer.flow_cytometry.gating.clusterers.HDBSCANFactory

.. autofunction:: alloviewer.flow_cytometry.gating.clusterers.FlowSOMFactory

.. autofunction:: alloviewer.flow_cytometry.gating.clusterers.PARCFactory


Clustering functions
--------------------

.. autoclass:: alloviewer.flow_cytometry.gating.clustering.ClusterPrediction
   :members:
   :show-inheritance:

.. autofunction:: alloviewer.flow_cytometry.gating.clustering.fit_clustering

.. autofunction:: alloviewer.flow_cytometry.gating.clustering.compute_outlier_threshold

.. autofunction:: alloviewer.flow_cytometry.gating.clustering.build_feature_blocks

.. autofunction:: alloviewer.flow_cytometry.gating.clustering.build_training_pool
