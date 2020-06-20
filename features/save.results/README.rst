Preserving Results
=====================================================

In this tutorial we show how Korali can be used to store important outputs from the computational model
that otherwise would be lost.

Computational Model
---------------------------

In the model we assign quantity of interest (QoI) to the Korali `Sample`:

.. code-block:: python

    # Store QoI
    d["Apples"]  = a
    d["Bananas"] = b


Execute
---------------------------

In Korali we have to set this flag to store the values of `Apples` and `Bananas`:

.. code-block:: python

    e["File Output"]["Store Samples"] = True

All evaluations can be found in the files in `_korali_results`.


To access the saved result of the first sample of the 8th generation, execute

.. code-block:: python
  
  import json
  with open('gen00000008.json') as f:
    data = json.load(f)

  data["Samples"][0]["Apples"]
  data["Samples"][0]["Bananas"]
