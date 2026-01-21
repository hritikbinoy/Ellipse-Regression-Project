The folder kria-kv260-build is used to build the dataflow build for the kria board using finn docker.

The notebook under the folder has to run in FINN docker only to get the results. use the output of notbook 2b-finn-verify-and-export as the input of this notebook, ie, ellipse_regression_hw_ready.onnx. Copy this model to the docker as well, so that the notebookacan load it when wwe run the notebook.

Inorder to get the bitfile as the result, which is used for hardware deployment, Vivado, Vitis, and Vitis HLS has to be mounted form the host system to the FINN docker environment.

The folders and the files you see under the kria-kv260-build folder are copied from the FINN docker after the build.
Apart form this, there are many other files and folders, which you will get after the build and everything has not copied to this folder. 

what we have here is only taken for the deployement, and if you need all the files then ypu have to run the build.

(Apart from this, the other files include, driver folder, intermediate_model folder and some otehr json files)