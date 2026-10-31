import joblib
import numpy as np
import pandas as pd
from stacking.neural import NumpyMLP

def test_portable_neural_inference_and_serialization(tmp_path):
    # An explicitly known ReLU network verifies standardization, neuron layout,
    # bias, zero activation and portability without importing training PyTorch.
    model=NumpyMLP(np.array([1.,2.,3.]),np.array([2.,2.,2.]),
        [np.array([[1.,0.,0.],[0.,1.,0.]],dtype=np.float32),np.array([0.,0.],dtype=np.float32),
         np.array([[2.,-1.]],dtype=np.float32),np.array([.5],dtype=np.float32)],10)
    x=pd.DataFrame([[5.,4.,3.],[-1.,0.,3.]],columns=['A','B','C'])
    np.testing.assert_allclose(model.decision_function(x),[3.5,.5])
    path=tmp_path/'neural.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).decision_function(x),model.decision_function(x))
