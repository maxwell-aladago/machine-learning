import numpy as np

def q1_mse(pred_Y, correct_Y):
# This function calculates the Mean Squared Error given two sets of output
# values, one set corresponding to the correct values, the other set
# representing the output values predicted by a regression model
# INPUT:
#  pred_Y: a numpy.ndarray vector of type 'float' containing m predicted values
#  correct_Y: a numpy.ndarray vector of type 'float' containing m correct values
#
# OUTPUT:
#  err: 'float' representing the Mean Squared Error
#

      residual = correct_Y - pred_Y
      err = np.dot(np.transpose(residual), residual) / residual.size
      
      return err
