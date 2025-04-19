# Singapore Total Monthly Rainfall Prediction 
Deep Learning Project Group 6

libraries required:
- numpy
- pandas
- torch
- sklearn

Our LSTM model has the best performance on the test set.

Training model files (just run all to get the model files):
- rnn.ipynb
- lstm.ipynb
- transformer_encoder.ipynb

Model Weights after training
- rnn_model.pth
- lstm_model.pth
- transformer_encoder_model.pth

Cells for loading and running best model weights for each model are indicated
with the markdown cell comment "Display losses for best model weights"

Best Model weights checkpoint
- models_cp/rnn_model-100.pth
- models_cp/lstm_model-100.pth
(No checkpoints for transformer encoder, only best weights saved
to transformer_encoder_model.pth)
