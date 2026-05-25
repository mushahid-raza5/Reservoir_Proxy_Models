# AI-Driven Reservoir Performance Proxy
### Group: Learning from Machine

This repository contains the codebase for our Final Project in Machine Learning 2. 

## Project Structure
The code should be evaluated in the following order to understand our progression from classical machine learning to physics-informed deep learning:

1. **ML2_FinalProject_Random_Forest_Baseline.ipynb**: Establishes the performance baseline using classical ensemble methods.
2. **ML2_FinalProject_MLPModel.ipynb**: Introduces deep learning (Feedforward Neural Network) with Huber Loss for scalar predictions.
3. **ML2_FinalProject_LSTM_Model.ipynb**: Extends the proxy to handle sequence modeling (22-year curves) with variable time-steps using an Encoder-Decoder architecture and Loss Weighting.
4. **ML2_FinalProject_Physics_Informed_NN.ipynb**: Implements a custom 	f.GradientTape training loop to enforce Darcy's Law and injection ceilings, ensuring the proxy obeys thermodynamic laws.

## Installation & Requirements
To run these notebooks, please install the dependencies using the provided 
equirements.txt file:

`ash
pip install -r requirements.txt
`

## Documentation
Please refer to the enclosed PDF report and presentation for our detailed findings, metrics, and methodology.
