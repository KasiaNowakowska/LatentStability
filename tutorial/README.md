# Tutorial

Running 1.-3., which consists of training and testing the architecture, requires ~20min with a dedicated GPU.

The structure is organised as follows:

1. train_cae.ipynb (i) trains the convolutional autoencoder (implemented in torch), (ii) shows the performance in the test set of the reconstruction and (iii) saves the encoded data for the ESN to be trained on. (GPU recommended)

2. validate_esn.ipynb (i) trains the echo state network to predict the latent space dynamics and (ii) shows the performance of the CAE-ESN in predicting the latent dynamics in the test set. (CPU)

3. calculate_LEs.ipynb shows (i) the calculation of Lyapunov exponents (ii) the procedure for how to calculate CLVS and the corresponding angles.

Comments are provided within each script.

We note here that calculation CLVs is resource and time-intensive procedure. We outline how to calculate them but we recommend writing a seperate script to calculate them. There is the option to callculate LEs and CLVs significantly faster using Google JAX on GPU. Pleaser reach out to @eliseoe directly for this. 