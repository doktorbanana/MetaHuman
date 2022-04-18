from Variational_Autoencoder_alla_Valerio import VAE
import numpy as np
import os

continue_from = 1000
step_size = 1000
continue_to = continue_from + step_size
subfolder = "1.0_16"
model_name = "VAE_1000samples_20Epochs"

BATCH_SIZE = 32
EPOCHS = 20

autoencoder = VAE.load("data_and_models\\" + model_name)
autoencoder.summary()

x_train = np.load("data_and_models\\" + subfolder + "\spectos.npy")
x_train = x_train[continue_from:continue_to]

autoencoder.train(x_train, BATCH_SIZE, EPOCHS)
autoencoder.save("data_and_models\\" + subfolder + "\\VAE_" + str(autoencoder.num_of_train_data) + "samples_" + str(EPOCHS) + "Epochs")