import os
from Variational_Autoencoder_alla_Valerio import VAE
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as plt

subfolder = "2.0_128"
load_path = os.path.join("D:\Daten\Studium\Semester_8\MetaHuman\MetaHuman\data_and_models\\2.0_128", "spectos_LedZep.npy")
x_train = np.load(load_path)
x_train, x_test, _, _ = train_test_split(x_train, x_train, test_size=0.2)
print("loaded data")

path = os.path.join("data_and_models/2.0_128", "VAE_LedZep128D_25140samples_20Epochs")
autoencoder = VAE.load(path)

"""
---------------------
Train the VAE
---------------------
"""

LEARNING_RATE = 0.0005
BATCH_SIZE = 128
EPOCHS = 20

#autoencoder.compile_model(LEARNING_RATE)
steps = 10
history = []
val_history = []

for i in range(steps):
    num = int(x_train.shape[0] / steps) * (i + 1)
    test_num = int(x_test.shape[0] / steps) * (i + 1)

    print("Start with subset " + str(i+1) + " of " + str(steps))
    print("Train from index " + str(int(num - (num / (i + 1)))) + " to index " + str(num))
    print("Use test indices " + str(int(test_num - (test_num / (i + 1)))) + " to " + str(test_num) +
          " as validation set")

    train_subset = x_train[int(num - (num / (i + 1))):num]
    test_subset = x_test[int(test_num - (test_num / (i + 1))):test_num]

    step_history = autoencoder.train(train_subset, test_subset, BATCH_SIZE, EPOCHS)
    history.extend(step_history.history['loss'])
    val_history.extend(step_history.history['val_loss'])

    """
    ----------------
    Save VAE
    ----------------
    """

    save_path = os.path.join("data_and_models", subfolder)
    name = "VAE_beatles" + str(autoencoder.latent_space_dim) + "D_" + \
           str(int(autoencoder.num_of_train_data / 2)) + "samples_" + str(20+EPOCHS) + "Epochs"
    model_path = os.path.join(save_path, name)
    autoencoder.save(model_path)

    print("saved at: " + save_path)


"""
Save the History
"""
history = np.asarray(history)
val_history = np.asarray(val_history)

hist_save_path = os.path.join(save_path, "history_beatles.npy")
val_hist_save_path = os.path.join(save_path, "val_history_beatles.npy")



with open(hist_save_path, 'wb') as f:
    np.save(f, history)
    np.save(f, val_history)

# with open(hist_save_path, 'rb') as f:
#     history = np.load(f)
#     val_history = np.load(f)

plt.figure()
plt.plot(history)
plt.xlabel("Epochs")
plt.xlabel("Loss")
plt.title("Loss History")
plt.show()

plt.figure()
plt.plot(val_history)
plt.xlabel("Epochs")
plt.xlabel("Validation Loss")
plt.title("Validation Loss History")
plt.show()