# EEGSym
Open implementation and code from the publication "EEGSym: Overcoming 
Intersubject Variability in Motor Imagery Based BCIs With Deep Learning". [1]

[1] S. Pérez-Velasco, E. Santamaría-Vázquez, V. Martínez-Cagigal, 
D. Marcos-Martínez and R. Hornero, "EEGSym: Overcoming Inter-subject 
Variability in Motor Imagery Based BCIs with Deep Learning," in IEEE 
Transactions on Neural Systems and Rehabilitation Engineering, 2022, 
doi: https://doi.org/10.1109/tnsre.2022.3186442

## Architecture details
![EEGSym architecture details](https://github.com/Serpeve/EEGSym/blob/main/EEGSym_scheme_online.png?raw=true)

## Example of use:
```
from EEGSym_architecture import EEGSym
from EEGSym_DataAugmentation import trial_iterator
from tensorflow.keras.callbacks import EarlyStopping as kerasEarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np

#%% Load signal and model
# Batch size used for fine-tuning
bs_EEGSym = 32  # bs_EEGSym = 256 for pre-training (To reduce compute time)

# Select if the pre-trained weight values on 5 datasets that include 280
# users are loaded from these electrode configurations:
# 8 Electrode configuration: ['F3', 'C3', 'P3', 'Cz', 'Pz', 'F4', 'C4', 'P4']
# 16 Electrode configuration: ['F7', 'F3', 'T7', 'C3', 'P7', 'P3', 'O1', 'Cz',
#             'Pz', 'F8', 'F4', 'T8', 'C4', 'P8', 'P4', 'O2']
pretrained = True  # Parameter to load pre-trained weight values

# Select the number of channels of your application (8/16 for pretrained=True)
ncha = 8

# Load the X EEG features and Y labels of your choice in the following format:
# X = Features in the form of [examples, samples, ncha, 1]
examples = 1000
samples = 128 * 3  # frequency = 128 Hz * time= 3 seconds
# Random signal for demonstrative purposes
X = np.random.normal(loc=0, scale=1, size=[examples, samples, ncha, 1])
# Y = one_hot encoded labels, i.e., [examples, 1, 1]
Y = np.random.randint(low=0,high=2,size=examples)
Y = to_categorical(Y)


# Divide the features into training, validation and test to obtain:
# X_train, Y_train, X_validate, Y_validate, X_test, Y_test
X_train = X[:int(0.6*examples)]
Y_train = Y[:int(0.6*examples)]
X_validate = X[int(0.6*examples):int(0.8*examples)]
Y_validate = Y[int(0.6*examples):int(0.8*examples)]
X_test = X[int(0.8*examples):]
Y_test = Y[int(0.8*examples):]

#%% Initialize the model
hyperparameters = dict()
hyperparameters["ncha"] = ncha
hyperparameters["dropout_rate"] = 0.4
hyperparameters["activation"] = 'elu'
hyperparameters["n_classes"] = 2
hyperparameters["learning_rate"] = 0.0001  # 1e-3 for pretraining and 1e-4
# for fine-tuning
hyperparameters["fs"] = 128
hyperparameters["input_time"] = 3*1000
hyperparameters["scales_time"] = np.tile([125, 250, 500], 1)
hyperparameters['filters_per_branch'] = 24
hyperparameters['ch_lateral'] = int((ncha / 2) - 1)  # 3/7 For 8/16 electrodes
# respectively. For the configurations present in the publication.
hyperparameters['residual'] = True
hyperparameters['symmetric'] = True

model = EEGSym(**hyperparameters)
model.summary()

# Select if the data augmentation is performed
augmentation = True  # Parameter to activate or deactivate DA

# Load pre-trained weight values
if pretrained:
    model.load_weights('EEGSym_pretrained_weights_{}_electrode.h5'.format(ncha))

# Early stopping
early_stopping = [(kerasEarlyStopping(mode='auto', monitor='val_loss',
                   min_delta=0.001, patience=25, verbose=1,
                   restore_best_weights=True))]
#%% OPTIONAL: Train the model
if pretrained:
    for layer in model.layers[:-1]:
        layer.trainable = False

fittedModel = model.fit(trial_iterator(X_train, Y_train,
                        batch_size=bs_EEGSym, shuffle=True,
                        augmentation=augmentation),
                        steps_per_epoch=X_train.shape[0] / bs_EEGSym,
                        epochs=500, validation_data=(X_validate, Y_validate),
                        callbacks=[early_stopping])

#%% Obtain the accuracies of the trained model
probs_test = model.predict(X_test)
pred_test = probs_test.argmax(axis=-1)
accuracy = (pred_test == Y_test.argmax(axis=-1))

#%% Optional: Store the weights
model.save_weights("\custom_path")
```
