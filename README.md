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

# Batch size used for fine-tuning
bs_EEGSym = 32  # bs_EEGSym = 256 for pre-training (To reduce compute time)

# Load the X EEG features and Y labels of your choiche in the followin format:
# X = Features in the form of [examples, samples, ncha, 1]
# Y = one_hot encoded labels, i.e., [examples, 1, 1]

# Divide the features into training, validation and test to obtain:
# X_train, Y_train, X_validate, Y_validate, X_test, Y_test

# Select if the pre-trained weight values on 5 datasets that include 280  
# users are loaded from these electrode configurations:
# 8 Electrode configuration: ['F3', 'C3', 'P3', 'Cz', 'Pz', 'F4', 'C4', 'P4']
# 16 Electrode configuration: ['F7', 'F3', 'T7', 'C3', 'P7', 'P3', 'O1', 'Cz', 
#             'Pz', 'F8', 'F4', 'T8', 'C4', 'P8', 'P4', 'O2']
pretrained = True  # Parameter to load pre-trained weight values

# Select the number of channels of your application
ncha = 8

# Initialize the model
hyperparameters = dict()
hyperparameters["ncha"] = channels
hyperparameters["dropout_rate"] = 0.4
hyperparameters["activation"] = 'elu'
hyperparameters["n_classes"] = 2
hyperparameters["learning_rate"] = 0.0001  # 1e-3 for pretraining and 1e-4 
# for fine-tuning
hyperparameters["fs"] = 128
hyperparameters["input_time"] = 3000
hyperparameters["scales_time"] = np.tile([125, 250, 500], 1)
hyperparameters['filters_per_branch'] = 24
hyperparameters['ch_lateral'] = (ncha - 2) / 7  # 3/7 For 8/16 electrodes 
# respectively. For the configurations present in the publication.
model_hyparams['residual'] = True
model_hyparams['symmetric'] = True

model = EEGSymv2(**model_hyparams)
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

# Train the model
fittedModel = model.fit(trial_iterator(X_train, Y_train, 
                        batch_size=bs_EEGSym, shuffle=True, 
                        augmentation=augmentation), 
                        steps_per_epoch=X_train.shape[0] / bs_EEGSym, 
                        epochs=500, validation_data=(X_validate, Y_validate), 
                        callbacks= [early_stopping])

# Obtain the accuracies of the trained model
probs_test = model.predict(X_test)
pred_test = probs_test.argmax(axis=-1)
accuracy = (pred_test == Y_test.argmax(axis=-1)) 

# Optional: Store the weights
model.save_weights("\cutom_path")   
```
