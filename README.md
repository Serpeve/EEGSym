# EEGSym
Open implementation and code from the publication "EEGSym: Overcoming 
Intersubject Variability in Motor Imagery Based BCIs With Deep Learning".

## Architecture details
![EEGSym architecture details](https://github.com/Serpeve/EEGSym/blob/main/EEGSym_scheme_online.png?raw=true)

## Example of use:
```
from EEGSym_architecture import EEGSym
from EEGSym_DataAugmentation import trial_iterator
from tensorflow.keras.callbacks import EarlyStopping as kerasEarlyStopping

# Batch size used for training
bs_EEGSym = 32

# Load the X EEG features and Y labels of your choiche in the followin format:
# X = Features in the form of [examples, samples, ncha, 1]
# Y = one_hot encoded labels, i.e., [examples, 1, 1]

# Divide the features into training, validation and test to obtain X_train, 
# Y_train, X_validate, Y_valicate, X_test, Y_test


# Initialize the model
hyperparameters = dict()
hyperparameters["ncha"] = channels
hyperparameters["dropout_rate"] = 0.4
hyperparameters["activation"] = 'elu'
hyperparameters["n_classes"] = 2
hyperparameters["learning_rate"] = 0.001  #0.001 for pretraining
hyperparameters["fs"] = 128
hyperparameters["input_time"] = 3000
hyperparameters["scales_time"] = np.tile([125, 250, 500], 1)
hyperparameters['filters_per_branch'] = 24
hyperparameters['ch_lateral'] = 3  # 3/7 For 8/16 electrodes respectively
model_hyparams['residual'] = True
model_hyparams['symmetric'] = True

model = EEGSymv2(**model_hyparams)
model.summary()

# Select if the data augmentation is performed
augmentation = True  # Parameter to activate or deactivate DA

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
