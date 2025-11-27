import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model , Model

# - - - - CONFIGURATION - - - -
BASE_DIR = os.path.expanduser("/home/kali/Documents/Block_cipher")
SPLIT_DATA_PATH = os.path.join(BASE_DIR , "Split_data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_models")
FEATURE_SAVE_PATH = os.path.join(SPLIT_DATA_PATH, "gnn_features")

os.makedirs(FEATURE_SAVE_PATH , exist_ok = True)

LSTM_MODEL_PATH = os.path.join(MODEL_SAVE_PATH , "lstm_model.h5")


print("Starting feature extraction----\n")

# 1. Load All Padded Sequence Data (X_seq_padded_all)
# We need the sequence data for ALL 114 SAMPLES (train + val)

print("Loading full padded sequence data from data_preprocessor.py file ...\n")

try:
    with open(os.path.join(SPLIT_DATA_PATH, "lstm_train_val.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    # Concatenate the training and validation padded sequences to get the full dataset
    X_train_seq = np.array(data['X_train_seq'])
    X_val_seq = np.array(data['X_val_seq'])
    
    # IMPORTANT: Ensure the shapes match for concatenation
    X_seq_padded_all = np.concatenate([X_train_seq, X_val_seq], axis=0)
    print(f"Total samples loaded: {X_seq_padded_all}\n")
    print(f"Total samples loaded: {X_seq_padded_all.shape[0]}\n")

except FileNotFoundError:
    print("ERROR: LSTM data file (lstm_train_val.pkl) not found. Check path and ensure data_preprocessor.py was run.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()
    
    
# 2. Load the Trained LSTM Model
print(f"Loading trained LSTM model from {LSTM_MODEL_PATH}...\n")
try:
    # Adding compile=False to handle potential custom objects, although it might not be strictly needed here.
    full_lstm_model = load_model(LSTM_MODEL_PATH, compile=False) 
except Exception as e:
    print(f"ERROR: Could not load model. Ensure training completed successfully and file exists.\nDetails: {e}")
    exit()
    
    
# 3. Truncate Model to Create Feature Extractor
# We want the output of the Dropout layer (index -2), which is the 128-D feature vector.
# The final layer (index -1) is the Dense Softmax layer (9 outputs).
# The second to last layer (index -2) is the Dropout layer (128 outputs).

try:
    # Get the output tensor from the target layer
    feature_layer_output = full_lstm_model.layers[-2].output

    # Create the feature extraction model with the same inputs but the truncated output
    feature_extractor_model = Model(
        inputs=full_lstm_model.inputs,
        outputs=feature_layer_output
    )
    print("Created feature extractor model successfully (outputting 128-D vector).")
    # feature_extractor_model.summary() # Optional: Uncomment to check the model layers

except IndexError:
    print("ERROR: Model layers indexing failed. Check if your LSTM model has at least 2 layers (LSTM/BiLSTM + Dropout + Dense).")
    exit()
except Exception as e:
    print(f"An error occurred during model truncation: {e}")
    exit()

# 4. Generate Sequence Features (X_seq_features)
print(f"Generating sequence features for all {X_seq_padded_all.shape[0]} samples...\n")
# The prediction output will be the (N, 128) feature matrix
X_seq_features = feature_extractor_model.predict(X_seq_padded_all)
print(f"SUCCESS: Generated sequence features array shape: {X_seq_features.shape}\n")


# 5. Save the Features
SAVE_FILE = os.path.join(FEATURE_SAVE_PATH, "X_seq_features_128D.pkl")
with open(SAVE_FILE, 'wb') as f:
    pickle.dump(X_seq_features, f)
print(f"\nSUCCESS: LSTM features extracted and saved to {SAVE_FILE}")

print("Proceed now to the gnn_trainer.py script!")











