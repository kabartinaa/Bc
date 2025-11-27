#data loading
import os
import pickle
import numpy as np

from tensorflow.keras.optimizers import Adam # Added for Learning Rate control

#data spliting and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.regularizers import l2 # <-- NEW IMPORT
from tensorflow.keras.layers import Input

#deep learning model
from tensorflow.keras.models import Sequential # to stack layers
from tensorflow.keras.layers import Embedding , LSTM , Dense , Dropout , Bidirectional
#Embedding : Converts opcode tokens into dense vector representations
#LSTM : Learns sequential dependencies in opcode sequences
#Dense : Fully connected layer for classification output
# Dropout : Regularization technique to prevent overfitting by randomly dropping neurons during training.
from tensorflow.keras.preprocessing.sequence import pad_sequences
#pad_sequences : add 0 to shorter seq
from tensorflow.keras.utils import to_categorical 
# convert int to binary : 2 -> [0,0,1,0]

# getting path
BASE_DIR = os.path.expanduser("/home/kali/Documents/Block_cipher")
SPLIT_DATA_PATH = os.path.join(BASE_DIR , "Split_data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR , "trained_models")
os.makedirs(MODEL_SAVE_PATH , exist_ok = True)

#fixing hyperparameters
#MAX_SEQUENCE_LENGTH = 100 # all seq should have same length
EMBEDDING_DIM = 128   # vector dim size
LSTM_UNITS = 128 # lstm memory capacity
DROPOUT_RATE = 0.5 # to prevent overfitting
LEARNING_RATE = 0.001
EPOCHS = 200

#             1. - - - LOAD INPUT FILE  - - -
print(f"Loading lstm from {SPLIT_DATA_PATH}")
try:
	with open(os.path.join(SPLIT_DATA_PATH, "lstm_train_val.pkl") , 'rb') as f:
		seq_data = pickle.load(f)
	
	X_train_seq = seq_data['X_train_seq']
	y_train = seq_data['y_train']
	X_val_seq = seq_data['X_val_seq']
	y_val = seq_data['y_val']
	
	print(f"Loaded {len(X_train_seq)} training sample\n\n")
	#print(f"\n training content is {X_train_seq}")
	
except FileFoundError:
	print("ERROR: Sequence data file not found. Ensure data_preprocessor.py was run.")
	
	
# 		2. - - - label encoding  - - - 

def clean_labels(raw_labels):
    if isinstance(raw_labels, np.ndarray):
        labels_list = raw_labels.tolist()
    else:
        labels_list = raw_labels
    
    # Flatten if necessary and convert every element to a Python str
    flat_labels = []
    for item in labels_list:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            for sub_item in item:
                flat_labels.append(str(sub_item))
        else:
            flat_labels.append(str(item))
    return flat_labels

y_train_list = clean_labels(y_train)
y_val_list = clean_labels(y_val)



all_unique_labels = sorted(list(set(y_train_list + y_val_list)))
NUM_CLASSES = len(all_unique_labels)
print(f"DEBUG: Identified {NUM_CLASSES} unique classes: {all_unique_labels}")

# 2. Create a manual mapping dictionary (e.g., {'AES': 0, 'DES': 1, ...})
label_to_id = {label: i for i, label in enumerate(all_unique_labels)}

# 3. Manually encode the lists using the dictionary
y_train_encoded = np.array([label_to_id[label] for label in y_train_list])
y_val_encoded = np.array([label_to_id[label] for label in y_val_list])





#MAX_TOKEN_VALUE = np.max(np.concatenate(X_train_seq)) if len(X_train_seq) >0  else 0
#print(f"MAX_TOKEN_VALUE : {MAX_TOKEN_VALUE} \n")
#VOCAB_SIZE = int(MAX_TOKEN_VALUE)+1
#print(f"VOCAB_SIZE : {VOCAB_SIZE} \n")
#print(f"num classes : {NUM_CLASSES} \n")


print("Sample y_train_list:", y_train_list[:20])
print("Sample y_val_list:", y_val_list[:20])
print("Label to id mapping:", label_to_id)
print("Unique encoded y_train:", np.unique(y_train_encoded))
print("Unique encoded y_val:", np.unique(y_val_encoded))
print("NUM_CLASSES:", NUM_CLASSES)



from tensorflow.keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train_encoded, num_classes=NUM_CLASSES)
y_val_one_hot = to_categorical(y_val_encoded, num_classes=NUM_CLASSES)

print("\n")
print("y_train_one_hot shape:", y_train_one_hot.shape)
print("y_val_one_hot shape:", y_val_one_hot.shape)




# ---------------- 3. - - - DYNAMIC SIZING AND PADDING (FIXED LOGIC) - - - ----------------

# 1. Calculate MAX_SEQUENCE_LENGTH dynamically from the loaded data
all_sequences = X_train_seq.tolist() + X_val_seq.tolist()
# This finds the length of the longest opcode sequence across all training/validation data
MAX_SEQUENCE_LENGTH = max(len(seq) for seq in all_sequences) 

# 2. Calculate VOCAB_SIZE dynamically
# Flatten the sequences to find the max token ID (excluding padding 0)
flat_tokens = [token for seq in all_sequences for token in seq]

# The max value will be the largest token ID (e.g., 119). 
# Add 1 because the vocabulary size is 0-indexed (0 to MAX_TOKEN_VALUE)
MAX_TOKEN_VALUE = np.max(flat_tokens) if flat_tokens else 0
VOCAB_SIZE = int(MAX_TOKEN_VALUE) + 1 

print(f"\nDEBUG: Dynamic MAX_SEQUENCE_LENGTH: {MAX_SEQUENCE_LENGTH}")
print(f"DEBUG: Dynamic MAX_TOKEN_VALUE: {MAX_TOKEN_VALUE}")
print(f"DEBUG: Dynamic VOCAB_SIZE for Embedding layer: {VOCAB_SIZE}")


X_train_seq_padded = pad_sequences(
	X_train_seq,
	maxlen = MAX_SEQUENCE_LENGTH,
	padding = 'post',
	truncating = 'post'
)

X_val_seq_padded = pad_sequences(
    X_val_seq,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding='post',
    truncating='post'
)

print("X_train_seq_padded shape:", X_train_seq_padded.shape)
print("X_val_seq_padded shape:", X_val_seq_padded.shape)
#print("X_train_seq_array dtype:", X_train_seq_array.dtype)


X_train_seq_array = np.array(X_train_seq_padded)
X_val_seq_array = np.array(X_val_seq_padded)







# - - - 3. MODEL TRAINING  - - - 
#from tensorflow.keras.layers import Input

def build_lstm_model():
	model = Sequential()
	
	model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
	
	model.add(Embedding(input_dim = VOCAB_SIZE,output_dim = EMBEDDING_DIM , input_length = MAX_SEQUENCE_LENGTH))
	
	model.add(Bidirectional(LSTM(LSTM_UNITS))) #Purpose: Learns sequential dependencies in opcode sequences.
	
	model.add(Dropout(DROPOUT_RATE))
	model.add(Dense(NUM_CLASSES , activation = 'softmax', kernel_regularizer=l2(0.001)))
	#activation='softmax' → ensures outputs are normalized probabilities that sum to 1.
	model.compile(optimizer = Adam(learning_rate=LEARNING_RATE) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
	
	return model  

	
	# - - - flow of training  - - -
	
	#Embedding → turns opcode IDs into dense vectors.

	#LSTM → learns sequential patterns in opcode sequences.

	#Dropout → prevents overfitting.

#	Dense + Softmax → outputs class probabilities.

	#Compile → sets training rules (optimizer, loss, metric).#
	
	
	# - - - execute training - - -
	
	
#X_train_seq_array = np.array(X_train_seq)
#X_val_seq_array = np.array(X_val_seq)
	#Converts your training and validation opcode sequences into NumPy arrays.

	#NumPy arrays are the standard input format for Keras/TensorFlow models.

	#Ensures consistency and compatibility when feeding data into the LSTM.
	


	
	
	
	
	
	
	
	
	
lstm_model = build_lstm_model()

print("\n - - - Starting LSTM Training - - - ")
lstm_model.summary()

history = lstm_model.fit(
	X_train_seq_array,
	y_train_one_hot ,
	epochs = EPOCHS,
	batch_size =4,
	validation_data = (X_val_seq_array , y_val_one_hot),
	verbose = 1
	)
	
print(history.history)

val_preds = lstm_model.predict(X_val_seq_array)
val_pred_labels = np.argmax(val_preds, axis=1)
print("Encoded val true:", y_val_encoded)
print("Encoded val pred:", val_pred_labels)


model_path = os.path.join(MODEL_SAVE_PATH, 'lstm_model.h5')
lstm_model.save(model_path)
print(f"\nSUCCESS: LSTM model trained and saved to {model_path}")

























