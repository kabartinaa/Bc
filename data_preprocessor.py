import numpy as np
import pandas as pd
import pickle
import os
from collections import defaultdict


BASE_DIR = "/home/kali/Documents/Block_cipher"
DATA_DIR = "/home/kali/Documents/Block_cipher/Processed_Dataset"

#extract these features 
gnn_data_list=[]
lstm_sequence=[]
numerical_features=[]
labels=[]

# loop over the  datasets .txt.pkl
for i in range(1,115):
	file_name = f"cipher{i}.txt.pkl"
	file_path = os.path.join(DATA_DIR , file_name)
	
	with open(file_path , 'rb') as f:
		data = pickle.load(f)
		# The 'data' variable now holds this dictionary:
	#data = {
 #   	'graph': <NetworkX Graph Object>, 
  #  	'true_label': "Block_Cipher_AES_DES",
   # 	'features_by_function': {
    #    'main': {
     #       'block_count': 5,
      #      'edge_count': 5,
            #'opcode_sequence': ['bllo', 'blvc', 'stmdbhs', ...]      # }#}#}
		
		
		
		
	#print("Keys in data:", data.keys())
	#Keys in data: dict_keys(['graph', 'feature_by_fuction', 'truth_label'])
	#print(f"cipher {i} graph",data['graph'])
	#print(f"cipher {i} label",data['truth_label'])
	#print(f"cipher {i} graph",data['feature_by_fuction'])
	
	
	
	# A. gnn
	gnn_data_list.append(data['graph'])
	labels.append(data['truth_label'])
	
	
	
		#B. lstm
	main_func_name = max(data['feature_by_fuction'],key=lambda k:data['feature_by_fuction'][k]['block_count'])
	main_func=data['feature_by_fuction'][main_func_name]
	#append opcoode
	lstm_sequence.append(main_func['opcode_sequence'])
	
	
	
	
    # C. numerical feature for xgboost
	numerical_features.append([main_func['block_count'],main_func['edge_count']])
	
	
X_num = np.array(numerical_features)
print("\n")
#print(f"numerical feature is {X_num}\n")
X_seq = np.array(lstm_sequence, dtype=object) 


# - - -- - -SCALING ( edges , blocks : converting high count to scaled count) - - - - - -
#use x_num
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_num_scaler = scaler.fit_transform(X_num)

#print(f"before scaling features are {X_num}\n after scaling features are {X_num_scaler} ")

#print(f"original numerical shape {X_num.shape}")
#print(f"Scaled numerical shape {X_num_scaler.shape}")


# - - - - - Tokenisation ( to convert the opcode next to numerical ) - - - - 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<unk>")
#print(f"x_seq is {X_seq}")
tokenizer.fit_on_texts(X_seq)
X_seq_tokenizer = tokenizer.texts_to_sequences(X_seq)
#print(f" conveted x_seq is {X_seq_tokenizer}")
#conveted x_seq is [[14, 15, 16, 4, 5, 6, 7, 4, 5, 6, 7, 17], [2, 3, 18

MAX_SEQUENCE_LENGTH = max(len(s) for s in X_seq_tokenizer)
#pad seq to same length
X_seq_padded = pad_sequences(
	X_seq_tokenizer,
	maxlen = MAX_SEQUENCE_LENGTH,
	# If samm add 0
	padding = 'post',
	# remove of too long
	truncating = 'post'
)
#print(f"padding is {X_seq_padded}")
#print(f"Vocabulary Size : {len(tokenizer.word_index)}")
#print(f"Max Sequence Length : {MAX_SEQUENCE_LENGTH}")
#print(f"Padding sequence shape :{X_seq_padded.shape}")
    	
    	# finally neeed 
    	#X_num_scaled
    	#X_seq_padded
#print(f"gnn list is {gnn_data_list} ")


from sklearn.model_selection import train_test_split
#  70% Train, 10% Validation, and 20% Test split.
# convert all to np array
y = np.array(labels)
X_seq_padded_array = np.array(X_seq_padded)
X_num_scaled_array = np.array(X_num_scaler)
X_gnn_array = np.array(gnn_data_list , dtype = object) 
print(f"\n label array {y} \n seq array {X_seq_padded_array} \nscaled array {X_num_scaled_array}")



# - - - - SPLITTING THE CLASSES INTO BALANECD WAY FOR TRAIN , VAL ,TEST TO GET RID TO VAL_ACCRACY IMBALANCE

unique_labels = sorted(list(set(y)))
label_to_id = {label : i for i , label in enumerate(unique_labels)}   # enum: (0,aes) (1,res)
y_ids = np.array([label_to_id[label]for label in y])  #[0,1]


TRAIN_PER_CLASS = 7
VAL_PER_CLASS = 1

class_indices = defaultdict(list)
for i , label_id in enumerate(y_ids):
	class_indices[label_id].append(i)
	
train_indices=[]
val_indices=[]
test_indices=[]

for label_id , indices in class_indices.items():
	train_indices.extend(indices[:TRAIN_PER_CLASS])
	val_indices.extend(indices[TRAIN_PER_CLASS:TRAIN_PER_CLASS+VAL_PER_CLASS])
	test_indices.extend(indices[TRAIN_PER_CLASS+VAL_PER_CLASS:])


# TRAINING SET (Perfectly balanced)
X_train_num = X_num_scaled_array[train_indices]
X_train_seq = X_seq_padded_array[train_indices]
X_train_gnn = X_gnn_array[train_indices]
y_train = y[train_indices]

# VALIDATION SET (One sample of every class)
X_val_num = X_num_scaled_array[val_indices]
X_val_seq = X_seq_padded_array[val_indices]
X_val_gnn = X_gnn_array[val_indices]
y_val = y[val_indices]

# TEST SET (Remaining data)
X_test_num = X_num_scaled_array[test_indices]
X_test_seq = X_seq_padded_array[test_indices]
X_test_gnn = X_gnn_array[test_indices]
y_test = y[test_indices]




import collections
print("\n")
print("collection result : \n")
print(collections.Counter(y))


print("\n--- Final Data Shapes ---\n")
print(f"Total Samples: {len(y)} \n")
print(f"Training Samples: {len(y_train)} (approx 70%) \n")
print(f"Validation Samples: {len(y_val)} (approx 10%) \n")
#print(f"Testing Samples: {len(y_test)} (approx 20%)")
#print(f"wahts inside {X_train_gnn} \n and lstm is {X_train_seq}")


print("\n")
print("Training labels:", y_train)
print("Validation labels:", y_val)
print("Test labels:", y_test)
print("\n")


# create a dir which contains the datas of GNN along of gnn_trainer.py
SPLIT_DATA_PATH  =  os.path.join(BASE_DIR , "Split_data")
os.makedirs(SPLIT_DATA_PATH , exist_ok = True)
gnn_train_data = {
'X_train_gnn' : X_train_gnn,
'y_train' : y_train,
'X_val_gnn' : X_val_gnn ,
'y_val' : y_val
}
print(type(gnn_train_data))
with open(os.path.join(SPLIT_DATA_PATH , "gnn_train_val.pkl") , 'wb') as f:
	pickle.dump(gnn_train_data , f)
print(f"SUCCESS: Saved GNN training and validation data to {SPLIT_DATA_PATH}")
print(BASE_DIR)
 
	# to train lstm 

SPLIT_DATA_PATH  =  os.path.join(BASE_DIR , "Split_data")
os.makedirs(SPLIT_DATA_PATH , exist_ok = True)
lstm_train_data = {
'X_train_seq' : X_train_seq,
'y_train' : y_train,
'X_val_seq' : X_val_seq ,
'y_val' : y_val
}
print(type(lstm_train_data))
with open(os.path.join(SPLIT_DATA_PATH , "lstm_train_val.pkl") , 'wb') as f:
	pickle.dump(lstm_train_data , f)
print(f"SUCCESS: Saved LSTM training and validation data to {SPLIT_DATA_PATH}")
print(BASE_DIR)
print(f"{X_train_seq} and {X_val_seq}")
 
 
 
# B. *** NEW: Save Tokenizer and MAX_SEQUENCE_LENGTH ***
lstm_aux_data = {
    'tokenizer': tokenizer,
    'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH
}
with open(os.path.join(SPLIT_DATA_PATH , "lstm_aux_data.pkl") , 'wb') as f:
	pickle.dump(lstm_aux_data , f)
print(f"SUCCESS: Saved LSTM auxiliary data (tokenizer, max_len) to {SPLIT_DATA_PATH}")

print(f"{X_train_seq} and {X_val_seq}")




































    	
    	
    	
    	
    	
    	
    	
    	
	
