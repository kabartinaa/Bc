import angr
import networkx as nx
import pickle
import os

BASE_DIR = os.path.expanduser('/home/kali/Documents/Block_cipher')
BINARY_DIR = os.path.join(BASE_DIR , 'Binaries')
OUTPUT_DIR = os.path.join(BASE_DIR , 'Processed_Dataset')
TARGET_ARC = 'ARM'
os.makedirs(OUTPUT_DIR,exist_ok=True)

def extract_file(binary_file , i):
	p = angr.Project(binary_file , main_opts = {'arch' :TARGET_ARC , 'base_addr' : 0x400000 , 'backend':'blob'})
	cfg = p.analyses.CFGFast()
	nx_graph = cfg.graph
	#print(cfg.functions)
	print("CFG Nodes:",nx_graph.number_of_nodes())
	
	#print("print first five functions\n")
	#for addr , fun in list(cfg.functions.items()) [:5]:
	#	print(hex(addr) ,fun.name)
		
		
		
	function_data={}
	
	for func in cfg.functions.values():
		if func.is_simprocedure or func.is_plt:
			continue
		#print("Count Blocks and Edges (for XGBoost)")
		block_count = len(func.transition_graph.nodes)
		edge_count = len(func.transition_graph.edges)
		
		#print("Instruction Opcodes (for LSTM Sequence Model")
		instructions_opcodes = []
		for block in func.blocks: #sequence of instruction . block ends with jump / or...
			if block.disassembly:
				instructions_opcodes.extend([i.mnemonic for i in block.disassembly.insns])
		 	
		function_data[func.name] = {
		 'addr' : hex(func.addr),
		 'block_count' : block_count,
		 'edge_count' : edge_count,
		 'opcode_sequence' : instructions_opcodes
		 }
	#the largest. That’s assumed to be the “main” function.
	if function_data:
		main_func_name = max(function_data , key = lambda k:function_data[k]['block_count'])
		main_func = function_data[main_func_name]
		print(f" CFG Total nodes (Blocks) : {nx_graph.number_of_nodes()}")
		print(f"Main function @{main_func['addr']}")
		print(f"Basic blocks {main_func['block_count']}")
		print(f"Edges : {main_func['edge_count']}")
		print(f"instruction opcode sequence : {main_func['opcode_sequence'][:10]}")
		
	if i==1:
		ground_truth = "BLOCK_CIPHER_AES"
	elif i==2:
		ground_truth = "HASH_SHA256"
	elif i==3:
		ground_truth = "HASH_SHA256"
	elif i==4:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==5:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==6:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==7:
		ground_truth = "STREAM_CIPHER_CHACHA"
	elif i==8:
		ground_truth = "Utility-Wrapper_Func"
	elif i==9:
		ground_truth = "BLOCK_CIPHER_AES"
	elif i==10:
		ground_truth = "STREAM_CIPHER_CHACHA"
	elif i==11:
		ground_truth = "STREAM_CIPHER_CHACHA"
	elif i==12:
		ground_truth = "KEY_DERIVATION_PBKDF2"
	elif i==13:
		ground_truth = "KEY_DERIVATION_PBKDF2"
	elif i==14:
		ground_truth = "BLOCK_CIPHER_AES"
	elif i==15:
		ground_truth = "BLOCK_CIPHER_AES"
	elif i==16:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==17:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==18:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==19:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==20:
		ground_truth = "HASH_SHA256"
	elif i==21:
		ground_truth = "HASH_SHA256"
	elif i==22:
		ground_truth = "BLOCK_CIPHER_AES"
	elif i==23:
		ground_truth = "BLOCK_CIPHER_AES"
	elif i==24:
		ground_truth = "STREAM_CIPHER_CHACHA"
	elif i==25:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==26:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==27:
		ground_truth = "KEY_DERIVATION_PBKDF2"
	elif i==28:
		ground_truth = "MAC_POLY1305"
	elif i==29:
		ground_truth = "MAC_POLY1305"
	elif i==30:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==31:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==32:
		ground_truth = "HASH_SHA256"
	elif i==33: 
		ground_truth = "HASH_SHA256"
	elif i==34:
		ground_truth = "LIGHTWEIGHT_SIMON"
	elif i==35:
		ground_truth = "LIGHTWEIGHT_SIMON"
	elif i==36:
		ground_truth = "STREAM_CIPHER_CHACHA"
	elif i==37:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==38:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==39:
		ground_truth = "BLOCK_CIPHER_DES"
	elif i==40:
		ground_truth = "KEY_DERIVATION_PBKDF2"
	elif i==41:
		ground_truth = "KEY_DERIVATION_PBKDF2"
	elif i==42:
		ground_truth = "KEY_DERIVATION_PBKDF2"
	elif i==43:
		ground_truth = "MAC_POLY1305"
	elif i==44:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==45:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==46:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==47:
		ground_truth = "LIGHTWEIGHT_SIMON"
	elif i==48:
		ground_truth = "Utility-Wrapper_Func"
	elif i==49:
		ground_truth = "Utility-Wrapper_Func"
	elif i==50:
		ground_truth = "Utility-Wrapper_Func"
	elif i==51:
		ground_truth= "BLOCK_CIPHER_AES"
	elif i==52:
		ground_truth= "BLOCK_CIPHER_AES"
	elif i==53:
		ground_truth= "BLOCK_CIPHER_AES"
	elif i==54:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==55:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==56:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==57:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==58:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==59:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==60:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==61:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==62:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==63:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==64:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==65:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==66:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==67:
		ground_truth= "BLOCK_CIPHER_DES"
	elif i==68:
		ground_truth= "BLOCK_CIPHER_DES"
	elif i==69:
		ground_truth= "BLOCK_CIPHER_DES"
	elif i==70:
		ground_truth= "BLOCK_CIPHER_DES"
	elif i==71:
		ground_truth= "KEY_DERIVATION_PBKDF2"
	elif i==72:
		ground_truth= "KEY_DERIVATION_PBKDF2"
	elif i==73:
		ground_truth= "KEY_DERIVATION_PBKDF2"
	elif i==74:
		ground_truth= "KEY_DERIVATION_PBKDF2"
	elif i==75:
		ground_truth= "KEY_DERIVATION_PBKDF2"
	elif i==76:
		ground_truth= "KEY_DERIVATION_PBKDF2"
	elif i==77:
		ground_truth= "KEY_DERIVATION_PBKDF2"
	elif i==78:
		ground_truth= "MAC_POLY1305"
	elif i==79:
		ground_truth= "MAC_POLY1305"
	elif i==80:
		ground_truth= "MAC_POLY1305"
	elif i==81:
		ground_truth= "MAC_POLY1305"
	elif i==82:
		ground_truth= "MAC_POLY1305"
	elif i==83:
		ground_truth= "ASYMMETRIC_RSA"
	elif i==84:
		ground_truth= "ASYMMETRIC_RSA"
	elif i==85:
		ground_truth= "ASYMMETRIC_RSA"
	elif i==86:
		ground_truth= "ASYMMETRIC_RSA"
	elif i==87:
		ground_truth= "ASYMMETRIC_RSA"
	elif i==88:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==89:
		ground_truth= "STREAM_CIPHER_CHACHA"
	elif i==90:
		ground_truth= "HASH_SHA256"
	elif i==91:
		ground_truth= "HASH_SHA256"
	elif i==92:
		ground_truth= "HASH_SHA256"
	elif i==93:
		ground_truth= "LIGHTWEIGHT_SIMON"
	elif i==94:
		ground_truth= "LIGHTWEIGHT_SIMON"
	elif i==95:
		ground_truth= "LIGHTWEIGHT_SIMON"
	elif i==96:
		ground_truth= "LIGHTWEIGHT_SIMON"
	elif i==97:
		ground_truth= "LIGHTWEIGHT_SIMON"
	elif i==98:
		ground_truth= "Utility-Wrapper_Func"
	elif i==99:
		ground_truth= "Utility-Wrapper_Func"
	elif i==100:
		ground_truth = "MAC_POLY1305"
	elif i==101:
		ground_truth = "MAC_POLY1305"
	elif i==102:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==103:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==104:
		ground_truth = "ASYMMETRIC_RSA"
	elif i==105:
		ground_truth = "LIGHTWEIGHT_SIMON"
	elif i==106:
		ground_truth = "LIGHTWEIGHT_SIMON"
	elif i==107:
		ground_truth = "LIGHTWEIGHT_SIMON"
	elif i==108:
		ground_truth = "LIGHTWEIGHT_SIMON"
	elif i==109:
		ground_truth = "Utility-Wrapper_Func"
	elif i==110:
		ground_truth = "Utility-Wrapper_Func"
	elif i==111:
		ground_truth = "Utility-Wrapper_Func"
	elif i==112:
		ground_truth = "Utility-Wrapper_Func"
	elif i==113:
		ground_truth = "Utility-Wrapper_Func"
	elif i==114:
		ground_truth = "Utility-Wrapper_Func"
	else:
		ground_truth = f"Cipher_{i}_UNKNOWN"
		

    # ... (Rest of the script remains the same) ...
		
	data = {
		'graph':nx_graph,
		'feature_by_fuction':function_data,
		'truth_label':ground_truth
	}
	
	return data		
        
if __name__ == "__main__":
	#path = os.path.expanduser('/home/kali/Documents/Block_cipher/Binaries/cipher1.txt')
	#extract_file(path)
	for i in range(1,115):
		file_name = f"cipher{i}.txt"
		binary_path = os.path.join(BINARY_DIR , file_name)
		output_file = os.path.join(OUTPUT_DIR , f"{file_name}.pkl")
		if not os.path.exists(binary_path):
			print(f"ERROR : Binary file path {binary_path} not found")
			continue
			
		print(f"\n- - - processing {file_name} - - -")
		
		try:
			structured_data = extract_file(binary_path , i)
			with open (output_file , 'wb') as f:
				pickle.dump(structured_data , f)
			print(f"file {file_name} saved to output file {output_file}")
		except Exception as e:
			print(f"Failed processing {file_name} Error :{e}")
					
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			



