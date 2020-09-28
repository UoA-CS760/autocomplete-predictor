#Building LSTM model

Steps:
1. Run generate_data.py to convert raw source code into sequence of tokens
Source code file names to be added to data\source_code_file_names.txt and the code file should be added to data\code

Usage: 
```
python generate_data.py --files_fp ..\..\..\data\source_code_file_names.txt --out_fp token-seq --base_dir ..\..\..\data\code\ token
```

2. Run generate_vocab.py to create vocab file from the token-seq file and save in file 
Usage: 
```
python .\generate_vocab.py  --input_fp .\token-seq --out_fp vocab.txt --input_type source_code
```

3. Convert token seq into integer vector by running convert_token_seq_to_int_vector.py and save to file int-seq.txt
Usage: 
```
python .\convert_token_seq_to_int_vector.py
```