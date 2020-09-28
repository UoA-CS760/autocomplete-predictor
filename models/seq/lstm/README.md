#Building LSTM model

Steps:
1. Run generate_data.py to convert raw source code into sequence of tokens

Usage: 
```
python generate_data.py --files_fp ..\..\..\data\source_code_file_names.txt --out_fp token-seq --base_dir ..\..\..\data\code\ token
```