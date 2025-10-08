from hailo whisper runtime code 

needed only when using HEF decoder mode (without KV-cache). It points to a directory containing token embedding weights for the decoder:

  1. Token embedding weights: token_embedding_weight_{variant}.npy 
  2. ONNX add input bias: onnx_add_input_{variant}.npy 

  These files are used in the _tokenization method (line 306) to convert token IDs into embeddings before feeding them to the HEF decoder.