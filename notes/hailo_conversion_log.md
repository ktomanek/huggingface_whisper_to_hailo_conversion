notes
* tricky with GPU, but A100 works, L4 doesn't (wrong type -- leads to it endlessly hangign)


python3 -u -m conversion.convert_whisper_encoder \
  --variant tiny \
  --hw-arch hailo8l \
  --load-calib-set \
  /tmp/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx \
  2>&1 | tee conversion_output.log

# Running compilation directly

python compile_har_to_hef.py --har-path converted/tiny_whisper_encoder_10s_hailo8l_try1/whisper_tiny_encoder_10s_hailo_final_optimized.har --hw-arch hailo8l --model-script optimization/tiny/hailo8l/encoder_model_script_tiny.alls 2>&1 | tee hef_compilation_try1.log

# Hailo conversion on Server, N100
python3 -u -m conversion.convert_whisper_encoder  --variant tiny --hw-arch hailo8l --load-calib-set  /tmp/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx 
[info] No GPU chosen and no suitable GPU found, falling back to CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1759859330.731148 1701077 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1759859330.735866 1701077 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

Model conversion started.
[info] - Variant: tiny
[info] - Hardware Architecture: hailo8l

Model parsing

[info] Translation started on ONNX model whisper_tiny_encoder_10s_hailo_final
[info] Restored ONNX model whisper_tiny_encoder_10s_hailo_final (completion time: 00:00:00.11)
[info] Extracted ONNXRuntime meta-data for Hailo model (completion time: 00:00:00.48)
[info] Start nodes mapped from original model: 'x.1': 'whisper_tiny_encoder_10s_hailo_final/input_layer1'.
[info] End nodes mapped from original model: '/layer_norm/LayerNormalization'.
[info] Translation completed on ONNX model whisper_tiny_encoder_10s_hailo_final (completion time: 00:00:01.00)
[info] Saved HAR to: /home/katrintomanek/dev/hailo_dfc/hailo-whisper/conversion/converted/tiny_whisper_encoder_10s_hailo8l/whisper_tiny_encoder_10s_hailo_final.har

Model Optimization

[info] - Loading calibration set from ./conversion/optimization/tiny/encoder_calib_set_tiny_10s.npy
[info] Loading model script commands to whisper_tiny_encoder_10s_hailo_final from ./conversion/optimization/tiny/hailo8l/encoder_model_script_tiny.alls
[info] Starting Model Optimization
[info] Model received quantization params from the hn
[info] MatmulDecompose skipped
[info] Starting Mixed Precision
[info] Model Optimization Algorithm Mixed Precision is done (completion time is 00:00:00.72)
[info] Starting LayerNorm Decomposition
[info] Using dataset with 1024 entries for calibration
Calibration: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [04:36<00:00,  3.70entries/s]
[info] Model Optimization Algorithm LayerNorm Decomposition is done (completion time is 00:04:44.90)
[info] Starting Statistics Collector
[info] Using dataset with 1024 entries for calibration
Calibration: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [09:55<00:00,  1.72entries/s]
[info] Model Optimization Algorithm Statistics Collector is done (completion time is 00:09:59.14)
[info] Starting Fix zp_comp Encoding
[info] Model Optimization Algorithm Fix zp_comp Encoding is done (completion time is 00:00:00.02)
[info] Starting Matmul Equalization
[info] Model Optimization Algorithm Matmul Equalization is done (completion time is 00:00:00.60)
[info] Starting MatmulDecomposeFix
[info] Model Optimization Algorithm MatmulDecomposeFix is done (completion time is 00:00:00.00)
[info] activation fitting started for whisper_tiny_encoder_10s_hailo_final/reduce_sum_softmax1/act_op
[info] activation fitting started for whisper_tiny_encoder_10s_hailo_final/reduce_sum_softmax2/act_op
[info] No shifts available for layer whisper_tiny_encoder_10s_hailo_final/conv10/conv_op, using max shift instead. delta=0.0602
[info] No shifts available for layer whisper_tiny_encoder_10s_hailo_final/conv10/conv_op, using max shift instead. delta=0.0301
[info] activation fitting started for whisper_tiny_encoder_10s_hailo_final/reduce_sum_softmax3/act_op
[info] No shifts available for layer whisper_tiny_encoder_10s_hailo_final/conv13/conv_op, using max shift instead. delta=0.4729
[info] No shifts available for layer whisper_tiny_encoder_10s_hailo_final/conv13/conv_op, using max shift instead. delta=0.4729
[info] No shifts available for layer whisper_tiny_encoder_10s_hailo_final/conv14/conv_op, using max shift instead. delta=0.4707
[info] No shifts available for layer whisper_tiny_encoder_10s_hailo_final/conv14/conv_op, using max shift instead. delta=0.2354
[info] No shifts available for layer whisper_tiny_encoder_10s_hailo_final/conv13/conv_op, using max shift instead. delta=0.4729
[info] activation fitting started for whisper_tiny_encoder_10s_hailo_final/reduce_sum_softmax4/act_op
[info] Finetune encoding skipped
[info] Bias Correction skipped
[info] Adaround skipped
[info] Starting Quantization-Aware Fine-Tuning
[info] Using dataset with 3340 entries for finetune
Epoch 1/6
 24/417 ━━━━━━━━━━━━━━━━━━━━ 2:18:40 21s/step - _distill_loss_whisper_tiny_encoder_10s_hailo_final/normalization13: 0.3379 - total_distill_loss: 0.3379^C^C^C^C
[warning] Training cut by the user, proceed at your own peril
[info] Model Optimization Algorithm Quantization-Aware Fine-Tuning is done (completion time is 00:13:27.26)
[info] Starting Layer Noise Analysis
Full Quant Analysis: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [05:21<00:00, 160.68s/iterations]
[info] Model Optimization Algorithm Layer Noise Analysis is done (completion time is 00:05:29.30)
[info] Output layers signal-to-noise ratio (SNR): measures the quantization noise (higher is better)
[info] 	whisper_tiny_encoder_10s_hailo_final/output_layer1 SNR:	9.179 dB
[info] Model Optimization is done
[info] Saved HAR to: /home/katrintomanek/dev/hailo_dfc/hailo-whisper/conversion/converted/tiny_whisper_encoder_10s_hailo8l/whisper_tiny_encoder_10s_hailo_final_optimized.har

Model compilation

[info] To achieve optimal performance, set the compiler_optimization_level to "max" by adding performance_param(compiler_optimization_level=max) to the model script. Note that this may increase compilation time.
[info] Loading network parameters
[info] Starting Hailo allocation and compilation flow
[info] Building optimization options for network layers...
[info] Successfully built optimization options - 11s 862ms
[info] Trying to compile the network in a single context
[info] Single context flow failed: Recoverable single context error
[info] Building optimization options for network layers...

[info] Successfully built optimization options - 12m 31s 782ms
[info] Using Multi-context flow
[info] Resources optimization params: max_control_utilization=60%, max_compute_utilization=60%, max_compute_16bit_utilization=60%, max_memory_utilization (weights)=60%, max_input_aligner_utilization=60%, max_apu_utilization=60%
[info] Finding the best partition to contexts...
[.......................<==>.............] Duration: 00:01:03                                                                                                                                    
Found valid partition to 7 contexts
[...<==>.................................] Duration: 00:00:30                                                                                                                                    
Found valid partition to 7 contexts, Performance improved by 12.2%
[info] Searching for a better partition...
[.....................<==>...............] Duration: 00:00:32                                                                                                                                    
Found valid partition to 7 contexts, Performance improved by 0.9%
[info] Searching for a better partition...
[.....<==>...............................] Duration: 00:00:23                                                                                                                                    
Found valid partition to 7 contexts, Performance improved by 17.0%
[info] Searching for a better partition...
[....................................<==>] Duration: 00:00:19                                                                                                                                    
Found valid partition to 7 contexts, Performance improved by 13.8%
[info] Searching for a better partition...
[......<==>..............................] Duration: 00:00:22                                                                                                                                    
Found valid partition to 7 contexts, Performance improved by 1.4%
[info] Searching for a better partition...
[.................<==>...................] Duration: 00:00:16                                                                                                                                    
Found valid partition to 7 contexts, Performance improved by 7.5%
[info] Searching for a better partition...
[...........................<==>.........] Duration: 00:00:27                                                                                                                                    
Found valid partition to 7 contexts, Performance improved by 7.9%
[info] Searching for a better partition...
[.....<==>...............................] Duration: 00:01:01                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 5.9%
[info] Searching for a better partition...
[.............<==>.......................] Duration: 00:01:17                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 0.6%
[info] Searching for a better partition...
[..........................<==>..........] Duration: 00:00:20                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 0.4%
[info] Searching for a better partition...
[........<==>............................] Duration: 00:03:04                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 0.4%
[info] Searching for a better partition...
[<==>....................................] Duration: 00:00:15                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 0.6%
[info] Searching for a better partition...
[....<==>................................] Duration: 00:00:15                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 2.6%
[info] Searching for a better partition...
[...<==>.................................] Duration: 00:00:14                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 0.0%
[info] Searching for a better partition...
[.....<==>...............................] Duration: 00:00:14                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 2.6%
[info] Searching for a better partition...
[............<==>........................] Duration: 00:00:16                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 1.8%
[info] Searching for a better partition...
[...<==>.................................] Duration: 00:00:14                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 1.6%
[info] Searching for a better partition...
[........<==>............................] Duration: 00:00:14                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 0.9%
[info] Searching for a better partition...
[.<==>...................................] Duration: 00:01:31                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 1.9%
[info] Searching for a better partition...
[..................<==>..................] Duration: 00:00:32                                                                                                                                    
Found valid partition to 8 contexts, Performance improved by 4.3%
[info] Searching for a better partition...
[........................<==>............] Duration: 00:04:37                                                                                                                                    
Found valid partition to 9 contexts, Performance improved by 2.4%
[info] Searching for a better partition...
[.........<==>...........................] Duration: 00:01:15                                                                                                                                    
Found valid partition to 9 contexts, Performance improved by 5.4%
[info] Searching for a better partition...
[....................<==>................] Duration: 00:00:40                                                                                                                                    
Found valid partition to 9 contexts, Performance improved by 6.0%
[info] Searching for a better partition...
[...............................<==>.....] Duration: 00:00:19                                                                                                                                    
Found valid partition to 9 contexts, Performance improved by 0.9%
[info] Searching for a better partition...
[......<==>..............................] Elapsed: 00:00:08                                                                                                                                     
[info] Partition to contexts finished successfully
[info] Partitioner finished after 362 iterations, Time it took: 20m 58s 302ms
[info] Applying selected partition to 9 contexts...
[info] Validating layers feasibility

Validating whisper_tiny_encoder_10s_hailo_final_context_0 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                                                                                                                     


Validating whisper_tiny_encoder_10s_hailo_final_context_1 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                                                                                                                     


Validating whisper_tiny_encoder_10s_hailo_final_context_2 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                                                                                                                     


Validating whisper_tiny_encoder_10s_hailo_final_context_3 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                                                                                                                     


Validating whisper_tiny_encoder_10s_hailo_final_context_4 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                                                                                                                     


Validating whisper_tiny_encoder_10s_hailo_final_context_5 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                                                                                                                     


Validating whisper_tiny_encoder_10s_hailo_final_context_6 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                                                                                                                     


Validating whisper_tiny_encoder_10s_hailo_final_context_7 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                                                                                                                     


Validating whisper_tiny_encoder_10s_hailo_final_context_8 layer by layer (100%)

 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 
 +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  + 

● Finished                                           

[info] Layers feasibility validated successfully
[info] Running resources allocation (mapping) flow, time per context: 59m 59s
Context:0/8 Iteration 4: Trying parallel mapping...  
          cluster_0  cluster_1  cluster_2  cluster_3  cluster_4  cluster_5  cluster_6  cluster_7  prepost 
 worker0  X          V          *          *          V          V          *          *          V       
 worker1  V          V          *          *          V          V          *          *          V       
 worker2  V          X          *          *          V          V          *          *          V       
 worker3  X          *          *          *          V          V          *          *          V       
Context:1/8 Iteration 4: Trying parallel mapping...  
          cluster_0  cluster_1  cluster_2  cluster_3  cluster_4  cluster_5  cluster_6  cluster_7  prepost 
 worker0  V          V          *          *          V          V          *          *          V       
 worker1  V          V          *          *          X          *          *          *          V       
 worker2  V          V          *          *          X          *          *          *          V       
 worker3  V          V          *          *          V          V          *          *          V       
Context:2/8 Iteration 4: Trying parallel mapping...  
          cluster_0  cluster_1  cluster_2  cluster_3  cluster_4  cluster_5  cluster_6  cluster_7  prepost 
 worker0  V          V          *          *          *          *          *          *          V       
 worker1  *          *          *          *          V          *          *          *          V       
 worker2  V          V          *          *          V          V          *          *          V       
 worker3  V          V          *          *          V          V          *          *          V       
Context:3/8 Iteration 4: Trying parallel mapping...  
          cluster_0  cluster_1  cluster_2  cluster_3  cluster_4  cluster_5  cluster_6  cluster_7  prepost 
 worker0  V          V          *          *          V          V          *          *          V       
 worker1  V          V          *          *          X          *          *          *          V       
 worker2  V          V          *          *          X          *          *          *          V       
 worker3  V          V          *          *          V          V          *          *          V       
Context:4/8 Iteration 4: Trying parallel mapping...  
          cluster_0  cluster_1  cluster_2  cluster_3  cluster_4  cluster_5  cluster_6  cluster_7  prepost 
 worker0  V          V          *          *          V          V          *          *          V       
 worker1  V          V          *          *          V          V          *          *          V       
 worker2  V          V          *          *          V          V          *          *          V       
 worker3  V          V          *          *          V          V          *          *          V       
Context:5/8 Iteration 4: Trying parallel mapping...  
          cluster_0  cluster_1  cluster_2  cluster_3  cluster_4  cluster_5  cluster_6  cluster_7  prepost 
 worker0  V          V          *          *          V          V          *          *          V       
 worker1  V          V          *          *          *          *          *          *          V       
 worker2  *          V          *          *          *          *          *          *          V       
 worker3  V          V          *          *          V          V          *          *          V       

  00:56
Reverts on cluster mapping: 0
Reverts on inter-cluster connectivity: 0
Reverts on pre-mapping validation: 0
Reverts on split failed: 0


[info] whisper_tiny_encoder_10s_hailo_final_context_0 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 100%                | 40.6%               | 47.7%              |
[info] | cluster_1 | 50%                 | 20.3%               | 43%                |
[info] | cluster_4 | 75%                 | 40.6%               | 50%                |
[info] | cluster_5 | 6.3%                | 9.4%                | 15.6%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 57.8%               | 27.7%               | 39.1%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] whisper_tiny_encoder_10s_hailo_final_context_1 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 25%                 | 6.3%                | 14.1%              |
[info] | cluster_1 | 68.8%               | 50%                 | 32%                |
[info] | cluster_4 | 62.5%               | 25%                 | 34.4%              |
[info] | cluster_5 | 87.5%               | 42.2%               | 36.7%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60.9%               | 30.9%               | 29.3%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] whisper_tiny_encoder_10s_hailo_final_context_2 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 50%                 | 18.8%               | 18.8%              |
[info] | cluster_1 | 50%                 | 73.4%               | 75.8%              |
[info] | cluster_4 | 43.8%               | 43.8%               | 36.7%              |
[info] | cluster_5 | 100%                | 46.9%               | 42.2%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60.9%               | 45.7%               | 43.4%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] whisper_tiny_encoder_10s_hailo_final_context_3 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 25%                 | 6.3%                | 14.1%              |
[info] | cluster_1 | 68.8%               | 50%                 | 32%                |
[info] | cluster_4 | 62.5%               | 25%                 | 33.6%              |
[info] | cluster_5 | 87.5%               | 42.2%               | 36.7%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60.9%               | 30.9%               | 29.1%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] whisper_tiny_encoder_10s_hailo_final_context_4 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 50%                 | 18.8%               | 18.8%              |
[info] | cluster_1 | 50%                 | 73.4%               | 75.8%              |
[info] | cluster_4 | 43.8%               | 43.8%               | 35.9%              |
[info] | cluster_5 | 100%                | 46.9%               | 42.2%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60.9%               | 45.7%               | 43.2%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] whisper_tiny_encoder_10s_hailo_final_context_5 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 25%                 | 6.3%                | 14.1%              |
[info] | cluster_1 | 68.8%               | 50%                 | 32%                |
[info] | cluster_4 | 62.5%               | 25%                 | 34.4%              |
[info] | cluster_5 | 87.5%               | 42.2%               | 36.7%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60.9%               | 30.9%               | 29.3%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] whisper_tiny_encoder_10s_hailo_final_context_6 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 50%                 | 18.8%               | 18.8%              |
[info] | cluster_1 | 50%                 | 73.4%               | 75.8%              |
[info] | cluster_4 | 43.8%               | 43.8%               | 36.7%              |
[info] | cluster_5 | 100%                | 46.9%               | 42.2%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60.9%               | 45.7%               | 43.4%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] whisper_tiny_encoder_10s_hailo_final_context_7 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 25%                 | 6.3%                | 14.1%              |
[info] | cluster_1 | 68.8%               | 50%                 | 32%                |
[info] | cluster_4 | 62.5%               | 25%                 | 33.6%              |
[info] | cluster_5 | 87.5%               | 42.2%               | 36.7%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60.9%               | 30.9%               | 29.1%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] whisper_tiny_encoder_10s_hailo_final_context_8 utilization: 
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | cluster_0 | 50%                 | 67.2%               | 78.9%              |
[info] | cluster_1 | 37.5%               | 20.3%               | 20.3%              |
[info] | cluster_4 | 75%                 | 51.6%               | 43.8%              |
[info] | cluster_5 | 81.3%               | 31.3%               | 31.3%              |
[info] +-----------+---------------------+---------------------+--------------------+
[info] | Total     | 60.9%               | 42.6%               | 43.6%              |
[info] +-----------+---------------------+---------------------+--------------------+