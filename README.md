# Multi-Objectives program REPAIR framework


```
mkdir output_model || true  # create model output directory
python3 MOTrain.py 1 # run the trainer, with the gamma value of 1
```

You can make MOTrain.py do single-objective fine-tuning by setting its first parameter to zero.

The fine-tuned model is saved in the output_model directory and can be tested for inference by executing the following command.

```
python3 test.py
```
