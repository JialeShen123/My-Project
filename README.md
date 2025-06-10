## How to use the model？

Using the transformer-based model for evaluation.
The model is used to estimate the daily average ET.





![图表, 表格  AI 生成的内容可能不正确。](file:////Users/yanyi/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image001.png)
The image above shows the attention map of the input values, darker means higher attention.
Throughout our work, this part of the code function is evaluation, but for the consumer, this part is really the process of running the model. 

### Setup

Code：pip install -r requirements.txt

Make sure you have all the required libraries.

### Usage

1. The two model (overpass time 10:30 and 14:00) were already saved in the directory “Model_pth”, and all csv files of 32 testing sites were also given in the directory “data/data_dir”.

2. The model.py and dataset.py modules do not need to be run, but serve the model evaluation.py modules.

3. In lines 224 and 234 of model evaluation.py, replace the input and output paths of your actual path.

​	Line 224: output_dir = # your own output path

​	Line 234: os.path.abspath(r"your input path ")

4. Run the code in the terminal：python model evaluation.py

**Before running this code, please make sure:**

​	Check that the dataset.py and model.py files exist and can be imported.

​	If the model is not configured to return attention weights, you may need to modify the initialization 	parameters of MultiheadAttention in model.py, for example, need_weights=True.

​	Ensure that the Model_pth/fluxAttention_train62val10test32_single_LE_1030.pth and data directories exist at the correct relative or absolute path.

```bash
# choice A: use only one LE observation per day
# TERRA, SPOT Overpass Time ~10:30
sparse = Le_all[[21,21,21,21]]
# FY-3D Satellite Overpass Time ~14:00
# sparse = le_all[[28,28, 28,28]]
```

​	When passing in the model corresponding to the transit time, you need to make sure that the relevant sentences in lines 70-74 of the code in the dataset.py module are also adjusted to the corresponding time, for example, if you enter the model at 10:30 a.m., you need to set it to "sparse = le_all[[21,21,21,21]]", if the time does not match, it will affect the final evaluation result. 

5. The estimation results and attention map visualization will be saved in output dir --Model_Output.

​	To visualize the attention map, modify the pytorch source codein:
/home/user/.local/lib/python3.xx/site-packages/torch/nn/modules/transformer.py

​	Find the TransformerDecoderLayer Class and in the _mha_block function,

​	Change need_weights=False to need_weights=True.

​	Then run the model evalution.py with visualization on:

​	python model evaluation.py –vis

 

 



 
