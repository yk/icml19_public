The code of our method is in tf_robustify. It works with both pytorch and tensorflow.
Two scripts are provided: tensorflow_example.py and torch_example.py.
The torch example is the easiest, as it can just be run and will fetch both data as well as model by itself.
For the tensorflow example, use fetch_Madry_ResNet.py to fetch the models and place them in the appropriate ckpt_dir (parameter of tensorflow_example.py).

