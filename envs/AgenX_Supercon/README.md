# Data

- source: https://www.kaggle.com/datasets/munumbutt/superconductor-dataset

# Training
```shell


/home/pym/anaconda3/bin/python /media/pym/date/vLab/Supercon/_train_supercon.py 
Loading and processing data from: ./data/superconductor_data.tsv
Loaded dataset with columns: ['num', 'name', 'element', 'str3', 'utc', 'tc', 'journal']
Dataset shape: (26321, 7)
Warning: Dataset contains 8027 missing values
After dropping rows with missing element or tc: (26321, 7)
Found 83 unique elements in the dataset
Found 426 unique structure types
Extracted feature matrix shape: (26321, 509)
Number of feature columns: 509
Processed 21056 training samples and 5265 test samples
Feature dimension: 509
Input feature size: 509
/home/pym/anaconda3/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:62: UserWarning:

The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.

Epoch [1/300], Train Loss: 343.2515, Test Loss: 230.0449, R²: 0.8041, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [2/300], Train Loss: 242.7944, Test Loss: 235.8619, R²: 0.7991, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [3/300], Train Loss: 233.5109, Test Loss: 186.9604, R²: 0.8408, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [4/300], Train Loss: 216.6565, Test Loss: 180.6987, R²: 0.8461, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [5/300], Train Loss: 208.9696, Test Loss: 187.1159, R²: 0.8406, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [6/300], Train Loss: 203.1333, Test Loss: 173.7016, R²: 0.8521, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [7/300], Train Loss: 195.1075, Test Loss: 173.5314, R²: 0.8522, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [8/300], Train Loss: 187.1085, Test Loss: 169.2936, R²: 0.8558, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [9/300], Train Loss: 184.2508, Test Loss: 166.1714, R²: 0.8585, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [10/300], Train Loss: 179.4395, Test Loss: 173.8220, R²: 0.8520, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [11/300], Train Loss: 179.5958, Test Loss: 165.5289, R²: 0.8590, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [12/300], Train Loss: 175.6575, Test Loss: 169.0492, R²: 0.8560, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [13/300], Train Loss: 172.1897, Test Loss: 176.5825, R²: 0.8496, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [14/300], Train Loss: 172.3676, Test Loss: 152.8421, R²: 0.8698, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [15/300], Train Loss: 170.1051, Test Loss: 154.8245, R²: 0.8681, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [16/300], Train Loss: 168.4055, Test Loss: 154.8709, R²: 0.8681, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [17/300], Train Loss: 166.6765, Test Loss: 164.3132, R²: 0.8601, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [18/300], Train Loss: 168.7275, Test Loss: 150.6323, R²: 0.8717, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [19/300], Train Loss: 164.4159, Test Loss: 157.0037, R²: 0.8663, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [20/300], Train Loss: 163.7963, Test Loss: 159.1434, R²: 0.8645, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [21/300], Train Loss: 162.2541, Test Loss: 162.8400, R²: 0.8613, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [22/300], Train Loss: 157.5040, Test Loss: 152.9172, R²: 0.8698, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [23/300], Train Loss: 160.4440, Test Loss: 150.2483, R²: 0.8720, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [24/300], Train Loss: 159.6231, Test Loss: 156.5951, R²: 0.8666, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [25/300], Train Loss: 157.0766, Test Loss: 152.8802, R²: 0.8698, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [26/300], Train Loss: 155.3949, Test Loss: 152.6068, R²: 0.8700, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [27/300], Train Loss: 154.1123, Test Loss: 161.7064, R²: 0.8623, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [28/300], Train Loss: 152.1644, Test Loss: 146.5644, R²: 0.8752, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [29/300], Train Loss: 150.2607, Test Loss: 144.0876, R²: 0.8773, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [30/300], Train Loss: 151.3252, Test Loss: 152.0246, R²: 0.8705, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [31/300], Train Loss: 149.3436, Test Loss: 145.2342, R²: 0.8763, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [32/300], Train Loss: 149.4515, Test Loss: 142.7079, R²: 0.8785, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [33/300], Train Loss: 148.8139, Test Loss: 137.3632, R²: 0.8830, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [34/300], Train Loss: 147.3257, Test Loss: 149.2832, R²: 0.8729, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [35/300], Train Loss: 145.4798, Test Loss: 146.0258, R²: 0.8756, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [36/300], Train Loss: 144.5334, Test Loss: 144.4973, R²: 0.8769, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [37/300], Train Loss: 145.7573, Test Loss: 139.5403, R²: 0.8812, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [38/300], Train Loss: 142.7517, Test Loss: 152.5095, R²: 0.8701, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [39/300], Train Loss: 142.4869, Test Loss: 142.9929, R²: 0.8782, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [40/300], Train Loss: 141.7313, Test Loss: 145.8579, R²: 0.8758, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [41/300], Train Loss: 142.9779, Test Loss: 139.0809, R²: 0.8816, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [42/300], Train Loss: 140.7750, Test Loss: 143.2934, R²: 0.8780, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [43/300], Train Loss: 142.1596, Test Loss: 142.3760, R²: 0.8787, LR: 0.001000
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [44/300], Train Loss: 137.1126, Test Loss: 147.5419, R²: 0.8743, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [45/300], Train Loss: 131.7312, Test Loss: 132.9777, R²: 0.8867, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [46/300], Train Loss: 127.5195, Test Loss: 132.8244, R²: 0.8869, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [47/300], Train Loss: 124.6990, Test Loss: 132.3162, R²: 0.8873, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [48/300], Train Loss: 126.0278, Test Loss: 130.4855, R²: 0.8889, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [49/300], Train Loss: 124.0074, Test Loss: 139.9967, R²: 0.8808, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [50/300], Train Loss: 124.9367, Test Loss: 142.5300, R²: 0.8786, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [51/300], Train Loss: 122.1255, Test Loss: 133.7242, R²: 0.8861, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [52/300], Train Loss: 123.5329, Test Loss: 131.4898, R²: 0.8880, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [53/300], Train Loss: 120.3934, Test Loss: 129.2024, R²: 0.8900, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [54/300], Train Loss: 122.1665, Test Loss: 134.1800, R²: 0.8857, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [55/300], Train Loss: 119.6575, Test Loss: 127.9920, R²: 0.8910, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [56/300], Train Loss: 121.9571, Test Loss: 132.2420, R²: 0.8874, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [57/300], Train Loss: 120.1868, Test Loss: 130.3781, R²: 0.8890, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [58/300], Train Loss: 117.6738, Test Loss: 130.7381, R²: 0.8887, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [59/300], Train Loss: 117.8498, Test Loss: 131.2396, R²: 0.8882, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [60/300], Train Loss: 117.2473, Test Loss: 127.2594, R²: 0.8916, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [61/300], Train Loss: 118.5541, Test Loss: 132.2531, R²: 0.8874, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [62/300], Train Loss: 117.2321, Test Loss: 130.5709, R²: 0.8888, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [63/300], Train Loss: 116.8645, Test Loss: 129.5574, R²: 0.8897, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [64/300], Train Loss: 117.1088, Test Loss: 126.8816, R²: 0.8919, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [65/300], Train Loss: 116.9268, Test Loss: 128.9329, R²: 0.8902, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [66/300], Train Loss: 117.7906, Test Loss: 129.8556, R²: 0.8894, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [67/300], Train Loss: 113.6907, Test Loss: 125.6175, R²: 0.8930, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [68/300], Train Loss: 116.0965, Test Loss: 130.7745, R²: 0.8886, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [69/300], Train Loss: 115.3376, Test Loss: 127.0301, R²: 0.8918, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [70/300], Train Loss: 114.7735, Test Loss: 124.6455, R²: 0.8938, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [71/300], Train Loss: 114.1389, Test Loss: 132.1808, R²: 0.8874, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [72/300], Train Loss: 113.5964, Test Loss: 124.9889, R²: 0.8936, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [73/300], Train Loss: 113.5542, Test Loss: 131.7780, R²: 0.8878, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [74/300], Train Loss: 112.8506, Test Loss: 129.5192, R²: 0.8897, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [75/300], Train Loss: 113.2619, Test Loss: 124.2258, R²: 0.8942, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [76/300], Train Loss: 110.4282, Test Loss: 130.3787, R²: 0.8890, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [77/300], Train Loss: 110.2608, Test Loss: 127.6199, R²: 0.8913, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [78/300], Train Loss: 110.5015, Test Loss: 123.9869, R²: 0.8944, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [79/300], Train Loss: 111.5319, Test Loss: 127.2083, R²: 0.8917, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [80/300], Train Loss: 111.3176, Test Loss: 128.1234, R²: 0.8909, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [81/300], Train Loss: 109.9657, Test Loss: 128.9429, R²: 0.8902, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [82/300], Train Loss: 109.6082, Test Loss: 123.6026, R²: 0.8947, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [83/300], Train Loss: 111.6909, Test Loss: 126.6708, R²: 0.8921, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [84/300], Train Loss: 110.7776, Test Loss: 127.5836, R²: 0.8913, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [85/300], Train Loss: 111.3004, Test Loss: 127.8193, R²: 0.8911, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [86/300], Train Loss: 109.6558, Test Loss: 127.6494, R²: 0.8913, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [87/300], Train Loss: 108.8141, Test Loss: 123.6948, R²: 0.8947, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [88/300], Train Loss: 109.5903, Test Loss: 125.9365, R²: 0.8927, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [89/300], Train Loss: 108.0291, Test Loss: 123.5476, R²: 0.8948, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [90/300], Train Loss: 109.0207, Test Loss: 124.0213, R²: 0.8944, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [91/300], Train Loss: 107.5241, Test Loss: 130.0935, R²: 0.8892, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [92/300], Train Loss: 108.1268, Test Loss: 123.5584, R²: 0.8948, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [93/300], Train Loss: 106.4584, Test Loss: 124.6209, R²: 0.8939, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [94/300], Train Loss: 107.1962, Test Loss: 121.4068, R²: 0.8966, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [95/300], Train Loss: 106.5932, Test Loss: 135.7865, R²: 0.8844, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [96/300], Train Loss: 106.6593, Test Loss: 122.4882, R²: 0.8957, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [97/300], Train Loss: 108.1770, Test Loss: 121.8828, R²: 0.8962, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [98/300], Train Loss: 106.0206, Test Loss: 122.8369, R²: 0.8954, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [99/300], Train Loss: 105.6580, Test Loss: 123.8336, R²: 0.8945, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [100/300], Train Loss: 105.0175, Test Loss: 120.9734, R²: 0.8970, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [101/300], Train Loss: 105.5894, Test Loss: 124.9806, R²: 0.8936, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [102/300], Train Loss: 105.1164, Test Loss: 120.8950, R²: 0.8970, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [103/300], Train Loss: 103.8250, Test Loss: 121.0990, R²: 0.8969, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [104/300], Train Loss: 105.7900, Test Loss: 117.6187, R²: 0.8998, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [105/300], Train Loss: 104.1361, Test Loss: 121.9722, R²: 0.8961, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [106/300], Train Loss: 103.7491, Test Loss: 122.4351, R²: 0.8957, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [107/300], Train Loss: 104.6630, Test Loss: 120.8313, R²: 0.8971, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [108/300], Train Loss: 105.2060, Test Loss: 121.6170, R²: 0.8964, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [109/300], Train Loss: 103.7744, Test Loss: 119.8298, R²: 0.8979, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [110/300], Train Loss: 103.0249, Test Loss: 123.1906, R²: 0.8951, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [111/300], Train Loss: 103.8936, Test Loss: 121.3858, R²: 0.8966, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [112/300], Train Loss: 102.6694, Test Loss: 123.8534, R²: 0.8945, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [113/300], Train Loss: 104.3312, Test Loss: 123.4163, R²: 0.8949, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [114/300], Train Loss: 102.9279, Test Loss: 117.4099, R²: 0.9000, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [115/300], Train Loss: 100.9890, Test Loss: 119.7383, R²: 0.8980, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [116/300], Train Loss: 102.3632, Test Loss: 121.7568, R²: 0.8963, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [117/300], Train Loss: 101.1955, Test Loss: 118.8278, R²: 0.8988, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [118/300], Train Loss: 101.0702, Test Loss: 120.8280, R²: 0.8971, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [119/300], Train Loss: 101.2419, Test Loss: 116.6729, R²: 0.9006, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [120/300], Train Loss: 101.5664, Test Loss: 115.6286, R²: 0.9015, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [121/300], Train Loss: 100.2217, Test Loss: 119.4592, R²: 0.8983, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [122/300], Train Loss: 102.9092, Test Loss: 129.6987, R²: 0.8895, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [123/300], Train Loss: 99.4569, Test Loss: 114.8863, R²: 0.9022, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [124/300], Train Loss: 99.9884, Test Loss: 119.6598, R²: 0.8981, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [125/300], Train Loss: 99.4924, Test Loss: 117.5451, R²: 0.8999, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [126/300], Train Loss: 100.1212, Test Loss: 118.4274, R²: 0.8991, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [127/300], Train Loss: 99.8819, Test Loss: 120.3368, R²: 0.8975, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [128/300], Train Loss: 99.0233, Test Loss: 117.1184, R²: 0.9003, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [129/300], Train Loss: 98.8865, Test Loss: 117.6313, R²: 0.8998, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [130/300], Train Loss: 98.1824, Test Loss: 118.2237, R²: 0.8993, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [131/300], Train Loss: 99.7220, Test Loss: 121.4942, R²: 0.8965, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [132/300], Train Loss: 98.4015, Test Loss: 115.3538, R²: 0.9018, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [133/300], Train Loss: 98.5035, Test Loss: 117.5302, R²: 0.8999, LR: 0.000500
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [134/300], Train Loss: 97.8858, Test Loss: 123.7246, R²: 0.8946, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [135/300], Train Loss: 93.4078, Test Loss: 117.7258, R²: 0.8997, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [136/300], Train Loss: 92.7109, Test Loss: 114.7563, R²: 0.9023, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [137/300], Train Loss: 90.1041, Test Loss: 111.6441, R²: 0.9049, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [138/300], Train Loss: 90.5296, Test Loss: 118.2328, R²: 0.8993, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [139/300], Train Loss: 90.1653, Test Loss: 115.9585, R²: 0.9012, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [140/300], Train Loss: 91.2815, Test Loss: 113.9807, R²: 0.9029, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [141/300], Train Loss: 88.1661, Test Loss: 112.2618, R²: 0.9044, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [142/300], Train Loss: 89.5488, Test Loss: 112.0168, R²: 0.9046, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [143/300], Train Loss: 89.6367, Test Loss: 112.2860, R²: 0.9044, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [144/300], Train Loss: 89.5542, Test Loss: 111.9517, R²: 0.9047, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [145/300], Train Loss: 89.8899, Test Loss: 109.5350, R²: 0.9067, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [146/300], Train Loss: 88.5840, Test Loss: 110.8543, R²: 0.9056, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [147/300], Train Loss: 87.5739, Test Loss: 114.0576, R²: 0.9029, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [148/300], Train Loss: 88.3515, Test Loss: 121.1626, R²: 0.8968, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [149/300], Train Loss: 87.4549, Test Loss: 110.0717, R²: 0.9063, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [150/300], Train Loss: 85.8347, Test Loss: 110.5964, R²: 0.9058, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [151/300], Train Loss: 86.7325, Test Loss: 111.6480, R²: 0.9049, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [152/300], Train Loss: 87.3984, Test Loss: 109.1256, R²: 0.9071, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [153/300], Train Loss: 85.7961, Test Loss: 112.6597, R²: 0.9041, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [154/300], Train Loss: 86.5713, Test Loss: 114.2065, R²: 0.9027, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [155/300], Train Loss: 86.2663, Test Loss: 110.5986, R²: 0.9058, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [156/300], Train Loss: 86.6593, Test Loss: 111.5351, R²: 0.9050, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [157/300], Train Loss: 85.9080, Test Loss: 109.5416, R²: 0.9067, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [158/300], Train Loss: 86.0573, Test Loss: 112.8963, R²: 0.9039, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [159/300], Train Loss: 86.6701, Test Loss: 112.2540, R²: 0.9044, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [160/300], Train Loss: 85.8843, Test Loss: 107.6050, R²: 0.9084, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [161/300], Train Loss: 85.5994, Test Loss: 110.1569, R²: 0.9062, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [162/300], Train Loss: 86.2588, Test Loss: 109.6803, R²: 0.9066, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [163/300], Train Loss: 84.5145, Test Loss: 107.4674, R²: 0.9085, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [164/300], Train Loss: 85.6449, Test Loss: 110.7905, R²: 0.9056, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [165/300], Train Loss: 84.1526, Test Loss: 108.1034, R²: 0.9079, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [166/300], Train Loss: 84.8566, Test Loss: 109.4622, R²: 0.9068, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [167/300], Train Loss: 82.4364, Test Loss: 109.2005, R²: 0.9070, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [168/300], Train Loss: 83.8428, Test Loss: 114.5256, R²: 0.9025, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [169/300], Train Loss: 83.2334, Test Loss: 109.3366, R²: 0.9069, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [170/300], Train Loss: 83.1682, Test Loss: 106.4807, R²: 0.9093, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [171/300], Train Loss: 82.6975, Test Loss: 106.3937, R²: 0.9094, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [172/300], Train Loss: 83.2499, Test Loss: 110.2619, R²: 0.9061, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [173/300], Train Loss: 84.5263, Test Loss: 106.9404, R²: 0.9089, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [174/300], Train Loss: 81.2051, Test Loss: 107.4511, R²: 0.9085, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [175/300], Train Loss: 81.8293, Test Loss: 106.0194, R²: 0.9097, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [176/300], Train Loss: 82.5866, Test Loss: 104.2995, R²: 0.9112, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [177/300], Train Loss: 81.5760, Test Loss: 105.1671, R²: 0.9104, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [178/300], Train Loss: 81.6705, Test Loss: 106.0615, R²: 0.9097, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [179/300], Train Loss: 80.6299, Test Loss: 107.3634, R²: 0.9086, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [180/300], Train Loss: 81.7721, Test Loss: 109.1727, R²: 0.9070, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [181/300], Train Loss: 82.2131, Test Loss: 104.7672, R²: 0.9108, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [182/300], Train Loss: 81.7704, Test Loss: 107.3247, R²: 0.9086, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [183/300], Train Loss: 81.6194, Test Loss: 106.0048, R²: 0.9097, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [184/300], Train Loss: 82.4157, Test Loss: 105.6961, R²: 0.9100, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [185/300], Train Loss: 80.6775, Test Loss: 105.9260, R²: 0.9098, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [186/300], Train Loss: 81.2243, Test Loss: 109.9766, R²: 0.9063, LR: 0.000250
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [187/300], Train Loss: 80.7077, Test Loss: 105.7506, R²: 0.9099, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [188/300], Train Loss: 77.7675, Test Loss: 106.9253, R²: 0.9089, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [189/300], Train Loss: 78.2326, Test Loss: 106.3574, R²: 0.9094, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [190/300], Train Loss: 77.6650, Test Loss: 104.0279, R²: 0.9114, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [191/300], Train Loss: 77.5641, Test Loss: 104.1283, R²: 0.9113, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [192/300], Train Loss: 76.9101, Test Loss: 107.7099, R²: 0.9083, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [193/300], Train Loss: 76.1134, Test Loss: 104.7545, R²: 0.9108, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [194/300], Train Loss: 75.5435, Test Loss: 104.1632, R²: 0.9113, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [195/300], Train Loss: 76.6897, Test Loss: 105.0660, R²: 0.9105, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [196/300], Train Loss: 76.4874, Test Loss: 105.1182, R²: 0.9105, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [197/300], Train Loss: 75.8867, Test Loss: 105.0805, R²: 0.9105, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [198/300], Train Loss: 75.8144, Test Loss: 105.5455, R²: 0.9101, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [199/300], Train Loss: 76.6933, Test Loss: 103.5969, R²: 0.9118, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [200/300], Train Loss: 76.0736, Test Loss: 105.0172, R²: 0.9106, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [201/300], Train Loss: 76.1791, Test Loss: 104.7694, R²: 0.9108, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [202/300], Train Loss: 74.9583, Test Loss: 103.3974, R²: 0.9119, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [203/300], Train Loss: 76.3749, Test Loss: 102.8327, R²: 0.9124, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [204/300], Train Loss: 75.2935, Test Loss: 102.9973, R²: 0.9123, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [205/300], Train Loss: 75.4039, Test Loss: 102.8310, R²: 0.9124, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [206/300], Train Loss: 75.3082, Test Loss: 104.4713, R²: 0.9110, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [207/300], Train Loss: 75.9587, Test Loss: 105.9349, R²: 0.9098, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [208/300], Train Loss: 75.9592, Test Loss: 105.3297, R²: 0.9103, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [209/300], Train Loss: 76.2975, Test Loss: 103.4590, R²: 0.9119, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [210/300], Train Loss: 75.2784, Test Loss: 103.4992, R²: 0.9119, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [211/300], Train Loss: 75.9316, Test Loss: 104.3947, R²: 0.9111, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [212/300], Train Loss: 74.2886, Test Loss: 102.8331, R²: 0.9124, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [213/300], Train Loss: 74.6235, Test Loss: 104.6244, R²: 0.9109, LR: 0.000125
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [214/300], Train Loss: 74.6573, Test Loss: 103.6543, R²: 0.9117, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [215/300], Train Loss: 72.6380, Test Loss: 102.5756, R²: 0.9126, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [216/300], Train Loss: 74.0418, Test Loss: 103.2851, R²: 0.9120, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [217/300], Train Loss: 73.4114, Test Loss: 103.5535, R²: 0.9118, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [218/300], Train Loss: 72.9311, Test Loss: 102.3030, R²: 0.9129, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [219/300], Train Loss: 71.9582, Test Loss: 103.1222, R²: 0.9122, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [220/300], Train Loss: 73.4097, Test Loss: 102.3198, R²: 0.9129, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [221/300], Train Loss: 72.2150, Test Loss: 103.8240, R²: 0.9116, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [222/300], Train Loss: 72.4820, Test Loss: 103.9491, R²: 0.9115, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [223/300], Train Loss: 73.3073, Test Loss: 103.7576, R²: 0.9116, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [224/300], Train Loss: 71.6025, Test Loss: 102.8007, R²: 0.9124, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [225/300], Train Loss: 72.4473, Test Loss: 103.3195, R²: 0.9120, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [226/300], Train Loss: 73.4258, Test Loss: 103.9788, R²: 0.9114, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [227/300], Train Loss: 72.5482, Test Loss: 102.9527, R²: 0.9123, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [228/300], Train Loss: 72.5381, Test Loss: 102.8934, R²: 0.9124, LR: 0.000063
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [229/300], Train Loss: 72.7140, Test Loss: 102.6212, R²: 0.9126, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [230/300], Train Loss: 72.2363, Test Loss: 102.4807, R²: 0.9127, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [231/300], Train Loss: 71.3388, Test Loss: 102.3187, R²: 0.9129, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [232/300], Train Loss: 71.6480, Test Loss: 101.9728, R²: 0.9132, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [233/300], Train Loss: 72.0245, Test Loss: 101.8442, R²: 0.9133, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [234/300], Train Loss: 71.1430, Test Loss: 102.1222, R²: 0.9130, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [235/300], Train Loss: 70.3393, Test Loss: 102.0834, R²: 0.9131, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [236/300], Train Loss: 70.6859, Test Loss: 101.9299, R²: 0.9132, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [237/300], Train Loss: 71.5996, Test Loss: 102.1592, R²: 0.9130, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [238/300], Train Loss: 71.2957, Test Loss: 101.8534, R²: 0.9133, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [239/300], Train Loss: 71.3955, Test Loss: 101.8524, R²: 0.9133, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [240/300], Train Loss: 71.2293, Test Loss: 102.4670, R²: 0.9127, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [241/300], Train Loss: 70.2799, Test Loss: 102.4758, R²: 0.9127, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [242/300], Train Loss: 70.9848, Test Loss: 101.9696, R²: 0.9132, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [243/300], Train Loss: 70.8215, Test Loss: 102.2101, R²: 0.9130, LR: 0.000031
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [244/300], Train Loss: 72.0167, Test Loss: 102.4557, R²: 0.9127, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [245/300], Train Loss: 70.1351, Test Loss: 101.8425, R²: 0.9133, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [246/300], Train Loss: 71.5477, Test Loss: 102.3132, R²: 0.9129, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [247/300], Train Loss: 69.9316, Test Loss: 101.8675, R²: 0.9132, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [248/300], Train Loss: 69.3364, Test Loss: 102.0435, R²: 0.9131, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [249/300], Train Loss: 70.1682, Test Loss: 102.0022, R²: 0.9131, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [250/300], Train Loss: 70.2613, Test Loss: 102.1679, R²: 0.9130, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [251/300], Train Loss: 70.6862, Test Loss: 102.1988, R²: 0.9130, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [252/300], Train Loss: 70.5034, Test Loss: 102.1854, R²: 0.9130, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [253/300], Train Loss: 70.5335, Test Loss: 102.0319, R²: 0.9131, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [254/300], Train Loss: 70.1650, Test Loss: 102.3502, R²: 0.9128, LR: 0.000016
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [255/300], Train Loss: 70.4972, Test Loss: 102.7145, R²: 0.9125, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [256/300], Train Loss: 70.5522, Test Loss: 102.4048, R²: 0.9128, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [257/300], Train Loss: 70.1056, Test Loss: 102.4393, R²: 0.9128, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [258/300], Train Loss: 70.6755, Test Loss: 102.3876, R²: 0.9128, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [259/300], Train Loss: 70.4742, Test Loss: 102.3794, R²: 0.9128, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [260/300], Train Loss: 71.2242, Test Loss: 102.2657, R²: 0.9129, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [261/300], Train Loss: 71.4070, Test Loss: 102.3136, R²: 0.9129, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [262/300], Train Loss: 70.6482, Test Loss: 102.2889, R²: 0.9129, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [263/300], Train Loss: 70.3833, Test Loss: 102.4371, R²: 0.9128, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [264/300], Train Loss: 70.1241, Test Loss: 102.0860, R²: 0.9131, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [265/300], Train Loss: 70.4684, Test Loss: 102.1343, R²: 0.9130, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [266/300], Train Loss: 68.5310, Test Loss: 101.9838, R²: 0.9131, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [267/300], Train Loss: 70.1374, Test Loss: 102.0574, R²: 0.9131, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [268/300], Train Loss: 69.8771, Test Loss: 101.9855, R²: 0.9131, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [269/300], Train Loss: 69.8220, Test Loss: 102.0329, R²: 0.9131, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [270/300], Train Loss: 70.5164, Test Loss: 102.0050, R²: 0.9131, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [271/300], Train Loss: 70.3739, Test Loss: 102.0847, R²: 0.9131, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [272/300], Train Loss: 70.0358, Test Loss: 102.0965, R²: 0.9130, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [273/300], Train Loss: 70.0270, Test Loss: 101.9282, R²: 0.9132, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [274/300], Train Loss: 70.0498, Test Loss: 102.0368, R²: 0.9131, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [275/300], Train Loss: 69.5625, Test Loss: 101.8925, R²: 0.9132, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [276/300], Train Loss: 68.9342, Test Loss: 101.8813, R²: 0.9132, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [277/300], Train Loss: 69.6253, Test Loss: 101.7327, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [278/300], Train Loss: 71.1188, Test Loss: 101.6924, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [279/300], Train Loss: 69.8624, Test Loss: 101.7949, R²: 0.9133, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [280/300], Train Loss: 69.0457, Test Loss: 101.5417, R²: 0.9135, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [281/300], Train Loss: 69.3245, Test Loss: 101.6195, R²: 0.9135, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [282/300], Train Loss: 69.3318, Test Loss: 101.7367, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [283/300], Train Loss: 68.9941, Test Loss: 101.4805, R²: 0.9136, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [284/300], Train Loss: 70.2911, Test Loss: 101.3699, R²: 0.9137, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [285/300], Train Loss: 70.4529, Test Loss: 101.5350, R²: 0.9135, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [286/300], Train Loss: 69.8752, Test Loss: 101.6284, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [287/300], Train Loss: 70.1583, Test Loss: 101.7009, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [288/300], Train Loss: 69.5980, Test Loss: 101.7361, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [289/300], Train Loss: 70.0377, Test Loss: 101.6372, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [290/300], Train Loss: 69.6558, Test Loss: 101.5187, R²: 0.9135, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [291/300], Train Loss: 70.0819, Test Loss: 101.4368, R²: 0.9136, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [292/300], Train Loss: 69.3397, Test Loss: 101.6065, R²: 0.9135, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [293/300], Train Loss: 70.9807, Test Loss: 101.8003, R²: 0.9133, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [294/300], Train Loss: 68.8069, Test Loss: 101.6974, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [295/300], Train Loss: 69.4452, Test Loss: 101.6645, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [296/300], Train Loss: 70.3742, Test Loss: 101.9141, R²: 0.9132, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [297/300], Train Loss: 69.9401, Test Loss: 101.6958, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [298/300], Train Loss: 69.6714, Test Loss: 101.7018, R²: 0.9134, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [299/300], Train Loss: 69.8489, Test Loss: 101.8588, R²: 0.9133, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Epoch [300/300], Train Loss: 69.1728, Test Loss: 101.7469, R²: 0.9133, LR: 0.000010
Loss and R² curves saved to ./figures/supercon_metrics.pdf
Training completed!
Best test loss: 101.3699
Best R² score: 0.9137
RMSE on test set: 10.0870 K

Sample predictions:
True: 5.7K, Predicted: 5.9K
True: 108.0K, Predicted: 110.6K
True: 2.3K, Predicted: 8.3K
True: 4.2K, Predicted: 4.6K
True: 128.0K, Predicted: 113.8K
Prediction vs. true values plot saved to ./figures/supercon_predictions.pdf

Process finished with exit code 0


```