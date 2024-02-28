import torch

# Define the tensor
tensor = torch.tensor([[ 537.7204,  357.1349,  689.9744,  703.6920],
                       [1034.6700,  352.0581,  489.8850,  699.1428],
                       [ 116.4958,  389.3210,  232.9917,  642.2138],
                       [ 862.8401,  251.8504,  193.2795,  107.3292],
                       [1173.2361,  659.6425,  150.2367,   55.9897],
                       [ 860.0164,  299.1110,  202.4672,  202.3514],
                       [ 153.4936,  351.3019,  306.9871,  692.4664],
                       [ 747.3392,   61.3405,   40.3863,   70.6922]], device='cuda:0')

# Convert tensor to NumPy array
numpy_array = tensor.cpu().numpy()

# Iterate over each numerical value in the tensor
for row in numpy_array:
    for number in row:
        print(number)