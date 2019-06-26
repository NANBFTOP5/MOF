
import numpy as np


trainfile="/home/suliangbu/work/wanghong/train_low_None_.pickle"
tmp = np.load(trainfile)

data = tmp.values


featur = np.zeros((186, 800, 1001))

for i in range(186):
	featur[i] = data[i * 800:(i+1) * 800, :1001]




def get_batch(self,batch_size,featur):
    """Create batch of n pairs, half same class, half different class"""
    n_examples = 800
    n_classes = 186

    
    categories = rng.choice(n_classes,size=(batch_size,),replace=False)
    
    pairs = [np.zeros((batch_size, 1001)) for i in range(2)]
    
    targets=np.zeros((batch_size))

    targets[batch_size//2:] = 1
    
    for i in range(batch_size):

        category = categories[i]
        idx_1 = rng.randint(0, n_examples)        
        pairs[0][i] = featur[category, idx_1]

        idx_2 = rng.randint(0, n_examples)

        if i >= batch_size // 2:
            category_2 = category  
        else: 
            category_2 = (category + rng.randint(1,n_classes)) % n_classes

        pairs[1][i] = featur[category_2,idx_2]

    return pairs, targets
