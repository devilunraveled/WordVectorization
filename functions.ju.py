# %% [md]
"""
SVD Functions Testing.
"""

# %%
from src.parser import Dataset
import time 

# %%
mainDataset = Dataset(fileName = "train.pkl",forceNew = True)

# %%
sampleDataset = Dataset(trainPath = "./corpus/sample.csv", 
                        fileName = "sample.pkl",
                        forceNew = True)

# %%
# Performing SVD.
def svd( dataset : Dataset ):
    startTime = time.time()
    dataset.performSVD()
    print(f"Time taken to perform SVD : {time.time() - startTime:.2f}s")

# %% [md]
"""
Inspecting the SVD
"""

# %% 
def inspectSVD( dataset : Dataset ):
    print(f"U : {dataset.U}\nS : {dataset.S}\nVt : {dataset.Vt}")

# %% 
def inspectWordEmbedding( dataset : Dataset, word : str ):
    return dataset.getWordEmbedding(word = word)

# %%
mainDataset.getWord2VecEmbedding()
