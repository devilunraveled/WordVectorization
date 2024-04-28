# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
"""
Classification Testing.
"""

# %%
import sys
print(sys.executable)

# %%
# custom installation for notebook.
!pip3.12 install nltk

# %% 
from src.classification import RNNClassifier
from src.parser import Dataset
from src.Config import SVDConfig

# %%
dataset = Dataset(fileName = "model.pkl", svdLoad = True)

# %%
dataset.getData()

# %%
# Run only if SVD Embeddings are not precomputed.
dataset.setSVDEmbeddings()

# %%
dataset.getTestData()

# %%
trainData = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.labels, dataset.tokenizedData)]
testData = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.testLabels, dataset.testData)]


# %%
trainData[0][1][1]


# %%
print(dataset.getSVDEmbedding(trainData[0][1][1]))

# %% [markdown]
"""
### Using SVD Embeddings for Classification
"""

# %%
svdBasedClassifier = RNNClassifier(inputSize=256, embeddingMap=dataset.getSVDEmbedding, 
                                   data=trainData, testData=testData, 
                                   fileName = f"svdClassifier_{SVDConfig.contextWindow}.pt")
svdBasedClassifier.createDataset()

# %%
svdBasedClassifier.trainModel()
# svdBasedClassifier.loadModel(fileName = svdBasedClassifier.fileName)

# %%
svdBasedClassifier.evaluation()

# %%
svdBasedClassifier.plotConfusionMatrix(name=f"SVD_{SVDConfig.EmbeddingSize}_{SVDConfig.contextWindow}")
