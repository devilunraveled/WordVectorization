from collections import Counter
from functools import cache
from typing import Optional
from nltk.tokenize import word_tokenize as Tokenizer
import re
from bidict import bidict
from ordered_set import OrderedSet
from scipy.sparse import lil_matrix as LilMatrix
from scipy.sparse.linalg import svds as SparseSVD
from numpy.linalg import svd as DenseSVD
from alive_progress import alive_bar
import pickle
import torch
import torch.optim as Optimizer
import torch.nn as NeuralNetwork
from torch import sigmoid as Sigmoid
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import random

from .Config import SVDConfig, Structure, Word2VecConfig

class Dataset:
    def __init__(self, 
                 trainPath : str  = Structure.trainPath,
                 testPath  : str  = Structure.testPath,
                 forceNew  : bool = False,
                 fileName  : str  = "dataset.pickle"
                 ):
        """
            @param trainPath: Path to train.csv
            @param testPath: Path to test.csv
        """
        self.trainPath = trainPath
        self.testPath = testPath
        self.fileName = fileName
  
        if not forceNew :
            loaded = self.loadDataset()
            if loaded :
                return
        
        self.vocab = OrderedSet(set())
        self.data = []
        self.tokenizedData = []

    def saveDataset(self) -> None:
        try :
            savePathFile = Structure.corpusPath + "/" + self.fileName
            with open(savePathFile, "wb") as f:
                pickle.dump(self, f)
        except:
            print("Failed to save dataset.")
    def loadDataset(self) -> None:
        try :
            savePathFile = Structure.corpusPath + "/" + self.fileName
            with open(savePathFile, "rb") as f:
                self.__dict__.update(pickle.load(f).__dict__)
        except:
            print("No dataset found. Dataset will be loaded from scratch.")
    
    def getVocabulary(self) -> OrderedSet | set:
        try :
            if not self.vocab :
                self.getData()
        except:
            import traceback 
            traceback.print_exc()
        finally:
            return self.vocab

    def getData(self) -> Optional[list]:
        try :
            if self.data :
                return self.data
            print("Parsing data...")
            with open(self.trainPath, "r") as f:
                self.vocab = {SVDConfig.startToken, SVDConfig.endToken, SVDConfig.unkToken}
                skipped = False
                numLines = sum(1 for _ in f)
                f.seek(0)
                with alive_bar(numLines-1) as bar:
                    for line in f:
                        # Skip the first line as it contains meta information.
                        if not skipped: 
                            skipped = True
                            continue
                        
                        description : str = ' '.join(line.split(",")[1:]).strip()
                        description = re.sub(SVDConfig.cleanser, " ", description)

                        self.data.append( SVDConfig.startToken + ' ' + description + ' ' + SVDConfig.endToken )
                        self.tokenizedData.append([ word for word in Tokenizer(description) ])
                        self.vocab.update(set(self.tokenizedData[-1]))
                        bar()
        except:
            import traceback 
            traceback.print_exc()
        finally:
            return self.data

    def getCoocurenceMatrix(self, sparse : bool = False) -> Optional[dict] | Optional[list[list]]:
        """ 
            This method returns a co-occurence matrix, 
            it can be of a sparse representation if the 
            sparse parameter is set True. Otherwise, it 
            creates the co-occurence based on the first k 
            words as found.
            @param sparse: Set to True if you want a sparse matrix representation.
        """
        try :
            if sparse:
                self.SparseCoOcMatrix = self.__getSparseCoocurenceMatrix__()
            else:
                self.StochasticCoOcMatrix = self.__getStochasticCoocurenceMatrix__()
        except:
            import traceback 
            traceback.print_exc()
        finally:
            self.saveDataset()
    
    def __getSparseCoocurenceMatrix__(self):
        """
            Returns a sparse co-occurence matrix
            as a list of lists.
        """
        if not hasattr(self, "mapping") :
            self.__constructBidict__()
        
        print("Creating sparse co-occurence matrix...")
        matrixDimension = len(self.vocab)
        sparseMatrix = LilMatrix((matrixDimension, matrixDimension), dtype = "float64")
        
        with alive_bar(len(self.tokenizedData)) as bar:
            for sentence in self.tokenizedData:
                for index, token in enumerate(sentence):
                    windowStart = max(0, index - SVDConfig.contextWindow)
                    for windowToken in sentence[windowStart : index]:
                        sparseMatrix[self.mapping[token], self.mapping[windowToken]] += 1
                        sparseMatrix[self.mapping[windowToken], self.mapping[token]] += 1
                bar()
        return sparseMatrix

    def __getStochasticCoocurenceMatrix__(self):
        """
            Returns a stochastic co-occurence matrix
            based on the number of words in the config.
        """

        if not self.mapping:
            self.__constructBidict__()

        def acceptable(word : str ) -> bool :
            if self.mapping[word] < SVDConfig.numWords:
                return True
            return False
        
        coOccMat = [ [ 0 for _ in range(SVDConfig.numWords) ] for _ in range(SVDConfig.numWords) ]

        print("Creating stochastic co-occurence matrix...")
        
        for i in range(min( len(self.vocab), SVDConfig.numWords)):
            word = self.mapping.inverse[i]
            context = self.__getWordContext__(word)
            for contextWord,freq in context.items():
                if acceptable(contextWord):
                    coOccMat[i][self.mapping[contextWord]] = freq
         
        return coOccMat

    def __getContextInLines__(self, word : str, line : list[str]) -> Optional[list]:
        """
            Returns the context of the word in a line.
            @param word: The word to search for.
            @param line: The line to search in.
            @return: The context of the word in the line.
        """
        context = []
        try :
            indices = [ i for i,token in enumerate(line) if token == word ]
            
            for index in indices:
                start = max(0,index - SVDConfig.contextWindow)
                end = min(len(line) - 1, index + SVDConfig.contextWindow)
                
                context.extend(line[start : end + 1])
        except:
            import traceback 
            traceback.print_exc()
        finally:
            return context
    
    @cache
    def __getWordContext__(self, word : str) -> Counter:
        """
            Returns the context of the word in the corpus.
            @param word: The word to search for.
            @return: The context of the word in the corpus.
        """
        context = Counter()
        try :
            if not self.tokenizedData :
                self.getData()

            for line in self.tokenizedData:
                context.update(self.__getContextInLines__(word = word, line = line))
            
        except:
            import traceback 
            traceback.print_exc()
        finally:
            return context

    def __constructBidict__(self):
        """
            Constructs a bidirectional mapping from the vocabulary to 
            the index for ease of mapping.
        """
        try :
            if not self.vocab :
                self.getData()

            self.mapping = bidict({ word : index for index, word in enumerate(self.vocab)} )
            ## Replace the first three mappings to the custom tokens.
        except:
            import traceback 
            traceback.print_exc()
    
    def performSVD(self, sparse = True):
        """
            Performs SVD on the co-occurence matrix
            @param sparse: Set to True if you want a sparse matrix representation.
        """
        
        try :
            if hasattr(self, 'U') :
                print("SVD already performed. Skipping...")
                return

            if sparse:
                if ( not hasattr(self, 'SparseCoOcMatrix') ):
                    self.getCoocurenceMatrix(sparse = True)
                
                print("Computing SVD on sparse co-occurence matrix...")
                U,S,Vt = SparseSVD(A = self.SparseCoOcMatrix, k = min(SVDConfig.EmbeddingSize, len(self.vocab) - 1) )

                self.U = U
                self.S = S
                self.Vt = Vt
            else:
                if ( not hasattr(self, 'StochasticCoOcMatrix') ):
                    self.getCoocurenceMatrix(sparse = False)
                
                U,S,Vt = DenseSVD(self.StochasticCoOcMatrix)
                
                self.U = U
                self.S = S
                self.Vt = Vt

            self.saveDataset()
        except:
            import traceback
            traceback.print_exc()

    def prepareData(self):
        try :
            self.getData()
            self.__constructBidict__()
            
            self.word2VecDataPoints = []
            print(f"Preparing data for Word2Vec...")
            with alive_bar(len(self.tokenizedData)) as bar:
                for sentence in self.tokenizedData:
                    for index, token in enumerate(sentence):
                        window = sentence[max(0,index - SVDConfig.contextWindow) : index] + sentence[index + 1 : index + 1 + SVDConfig.contextWindow]
                        for word in window:
                            if word in self.vocab:
                                self.word2VecDataPoints.append( (self.mapping[token], self.mapping[word], True) )
                            for _ in range(Word2VecConfig.negativeSamples):
                                randomWord = random.randint(0, len(self.vocab) - 1)
                                self.word2VecDataPoints.append( (self.mapping[token], randomWord, False) )
                    bar()
        except:
            import traceback 
            traceback.print_exc()
    
    def getWordEmbedding(self, word : str) -> list:
        """
            Get the embedding for the word in the vocabulary.
        """
        try :
            if not hasattr(self,'U') :
                self.performSVD(sparse = True)
        except:
            import traceback 
            traceback.print_exc()
            return []
        finally:
            if self.U is None:
                return []
            return self.U[self.mapping[word]]
    
    def getWord2VecEmbedding(self) -> None:
        """
            Get the embedding for the word in the vocabulary.
        """
        
        self.prepareData()
        
        try :
            if not hasattr(self, 'word2Vec'):
                self.word2Vec = Word2Vec(mapping = self.mapping, word2VecDataPoints = self.word2VecDataPoints)
        except:
            import traceback 
            traceback.print_exc()
        finally:
            if self.word2Vec is not None:
                self.word2Vec.trainEmbeddings()
    
class CustomDataset(TorchDataset):
    def __init__(self, word2VecDataPoints : list):
        self.word2VecDataPoints = word2VecDataPoints
    
    def __len__(self):
        return len(self.word2VecDataPoints)

    def __getitem__(self, index):
        point = self.word2VecDataPoints[index]
        return torch.tensor(point[0], dtype = torch.int64), torch.tensor(point[1], dtype = torch.int64), torch.tensor(point[2], dtype = torch.float)

class Word2Vec(NeuralNetwork.Module):
    def __init__(self, mapping : bidict, embeddingSize : int = Word2VecConfig.EmbeddingSize, word2VecDataPoints : list = [] ) -> None:
        self.mapping = mapping
        super().__init__()

        self.EmbeddingSize = embeddingSize
        self.contextEmbedding = NeuralNetwork.Embedding(num_embeddings = len(self.mapping), embedding_dim = embeddingSize)
        self.wordEmbedding = NeuralNetwork.Embedding(num_embeddings = len(self.mapping), embedding_dim = embeddingSize)
        self.criterion = NeuralNetwork.BCELoss()
        self.optimizer = Optimizer.Adam(self.parameters(), lr = 0.01)
        self.dataset = CustomDataset(word2VecDataPoints)
        self.data = DataLoader(self.dataset, batch_size = Word2VecConfig.batchSize, shuffle = True)
        self.to(torch.device(('cuda' if torch.cuda.is_available() else 'cpu')))

    def forward(self, word, context):
        out1 = self.contextEmbedding(context)
        out2 = self.wordEmbedding(word)
        out = torch.bmm(out1.unsqueeze(1), out2.unsqueeze(2)).squeeze()
        return Sigmoid(out)

    def trainEmbeddings(self, numEpochs : int = Word2VecConfig.epochs):
        try :
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(self, 'wordEmbeddings') :
                print("Word2Vec already trained. Skipping...")
                return
            
            for epoch in range(numEpochs):
                avgLoss = 0
                with alive_bar(len(self.data)) as bar:
                    for word, context, label in self.data:
                        word = word.to(device)
                        context = context.to(device)
                        label = label.to(device)
                        output = self(word, context)
                        loss = self.criterion(output, label)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        avgLoss += loss.item()
                        self.optimizer.step()
                        bar()
                    avgLoss /= len(self.dataset)
                    print(f"Epoch : {epoch+1}, Loss : {avgLoss:.4f}")
        except:
            import traceback
            traceback.print_exc()
