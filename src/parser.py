from collections import Counter, namedtuple
from functools import cache
from typing import Optional
from nltk.tokenize import word_tokenize as Tokenizer
import re
from bidict import bidict
from ordered_set import OrderedSet
from scipy.sparse import lil_matrix as LilMatrix
from scipy.sparse.linalg import svds as SparseSVD
from numpy.linalg import svd as DenseSVD

from .Config import SVDConfig, Structure

class Dataset:
    def __init__(self, 
                 trainPath : str = Structure.trainPath,
                 testPath  : str = Structure.testPath
                 ):
        """
            @param trainPath: Path to train.csv
            @param testPath: Path to test.csv
        """
        self.trainPath = trainPath
        self.testPath = testPath
        
        self.vocab = OrderedSet()
        self.data = []
        self.tokenizedData = []
        self.mapping = bidict()

    def getVocabulary(self) -> set:
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
            print("Parsing data...")
            with open(self.trainPath, "r") as f:
                self.vocab = {SVDConfig.startToken, SVDConfig.endToken, SVDConfig.unkToken}
                skipped = False
                
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
    
    def __getSparseCoocurenceMatrix__(self):
        """
            Returns a sparse co-occurence matrix
            as a list of lists.
        """
        if not self.mapping :
            self.__constructBidict__()
        
        matrixDimension = max(SVDConfig.EmbeddingSize, len(self.vocab))
        sparseMatrix = LilMatrix((matrixDimension, matrixDimension))

        for i in range(min( len(self.vocab), SVDConfig.numWords)):
            word = self.mapping.inverse[i]
            context = self.__getWordContext__(word).most_common(5)
            for contextWord,freq in context:
                sparseMatrix[i, self.mapping[contextWord]] = freq

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
    
    def performSVD(self, sparse = False):
        """
            Performs SVD on the co-occurence matrix
            @param sparse: Set to True if you want a sparse matrix representation.
        """
        
        try :
            if sparse:
                if ( not hasattr(self, 'SparseCoOcMatrix') ):
                    self.SparseCoOcMatrix = self.__getSparseCoocurenceMatrix__()

                U,S,Vt = SparseSVD(A = self.SparseCoOcMatrix, k = SVDConfig.EmbeddingSize)

                self.U = U
                self.S = S
                self.Vt = Vt
            else:
                if ( not hasattr(self, 'StochasticCoOcMatrix') ):
                    self.StochasticCoOcMatrix = self.__getStochasticCoocurenceMatrix__()
                
                U,S,Vt = DenseSVD(self.StochasticCoOcMatrix)
                
                self.U = U
                self.S = S
                self.Vt = Vt
        except:
            import traceback 
            traceback.print_exc()

    def prepareData(self):
        try :
            self.getData()
            self.__constructBidict__()
        except:
            import traceback 
            traceback.print_exc()

    def getFrequencies(self):
        pass
