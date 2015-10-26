from Bio import SeqIO
import numpy
from amino_acid_mapping import alphabetic_amino_acid_mapping

class FeatureExtractor(object):
    """
    The job of a feature extractor is to take an example and identify its
    salient features. For this family of feature extractors, the examples are
    protein sequences (list of amino acids), and the extract_features() method
    does the work of producing the feature vector.
    """
    def __init__(self, amino_acid_mapping=alphabetic_amino_acid_mapping):
        """
        :param dictionary   a dictionary that maps the 20 characters of the amino
                            acid alphabet to a an integer from 0 to 20.
        :type  dictionary   dictionary
        """
        self.amino_acid_mapping = amino_acid_mapping
    

    def dipeptide_frequency_table(self, seq):
        """
        Produces a 20x20 table of dipeptide frequencies, where the row and 
        column correspond to the index of a particular amino acid in the 
        dictionary that defines this feature extractor. For instance, if the 
        dictionary maps alanine to 0 and cysteine to 1 (i.e. alphabetical order) 
        then as we scan the sequence, if we see an instance of AC, we augment
        the value in row 0, column 1. The values in the table are normalized
        so that the sum of all values is 1.
        
        :type  seq      Bio.Seq
        :param seq      the amino acid sequence to process. Note that we are
                        restricted to amino acid (not nucleotide) sequences.
                        
        :return a table representing the dipeptide frequencies.  
        """
        frequencies = numpy.zeros((20,20))       # 20x20 dipeptide pair frequencies.
        length = len(seq)           
        last_index = length-1                    # Allows for the length of the dipeptide.
        weight = 1.0/(length-1)                  # Each dipeptide pair contributes this much to overall frequency.
        for i in range (0, last_index):          # Look at each dipeptide pair.            
            row=self.amino_acid_mapping.amino_acid_to_index(seq[i])     # First character in the pair is the "row".
            col=self.amino_acid_mapping.amino_acid_to_index(seq[i+1])   # Second character in the pair is the "column".
            frequencies[row,col] += weight 
        return frequencies
            
            
    def extract_features(self, seq):
        """
        Extracts features from a sequence and returns a feature vecture. This
        base class version simply returns a 400-wide set of dipeptide 
        frequencies. It mimics a 20x20 array where the row and column of the
        array are provided by the indices in the dictionary.
        
        :param seq  The amino acid sequence (BioSeq form)
        :type  seq  Bio.Seq
        
        :return A feature vector for the sequence. In the case of this base
                class, that's just a flattened dipeptide frequency table (i.e. a
                400x1 array instead of a 20x20 array, and the first 20 elements
                are the first row from the tablem the second 20 are the second
                row, etc.
        """
        frequencies = self.dipeptide_frequency_table(seq)
        feature_vector = numpy.zeros(400)
        for row in range(0,20):
            for col in range(0,20):
                feature_vector[20*row + col] = frequencies[row,col]
        return feature_vector

    
