from Bio import SeqIO
import numpy

# This is a default dictionary for converting a letter in the amino acid 
# alphabet to an integer index from 0 to 19. For this case, the index is just 
# alphabetical order.
#
# Other dictionaries can mimmick this with different ordering
alphabetical_aa_dictionary = {
    'A':0,    # Alanine
    'C':1,    # Cysteine
    'D':2,    # Aspartic AciD
    'E':3,    # Glutamic Acid
    'F':4,    # Phenylalanine
    'G':5,    # Glycine
    'H':6,    # Histidine
    'I':7,    # Isoleucine
    'K':8,    # Lysine
    'L':9,    # Leucine  
    'M':10,   # Methionine
    'N':11,   # AsparagiNe
    'P':12,   # Proline       
    'Q':13,   # Glutamine
    'R':14,   # ARginine
    'S':15,   # Serine
    'T':16,   # Threonine
    'V':17,   # Valine
    'W':18,   # Tryptophan
    'Y':19 }  # TYrosine


class FeatureExtractor(object):
    """
    The job of a feature extractor is to take an example and identify its
    salient features. For this family of feature extractors, the examples are
    protein sequences (list of amino acids), and the extract_features() method
    does the work of producing the feature vector.
    """
    def __init__(self, dictionary=alphabetical_aa_dictionary):
        """
        :param dictionary   a dictionary that maps the 20 characters of the amino
                            acid alphabet to a an integer from 0 to 20.
        :type  dictionary   dictionary
        """
        self.dictionary = dictionary

    
    def letter_to_index(self, letter):
        """
        Converts a letter to the row/column index in the 20x20 frequency matrix
        using the mapping from the dictionary that defines this feature 
        extractor.
        
        :type  letter   character
        :param letter   the letter corresponding to the amino acid in the 
                        sequence. A for alanine, for instance, C for cystiene,
                        etc. All letters will be converted to uppercase before
                        processing.
        """
        return self.dictionary[letter.upper()]
    

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
            row=self.letter_to_index(seq[i])     # First character in the pair is the "row".
            col=self.letter_to_index(seq[i+1])   # Second character in the pair is the "column".
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

import unittest
class TestFeatureExtractor(unittest.TestCase):
    def setUp(self): 
        self.feature_extractor = FeatureExtractor()
        
        self.record1 = Bio.SeqRecord(Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF",
            IUPAC.protein),
            id="YP_025292.1", name="HokC",
            description="toxic membrane protein, small")        
        self.seq1 = record1.seq 
        self.feature_vector1 = feature_extractor.extract_features(seq1)
        
    def test_feature_vector_length(self):
        self.assertEqual(len(self.feature_vector1), 400, msg="Feature vector not 400 long")
        
    def test_dipeptide_frequency_sum(self):
        checksum = 0.0
        for i in range(0,400):
            checksum += self.feature_vector1[i]
        self.assertAlmostEqual(checksum, 1.0, places=5, msg="Frequencies don't sum to 1") 
        
# A bit of test code.
#feature_extractor = FeatureExtractor()                   # Create feature extractor with default dictionary.
#handle = open('baseplate_3370.faa', "rU")                # Open a file.
#record = SeqIO.parse(handle, "fasta").next()             # Get the first sequence as a SeqRecord
#handle.close()                                           # Good housekeeping. We're done with it, close it.
#seq = record.seq                                         # You have to pull the sequence from the SeqRecord
#feature_vector = feature_extractor.extract_features(seq) # Get the feature vector.
#print feature_vector                                     # All 400 numbers should be small, most 0.0.


if __name__ == '__main__':
    unittest.main()
    
