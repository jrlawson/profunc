import unittest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import numpy
from feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    '''
    Unit tests for the FeatureExtractor class. Does simple tests to insure that 
    the feature vector we get back is of the right length and has frequency
    data that makes sense. More tests should be added.
    ''' 
    def setUp(self):
        '''Sets up the test by constructing feature vectors to get tested'''       
        self.record1 = SeqRecord(Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF",
            IUPAC.protein),
            id="YP_025292.1", name="HokC",
            description="toxic membrane protein, small")        
        self.seq1 = self.record1.seq
        self.feature_extractor = FeatureExtractor()  
        self.feature_vector1 = self.feature_extractor.extract_features(self.seq1)
        
    def test_feature_vector_length(self):
        '''Tests that the feature vector is 400 elements long'''
        self.assertEqual(len(self.feature_vector1), 400, msg="Feature vector not 400 long")
        
    def test_dipeptide_frequency_sum(self):
        '''Tests that the dipeptide frequencies sum to 1'''
        checksum = 0.0
        for i in range(0,400):
            checksum += self.feature_vector1[i]
        self.assertAlmostEqual(checksum, 1.0, places=5, msg="Frequencies don't sum to 1")
        
        
if __name__ == '__main__':
    unittest.main()
