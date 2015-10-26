import unittest
from amino_acid_mapping import alphabetic_amino_acid_mapping

class TestAminoAcidMapping(unittest.TestCase):
    '''
    Tests the AminoAcidMapping class to make sure that the mappings make sense.
    '''
    def setUp(self):
        self.mapping = alphabetic_amino_acid_mapping
    
    def test_dictionary_completeness(self):
        '''Tests to insure that each character maps to an integer from 0-20 and
        that no two characters map to the same value'''
        cover = [0]*20        
        targets = "ACDEFGHIKLMNPQRSTVWY"    # The set of amino acids
        for i in range(0,20):               # Test each of them
            letter = targets[i]
            value = self.mapping.amino_acid_to_index(letter)
            # Value is in proper range
            self.assertGreaterEqual(value,0,msg="Letter maps to value<0")
            self.assertLess(value,20,msg="Letter maps to value>=20")
            # No other letter maps to the same value.
            self.assertEquals(cover[value],0,msg="Letter maps to non-unique value")
            cover[value]=1
            
    def test_key_error_raised_on_invalid_amino_acid(self):
        '''Tests to insure that when an unknown amino acid character is given, 
        a KeyError is raised.'''
        with self.assertRaises(KeyError):
            self.mapping.amino_acid_to_index('Z')            
            
if __name__ == '__main__':
    unittest.main()            
