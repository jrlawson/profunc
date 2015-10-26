# This is a default dictionary for converting a letter in the IUPAC amino acid 
# alphabet to an integer index from 0 to 19. For this case, the index is just 
# alphabetical order.
#
# Other dictionaries can mimmick this with different ordering
iupac_aa_dictionary_ordered_alphabetically = {
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
    
class AminoAcidMapping(object):
    '''
    Maps that take a letter associated with an amino acid and convert it to an
    index from 0-19.
    
    :type  dictionary dictionary
    :param dictionary The map to be used. Default is the IUPAC amino acid 
                      dictionary ordered alphabetically.
    '''
    def __init__(self, dictionary=iupac_aa_dictionary_ordered_alphabetically):
        self.dictionary = dictionary
        
        
    def amino_acid_to_index(self, letter):
        """
        Converts a letter in the amino acid alphabet to an index
        using the mapping from the dictionary. If the letter is not in the 
        dictionary, a KeyError will be raised.
        
        :type  letter   character
        :param letter   the letter corresponding to the amino acid in the 
                        sequence. A for alanine, for instance, C for cystiene,
                        etc. All letters will be converted to uppercase before
                        processing.
        """
        return self.dictionary[letter.upper()]
        
        
alphabetic_amino_acid_mapping = AminoAcidMapping()

