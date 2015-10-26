from array import array
from Bio import SeqIO
from Bio import Seq
from feature_extractor import FeatureExtractor

dir = "/home/jlawson/Dropbox/ProteinFunctionData/"      # Where the files live.
names =[                                                # Names of all of the files.
        "baseplate_3370", 
        "collar_1385", 
        "htj_2258_nofg", 
        "major_tail_1512", 
        "mcp_3589", 
        "minor_capsid_1500_nofg", 
        "minor_tail_2033", 
        "portal_2141",
        "tail_fiber_3007",
        "tail_sheath_2350"]

feature_extractor = FeatureExtractor()
raw_data = list()            
for i in range(0,len(names)) :
    print(names[i])
    handle = open(dir+names[i]+".faa", "rU")            # Open a file.
    for record in SeqIO.parse(handle, "fasta") :
        print(record.id)
        feature_vector = feature_extractor.extract_features(record)
        raw_data.append(feature_vector)
    handle.close() 
 
print len(raw_data)            
#for i in range(0,len(raw_data)) :
#    print(raw_data[i].id)
