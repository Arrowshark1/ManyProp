import random
from rdkit import Chem

def rand_mol(counter=0):
    if counter>10: # guards against infinite recursion
        return "CCC"
    smi = "C"
    chain_len = random.randint(1, 20)
    for _ in range(chain_len):
        new_str = ""
        rand = random.randint(0,18)
        match rand: 
            case 0:
                new_str="C"
            case 1:
                new_str="c1ccccc1"
            case 2:
                new_str="(O)"
            case 3:
                new_str="="
            case 4:
                new_str="(CC)"
            case 5:
                new_str="(CCC)"
            case 6:
                new_str="c1c"
            case 7:
                new_str="(=O)"
            case 8: 
                new_str="(N(O)=O)"
            case 9:
                new_str="C=1CCCCC1"
            case _: 
                new_str = "C"

        smi = smi+new_str
    smi = smi+"C"
    if Chem.MolFromSmiles(smi) is not None:
        return smi
    else:
        return rand_mol(counter=counter+1)