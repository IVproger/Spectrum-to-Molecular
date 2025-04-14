# import random
# from collections import Counter

# import pandas as pd
# from tqdm import tqdm

# from rdkit import Chem
# from rdkit import RDLogger
# from rdkit.Chem import Descriptors

# random.seed(42)

# lg = RDLogger.logger()
# lg.setLevel(RDLogger.CRITICAL)

# def read_from_sdf(path):
#     res = []
#     app = False
#     with open(path, 'r') as f:
#         for line in tqdm(f.readlines(), desc='Loading SDF structures', leave=False):
#             if app:
#                 res.append(line.strip())
#                 app = False
#             if line.startswith('> <SMILES>'):
#                 app = True

#     return res

# def filter(mol):
#     try:
#         smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
#         mol = Chem.MolFromSmiles(smi)

#         if "." in smi:
#             return False
        
#         if Descriptors.MolWt(mol) >= 1500:
#             return False
        
#         for atom in mol.GetAtoms():
#             if atom.GetFormalCharge() != 0:
#                 return False
#     except:
#         return False
    
#     return True

# FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P'}

# def filter_with_atom_types(mol):
#     try:
#         smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
#         mol = Chem.MolFromSmiles(smi)

#         if "." in smi:
#             return False
        
#         if Descriptors.MolWt(mol) >= 1500:
#             return False
        
#         for atom in mol.GetAtoms():
#             if atom.GetFormalCharge() != 0:
#                 return False
#             if atom.GetSymbol() not in FILTER_ATOMS:
#                 return False
#     except:
#         return False
    
#     return True

# ########## CANOPUS DATASET ##########

# canopus_split = pd.read_csv('../data/canopus/splits/canopus_hplus_100_0.tsv', sep='\t')

# canopus_labels = pd.read_csv('../data/canopus/labels.tsv', sep='\t')
# canopus_labels["name"] = canopus_labels["spec"]
# canopus_labels = canopus_labels[["name", "smiles"]].reset_index(drop=True)

# canopus_labels = canopus_labels.merge(canopus_split, on="name")

# canopus_train_inchis = []
# canopus_test_inchis = []
# canopus_val_inchis = []

# for i in tqdm(range(len(canopus_labels)), desc="Converting CANOPUS SMILES to InChI", leave=False):
    
#     mol = Chem.MolFromSmiles(canopus_labels.loc[i, "smiles"])
#     smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
#     mol = Chem.MolFromSmiles(smi)
#     inchi = Chem.MolToInchi(mol)

#     if canopus_labels.loc[i, "split"] == "train":
#         if filter(mol):
#             canopus_train_inchis.append(inchi)
#     elif canopus_labels.loc[i, "split"] == "test":
#         canopus_test_inchis.append(inchi)
#     elif canopus_labels.loc[i, "split"] == "val":
#         canopus_val_inchis.append(inchi)

# canopus_train_df = pd.DataFrame(set(canopus_train_inchis), columns=["inchi"])
# canopus_train_df.to_csv("../data/fp2mol/canopus/preprocessed/canopus_train.csv", index=False)

# canopus_test_df = pd.DataFrame(canopus_test_inchis, columns=["inchi"])
# canopus_test_df.to_csv("../data/fp2mol/canopus/preprocessed/canopus_test.csv", index=False)

# canopus_val_df = pd.DataFrame(canopus_val_inchis, columns=["inchi"])
# canopus_val_df.to_csv("../data/fp2mol/canopus/preprocessed/canopus_val.csv", index=False)

# excluded_inchis = set(canopus_test_inchis + canopus_val_inchis)

# ########## MSG DATASET ##########

# msg_split = pd.read_csv('../data/msg/split.tsv', sep='\t')

# msg_labels = pd.read_csv('../data/msg/labels.tsv', sep='\t')
# msg_labels["name"] = msg_labels["spec"]
# msg_labels = msg_labels[["name", "smiles"]].reset_index(drop=True)

# msg_labels = msg_labels.merge(msg_split, on="name")

# msg_train_inchis = []
# msg_test_inchis = []
# msg_val_inchis = []

# for i in tqdm(range(len(msg_labels)), desc="Converting MSG SMILES to InChI", leave=False):
    
#     mol = Chem.MolFromSmiles(msg_labels.loc[i, "smiles"])
#     smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
#     mol = Chem.MolFromSmiles(smi)
#     inchi = Chem.MolToInchi(mol)

#     if msg_labels.loc[i, "split"] == "train":
#         if filter(mol):
#             msg_train_inchis.append(inchi)
#     elif msg_labels.loc[i, "split"] == "test":
#         msg_test_inchis.append(inchi)
#     elif msg_labels.loc[i, "split"] == "val":
#         msg_val_inchis.append(inchi)

# msg_train_df = pd.DataFrame(set(msg_train_inchis), columns=["inchi"])
# msg_train_df.to_csv("../data/fp2mol/msg/preprocessed/msg_train.csv", index=False)

# msg_test_df = pd.DataFrame(msg_test_inchis, columns=["inchi"])
# msg_test_df.to_csv("../data/fp2mol/msg/preprocessed/msg_test.csv", index=False)

# msg_val_df = pd.DataFrame(msg_val_inchis, columns=["inchi"])
# msg_val_df.to_csv("../data/fp2mol/msg/preprocessed/msg_val.csv", index=False)

# excluded_inchis.update(msg_test_inchis + msg_val_inchis)

# ########## HMDB DATASET ##########

# hmdb_set = set()
# raw_smiles = read_from_sdf('../data/fp2mol/raw/structures.sdf')
# for smi in tqdm(raw_smiles, desc='Cleaning HMDB structures', leave=False):
#     try:
#         mol = Chem.MolFromSmiles(smi)
#         smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
#         mol = Chem.MolFromSmiles(smi)
#         if filter_with_atom_types(mol):
#             hmdb_set.add(Chem.MolToInchi(mol))
#     except:
#         pass

# hmdb_inchis = list(hmdb_set)
# random.shuffle(hmdb_inchis)

# hmdb_train_inchis = hmdb_inchis[:int(0.95 * len(hmdb_inchis))]
# hmdb_val_inchis = hmdb_inchis[int(0.95 * len(hmdb_inchis)):]

# hmdb_train_inchis = [inchi for inchi in hmdb_train_inchis if inchi not in excluded_inchis]

# hmdb_train_df = pd.DataFrame(hmdb_train_inchis, columns=["inchi"])
# hmdb_train_df.to_csv("../data/fp2mol/hmdb/preprocessed/hmdb_train.csv", index=False)

# hmdb_val_df = pd.DataFrame(hmdb_val_inchis, columns=["inchi"])
# hmdb_val_df.to_csv("../data/fp2mol/hmdb/preprocessed/hmdb_val.csv", index=False)

# ########## DSSTox DATASET ##########

# dss_set_raw = set()
# for i in tqdm(range(1, 14), desc='Loading DSSTox structures', leave=False):
#     df = pd.read_excel(f'../data/fp2mol/raw/DSSToxDump{i}.xlsx')
#     dss_set_raw.update(df[df['SMILES'].notnull()]['SMILES'])

# dss_set = set()
# for smi in tqdm(dss_set_raw, desc='Cleaning DSSTox structures', leave=False):
#     try:
#         mol = Chem.MolFromSmiles(smi)
#         smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
#         mol = Chem.MolFromSmiles(smi)
#         if filter_with_atom_types(mol):
#             dss_set.add(Chem.MolToInchi(mol))
#     except:
#         pass

# dss_inchis = list(dss_set)
# random.shuffle(dss_inchis)

# dss_train_inchis = dss_inchis[:int(0.95 * len(dss_inchis))]
# dss_val_inchis = dss_inchis[int(0.95 * len(dss_inchis)):]

# dss_train_inchis = [inchi for inchi in dss_train_inchis if inchi not in excluded_inchis]

# dss_train_df = pd.DataFrame(dss_train_inchis, columns=["inchi"])
# dss_train_df.to_csv("../data/fp2mol/dss/preprocessed/dss_train.csv", index=False)

# dss_val_df = pd.DataFrame(dss_val_inchis, columns=["inchi"])
# dss_val_df.to_csv("../data/fp2mol/dss/preprocessed/dss_val.csv", index=False)

# ########## COCONUT DATASET ##########

# coconut_df = pd.read_csv('../data/fp2mol/raw/coconut_csv-03-2025.csv')

# coconut_set_raw = set(coconut_df["canonical_smiles"])

# coconut_set = set()
# for smi in tqdm(coconut_set_raw, desc='Cleaning COCONUT structures', leave=False):
#     try:
#         mol = Chem.MolFromSmiles(smi)
#         smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
#         mol = Chem.MolFromSmiles(smi)
#         if filter_with_atom_types(mol):
#             coconut_set.add(Chem.MolToInchi(mol))
#     except:
#         pass

# coconut_inchis = list(coconut_set)
# random.shuffle(coconut_inchis)

# coconut_train_inchis = coconut_inchis[:int(0.95 * len(coconut_inchis))]
# coconut_val_inchis = coconut_inchis[int(0.95 * len(coconut_inchis)):]

# coconut_train_inchis = [inchi for inchi in coconut_train_inchis if inchi not in excluded_inchis]

# coconut_train_df = pd.DataFrame(coconut_train_inchis, columns=["inchi"])
# coconut_train_df.to_csv("../data/fp2mol/coconut/preprocessed/coconut_train.csv", index=False)

# coconut_val_df = pd.DataFrame(coconut_val_inchis, columns=["inchi"])
# coconut_val_df.to_csv("../data/fp2mol/coconut/preprocessed/coconut_val.csv", index=False)


# ########## MOSES DATASET ##########

# moses_df = pd.read_csv('../data/fp2mol/raw/moses.csv')

# moses_set_raw = set(moses_df["SMILES"])

# moses_set = set()
# for smi in tqdm(moses_set_raw, desc='Cleaning MOSES structures', leave=False):
#     try:
#         mol = Chem.MolFromSmiles(smi)
#         smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
#         mol = Chem.MolFromSmiles(smi)
#         if filter_with_atom_types(mol):
#             moses_set.add(Chem.MolToInchi(mol))
#     except:
#         pass

# moses_inchis = list(moses_set)
# random.shuffle(moses_inchis)

# moses_train_inchis = moses_inchis[:int(0.95 * len(moses_inchis))]
# moses_val_inchis = moses_inchis[int(0.95 * len(moses_inchis)):]

# moses_train_inchis = [inchi for inchi in moses_train_inchis if inchi not in excluded_inchis]

# moses_train_df = pd.DataFrame(moses_train_inchis, columns=["inchi"])
# moses_train_df.to_csv("../data/fp2mol/moses/preprocessed/moses_train.csv", index=False)

# moses_val_df = pd.DataFrame(moses_val_inchis, columns=["inchi"])
# moses_val_df.to_csv("../data/fp2mol/moses/preprocessed/moses_val.csv", index=False)

# ########## COMBINED DATASET ##########

# combined_inchis = hmdb_inchis + dss_inchis + coconut_inchis + moses_inchis
# combined_inchis = list(set(combined_inchis))
# random.shuffle(combined_inchis)

# combined_train_inchis = combined_inchis[:int(0.95 * len(combined_inchis))]
# combined_val_inchis = combined_inchis[int(0.95 * len(combined_inchis)):]
# combined_train_inchis = [inchi for inchi in combined_train_inchis if inchi not in excluded_inchis]

# combined_train_df = pd.DataFrame(combined_train_inchis, columns=["inchi"])
# combined_train_df.to_csv("../data/fp2mol/combined/preprocessed/combined_train.csv", index=False)

# combined_val_df = pd.DataFrame(combined_val_inchis, columns=["inchi"])
# combined_val_df.to_csv("../data/fp2mol/combined/preprocessed/combined_val.csv", index=False)

import random
from collections import Counter
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

random.seed(42)

# Suppress RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Number of processes to use (adjust based on your CPU)
N_CORES = mp.cpu_count() - 1

def read_from_sdf(path):
    res = []
    with open(path, 'r') as f:
        lines = f.readlines()
        
    app = False
    for line in lines:
        if app:
            res.append(line.strip())
            app = False
        if line.startswith('> <SMILES>'):
            app = True
            
    return res

def filter(mol):
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)

        if mol is None or "." in smi:
            return False
        
        if Descriptors.MolWt(mol) >= 1500:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
    except:
        return False
    
    return True

FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P'}

def filter_with_atom_types(mol):
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)

        if mol is None or "." in smi:
            return False
        
        if Descriptors.MolWt(mol) >= 1500:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
            if atom.GetSymbol() not in FILTER_ATOMS:
                return False
    except:
        return False
    
    return True

def process_smiles_batch(smiles_list, filter_func=filter):
    """Process a batch of SMILES strings in parallel"""
    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
                
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            mol = Chem.MolFromSmiles(smi)
            
            if filter_func(mol):
                results.append(Chem.MolToInchi(mol))
        except:
            continue
    return results

def process_dataset_row(row, column="smiles", filter_func=filter):
    """Process a single dataset row"""
    try:
        mol = Chem.MolFromSmiles(row[column])
        if mol is None:
            return None
            
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)
        
        if filter_func(mol):
            return {"split": row.get("split", "train"), "inchi": Chem.MolToInchi(mol)}
    except:
        pass
    return None

def chunks(lst, n):
    """Split list into n chunks"""
    chunk_size = max(1, len(lst) // n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def ensure_dir(file_path):
    """Make sure directory exists"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Create a process pool
    pool = mp.Pool(N_CORES)
    
    # Ensure output directories exist
    for dataset in ['canopus', 'msg', 'hmdb', 'dss', 'coconut', 'moses', 'combined']:
        ensure_dir(f"../data/fp2mol/{dataset}/preprocessed/")
        
    #################################################
    # Process CANOPUS dataset
    #################################################
    print("Processing CANOPUS dataset...")
    canopus_split = pd.read_csv('../data/canopus/splits/canopus_hplus_100_0.tsv', sep='\t')
    canopus_labels = pd.read_csv('../data/canopus/labels.tsv', sep='\t')
    canopus_labels["name"] = canopus_labels["spec"]
    canopus_labels = canopus_labels[["name", "smiles"]].merge(canopus_split, on="name")
    
    # Process rows in parallel
    canopus_results = list(tqdm(
        pool.imap(partial(process_dataset_row, column="smiles"), 
                  [row for _, row in canopus_labels.iterrows()]),
        total=len(canopus_labels),
        desc="Processing CANOPUS"
    ))
    
    # Filter None results
    canopus_results = [r for r in canopus_results if r]
    
    # Split by dataset split
    canopus_train = [r["inchi"] for r in canopus_results if r["split"] == "train"]
    canopus_test = [r["inchi"] for r in canopus_results if r["split"] == "test"]
    canopus_val = [r["inchi"] for r in canopus_results if r["split"] == "val"]
    
    # Save results
    pd.DataFrame(set(canopus_train), columns=["inchi"]).to_csv("../data/fp2mol/canopus/preprocessed/canopus_train.csv", index=False)
    pd.DataFrame(canopus_test, columns=["inchi"]).to_csv("../data/fp2mol/canopus/preprocessed/canopus_test.csv", index=False)
    pd.DataFrame(canopus_val, columns=["inchi"]).to_csv("../data/fp2mol/canopus/preprocessed/canopus_val.csv", index=False)
    
    # Keep track of excluded inchis
    excluded_inchis = set(canopus_test + canopus_val)
    
    #################################################
    # Process MSG dataset
    #################################################
    print("Processing MSG dataset...")
    msg_split = pd.read_csv('../data/msg/split.tsv', sep='\t')
    msg_labels = pd.read_csv('../data/msg/labels.tsv', sep='\t')
    msg_labels["name"] = msg_labels["spec"]
    msg_labels = msg_labels[["name", "smiles"]].merge(msg_split, on="name")
    
    # Process rows in parallel
    msg_results = list(tqdm(
        pool.imap(partial(process_dataset_row, column="smiles"), 
                  [row for _, row in msg_labels.iterrows()]),
        total=len(msg_labels),
        desc="Processing MSG"
    ))
    
    # Filter None results
    msg_results = [r for r in msg_results if r]
    
    # Split by dataset split
    msg_train = [r["inchi"] for r in msg_results if r["split"] == "train"]
    msg_test = [r["inchi"] for r in msg_results if r["split"] == "test"]
    msg_val = [r["inchi"] for r in msg_results if r["split"] == "val"]
    
    # Save results
    pd.DataFrame(set(msg_train), columns=["inchi"]).to_csv("../data/fp2mol/msg/preprocessed/msg_train.csv", index=False)
    pd.DataFrame(msg_test, columns=["inchi"]).to_csv("../data/fp2mol/msg/preprocessed/msg_test.csv", index=False)
    pd.DataFrame(msg_val, columns=["inchi"]).to_csv("../data/fp2mol/msg/preprocessed/msg_val.csv", index=False)
    
    # Update excluded inchis
    excluded_inchis.update(msg_test + msg_val)
    
    #################################################
    # Process HMDB dataset
    #################################################
    print("Processing HMDB dataset...")
    raw_smiles = read_from_sdf('../data/fp2mol/raw/structures.sdf')
    
    # Split input into chunks and process in parallel
    smiles_chunks = chunks(raw_smiles, N_CORES * 4)
    hmdb_results = []
    
    for chunk_results in tqdm(
        pool.imap(partial(process_smiles_batch, filter_func=filter_with_atom_types), 
                  smiles_chunks),
        total=len(smiles_chunks),
        desc="Processing HMDB"
    ):
        hmdb_results.extend(chunk_results)
    
    # Deduplicate
    hmdb_inchis = list(set(hmdb_results))
    random.shuffle(hmdb_inchis)
    
    # Split train/val
    hmdb_train = hmdb_inchis[:int(0.95 * len(hmdb_inchis))]
    hmdb_val = hmdb_inchis[int(0.95 * len(hmdb_inchis)):]
    
    # Remove excluded molecules
    hmdb_train = [inchi for inchi in hmdb_train if inchi not in excluded_inchis]
    
    # Save results
    pd.DataFrame(hmdb_train, columns=["inchi"]).to_csv("../data/fp2mol/hmdb/preprocessed/hmdb_train.csv", index=False)
    pd.DataFrame(hmdb_val, columns=["inchi"]).to_csv("../data/fp2mol/hmdb/preprocessed/hmdb_val.csv", index=False)
    
    #################################################
    # Process DSSTox dataset
    #################################################
    print("Processing DSSTox dataset...")
    dss_smiles = []
    for i in tqdm(range(1, 14), desc='Loading DSSTox'):
        df = pd.read_excel(f'../data/fp2mol/raw/DSSToxDump{i}.xlsx')
        dss_smiles.extend(df[df['SMILES'].notnull()]['SMILES'].tolist())
    
    # Split input into chunks and process in parallel
    smiles_chunks = chunks(dss_smiles, N_CORES * 4)
    dss_results = []
    
    for chunk_results in tqdm(
        pool.imap(partial(process_smiles_batch, filter_func=filter_with_atom_types), 
                  smiles_chunks),
        total=len(smiles_chunks),
        desc="Processing DSSTox"
    ):
        dss_results.extend(chunk_results)
    
    # Deduplicate
    dss_inchis = list(set(dss_results))
    random.shuffle(dss_inchis)
    
    # Split train/val
    dss_train = dss_inchis[:int(0.95 * len(dss_inchis))]
    dss_val = dss_inchis[int(0.95 * len(dss_inchis)):]
    
    # Remove excluded molecules
    dss_train = [inchi for inchi in dss_train if inchi not in excluded_inchis]
    
    # Save results
    pd.DataFrame(dss_train, columns=["inchi"]).to_csv("../data/fp2mol/dss/preprocessed/dss_train.csv", index=False)
    pd.DataFrame(dss_val, columns=["inchi"]).to_csv("../data/fp2mol/dss/preprocessed/dss_val.csv", index=False)
    
    #################################################
    # Process COCONUT dataset
    #################################################
    print("Processing COCONUT dataset...")
    coconut_df = pd.read_csv('../data/fp2mol/raw/coconut_csv-03-2025.csv')
    coconut_smiles = list(coconut_df["canonical_smiles"])
    
    # Split input into chunks and process in parallel
    smiles_chunks = chunks(coconut_smiles, N_CORES * 4)
    coconut_results = []
    
    for chunk_results in tqdm(
        pool.imap(partial(process_smiles_batch, filter_func=filter_with_atom_types), 
                  smiles_chunks),
        total=len(smiles_chunks),
        desc="Processing COCONUT"
    ):
        coconut_results.extend(chunk_results)
    
    # Deduplicate
    coconut_inchis = list(set(coconut_results))
    random.shuffle(coconut_inchis)
    
    # Split train/val
    coconut_train = coconut_inchis[:int(0.95 * len(coconut_inchis))]
    coconut_val = coconut_inchis[int(0.95 * len(coconut_inchis)):]
    
    # Remove excluded molecules
    coconut_train = [inchi for inchi in coconut_train if inchi not in excluded_inchis]
    
    # Save results
    pd.DataFrame(coconut_train, columns=["inchi"]).to_csv("../data/fp2mol/coconut/preprocessed/coconut_train.csv", index=False)
    pd.DataFrame(coconut_val, columns=["inchi"]).to_csv("../data/fp2mol/coconut/preprocessed/coconut_val.csv", index=False)
    
    #################################################
    # Process MOSES dataset
    #################################################
    print("Processing MOSES dataset...")
    moses_df = pd.read_csv('../data/fp2mol/raw/moses.csv')
    moses_smiles = list(moses_df["SMILES"])
    
    # Split input into chunks and process in parallel
    smiles_chunks = chunks(moses_smiles, N_CORES * 4)
    moses_results = []
    
    for chunk_results in tqdm(
        pool.imap(partial(process_smiles_batch, filter_func=filter_with_atom_types), 
                  smiles_chunks),
        total=len(smiles_chunks),
        desc="Processing MOSES"
    ):
        moses_results.extend(chunk_results)
    
    # Deduplicate
    moses_inchis = list(set(moses_results))
    random.shuffle(moses_inchis)
    
    # Split train/val
    moses_train = moses_inchis[:int(0.95 * len(moses_inchis))]
    moses_val = moses_inchis[int(0.95 * len(moses_inchis)):]
    
    # Remove excluded molecules
    moses_train = [inchi for inchi in moses_train if inchi not in excluded_inchis]
    
    # Save results
    pd.DataFrame(moses_train, columns=["inchi"]).to_csv("../data/fp2mol/moses/preprocessed/moses_train.csv", index=False)
    pd.DataFrame(moses_val, columns=["inchi"]).to_csv("../data/fp2mol/moses/preprocessed/moses_val.csv", index=False)
    
    #################################################
    # Process COMBINED dataset
    #################################################
    print("Processing COMBINED dataset...")
    combined_inchis = list(set(hmdb_inchis + dss_inchis + coconut_inchis + moses_inchis))
    random.shuffle(combined_inchis)
    
    # Split train/val
    combined_train = combined_inchis[:int(0.95 * len(combined_inchis))]
    combined_val = combined_inchis[int(0.95 * len(combined_inchis)):]
    
    # Remove excluded molecules
    combined_train = [inchi for inchi in combined_train if inchi not in excluded_inchis]
    
    # Save results
    pd.DataFrame(combined_train, columns=["inchi"]).to_csv("../data/fp2mol/combined/preprocessed/combined_train.csv", index=False)
    pd.DataFrame(combined_val, columns=["inchi"]).to_csv("../data/fp2mol/combined/preprocessed/combined_val.csv", index=False)
    
    # Clean up
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()