from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from tqdm import tqdm
import os
import json
import pandas as pd
import multiprocessing as mp
from functools import partial
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def process_smiles_to_inchi(smile):
    """Process a single SMILES string to InChI"""
    try:
        mol = Chem.MolFromSmiles(smile)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchi(mol)
    except Exception as e:
        print(f"{e}")
        return None

    
def smiles_to_inchi_parallel(df, smile_column_name,n_jobs=None):
    """
    Convert SMILES to InChI in parallel
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing SMILES
    smile_column_name : str
        Column name containing SMILES
    n_jobs : int, optional
        Number of processes to use, defaults to number of CPU cores
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'InChI' column
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
        
    print("Extract all SMILES to a list for processing")
    smiles_list = df[smile_column_name].tolist()
    
    with mp.Pool(processes=n_jobs) as pool:
        inchis = list(
            tqdm(
                pool.imap(process_smiles_to_inchi,smiles_list),
                total=len(smiles_list),
                desc="Converting SMILES to InChI",
            )
        )
        
        df_result = df.copy()
        df_result['InChI'] = inchis
        
    return df_result

def read_spec_json(file_name, data_path):
    """
    Process the JSON file for a given spec value.
    
    Parameters:
    -----------
    file_name, : str
        The filename or identifier from the spec field
    data_path : str
        The directory path where the JSON files are stored
    
    Returns:
    --------
    dict or None
        Processed data from the JSON file, or None if file not found
    """
    
    # Construct the full path to the JSON file
    json_path = os.path.join(data_path,file_name)
    
    # Check if the path exists as is
    if not os.path.exists(json_path):
        if not json_path.endswith('.json'):
            json_path = f"{json_path}.json"
            
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found for spec: {file_name}")
            return None
        
    # Read and parse the JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        return file_name, data
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return file_name, None
    
def extract_spectal_info(df,spec_column, data_path,n_jobs=None):
    """
    Process all spec files in a DataFrame in parallel
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing spec column
    spec_column : str
        Name of the column containing spec filenames
    data_path : str
        Directory path where JSON files are stored
    n_jobs : int, optional
        Number of processes to use, defaults to CPU count
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'extracted_spectral_info' column
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    read_func = partial(read_spec_json,data_path=data_path)
    
    spec_files_names = df[spec_column].tolist()
    results = {}
    with mp.Pool(processes=n_jobs) as pool:
        for file_name, data in tqdm(
            pool.imap_unordered(read_func,spec_files_names),
            total=len(spec_files_names),
            desc="Extracting JSON files"
        ):
            results[file_name] = data
    
    successful = sum (1 for v in results.values() if v is not None)
    print(f"Successfully processed {successful} out of {len(results)} files ({successful/len(results)*100:.2f}%)")
    df_result = df.copy()
    df_result['extracted_spectral_info'] = df[spec_column].map(results)
    return df_result

def process_mol(mol):
    """Process a single molecule"""
    if mol is None:
        return None
    
    record = {
        'SMILES': Chem.MolToSmiles(mol, isomericSmiles=False),
        'InChI': Chem.MolToInchi(mol),
        'Formula': Chem.rdMolDescriptors.CalcMolFormula(mol)
    }

    return record

def parse_sdf_parallel(sdf_file_path, n_jobs=None):
    """Parse SDF using parallel processing"""
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    print("Loading molecules from SDF...")
    supplier = Chem.SDMolSupplier(sdf_file_path)
    molecules = [mol for mol in tqdm(supplier,desc="Reading molecules")]
        
    print(f"Processing {len(molecules)} molecules with {n_jobs} workers...")
    
    with mp.Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(process_mol,molecules),
                total=len(molecules),
                desc="Processing"
            )
        )    
        
        results = [r for r in results if r is not None]
        
        df = pd.DataFrame(results)
        print(f"Successfully processed {len(df)} molecules")
        return df
    

def process_smile(smile):
    """Process a single SMILES string"""    
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print("Can't recognize the mol")
            return None
            
        inchi = Chem.MolToInchi(mol)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        return {
            'SMILES': smile,
            'InChI': inchi,
            'Formula': formula
        }
    except Exception as e:
        print(e)
        return None
    
def process_excel_files_parallel(base_path='../../data/raw/diffms_spectrum_db/raw/', 
                                        start=1, end=13, n_jobs=None):
    """Process Excel files with parallel processing, extracting only SMILES, InChI and formula"""
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    all_data = []
    
    for i in tqdm(range(start, end+1), desc='Processing Excel files'):
        file_path = f'{base_path}DSSToxDump{i}.xlsx'
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        
        try:
            # Read only the SMILES column
            df = pd.read_excel(file_path, usecols=['MS_READY_SMILES'])
            smiles_list = df['MS_READY_SMILES'].dropna().tolist()
    
            # Process in parallel
            with mp.Pool(processes=n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(process_smile, smiles_list),
                    total=len(smiles_list),
                    desc=f'Processing SMILES in file {i}'
                ))
            
            # Filter out None results
            results = [r for r in results if r is not None]
            
            if results:
                file_df = pd.DataFrame(results)
                all_data.append(file_df)
                print(f"Extracted {len(file_df)} valid compounds from {file_path}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Combine and deduplicate
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates based on SMILES
        initial_count = len(final_df)
        final_df = final_df.drop_duplicates(subset=['SMILES'])
        
        print(f"Final DataFrame contains {len(final_df)} unique compounds")
        print(f"Removed {initial_count - len(final_df)} duplicates")
        
        return final_df
    else:
        return pd.DataFrame(columns=['SMILES', 'InChI', 'Formula'])
