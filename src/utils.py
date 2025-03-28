from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, PandasTools
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
from pyteomics import mzxml
import os
import json
import pandas as pd
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Configure RDKit logging
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)

# ============================================================================
# Core Chemical Structure Processing Functions
# ============================================================================

def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert a SMILES string to an RDKit molecule object.
    
    Args:
        smiles: SMILES string representation of molecule
        
    Returns:
        RDKit Mol object or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to convert SMILES to molecule: {smiles}")
        return mol
    except Exception as e:
        logger.error(f"Error converting SMILES to molecule: {e}")
        return None

def standardize_smiles(smiles: str, remove_stereo: bool = True) -> Optional[str]:
    """Standardize SMILES representation.
    
    Args:
        smiles: Input SMILES string
        remove_stereo: Whether to remove stereochemical information
        
    Returns:
        Standardized SMILES string or None if conversion fails
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=not remove_stereo)
    except Exception as e:
        logger.error(f"Error standardizing SMILES: {e}")
        return None

def mol_to_inchi(mol: Chem.Mol) -> Optional[str]:
    """Convert RDKit molecule to InChI.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        InChI string or None if conversion fails
    """
    if mol is None:
        return None
    
    try:
        return Chem.MolToInchi(mol)
    except Exception as e:
        logger.error(f"Error converting molecule to InChI: {e}")
        return None

def get_molecular_formula(mol: Chem.Mol) -> Optional[str]:
    """Get molecular formula from RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Molecular formula or None if calculation fails
    """
    if mol is None:
        return None
    
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception as e:
        logger.error(f"Error calculating molecular formula: {e}")
        return None

def smiles_to_inchi(smiles: str, remove_stereo: bool = True) -> Optional[str]:
    """Convert SMILES to InChI with option to remove stereochemistry.
    
    Args:
        smiles: SMILES string
        remove_stereo: Whether to remove stereochemical information
        
    Returns:
        InChI string or None if conversion fails
    """
    try:
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None
            
        # Standardize SMILES if removing stereo information
        if remove_stereo:
            std_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            mol = smiles_to_mol(std_smiles)
            if mol is None:
                return None
                
        return mol_to_inchi(mol)
    except Exception as e:
        logger.error(f"Error converting SMILES to InChI: {e}")
        return None

def extract_molecule_properties(mol: Chem.Mol, remove_stereo: bool = True) -> Optional[Dict[str, str]]:
    """Extract common molecular properties from RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        remove_stereo: Whether to remove stereochemical information in SMILES
        
    Returns:
        Dictionary with SMILES, InChI and Formula or None if processing fails
    """
    if mol is None:
        return None
        
    try:
        return {
            'SMILES': Chem.MolToSmiles(mol, isomericSmiles=not remove_stereo),
            'InChI': mol_to_inchi(mol),
            'Formula': get_molecular_formula(mol)
        }
    except Exception as e:
        logger.error(f"Error extracting molecule properties: {e}")
        return None

def process_smiles(smiles: str, remove_stereo: bool = True) -> Optional[Dict[str, str]]:
    """Process a SMILES string to extract molecular properties.
    
    Args:
        smiles: SMILES string
        remove_stereo: Whether to remove stereochemical information
        
    Returns:
        Dictionary with SMILES, InChI and Formula or None if processing fails
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
        
    result = extract_molecule_properties(mol, remove_stereo)
    if result is not None:
        # Preserve the input SMILES rather than using the standardized version
        result['SMILES'] = smiles
    return result

def extract_mzxml_data(filepath: str) -> Optional[Dict[str, Any]]:
    """Extract data from an mzXML file.
    
    Args:
        filepath: Path to mzXML file
        
    Returns:
        Dictionary with extracted data or None if processing fails
    """
    try:
        # Basic file info
        file_info = {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
        }
        
        # Read mzXML file
        with mzxml.MzXML(filepath) as reader:
            # Extract run information
            file_info['instrument'] = reader.header.get('instrumentModel', 'Unknown')
            file_info['start_time'] = reader.header.get('startTime', None)
            
            # Count spectra
            reader.reset()
            spectra_count = sum(1 for _ in reader)
            file_info['spectra_count'] = spectra_count
            
            # Get info from first spectrum for sample data
            reader.reset()
            first_spectrum = next(reader, None)
            if first_spectrum:
                file_info['ms_levels'] = first_spectrum.get('msLevel', None)
                file_info['retention_time'] = first_spectrum.get('retentionTime', None)
            
        return file_info
    
    except Exception as e:
        logger.error(f"Error extracting data from mzXML file {filepath}: {e}")
        return None

# ============================================================================
# File Processing Functions
# ============================================================================

def read_json_file(file_name: str, data_path: str) -> Tuple[str, Optional[Dict]]:
    """Read and parse a JSON file.
    
    Args:
        file_name: The filename or identifier
        data_path: Directory path where JSON files are stored
        
    Returns:
        Tuple of (file_name, data) where data is the parsed JSON or None if failed
    """
    # Construct the full path to the JSON file
    json_path = os.path.join(data_path, file_name)
    
    # Check if the path exists as is or needs .json extension
    if not os.path.exists(json_path):
        if not json_path.endswith('.json'):
            json_path = f"{json_path}.json"
            
        if not os.path.exists(json_path):
            logger.warning(f"JSON file not found: {file_name}")
            return file_name, None
        
    # Read and parse the JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return file_name, data
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return file_name, None

# ============================================================================
# Parallel Processing Functions
# ============================================================================

def process_in_parallel(items: List[Any], 
                        process_func: Callable, 
                        n_jobs: Optional[int] = None,
                        desc: str = "Processing") -> List[Any]:
    """Generic parallel processing function.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        n_jobs: Number of processes to use (defaults to CPU count)
        desc: Description for the progress bar
        
    Returns:
        List of processed results
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
        
    with mp.Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, items),
                total=len(items),
                desc=desc
            )
        )
    return results

def extract_spectral_info(df: pd.DataFrame, 
                         spec_column: str, 
                         data_path: str, 
                         n_jobs: Optional[int] = None) -> pd.DataFrame:
    """Process all spec files in a DataFrame in parallel.
    
    Args:
        df: DataFrame containing spec column
        spec_column: Column name containing spec filenames
        data_path: Directory path where JSON files are stored
        n_jobs: Number of processes to use (defaults to CPU count)
        
    Returns:
        DataFrame with added 'extracted_spectral_info' column
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    read_func = partial(read_json_file, data_path=data_path)
    spec_files_names = df[spec_column].tolist()
    
    results_list = process_in_parallel(
        spec_files_names, 
        read_func, 
        n_jobs=n_jobs,
        desc="Extracting JSON files"
    )
    
    # Convert results to dictionary
    results = {file_name: data for file_name, data in results_list}
    
    successful = sum(1 for v in results.values() if v is not None)
    print(f"Successfully processed {successful} out of {len(results)} files "
          f"({successful/len(results)*100:.2f}%)")
    
    df_result = df.copy()
    df_result['extracted_spectral_info'] = df[spec_column].map(results)
    return df_result

def parse_sdf_file(sdf_file_path: str, n_jobs: Optional[int] = None) -> pd.DataFrame:
    """Parse SDF file using parallel processing.
    
    Args:
        sdf_file_path: Path to the SDF file
        n_jobs: Number of processes to use (defaults to CPU count)
        
    Returns:
        DataFrame with molecule properties
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    print("Loading molecules from SDF...")
    supplier = Chem.SDMolSupplier(sdf_file_path)
    molecules = [mol for mol in tqdm(supplier, desc="Reading molecules")]
        
    print(f"Processing {len(molecules)} molecules with {n_jobs} workers...")
    
    results = process_in_parallel(
        molecules, 
        extract_molecule_properties, 
        n_jobs=n_jobs,
        desc="Processing molecules"
    )
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    df = pd.DataFrame(results)
    print(f"Successfully processed {len(df)} molecules")
    return df
    
def process_smiles_dataframe(df: pd.DataFrame, 
                           smiles_column: str = "SMILES", 
                           n_jobs: Optional[int] = None,
                           remove_stereo: bool = True) -> pd.DataFrame:
    """Process SMILES in a DataFrame to extract InChI and Formula.
    
    Args:
        df: DataFrame containing SMILES column
        smiles_column: Column name containing SMILES strings
        n_jobs: Number of processes to use (defaults to CPU count)
        remove_stereo: Whether to remove stereochemical information
        
    Returns:
        DataFrame with added 'InChI' and 'Formula' columns
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
        
    print("Extracting SMILES from dataframe")
    smiles_list = df[smiles_column].dropna().tolist()
    
    print(f"Processing {len(smiles_list)} SMILES strings using {n_jobs} processes...")
    
    process_func = partial(process_smiles, remove_stereo=remove_stereo)
    results = process_in_parallel(
        smiles_list, 
        process_func, 
        n_jobs=n_jobs,
        desc="Processing SMILES"
    )
    
    # Filter valid results and create mapping
    valid_results = [r for r in results if r is not None]
    smiles_to_props = {result["SMILES"]: {
        'InChI': result['InChI'],
        'Formula': result['Formula']
    } for result in valid_results}
    
    # Apply mapping to dataframe
    result_df = df.copy()
    result_df['InChI'] = df[smiles_column].map(
        lambda s: smiles_to_props.get(s, {}).get('InChI') if pd.notna(s) else None
    )
    result_df['Formula'] = df[smiles_column].map(
        lambda s: smiles_to_props.get(s, {}).get('Formula') if pd.notna(s) else None
    )
    
    # Remove rows with failed conversions
    result_df = result_df.dropna(subset=['InChI', 'Formula'])
    result_df = result_df.reset_index(drop=True)
    
    print(f"Successfully processed {len(result_df)} of {len(df)} entries")
    return result_df

def process_excel_files(base_path: str = '../../data/raw/diffms_spectrum_db/raw/', 
                       start: int = 1, 
                       end: int = 13, 
                       n_jobs: Optional[int] = None) -> pd.DataFrame:
    """Process Excel files in parallel, extracting molecular properties.
    
    Args:
        base_path: Directory containing Excel files
        start: Starting file number
        end: Ending file number
        n_jobs: Number of processes to use (defaults to CPU count)
        
    Returns:
        DataFrame with unique compounds and their properties
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    all_data = []
    
    for i in tqdm(range(start, end+1), desc='Processing Excel files'):
        file_path = f'{base_path}DSSToxDump{i}.xlsx'
        
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found, skipping.")
            continue
        
        try:
            # Read only the SMILES column
            df = pd.read_excel(file_path, usecols=['MS_READY_SMILES'])
            smiles_list = df['MS_READY_SMILES'].dropna().tolist()
    
            # Process in parallel
            results = process_in_parallel(
                smiles_list, 
                process_smiles, 
                n_jobs=n_jobs,
                desc=f'Processing file {i}'
            )
            
            # Filter out None results
            results = [r for r in results if r is not None]
            
            if results:
                file_df = pd.DataFrame(results)
                all_data.append(file_df)
                print(f"Extracted {len(file_df)} valid compounds from {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
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


def process_smiles_file(filepath: str, n_jobs: Optional[int] = None) -> List[Dict[str, str]]:
    """Process SMILES strings from a file and extract molecular properties in parallel.
    
    Args:
        filepath: Path to file containing SMILES (one per line)
        n_jobs: Number of parallel processes to use
        
    Returns:
        List of dictionaries with molecular properties
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    with open(filepath,'r') as r:
        smiles_list = [line.strip() for line in r if line.strip()]
    
    results = process_in_parallel(
        items=smiles_list,
        process_func=process_smiles,
        n_jobs=n_jobs,
        desc='Processing molecules'
    )
    
    valid_results = [r for r in results if r is not None]
    
    print(f"Successfully processed {len(valid_results)} out of {len(smiles_list)} molecules")
    return valid_results

def process_mzxml_files(filepaths: List[str], n_jobs: Optional[int] = None) -> pd.DataFrame:
    """Process multiple mzXML files in parallel.
    
    Args:
        filepaths: List of paths to mzXML files
        n_jobs: Number of processes to use (defaults to CPU count)
        
    Returns:
        DataFrame with extracted data
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    results = process_in_parallel(
        items=filepaths,
        process_func=process_mzxml_files,
        n_jobs=n_jobs,
        desc="Processing mzXML files"
    )
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    print(f"Successfully processed {len(valid_results)} out of {len(filepaths)} mzXML files")
    return pd.DataFrame(valid_results)