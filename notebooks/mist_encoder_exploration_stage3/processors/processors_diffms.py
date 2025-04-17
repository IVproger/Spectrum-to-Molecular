from pathlib import Path
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Callable

import h5py

import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

import utils
import data.data as data


class SpectrumProcessor:
    """Process raw spectral data directly without file reading."""

    cat_types = {"frags": 0, "loss": 1, "ab_loss": 2, "cls": 3}
    num_inten_bins = 10
    num_types = len(cat_types)
    cls_type = cat_types.get("cls")
    
    def __init__(
        self,
        augment_data: bool = False,
        augment_prob: float = 1,
        remove_prob: float = 0.1,
        remove_weights: str = "uniform",
        inten_prob: float = 0.1,
        cls_type: str = "ms1",
        max_peaks: int = None,
        inten_transform: str = "float",
        magma_modulo: int = 512,
        magma_aux_loss: bool = False,
        magma_folder: str = None,
    ):
        self.cls_type = cls_type
        self.augment_data = augment_data
        self.remove_prob = remove_prob
        self.augment_prob = augment_prob
        self.remove_weights = remove_weights
        self.inten_prob = inten_prob
        self.max_peaks = max_peaks
        self.inten_transform = inten_transform
        self.aug_nbits = magma_modulo
        self.magma_aux_loss = magma_aux_loss
        self.magma_folder = Path(magma_folder) if magma_folder else None
        self.spec_name_to_magma_file = {}
        
        # Initialize magma file mapping if enabled
        if self.magma_aux_loss and self.magma_folder:
            self._initialize_magma_mapping()
            
    def _initialize_magma_mapping(self):
        """Initialize the mapping of spectrum names to MAGMA files"""
        if not self.magma_folder.exists():
            print(f"Warning: MAGMA folder {self.magma_folder} does not exist")
            return
            
        # Create mapping of spectrum names to MAGMA files
        magma_files = list(self.magma_folder.glob("*.magma"))
        self.spec_name_to_magma_file = {
            file.stem: file for file in magma_files
        }
        print(f"Found {len(self.spec_name_to_magma_file)} MAGMA files")
        
    def process_raw_spectrum(self, raw_spectral_json, spec_id=None, train_mode=False):
        """
        Process raw spectral data from JSON string or dict
        
        Args:
            raw_spectral_json: JSON string or dict containing spectral info
            spec_id: Optional identifier for the spectrum
            train_mode: Whether to use training mode (for augmentation)
            
        Returns:
            Dict of processed spectral features
        """
        # Convert string to dict if needed
        if isinstance(raw_spectral_json, str):
            tree = json.loads(raw_spectral_json)
        else:
            tree = raw_spectral_json
            
        # Extract spectrum name if not provided
        if spec_id is None and "name" in tree:
            spec_id = tree["name"]
        
        # Extract the peak dictionary
        peak_dict = self._get_peak_dict_from_raw(tree)
        
        # Augment if in training mode
        if train_mode and self.augment_data:
            augment_peak = np.random.random() < self.augment_prob
            if augment_peak:
                peak_dict = self.augment_peak_dict(peak_dict)
                
        # Generate feature dictionary
        features = self._generate_features(peak_dict, spec_id)
        
        return features
    
    def _get_peak_dict_from_raw(self, tree: dict) -> dict:
        """
        Extract peak information from raw spectral data
        
        Args:
            tree: Dictionary of raw spectral data
            
        Returns:
            dict: Dictionary with peak information
        """
        root_form = tree["cand_form"]
        root_ion = tree["cand_ion"]
        output_tbl = tree["output_tbl"]
        
        if output_tbl is None:
            frags = []
            intens = []
            ions = []
        else:
            frags = output_tbl["formula"]
            intens = output_tbl["ms2_inten"]
            ions = output_tbl["ions"]
        
        out_dict = {
            "frags": frags,
            "intens": intens,
            "ions": ions,
            "root_form": root_form,
            "root_ion": root_ion,
        }
        
        # If we have a max peaks, then we need to filter
        if self.max_peaks is not None and len(intens) > 0:
            # Sort by intensity
            inten_list = list(out_dict["intens"])
            new_order = np.argsort(inten_list)[::-1]
            cutoff_ind = min(len(inten_list), self.max_peaks)
            new_inds = new_order[:cutoff_ind]
            
            # Get new frags, intens, ions and assign to outdict
            inten_list = np.array(inten_list)[new_inds].tolist()
            frag_list = np.array(out_dict["frags"])[new_inds].tolist()
            ion_list = np.array(out_dict["ions"])[new_inds].tolist()
            
            out_dict["frags"] = frag_list
            out_dict["intens"] = inten_list
            out_dict["ions"] = ion_list
            
        return out_dict
    
    def augment_peak_dict(self, peak_dict: dict):
        """
        Augment peak dictionary by removing peaks or rescaling intensities
        
        Args:
            peak_dict: Dictionary containing peak info
            
        Returns:
            Augmented peak dictionary
        """
        # Only scale frags
        frags = np.array(peak_dict["frags"])
        intens = np.array(peak_dict["intens"])
        ions = np.array(peak_dict["ions"])
        
        if len(frags) == 0:
            return peak_dict
            
        # Compute removal probability
        num_modify_peaks = len(frags)
        keep_prob = 1 - self.remove_prob
        num_to_keep = np.random.binomial(n=num_modify_peaks, p=keep_prob)
        
        keep_inds = np.arange(0, num_modify_peaks)
        
        # Determine weights for removal
        if self.remove_weights == "quadratic":
            keep_probs = intens.reshape(-1) ** 2 + 1e-9
            keep_probs = keep_probs / keep_probs.sum()
        elif self.remove_weights == "uniform":
            keep_probs = np.ones(len(intens)) / len(intens)
        elif self.remove_weights == "exp":
            keep_probs = np.exp(intens.reshape(-1) + 1e-5)
            keep_probs = keep_probs / keep_probs.sum()
        else:
            raise NotImplementedError()
            
        # Sample indices to keep
        ind_samples = np.random.choice(
            keep_inds, size=num_to_keep, replace=False, p=keep_probs
        )
        
        # Re-index frags, intens, and ions
        frags, intens, ions = frags[ind_samples], intens[ind_samples], ions[ind_samples]
        
        # Apply random intensity scaling
        rescale_prob = np.random.random(len(intens))
        inten_scalar_factor = np.random.normal(loc=1, size=len(intens))
        inten_scalar_factor[inten_scalar_factor <= 0] = 0
        
        # Where rescale prob is >= self.inten_prob set inten rescale to 1
        inten_scalar_factor[rescale_prob >= self.inten_prob] = 1
        
        # Rescale intens
        intens = intens * inten_scalar_factor
        new_max = intens.max() + 1e-12 if len(intens) > 0 else 1
        intens /= new_max
        
        # Replace peak dict with new values
        peak_dict["intens"] = intens
        peak_dict["frags"] = frags
        peak_dict["ions"] = ions
        
        return peak_dict
    
    def _process_magma_file(self, spec_name, mz_vec, forms_vec):
        """
        Process MAGMA file for fingerprint data

        Args:
            spec_name: Name of the spectrum
            mz_vec: Vector of m/z values
            forms_vec: Vector of formula vectors

        Returns:
            numpy array of fingerprints
        """
        # Initialize fingerprints with -1 (no data)
        fingerprints = np.zeros((forms_vec.shape[0], self.aug_nbits)) - 1

        # Return early if MAGMA is disabled or no spec name
        if not self.magma_aux_loss or not spec_name:
            return fingerprints

        # Get MAGMA file path
        magma_file = self.spec_name_to_magma_file.get(spec_name)

        # If specified spec name not found, try with file extension stripped
        if not magma_file and '.' in spec_name:
            base_name = spec_name.split('.')[0]
            magma_file = self.spec_name_to_magma_file.get(base_name)

        if magma_file and magma_file.exists():
            try:
                # Ensure 'frag_fp' is read as string type
                magma_df = pd.read_csv(magma_file, sep="\t", dtype={'frag_fp': str})

                if len(magma_df) > 0 and len(mz_vec) > 0:
                    mz_vec_array = np.array(mz_vec)
                    magma_masses = magma_df["mz_corrected"].values

                    # Get closest mz within 1e-4
                    diff_mat = np.abs(mz_vec_array[:, None] - magma_masses[None, :])
                    min_inds = diff_mat.argmin(1)

                    for row_ind, (mz_val, min_ind) in enumerate(zip(mz_vec_array, min_inds)):
                        diff_val = diff_mat[row_ind, min_ind]
                        if diff_val < 1e-4:
                            # Get fingerprint
                            magma_row = magma_df.iloc[min_ind]
                            # Check if 'frag_fp' exists and is not NaN/null before splitting
                            if "frag_fp" in magma_row and pd.notna(magma_row["frag_fp"]):
                                # Convert to string just in case, although dtype should handle it
                                frag_fp_str = str(magma_row["frag_fp"])
                                # Check if the string is not empty before splitting
                                if frag_fp_str:
                                    magma_fp_bits = [
                                        int(i) % self.aug_nbits
                                        for i in frag_fp_str.split(",") if i # Ensure i is not empty string
                                    ]

                                    # Set base to 0
                                    fingerprints[row_ind, :] = 0

                                    # Update to 1 where active bit
                                    if magma_fp_bits: # Check if list is not empty
                                        fingerprints[row_ind, magma_fp_bits] = 1
            except Exception as e:
                # Provide more context in the error message
                print(f"Error processing MAGMA file {magma_file} for spectrum {spec_name}: {e}")
                # Optionally re-raise or handle differently
                # raise e

        return fingerprints
    
    def _generate_features(self, peak_dict: dict, spec_name=None):
        """
        Generate features from peak dictionary
        
        Args:
            peak_dict: Dictionary containing peak info
            spec_name: Optional spectrum name/identifier
            
        Returns:
            Dictionary of features
        """
        import utils  # Assuming utils is defined elsewhere
        
        # Add in chemical formulae
        root = peak_dict["root_form"]
        
        forms_vec = [utils.formula_to_dense(i) for i in peak_dict["frags"]]
        if len(forms_vec) == 0:
            mz_vec = []
        else:
            mz_vec = (np.array(forms_vec) * utils.VALID_MONO_MASSES).sum(-1).tolist()
        
        root_vec = utils.formula_to_dense(root)
        root_ion = utils.get_ion_idx(peak_dict["root_ion"])
        root_mass = (root_vec * utils.VALID_MONO_MASSES).sum()
        inten_vec = list(peak_dict["intens"])
        ion_vec = [utils.get_ion_idx(i) for i in peak_dict["ions"]]
        type_vec = len(forms_vec) * [self.cat_types["frags"]]
        instrument = 0  # Default instrument value - adjust as needed
        
        # Add classification token
        if self.cls_type == "ms1":
            cls_ind = self.cat_types.get("cls")
            inten_vec.append(1.0)
            type_vec.append(cls_ind)
            forms_vec.append(root_vec)
            mz_vec.append(root_mass)
            ion_vec.append(root_ion)
        elif self.cls_type == "zeros":
            cls_ind = self.cat_types.get("cls")
            inten_vec.append(0.0)
            type_vec.append(cls_ind)
            forms_vec.append(np.zeros_like(root_vec))
            mz_vec.append(0)
            ion_vec.append(root_ion)
        else:
            raise NotImplementedError()
            
        # Process intensity values
        inten_vec = np.array(inten_vec)
        if self.inten_transform == "float":
            self.inten_feats = 1
        elif self.inten_transform == "zero":
            self.inten_feats = 1
            inten_vec = np.zeros_like(inten_vec)
        elif self.inten_transform == "log":
            self.inten_feats = 1
            inten_vec = np.log(inten_vec + 1e-5)
        elif self.inten_transform == "cat":
            self.inten_feats = self.num_inten_bins
            bins = np.linspace(0, 1, self.num_inten_bins)
            inten_vec = np.digitize(inten_vec, bins)
        else:
            raise NotImplementedError()
            
        forms_vec = np.array(forms_vec)
        
        # Process MAGMA fingerprints if enabled
        fingerprints = self._process_magma_file(spec_name, mz_vec, forms_vec)
        
        # Create output dictionary
        out_dict = {
            "peak_type": np.array(type_vec),
            "form_vec": forms_vec,
            "ion_vec": ion_vec,
            "frag_intens": inten_vec,
            "name": spec_name,
            "magma_fps": fingerprints,
            "magma_aux_loss": self.magma_aux_loss,
            "instrument": instrument,
        }
        
        return out_dict

class FingerprintProcessor:
    """Process raw spectral data and generate molecular fingerprints."""
    
    def __init__(
        self,
        fp_names: List[str] = ["morgan2048"],
        morgan_radius: int = 2,
        nbits: int = 2048,
        augment_data: bool = False,
        augment_prob: float = 0.1,
    ):
        """
        Initialize the FingerprintProcessor
        
        Args:
            fp_names: List of fingerprint types to generate
            morgan_radius: Radius for Morgan fingerprints
            nbits: Number of bits for fingerprints
            augment_data: Whether to augment fingerprints
            augment_prob: Probability of augmenting fingerprints
        """
        self.fp_names = fp_names
        self.morgan_radius = morgan_radius
        self.nbits = nbits
        self.augment_data = augment_data
        self.augment_prob = augment_prob
        self._morgan_projection = np.random.randn(50, 2048)
        
    def process_raw_spectrum(self, raw_spectral_json, spec_id=None, train_mode=False):
        """
        Process raw spectral data from JSON string or dict and generate fingerprints
        
        Args:
            raw_spectral_json: JSON string or dict containing spectral info
            spec_id: Optional identifier for the spectrum
            train_mode: Whether to use training mode (for augmentation)
            
        Returns:
            Dict of processed fingerprint features
        """
        # Convert string to dict if needed
        if isinstance(raw_spectral_json, str):
            tree = json.loads(raw_spectral_json)
        else:
            tree = raw_spectral_json
            
        # Extract the SMILES or InChI from the data
        smiles = tree.get("smiles", tree.get("SMILES", None))
        inchi = tree.get("inchi", tree.get("InChI", None))
        
        # Create RDKit molecule from SMILES or InChI
        mol = None
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        elif inchi:
            mol = Chem.MolFromInchi(inchi)
        
        if not mol:
            # Return empty fingerprints if molecule can't be created
            return self._empty_fingerprints(spec_id)
            
        # Generate fingerprints
        fingerprints = self._generate_fingerprints(mol)
        
        # Augment fingerprints if in training mode
        if train_mode and self.augment_data and np.random.random() < self.augment_prob:
            fingerprints = self._augment_fingerprints(fingerprints)
            
        # Add metadata
        fingerprints["name"] = spec_id
        fingerprints["smiles"] = smiles if smiles else Chem.MolToSmiles(mol)
        fingerprints["inchi"] = inchi if inchi else Chem.MolToInchi(mol)
        
        return fingerprints
    
    def _empty_fingerprints(self, spec_id=None):
        """Generate empty fingerprint features when molecule can't be created"""
        fp_dict = {}
        for fp_name in self.fp_names:
            fp_size = self._get_fp_size(fp_name)
            fp_dict[fp_name] = np.zeros(fp_size)
            
        return {
            "fingerprints": fp_dict,
            "name": spec_id,
            "smiles": "",
            "inchi": ""
        }
    
    def _get_fp_size(self, fp_name):
        """Get the size of the fingerprint based on type"""
        fp_name_to_bits = {
            "morgan256": 256,
            "morgan512": 512,
            "morgan1024": 1024,
            "morgan2048": 2048,
            "morgan_project": 50,
            "morgan4096": 4096,
            "maccs": 167,
        }
        return fp_name_to_bits.get(fp_name, self.nbits)
    
    def _generate_fingerprints(self, mol):
        """
        Generate different types of fingerprints for a molecule
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dict of fingerprint arrays
        """
        fp_dict = {}
        
        for fp_name in self.fp_names:
            fp_dict[fp_name] = self._get_fingerprint(mol, fp_name)
            
        return {"fingerprints": fp_dict}
    
    def _get_fingerprint(self, mol, fp_name):
        """Generate a specific type of fingerprint"""
        if "morgan" in fp_name and "project" not in fp_name:
            # Extract bit number from name (e.g., morgan2048 -> 2048)
            nbits = int(fp_name.replace("morgan", "")) if fp_name != "morgan" else self.nbits
            return self._get_morgan_fp(mol, nbits=nbits)
        elif fp_name == "morgan_project":
            return self._get_morgan_projection(mol)
        elif fp_name == "maccs":
            return self._get_maccs(mol)
        else:
            # Default to Morgan with specified nbits
            return self._get_morgan_fp(mol, nbits=self.nbits)
    
    def _get_morgan_fp(self, mol, nbits=2048):
        """Generate Morgan fingerprint"""
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=nbits)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array
    
    def _get_morgan_projection(self, mol):
        """Generate projected Morgan fingerprint"""
        morgan_fp = self._get_morgan_fp(mol, nbits=2048)
        output_fp = np.einsum("ij,j->i", self._morgan_projection, morgan_fp)
        return output_fp
    
    def _get_maccs(self, mol):
        """Generate MACCS fingerprint"""
        fingerprint = GetMACCSKeysFingerprint(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array
    
    def _augment_fingerprints(self, fingerprints):
        """
        Augment fingerprints by randomly flipping bits
        
        Args:
            fingerprints: Dictionary containing fingerprint data
            
        Returns:
            Augmented fingerprints dictionary
        """
        fp_dict = fingerprints["fingerprints"]
        
        for fp_name, fp in fp_dict.items():
            # Randomly flip a small percentage of bits (0.5-2%)
            num_bits = len(fp)
            num_to_flip = max(1, int(np.random.uniform(0.005, 0.02) * num_bits))
            
            # Select random bit positions to flip
            flip_positions = np.random.choice(num_bits, num_to_flip, replace=False)
            
            # Create a copy and flip the bits
            augmented_fp = fp.copy()
            for pos in flip_positions:
                augmented_fp[pos] = 1 - augmented_fp[pos]  # Flip 0 to 1 or 1 to 0
            
            fp_dict[fp_name] = augmented_fp
            
        return {"fingerprints": fp_dict}
    
    def dist(self, fp1_data, fp2_data, fp_type="morgan2048") -> float:
        """
        Calculate Tanimoto distance between two fingerprints

        Args:
            fp1_data: First fingerprint data (raw JSON or processed fingerprint dict)
            fp2_data: Second fingerprint data (raw JSON or processed fingerprint dict)
            fp_type: Fingerprint type to use for comparison
            
        Returns:
            Tanimoto distance (float between 0 and 1)
        """
        # Process input data if it's raw JSON
        if isinstance(fp1_data, (str, dict)) and not isinstance(fp1_data, dict) or "fingerprints" not in fp1_data:
            fp1 = self.process_raw_spectrum(fp1_data)["fingerprints"][fp_type]
        else:
            fp1 = fp1_data["fingerprints"][fp_type]
            
        if isinstance(fp2_data, (str, dict)) and not isinstance(fp2_data, dict) or "fingerprints" not in fp2_data:
            fp2 = self.process_raw_spectrum(fp2_data)["fingerprints"][fp_type]
        else:
            fp2 = fp2_data["fingerprints"][fp_type]
        
        # Calculate Tanimoto distance
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        # Avoid division by zero
        if union == 0:
            return 1.0
        
        tanimoto = 1.0 - (intersection / union)
        return tanimoto

    def dist_batch(self, mol_list, fp_type="morgan2048") -> np.ndarray:
        """
        Calculate pairwise Tanimoto distances for a batch of molecules
        
        Args:
            mol_list: List of fingerprint data (raw JSON or processed fingerprint dicts)
            fp_type: Fingerprint type to use for comparison
            
        Returns:
            Square matrix of Tanimoto distances
        """
        if len(mol_list) == 0:
            return np.array([[]])

        # Process all fingerprints
        fps = []
        for mol_data in mol_list:
            if isinstance(mol_data, (str, dict)) and not isinstance(mol_data, dict) or "fingerprints" not in mol_data:
                fp = self.process_raw_spectrum(mol_data)["fingerprints"][fp_type]
            else:
                fp = mol_data["fingerprints"][fp_type]
            fps.append(fp)

        fps = np.vstack(fps)

        # Calculate all pairwise distances efficiently
        fps_a = fps[:, None, :]
        fps_b = fps[None, :, :]

        intersect = (fps_a & fps_b).sum(-1)
        union = (fps_a | fps_b).sum(-1)
        
        # Avoid division by zero
        union = np.maximum(union, 1)
        
        tanimoto = 1.0 - intersect / union
        return tanimoto

    def dist_one_to_many(self, mol_data, mol_list, fp_type="morgan2048") -> np.ndarray:
        """
        Calculate Tanimoto distances from one molecule to many molecules
        
        Args:
            mol_data: Fingerprint data for the reference molecule
            mol_list: List of fingerprint data to compare against
            fp_type: Fingerprint type to use for comparison
            
        Returns:
            Array of Tanimoto distances
        """
        if len(mol_list) == 0:
            return np.array([])

        # Process reference fingerprint
        if isinstance(mol_data, (str, dict)) and not isinstance(mol_data, dict) or "fingerprints" not in mol_data:
            fp_a = self.process_raw_spectrum(mol_data)["fingerprints"][fp_type]
        else:
            fp_a = mol_data["fingerprints"][fp_type]

        # Process all comparison fingerprints
        fps = []
        for mol_temp in mol_list:
            if isinstance(mol_temp, (str, dict)) and not isinstance(mol_temp, dict) or "fingerprints" not in mol_temp:
                fp = self.process_raw_spectrum(mol_temp)["fingerprints"][fp_type]
            else:
                fp = mol_temp["fingerprints"][fp_type]
            fps.append(fp)

        fps_b = np.vstack(fps)
        fps_a = fp_a[None, :]

        # Calculate distances
        intersect = (fps_a & fps_b).sum(-1)
        union = (fps_a | fps_b).sum(-1)
        
        # Avoid division by zero
        union = np.maximum(union, 1)

        tanimoto = 1.0 - intersect / union
        return tanimoto

class GraphProcessor:
    """Process raw spectral data and generate molecular graph representations."""
    
    ATOM_DECODER = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
    TYPES = {atom: i for i, atom in enumerate(ATOM_DECODER)}
    BONDS = {
        Chem.rdchem.BondType.SINGLE: 0, 
        Chem.rdchem.BondType.DOUBLE: 1, 
        Chem.rdchem.BondType.TRIPLE: 2, 
        Chem.rdchem.BondType.AROMATIC: 3
    }
    
    def __init__(
        self,
        augment_data: bool = False,
        augment_prob: float = 0.1,
        morgan_r: int = 2,
        morgan_nbits: int = 2048,
        add_node_features: bool = True,
    ):
        """
        Initialize the GraphProcessor
        
        Args:
            augment_data: Whether to augment graph representations
            augment_prob: Probability of augmenting graphs
            morgan_r: Radius for Morgan fingerprints as node features
            morgan_nbits: Number of bits for node fingerprints
            add_node_features: Whether to add extended node features
        """
        self.augment_data = augment_data
        self.augment_prob = augment_prob
        self.morgan_r = morgan_r
        self.morgan_nbits = morgan_nbits
        self.add_node_features = add_node_features
        
    def process_raw_spectrum(self, raw_spectral_json, spec_id=None, train_mode=False):
        """
        Process raw spectral data from JSON string or dict and generate graph representation
        
        Args:
            raw_spectral_json: JSON string or dict containing spectral info
            spec_id: Optional identifier for the spectrum
            train_mode: Whether to use training mode (for augmentation)
            
        Returns:
            PyTorch Geometric Data object representing the molecular graph
        """
        # Convert string to dict if needed
        if isinstance(raw_spectral_json, str):
            tree = json.loads(raw_spectral_json)
        else:
            tree = raw_spectral_json
            
        # Extract the SMILES or InChI from the data
        smiles = tree.get("smiles", tree.get("SMILES", None))
        inchi = tree.get("inchi", tree.get("InChI", None))
        
        # Create RDKit molecule from SMILES or InChI
        mol = None
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        elif inchi:
            mol = Chem.MolFromInchi(inchi)
        
        if not mol:
            # Return empty graph if molecule can't be created
            return self._empty_graph(spec_id)
            
        # Generate molecular graph
        graph = self._molecule_to_graph(mol, spec_id)
        
        # Augment graph if in training mode
        if train_mode and self.augment_data and np.random.random() < self.augment_prob:
            graph = self._augment_graph(graph)
            
        return graph
    
    def _empty_graph(self, spec_id=None):
        """Generate empty graph when molecule can't be created"""
        # Create a minimal graph with no nodes or edges
        x = torch.zeros((0, len(self.TYPES)), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(self.BONDS) + 1), dtype=torch.float)
        
        # Create empty fingerprint
        y = torch.zeros((1, self.morgan_nbits), dtype=torch.float)
        
        return Data(
            x=x, 
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            name=spec_id if spec_id else "",
            smiles="",
            num_nodes=0
        )
    
    def _molecule_to_graph(self, mol, spec_id=None):
        """
        Convert an RDKit molecule to a PyTorch Geometric graph
        
        Args:
            mol: RDKit molecule object
            spec_id: Optional identifier for the spectrum
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get basic molecule information
        N = mol.GetNumAtoms()
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        
        # Node features (atom types)
        type_idx = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            # Default to C if atom type not in TYPES
            type_idx.append(self.TYPES.get(symbol, self.TYPES['C']))
            
        # One-hot encode atom types
        x = F.one_hot(torch.tensor(type_idx), num_classes=len(self.TYPES)).float()
        
        # Edge features (bonds)
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            
            # Add edges in both directions
            row += [start, end]
            col += [end, start]
            
            # Add 1 to bond type to reserve 0 for no bond
            edge_type += 2 * [self.BONDS.get(bond_type, 0) + 1]
            
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        # One-hot encode bond types
        edge_attr = F.one_hot(edge_type, num_classes=len(self.BONDS) + 1).to(torch.float)
        
        # Sort edges by source then target node
        if edge_index.shape[1] > 0:  # If there are any edges
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]
        
        # Add Morgan fingerprint as a global feature
        y = torch.tensor(np.asarray(
            Chem.AllChem.GetMorganFingerprintAsBitVect(
                mol, self.morgan_r, nBits=self.morgan_nbits
            ), dtype=np.int8
        )).unsqueeze(0).float()
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            name=spec_id if spec_id else "",
            smiles=smiles,
            num_nodes=N
        )
        
        # Add extended node features if requested
        if self.add_node_features:
            data = self._add_node_features(data, mol)
            
        return data
    
    def _add_node_features(self, data, mol):
        """
        Add extended node features to the graph
        
        Args:
            data: PyTorch Geometric Data object
            mol: RDKit molecule object
            
        Returns:
            PyTorch Geometric Data with extended node features
        """
        # Initialize additional feature array
        additional_features = []
        
        for atom in mol.GetAtoms():
            features = [
                atom.GetDegree() / 10.0,  # Normalized degree
                atom.GetFormalCharge() / 8.0,  # Normalized formal charge
                atom.GetNumImplicitHs() / 8.0,  # Normalized implicit Hs
                atom.GetIsAromatic() * 1.0,  # Is aromatic
                atom.IsInRing() * 1.0,  # Is in ring
                atom.GetHybridization() / 8.0,  # Normalized hybridization
            ]
            additional_features.append(features)
            
        # Convert to tensor and concatenate with existing features
        if additional_features:
            additional_features = torch.tensor(additional_features, dtype=torch.float)
            data.x = torch.cat([data.x, additional_features], dim=1)
            
        return data
    
    def _augment_graph(self, graph):
        """
        Augment the graph representation by perturbing node and edge features
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            Augmented PyTorch Geometric Data object
        """
        # Make a copy of the graph
        augmented_graph = Data(
            x=graph.x.clone(),
            edge_index=graph.edge_index.clone(),
            edge_attr=graph.edge_attr.clone(),
            y=graph.y.clone(),
            name=graph.name,
            smiles=graph.smiles,
            num_nodes=graph.num_nodes
        )
        
        # Randomly perturb node features (if there are any nodes)
        if augmented_graph.num_nodes > 0:
            # Add small noise to continuous node features
            if augmented_graph.x.size(1) > len(self.TYPES):
                # Identify continuous features (those after one-hot atom types)
                continuous_features = augmented_graph.x[:, len(self.TYPES):]
                
                # Add small Gaussian noise
                noise = torch.randn_like(continuous_features) * 0.02
                augmented_graph.x[:, len(self.TYPES):] = continuous_features + noise
        
        # Randomly perturb edge features (if there are any edges)
        if augmented_graph.edge_attr.size(0) > 0:
            # With small probability, change some edge attributes
            mask = torch.rand(augmented_graph.edge_attr.size(0)) < 0.02
            
            if mask.any():
                # For selected edges, slightly modify the edge features
                selected_edges = augmented_graph.edge_attr[mask]
                
                # Add small noise and renormalize
                noise = torch.randn_like(selected_edges) * 0.1
                augmented_edge_attr = selected_edges + noise
                augmented_edge_attr = F.softmax(augmented_edge_attr, dim=1)
                
                augmented_graph.edge_attr[mask] = augmented_edge_attr
                
        return augmented_graph