"""
Molecule data loading utilities.

Supports loading molecules from various file formats including CSV, SDF, and SMILES files.
"""

from typing import List, Optional, Union
from pathlib import Path
import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

logger = logging.getLogger(__name__)


class MoleculeLoader:
    """
    Load molecules from various file formats.
    
    Supported formats:
    - CSV with SMILES column
    - SDF files
    - Plain text files with SMILES (one per line)
    
    Parameters
    ----------
    filepath : str or Path
        Path to the molecule file.
    smiles_col : str, optional
        Name of the SMILES column for CSV files. Default is 'smiles'.
    mol_col : str, optional
        Name of the molecule column for DataFrame output. Default is 'mol'.
    
    Examples
    --------
    >>> loader = MoleculeLoader("molecules.csv")
    >>> molecules = loader.load()
    >>> print(f"Loaded {len(molecules)} molecules")
    """
    
    SUPPORTED_FORMATS = {'.csv', '.sdf', '.smi', '.txt'}
    
    def __init__(
        self,
        filepath: Union[str, Path],
        smiles_col: str = 'smiles',
        mol_col: str = 'mol'
    ):
        self.filepath = Path(filepath)
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        suffix = self.filepath.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
    
    def load(self, max_molecules: Optional[int] = None) -> List[Chem.Mol]:
        """
        Load molecules from file.
        
        Parameters
        ----------
        max_molecules : int, optional
            Maximum number of molecules to load. None for all.
        
        Returns
        -------
        List[Chem.Mol]
            List of RDKit molecule objects.
        """
        suffix = self.filepath.suffix.lower()
        
        if suffix == '.csv':
            molecules = self._load_csv(max_molecules)
        elif suffix == '.sdf':
            molecules = self._load_sdf(max_molecules)
        else:  # .smi or .txt
            molecules = self._load_smiles(max_molecules)
        
        # Filter out None values (failed parsing)
        valid_molecules = [mol for mol in molecules if mol is not None]
        
        n_failed = len(molecules) - len(valid_molecules)
        if n_failed > 0:
            logger.warning(f"Failed to parse {n_failed} molecules")
        
        logger.info(f"Loaded {len(valid_molecules)} valid molecules from {self.filepath}")
        
        return valid_molecules
    
    def load_with_metadata(
        self, 
        max_molecules: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load molecules with associated metadata.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with molecules and metadata.
        """
        suffix = self.filepath.suffix.lower()
        
        if suffix == '.csv':
            df = pd.read_csv(self.filepath, nrows=max_molecules)
            PandasTools.AddMoleculeColumnToFrame(
                df, self.smiles_col, self.mol_col, includeFingerprints=False
            )
        elif suffix == '.sdf':
            df = PandasTools.LoadSDF(
                str(self.filepath), 
                molColName=self.mol_col,
                smilesName=self.smiles_col
            )
            if max_molecules:
                df = df.head(max_molecules)
        else:
            molecules = self._load_smiles(max_molecules)
            df = pd.DataFrame({
                self.mol_col: molecules,
                self.smiles_col: [Chem.MolToSmiles(m) if m else None for m in molecules]
            })
        
        # Filter rows with valid molecules
        df = df[df[self.mol_col].notna()].reset_index(drop=True)
        
        return df
    
    def _load_csv(self, max_molecules: Optional[int] = None) -> List[Chem.Mol]:
        """Load molecules from CSV file."""
        df = pd.read_csv(self.filepath, nrows=max_molecules)
        
        if self.smiles_col not in df.columns:
            raise ValueError(
                f"Column '{self.smiles_col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        
        molecules = []
        for smiles in df[self.smiles_col]:
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                molecules.append(mol)
            except Exception as e:
                logger.debug(f"Failed to parse SMILES '{smiles}': {e}")
                molecules.append(None)
        
        return molecules
    
    def _load_sdf(self, max_molecules: Optional[int] = None) -> List[Chem.Mol]:
        """Load molecules from SDF file."""
        suppl = Chem.SDMolSupplier(str(self.filepath))
        
        molecules = []
        for i, mol in enumerate(suppl):
            if max_molecules and i >= max_molecules:
                break
            molecules.append(mol)
        
        return molecules
    
    def _load_smiles(self, max_molecules: Optional[int] = None) -> List[Chem.Mol]:
        """Load molecules from SMILES file (one per line)."""
        molecules = []
        
        with open(self.filepath, 'r') as f:
            for i, line in enumerate(f):
                if max_molecules and i >= max_molecules:
                    break
                
                smiles = line.strip().split()[0]  # Take first token
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    molecules.append(mol)
                except Exception as e:
                    logger.debug(f"Failed to parse SMILES '{smiles}': {e}")
                    molecules.append(None)
        
        return molecules


def load_molecules_from_url(
    url: str, 
    save_path: Optional[str] = None,
    **kwargs
) -> List[Chem.Mol]:
    """
    Load molecules from a URL.
    
    Parameters
    ----------
    url : str
        URL to the molecule file.
    save_path : str, optional
        Path to save the downloaded file.
    **kwargs
        Additional arguments passed to MoleculeLoader.
    
    Returns
    -------
    List[Chem.Mol]
        List of RDKit molecule objects.
    """
    import urllib.request
    import tempfile
    
    if save_path is None:
        # Determine extension from URL
        suffix = Path(url).suffix or '.csv'
        fd, save_path = tempfile.mkstemp(suffix=suffix)
    
    logger.info(f"Downloading molecules from {url}")
    urllib.request.urlretrieve(url, save_path)
    
    loader = MoleculeLoader(save_path, **kwargs)
    return loader.load()
