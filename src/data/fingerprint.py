"""
Molecular fingerprint extraction.

Supports various fingerprint types including Morgan (ECFP), RDKit, and MACCS keys.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import numpy as np
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit.DataStructs import ExplicitBitVect

logger = logging.getLogger(__name__)


class FingerprintType(Enum):
    """Supported fingerprint types."""
    MORGAN = "morgan"
    RDKIT = "rdkit"
    MACCS = "maccs"


@dataclass
class FingerprintResult:
    """
    Container for fingerprint extraction results.
    
    Attributes
    ----------
    matrix : scipy.sparse.csr_matrix
        Binary incidence matrix (molecules × features).
    feature_ids : List[int]
        List of feature identifiers (e.g., Morgan bit IDs).
    bit_info : List[Dict]
        Per-molecule bit information (atom mapping for Morgan FPs).
    """
    matrix: sparse.csr_matrix
    feature_ids: List[int]
    bit_info: Optional[List[Dict]] = None
    
    @property
    def n_molecules(self) -> int:
        return self.matrix.shape[0]
    
    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]
    
    def get_molecules_with_feature(self, feature_idx: int) -> np.ndarray:
        """Get indices of molecules containing a specific feature."""
        return self.matrix.getcol(feature_idx).nonzero()[0]
    
    def get_features_of_molecule(self, mol_idx: int) -> np.ndarray:
        """Get feature indices present in a specific molecule."""
        return self.matrix.getrow(mol_idx).nonzero()[1]


class FingerprintExtractor:
    """
    Extract molecular fingerprints and build incidence matrix.
    
    Parameters
    ----------
    fp_type : str or FingerprintType
        Type of fingerprint to use. Options: 'morgan', 'rdkit', 'maccs'.
    radius : int
        Radius for Morgan fingerprints. Default is 2 (ECFP4).
    n_bits : int, optional
        Number of bits for hashed fingerprints. None for unhashed.
    use_features : bool
        Use feature-based fingerprints (FCFP instead of ECFP).
    use_chirality : bool
        Include chirality in fingerprints.
    
    Examples
    --------
    >>> extractor = FingerprintExtractor(fp_type='morgan', radius=2)
    >>> result = extractor.extract(molecules)
    >>> print(f"Generated {result.n_features} unique features")
    """
    
    def __init__(
        self,
        fp_type: Union[str, FingerprintType] = FingerprintType.MORGAN,
        radius: int = 2,
        n_bits: Optional[int] = None,
        use_features: bool = False,
        use_chirality: bool = False
    ):
        if isinstance(fp_type, str):
            fp_type = FingerprintType(fp_type.lower())
        
        self.fp_type = fp_type
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        self.use_chirality = use_chirality
    
    def extract(
        self, 
        molecules: List[Chem.Mol],
        return_bit_info: bool = True
    ) -> FingerprintResult:
        """
        Extract fingerprints from molecules.
        
        Parameters
        ----------
        molecules : List[Chem.Mol]
            List of RDKit molecule objects.
        return_bit_info : bool
            Whether to return bit mapping information.
        
        Returns
        -------
        FingerprintResult
            Fingerprint extraction results.
        """
        if self.fp_type == FingerprintType.MORGAN:
            return self._extract_morgan(molecules, return_bit_info)
        elif self.fp_type == FingerprintType.RDKIT:
            return self._extract_rdkit(molecules)
        elif self.fp_type == FingerprintType.MACCS:
            return self._extract_maccs(molecules)
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fp_type}")
    
    def _extract_morgan(
        self, 
        molecules: List[Chem.Mol],
        return_bit_info: bool = True
    ) -> FingerprintResult:
        """Extract Morgan (ECFP/FCFP) fingerprints."""
        all_features: Dict[int, List[int]] = {}  # feature_id -> [mol_indices]
        bit_info_list = []
        
        for mol_idx, mol in enumerate(molecules):
            if mol is None:
                bit_info_list.append({})
                continue
            
            bit_info = {}
            
            if self.n_bits is not None:
                # Hashed fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, 
                    self.radius, 
                    nBits=self.n_bits,
                    useFeatures=self.use_features,
                    useChirality=self.use_chirality,
                    bitInfo=bit_info
                )
                on_bits = list(fp.GetOnBits())
            else:
                # Unhashed fingerprint (count-based)
                fp = AllChem.GetMorganFingerprint(
                    mol,
                    self.radius,
                    useFeatures=self.use_features,
                    useChirality=self.use_chirality,
                    bitInfo=bit_info
                )
                on_bits = list(fp.GetNonzeroElements().keys())
            
            bit_info_list.append(bit_info if return_bit_info else {})
            
            for bit in on_bits:
                if bit not in all_features:
                    all_features[bit] = []
                all_features[bit].append(mol_idx)
        
        # Build sparse matrix
        matrix, feature_ids = self._build_sparse_matrix(
            all_features, len(molecules)
        )
        
        logger.info(
            f"Extracted {len(feature_ids)} unique Morgan features "
            f"(radius={self.radius}) from {len(molecules)} molecules"
        )
        
        return FingerprintResult(
            matrix=matrix,
            feature_ids=feature_ids,
            bit_info=bit_info_list if return_bit_info else None
        )
    
    def _extract_rdkit(self, molecules: List[Chem.Mol]) -> FingerprintResult:
        """Extract RDKit topological fingerprints."""
        all_features: Dict[int, List[int]] = {}
        
        for mol_idx, mol in enumerate(molecules):
            if mol is None:
                continue
            
            fp = RDKFingerprint(mol)
            on_bits = list(fp.GetOnBits())
            
            for bit in on_bits:
                if bit not in all_features:
                    all_features[bit] = []
                all_features[bit].append(mol_idx)
        
        matrix, feature_ids = self._build_sparse_matrix(
            all_features, len(molecules)
        )
        
        logger.info(
            f"Extracted {len(feature_ids)} unique RDKit features "
            f"from {len(molecules)} molecules"
        )
        
        return FingerprintResult(matrix=matrix, feature_ids=feature_ids)
    
    def _extract_maccs(self, molecules: List[Chem.Mol]) -> FingerprintResult:
        """Extract MACCS keys."""
        all_features: Dict[int, List[int]] = {}
        
        for mol_idx, mol in enumerate(molecules):
            if mol is None:
                continue
            
            fp = MACCSkeys.GenMACCSKeys(mol)
            on_bits = list(fp.GetOnBits())
            
            for bit in on_bits:
                if bit not in all_features:
                    all_features[bit] = []
                all_features[bit].append(mol_idx)
        
        matrix, feature_ids = self._build_sparse_matrix(
            all_features, len(molecules)
        )
        
        logger.info(
            f"Extracted {len(feature_ids)} unique MACCS keys "
            f"from {len(molecules)} molecules"
        )
        
        return FingerprintResult(matrix=matrix, feature_ids=feature_ids)
    
    def _build_sparse_matrix(
        self, 
        features: Dict[int, List[int]], 
        n_molecules: int
    ) -> Tuple[sparse.csr_matrix, List[int]]:
        """Build sparse incidence matrix from feature dictionary."""
        feature_ids = sorted(features.keys())
        feature_to_idx = {fid: idx for idx, fid in enumerate(feature_ids)}
        
        rows = []
        cols = []
        
        for feature_id, mol_indices in features.items():
            col_idx = feature_to_idx[feature_id]
            for mol_idx in mol_indices:
                rows.append(mol_idx)
                cols.append(col_idx)
        
        data = np.ones(len(rows), dtype=np.float32)
        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_molecules, len(feature_ids))
        )
        
        return matrix, feature_ids
    
    def get_feature_counts(self, result: FingerprintResult) -> np.ndarray:
        """Get the count of molecules for each feature."""
        return np.asarray(result.matrix.sum(axis=0)).flatten()
    
    def filter_features(
        self,
        result: FingerprintResult,
        min_count: int = 2,
        max_count: Optional[int] = None
    ) -> FingerprintResult:
        """
        Filter features by occurrence count.
        
        Parameters
        ----------
        result : FingerprintResult
            Original fingerprint result.
        min_count : int
            Minimum number of molecules containing the feature.
        max_count : int, optional
            Maximum number of molecules containing the feature.
        
        Returns
        -------
        FingerprintResult
            Filtered fingerprint result.
        """
        counts = self.get_feature_counts(result)
        
        mask = counts >= min_count
        if max_count is not None:
            mask &= counts <= max_count
        
        kept_indices = np.where(mask)[0]
        
        filtered_matrix = result.matrix[:, kept_indices]
        filtered_ids = [result.feature_ids[i] for i in kept_indices]
        
        logger.info(
            f"Filtered features: {result.n_features} -> {len(filtered_ids)} "
            f"(min_count={min_count}, max_count={max_count})"
        )
        
        return FingerprintResult(
            matrix=filtered_matrix,
            feature_ids=filtered_ids,
            bit_info=result.bit_info
        )
