#!/usr/bin/env python3


import os
import sys
import json
import subprocess
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Bioinformatics libraries
try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
except ImportError:
    print("Installing required bioinformatics libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "biotite", "biopython"])
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

# ML/AI libraries
try:
    import torch
    import torch.nn.functional as F
    from transformers import EsmModel, EsmTokenizer
except ImportError:
    print("Installing PyTorch and transformers...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers"])
    import torch
    import torch.nn.functional as F
    from transformers import EsmModel, EsmTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binder_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BinderCandidate:
    """Data structure for storing binder candidate information"""
    id: str
    sequence: str
    structure_path: str
    binding_score: float
    folding_confidence: float
    interaction_energy: float
    clash_score: float
    druglikeness_score: float
    expressibility_score: float
    generation_method: str
    target_region: str
    
class PipelineConfig:
    """Configuration class for the binder design pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_config = {
            # Target protein configuration
            "target_pdb_path": "kainate_receptor_ATD.pdb",
            "target_chain": "A",
            "binding_site_residues": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],  # hop bot region
            
            # RFdiffusion parameters
            "rfdiffusion": {
                "num_designs": 1000,
                "length": [50, 80],  # binder length range
                "hotspot_residues": [],
                "contigs": "50-80",
                "iterations": 50,
                "noise_scale": 1.0,
                "scaffold_guided": True
            },
            
            # ProteinMPNN parameters
            "proteinmpnn": {
                "num_sequences": 8,
                "temperature": 0.1,
                "batch_size": 1,
                "omit_AA": "CX",  # Omit cysteine and unknown amino acids
                "bias_AA": {"K": 0.1, "R": 0.1, "E": -0.1, "D": -0.1}  # Bias for/against certain AAs
            },
            
            # AlphaFold parameters
            "alphafold": {
                "model_type": "alphafold3",  # or "alphafold2"
                "max_msa_clusters": 512,
                "max_extra_msa": 1024,
                "relaxation_steps": 200
            },
            
            # Filtering thresholds
            "filtering": {
                "min_binding_score": 5.0,
                "min_folding_confidence": 0.7,
                "max_clash_score": 2.0,
                "min_interaction_energy": -10.0,
                "min_expressibility": 0.6
            },
            
            # Output configuration
            "output_dir": "binder_designs",
            "max_final_candidates": 20,
            "parallel_processes": 4
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._update_nested_dict(self.base_config, user_config)
    
    def _update_nested_dict(self, base_dict, update_dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        current = self.base_config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

class EnvironmentManager:
    """Manages conda environments and dependencies"""
    
    @staticmethod
    def setup_rfdiffusion_env():
        """Set up RFdiffusion environment"""
        env_name = "rfdiffusion"
        logger.info(f"Setting up {env_name} environment...")
        
        commands = [
            f"conda create -n {env_name} python=3.9 -y",
            f"conda activate {env_name}",
            "conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y",
            "pip install hydra-core omegaconf",
            "git clone https://github.com/RosettaCommons/RFdiffusion.git",
            "cd RFdiffusion && pip install -e .",
            "cd .."
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                logger.info(f"Successfully executed: {cmd}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to execute: {cmd}")
                logger.error(f"Error: {e.stderr}")
                raise
    
    @staticmethod
    def setup_proteinmpnn_env():
        """Set up ProteinMPNN environment"""
        env_name = "proteinmpnn"
        logger.info(f"Setting up {env_name} environment...")
        
        commands = [
            f"conda create -n {env_name} python=3.8 -y",
            f"conda activate {env_name}",
            "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y",
            "git clone https://github.com/dauparas/ProteinMPNN.git",
            "cd ProteinMPNN && pip install -r requirements.txt",
            "cd .."
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                logger.info(f"Successfully executed: {cmd}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to execute: {cmd}")
                logger.error(f"Error: {e.stderr}")
                raise
    
    @staticmethod
    def setup_alphafold_env():
        """Set up AlphaFold environment"""
        env_name = "alphafold"
        logger.info(f"Setting up {env_name} environment...")
        
        commands = [
            f"conda create -n {env_name} python=3.9 -y",
            f"conda activate {env_name}",
            "pip install colabfold[alphafold] --no-warn-conflicts",
            "pip install alphafold3-pytorch",
            "pip install jax jaxlib",
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                logger.info(f"Successfully executed: {cmd}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to execute: {cmd}")
                logger.error(f"Error: {e.stderr}")
                raise

class StructureAnalyzer:
    """Analyzes protein structures and binding interactions"""
    
    @staticmethod
    def load_structure(pdb_path: str):
        """Load protein structure from PDB file"""
        try:
            with open(pdb_path, 'r') as f:
                structure = pdb.PDBFile.read(f)
            return structure
        except Exception as e:
            logger.error(f"Failed to load structure from {pdb_path}: {e}")
            raise
    
    @staticmethod
    def calculate_binding_score(target_struct, binder_struct) -> float:
        """Calculate binding score between target and binder"""
        # Simplified binding score calculation
        # In practice, you'd use more sophisticated methods like PyRosetta scoring
        
        # Get atom coordinates
        target_coords = target_struct.coord
        binder_coords = binder_struct.coord
        
        # Calculate minimum distance between structures
        min_distances = []
        for t_coord in target_coords:
            distances = np.linalg.norm(binder_coords - t_coord, axis=1)
            min_distances.append(np.min(distances))
        
        # Simple scoring based on interface area and distance
        interface_contacts = np.sum(np.array(min_distances) < 4.0)  # Within 4Å
        binding_score = interface_contacts / len(min_distances) * 10.0
        
        return binding_score
    
    @staticmethod
    def check_clashes(target_struct, binder_struct, clash_threshold: float = 2.0) -> float:
        """Check for steric clashes between target and binder"""
        target_coords = target_struct.coord
        binder_coords = binder_struct.coord
        
        clash_count = 0
        total_contacts = 0
        
        for t_coord in target_coords:
            distances = np.linalg.norm(binder_coords - t_coord, axis=1)
            contacts = distances < 4.0
            clashes = distances < clash_threshold
            
            total_contacts += np.sum(contacts)
            clash_count += np.sum(clashes)
        
        if total_contacts == 0:
            return 0.0
        
        return clash_count / total_contacts * 100.0

class RFDiffusionRunner:
    """Wrapper for running RFdiffusion"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.rfdiffusion_path = "RFdiffusion"
        
    def run_generation(self, target_pdb: str, output_dir: str) -> List[str]:
        """Run RFdiffusion to generate binder backbones"""
        logger.info("Starting RFdiffusion binder generation...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare RFdiffusion command
        cmd = [
            "conda", "run", "-n", "rfdiffusion", "python", 
            f"{self.rfdiffusion_path}/scripts/run_inference.py",
            f"inference.output_prefix={output_dir}/binder",
            f"inference.input_pdb={target_pdb}",
            f"inference.num_designs={self.config.get('rfdiffusion.num_designs')}",
            f"contigmap.contigs=[{self.config.get('rfdiffusion.contigs')}]",
            f"inference.ckpt_override_path={self.rfdiffusion_path}/models/Base_ckpt.pt",
            "inference.write_trajectory=False"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("RFdiffusion completed successfully")
            
            # Collect generated PDB files
            generated_pdbs = list(Path(output_dir).glob("binder_*.pdb"))
            logger.info(f"Generated {len(generated_pdbs)} binder backbones")
            
            return [str(pdb) for pdb in generated_pdbs]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"RFdiffusion failed: {e.stderr}")
            raise

class ProteinMPNNRunner:
    """Wrapper for running ProteinMPNN"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.proteinmpnn_path = "ProteinMPNN"
        
    def design_sequences(self, backbone_pdbs: List[str], output_dir: str) -> Dict[str, List[str]]:
        """Design sequences for binder backbones using ProteinMPNN"""
        logger.info(f"Designing sequences for {len(backbone_pdbs)} backbones...")
        
        os.makedirs(output_dir, exist_ok=True)
        sequence_designs = {}
        
        for pdb_path in backbone_pdbs:
            pdb_name = Path(pdb_path).stem
            
            # Prepare ProteinMPNN command
            cmd = [
                "conda", "run", "-n", "proteinmpnn", "python",
                f"{self.proteinmpnn_path}/protein_mpnn_run.py",
                "--pdb_path", pdb_path,
                "--out_folder", output_dir,
                "--num_seq_per_target", str(self.config.get('proteinmpnn.num_sequences')),
                "--sampling_temp", str(self.config.get('proteinmpnn.temperature')),
                "--batch_size", str(self.config.get('proteinmpnn.batch_size')),
                "--omit_AAs", self.config.get('proteinmpnn.omit_AA', '')
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Parse output sequences
                fasta_file = Path(output_dir) / f"seqs/{pdb_name}.fa"
                if fasta_file.exists():
                    sequences = []
                    for record in SeqIO.parse(fasta_file, "fasta"):
                        sequences.append(str(record.seq))
                    sequence_designs[pdb_name] = sequences
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"ProteinMPNN failed for {pdb_path}: {e.stderr}")
                continue
        
        total_sequences = sum(len(seqs) for seqs in sequence_designs.values())
        logger.info(f"Generated {total_sequences} sequence designs")
        
        return sequence_designs

class AlphaFoldRunner:
    """Wrapper for running AlphaFold predictions"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def predict_complex_structure(self, target_seq: str, binder_seq: str, output_dir: str) -> Tuple[str, float]:
        """Predict structure of target-binder complex"""
        logger.info("Predicting complex structure with AlphaFold...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create combined sequence for complex prediction
        complex_seq = target_seq + ":" + binder_seq
        
        # Use ColabFold for fast structure prediction
        cmd = [
            "conda", "run", "-n", "alphafold",
            "colabfold_batch",
            "--amber", "--use-gpu-relax",
            complex_seq,
            output_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find predicted structure file
            pdb_files = list(Path(output_dir).glob("*.pdb"))
            if pdb_files:
                pdb_path = str(pdb_files[0])
                
                # Extract confidence score from file
                confidence = self._extract_confidence_score(pdb_path)
                
                return pdb_path, confidence
            else:
                logger.error("No PDB file generated by AlphaFold")
                return None, 0.0
                
        except subprocess.CalledProcessError as e:
            logger.error(f"AlphaFold prediction failed: {e.stderr}")
            return None, 0.0
    
    def _extract_confidence_score(self, pdb_path: str) -> float:
        """Extract confidence score from AlphaFold PDB file"""
        try:
            structure = self._load_structure(pdb_path)
            # AlphaFold stores confidence in B-factor field
            confidence_scores = structure.b_factor
            return np.mean(confidence_scores) / 100.0  # Convert to 0-1 scale
        except:
            return 0.0
    
    def _load_structure(self, pdb_path: str):
        """Load structure using biotite"""
        with open(pdb_path, 'r') as f:
            return pdb.PDBFile.read(f)

class BinderScorer:
    """Comprehensive scoring system for binder candidates"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Load ESM model for protein representation
        try:
            self.esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.esm_model.eval()
        except:
            logger.warning("Could not load ESM model for advanced scoring")
            self.esm_model = None
    
    def score_binder(self, candidate: BinderCandidate, target_structure_path: str) -> BinderCandidate:
        """Comprehensive scoring of binder candidate"""
        
        # Load structures
        try:
            target_struct = StructureAnalyzer.load_structure(target_structure_path)
            binder_struct = StructureAnalyzer.load_structure(candidate.structure_path)
            
            # Calculate binding score
            candidate.binding_score = StructureAnalyzer.calculate_binding_score(target_struct, binder_struct)
            
            # Calculate clash score
            candidate.clash_score = StructureAnalyzer.check_clashes(target_struct, binder_struct)
            
            # Calculate interaction energy (simplified)
            candidate.interaction_energy = self._estimate_interaction_energy(candidate.sequence)
            
            # Calculate druglikeness score
            candidate.druglikeness_score = self._calculate_druglikeness(candidate.sequence)
            
            # Calculate expressibility score
            candidate.expressibility_score = self._calculate_expressibility(candidate.sequence)
            
        except Exception as e:
            logger.error(f"Error scoring candidate {candidate.id}: {e}")
            # Set default low scores
            candidate.binding_score = 0.0
            candidate.clash_score = 100.0
            candidate.interaction_energy = 0.0
            candidate.druglikeness_score = 0.0
            candidate.expressibility_score = 0.0
        
        return candidate
    
    def _estimate_interaction_energy(self, sequence: str) -> float:
        """Estimate interaction energy based on sequence composition"""
        # Simplified energy calculation based on amino acid properties
        energy_values = {
            'R': -1.5, 'K': -1.2, 'H': -0.8,  # Positive charges
            'D': -1.3, 'E': -1.1,             # Negative charges
            'F': -0.9, 'W': -1.0, 'Y': -0.7,  # Aromatics
            'L': -0.5, 'I': -0.5, 'V': -0.4,  # Hydrophobics
            'S': -0.2, 'T': -0.2, 'N': -0.3, 'Q': -0.3,  # Polar
            'G': 0.0, 'A': -0.1, 'P': 0.2,    # Small/flexible
            'C': -0.6, 'M': -0.4               # Sulfur-containing
        }
        
        total_energy = sum(energy_values.get(aa, 0.0) for aa in sequence)
        return total_energy
    
    def _calculate_druglikeness(self, sequence: str) -> float:
        """Calculate druglikeness score based on sequence properties"""
        # Check for problematic sequences
        if len(sequence) > 100:  # Too long
            return 0.1
        if len(sequence) < 20:   # Too short
            return 0.2
        
        # Count problematic amino acids
        problematic_count = sequence.count('C') + sequence.count('M') + sequence.count('P')
        if problematic_count > len(sequence) * 0.2:  # More than 20% problematic
            return 0.3
        
        # Balanced composition score
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # Check for diversity
        diversity = len(aa_counts) / 20.0  # Number of different AAs / 20
        
        return min(1.0, diversity + 0.3)
    
    def _calculate_expressibility(self, sequence: str) -> float:
        """Calculate bacterial expressibility score"""
        # Factors that affect expression in E. coli
        
        # Avoid rare codons (simplified)
        rare_aa_penalty = sequence.count('R') + sequence.count('L') * 0.5
        rare_penalty = min(0.5, rare_aa_penalty / len(sequence))
        
        # Avoid aggregation-prone sequences
        hydrophobic_stretch = self._find_max_hydrophobic_stretch(sequence)
        aggregation_penalty = min(0.3, hydrophobic_stretch / 10.0)
        
        # Avoid too many charged residues
        charged_count = sequence.count('R') + sequence.count('K') + sequence.count('D') + sequence.count('E')
        charge_penalty = min(0.2, (charged_count / len(sequence) - 0.3))
        
        base_score = 1.0
        final_score = base_score - rare_penalty - aggregation_penalty - max(0, charge_penalty)
        
        return max(0.0, final_score)
    
    def _find_max_hydrophobic_stretch(self, sequence: str) -> int:
        """Find maximum consecutive hydrophobic amino acid stretch"""
        hydrophobic = set(['F', 'W', 'Y', 'L', 'I', 'V', 'M', 'A'])
        
        max_stretch = 0
        current_stretch = 0
        
        for aa in sequence:
            if aa in hydrophobic:
                current_stretch += 1
                max_stretch = max(max_stretch, current_stretch)
            else:
                current_stretch = 0
        
        return max_stretch

class BinderPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = PipelineConfig(config_path)
        self.output_dir = Path(self.config.get('output_dir'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.rfdiffusion = RFDiffusionRunner(self.config)
        self.proteinmpnn = ProteinMPNNRunner(self.config)
        self.alphafold = AlphaFoldRunner(self.config)
        self.scorer = BinderScorer(self.config)
        
        # Results storage
        self.candidates: List[BinderCandidate] = []
        
    def setup_environments(self):
        """Set up all required conda environments"""
        logger.info("Setting up computational environments...")
        
        try:
            EnvironmentManager.setup_rfdiffusion_env()
            EnvironmentManager.setup_proteinmpnn_env()
            EnvironmentManager.setup_alphafold_env()
            logger.info("All environments set up successfully")
        except Exception as e:
            logger.error(f"Failed to set up environments: {e}")
            raise
    
    def run_pipeline(self, target_pdb_path: str, target_sequence: str):
        """Run the complete binder design pipeline"""
        logger.info("Starting binder design pipeline...")
        
        # Step 1: Generate binder backbones with RFdiffusion
        backbone_dir = self.output_dir / "backbones"
        backbone_pdbs = self.rfdiffusion.run_generation(target_pdb_path, str(backbone_dir))
        
        # Filter backbones (remove obviously bad structures)
        filtered_backbones = self._filter_backbones(backbone_pdbs)
        logger.info(f"Filtered to {len(filtered_backbones)} good backbones")
        
        # Step 2: Design sequences with ProteinMPNN
        sequence_dir = self.output_dir / "sequences"
        sequence_designs = self.proteinmpnn.design_sequences(filtered_backbones, str(sequence_dir))
        
        # Step 3: Predict complex structures and score candidates
        self._process_candidates(sequence_designs, target_sequence, target_pdb_path)
        
        # Step 4: Filter and rank candidates
        final_candidates = self._filter_and_rank_candidates()
        
        # Step 5: Generate final report
        self._generate_report(final_candidates)
        
        logger.info("Pipeline completed successfully")
        return final_candidates
    
    def _filter_backbones(self, backbone_pdbs: List[str]) -> List[str]:
        """Filter backbone structures based on quality metrics"""
        filtered = []
        
        for pdb_path in backbone_pdbs:
            try:
                structure = StructureAnalyzer.load_structure(pdb_path)
                
                # Basic quality checks
                if len(structure.coord) < 20:  # Too small
                    continue
                if len(structure.coord) > 200:  # Too large
                    continue
                
                # Check for reasonable secondary structure (simplified)
                # In practice, you'd use DSSP or similar
                
                filtered.append(pdb_path)
                
            except Exception as e:
                logger.warning(f"Failed to analyze backbone {pdb_path}: {e}")
                continue
        
        return filtered
    
    def _process_candidates(self, sequence_designs: Dict[str, List[str]], 
                          target_sequence: str, target_pdb_path: str):
        """Process all candidate binders"""
        logger.info("Processing candidate binders...")
        
        candidate_id = 0
        complex_dir = self.output_dir / "complexes"
        complex_dir.mkdir(exist_ok=True)
        
        for backbone_name, sequences in sequence_designs.items():
            for seq_idx, sequence in enumerate(sequences):
                candidate_id += 1
                
                # Create candidate
                candidate = BinderCandidate(
                    id=f"binder_{candidate_id:04d}",
                    sequence=sequence,
                    structure_path="",  # Will be filled after AlphaFold
                    binding_score=0.0,
                    folding_confidence=0.0,
                    interaction_energy=0.0,
                    clash_score=0.0,
                    druglikeness_score=0.0,
                    expressibility_score=0.0,
                    generation_method="RFdiffusion+ProteinMPNN",
                    target_region="amino_terminal_domain"
                )
                
                # Predict complex structure
                complex_output_dir = complex_dir / candidate.id
                complex_output_dir.mkdir(exist_ok=True)
                
                structure_path, confidence = self.alphafold.predict_complex_structure(
                    target_sequence, sequence, str(complex_output_dir)
                )
                
                if structure_path:
                    candidate.structure_path = structure_path
                    candidate.folding_confidence = confidence
                    
                    # Score the candidate
                    candidate = self.scorer.score_binder(candidate, target_pdb_path)
                    
                    self.candidates.append(candidate)
                
                # Progress logging
                if candidate_id % 50 == 0:
                    logger.info(f"Processed {candidate_id} candidates...")
    
    def _filter_and_rank_candidates(self) -> List[BinderCandidate]:
        """Filter and rank candidate binders"""
        logger.info(f"Filtering and ranking {len(self.candidates)} candidates...")
        
        # Apply filtering thresholds
        filtered_candidates = []
        thresholds = self.config.get('filtering')
        
        for candidate in self.candidates:
            if (candidate.binding_score >= thresholds['min_binding_score'] and
                candidate.folding_confidence >= thresholds['min_folding_confidence'] and
                candidate.clash_score <= thresholds['max_clash_score'] and
                candidate.interaction_energy <= thresholds['min_interaction_energy'] and
                candidate.expressibility_score >= thresholds['min_expressibility']):
                
                filtered_candidates.append(candidate)
        
        logger.info(f"Filtered to {len(filtered_candidates)} candidates")
        
        # Rank by composite score
        for candidate in filtered_candidates:
            candidate.composite_score = self._calculate_composite_score(candidate)
        
        # Sort by composite score (higher is better)
        ranked_candidates = sorted(filtered_candidates, 
                                 key=lambda x: x.composite_score, reverse=True)
        
        # Return top candidates
        max_candidates = self.config.get('max_final_candidates')
        return ranked_candidates[:max_candidates]
    
    def _calculate_composite_score(self, candidate: BinderCandidate) -> float:
        """Calculate composite score for ranking"""
        # Weighted combination of different scores
        weights = {
            'binding': 0.3,
            'confidence': 0.25,
            'interaction': 0.2,
            'expressibility': 0.15,
            'druglikeness': 0.1
        }
        
        # Normalize scores to 0-1 range
        normalized_binding = min(1.0, candidate.binding_score / 10.0)
        normalized_interaction = min(1.0, abs(candidate.interaction_energy) / 20.0)
        clash_penalty = max(0.0, 1.0 - candidate.clash_score / 10.0)
        
        composite = (
            weights['binding'] * normalized_binding +
            weights['confidence'] * candidate.folding_confidence +
            weights['interaction'] * normalized_interaction +
            weights['expressibility'] * candidate.expressibility_score +
            weights['druglikeness'] * candidate.druglikeness_score
        ) * clash_penalty  # Apply clash penalty
        
        return composite
    
    def _generate_report(self, final_candidates: List[BinderCandidate]):
        """Generate comprehensive report of results"""
        logger.info("Generating final report...")
        
        # Create report directory
        report_dir = self.output_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        
        # Generate summary statistics
        summary = {
            'total_candidates_generated': len(self.candidates),
            'candidates_passing_filters': len(final_candidates),
            'pipeline_completion_time': datetime.now().isoformat(),
            'top_candidates': []
        }
        
        # Detailed candidate information
        for i, candidate in enumerate(final_candidates[:10]):  # Top 10
            candidate_info = {
                'rank': i + 1,
                'id': candidate.id,
                'sequence': candidate.sequence,
                'sequence_length': len(candidate.sequence),
                'binding_score': round(candidate.binding_score, 3),
                'folding_confidence': round(candidate.folding_confidence, 3),
                'interaction_energy': round(candidate.interaction_energy, 3),
                'clash_score': round(candidate.clash_score, 3),
                'expressibility_score': round(candidate.expressibility_score, 3),
                'druglikeness_score': round(candidate.druglikeness_score, 3),
                'composite_score': round(candidate.composite_score, 3)
            }
            summary['top_candidates'].append(candidate_info)
        
        # Save JSON report
        with open(report_dir / "summary_report.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate FASTA file with top sequences
        top_sequences = []
        for i, candidate in enumerate(final_candidates):
            record = SeqRecord(
                Seq(candidate.sequence),
                id=candidate.id,
                description=f"rank_{i+1}_score_{candidate.composite_score:.3f}"
            )
            top_sequences.append(record)
        
        SeqIO.write(top_sequences, report_dir / "top_binder_sequences.fasta", "fasta")
        
        # Generate detailed CSV report
        df_data = []
        for candidate in final_candidates:
            df_data.append({
                'ID': candidate.id,
                'Sequence': candidate.sequence,
                'Length': len(candidate.sequence),
                'Binding_Score': candidate.binding_score,
                'Folding_Confidence': candidate.folding_confidence,
                'Interaction_Energy': candidate.interaction_energy,
                'Clash_Score': candidate.clash_score,
                'Expressibility_Score': candidate.expressibility_score,
                'Druglikeness_Score': candidate.druglikeness_score,
                'Composite_Score': candidate.composite_score,
                'Structure_Path': candidate.structure_path
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(report_dir / "detailed_candidates.csv", index=False)
        
        # Generate analysis plots (if matplotlib available)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Score distribution plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            scores = ['binding_score', 'folding_confidence', 'interaction_energy', 
                     'expressibility_score', 'druglikeness_score', 'composite_score']
            
            for i, score in enumerate(scores):
                values = [getattr(candidate, score) for candidate in final_candidates]
                axes[i].hist(values, bins=20, alpha=0.7)
                axes[i].set_title(f'{score.replace("_", " ").title()} Distribution')
                axes[i].set_xlabel(score.replace("_", " ").title())
                axes[i].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(report_dir / "score_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            logger.info("Matplotlib not available, skipping plot generation")
        
        logger.info(f"Report generated in {report_dir}")
    
    def save_checkpoint(self, checkpoint_name: str = "pipeline_checkpoint.pkl"):
        """Save pipeline state for recovery"""
        checkpoint_path = self.output_dir / checkpoint_name
        
        checkpoint_data = {
            'config': self.config.base_config,
            'candidates': self.candidates,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name: str = "pipeline_checkpoint.pkl"):
        """Load pipeline state from checkpoint"""
        checkpoint_path = self.output_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file {checkpoint_path} not found")
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.candidates = checkpoint_data['candidates']
            logger.info(f"Loaded {len(self.candidates)} candidates from checkpoint")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

class DockerManager:
    """Manages Docker containers for isolated execution"""
    
    @staticmethod
    def create_dockerfile():
        """Create Dockerfile for the pipeline"""
        dockerfile_content = """
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and setup script
COPY requirements.txt .
COPY setup_environments.sh .

# Make setup script executable
RUN chmod +x setup_environments.sh

# Run environment setup
RUN ./setup_environments.sh

# Copy pipeline code
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Set up entrypoint
ENTRYPOINT ["python", "binder_pipeline.py"]
"""
        
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_content)
    
    @staticmethod
    def create_requirements_file():
        """Create requirements.txt file"""
        requirements = """
biotite>=0.37.0
biopython>=1.79
torch>=1.11.0
transformers>=4.20.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
plotly>=5.0.0
"""
        
        with open("requirements.txt", 'w') as f:
            f.write(requirements)
    
    @staticmethod
    def create_setup_script():
        """Create environment setup script"""
        setup_script = """#!/bin/bash
set -e

echo "Setting up RFdiffusion environment..."
conda create -n rfdiffusion python=3.9 -y
source activate rfdiffusion
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install hydra-core omegaconf
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion && pip install -e . && cd ..

echo "Setting up ProteinMPNN environment..."
conda create -n proteinmpnn python=3.8 -y
source activate proteinmpnn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
git clone https://github.com/dauparas/ProteinMPNN.git
cd ProteinMPNN && pip install -r requirements.txt && cd ..

echo "Setting up AlphaFold environment..."
conda create -n alphafold python=3.9 -y
source activate alphafold
pip install colabfold[alphafold] --no-warn-conflicts
pip install alphafold3-pytorch
pip install jax jaxlib

echo "Environment setup complete!"
"""
        
        with open("setup_environments.sh", 'w') as f:
            f.write(setup_script)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Protein Binder Design Pipeline")
    
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--target-pdb", type=str, required=True, 
                       help="Target protein PDB file path")
    parser.add_argument("--target-sequence", type=str, required=True,
                       help="Target protein sequence")
    parser.add_argument("--setup-envs", action="store_true",
                       help="Set up conda environments")
    parser.add_argument("--docker-setup", action="store_true",
                       help="Create Docker setup files")
    parser.add_argument("--load-checkpoint", type=str,
                       help="Load from checkpoint file")
    parser.add_argument("--output-dir", type=str, default="binder_designs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Docker setup mode
    if args.docker_setup:
        logger.info("Creating Docker setup files...")
        DockerManager.create_dockerfile()
        DockerManager.create_requirements_file()
        DockerManager.create_setup_script()
        logger.info("Docker files created successfully")
        return
    
    # Initialize pipeline
    pipeline = BinderPipeline(args.config)
    
    # Set output directory if specified
    if args.output_dir:
        pipeline.output_dir = Path(args.output_dir)
        pipeline.output_dir.mkdir(exist_ok=True)
    
    try:
        # Environment setup mode
        if args.setup_envs:
            pipeline.setup_environments()
            return
        
        # Load checkpoint if specified
        if args.load_checkpoint:
            if not pipeline.load_checkpoint(args.load_checkpoint):
                logger.error("Failed to load checkpoint, starting fresh")
        
        # Validate input files
        if not os.path.exists(args.target_pdb):
            logger.error(f"Target PDB file not found: {args.target_pdb}")
            return
        
        if not args.target_sequence:
            logger.error("Target sequence is required")
            return
        
        # Run the pipeline
        logger.info("Starting binder design pipeline...")
        final_candidates = pipeline.run_pipeline(args.target_pdb, args.target_sequence)
        
        # Save final checkpoint
        pipeline.save_checkpoint("final_checkpoint.pkl")
        
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Generated {len(final_candidates)} high-quality binder candidates")
        logger.info(f"Results saved in: {pipeline.output_dir}")
        
        # Print top 5 candidates
        print("\n" + "="*50)
        print("TOP 5 BINDER CANDIDATES")
        print("="*50)
        
        for i, candidate in enumerate(final_candidates[:5]):
            print(f"\nRank {i+1}: {candidate.id}")
            print(f"Sequence: {candidate.sequence}")
            print(f"Length: {len(candidate.sequence)} residues")
            print(f"Composite Score: {candidate.composite_score:.3f}")
            print(f"Binding Score: {candidate.binding_score:.3f}")
            print(f"Folding Confidence: {candidate.folding_confidence:.3f}")
            print(f"Expressibility: {candidate.expressibility_score:.3f}")
            print("-" * 30)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        pipeline.save_checkpoint("interrupted_checkpoint.pkl")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        pipeline.save_checkpoint("error_checkpoint.pkl")
        raise

if __name__ == "__main__":
    main()

# Additional utility functions for pipeline customization

class CustomScorer:
    """Extended scoring functions for specialized applications"""
    
    @staticmethod
    def calculate_membrane_compatibility(sequence: str) -> float:
        """Score for membrane protein binders"""
        hydrophobic_aas = set(['F', 'W', 'I', 'L', 'V', 'M', 'Y', 'A'])
        hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aas)
        return hydrophobic_count / len(sequence)
    
    @staticmethod
    def calculate_protease_resistance(sequence: str) -> float:
        """Score for protease resistance"""
        # Avoid protease cleavage sites
        problematic_motifs = ['KR', 'RR', 'KK', 'FK', 'YK', 'WK']
        penalty = sum(sequence.count(motif) for motif in problematic_motifs)
        return max(0.0, 1.0 - penalty / len(sequence))
    
    @staticmethod
    def calculate_immunogenicity_risk(sequence: str) -> float:
        """Estimate immunogenicity risk"""
        # Simplified immunogenicity assessment
        # Avoid human-like sequences and common T-cell epitopes
        
        # Check for poly-basic regions
        basic_stretch = 0
        max_basic_stretch = 0
        basic_aas = set(['K', 'R', 'H'])
        
        for aa in sequence:
            if aa in basic_aas:
                basic_stretch += 1
                max_basic_stretch = max(max_basic_stretch, basic_stretch)
            else:
                basic_stretch = 0
        
        risk_score = min(1.0, max_basic_stretch / 5.0)  # Penalize stretches > 5
        return 1.0 - risk_score

class PipelineOptimizer:
    """Optimize pipeline parameters based on results"""
    
    def __init__(self, pipeline: BinderPipeline):
        self.pipeline = pipeline
    
    def optimize_parameters(self, validation_set: List[BinderCandidate]) -> Dict:
        """Optimize pipeline parameters using validation results"""
        
        # Analyze which parameters correlate with success
        successful_candidates = [c for c in validation_set if c.composite_score > 0.7]
        
        if not successful_candidates:
            logger.warning("No successful candidates found for optimization")
            return {}
        
        # Analyze sequence properties of successful candidates
        avg_length = np.mean([len(c.sequence) for c in successful_candidates])
        avg_hydrophobic = np.mean([
            sum(1 for aa in c.sequence if aa in 'FWILVMA') / len(c.sequence)
            for c in successful_candidates
        ])
        
        optimized_params = {
            'rfdiffusion': {
                'length': [int(avg_length * 0.8), int(avg_length * 1.2)],
                'num_designs': min(2000, self.pipeline.config.get('rfdiffusion.num_designs') * 2)
            },
            'filtering': {
                'min_binding_score': np.percentile([c.binding_score for c in successful_candidates], 25),
                'min_folding_confidence': np.percentile([c.folding_confidence for c in successful_candidates], 25)
            }
        }
        
        logger.info(f"Optimized parameters: {optimized_params}")
        return optimized_params

# Example configuration file template
EXAMPLE_CONFIG = """
{
  "target_pdb_path": "kainate_receptor_ATD.pdb",
  "target_chain": "A",
  "binding_site_residues": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
  
  "rfdiffusion": {
    "num_designs": 1000,
    "length": [50, 80],
    "contigs": "50-80",
    "iterations": 50,
    "scaffold_guided": true
  },
  
  "proteinmpnn": {
    "num_sequences": 8,
    "temperature": 0.1,
    "omit_AA": "CX",
    "bias_AA": {"K": 0.1, "R": 0.1, "E": -0.1, "D": -0.1}
  },
  
  "alphafold": {
    "model_type": "alphafold3",
    "max_msa_clusters": 512,
    "relaxation_steps": 200
  },
  
  "filtering": {
    "min_binding_score": 5.0,
    "min_folding_confidence": 0.7,
    "max_clash_score": 2.0,
    "min_interaction_energy": -10.0,
    "min_expressibility": 0.6
  },
  
  "output_dir": "binder_designs",
  "max_final_candidates": 20,
  "parallel_processes": 4
}
"""

def create_example_config():
    """Create example configuration file"""
    with open("example_config.json", 'w') as f:
        f.write(EXAMPLE_CONFIG)
    logger.info("Example configuration file created: example_config.json")

# Usage examples and documentation
USAGE_EXAMPLES = """
USAGE EXAMPLES:
===============

1. Set up environments (run once):
   python binder_pipeline.py --setup-envs

2. Create Docker files:
   python binder_pipeline.py --docker-setup

3. Run full pipeline:
   python binder_pipeline.py \\
     --target-pdb kainate_receptor.pdb \\
     --target-sequence "MKVLWAALLV..." \\
     --config my_config.json \\
     --output-dir my_binders

4. Resume from checkpoint:
   python binder_pipeline.py \\
     --target-pdb kainate_receptor.pdb \\
     --target-sequence "MKVLWAALLV..." \\
     --load-checkpoint interrupted_checkpoint.pkl

5. Docker execution:
   docker build -t binder-pipeline .
   docker run -v $(pwd)/data:/app/data binder-pipeline \\
     --target-pdb data/target.pdb \\
     --target-sequence "SEQUENCE..."

TROUBLESHOOTING:
================

1. Environment issues:
   - Ensure CUDA is available for GPU acceleration
   - Use conda to manage environments
   - Check disk space (pipeline needs ~50GB)

2. Memory issues:
   - Reduce num_designs in config
   - Increase swap space
   - Use smaller batch sizes

3. Network issues:
   - Download models manually if automatic download fails
   - Check firewall settings for git clone operations
"""

if __name__ == "__main__":
    # Print usage if no arguments provided
    if len(sys.argv) == 1:
        print(USAGE_EXAMPLES)
        create_example_config()
    else:
        main()

