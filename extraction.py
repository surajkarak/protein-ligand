#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
from tqdm import tqdm
import prolif as plf
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_types
from pathlib import Path


# In[2]:


IPythonConsole.ipython_useSVG = True

# Initialize progress bar for pandas
# tqdm.pandas()


# In[3]:


def calculate_center_of_mass(ligand_path):
    """
    Calculate the center of mass of one ligand.

    Parameters
    ----------
    ligand_path : Path
        Path to sdf file with ligands.

    Returns
    -------
    np.array
        np.array with x, y, z coordinates of the ligand.
    """

    suppl = Chem.SDMolSupplier(ligand_path)
    mol = suppl[0]  # Assuming only one molecule in sdf file

    conformer = mol.GetConformer()  #Gets the 3D conformer of the molecule. 
    positions = conformer.GetPositions() # Retrieves the 3D coordinates (x, y, z) of each atom.
    masses = np.array([atom.GetMass() for atom in mol.GetAtoms()]) #Creates an array of atomic masses for each atom in the molecule.

    center_of_mass = np.average(positions, axis=0, weights=masses) #Calculates the weighted average of the atom positions using their masses to get the center of mass.
    return center_of_mass


# In[4]:


def get_aminoacids_df(df: pd.DataFrame) -> list:
    """
    Get list of amino acids which interact with one specific ligand.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with interaction fingerprints per one ligand and all residues in boolean True/False representation.

    Returns
    -------
    list
        List of residues involved in interaction with one specific ligand.
    """
    index = pd.MultiIndex.from_tuples(df.keys(), names=['ligand', 'protein', 'interaction']) #creates a MultiIndex from the keys of the dataframe df
    df2 = pd.DataFrame(df, columns=index) # new dataframe using  original dataframe df but columns indexed by the ligand, protein, and interaction levels.
    aas = [col[1] for col in df2.columns] #extracts the second element (protein) from each column's tuple.
    return aas


# In[5]:


def get_df_from_fp(fp1: dict, residue: str) -> pd.DataFrame:
    """
    Calculation of DataFrame from interaction fingerprint.

    Parameters
    ----------
    fp1 : dict
        Individual interaction fingerprint per one ligand with one specific residue.
    residue : str
        Residue name.

    Returns
    -------
    pd.DataFrame
        DataFrame with individual interaction fingerprint per one ligand with one specific residue and calculated parameters for bonds.
    """
    rows = []
    for interaction, details in fp1.items(): 
        for detail in details:
            row = {'interaction': interaction}
            for key, value in detail.items():
                if key not in ['indices', 'parent_indices']:  # Be careful, we are missing info about interacting indices
                    row[f'{interaction}.{key}'] = value
            rows.append(row)
    df1 = pd.DataFrame(rows)
    df1.insert(0, 'Residue', residue)
    df1 = df1.drop(columns=['interaction'])
    return df1 


# In[6]:


def prepare_complex(complex_path, box_size, ligand_name) -> tuple[plf.Molecule, list[plf.Molecule]]:
    """
    When the input is based only on the Path to complex (PDB) without sdf file with ligands.

    Parameters
    ----------
   --complex_path : Path or str
        Path or str to individual PDB file (complex of protein + ligand).
    --box_size : float
        Value in Angstroems defining protein area around ligand
    --ligand_name : str
        Name of the residue of ligand in PDB complex.

    Returns
    -------
    tuple[plf.Molecule, list[plf.Molecule]]
        Two prolif Molecule objects: selected protein and ligands from sdf file.
    """
    complex = mda.Universe(complex_path, guess_bonds=True) 
    elements = guess_types(complex.atoms.names)
    complex.add_TopologyAttr("elements", elements)
    ligand_selection = complex.select_atoms(f"resname {ligand_name}")
    if len(ligand_selection) == 0:
        raise ValueError(f"Ligand {ligand_name} is not detected in the protein.")
    protein_selection = complex.select_atoms(
        f"protein and byres around {box_size} group ligand", ligand=ligand_selection
    )
    protein_mol = plf.Molecule.from_mda(protein_selection) #Converts selected protein atoms to a prolif Molecule object.
    ligand_mol = plf.Molecule.from_mda(ligand_selection) #Converts selected ligand atoms to a prolif Molecule object.
    ligand_path = Path("ligand.sdf")
    writer = Chem.SDWriter(ligand_path)
    writer.write(ligand_mol)
    writer.close()
    lig_suppl = list(plf.sdf_supplier(ligand_path))
    return protein_mol, lig_suppl


# In[7]:


def prepare_receptor_ligands(receptor_path, ligand_path, box_size) -> tuple[plf.Molecule, list[plf.Molecule]]:
    """
    When the input is based on the Path to receptor (PDB, protein) with sdf file with ligands.

    Parameters
    ----------
    --receptor_path : Path or str
        Path or str to individual PDB file (protein only).
    --ligand_path : Path or str
        Path or str to sdf file with docked ligand.
    --box_size : float
        Value in Angstroems defining protein area around ligand.

    Returns
    -------
    tuple[plf.Molecule, list[plf.Molecule]]
        Two prolif Molecule objects: selected protein and ligands from sdf file.
    """
    receptor = mda.Universe(receptor_path, guess_bonds=True)
    elements = guess_types(receptor.atoms.names)
    receptor.add_TopologyAttr("elements", elements)
    center = calculate_center_of_mass(ligand_path)
    if center.all():
        x, y, z = center
        selection_string = f"protein and (byres point {x} {y} {z} {box_size})"
        protein_selection = receptor.select_atoms(selection_string)
        protein_mol = plf.Molecule.from_mda(protein_selection)
        lig_suppl = list(plf.sdf_supplier(ligand_path))
        return protein_mol, lig_suppl


# In[8]:


def prepare_complex_ligands(complex_path, ligand_path, box_size, ligand_name) -> tuple[plf.Molecule, list[plf.Molecule]]:
    """
    When the input is based on the Path to complex (PDB) with sdf file with ligands.

    Parameters
    ----------
   --complex_path : Path or str
        Path or str to individual PDB file (complex of protein + ligand).
        --ligand_path : Path or str
        Path or str to sdf file with docked ligand.
    --box_size : float
        Value in Angstroems defining protein area around ligand
    --ligand_name : str
        Name of the residue of ligand in PDB complex.

    Returns
    -------
    tuple[plf.Molecule, list[plf.Molecule]]
        Two prolif Molecule objects: selected protein and ligands from sdf file.
    """
    complex = mda.Universe(complex_path, guess_bonds=True)
    elements = guess_types(complex.atoms.names)
    complex.add_TopologyAttr("elements", elements)
    ligand_selection = complex.select_atoms(f"resname {ligand_name}")
    protein_selection = complex.select_atoms(
        f"protein and byres around {box_size} group ligand", ligand=ligand_selection
    )
    protein_mol = plf.Molecule.from_mda(protein_selection)
    lig_suppl = list(plf.sdf_supplier(ligand_path))
    return protein_mol, lig_suppl


# In[9]:


def calculate_ifp(protein_mol, lig_suppl, flag_save: bool = True) -> pd.DataFrame:
    """
    Calculate interaction fingerprint with all bonds parameters (angle, length, DHAngel etc.)
    for two prolif objects: selected protein and ligand.

    Parameters
    ----------
   --protein_mol : plf.Molecule
        Selected area of protein for IFP calculation as prolif Molecule object.
        --lig_suppl : list[plf.Molecule]
        Ligand object for IFP calculation as prolif Molecule object.

    Returns
    -------
    Union[pd.DataFrame, None]
        pd.DataFrame with individual interaction fingerprint per one ligand with one specific residue and calculated parameters for bonds.
    """
    fp = plf.Fingerprint()
    fp.run_from_iterable(lig_suppl, protein_mol)
    df = fp.to_dataframe()
    dfs = []
    residues_list = get_aminoacids_df(df)
    for residue in residues_list:
        fp1 = fp.ifp[0][('UNL1', residue)]  # In individual IFP (interaction fingerprint), name of ligand and frame are fixed
        df = get_df_from_fp(fp1, residue)
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)
    if flag_save:
        final_df.to_csv('IFP_test.csv', index=False)
    return final_df


# In[10]:


def main(complex_path=None, receptor_path=None, ligand_path=None, box_size=None, ligand_name=None, flag_save=True):
    def validate_and_execute(complex_path=None, receptor_path=None, ligand_path=None, box_size=None, ligand_name=None, flag_save=True):
        if complex_path and ligand_name and not ligand_path:
            return prepare_complex(complex_path, box_size, ligand_name)
        elif complex_path and ligand_path and ligand_name:
            return prepare_complex_ligands(complex_path, ligand_path, box_size, ligand_name)
        elif receptor_path and ligand_path:
            return prepare_receptor_ligands(receptor_path, ligand_path, box_size)
        else:
            raise ValueError("Invalid input combination. Please provide the correct arguments: complex PDB alone, complex PDB and ligands (sdf file) or receptor and ligands (sdf file).")

    protein_mol, lig_suppl = validate_and_execute(
        complex_path=complex_path,
        receptor_path=receptor_path,
        ligand_path=ligand_path,
        box_size=box_size,
        ligand_name=ligand_name,
        flag_save=flag_save
    )

    final_df = calculate_ifp(protein_mol, lig_suppl, flag_save)
    return final_df 


# In[11]:


#from main import main  

def process_multiple_pdb_files(pdb_files, box_size, ligand_name, ligand_path=None):
    results = []
    
    for pdb_file in pdb_files:
        #print(f"Processing {pdb_file}")
        try:
            result_df = main(
                complex_path=pdb_file,
                ligand_path=ligand_path,
                box_size=box_size,
                ligand_name=ligand_name,
                flag_save=False   
            )
            if result_df is not None and not result_df.empty:
                result_df['PDB_File'] = pdb_file  # Add a column to identify the source PDB file
                results.append(result_df)
            else:
                print(f"No data returned for {pdb_file}")
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        print("No results to concatenate")
        return pd.DataFrame()  # Return an empty dataframe if there are no results




# In[13]:


# pdb_files = [f"frames/tmp_{i}.pdb" for i in range(501)]
# box_size = 10.0
# ligand_name = "LLM"
# ligand_path = None  # If you have a common ligand file, otherwise set to None

# final_results_df = process_multiple_pdb_files(pdb_files, box_size, ligand_name, ligand_path)


# # In[10]:


# final_results_df['PDB_File'] = final_results_df['PDB_File'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))


# # In[11]:


# final_results_df.shape


# # In[12]:


# final_results_df.to_csv('processed.csv', index=False)


# # # Extracting dataframe from 2nd & 3rd set of PDB files

# # In[15]:


# pdb_files2 = [f"/Users/surajkwork/Documents/Projects/ProteinLigand/protein-ligand/MD/md_conf_snap_2/frames/tmp_{i}.pdb" for i in range(201)]
# box_size = 10.0
# ligand_name = "LLM"
# ligand_path = None   

# df2 = process_multiple_pdb_files(pdb_files2, box_size, ligand_name, ligand_path)
# df2['PDB_File'] = df2['PDB_File'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))


# # In[16]:


# df2.shape


# # In[17]:


# df2.head(10)


# # In[18]:


# df2.to_csv('md_conf_snap_2_processed.csv', index=False)


# # In[19]:


# pdb_files3 = [f"/Users/surajkwork/Documents/Projects/ProteinLigand/protein-ligand/MD/md_conf_snap_4/frames/tmp_{i}.pdb" for i in range(201)]
# box_size = 10.0
# ligand_name = "LLM"
# ligand_path = None   

# df3 = process_multiple_pdb_files(pdb_files2, box_size, ligand_name, ligand_path)
# df3['PDB_File'] = df3['PDB_File'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))


# # In[20]:


# df3.shape


# # In[21]:


# df3.head(10)


# # In[22]:


# df3.to_csv('md_conf_snap_4_processed.csv', index=False)

