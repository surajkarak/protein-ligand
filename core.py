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
import matplotlib.pyplot as plt
import os


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

def df_from_pdb(complex_path=None, receptor_path=None, ligand_path=None, box_size=None, ligand_name=None, flag_save=True):
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


def process_multiple_pdb_files(pdb_files, box_size, ligand_name, ligand_path=None):
    results = []
    
    for pdb_file in pdb_files:
        #print(f"Processing {pdb_file}")
        try:
            result_df = df_from_pdb(
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
    


def build_smooth_distribution(dataframe, merge_distributions, group_ids=None, columns=None, smooth_distribution=0, 
                              smooth_tolerance=3, bin_size=0.1, n_bins=50, range=None):
    if group_ids is not None:  
        dataframe = dataframe[dataframe['Residue'].isin(group_ids)] # if no group_ids is specified, we calculate distributions across all residues. If specidied, we filter dataframe for only those residues
    
    if columns is None: # Default case of all columns
        columns = [col for col in dataframe.columns if col != 'Residue']
    
    distributions = {} #Dictionary to store distributions
    
    for column in columns: # Either default case or list of columns we specify
        data = dataframe[column].dropna()
        
        # If the range is not specified, calculate it from the data
        if range is None:
            data_range = (data.min(), data.max())
        else:
            data_range = range
            
        bins = np.linspace(data_range[0], data_range[1], n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Initialize the distribution
        dist = np.zeros(n_bins)
        
        for value in data:
            if smooth_distribution > 0:
                gaussian_contributions = norm.pdf(bin_centers, loc=value, scale=smooth_distribution)
                gaussian_contributions[gaussian_contributions < norm.pdf(smooth_tolerance)] = 0
                dist += gaussian_contributions
            else:
                # Place the value in the appropriate bin without smoothing
                bin_index = np.digitize(value, bins) - 1
                if 0 <= bin_index < len(dist):
                    dist[bin_index] += 1
        dist /= dist.sum() * bin_size  # Normalize the distribution
        
        if merge_distributions: # If we want to merge across columns and multiple columns are specified
            if 'Merged' not in distributions: # For 1st merge
                distributions['Merged'] = dist
            else:
                distributions['Merged'] += dist # Adding next column's distribution
        else: # If we want the distributions separately for each column, output as a dictionary (append to it) with entries for each column
            distributions[column] = dist
    
    return distributions, bin_centers


def process_dataframes(folder, columns, smooth_distribution, smooth_tolerance, bin_size, n_bins, merge_distributions = False, group_ids=None, range=None, sliding_window=0):
    file_list = [f for f in os.listdir(folder) if f.endswith('.pdb')]
    all_distributions = []
    all_bin_centers = []

    pdb_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pdb')]
    df  = process_multiple_pdb_files(pdb_files, box_size, ligand_name, ligand_path)
    print("Processing done")

    # for file in file_list:
    #     filepath = os.path.join(folder, file)
        
        
    distributions, bin_centers = build_smooth_distribution(df, merge_distributions = merge_distributions, group_ids = group_ids, columns = columns, smooth_distribution = smooth_distribution, smooth_tolerance = smooth_tolerance, bin_size = bin_size, n_bins = n_bins, range=None)
    if merge_distributions:
            # Append the merged distribution to the list
            all_distributions.append(distributions['Merged'])
            # Combine all merged distributions into a single plot
            merged_distribution = np.mean(all_distributions, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(bin_centers, merged_distribution, label='Merged Distribution', color='blue')
            plt.fill_between(bin_centers, merged_distribution, color='blue', alpha=0.2)
            plt.title('Merged Distribution Across All Dataframes')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.show()


    else:
        # If not merging, calculate the mean and variance over the distributions
        for col in distributions:
                all_distributions.append(distributions[col])
                all_bin_centers.append(bin_centers)

        # mean_distribution = np.mean(all_distributions, axis=0)
        # variance_distribution = np.var(all_distributions, axis=0)
        
        # Plotting the results
        for i, col in enumerate(distributions.keys()):
            plt.figure(figsize=(10, 6))
            
            plt.plot(all_bin_centers[i], all_distributions[i], label=f'Distribution of {col}', color='blue')
            if sliding_window:
                smoothed_distribution = np.convolve(distributions[col], np.ones(sliding_window)/sliding_window, mode='same')
                plt.plot(all_bin_centers[i], smoothed_distribution, label=f'Smoothed Distribution of {col}', color='orange')
            
            plt.title(f'Distribution of {col} Across Dataframes')
            plt.xlabel('Bins')
            plt.ylabel('Value')
            plt.legend()
            plt.show()


# columns = ['Hydrophobic.distance','VdWContact.distance'] # Change to the columns you need, or set to None to include all columns
# smooth_distribution=0
# smooth_tolerance=3
# bin_size=0.1
# n_bins=50

# # For visualization
# sliding_window=3

# ## For processing PDB files
# box_size = 10.0
# ligand_name = "LLM" # Change to the corresponding letters in the new folder a2a, like ADN, NEC etc.
# ligand_path = None  # If you have a common ligand file, otherwise set to None

def core():
    """
    Main core function to visualize distributions.
    $ python core.py --complex_path /Users/surajkwork/Documents/Projects/ProteinLigand/protein-ligand/MD/md_conf_snap_2/frames --smooth_distribution 0 --smooth_tolerance 3 --bin_size 0.1 --n_bins 50 --merge_distributions True --sliding_window 3

    """
    parser = argparse.ArgumentParser(description="Process complex, receptor, and ligand files.")
    
    # process_dataframes(folder_path, columns, smooth_distribution, smooth_tolerance, bin_size, n_bins, merge_distributions=True, sliding_window=3)

    parser.add_argument('--folder_path', type=Path, help="Path to the folder containing PDB files")
    parser.add_argument('--columns', type=str, help="Name of metric/parameter")
    parser.add_argument('--smooth_distribution', type=float, help="Sigma of the gaussian applied for each sample", default=0)
    parser.add_argument('--smooth_tolerance', type=float, help="float parameter to neglect the gaussian tails")
    parser.add_argument('--bin_size', type=float, help="Size of the bin")
    parser.add_argument('--n_bins', type=int, help="No. of bins")
    parser.add_argument('--merge_distributions', type=bool, help="Merge distributions or not") 
    parser.add_argument('--range', type=tuple, help="tuple of (left_border, right_border) to determine the interval of the distribution.")

    args = parser.parse_args()

if __name__ == 'core':
    core()

