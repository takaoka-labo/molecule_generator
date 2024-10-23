import os
import subprocess
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import cclib
from rdkit.Chem import rdmolfiles

def read_smiles_from_csv(csv_filename):
    smiles_list = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                smiles_list.append(row[0])
    return smiles_list

def generate_molecule_from_smiles(smiles, index):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"SMILESのパースに失敗しました: {smiles}")
        return None
    mol = Chem.AddHs(mol)
    return mol

def generate_conformers_with_rdkit(mol, index, num_confs=1):
    params = AllChem.ETKDGv3()
    params.numThreads = 0  # 0は利用可能なすべてのCPUを使用
    params.pruneRmsThresh = 0.1  # 重複コンフォーマーの削除閾値
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if not ids:
        print(f"RDKitでのコンフォーマー生成に失敗しました: 分子 {index}")
        return False
    # UFFまたはMMFFで最適化
    for confId in ids:
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=confId)
        except:
            print(f"UFF最適化に失敗しました: 分子 {index}, コンフォーマー {confId}")
            return False
    # 最適化されたコンフォーマーをSDFファイルに書き出し
    sdf_file = f"mol_{index}/mol_{index}_rdkit_optim.sdf"
    writer = Chem.SDWriter(sdf_file)
    for confId in ids:
        writer.write(mol, confId=confId)
    writer.close()
    print(f"RDKitでのコンフォーマー生成と最適化が完了しました: {sdf_file}")
    return True

def save_molecule_to_mol2_with_openbabel(sdf_file, index):
    mol2_file = f"mol_{index}/mol_{index}_before_gaussian.mol2"
    
    # Open Babelを使ってSDFファイルをMOL2に変換
    command = ['obabel', sdf_file, '-O', mol2_file, '--gen3D']
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Open BabelでMOL2ファイルの生成に失敗しました: {result.stderr}")
        return False
    
    print(f"MOL2ファイルに保存しました: {mol2_file}")
    return True

def generate_gaussian_input(index):
    # RDKitで最適化されたSDFを読み込み
    optimized_sdf = f"mol_{index}/mol_{index}_rdkit_optim.sdf"
    suppl = Chem.SDMolSupplier(optimized_sdf, removeHs=False)
    mol = suppl[0]
    if mol is None:
        print(f"最適化されたSDFファイルの読み込みに失敗しました: {optimized_sdf}")
        return False

    # Gaussian入力ファイルの作成（構造最適化用）
    gaussian_input_file = f"mol_{index}/mol_opt_{index}.gjf"
    conformer = mol.GetConformer()
    with open(gaussian_input_file, 'w') as f:
        f.write('%nprocshared=64\n')
        f.write('%chk=temp\n')
        f.write('# b3lyp/6-311g* opt pop=full gfprint\n\n')
        f.write('Optimization\n\n')
        f.write('0 1\n')
        for atom in mol.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            f.write(f'{symbol}    {pos.x:.6f}    {pos.y:.6f}    {pos.z:.6f}\n')
        f.write('\n')
    print(f"Gaussian入力ファイルを生成しました: {gaussian_input_file}")
    return True

def run_gaussian_optimization(index):
    gaussian_input_file = f"mol_{index}/mol_opt_{index}.gjf"
    gaussian_output_file = f"mol_{index}/mol_opt_{index}.log"

    if not os.path.exists(gaussian_input_file):
        print(f"Gaussianの入力ファイルが存在しません: {gaussian_input_file}")
        return False

    # Gaussianの実行（構造最適化）
    process = subprocess.Popen(['g16', gaussian_input_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()  # プロセスの完了を待つ
    if process.returncode != 0:
        print(f"Gaussianの実行に失敗しました: {stderr.decode()}")
        return False
    else:
        # 出力ファイルの存在確認
        if not os.path.exists(gaussian_output_file):
            print(f"Gaussianの出力ファイルが見つかりません: {gaussian_output_file}")
            return False
        if os.path.getsize(gaussian_output_file) == 0:
            print(f"Gaussianの出力ファイルが空です: {gaussian_output_file}")
            return False
        print(f"Gaussianの実行が完了しました: {gaussian_output_file}")
        return True

def extract_optimized_coords(gaussian_output_file):
    data = cclib.io.ccread(gaussian_output_file)
    if data is None:
        print(f"Gaussian出力ファイルの読み込みに失敗しました: {gaussian_output_file}")
        return None
    atoms = data.atomnos
    coords = data.atomcoords[-1]  # 最終ステップの座標
    optimized_coords = []
    for atom_num, coord in zip(atoms, coords):
        symbol = Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(atom_num))
        x, y, z = coord
        optimized_coords.append(f'{symbol}    {x:.6f}    {y:.6f}    {z:.6f}')
    return optimized_coords

def generate_gaussian_esp_input(index):
    # 最適化された構造をGaussian出力ファイルから抽出
    gaussian_output_file = f"mol_{index}/mol_opt_{index}.log"
    optimized_coords = extract_optimized_coords(gaussian_output_file)
    if optimized_coords is None:
        print(f"最適化された構造の抽出に失敗しました。")
        return False

    # Gaussian入力ファイルの作成（ESP計算用）
    esp_input_file = f"mol_esp_{index}.gjf"
    with open(esp_input_file, 'w') as f:
        f.write('%nprocshared=64\n')
        f.write('%chk=temp\n')
        f.write('#p b3lyp/6-311G* pop=mk gfprint scf=tight\n')
        f.write('iop(6/41=10,6/42=17,6/50=1)\n\n')
        f.write('ESPcalculation\n\n')
        f.write('0 1\n')
        for coord_line in optimized_coords:
            f.write(coord_line + '\n')
        f.write('\n')
        f.write(f'mol_{index}/mol_esp_{index}.esp\n')
        
    print(f"ESP計算用のGaussian入力ファイルを生成しました: {esp_input_file}")
    return True

def run_gaussian_esp_calculation(index):
    esp_input_file = f"mol_{index}/mol_esp_{index}.gjf"
    esp_output_file = f"mol_{index}/mol_esp_{index}.esp"

    if not os.path.exists(esp_input_file):
        print(f"Gaussianの入力ファイルが存在しません: {esp_input_file}")
        return False

    # Gaussianの実行（ESP計算）
    process = subprocess.Popen(['g16', esp_input_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()  # プロセスの完了を待つ
    if process.returncode != 0:
        print(f"Gaussianの実行に失敗しました: {stderr.decode()}")
        return False
    else:
        # 出力ファイルの存在確認
        if not os.path.exists(esp_output_file):
            print(f"Gaussianの出力ファイルが見つかりません: {esp_output_file}")
            return False
        if os.path.getsize(esp_output_file) == 0:
            print(f"Gaussianの出力ファイルが空です: {esp_output_file}")
            return False
        print(f"Gaussianの実行が完了しました: {esp_output_file}")
        return True

def extract_charge_and_multiplicity(esp_file):
    try:
        with open(esp_file, 'r') as f:
            for line in f:
                if "CHARGE =" in line and "MULTIPLICITY =" in line:
                    parts = line.split()
                    charge = int(parts[2])
                    multiplicity = int(parts[6])
                    return charge, multiplicity
    except Exception as e:
        print(f"CHARGEとMULTIPLICITYの取得に失敗しました。エラー: {e}")
    return None, None

def run_resp_fitting(index):
    esp_file = f"mol_{index}/mol_esp_{index}.esp"
    
    # CHARGEとMULTIPLICITYをESPファイルから取得
    charge, multiplicity = extract_charge_and_multiplicity(esp_file)
    if charge is None or multiplicity is None:
        print("CHARGEとMULTIPLICITYの取得に失敗しました。")
        return False

    mol2_file = f"mol_{index}/mol_{index}.mol2"
    command = [
        'antechamber',
        '-fi', 'gesp',
        '-i', esp_file,
        '-fo', 'mol2',
        '-o', mol2_file,
        '-c', 'resp',
        '-nc', str(charge),
        '-m', str(multiplicity),
        '-dr', 'no'
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"RESPの実行に失敗しました: {result.stderr}")
        return False
    else:
        print(f"RESPの実行が完了しました: {mol2_file}")
        return True
    
def rewrite_mol2_file(index):
    # Define the mapping from GAFF to standard atom types
    atom_type_mapping = {
        # 炭素 (Carbon)
        'c1': 'C.sp',    # sp炭素
        'c2': 'C.sp2',   # sp2炭素
        'c3': 'C.3',     # sp3炭素
        'ca': 'C.ar',    # 芳香族炭素 (sp2)
        'cp': 'C.2',     # 共役系の炭素 (sp2)
        'cc': 'C.2',     # カルバニオン系炭素 (sp2)
        
        # 水素 (Hydrogen)
        'h1': 'H.sp',    # sp炭素に結合した水素
        'h2': 'H.sp2',   # sp2炭素に結合した水素
        'h3': 'H.sp3',   # sp3炭素に結合した水素
        'ha': 'H.ar',    # 芳香族炭素に結合した水素
        'hc': 'H',       # 飽和炭化水素に結合した水素 (sp3炭素)
        'ho': 'H',       # ヒドロキシ基 (-OH) に結合した水素
        'hs': 'H',       # 硫黄に結合した水素

        # 酸素 (Oxygen)
        'o': 'O.2',      # カルボニル基 (C=O) の酸素
        'oh': 'O.3',     # ヒドロキシ基の酸素 (sp3)
        'os': 'O.3',     # エーテルやアルコールの酸素 (sp3)

        # 窒素 (Nitrogen)
        'n': 'N.2',      # sp2窒素 (非芳香環)
        'n2': 'N.2',     # sp2窒素 (アゾ基)
        'n3': 'N.3',     # sp3窒素 (アミン)
        'n4': 'N.4',     # sp3窒素 (四級アンモニウム)
        'na': 'N.ar',    # 芳香環窒素 (sp2, ピリジン)
        'nb': 'N.ar',    # 五員芳香環窒素 (sp2)
        'nc': 'N.2',     # カルバニオン系窒素 (sp2)
        'nd': 'N.2',     # 五員環のsp2窒素 (共役)

        # 硫黄 (Sulfur)
        's': 'S.2',      # スルホン基の硫黄
        'ss': 'S.3',     # スルフィド/チオールの硫黄
        'sh': 'S.3',     # スルファニル基の硫黄

        # リン (Phosphorus)
        'p3': 'P.3',     # sp3リン
        'p5': 'P.3',

        # ハロゲン (Halogen)
        'f': 'F',        # フッ素
        'cl': 'Cl',      # 塩素
        'br': 'Br',      # 臭素
        'i': 'I'         # ヨウ素
            
            # Add more mappings as necessary
    }
    
    # Read the MOL2 file content
    with open(f'mol_{index}/mol_{index}.mol2', 'r') as file:
        mol2_content = file.readlines()  # Read line by line
    
    converted_lines = []
    atom_section = False

    for line in mol2_content: 
        
        if 'resp' in line:
            line = line.replace('resp', 'USER_CHARGES')
        
        # Detect the beginning of the @<TRIPOS>ATOM section
        if line.strip() == "@<TRIPOS>ATOM":
            atom_section = True
        elif line.strip() == "@<TRIPOS>BOND":
            atom_section = False
        
        # Only process lines in the ATOM section
        if atom_section and len(line.split()) >= 9:
            # Split the line into components, keeping track of positions
            components = line.split()

            # Extract the atom type (6th column)
            atom_type = components[5]

            # Replace GAFF type with standard type if in the mapping
            if atom_type in atom_type_mapping:
                components[5] = atom_type_mapping[atom_type.lower()]  # Convert atom type to standard

            # Reassemble the line with the exact spacing preserved
            line = f"{components[0]:>7} {components[1]:<8} {float(components[2]):>14.6f} {float(components[3]):>14.6f} {float(components[4]):>14.6f} {components[5]:<8} {components[6]:>3} {components[7]:<6} {float(components[8]):>14.9f}\n"
        
        # Add the (potentially modified) line to the converted lines
        converted_lines.append(line)

    # Write the converted content to the output file
    with open(f'mol_{index}/mol_{index}_.mol2', 'w') as file:
        file.writelines(converted_lines)
    
    print(f"Conversion complete. Output saved to mol_{index}/mol_{index}_.mol2")


def main():
    smiles_list = read_smiles_from_csv('smiles_list.csv')

    for index, smiles in enumerate(smiles_list, start=1):
        print(f"\n=== 分子 {index} の処理を開始します ===")
        mol = generate_molecule_from_smiles(smiles, index)
        if mol is None:
            continue

        if not generate_conformers_with_rdkit(mol, index):
            continue
        
        sdf_file = f"mol_{index}/mol_{index}_rdkit_optim.sdf"
        if not save_molecule_to_mol2_with_openbabel(sdf_file, index):
            continue

        if not generate_gaussian_input(index):
            continue

        if not run_gaussian_optimization(index):
            continue

        if not generate_gaussian_esp_input(index):
            continue

        if not run_gaussian_esp_calculation(index):
            continue

        if not run_resp_fitting(index):
            continue
        
        rewrite_mol2_file(index)
        print(f"分子 {index} の処理が完了しました。最終的なMOL2ファイル: mol_{index}/mol_{index}.mol2")

if __name__ == '__main__':
    main()
