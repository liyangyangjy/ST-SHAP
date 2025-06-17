# ST-SHAP
Code and data for key residue prediction in enzymes 
## 1. Project Structure

Before running the code, please set up the following folder structure in your project root directory:
```
project_root/
├── Code/
├── Data/
│ ├── 1ast_73393/1ast_73393_c.nc
│ ├── 1ast_73393/1ast_c.nc
│ └── 1ast_73393/out.pdb
└── Result/
```

---

## 2. Data Download

Due to file size limitations on GitHub, the full dataset is hosted externally:

📦 **Download here**: [Zenodo archive (with DOI)](https://zenodo.org/records/15682888)

After downloading, place the files into the `Data/` directory following the structure above.

---

## 3. Environment Setup

Create and activate the required conda environment:

```bash
conda env create -f st-shap_py37.yml
conda activate st-shap_py37
```

---

## 4. Run the Code

Run the main script with:

```bash
python Code/main_st-shap_multi_100_test.py
```
---

## 5. Results

Detailed prediction results will be saved in:

```
Data/results_st-shap_100_test/
```

Statistical summaries will be stored in:

```
Result/
```

---

## 6. Citation

If you use this code or dataset in your research, please cite the Zenodo DOI or the corresponding paper.
