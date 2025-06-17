# ST-SHAP
Code and data for key residue prediction in enzymes 
1. Create 'Data' and 'Result' folders, parallel to the 'Code' folder.

project_root/  
├── Code/  
├── Data/
|     └──1ast_73393 
|             ├──1ast_73393_c.nc
|             ├──1ast_c.nc
|             └──out.pdb
└── Result/  

3. ## Data Download

Due to file size limitations, the full dataset is hosted externally:
- [Zenodo archive (with DOI)](https://zenodo.org/records/15682888)

4. Then run the code using the following command:

     conda env create -f st-shap_py37.yml
     conda activate st-shap_py37
     python Code/main_st-shap_multi_100_test.py

5. View detailed results in 'Data\results_st-shap_100_test' and check statistical summaries in the 'Result' folder.
