# Introduction
This project implements a test case prioritization method, based on test case diversification and fault-proneness estimation using defect prediction.
Test case prioritization methods aim to benefit testing of a software (specifically regression testing), by prioritizing test cases in an order that minimizes the expected time of executing failing test cases. This details of the algorithm is described in this paper: [Test case prioritization using test case diversification and fault-proneness estimations](https://arxiv.org/abs/2106.10524).

# Usage
This package is used in multiple steps: defect prediction, prioritization and result aggregation. The neccesary steps in order to execture the whole package once are listed below:

1. Get the code:
    ```
    git clone https://github.com/mostafamahdieh/ClusteringFaultPronenessTCP
    ```
2. Get the [Defects4J+M](https://github.com/khesoem/Defects4J-Plus-M) repository in the same main directory, naming it WTP-data:
    ```
    git clone https://github.com/khesoem/Defects4J-Plus-M.git WTP-data
    ```
3. Install python and neccesary packages:
    ```
    sudo apt-get install python3 python3-pip python3-venv
    ```
4. Create a python virtual environment and install neccesary pip packages:
    ```
    cd ClusteringFaultPronenessTCP
    python3 -m venv venv
    source ./venv/bin/activate
    pip3 install -r requirements.txt
    ```
5. Defect prediction: The defect prediction step can be executed using the bugprediction_runner.py script as follows. This script runs the bug prediction step for the specific versions of all projects.
    ```
    cd bugprediction
    python3 -u bugprediction_runner.py
    ```

6. CovClustering Test case prioritization: The prioritization_clustering_nofp_runner.py script is used to execute the CovClustering TCP method, which is one of the proposed methods presented in the introduction. This method does not use fault-proneness and is solely based on coverage using a clustering method to diversify test cases.
    ```
    python3 -u prioritization_clustering_nofp_runner.py
    ```


7. CovClustering+FP Test case prioritization: The prioritization_clustering_fp_runner.py script is used to execute the CovClustering+FP TCP method, which is the proposed method presented in the introduction which utilizes fault-proneness results.
    ```
    python3 -u prioritization_clustering_fp_runner.py
    ```


8. Aggregating the results: The results are aggregated using the aggregate_results_runner_main_results.py script:
    ```
    python3 -u aggregate_results_runner_main_results.py
    ```

## Citing in academic work
If you are using this project for your research, we would be really glad if you cite our paper using the following bibtex:
```
@article{mahdieh2021test,
  title={Test case prioritization using test case diversification and fault-proneness estimations},
  author={Mahdieh, Mostafa and Mirian-Hosseinabadi, Seyed-Hassan and Mahdieh, Mohsen},
  journal={arXiv preprint arXiv:2106.10524},
  year={2021}
}
```
