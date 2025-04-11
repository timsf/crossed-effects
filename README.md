# crossed-effects

Reference implementation for scalable Bayesian crossed effects models.
Algorithms are documented in the paper at https://arxiv.org/abs/2103.10875.

    .
    ├── demos   # Demonstration notebooks for a selection of models
    ├── paper   # Materials pertaining to the academic paper
    ├── tests   # Test root
    └── xfx     # Code root


## Instructions for Ubuntu/OSX

0. Install the generic dependencies: `Python 3.10`, `pip`, `poetry`, `git`:

1. Define your project root (`[project]`) and navigate there:

    ```shell
    mkdir [project]
    cd [project]
    ```

2. Clone the repository:

    ```shell
    git clone https://github.com/timsf/crossed-effects.git
    ```

3. Create a virtual environment and install dependencies:

    ```shell
    poetry install
    ```

4. Start a shell within the virtual environment:

    ```shell
    poetry shell
    ```

5. Start the `Jupyter` server:

    ```shell
    jupyter notebook
    ```

6. Access the `Jupyter` server in your browser and navigate to the notebook of interest.


## Reference
@article{Papaspiliopoulos_2023,
   title={Scalable Bayesian computation for crossed and nested hierarchical models},
   volume={17},
   ISSN={1935-7524},
   url={http://dx.doi.org/10.1214/23-EJS2172},
   DOI={10.1214/23-ejs2172},
   number={2},
   journal={Electronic Journal of Statistics},
   publisher={Institute of Mathematical Statistics},
   author={Papaspiliopoulos, Omiros and Stumpf-Fétizon, Timothée and Zanella, Giacomo},
   year={2023},
   month=jan }