# crossed-effects

Reference implementation for scalable Bayesian crossed effects models.
Algorithms are documented in the paper at https://arxiv.org/abs/2103.10875.

    .
    ├── demos   # Demonstration notebooks for a selection of models
    ├── paper   # Materials pertaining to the academic paper
    ├── tests   # Test root
    └── xfx     # Code root


## Instructions for Ubuntu/OSX

0. Install the generic dependencies `Python 3.11`, `uv`, `git`.

1. Define your project root (`[project]`) and navigate there:

    ```shell
    mkdir [project]
    cd [project]
    ```

2. Clone the repository:

    ```shell
    git clone https://github.com/timsf/crossed-effects.git
    ```

3. Start the `Jupyter` server:

    ```shell
    uv run jupyter notebook
    ```

4. Access the `Jupyter` server in your browser and navigate to the notebook of interest.


## Reference

    @article{papaspiliopoulos2023scalable,
        title={Scalable Bayesian computation for crossed and nested hierarchical models},
        author={Papaspiliopoulos, Omiros and Stumpf-F{\'e}tizon, Timoth{\'e}e and Zanella, Giacomo},
        journal={Electronic Journal of Statistics},
        volume={17},
        number={2},
        pages={3575--3612},
        year={2023},
        publisher={The Institute of Mathematical Statistics and the Bernoulli Society}
    }