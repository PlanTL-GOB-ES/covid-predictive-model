# Covid Predictive Model

A PyTorch Recurrent Neural Network implementation for Covid-19 fatality prediction.

## Characteristics of the RNN

- Daily predictions.
- Attention (static wrt. dynamic(all events),
dynamic(current event) wrt. dynamic(all events) and both mechanisms).
- Custom Loss function.

## RNN Architecture
![alt text](https://raw.githubusercontent.com/PlanTL-SANIDAD/covid-predictive-model/main/documentation/images/rnn.png "RNN Architecture")

## Code Structure
```
+-+--------------------+
| |                    |
| +-> baselines/       |
| |                    |
| +-> src/             |
| |                    |
| +-> README.md        |
| |                    |
| +-> requirements.txt |
| |                    |
| +-> setup.sh         |
| |                    |
| +-> .gitignore       |
+----------------------+
```

- baselines: Folder containing code of both baselines.
- src: Folder containing the code for all the operations involved.
- README.md: Instructions document.
- requirements.txt: Required libraries.
- setup.sh: Script that runs the virtual environment generation and installs all
 required libraries.
 - .gitignore: File specifying which files not to version.

## Setup
In order to start running any of the scripts it is important to create the virtual environment
by typing the following command in the CLI.

Linux:
```
$ ./setup.sh
```

Windows:
Create a virtual environment and run:
```
> pip install -r requirements.txt
```

 
### Train-Evaluate a Model
Script src/training/train.py, as explained above, performs the whole training-evaluation
pipeline.

The basic way of running the training script is the following one:
```
(virtual-env) $ python3 ./training/train.py <experiment-name> <data-path>
```

For more running options try
```
(virtual-env) $ python3 ./training/train.py --help
```
 
### Perform a Grid Search
Script src/training/train.py also performs the Grid Search. A file containing an specification
for the Grid Search parameters has to be included when running the script:

```
(virtual-env) $ python3 ./training/train.py <experiment-name> <path-to-data> --param-search-config \
<path-to-config>.json
```

The format of the Grid Search parameter specification should have the following structure (JSON):
```
{
<field>: [<value>, <value>, ...]
}
```
where \<field\> is one of the parameters present at the training script as input parameter.
For instance \<field\> could be "dropout" and the values array could be 0.0, 0.1 and 0.2. 
 
In order to perform a Grid Search with multiple computing nodes, Grid Search parameter configuration can be
split with the following command:
```
(virtual-env) $ python3 ./training/split.py <path-to-parameter-specification> \
<output_folder> --nsplits <N>
```

## Explainability
![alt text](https://raw.githubusercontent.com/PlanTL-SANIDAD/covid-predictive-model/main/documentation/images/patient_dynamic_attention.gif "Patient Dynamic Attention")

## Paper
Cite our pre-print:
```
@article {Villegas2020.12.22.20244061,
	author = {Villegas, Marta and Gonzalez-Agirre, Aitor and Guti{\'e}rrez-Fandi{\~n}o, Asier and Armengol-Estap{\'e}, Jordi and Carrino, Casimiro Pio and Fern{\'a}ndez, David P{\'e}rez and Soares, Felipe and Serrano, Pablo and Pedrera, Miguel and Garc{\'\i}a, Noelia and Valencia, Alfonso},
	title = {Predicting the Evolution of COVID-19 Mortality Risk: a Recurrent Neural Network Approach},
	elocation-id = {2020.12.22.20244061},
	year = {2020},
	doi = {10.1101/2020.12.22.20244061},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2020/12/23/2020.12.22.20244061},
	eprint = {https://www.medrxiv.org/content/early/2020/12/23/2020.12.22.20244061.full.pdf},
	journal = {medRxiv}
}

```

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2020 Secretaría de Estado de Digitalización e Inteligencia Artificial (SEDIA)

## Disclaimer

The models published in this repository are intended for a generalist purpose and are available to third parties. These models may have bias and/or any other undesirable distortions.

When third parties, deploy or provide systems and/or services to other parties using any of these models (or using systems based on these models) or become users of the models, they should note that it is their responsibility to mitigate the risks arising from their use and, in any event, to comply with applicable regulations, including regulations regarding the use of artificial intelligence.

In no event shall the owner of the models (SEDIA – State Secretariat for digitalization and artificial intelligence) nor the creator (BSC – Barcelona Supercomputing Center) be liable for any results arising from the use made by third parties of these models.


Los modelos publicados en este repositorio tienen una finalidad generalista y están a disposición de terceros. Estos modelos pueden tener sesgos y/u otro tipo de distorsiones indeseables.

Cuando terceros desplieguen o proporcionen sistemas y/o servicios a otras partes usando alguno de estos modelos (o utilizando sistemas basados en estos modelos) o se conviertan en usuarios de los modelos, deben tener en cuenta que es su responsabilidad mitigar los riesgos derivados de su uso y, en todo caso, cumplir con la normativa aplicable, incluyendo la normativa en materia de uso de inteligencia artificial.

En ningún caso el propietario de los modelos (SEDIA – Secretaría de Estado de Digitalización e Inteligencia Artificial) ni el creador (BSC – Barcelona Supercomputing Center) serán responsables de los resultados derivados del uso que hagan terceros de estos modelos.
