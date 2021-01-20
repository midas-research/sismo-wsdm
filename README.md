# Towards Ordinal Suicide Ideation Detection on Social Media

This codebase contains the python scripts for SISMO, the base model for Towards Ordinal Suicide Ideation Detection on Social Media.

WSDM '21 paper [coming soon](#)

## Environment & Installation Steps

Python 3.6 & Pytorch 1.5

## Run

Execute the following steps in the same environment:

```bash
cd sismo-wsdm & python main.py
```

## Command Line Arguments

To run different variants of SISMO, perform ablation or tune hyperparameters, the following command-line arguments may be used:

```
usage: main.py [-h] [--expt-type {4,5}] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
               [--num-runs NUM_RUNS] [--early-stop EARLY_STOP] [--hidden-dim HIDDEN_DIM]
               [--embed-dim EMBED_DIM] [--num-layers NUM_LAYERS] [--dropout DROPOUT]
               [--learning-rate LEARNING_RATE] [--scale SCALE]
               [--embedding-type {bert,distil,roberta,longformer,xlnet}] [--data-dir DATA_DIR]
               [--model-type {avg-pool,lstm+att,lstm}]

optional arguments:
  -h, --help            show this help message and exit
  --expt-type {4,5}     expt type (default: 5)
  --batch-size BATCH_SIZE
                        batch size (default: 8)
  --epochs EPOCHS       number of epochs (default: 50)
  --num-runs NUM_RUNS   number of runs (default: 50)
  --early-stop EARLY_STOP
                        early stop limit (default: 10)
  --hidden-dim HIDDEN_DIM
                        hidden dimensions (default: 512)
  --embed-dim EMBED_DIM
                        embedding dimensions (default: 768)
  --num-layers NUM_LAYERS
                        number of layers (default: 1)
  --dropout DROPOUT     dropout probablity (default: 0.3)
  --learning-rate LEARNING_RATE
                        learning rate (default: 0.01)
  --scale SCALE         scale factor alpha (default: 1.8)
  --embedding-type {bert,distil,roberta,longformer,xlnet}
                        type of embedding (default: longformer)
  --data-dir DATA_DIR   directory for data (default: )
  --model-type {avg-pool,lstm+att,lstm}
                        type of model (default: lstm+att)
```

## Dataset

We use the dataset released by [1] that consists of reddit posts of 500 users across 9 mental health and suicide related subreddits.

https://github.com/AmanuelF/Suicide-Risk-Assessment-using-Reddit

Processed dataset format should be a DataFrame as a .pkl file having the following columns:

1. label : 0, 1, ... 4 denoting the risk level of the user.
2. enc : list of lists consisting of 768-dimensional encoding for each post. (SISMO uses longformer embeddings [2])

We provide longformer embeddings (reddit-longformer.pkl)

## Ethical Considerations

We work within the purview of acceptable privacy practices suggested by [3] and considerations discussed by [4] to avoid coercion and intrusive treatment.
For the dataset [1] used in this research, the original Reddit data is publicly available.
Our work focuses on developing a neural model for screening users and does not make any diagnostic claims related to suicide.
We study Reddit posts in a purely observational capacity, and do not intervene with user experience.
The assessments made by SISMO are sensitive and should be shared selectively and subject to IRB approval to avoid misuse like Samaritan’s Radar [6].

## Cite

If our work was helpful in your research, please kindly cite this work:

```
@inproceedings{sawhney2021ordinal,
    author={Sawhney, Ramit  and
            Joshi, Harshit  and
            Gandhi, Saumya  and
            Shah, Rajiv Ratn},
    title = {Towards Ordinal Suicide Ideation Detectionon Social Media},
    year = {2021},
    month=mar,
    booktitle = {Proceedings of 14th ACM International Conference On Web Search And Data Mining},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    keywords = {social media, suicide ideation, ordinal regression, reddit},
    location = {Virtual Event, Israel},
    series = {WSDM '21}
}
```

### References

[1] Gaur, M., Alambo, A., Sain, J. P., Kursuncu, U., Thirunarayan, K., Kavuluru, R., ... & Pathak, J. (2019, May). Knowledge-aware assessment of severity of suicide risk for early intervention. In The World Wide Web Conference (pp. 514-525).

[2] Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

[3] Chancellor, S., Birnbaum, M. L., Caine, E. D., Silenzio, V. M., & De Choudhury, M. (2019, January). A taxonomy of ethical tensions in inferring mental health states from social media. In Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 79-88).

[4] Fiesler, C., & Proferes, N. (2018). “Participant” perceptions of Twitter research ethics. Social Media+ Society, 4(1), 2056305118763366.

[5] Zirikly, A., Resnik, P., Uzuner, O., & Hollingshead, K. (2019, June). CLPsych 2019 shared task: Predicting the degree of suicide risk in Reddit posts. In Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology (pp. 24-33).

[6] Hsin, H., Torous, J., & Roberts, L. (2016). An adjuvant role for mobile health in psychiatry. JAMA psychiatry, 73(2), 103-104.
