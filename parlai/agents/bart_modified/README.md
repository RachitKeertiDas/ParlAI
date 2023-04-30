BART-Modified README

This README assumes that you have already setup parlAi and HAdes in the same environment.

A modification of BART to curb hallucination.

1) Run feature_clf.py in this  via `python3 feature_clf.py`
2) Now, train the model using the following command:
```
parlai train_model -m bart_modified -mf zoo:bart/bart_large_modified/model -t wizard_of_wikipedia --batchsize 16 --fp16 True --gradient-clip 0.1 --label-truncate 128 --log-every-n-secs 30 --lr-scheduler reduceonplateau --lr-scheduler-patience 1 --model-parallel True --optimizer adam --text-truncate 512 --truncate 512 -lr 1e-05 -vmm min -veps 0.25 -vme 1000 -vmt ppl -vp 5 -eps 1 --beam_size 5 --inference beam

```
3) A folder named data_collections must be present one level above the current directory with the corresponding data required for the HaDes Model.

Then you can proceed to run the parali command to train, interact and evaluate the model. However, please make sure to be in the current working directory to make sure the data is passed correctly to the feature_clf.py
