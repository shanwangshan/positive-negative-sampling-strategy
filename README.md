# Positive-negative-sampling-strategy

This git repo details the implementation of paper - "Positive and negative sampling strategies for self-supervised learning on audio-video data". We have separate folders for each method for simplicity and clarity.

## Clone the repo

```console
git clone https://github.com/shanwangshan/positive-negative-sampling-strategy.git
```
## Set up the same conda environemnt

```console
conda env create -f environment.yml
```

## Examples of running the code
Since we have different folders for different sampling methods, here we show one example only.

### Proposed easy-negative sampling technique
```console
cd soft-positive
```
#### Pretext task - Audio video correspondence (AVC)
One needs to have the vggsound dataset ready, then run
```console
sbatch gpu_train.sh
```
The checkpoint is saved at checkpoint folder, one can download the pre-trained weights here. The training log is saved in err_out folder.

#### Test on downstream tasks

- In-domain tasks on vggsound, for linear evaluation, one need to comment out other lines in gpu_train.sh except

	```console
	python train.py config.yaml --model_type 'audio' --lin_prob
	```
	Then, run
	```console
	sbatch gpu_train.sh
	```

   for finetune, one need to comment out other lines in gpu_train.sh except

	```console
	python train.py config.yaml --model_type 'audio'
	```
	Then, run
	```console
	sbatch gpu_train.sh
	```

   The pretrained weights of linear evaluation and finetune are saved in folder audio_model_lin and audio_model_ft, respectively. You can download them here.


	For testing on vggsound test data using linear evaluation,  one need to comment out other lines in gpu_test.sh except

	```console
	python test_batch.py config.yaml --model_type 'audio' --lin_prob
	```
	Then, run
	```console
	sbatch gpu_test.sh
	```
	The test outputs are saved in folder err_out.

- Other two out-of-domain downstream tasks are seen from folder test_TAU, test_fsd.
