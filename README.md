# FCF
Fortified Concept Forgetting for Text-to-Image Diffusion Models by Machine Unlearning on CLIP

## Requirements
A environment named ``` fcf ``` can be created and activated with:
```
conda env create -f environment.yaml
conda activate fcf
```
## Training
To train the model. you would need a csv file with ```prompt_f```, ```prompt_n``` and ```prompt_r```.(Sample data in ```data/train/```)

To choose the type of the model, you would need to adjust the value of ```eta``` in ```concept_forgetting_train.py```, ```eta_experience``` in ```features_forgetting_E.py``` and ```eta_clean``` in ```features_forgetting_P.py```
### Explicit Concept Forgetting Process
```
python concept_forgetting_train.py --input_prompts 'prompts_path' --save_path 'saved_model_path'
```
### Implicit Concept Forgetting Process
Using the Projection Feature Forgetting Method:
```
python features_forgetting_P.py --model_path 'your_trained_model_path'
```
After training, the model will be saved in the original ```model path```.

Using the Empirical Feature Forgetting Method:
```
python features_forgetting_E.py --model_path 'your_trained_model_path' --experienxe_path 'your_experience_path'
```
After training, the model will be saved in the original ```model path```.

## Generating
To generate images, you would need a csv file with ```prompt```, ```evaluation_seed``` and ```case_number```. (Sample data in ```data/```)
```
python fcf-generate.py --model_name 'name of the model to load' --prompts_path 'path for the csv file with prompts and corresponding seeds.' --save_path 'save directory for images.'
```
```--ddim_steps```: number of denoising steps. Default: 50.

```--num_samples```: number of samples generated per prompt. Default: 1.

```--from_case```: The starting offset in csv to generate images. Default: 0

## Citation
If you like or use our work please cite us:
```
TBD...
```
