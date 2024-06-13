# D-BADGE
Code for the paper "[D-BADGE: Decision-based Adversarial Batch Attack with Directional Gradient Estimation](https://ieeexplore.ieee.org/abstract/document/10542123)" (IEEE Access).
### Abstract
> The susceptibility of deep neural networks (DNNs) to adversarial examples has prompted an increase in the deployment of adversarial attacks.
Image-agnostic universal adversarial perturbations (UAPs) are much more threatening, but many limitations exist to implementing UAPs in real-world scenarios where only binary decisions are returned.
In this research, we propose D-BADGE, a novel method to craft universal adversarial perturbations for executing decision-
To primarily optimize perturbation by focusing on decisions, we consider the direction of these updates as the primary factor and the magnitude of updates as the secondary factor. 
First, we employ Hamming loss that measures the distance from distributions of ground truth and accumulating decisions in batches to determine the magnitude of the gradient. 
This magnitude is applied in the direction of the revised simultaneous perturbation stochastic approximation (SPSA) to update the perturbation. 
This simple yet efficient decision-based method functions similarly to a score-based attack, enabling the generation of  UAPs in real-world scenarios, and can be easily extended to targeted attacks.
Experimental validation across multiple victim models demonstrates that the D-BADGE outperforms existing attack methods, even image-specific and score-based attacks. 
In particular, our proposed method shows a superior attack success rate with less training time. 
The research also shows that D-BADGE can successfully deceive unseen victim models and accurately target specific classes.
### Architecture
![Architecture](./figures/f1.png)
### Experiments
#### Figures
![Visualization](./figures/f2.png)
![Transferability & Convergence](./figures/f3.png)
#### Benchmarks
![Targeted](./figures/f4.png)
![Others](./figures/f5.png)
### Supplementary Material
![L2 & Tables](./figures/f6.png)
![Statistical Analysis](./figures/f7.png)
![Decision Space](./figures/f8.png)


## Usage
### Setup Workspace
Clone this Git repository.
```bash
git clone "https://github.com/AIRLABkhu/D-BADGE.git"
cd "D-BADGE"

# Unzip checkpoitns. (OPTIONAL)
cd "log"
cat "log.tar.gz*" | tar xvzf -
rm "log.tar.gz*"
cd ".."
```
### Run Training
You can train your own victim model. (OPTIONAL)
```bash
# https://github.com/AIRLABkhu/D-BADGE/blob/main/train_victim.py
python train_victim.py --device cuda:{ID} --model resnet18 --tag "cifar10_resnet18"
```
Then train a new perturbation.
```bash
# https://github.com/AIRLABkhu/D-BADGE/blob/main/train_attack_spsa.py
python train_attack_spsa.py --device cuda:{ID} --checkpoint "cifar10_resnet18" --tag "_baselines/00"
```
All files related with this training will be saved in ```"log/cifar10_resnet18/_baselines/00"```. \
If you want to use Adam optimizer, try ```-c "cifar10-optim/adam"``` option.
```bash
python train_attack_spsa.py --device cuda:{ID} --checkpoint "cifar10_resnet18" -c "cifar10-optim/adam" --tag "_baselines/00"
```
There are several option you could try.
```bash
python  train_attack_spsa.py --device cuda:{ID} --checkpoint "cifar10_resnet18" --tag "_baselines/00" \
        --seed {SEED}               \  # The random seed for training.
        --benchmark                 \  # Specify to use benchmark algorithm or not. Deterministic algorithms will be applied if not specified.
        --epochs {EPOCHS}           \  # The number of epochs for training.
        --beta {BETA}               \  # A hyperpameter that denotes the standard deviation of normal step noise.
        --learning-rate {LR}        \  # (or -lr) specifies the step size.
        --batch-size {BS}           \  # The size of a batch in training phase.
        --eval-batch-size {EBS}     \  # The size of a batch in evaluation phase.
        --accumulation {ACCM}       \  # How many batches to use to update the perturbation once.
        --max-iters {ITERS}         \  # The number of batches to train in one epoch.
        --sliding-window-batch      \  # (or -swb). If you enable this flag, accumulated batches will be reused, dropping the first one.
        --augmentations {AUG...}    \  # Augmentation recipe filenames.
        --target {TARGET}           \  # The target class of an attack.
        --budget {BUDGET}           \  # The maximum permitted magnitude of a perturbation.
        --regulation {REG}          \  # Which `l` to use for `p_l`-norm in regulation.
        --eval-step-size {ESS}      \  # The period of evalation epoch.
        --loss-func {LOSS}          \  # The name of a loss function.
        --use-logits                   # If you enable this flag, the perturbation will be trained using the scores, not the decisions.
```
You can also try `train_attack.py` with RGF and `train_attack_nes.py` with NES optimization.

## Citation
If you use this code in your paper, please consider citing this ```BibTeX``` entry.
```bibtex
@ARTICLE{yu-2024-dbadge,
  author={Yu, Geunhyeok and Jeon, Minwoo and Hwang, Hyoseok},
  journal={IEEE Access}, 
  title={D-BADGE: Decision-Based Adversarial Batch Attack With Directional Gradient Estimation}, 
  year={2024},
  volume={12},
  number={},
  pages={80770-80780},
  keywords={Perturbation methods;Closed box;Transformers;Estimation;Vectors;Measurement;Hamming distances;Artificial neural networks;Adversarial machine learning;Image classification;Representation learning;Deep neural networks;universal decision-based adversarial attack;image classification;representation learning;vulnerability;zeroth-order optimization},
  doi={10.1109/ACCESS.2024.3407097}}
}
```
