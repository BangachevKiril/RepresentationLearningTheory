This folder contains the different experiments asscoiated to the paper [Global Minimizers of Sigmoid Contrastive Loss]() by Kiril Bangachev, Iliyas Noman, Guy Bresler, and Yury Polyanskiy. It performs rigorus analysis of using sigmoid constrastive loss *with trainable inverse temperature and bias* for aligning across modalities which was first introduced in [The SigLIP paper by Google DeepMind](https://www.computer.org/csdl/proceedings-article/iccv/2023/071800l1941/1TJfkEkV3RC) and later expanded in [SigLIP2](https://arxiv.org/abs/2502.14786). Relevant is also the theoretical analysis of [sigoid loss with fixed temperature and bias](https://proceedings.mlr.press/v238/lee24a/lee24a.pdf).
There are two classes of experiments.

I. First,we perform experiments with pre-trained siglip models, which we acquire from [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/siglip). Specifically, we embedthe ImageNet validation dataset consisting of 50,000 imagesfrom 1000 classes. All of these experiments are in the ImageNetEmbedding.ipynb notebook. We:

1. Show that the trained models actually form a constellation as the theory of our paper predicts (Figure 1 in the paper).

2. The trained models exhibit a modality gap (Figure 3 in the paper).

II. Then, we perform experiments based on synthetic data where we directly train representations using Adam instead of neural networks. The represnetations are synthetically created spherical vectors. The loss function and experimental set-upfiles are in the utils folder. The output images are in the logs folder. Each separate experiments is in a separate notebook, as follows:

1. BasicExperiment.ipynb. We simply run the optimization with traibale temperature. Figure 5 of the paper is here.

2. MoreModalities.ipynb. We run the optimization in the case of more than two modalities (4, 6, 8, 10). There is an ablation study for training with fixed temperature. Figure 8.

3. FrozenModaltyExperiment.ipynb. We run the optimization when one modality is locked.This corresponds to the set-up of aligning with a locked encoder.

4. Ablation Study.ipynb. We perform an ablation study by comparing to the case of training with a fixed large temperature.

5. BiasParamLeadsToZeroRB.ipynb. We performan experiment if training with the usual bias parameterization versus the relative bias parameterization that we introduce in our work. Figures from Appendix E.4.     

6. FixedRelativeBias.ipynb. Experiments with training using the relative bias parameterization with a fixed relative bias. 


                                                        
                                                          

                                                    

