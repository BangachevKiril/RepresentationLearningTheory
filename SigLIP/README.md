Experiments for trainable bias and temperature using the [Sigmoid Loss for Language-Image pretraining](https://www.computer.org/csdl/proceedings-article/iccv/2023/071800l1941/1TJfkEkV3RC).
                                                          
The experiments follow the [upcoming paper](). In the paper, we describe the geometry of zero-loss solutions to SigLIP and test properties of it:

1. We perform experiments with real data using the [image-text model of Google Research](https://github.com/google-research/big_vision/tree/main/big_vision/models/proj/image_text) trained with SigLIP.
2. Our theoretical results suggest several modifications and reparamterizations depending on the concrete task of interest. These include:
   
   A) Extension to more modalities
   
   B) Extension to a forzen text/image embedding
   
   C) Extension to a desired temperature/bias tradeoff
   
4. We perform an ablation study on why zero-loss configurations need trainable temperature rather than a fixed low temperature

                                                        
                                                          

                                                    

