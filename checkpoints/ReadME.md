Model Checkpoints:
* MobilenetV2 based text segmentation is listed above. 

* Xecption based text segmentation: [Here](https://drive.google.com/open?id=1iiWjf-PKBq_nfD9lqKtuyU4bqEbUaq_5). The model is trained with SGD with nesterov. Weight decay is 1e-3. Learning rate use cyclical learning rate, ranging from 1e-4 to 4e-4. 5e-5 seems too small, and 1e-3 will diverge the model. 
 