## Ideas For Model Improvements:

<u>original Paper</u>: https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf

we can use hybrid model of VAE with CNN or RNN. There are examples of hybrid model with AE here:

* https://dl.acm.org/doi/pdf/10.1145/3285029

#### Encoder:

* Denoising VAE
  * Daniel Im Jiwoong Im, Sungjin Ahn, Roland Memisevic, and Yoshua Bengio. 2017. Denoising criterion for variational auto-encoding framework
  * Rui Shu, Hung H Bui, Shengjia Zhao, Mykel J Kochenderfer, and Stefano Ermon. 2018. Amortized inference regularization
* Mult VAE - multinomial likelihood for data distribution. different z for each user
  * Dawen Liang, Rahul G Krishnan, Matthew D Hoffman, and Tony Jebara. 2018. Variational autoencoders for collaborative filtering
* Partial VAE
  * http://bayesiandeeplearning.org/2018/papers/75.pdf

#### Decoder:

#### Quality Measurements: 

-  NDCG@k (try different k)

- Recall@k  (try different k)

- we need to keep the measurements from the original artical (with attention for comparison)
