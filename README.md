# DUMN:Deep User Match Network for Click Through Rate Prediction

花了几天时间复现了一下阿里的一篇论文：https://dl.acm.org/doi/abs/10.1145/3404835.3463078

文章为SIGIR21的一篇短文，主要考虑的是通常我们在做ctr等预估的时候只考虑item之间的关系，比如用户的历史点击序列和目标item之间的关系，
从而反映用户的兴趣。但是忽略了用户之间的关系，本文主要就是构建了user-to-user的模块DUMN来做用户匹配。


数据和特征可以换成自己的，代码中在这部分都做了删减

