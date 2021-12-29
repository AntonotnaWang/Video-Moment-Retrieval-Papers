# Video-Moment-Retrieval-Papers
Summary of the papers related to video moment retrieval / video grounding / video moment localization ...

* This stream is to identify the clips related to a given sentence in a given video.

* [review blog1](https://blog.csdn.net/qq_39388410/article/details/107316185)

* [review blog2](https://zhuanlan.zhihu.com/p/101555506)

* [review bolg3](https://zhuanlan.zhihu.com/p/67002279?from_voters_page=true)

# 2017
The beginning of this stream

* - [TALL Temporal Activity Localization via Language Query](http://openaccess.thecvf.com/content_iccv_2017/html/Gao_TALL_Temporal_Activity_ICCV_2017_paper.html) `ICCV 2017`.

点出该问题需要解决的两大困难：1作为跨模态任务，如何得到适当的文本和视频表示特征，以允许跨模态匹配操作和完成语言查询。2理论上可以生成无限粒度的视频片段，然后逐一比较。但时间消耗过大，那么如何能够从有限粒度的滑动窗口做到准确的具体帧定位。

特征抽取--使用全局的sentence2vec和C3D; 多模态交互。

![TALL / CTRL](figs/TALL.png)

* - [Localizing Moments in Video with Natural Language](https://openaccess.thecvf.com/content_iccv_2017/html/Hendricks_Localizing_Moments_in_ICCV_2017_paper.html) `ICCV2017`.

特征抽取--使用全局的LSTM和VGG; 无多模态交互。

![MCN](figs/MCN.png)

# 2018

* - [Cross-modal Moment Localization in Videos](https://dl.acm.org/doi/abs/10.1145/3240508.3240549) `MM 2018`.

* - [Attentive Moment Retrieval in Videos](https://dl.acm.org/doi/abs/10.1145/3209978.3210003?casa_token=ZxROouILgG4AAAAA:_xPwkaKYbwslcQnqEbrj7c4oZImYqDIVbtD0U4DJeHpKNzTknTd5qu4w5wK8NoVClmebtXU3InjS) `SIGIR 18`.

# 2021

* - [Multi-Modal Relational Graph for Cross-Modal Video Moment Retrieval]() `CVPR 2021`

* - [LoGAN: Latent Graph Co-Attention Network for Weakly-Supervised Video Moment Retrieval]() `WACV 2021` 

* - [A Closer Look at Temporal Sentence Grounding in Videos Datasets and Metrics]() `arxiv 2021`

* - [Interventional Video Grounding with Dual Contrastive Learning]() `CVPR 2021`

* - [Video Corpus Moment Retrieval with Contrastive Learning]() `SIGIR 2021`

* - [Deconfounded Video Moment Retrieval with Causal Intervention]() `SIGIR2021`

* - [Cross Interaction Network for Natural Language Guided Video Moment Retrieval]() `SIGIR 2021`

* - [mTVR: Multilingual Moment Retrieval in Videos]() `ACL 2021`

* - [Fast Video Moment Retrieval]() `ICCV 2021`

* - [CONQUER: Contextual Query-aware Ranking for Video Corpus Moment Retrieval]() `MM2021`

* - [Interaction-Integrated Network for Natural Language Moment Localization]() `TIP 2021`

* - [Boundary Proposal Network for Two-Stage Natural Language Video Localization]() `AAAI 2021`

* - [Context-Aware Biaffine Localizing Network for Temporal Sentence Grounding]() `CVPR 2021`

* - [Thinking Fast and Slow: Efficient Text-to-Visual Retrieval With Transformers]() `CVPR 2021`

* - [Fast Video Moment Retrieval]() `ICCV 2021`