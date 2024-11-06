# FAKE NEWS DETECTION



### Table of Contents

- [Survey](#Survey)
- [Text-based Detection](#Text-based Detection)
- [Multi-modal Detection](#multi-modal-detection) 
- [Using Social Context](#using-social-context)
- [Fact-checking](#fact-checking)
- [Datasets](#Datasets)

Markdown format:

```markdown
- Paper Name. 
  [[link]](link) 
  [[code]](link).
  - Author 1, Author 2, Author 3. *Conference/Journal*, Year.
```



### Survey

- The Future of False Information Detection on Social Media: New Perspectives and Trends. [[link]](https://dl.acm.org/doi/10.1145/3393880)
  - BIN GUO, YASAN DING. ACM Computing Surveys, 2020.

- Fighting False Information from Propagation Process: A Survey. [[link]](https://dl.acm.org/doi/10.1145/3563388)
  - LING SUN, YUAN RAO. ACM Computing Surveys, 2023.

- 基于传播意图特征的虚假新闻检测方法综述.
  - 毛震东, 赵博文. 信号处理. 2022.

- 基于事实信息核查的虚假新闻检测综述.
  - 杨昱洲, 周杨铭, 应祺超. 中国传媒大学学报. 2023.

- Detecting and Mitigating the Dissemination of Fake  News: Challenges and Future  Research Opportunities. [[link]](https://ieeexplore.ieee.org/document/9789171)
  - Wajiha Shahid, Bahman Jamshidi, Saqib Hakak. IEEE Transactions on Computational Social Systems, 2022
- Decoding the AI Pen: Techniques and Challenges in Detecting AI-Generated Text. [[link]](https://dl.acm.org/doi/10.1145/3637528.3671463) 
  - Sara Abdali, Richard Anarfi, CJ Barberan, Jia He. *KDD*, 2024.





### Text-based Detection

- Memory-Guided Multi-View Multi-Domain  Fake News Detection. [[link]](https://ieeexplore.ieee.org/abstract/document/9802916) [[code]](https://github.com/ICTMCG/M3FEND)
  - Yongchun Zhu, Qiang Sheng, Juan Cao. IEEE Transactions on Knowledge and Data Engineering, 2020.

- Bad Actor, Good Advisor:  Exploring the Role of Large Language Models in Fake News Detection. [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/30214) [[code]](https://github.com/ictmcg/arg)
  - Beizhe Hu, Qiang Sheng, Juan Cao. *AAAI*, 2024
- An Integrated Multi-Task Model  for Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/9339883/) 
  - Qing Liao, Hao Han, Xuan Wang. IEEE Transactions on Knowledge and Data Engineering, 2021.
- TELLER: A Trustworthy Framework For Explainable, Generalizable and  Controllable Fake News Detection. [[link]](https://aclanthology.org/2024.findings-acl.919) [[code]](https://github.com/less-and-less-bugs/Trust_TELLER)
  - Hui Liu, Wenya Wang, Haoru Li. *ACL*, 2024
  - Explainable Detection
- Weak Supervision for Fake News Detection via Reinforcement Learning. [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/5389) [[code]](https://github.com/yaqingwang/WeFEND-AAAI20)
  - Yaqing Wang, Weifeng Yang, Fenglong Ma. *AAAI*, 2020
- MSynFD: Multi-hop Syntax Aware Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3589334.3645468) 
  - Liang Xiao, Qi Zhang, Chongyang Shi. *WWW*, 2024
- Mixed Graph Neural Network-Based Fake News  Detection for Sustainable Vehicular  Social Networks. [[link]](https://ieeexplore.ieee.org/document/9819856/)
  - Zhiwei Guo, Keping Yu, Alireza Jolfaei. IEEE Transactions on Intelligent Transportation Systems. 2022
- Fake Review Detection Using Deep Neural  Networks with Multimodal Feature Fusion  Method. [[link]](https://ieeexplore.ieee.org/document/10476070) 
  - Xin Li, Lirong Chen. *ICPADS*, 2023
- On Fake News Detection with LLM Enhanced Semantics Mining. [[link]](https://openreview.net/forum?id=Fd2mmQKFKW) 
  - Xiaoxiao Ma, Yuchen Zhang, Kaize Ding. *EMNLP*, 2024

- Decoding Susceptibility: Modeling Misbelief to Misinformation Through a Computational Approach. [[link]](https://arxiv.org/abs/2311.09630) 
  - Yanchen Liu, Mingyu Derek Ma, Diyi Yang. *EMNLP*, 2024

- Fake News in Sheep’s Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks. [[link]](https://dl.acm.org/doi/10.1145/3637528.3671977) 
  - Jiaying Wu, Jiafeng Guo, Bryan Hooi. *KDD*, 2024




### Multi-modal Detection

- Bootstrapping Multi-view Representations for Fake News Detection. [[link]](https://arxiv.org/abs/2206.05741) [[code]](https://github.com/yingqichao/fnd-bootstrap)
  - Q Ying, X Hu, Y Zhou, Z Qian, D Zeng. *AAAI*, 2023.
- Cross-modal Ambiguity Learning for Multimodal Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3485447.3511968) [[code]](https://github.com/cyxanna/CAFE)
  - Y Chen, D Li, P Zhang, J Sui, Q Lv, L Tun. *WWW*, 2022.
- Causal Inference for Leveraging Image-Text  Matching Bias in Multi-Modal Fake News  Detection. [[link]](https://ieeexplore.ieee.org/document/9996587)
  - Linmei Hu, Ziwei Chen, Ziwang Zhao. IEEE Transactions on Knowledge and Data Engineering, 2022.
- MVAE: Multimodal Variational Autoencoder for Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3308558.3313552) [[code]](https://github.com/dhruvkhattar/MVAE)
  - D Khattar, JS Goud, M Gupta, V Varma. *WWW*, 2019.
- Hierarchical Multi-modal Contextual Attention Network for Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3404835.3462871) [[code]](https://github.com/wangjinguang502/HMCAN)
  - S Qian, J Wang, J Hu, Q Fang, C Xu. *SIGIR*, 2021.
- Unraveling the Tangle of Disinformation: A Multimodal Approach for Fake News Identification on Social Media. [[link]](https://dl.acm.org/doi/10.1145/3589335.3651972)
  - Junaid Rashid, Jungeun Kim, Anum Masood. *WWW*, 2024
- Detecting and Grounding Multi-Modal Media  Manipulation and Beyond. [[link]](https://ieeexplore.ieee.org/document/10440475/) [[code]](https://github.com/rshaojimmy/MultiModal-DeepFake)
  - Rui Shao, Tianxing Wu, Jianlong Wu. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024
- Unsupervised Domain-Agnostic Fake News  Detection Using Multi-Modal Weak Signals. [[link]](https://ieeexplore.ieee.org/document/10517660)
  - Amila Silva, Ling Luo, Shanika Karunasekera. IEEE Transactions on Knowledge and Data Engineering, 2024.
- Leveraging Intra and Inter Modality Relationship for Multimodal Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3487553.3524650) [[code]](https://github.com/shiivangii/Leveraging-Intra-and-Inter-Modality-Relationship-for-Multimodal-Fake-News-Detection)
  - S Singhal, T Pandey, S Mrig, RR Shah. *WWW*, 2022.
- Improving Generalization for Multimodal Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3591106.3592230) [[code]](https://github.com/TIBHannover/MM-FakeNews-Detection)
  - S Tahmasebi, S Hakimov, R Ewerth. *ICMR*, 2023.
- Fake News Detection via Multi-scale Semantic Alignment and Cross-modal Attention. [[link]](https://dl.acm.org/doi/10.1145/3626772.3657905) [[code]](https://github.com/kingdommm/MSACA_pytorch)
  - Jiandong Wang, Hongguang Zhang, Chun Liu. *SIGIR*, 2024
- Cross-modal Contrastive Learning for Multimodal Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3581783.3613850) [[code]](https://github.com/wishever/COOLANT)
  - Longzheng Wang, Chuang Zhang, Hongbo Xu. *ACM MM*, 2023.
- Intra and Inter-modality Incongruity Modeling and Adversarial Contrastive Learning for Multimodal Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3652583.3658118) 
  - Siqi Wei, Bin Wu. *ICMR*, 2024
- Human Cognition-Based Consistency Inference  Networks for Multi-Modal Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/10138033) 
  - Lianwei Wu, Pusheng Liu, Yongqiang Zhao. IEEE Transactions on Knowledge and Data Engineering, 2023.
- Multimodal Fusion with Co-Attention Networks for Fake News Detection. [[link]](https://aclanthology.org/2021.findings-acl.226) [[code]](https://github.com/wuyang45/MCAN_code)
  - Y Wu, P Zhan, Y Zhang, L Wang. *ACL*, 2021.
- Hierarchical Semantic Enhancement Network for Multimodal Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3581783.3612423)
  - Qiang Zhang, Jiawei Liu, Fanrui Zhang. *ACM MM*, 2023
- Multi-modal Fake News Detection on Social Media via Multi-grained Information Fusion. [[link]](https://dl.acm.org/doi/10.1145/3591106.3592271) [[code]](https://github.com/ml-master/MMFN_yyt)
  - Y Zhou, Y Yang, Q Ying, Z Qian, X Zhang. *ICMR*, 2023.
- Knowledge Enhanced Vision and Language Model  for Multi-Modal Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/10505027) 
  - Xingyu Gao, Xi Wang, Zhenyu Chen. IEEE Transactions on Multimedia, 2024
- Multimodal fake news detection via progressive fusion networks. [[link]](https://www.sciencedirect.com/science/article/pii/S0306457322002217) 
  - J Jing, H Wu, J Sun, X Fang, H Zhang. Information Processing & Management, 2023.
- Positive Unlabeled Fake News Detection via  Multi-Modal Masked Transformer Network. [[link]](https://ieeexplore.ieee.org/document/10089519) 
  - Jinguang Wang, Shengsheng Qian, Jun Hu. IEEE Transactions on Multimedia, 2024.
- Modeling Both Intra- and Inter-Modality Uncertainty  for Multimodal Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/10261246)
  - Lingwei Wei, Dou Hu, Wei Zhou. IEEE Transactions on Multimedia, 2023
- Multimodal Fake News Detection via CLIP-Guided  Learning. [[link]](https://ieeexplore.ieee.org/document/10219997) 
  - Yangming Zhou, Yuzhou Yang, Qichao Ying. *ICME*, 2023.
- Learning Frequency-Aware Cross-Modal Interaction  for Multimodal Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/10586835) 
  - Yan Bai, Yanfeng Liu, Yongjun Li. IEEE Transactions on Computational Social Systems, 2024.
- MAFE: Multi-modal Alignment via Mutual  Information Maximum Perspective in Multi-modal  Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/10580548)
  - Haimei Qin, Yaqi Jing. *CSCWD*, 2024.
- Modality and Event Adversarial Networks for  Multi-Modal Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/9794602)
  - Pengfei Wei, Fei Wu, Ying Sun. IEEE Signal Processing Letters. 2022.
- FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms. [[link]](https://ojs.aaai.org/index.php/AAAI/article/view/26689) [[code]](https://github.com/ICTMCG/FakeSV)
  - Peng Qi, Yuyan Bu, Juan Cao. *AAAI*, 2023. 
- “Image, Tell me your story!” Predicting the original meta-context of visual misinformation. [[link]](https://www.arxiv.org/abs/2408.09939) [[code]](https://github.com/UKPLab/5pils)
  - Jonathan Tonglet, Marie-Francine Moens, Iryna Gurevych. *EMNLP*, 2024

- FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process. [[link]](https://dl.acm.org/doi/10.1145/3664647.3680663) [[code]](https://github.com/ICTMCG/FakingRecipe)
  - Yuyan Bu, Qiang Sheng, Juan Cao, Peng Qi. *ACM MM*, 2024

- Mitigating World Biases: A Multimodal Multi-View Debiasing Framework for Fake News Video Detection. [[link]](https://dl.acm.org/doi/10.1145/3664647.3681673) 
  - Zhi Zeng, Minnan Luo, Xiangzheng Kong. *ACM MM*, 2024

- FKA-Owl: Advancing Multimodal Fake News Detection through Knowledge-Augmented LVLMs. [[link]](https://dl.acm.org/doi/10.1145/3664647.3681089) [[code]](https://github.com/liuxuannan/FAK-Owl)
  - Xuannan Liu, Peipei Li, Huaibo Huang. *ACM MM*, 2024

- Harmfully Manipulated Images Matter in Multimodal Misinformation Detection. [[link]](https://dl.acm.org/doi/10.1145/3664647.3681322) 
  - Bing Wang, Shengsheng Wang, Changchun Li. *ACM MM*, 2024

- MMDFND: Multi-modal Multi-Domain Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3664647.3681317) 
  - Yu Tong, Weihai Lu, Zhe Zhao. *ACM MM*, 2024

- Vaccine Misinformation Detection in X using Cooperative Multimodal Framework. [[link]](https://dl.acm.org/doi/10.1145/3664647.3681422)
  - Usman Naseem, Adam Dunn, Matloob Khushi, Jinman Kim. *ACM MM*, 2024




### Using Social Context

- MFAN: Multi-modal Feature-enhanced Attention Networks for Rumor Detection. [[link]](https://www.ijcai.org/proceedings/2022/335) [[code]](https://github.com/drivsaf/MFAN)
  - J Zheng, X Zhang, S Guo, Q Wang, W Zang, Y Zhang. *IJCAI*, 2022.
- User Preference-aware Fake News Detection. [[link]](https://dl.acm.org/doi/abs/10.1145/3404835.3462990) [[code]](https://github.com/safe-graph/GNN-FakeNews)
  - Y Dou, K Shu, C Xia, PS Yu, L Sun. *SIGIR*, 2021.
- A Self-Attention Mechanism-Based Model for  Early Detection of Fake News. [[link]](https://ieeexplore.ieee.org/document/10286430)
  - Bahman Jamshidi, Saqib Hakak, Rongxing Lu. IEEE Transactions on Computational Social Systems, 2023.
- Explainable Detection of Fake News on Social  Media Using Pyramidal Co-Attention Network. [[link]](https://ieeexplore.ieee.org/document/9908576) 
  - Fazlullah Khan, Ryan Alturki, Gautam Srivastava. IEEE Transactions on Computational Social Systems, 2022.
- FIND: Privacy-Enhanced Federated Learning for  Intelligent Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/10225618) 
  - Zhuotao Lian, Chen Zhang, Chunhua Su. IEEE Transactions on Computational Social Systems, 2023.
- MEFaND: A Multimodel Framework for  Early Fake News Detection. [[link]](https://ieeexplore.ieee.org/document/10430208)
  - Asma Sormeily, Sajjad Dadkhah. IEEE Transactions on Computational Social Systems, 2024.
- Propagation Structure-Aware Graph Transformer for Robust and Interpretable Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3637528.3672024) [[code]](https://github.com/JYZHU03/PSGT)
  - Junyou Zhu, Chao Gao, Ze Yin, Xianghua Li, Juergen Kurths. *KDD*, 2024

-  Leveraging Exposure Networks for Detecting Fake News Sources. [[link]](https://dl.acm.org/doi/10.1145/3637528.3671539) 
  - Maor Reuben, Lisa Friedland, Rami Puzis, Nir Grinberg. *KDD*, 2024

- Mitigating Social Hazards: Early Detection of Fake News via Diffusion-Guided Propagation Path Generation. [[link]](https://dl.acm.org/doi/10.1145/3664647.3681087) 
  - Litian Zhang, Xiaoming Zhang, Chaozhuo Li. *ACM MM*, 2024






### Fact-checking

- Improving Fake News Detection by Using an Entity-enhanced Framework to Fuse Diverse Multimodal Clues. [[link]](https://dl.acm.org/doi/10.1145/3474085.3481548) 
  - P Qi, J Cao, X Li, H Liu, Q Sheng, X Mi, Q He. *ACM MM*, 2021.
  - Multi-modal Detection
- SNIFFER: Multimodal Large Language Model  for Explainable Out-of-Context Misinformation Detection. [[link]](https://ieeexplore.ieee.org/document/10656149) [[code]](https://github.com/MischaQI/Sniffer)
  - Peng Qi, Zehong Yan, Wynne Hsu. *CVPR*, 2024.
  - Multi-modal Detection, Explainable Detection
- Do We Need Language-Specific Fact-Checking Models? The Case of Chinese. [[link]](https://arxiv.org/abs/2401.15498) 
  - Caiqi Zhang, Zhijiang Guo, Andreas Vlachos. *EMNLP*, 2024


- MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. [[link]](https://arxiv.org/abs/2404.10774) [[code]](https://github.com/Liyan06/MiniCheck)
  - Liyan Tang, Philippe Laban, Greg Durrett. *EMNLP*, 2024



### Datasets

- MCFEND: A Multi-source Benchmark Dataset for Chinese Fake News Detection. [[link]](https://dl.acm.org/doi/10.1145/3589334.3645385) [[code]](https://github.com/TrustworthyComp/mcfend)
  - Yupeng Li, Haorui He. *WWW*, 2024
- The Largest Social Media Ground-Truth Dataset  for Real/Fake Content: TruthSeeker. [[link]](https://ieeexplore.ieee.org/document/10286332) [[code]](https://www.unb.ca/cic/datasets/truthseeker-2023.html)
  - Sajjad Dadkhah, Xichen Zhang, Alexander Gerald Weismann. IEEE Transactions on Computational Social Systems. 2023



#### Reference

[[Fudan MAS]](https://fdmas.github.io/research/fake-news-detection.html)

[SIGIR 2024 Tutorial: Preventing and Detecting Misinformation Generated by Large Language Models](https://sigir24-llm-misinformation.github.io/)

[ICTMCG/LLM-for-misinformation-research: Paper list of misinformation research using (multi-modal) large language models, i.e., (M)LLMs.](https://github.com/ICTMCG/LLM-for-misinformation-research/)

[wangbing1416/Awesome-Fake-News-Detection: An awesome paper list of fake news detection (FND) and rumor detection.](https://github.com/wangbing1416/Awesome-Fake-News-Detection)

[llm-misinformation-survey](https://github.com/llm-misinformation/llm-misinformation-survey)

[Awesome-Misinfo-Video-Detection](https://github.com/ICTMCG/Awesome-Misinfo-Video-Detection)



