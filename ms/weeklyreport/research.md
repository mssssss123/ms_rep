
# 0 题目

Full Text Modeling Relieves Cold Start Problem in Sequential Recommendation

# 1 文献调研

### 1.1 相关工作

* 序列化推荐系统
  
  早期序列化推荐系统基于马尔可夫链假设以及矩阵因子分解方法，去预测用户下一个交互的item[1,2,3,4]，但处理不了复杂的序列。随着神经网络的发展，序列化推荐系统也转向了这些架构，如利用cnn[5,6]，rnn[7,8,9,10,11,12]，gnn[13,14]去捕获序列中item与item之间的交互。随着transformer模型在nlp等领域大放异彩，自注意力机制模型凭借其能够捕捉item之间长期依赖的能力，被广泛使用在推荐系统领域。代表性的模型有SARSRec模型[15]，将自注意力机制用于提取序列特征，随后的Bert4Rec模型[16]将完形填空训练以及双向注意力机制引入，二者都成为研究序列化推荐模型的重要基线。人们对自注意力机制模型的改进也得到了广泛的研究，近年来比较值得注意的是将辅助信息引入进来，而不是单纯的只使用id表示item。简单的做法是直接将辅助信息表示与id表示进行拼接[17]。最近的工作研究纷纷对如何更好地融合辅助信息提出了改进，FDSA模型[18]使用了不同的注意力模块去编码item及辅助信息，并在最后阶段融合，S3Rec[19]设计了用于预训练的自监督任务去结合辅助信息的监督信号，ICAI-SR[20]模型通过构建异构图去学习item和属性的表示并将融合后的向量用于序列推荐。NOVA[21]模型认为直接融合会破坏掉item向量空间的一致性，因此只对注意力计算的query和key使用融合的embeddings，value只使用id embedding。DIF-SR模型[22]进一步作出改进，将融合的过程由输入层改到注意力层之后，使用不同的注意力模块去计算id、属性各自的注意力矩阵，并对它们进行融合，而value也只使用id embedding。
* 冷启动问题
  
  推荐系统的冷启动问题是一个需要解决且具有挑战性的问题。现有的冷启动方法可分为两类：单独训练和联合训练方法。单独训练[23,24,25,26,27]是指，分别训练两个模型，一个是cf协同过滤模型，用于去学习到热启动item的embedding，一个是content模型，学习如何通过热启动item的content featrues转换到它的item embedding上。通过分别训练这样两个模型，在inference的过程中，就可以将冷启动的item，通过其content features生成出cf embeddings。联合训练[28,29,30]将上述两个模型进行一起训练。
* 文本建模
  
  如[31]所说，现有的推荐模型推广到跨域是困难的，尽管序列化推荐系统通过将用户转化为其历史行为item解决掉了新用户的建模问题，但对于没有看到新出现且可能和用户相关的新的item模型很难去编码，而根本原因在于目前使用分类索引（id)去编码。为了解决这一问题[31,32,33,34]，开始尝试去id化,使用item的其他描述信息.但现有的工作大多是聚焦在zero-shot领域，很少有工作尝试在传统的推荐领域。最近的工作[35]通过对传统推荐模型进行改造，利用文本或图像替换id取得了比原有模型要好的效果，但没有去和sota模型进行比较，只是和本身基线做了比较。
  

### 1.2 存在问题

1.id vs 文本

推荐系统发展至今，id作为标识占据着主导地位。随着nlp等领域的发展，模型在建模文本等模态越来越强大，因此自然而然有这样一个问题，只基于id的模型真的比基于其他模态的模型要好吗？
* id model依赖于交互，当用户和item交互较少时无法提供很好的推荐，即冷启动问题。
* 推荐领域由于id的不互通，无法实现nlp等其他领域的通用大模型
* 工业界为此需要维护一个庞大且经常更新的embedding矩阵
* id model对推荐目标可解释性差
  
2.冷启动

推荐系统的冷启动问题一直是困扰这个领域的一个难题。现有的工作大多是通过在热启动item上训练出一个转换模型，由item的特征生成相应的id embedding，但这样做并不是一个很好的解决办法。

3.长序列建模

出于性能与效率的权衡，现有的序列化推荐系统大多取用户交互记录的后n个（通常n=50）。但随着互联网的发展，如淘宝每个月用户的交互记录都是非常庞大的，如何有效的建模长序列是一个值得引起重视的问题。并且显然通过引入更多的交互信息可以提高模型性能。在广告ctr领域，现在流行的做法是利用目标item去反过来作为query去搜索用户序列，进而缩短为子序列。但序列化推荐系统是预测next item的任务，是否有一个更好的建模长序列的方法

# 2 研究目标

面向序列化推荐系统领域，分析目前id建模带来冷启动等问题的缺点，提出了使用纯文本建模序列与item表示替代id，为推荐领域解决冷启动问题提供解决方案，并且对于通用推荐模型的研究也能起到启示作用，最后对于文本建模长序列提供一种方法，希望通过我们的研究可以对推荐系统领域的文本建模、冷启动问题、长序列建模做出贡献。

# 3 研究内容

* 具体研究内容
  1.首先探究文本建模的可行性，因为推荐系统领域这方面的研究很少，需要去验证是否可行，使用文本的推荐模型是否与使用id的模型具有竞争力
  2.纯文本建模对于冷启动问题是否可以起到缓解作用，即能否比现有id模型冷启动场景下效果好
  3.设计的长序列建模方法是否对提升性能表现起到帮助
* 关键问题或难点
  冷启动问题的探究
* 预期达到的效果或提升
  预训练语言模型可以为学习的文本表征提供额外知识，理应比随机初始化的向量在推荐以及冷启动场景效果要好

# 4 技术路线

### 4.1 解决方案

1，数据端不同于传统的推荐系统，我们舍弃了id，完全使用文本来表示item以及序列。具体做法是使用item的title文本作为id的替代，并采用title:,category:这样的提示词来去融入辅助信息。
2.模型上使用t5模型作为编码器，对序列端和item端进行编码。
3.由于显存等限制，为了使序列能够在训练过程中见到更多的item，训练过程使用inbatch+random negative进行训练，损失函数为cross- entropy
4.对于长序列建模，采用fid的方式，对模型的ecnoder进行改进，将序列中的item进行独立编码再decode为一个向量表示序列

### 4.2 实验方法

* Dataset
* yelp:数据来自美国著名点评网站，数据集由来自8大都市区域约16万商户，约863万条评论组成。遵照[19,22]处理方式，只保留2019年1.1日之后的交互记录。
* Amazon-beauty:数据来自Amazon review datasets[36]，beauty数据集为用户对化妆品的打分
* 按照[15,19,20,22]预处理方式，移除了数据集中出现次数少于5次的用户和item，所以交互记录都被视作隐性反馈

|Dataset|Users|Items|avg.Actions/Users|avg.Actions/Items|Actions|Sparsity|
|-----|-----|-----|-----|----|----|----|
|beauty|22363|12101|8.9|16.4|198502|99.93%|
|yelp|30499|20068|10.4|15.8|317182|99.95%|

* Baseline

|模型|介绍|引用|
|---|----|---|
|GRU4REC|应用GRU对用户序列进行建模。这是第一个顺序推荐的recurrent模型|[9]|
|GRU4RECF|将side information添加到Gru4Rec中|[39]|
|Caser|是一种基于CNN的方法，通过对顺序推荐应用水平和垂直卷积运算来捕获高阶模式|[5]|
|BERT4Rec|使用掩蔽item训练方案模拟NLP中的掩蔽语言模型。主干是双向的自我注意机制|[16]|
|SASRec|使用self-attention对序列特征进行提取，是一种单向的自注意模型。这是顺序推荐中的一个强有力的基线|[15]|
|SASRecF|在送进模型前，将item和attribute的表示进行拼接|-|
|S3-Rec|模型还是embedding+自注意力机制，但设计了4个自监督任务序列-商品，商品-属性，序列-属性，序列-子序列解决数据稀疏问题|[19]|
|ICAI-SR|一个通用框架，它仔细设计了基于注意力的项目属性聚合模型（IAA）和实体顺序（ES）模型，以利用项目和属性之间的各种关系.DIF论文中将es模型设置为sasrec模型|[20]|
|NOVA|一种采用非侵入式自我注意（NOVA）机制的框架，用于更好的注意力分布学习.DIF论文中在sasrec模型实施nova机制|[21]|
|DIF-SR|将side-information的融合从输入层转移到注意力层，增强item融合辅助信息的能力|[22]|

* Evaluation Metric
  遵照[15,16]，采用leave-one-out策略，即将序列后两个商品保留为验证集数据和测试集数据。使用tok-recall和topk-ndcg对模型进行评估，k取值=「10，20」.[37,38]指出，使用模型对数据集中全部商品进行全排得到的结果更加公平。


# 5 分析实验

* 主要目的
* 验证文本作为item标识而不是id这一做法是否可行，以及能否带来模型表现上的提升
* 验证利用文本去建模序列和item是否对冷启动问题提供一个很好的解决方案
* 提升序列长度可以提高模型性能，能否提供一个有效的建模长序列的方法
* 分析实验（一）
* 目的：验证基于文本的模型和传统的id模型哪个更好
* 方法：与现有的序列化基线模型进行比较
* 预期结论：由于预训练语言模型可以提供额外的knowledge，效果应该比只使用未预训练的transformer架构要好
* 分析实验（二）
* 目的：验证融入辅助信息是否有提升
* 方法：在amazon-beauty数据集和yelp数据集设置了对照实验，如yelp数据集上文本只使用name，以及在序列和商品上加入category文本
* 预期结论：引入了额外的辅助信息能够帮助模型更好地推荐
* 分析实验（三）
* 目的：验证提出的模型是否对冷启动问题起到缓解效果
* 方法：重新构建下数据集，让验证集、测试集中商品不出现在训练集中，比较我们的模型和基线模型效果
* 预期结论：我们的模型通过预训练语言模型生成的embedding，而不是训练随机初始化的向量表，能取得比现有模型要好的效果
* 分析实验（四）
* 目的：验证fid建模长序列是否更有效
* 方法：和我们现有的直接文本的模型基线做比较
* 预期结论：fid独立编码每个item向量，并且可以规避序列受512token长的限制，效果应该比我们目前基线模型要好


# Reference

* [1] Guy Shani, David Heckerman, and Ronen I. Brafman. 2005.  An MDP-BasedRecommender System.J. Mach. Learn. Res.6 (Dec. 2005), 1265–1295
* [2]Ruining He and Julian McAuley. 2016.  Fusing similarity models with markovchains for sparse sequential recommendation. In2016 IEEE 16th InternationalConference on Data Mining (ICDM). IEEE, 191–200
* [3]Santosh  Kabbur,  Xia  Ning,  and  George  Karypis.  2013.   Fism:  factored  itemsimilarity models for top-n recommender systems. InProceedings of the 19thACM SIGKDD international conference on Knowledge discovery and data mining.659–667
* [4]Steffen Rendle. 2010. Factorization machines. In2010 IEEE International confer-ence on data mining. IEEE, 995–1000.
* [5]Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendationvia convolutional sequence embedding. InProceedings of the Eleventh ACMInternational Conference on Web Search and Data Mining. 565–573
* [6]Fajie Yuan, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M Jose, andXiangnan He. 2019.  A simple convolutional generative network for next itemrecommendation. InProceedings of the Twelfth ACM International Conference onWeb Search and Data Mining. 582–590.
* [7]Tim Donkers, Benedikt Loepp, and Jürgen Ziegler. 2017. Sequential User-basedRecurrent Neural Network Recommendations. InProceedings of RecSys. ACM,New York, NY, USA, 152–160
* [8]Balázs Hidasi and Alexandros Karatzoglou. 2018.  Recurrent Neural Networkswith Top-k Gains for Session-based Recommendations. InProceedings of CIKM.ACM, New York, NY, USA, 843–852.
* [9]Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk.2015.  Session-based recommendations with recurrent neural networks.arXivpreprint arXiv:1511.06939(2015)
* [10]Chen Ma, Peng Kang, and Xue Liu. 2019.  Hierarchical gating networks for se-quential recommendation. InProceedings of the 25th ACM SIGKDD internationalconference on knowledge discovery & data mining. 825–833.
* [11]Bo Peng, Zhiyun Ren, Srinivasan Parthasarathy, and Xia Ning. 2021.   HAM:hybrid associations models for sequential recommendation.IEEE Transactionson Knowledge and Data Engineering(2021).
* [12]Massimo Quadrana, Alexandros Karatzoglou, Balázs Hidasi, and Paolo Cremonesi.2017. Personalizing session-based recommendations with hierarchical recurrentneural networks. InProceedings of the Eleventh ACM Conference on Recom-mender Systems. 130–137
* [13]Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan. 2019.Session-based recommendation with graph neural networks. InProceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 33. 346–353
* [14]Jianxin Chang, Chen Gao, Yu Zheng, Yiqun Hui, Yanan Niu, Yang Song, DepengJin, and Yong Li. 2021. Sequential Recommendation with Graph Neural Networks.InProceedings of the 44th International ACM SIGIR Conference on Research andDevelopment in Information Retrieval. 378–387
* [15]Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recom-mendation. In2018 IEEE International Conference on Data Mining (ICDM).IEEE, 197–206
* [16]Fei  Sun,  Jun  Liu,  Jian  Wu,  Changhua  Pei,  Xiao  Lin,  Wenwu  Ou,  and  PengJiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoderrepresentations from transformer. InProceedings of the 28th ACM internationalconference on information and knowledge management. 1441–1450
* [17]Balázs Hidasi, Massimo Quadrana, Alexandros Karatzoglou, and Domonkos Tikk.2016. Parallel recurrent neural network architectures for feature-rich session-basedrecommendations. InProceedings of the 10th ACM conference on recommendersystems. 241–248
* [18]Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, DeqingWang,  Guanfeng  Liu,  and  Xiaofang  Zhou.  2019.   Feature-level  Deeper  Self-Attention Network for Sequential Recommendation.. InIJCAI. 4320–4326
* [19]Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang,Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning forsequential recommendation with mutual information maximization. InProceed-ings of the 29th ACM International Conference on Information & KnowledgeManagement. 1893–1902
* [20]Xu Yuan, Dongsheng Duan, Lingling Tong, Lei Shi, and Cheng Zhang. 2021.ICAI-SR: Item Categorical Attribute Integrated Sequential Recommendation. InProceedings of the 44th International ACM SIGIR Conference on Research andDevelopment in Information Retrieval. 1687–1691
* [21]Chang Liu, Xiaoguang Li, Guohao Cai, Zhenhua Dong, Hong Zhu, and LifengShang. 2021. Non-invasive Self-attention for Side Information Fusion in Sequen-tial Recommendation.arXiv preprint arXiv:2103.03578(2021).
* [22]Yueqi Xie, Peilin Zhou, and Sunghun Kim. 2022. Decoupled Side Infor-mation Fusion for Sequential Recommendation. InProceedings of the 45th
* [23]Iman Barjasteh, Rana Forsati, Farzan Masrour, Abdol-Hossein Esfahanian, and Hayder Radha. 2015. Cold-start item and user recommendation with decoupled completion and transduction. In Proceedings of the 9th ACM Conference on Recommender Systems. ACM, 91–98
* [24]Zeno Gantner, Lucas Drumond, Christoph Freudenthaler, Steffen Rendle, and Lars Schmidt-Thieme. 2010. Learning Attribute-to-Feature Mappings for Cold-Start Recommendations.. In ICDM, Vol. 10. Citeseer, 176–185.
* [25]Martin Saveski and Amin Mantrach. 2014. Item cold-start recommendations: learning local collective embeddings. In Proceedings of the 8th ACM Conference on Recommender systems. ACM, 89–96.
* [26]Ajit P Singh and Geoffrey J Gordon. 2008. Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 650–658.
* [27]Aaron Van den Oord, Sander Dieleman, and Benjamin Schrauwen. 2013. Deep content-based music recommendation. In Advances in Neural Information Processing Systems. 2643–2651.
* [28]Jingjing Li, Mengmeng Jing, Ke Lu, Lei Zhu, Yang Yang, and Zi Huang. 2019. From Zero-Shot Learning to Cold-Start Recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence
* [29]Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, Lexing Xie, and Darius Braziunas. 2017. Low-rank linear cold-start recommendation from social data. InThirty-First AAAI Conference on Artificial Intelligence.
* [30]Maksims Volkovs, Guangwei Yu, and Tomi Poutanen. 2017. Dropoutnet: Addressing cold start in recommender systems. In Advances in Neural Information Processing Systems. 4957–4966.
* [31]Hao Ding, Yifei Ma, Anoop Deoras, Yuyang Wang, and Hao Wang. 2021. Zero-Shot Recommender Systems.arXiv preprint arXiv:2105.08318(2021).
* [32]Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, Ji-Rong Wen. 2022. Towards Universal Sequence Representation Learningfor Recommender Systems. InProceedings of the 28th ACM SIGKDD Con-ference on Knowledge Discovery and Data Mining (KDD ’22), August 14–18,2022, Washington, DC, USA.ACM, New York, NY, USA, 9 pages
* [33]Language Models as Recommender Systems:Evaluations and Limitations
* [34]M6-Rec: Generative Pretrained Language Models areOpen-Ended Recommender Systems
* [35]WHERE  TOGONEXT  FORRECOMMENDERSYSTEMS?ID-VS.   MODALITY-BASED  RECOMMENDER  MODELSREVISITED
* [36]Julian J. McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, Santiago, Chile, August 9-13, 2015, Ricardo Baeza-Yates, Mounia Lalmas, Alistair Moffat, and Berthier A. Ribeiro-Neto (Eds.). ACM, 43–52. https://doi.org/10.1145/2766462.2767755
* [37]Alexander Dallmann, Daniel Zoller, and Andreas Hotho. 2021. A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models. In Fifteenth ACM Conference on Recommender Systems. 505–514.
* [38]Walid Krichene and Steffen Rendle. 2020. On sampled metrics for item recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1748–1757.
* [39]Parallel Recurrent Neural Network Architectures for Feature-rich Session-based Recommendations.
