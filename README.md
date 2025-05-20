# 代码核心功能说明
## 算法基础
该仓库采用多项式朴素贝叶斯分类器,其基于条件概率的特征独立性假设为：
1. 特征条件独立性：假设所有特征（如单词）在给定类别下相互独立
2. 多项式分布：适用于离散型特征（如词频或TF-IDF值），建模特征出现的次数
***
贝叶斯定理在邮件分类中的具体应用形式为：
1. 对于邮件分类（垃圾/普通），算法计算后验概率：P(类别∣特征)∝P(类别)⋅ ∏ P(特征i∣类别)
2. 先验概率P(类别)：通过训练数据中各类别的比例计算（如垃圾邮件占比127/151）
3. 似然概率P(类别∣特征)：统计每个词在各类别中的出现频率（平滑处理避免零概率）
***
邮件分类中的具体应用:
1. 输入：邮件的词频向量（如[0, 2, 1, ...]表示各特征词的出现次数）
2. 输出：选择使后验概率最大的类别（`argmax P(类别|特征)`）
## 数据处理流程
分词处理:
1. 使用`jieba.cut()`对中文文本分词（如将`"自然语言处理"`切分为`["自然", "语言", "处理"]`）
2. 过滤单字词（`len(word) > 1`），减少噪声特征。
***
停用词与无效字符过滤:
1. 正则表达式清洗
   + `re.sub(r'[.【】0-9、——。，！~\*]', '', text)  # 移除标点符号和数字`
2. 停用词处理
   + 代码中未显式定义停用词表，但通过长度过滤（`len(word) > 1`）隐式去除了部分无意义词;若需增强效果，可添加自定义停用词表（如`stop_words = ["的", "是", ...]`）
***
文本标准化:
+ 分词后使用空格拼接（`' '.join(words)`），以满足`TfidfVectorizer`的输入格式要求
## 特征构建过程
高频词特征选择:
1. 数学表达
   + 对每个文档d，生成词频向量x，其中`xi=count(wi)`（词wi的出现次数）
2. 实现差异
   + 手动统计词频（`Counter + map`），选择前`top_num`高频词
   + 特征值为原始计数，未考虑词的全局重要性
***
TF-IDF加权特征：
1. 数学表达

$$
\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \log\left(\frac{N}{\text{DF}(w)}\right)
$$
   + TF(w,d)：词w在文档d中的频率
   + DF(w)：包含词w的文档数
2. 实现差异
   + 使用`TfidfVectorizer`自动计算加权值，特征值为归一化后的权重
   + 通过`max_features`限制特征维度，类似高频词截断
## 高频词/TF-IDF两种特征模式的切换方法
通过`extract_features(method, top_num)`函数实现切换：
1. 高频词模式（`method='frequency'`）：
   + `vector = [ [words.count(word) for word in top_words] for words in all_words ]`
2. TF-IDF模式（`method='tfidf'`）：
   + `tfidf = TfidfVectorizer(max_features=top_num)
   vector = tfidf.fit_transform(texts).toarray()`
# 样本平衡处理
SMOTE过采样集成:
1. 在模型训练前添加SMOTE过采样步骤
2. 使用`smote.fit_resample()`生成平衡数据集
3. 打印过采样前后的类别分布以验证效果
***
样本平衡验证:
+ `print("过采样前类别分布:", np.bincount(labels))  # 输出: [24 127]
print("过采样后类别分布:", np.bincount(labels_resampled))  # 输出: [127 127]`
***
模型训练调整:
1. 使用平衡后的数据(`vector_resampled, labels_resampled`)训练模型
2. 保持预测函数不变，确保接口一致性
# 增加模型评估指标
新增训练测试机划分:
1. `from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    vector, labels, test_size=0.2, random_state=42, stratify=labels)`
2. 使用stratify参数保持原始类别比例
3. 测试集比例为20%（可调整）
***
SMOTE过采样优化:
1. 现在仅对训练集进行过采样，测试集保持原始分布
2. 打印过采样前后的类别分布对比
***
分类评估报告集成:
1. `from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['普通邮件', '垃圾邮件']))`
2. 输出精度(precision)、召回率(recall)、F1值等指标
3. 按类别（普通邮件/垃圾邮件）分别显示


