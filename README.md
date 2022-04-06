<font face="微软雅黑" color=white size=6>NLP</font><br/>
# 1. NLP 四大基本任务
**NLG (自然语言生成）NLU(自然语言处理）**
## 1.1. 序列标注
分词 词性标注 命名实体识别

## 1.2. 分类任务
文本分类、 情感分析

## 1.3. 句子关系
语义相似度 句子成分分析 、依存句法分析、语义角色标注、 问答系统、 信息抽取

## 1.4. 生成任务
机器翻译 文本生成 文本摘要




# 2. 自然语言处理的基本过程

## 2.1. 获取语料

## 2.2. 预处理

## 2.3. 特征工程

## 2.4. 构建模型

## 2.5. 模型评估

**评测指标**<br/>
Precision <br/>
Accuracy<br/>
Recall<br/>
F1<br/>
Rouge-L<br/>
BLEU<br/>

 ```flow
 st=>start: Start
 i=>inputoutput: 输入年份n
 cond1=>condition: n能否被4整除？
 cond2=>condition: n能否被100整除？
 cond3=>condition: n能否被400整除？
 o1=>inputoutput: 输出非闰年
 o2=>inputoutput: 输出非闰年
 o3=>inputoutput: 输出闰年
 o4=>inputoutput: 输出闰年
 e=>end
 st->i->cond1
 cond1(no)->o1->e
 cond1(yes)->cond2
 cond2(no)->o3->e
 cond2(yes)->cond3
 cond3(yes)->o2->e
 cond3(no)->o4->e
  ```
