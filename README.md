# SYSZUXface
DeepVAC-face test dataset.
一个高质量的用于1:N人脸识别的开源人脸测试集。
人脸的检测和识别算法均需自定义。

dataset目录中的图片文件使用git lfs维护，克隆该项目前，你需要首先安装git-lfs：
```bash
#on Linux
apt install git-lfs

#on macOS
brew install git-lfs
```
然后：
```bash
#克隆该项目
git clone https://github.com/DeepVAC/SYSZUXface

#拉取dataset图片
git lfs pull
```

## 使用说明
定义如下概念：
- TP: 将正类预测为正类数，也即已注册db的ds 匹配到了 对应的db；
- FN: 将正类预测为负类数；也即已注册db的ds 没有匹配 对应的db；
- FP: 将负类预测为正类数；也即没注册db的ds 匹配到了 某个db；
- TN: 将负类预测为负类数；也即没注册db的ds 没有匹配 任何db；

定义准确率、精确率、召回率如下：
- 准确率(accuracy) = (TP+TN)/(TP+FN+FP+TN)
- 精确率(precision) = TP/(TP+FP)
- 召回率(recall) = TP/(TP+FN)
- 漏检率(漏检率) = FN/(TP+FN)
- 错误率(错误率) = (FN+FP)/(TP+FN+FP+TN)

项目的目录说明如下：
|  目录   |  说明   |
|---------|---------|
|dataset  |数据集   |
|src     |测试示例代码|
|dataset/db | 底库图片，包含多个子目录，默认合并使用|
|dataset/db/famous | 底库图片，ds/famous测试图片对应的底库图片|
|dataset/db/soccer | 底库图片，ds/soccer测试图片对应的底库图片|
|dataset/db/allage | 底库图片，底库的干扰项图片|
|dataset/db/hancheng | 底库图片，底库的干扰项图片|
|dataset/db/manyi | 底库图片，底库的干扰项图片|
|dataset/ds | 待测试图片，包含多个子目录 |
|dataset/ds/famous | 待测试图片，以公众人物为主，对应的底库图片为db/famous |
|dataset/ds/soccer | 待测试图片，以足球运动员为主，对应的底库图片为db/soccer |

经典的测试步骤如下：
- 合并dataset/db下的子目录，组成大而全的底库，目前具备4w+的底库图片；
- 调整自己的人脸检测和识别算法，确保每个每个底库图片都能提取出特征；
- 使用自定义人脸检测和识别算法，生成适用于自己算法的底库；
- 使用自定义人脸检测和识别算法，对ds的子目录进行检测、特征提取、底库特征匹配；
- 计算ds/soccer的准确率、精确率、召回率、漏检率、错误率；
- 计算ds/famous的准确率、精确率、召回率、漏检率、错误率；

## 使用许可
本项目仅限用于纯粹的学术研究，如：
- 个人学习；
- 比赛排名；
- 公开发表且开源其实现的论文；

不得用于任何形式的商业牟利，包括但不限于：
- 任何形式的商业获利行为；
- 任何形式的商务机会获取；
- 任何形式的商业利益交换；


## 项目贡献
我们欢迎各种形式的贡献，包括但不限于：
- 提交自己的作品/产品在SYSZUXface上的成绩；
- 发现和Fix项目的bug；
- 提交高质量的测试集数据；
