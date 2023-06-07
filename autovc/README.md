# 基于Autoencoder的语音转换

## 介绍

Autoencoder是一种无监督学习模型，通常用于数据的降维、特征提取和数据重构等任务。它由两部分组成：

- 编码器（encoder）

- 解码器（decoder）

编码器将输入数据映射到低维度的潜在空间表示，通常称为编码（encoding）；解码器将编码映射回原始数据的表示，通常称为解码（decoding）。在训练过程中，Autoencoder的目标是尽可能地重构输入数据，即最小化输入数据与解码数据之间的差异。

详细的解释见[presentation.ipynb](../presentation.ipynb)



## 推理

详细的推理脚本见[inference.py](inference.py)，其中需要填写的参数如下：

```python
generator_path = ''  # autoencoder模型
hifigan_path = ''  # hifigan模型
hifigan_config_path = ''  # hifigan配置路径
speaker_encoder_path = ''  # speaker encoder路径
src_wav_path = ''  # 源语音路径，注意采样率为16000，单声道
trg_wav_path = ''  # 目标语者的语音路径，注意采样率为16000，单声道
save_path = ''  # 生成语音的保存路径
```

模型在[这里](https://github.com/Francis-Komizu/data-science-experiment/releases/tag/0)下载。



## 实验

通过融合source与target的speaker embedding，得到不同的输出语音。

仅修改weight。

<table>
  <tr>
    <th>source</th>
    <th>target</th>
    <th>converted</th>
    <th>weight</th>
    <th>scale</th>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/p225xp270.wav"></audio></td>
    <td>1.0</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/exp1/p225xp270_0.5_1.0.wav"></audio></audio></td>
    <td>0.5</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/exp1/p225xp270_0.2_1.0.wav"></audio></audio></td>
    <td>0.2</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/exp1/p225xp270_0.7_1.0.wav"></audio></audio></td>
    <td>0.7</td>
    <td>1.0</td>
  </tr>
</table>

仅修改scale。

<table>
  <tr>
    <th>source</th>
    <th>target</th>
    <th>converted</th>
    <th>weight</th>
    <th>scale</th>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/p225xp270.wav"></audio></td>
    <td>1.0</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/exp1/p225xp270_1.0_0.1.wav"></audio></audio></td>
    <td>1.0</td>
    <td>0.1</td>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/exp1/p225xp270_1.0_0.5.wav"></audio></audio></td>
    <td>1.0</td>
    <td>0.5</td>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/exp1/p225xp270_1.0_1.5.wav"></audio></audio></td>
    <td>1.0</td>
    <td>1.5</td>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/exp1/p225xp270_1.0_2.0.wav"></audio></audio></td>
    <td>1.0</td>
    <td>2.0</td>
  </tr>
  <tr>
    <td><audio src="wavs/p225xp225.wav"></audio></td>
    <td><audio src="wavs/p270xp270.wav"></audio></td>
    <td><audio src="wavs/exp1/p225xp270_1.0_10.0.wav"></audio></audio></td>
    <td>1.0</td>
    <td>10.0</td>
  </tr>
</table>




## 参考

[auspicious3000/autovc: AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss (github.com)](https://github.com/auspicious3000/autovc)

