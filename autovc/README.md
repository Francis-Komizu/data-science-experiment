# 基于AutoVC的实验

## 介绍


## 实验

### 实验1：修改speaker embedding

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



### 实验2：可视化speaker embedding


## 参考

[auspicious3000/autovc: AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss (github.com)](https://github.com/auspicious3000/autovc)

