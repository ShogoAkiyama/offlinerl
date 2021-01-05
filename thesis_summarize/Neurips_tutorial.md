# NeurIPS 2020 Offline Reinforcement Learning Tutorial



## Types of Reinforcement Learning 

![Screenshot at 2020-12-16 12-35-08](/home/shogo/Desktop/offlinerl/thesis_summarize/image/Screenshot at 2020-12-16 12-35-08.png)

(a) On-policy RLは挙動方策$\pi_k$でサンプルしたデータを用いて方策$\pi_{k+1}$に更新する。以下$k←k+1$を繰り返す。

(b) Off-Policy RLは挙動方策$\pi_k$でサンプルしたデータをリプレイバッファ$D$に貯めておく。サンプルデータを複数回使って方策$\pi_{k+1}$に更新する。以下$k←k+1$を繰り返す。

Off-Policyは集めたデータを何回も学習に利用するため、サンプルデータの利用効率（サンプル効率）は良い。しかし、リプレイバッファ$D$に貯まっているデータは方策$\pi_k$やその前の方策$\pi_{k-1}$など過去の方策で集めたデータであるため、既に改善された方策を再度学習している可能性がある（学習効率が悪い）。

(c) Offline RLは方策$\pi_\beta$で集めたデータをリプレイバッファ$D$に貯めておき、方策$\pi$を学習する。

上記の2つは方策の修正と環境からのサンプルを繰り返すが、Offlineでは既にサンプル済みのデータのみでの学習を行う。(a),(b)をまとめてOnline RLともいう。Offline RLはOnline RLより現実的で、環境との相互作用にはシミュレータや実際の運用システム上での学習が必要であるが、それらがない・できない環境で役に立つ。（例えば自動運転、広告最適化、医療現場など）



## Direct Policy Gradient

（余談）私が強化学習を勉強して最初に勉強したのはQ-Learningだった。そのあとActor-Critic、Policy-Gradientを勉強したが、Q-Learningとは独立した考え方だと思っていた。このチュートリアルで分かったのは、それは勘違いでPolicy-Gradient→Actor-Critic→Q-Learningとして数式で導出できるということだ。



強化学習は$T$ステップ間で得られる報酬の合計を最大化することである。
$$
max E_{\tau \sim \pi_\theta(\tau)} [\sum_{t=0}^{T}r^tr(s_t,a_t)]
$$
サンプルされる$s,a$の生成確率を$\pi_\theta(\tau) = p_\theta(s_1,a_1,...,s_T,a_T)$とおく。$\tau$は$trajectory$であり$\tau=(s_1,a_1,....,s_T,a_T)$である。



ここで強化学習では上記の式を最大化する方策を求めたい。したがって、学習するのは方策$\pi(a|s)$であることから、学習パラメータを$\theta$と置き$\pi_\theta(a|s)$を推定する。ここで求めるのは上記の式を最大化する最適方策$\pi^*$である。

この式をパラメータ$\theta$を求めるという意味で$J(\theta)$とかく。方策勾配法では、$J(\theta)$を最大化する$\theta$を求めるため、勾配上昇法により最適パラメータ$\theta^*$を求める。

勾配上昇法
$$
\theta \leftarrow\theta+\alpha \nabla J(\theta)
$$
以下では、$J(\theta)$の微分を求める。
$$
\begin{align*}
J(\theta) 
&= E_{\tau \sim \pi_\theta(\tau)} [\sum_{t=0}^{T}r^tr(s_t,a_t)] \\
&= E_{\tau \sim \pi_\theta(\tau)} [r(\tau)] 
\quad
(r(\tau)=\sum_{t=0}^{T}r^tr(s_t,a_t)) \\
&= \int r(\tau)\pi_\theta(\tau) d\tau \\
\end{align*}
$$
以下では、$J(\theta)$の微分を求める。
$$
\begin{align*}
\nabla J(\theta) 
&= \int r(\tau) \nabla\pi_\theta(\tau) d\tau \\
&= \int r(\tau) \pi_\theta(\tau)\nabla_\theta log \pi_\theta(\tau) d\tau
\quad
(\nabla_\theta\pi_\theta(\tau)=\pi_\theta(\tau) \nabla_\theta log \pi_\theta(\tau)) \\
&= E_{\tau \sim \pi_\theta(\tau)} [r(\tau) \nabla_{\theta} log \pi_\theta(\tau)] \\
&= E_{\tau \sim \pi_\theta(\tau)} [\nabla_{\theta} log \pi_\theta(\tau) r(\tau)]\\
&= E_{\tau \sim \pi_\theta(\tau)} [\nabla_{\theta} log \pi_\theta(\tau) \sum_{t=0}^{T}r^tr(s_t,a_t)] \\
&= E_{\tau \sim \pi_\theta(\tau)} [\nabla_{\theta} log \pi_\theta(\tau) \hat{Q}] \\
\end{align*}
$$
$\sum_{t=0}^{T}\gamma^t r(s_t,a_t)$を直接報酬で計算すると、

1. 報酬がスパースな場合に、$\hat{Q}=0$になり勾配が計算できない。
2. 報酬が大きすぎる場合に勾配が振動・発散する。

これを解決するために$\hat{Q}$も推定するのが$Actor~Critic$である。



先程の式は$Law~of~total~Expectations$の定理$E[X]=E[E[X|Y]]$より、
$$
\begin{align*}
\nabla J(\theta)
&= E_{\tau \sim \pi_\theta(\tau)} 
[\nabla_{\theta} log \pi_\theta(\tau) \sum_{t=0}^{T}r^tr(s_t,a_t)] \\
&= E_{\tau \sim \pi_\theta(\tau)} 
[E[\nabla_{\theta} log \pi_\theta(\tau) \sum_{t=0}^{T}r^tr(s_t,a_t)|\tau]] \\
&= E_{\tau \sim \pi_\theta(\tau)} 
[\nabla_{\theta} log \pi_\theta(\tau) E[\sum_{t=0}^{T}r^tr(s_t,a_t)|\tau]] \\
&= E_{\tau \sim \pi_\theta(\tau)} 
[\nabla_{\theta} log \pi_\theta(\tau) Q^{\pi_\theta}(s_t,a_t)] \\
\end{align*}
$$




$\hat{Q}(s_t, a_t)=E_{\tau \sim \pi_\theta(\tau)}[\sum_{t=0}^{T}r^tr(s_t,a_t)]$は累積報酬の期待値であるため、環境からの報酬では求められない（確率は分からない）ため推定する。求めるには以下の$Bellman~Equations$を使う。つまり動的計画法。
$$
Q^\pi(s_t,a_t)=r(s_t, a_t)+\gamma E[Q^\pi(s_{t+1},a_{t+1})]
$$
つまり、状態行動価値$Q(s_t,a_t)$は、次に受け取る報酬と次の状態行動価値の期待値$E[Q(s_{t+1},a_{t+1})]$で表現できる。ここで右辺を教師データと考えると、次の用に書ける。
$$
E_{(s,a) \sim \pi_\beta(s,a)}[\{Q_(s,a) - (r(s,a)+\gamma E[Q(s',a')])\}^2]
$$
アルゴリズムはしたのような感じ。

![Screenshot at 2020-12-18 18-52-05](/home/shogo/Desktop/offlinerl/thesis_summarize/image/Screenshot at 2020-12-18 18-52-05.png)

最後に$Q$学習とはなんなのか。Q学習は行動価値を推定し、$argmaxQ(s,a)$を探索するアルゴリズムと見れる一方で、学習した際に強化学習で大事な方策の概念がいまいちピンとこなかった。実際には$Q$学習は方策が常に決定的な$Actor~Critic$として理解可能である。

![Screenshot at 2020-12-18 18-54-16](/home/shogo/Desktop/offlinerl/thesis_summarize/image/Screenshot at 2020-12-18 18-54-16.png)

## Offline Reinforcement Learning

オフライン強化学習について説明する。

### オフライン強化学習とは

オフライン強化学習とは何か。区別するために今までの強化学習アルゴリズムはオンライン強化学習と呼ぶ。

基本的に強化学習は環境との相互作用という表現を使う。我々が学習させたい学習モデルをAgentと呼び、Agentは環境上で何回も行動をすることで失敗と成功の経験から最適行動を探し出す。これがオンライン強化学習である。

一方で、実問題の環境上ではAgentが何回も行動できるとは限らない。例えば自動運転、何回も失敗してたら車がペシャンコである。オフライン強化学習の目的は、人間か誰かからのデータから強いAgentを作ることである。

オフライン強化学習以前には、模倣学習や逆強化学習などのエキスパートのデータからAgentを学習する方法があった。しかし逆強化はエキスパートが最適な方策であるという条件があり、これは制約として厳しい。

オフライン強化学習はAgentはデータセットから最適な方策を探す。また、データセット内での方策にsuboptimalでも良い手法である。



### オフライン強化学習への期待

実問題では自動運転、広告最適化、医療などが期待できる。

マルチタスクでは、例えば1.物をつかんで、2.持ち上げ、3.運んで、4.置くという動作があるとする。この問題ではサブゴールなどの方法を取るなどが上げられるが、完璧なAgentができるのには何回も学習させる必要があるだろう。これを、例えばそれぞれのタスクでだけ学習させたAgentでデータセットに保存しておき、そのデータセットから学習させられれば、学習する時間は早くなるのではないかと考えられる。



### オフライン強化学習の問題

オフライン強化学習の問題は、データセットにない状態行動対に過大評価することである。オンライン強化学習の場合、過大評価された行動は、環境に対して実際に行動することで修正されるのに対し、オフライン強化学習はこれができないため、実際の価値より過大評価され、それらが$Bellman~Operator$によって伝播され、$Q$テーブルが壊れるという問題がある。

![Screenshot at 2020-12-17 19-14-28](/home/shogo/Desktop/Screenshot at 2020-12-17 19-14-28.png)



過大評価が起きると、Agentはその行動を選択する。その過大評価された行動は実際にはデータセットに無い行動があるためデータセットの方策と異なった方策を学習したことになる。（distributinal shift）



### アルゴリズム

オフライン強化学習のアルゴリズムは基本的には過大評価を解決するために以下の２つの方法をとる。

1. **学習方策に制約**を与えることで、リプレイメモリ内での方策となるべく近くする。つまり、なるべくデータセット内に存在する状態行動対を選択する。
2. データセット内に**存在しない**状態行動対の報酬には**罰則**を与える。

1のアプローチで有名なのはBCQ、BEAR、BRACなどである。

2はおそらくCQL、MOPOなどである。



