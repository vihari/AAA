Code for the NeurIPS 2021 paper:   
[Active Assessment of Prediction Services as Accuracy Surface Over Attribute Combinations](https://arxiv.org/pdf/2108.06514.pdf).  

# Table of Contents

1.  [Package dependencies](#orgefa07b1)
2.  [Datasets](#orgb64c62a)
3.  [Code files and their utility](#orge5bb09c)
4.  [Instructions for running](#org08d262b)
    1.  [For Estimation only Experiments of Section 4.5 Reported in Table 2, Figure 3](#org968aba9)
    2.  [For Estimation + Exploration reported in Figure 4](#org85a290e)
    3.  [Impact of Calibration of Section 4.7 and Figure 5](#org7bfa49b)


<a id="orgefa07b1"></a>

# Package dependencies

-   Python 1.6
-   Pytorch 1.6.0
-   GPytorch
-   Pickle
-   tqdm


<a id="orgb64c62a"></a>

# Datasets

The following datasets are used. If you wish to run the MF-\* tasks, you are required to download the CelebA dataset from the below link and place it in `datasets` of your home folder.
You do not have to download COCOS dataset for running AC-\* tasks. See "Instructions for Running" section. 

-   CelebA (<http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>)
-   COCOS (<https://github.com/nightrome/cocostuff>)


<a id="orge5bb09c"></a>

# Code files and their utility

The task and the corresponding file name is mentioned is as below.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Task</th>
<th scope="col" class="org-left">File name</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">MF-CelebA</td>
<td class="org-left">celeba.py</td>
</tr>


<tr>
<td class="org-left">MF-IMDB</td>
<td class="org-left">imdb.py</td>
</tr>


<tr>
<td class="org-left">AC-COCOS</td>
<td class="org-left">cocos.py</td>
</tr>


<tr>
<td class="org-left">AC-COCOS10K</td>
<td class="org-left">cocos10k.py</td>
</tr>
</tbody>
</table>

Important source tree files and their utiltity is shown in table below.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">File name</th>
<th scope="col" class="org-left">Utility</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left"><code>src/beta_gp_rloss_explorer.py</code></td>
<td class="org-left">Implements BetaGP, BetaGP-SL, BetaGP-SLP</td>
</tr>


<tr>
<td class="org-left"><code>src/simple_explorer.py</code></td>
<td class="org-left">Implements Beta-I</td>
</tr>


<tr>
<td class="org-left"><code>src/bern_gp_rloss_explorer.py</code></td>
<td class="org-left">Implements BernGP</td>
</tr>


<tr>
<td class="org-left"><code>src/betaab_gp_rloss_explorer.py</code></td>
<td class="org-left">BetaGPab</td>
</tr>


<tr>
<td class="org-left"><code>src/likelihoods/beta_gp_likelihoods.py</code></td>
<td class="org-left">Implements different data likelihoods. The routines: `baseline`, `simplev3_rloss` correspond to BetaGP-SL, BetaGP-SLP</td>
</tr>


<tr>
<td class="org-left"><code>src/dataset.py</code></td>
<td class="org-left">Abstract classes for Dataset, service model</td>
</tr>


<tr>
<td class="org-left"><code>src/dataset_fitter.py</code></td>
<td class="org-left">Routines for calibration, sampling from arms, warm start etc.</td>
</tr>


<tr>
<td class="org-left"><code>notebooks/ToyGP.ipynb</code></td>
<td class="org-left">Has the code for the simple setting described in the Appendix.</td>
</tr>
</tbody>
</table>


<a id="org08d262b"></a>

# Instructions for running

1.  Setup your environment by installing all the package dependencies listed above.
2.  Download `data.zip` from this [drive link](https://drive.google.com/file/d/1ka6D2_LorQ_GCGzFgn4FmUdXIxbvcCe8/view?usp=sharing) (300MB). Unzip `data.zip` in the working directory.
3.  (Optional) If you wish to run `MF-*` tasks, download [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place it in `$HOME/datasets`.
4.  Follow the description below for the specific python commands.


<a id="org968aba9"></a>

## For Estimation only Experiments of Section 4.5 Reported in Table 2, Figure 3

**Beta-I**
`python <task>.py --explorer simple --seed 0 --ablation`

**BernGP**
`python <task>.py --explorer bern_gp_rloss --seed 0 --ablation`

**BetaGP**  
`python <task>.py --explorer beta_gp_rloss --seed 0 --approx_type baseline --no_scale_loss --ablation`

**BetaGP-SL**
`python <task>.py --explorer beta_gp_rloss --seed 0 --approx_type baseline --ablation`

**BernGP-SLP**  
`python <task>.py --explorer beta_gp_rloss --seed 0 --approx_type simplev3 --ablation`

`<task>` should be set to one of `celeba, celeba_private, cocos3, cocos3_10k`.


<a id="org85a290e"></a>

## For Estimation + Exploration reported in Figure 4

Use the same commands described in the previous section but with `--ablation` flag removed.


<a id="org7bfa49b"></a>

## Impact of Calibration of Section 4.7 and Figure 5

`python <task>.py --explorer simple --seed 0 --ablation --sample_type [option]`

Sample type options should be one of `correctedwep`, `correctednoep`, `raw` and correspond to `Cal:Full`, `Cal:Temp`, `Cal:Simple` respectively of the paper.  
`<task>` should be set to one of `celeba, celeba_private, cocos3, cocos3_10k`.

