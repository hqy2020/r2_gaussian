# SeCuRe: Toward Sparse-View 3D Curve Reconstruction with Gaussian Splatting 

## ARTICLE INFO

Article history:

Keywords: 3D Curve Reconstruction, Sparse-View Novel View Synthesis, 3D Gaussian Splatting

## ABSTRACT

3D curve reconstruction provides essential geometric cues for various vision and graphics applications. While recent advances in 3D Gaussian Splatting (3DGS) have enabled efficient and high-quality curve reconstruction, existing methods typically require dense multi-view inputs to achieve reliable results. Under sparse-view conditions, they often suffer from unsatisfactory initialization, overfitting, and insignificance of edge areas, leading to fragmented or noisy reconstructions. In this paper, we present SeCuRe, a novel auxiliary framework for sparse-view 3D curve reconstruction with 3DGS. Our SeCuRe introduces an Edge-Guided Gaussian Initialization (EGGI) strategy that extracts robust geometric priors from sparse inputs by leveraging external dense reconstruction and edge detection models. It further incorporates a Structure-Aware Gaussian Pruning (SAGP) strategy to suppress overfitting through spatial coherence and visibility consistency constraints. Additionally, we propose a sparsity loss to encourage compact and clean curves by penalizing redundant or scattered Gaussian primitives. Extensive experiments on the ABC-NEF and DTU datasets show that integrating our SeCuRe with 3DGS-based methods consistently improves performance under extremely sparse-view conditions, achieving improvements of up to $1852.94 \%, 1129.57 \%$, and $1554.01 \%$ in Precision, Recall, and F-score, respectively.
(c) 2025 Elsevier B.V. All rights reserved.

## 1. Introduction

Curves are critical geometric cues for characterizing the structures of 3D objects and scenes [1, 2]. They efficiently encode structural information while maintaining compatibility with computer-aided design (CAD) systems [3]. This enables a wide range of vision and graphics applications, including sur-
face reconstruction [4], rendering [5], and simultaneous localization and mapping (SLAM) [6].

Recent studies, inspired by notable developments in novel view synthesis (NVS) [7, 8], have attempted to reconstruct 3D curves from multi-view edge maps, which are extracted from raw images using edge detection models. Neural implicit fields, in particular, have shown remarkable promise for 3D curve reconstruction $[9,10]$. However, they still incur considerable computational costs during training, even for simple CAD models [11]. To address this, subsequent studies [12, 13, 14] have leveraged 3D Gaussian Splatting (3DGS) [8], which supports
high-fidelity and real-time rendering, to represent edges for impressive curve reconstruction quality and efficiency.

However, the performance of these 3DGS-based methods rely on well-captured dense multi-view images, which is often cumbersome and impractical in real-world applications [15]. Specifically, sparse views as input can cause significant portions of the scene structure to be incorrectly learned. This results in failures to reconstruct continuous and complete curves, as illustrated in Fig. 1. Although several 3DGS-based methods have been proposed for sparse-view NVS [16, 17], their strategies of incorporating priors as depth and semantic information can be hardly applied to 3D curve reconstruction. This is because edge areas inherently possess weak appearance cues, ambiguous geometric priors, and limited semantic consistency across views. Consequently, regularization techniques effective for NVS [7] often fail to provide adequate constraints for curves.

We identify the following challenges in reconstructing 3D curves from sparse multi-view images:

---

![img-0.jpeg](assets/2025_JJK_CAG_SparseViewCurve%20(1)_img-0.jpeg)

Fig. 1: Recent 3DGS-based methods (e.g., EdgeGaussians [12]) exhibit substantial degradation in curve reconstruction quality when presented with sparse multi-view inputs (e.g., 3 views). Integrating the proposed auxiliary framework, ${ }^{95}$ SeCuRe, enables existing 3DGS-based curve reconstruction methods to consistently achieve improved performance under sparse-view conditions, while maintaining acceptable computational efficiency.

- Initialization Dependency. In sparse-view scenarios, re- ${ }_{9}$ liable reconstruction depends on strong geometric priors ${ }^{90}$ for initialization [18][19]. However, limited input im- ${ }^{91}$ ages provide insufficient constraints and often cause noisy, ${ }^{92}$ fragmented and incomplete points. Consequently, 3DGSbased methods struggle to recover accurate and continuous curves.
- Sparse-View Overfitting. Given limited multi-view images, the optimization process is prone to overfitting [20]. ${ }^{95}$ This problem is especially noticeable in 3D curve reconstruction. Specifically, edge areas tend to exhibit weak ap- ${ }^{97}$ pearance cues, which would drive the method to exploit ${ }^{98}$ shortcuts rather than learning true geometric correspon- ${ }^{99}$ dences.
- Insignificant Edge Areas. In most cases, edge areas oc- ${ }^{102}$ cupy a smaller proportion of image pixels compared to ${ }^{103}$ surface areas, causing the focus of Gaussian optimization ${ }^{104}$ to shift away from edges during pixel-based optimization. ${ }^{105}$ This phenomenon negatively affects both the quality of ${ }^{106}$ curve reconstruction and the convergence behavior.

To address the above-discussed challenges, we propose $\mathrm{Se}_{-109}$ CuRe, a novel auxiliary framework that can integrate with ${ }^{110}$ existing 3DGS-based methods for Sparse-view 3D Curve ${ }^{111}$ Reconstruction. It is capable of reconstructing structurally ${ }^{112}$ sound and geometrically accurate curves with extremely lim-113 ited input views by introducing designated strategies on Gaus-sian initialization and pruning. Specifically, we design an Edge-115 Guided Gaussian Initialization (EGGI) strategy to extract a116 dense and reliable edge point cloud from sparse-view images by ${ }^{117}$ exploiting dense generation [21] and edge detection [22] mod-118 els for geometric priors. Compared to existing methods, it cap-119 tures continuous edges more effectively. Meanwhile, we devise ${ }^{120}$ a Structure-Aware Gaussian Pruning (SAGP) strategy to miti-121 gate overfitting usually caused by sparse inputs while reinforc-122 ing geometric constraints based two regularization techniques ${ }^{123}$ on spatial coherence and visibility consistency. It enables the ${ }^{124}$

Gaussian optimization process to focus on learning true geometric correspondences on edges and thus leads to less artifacts. In addition, we present a sparsity loss to encourage compact and clean curve reconstruction by penalizing redundant or scattered Gaussian primitives, which further improves reconstruction quality and accelerates convergence. Experimental results on representative datasets demonstrate that our SeCuRe is compatible with 3DGS-based methods for 3D curve reconstruction, consistently enhancing reconstruction performance under extremely sparse-view conditions. The contributions of this paper can be summarized as follows:

- We propose a novel auxiliary framework enabling 3DGSbased 3D curve reconstruction methods to achieve superior performance under sparse-view conditions.
- We design an edge-guided strategy that leverages dense reconstruction and edge detection models to extract geometric priors from sparse views for Gaussian initialization.
- We design a structure-aware strategy that mitigates overfitting and reinforces geometric constraints during Gaussian optimization from the perspectives of spatial coherence and visibility consistency.


## 2. Related Work

### 2.1. 3D Curve Reconstruction from Multi-View Images

Compared to conventional 3D curve reconstruction methods, which rely on clean 3D point clouds for curve fitting [23, 24], those based on multi-view images [25, 26] leverage Structure-from-Motion (SfM) [27] techniques to represent edges that are then matched across different viewpoints and triangulated into 3D curves. However, these methods face a significant challenge: edges may be represented incompletely or even be absent due to inadequate observations upon sparse multi-view images. This would cause degraded curve reconstruction quality.

Inspired by the success of neural implicit fields in NVS, such as Neural Radiance Field (NeRF) [7], recent methods have explored learning a continuous field to represent edges for 3D edge prediction under the supervision of 2D edge maps [9, 10]. Despite their demonstrated promise, they are limited by the computational inefficiency of NeRF-based optimization processes, which are resource-intensive and require an unsatisfactory amount of training time usage even for simple CAD models [11]. To address this issue, more recent studies leverage 3D Gaussian Splatting (3DGS) [8] for 3D curve reconstruction, achieving superior performance under dense scene observations. For example, EdgeGaussians [12] represent edges with oriented Gaussians primitives; SGCR [13] constrains the primitives to Spherical Gaussians for more accurate edge representation; CurveGaussian [14] introduces a bi-directional coupling mechanism between curves and edge-oriented Gaussian primitives. While these methods have explored various intermediate representations for curve reconstruction, they are yet to fundamentally address the afore-mentioned challenge of severe performance degradation encountered under sparse-view conditions, which is our focus in this paper.

---

### 2.2. Dense Reconstruction via Multi-View Stereo

Multi-view Stereo (MVS) aims to reconstruct dense 3D ge- ${ }^{178}$ ometry from a set of calibrated images with known camera ${ }^{179}$ poses. Conventional SfM-MVS pipelines (e.g., COLMAP [27]) ${ }^{180}$ estimate depth by evaluating patch similarity across hypothe-181 sized depth planes, regularizing the resulting cost volume, and fusing per-view depth maps into a unified dense model. While effective in textured Lambertian scenes, they struggle in the ${ }^{183}$ presence of low texture, occlusions, or non-Lambertian surfaces. The advent of learning-based methods, exemplified by ${ }^{185}$ MVSNet [28], replaced handcrafted similarity metrics and filtering with deep feature extraction and 3D CNN-based [29] cost ${ }^{187}$ volume regularization, significantly improving completeness, ${ }^{188}$ but at the expense of increased memory consumption. Subsequent studies (e.g., R-MVSNet [30] and CasMVSNet [30]) ${ }^{190}$ have partially addressed this issue through recurrent and coarse- ${ }^{191}$ to-fine strategies.

Recently, the introduction of the Transformer model [31] has ${ }^{193}$ driven a paradigm shift from cost-volume regularization to the ${ }^{194}$ direct establishment of global pixel correspondences. Following this trend, DUSt3R [21] and MASt3R [32] regress dense ${ }^{196}$ and consistent 3D point maps across viewpoints, jointly solving ${ }^{197}$ for structure and alignment without relying on pre-calibrated ${ }^{198}$ cameras; VGGT [33] employs a feed-forward transformer to directly predict camera parameters, dense depth, and point tracks in a single pass. The outputs of these methods as priors for 3D reconstruction have been widely applied to various downstream ${ }^{201}$ tasks, such as SLAM [34], scene understanding [35], and pose ${ }^{202}$ estimation [36]. Due to the characteristics of edges, existing ${ }^{203}$ methods have not yet effectively leveraged such priors for 3D ${ }^{204}$ curve reconstruction.

## 3. Method

### 3.1. Preliminaries

3D Gaussian Splatting. 3DGS [8] has recently emerged as a ${ }_{211}$ promising NVS method and achieved an unprecedented combination of photorealistic quality and real-time rendering per- ${ }_{212}$ formance. Unlike NeRF [7] with its implicit neural representa- ${ }_{214}$ tions, 3DGS models a scene using an explicit set of 3D Gaus- ${ }_{215}$ sian primitives from a set of images of the scene as well as ${ }_{216}$ the corresponding cameras calibrated by a SfM pipeline [27], ${ }_{217}$ This explicit representation facilitates direct manipulation and enables efficient rendering. Specifically, each Gaussian primi- ${ }_{219}$ tive $\theta$ is defined by its geometric properties: a mean $\mu \in \mathbb{R}^{3}$ and ${ }_{220}$ a 3D covariance matrix $\Sigma \in \mathbb{R}^{3 \times 3}$. These parameters together describe the Gaussian's position and shape. The Gaussian's ${ }_{221}$ spatial influence on a 3D position $x$ is given by the following ${ }_{222}$ distribution:

$$
G(x)=\exp \left(-\frac{1}{2}(x-\mu)^{\top} \Sigma^{-1}(x-\mu)\right)
$$

In addition, each Gaussian primitive $\theta$ possesses photometric at-227 tributes: an opacity value $\alpha \in[0,1]$ to control its transparency,228 and a set of Spherical Harmonics (SH) coefficients $c$ to model229 complex and view-dependent appearance effects. To ensure230 the covariance matrix $\Sigma$ remains physically valid (i.e., positive231
semi-definite) throughout optimization, it is parameterized by an optimizable scaling vector $s \in \mathbb{R}^{3}$ and a rotation quaternion $q \in \mathbb{R}^{4}$. These parameters construct a scaling matrix $S$ and a rotation matrix $R$, respectively, such that $\Sigma=R S S^{\top} R^{\top}$. All these parameters are differentiable and optimized end-to-end.

DUSt3R. As a Transformer-based dense reconstruction method, DUSt3R [21] directly predicts dense 3D point maps from image pairs. It operates without requiring camera intrinsics or extrinsics. Unlike traditional SfM pipelines [27], DUSt3R regresses a dense field of 3D points for each pixel, establishing pixel-to-geometry correspondences, and outputs point maps and confidence maps for each input image. It is trained with a confidence-weighted 3D regression loss to enforce geometric consistency. Given a pair of images $I_{1}, I_{2}$, DUSt3R outputs the corresponding dense point maps $P_{1,1}, P_{2,1} \in \mathbb{R}^{W \times H \times 3}$ with associated confidence maps. These are expressed in the coordinate frame of $I_{1}$. Each point map assigns image pixels to their corresponding 3D points. When the input consists of more than two images, DUSt3R aggregates all pairwise point maps and aligns them into a unified point cloud $\mathcal{X}$. Simultaneously, it estimates camera parameters for each view. These parameters are leveraged to map the corresponding image pixels to 3D points in $\mathcal{X}$.

### 3.2. Overview

Given a set of sparse multi-view images, our task is to reconstruct 3D curves that accurately represent the underlying geometry. In contrast to dense reconstruction, 3D curve reconstruction faces distinct challenges due to weak appearance cues and less structural details, which are further amplified under sparse-view conditions. Inspired by the success of Gaussian Splatting in dense reconstruction, we propose SeCuRe, a novel framework that effectively alleviates the challenges of sparseview curve reconstruction through robust initialization and effective pruning strategy. Our framework is composed of three core components: (1) an Edge-Guided Gaussian Initialization strategy that leverages pre-trained models to extract robust geometric priors from sparse views, ensuring reliable and accurate initialization; (2) Structure-Aware Gaussian Pruning, which integrates spatial coherence and visibility consistency regularizations to effectively suppress overfitting and reinforce geometric constraints; and (3) a sparsity loss term designed to encourage compact and clean curve reconstructions by penalizing redundant or scattered Gaussian primitives. The overall pipeline is illustrated in Fig. 2.

### 3.3. Edge-Guided Gaussian Initialization

Vanilla 3DGS-based methods for curve reconstruction typically accomplish Gaussian initialization using sparse points and camera poses output by a SfM pipeline (e.g., COLMAP [26, 37]). Under sparse-view conditions, however, SfM often fails to provide reliable results due to insufficient feature correspondences and limited geometric constraints. Therefore, we design an Edge-Guided Gaussian Initialization (EGGI) strategy for robust geometric priors from sparse views to address this issue.

At first, a state-of-the-art dense reconstruction method (i.e., DUSt3R [21]) is adopted to generate a unified point cloud $\mathcal{X}$ and

---

![img-1.jpeg](assets/2025_JJK_CAG_SparseViewCurve%20(1)_img-1.jpeg)

Fig. 2: Pipeline of the proposed auxiliary framework SeCuRe for sparse-view 3D curve reconstruction, which can be integrated seamlessly with existing 3DGS-based methods. We design an Edge-Guided Gaussian Initialization (EGGI) strategy to provide accurate initial positions for 3D Gaussian primitives by leveraging robust geometric priors. During the Gaussian optimization process, our Structure-Aware Gaussian Pruning (SAGP) strategy dynamically removes outlier and redundant Gaussians based on spatial coherence and visibility consistency regularization techniques, ensuring that the remaining primitives are geometrically consistent and well-aligned with the underlying edges to achieve high-quality curve reconstruction. In addition a sparsity loss is introduced to penalize redundant or scattered Gaussian primitives corresponding to non-edge areas.
camera poses $\mathcal{C}=\left\{C_{i}\right\}_{i=1}^{n}$ from the training views $\mathcal{I}$. Since the ${ }_{252}$ quality of reconstructed scenes via 3DGS is highly sensitive to ${ }_{253}$ initial Gaussian primitives in sparse-view settings [18, 19], ini- ${ }_{254}$ tialization with $X$ by directly aggregating pairwise point maps ${ }_{255}$ would cause numerous Gaussian primitives that may not corre- ${ }_{256}$ spond to actual edge areas. In particular, such outliers converge ${ }_{257}$ slowly under sparse supervision and often persist as artifacts, ${ }_{258}$ which degrades curve reconstruction due to the introduction ${ }_{259}$ of incorrect geometric priors. To ensure accurate capture of ${ }_{270}$ the edge structure, we employ a pre-trained 2D edge detection ${ }_{271}$ model (i.e., PiDiNet [22]) for probability edge maps $\mathcal{E}=\{E_{i}\}_{i=1,272}^{n}$ from $\mathcal{I}$, and use $\mathcal{X}$ and $\mathcal{C}$ to remap $\mathcal{I}$ to point maps $\hat{\mathcal{P}}=\{\hat{P}_{i}\}_{i=1,273}^{n}$ in the same coordinate frame. For each $\hat{P}_{i}$, all points $\left\{p_{i, j}\right\}$ are ${ }_{274}$ projected onto the corresponding edge map $E_{i}$ using the camera ${ }_{275}$ projection function $\pi_{i}(\cdot)$, and we keep only those points whose ${ }_{276}$ projections intersect with pixels on the detected edges to obtain ${ }_{277}$ an edge point map as follows:

$$
\hat{P}_{i}=\left\{p_{i, j} \in \hat{P}_{i} \mid E_{i}\left(\pi_{i}\left(p_{i, j}\right)\right)>0\right\}
$$

It is noteworthy that our EGGI strategy avoids projecting the ${ }_{281}$ dense point cloud $\mathcal{X}$ onto the edge map $E_{i}$ for each training ${ }_{282}$ view $I_{i}$, since this simple operation would otherwise cause edge ${ }_{283}$ points to be removed if their corresponding edges are occluded ${ }_{284}$ in any other view. Instead, the point map $\hat{P}_{i}$ after remapping ${ }_{285}$ is leveraged for projection to retain all edge points as long as ${ }_{286}$ their corresponding edges are visible in at least one view. Sub- ${ }_{287}$ sequently, we aggregate all edge point maps $\hat{\mathcal{P}}=\{\hat{P}_{i}\}_{i=1}^{n}$ to ob- ${ }_{288}$ tain an edge point cloud $\mathcal{X}$. This representation better preserves ${ }_{289}$ the continuity and completeness of edges. Hence, our EGGI ${ }_{290}$ strategy provides robust geometric priors for initializing Gaus-sian primitives with camera poses $\mathcal{C}$, facilitating the following ${ }_{292}$ Gaussian optimization process.

### 3.4. Structure-Aware Gaussian Pruning

For sparse-view 3D curve reconstruction with 3DGS, the optimization of Gaussian primitives is likely to result in overfitting, because edge areas are usually characterized by weak appearance cues. This challenge escalates the difficulty of learning true geometric correspondences.

Recent studies on sparse-view NVS [20] have managed to alleviate overfitting by randomly dropping Gaussian primitives. Considering the application of a similar strategy to 3D curve reconstruction, where structural information is crucial, we argue that selectively pruning Gaussian primitives based on geometric priors could more effectively suppress overfitting and thus improve the quality of curve reconstruction compared to random pruning. Therefore, we design a Structure-Aware Gaussian Pruning (SAGP) strategy to address the afore-mentioned challenge by integrating regularization techniques on spatial coherence and visibility consistency during the Gaussian optimization stage.

Spatial Coherence Regularization. Unlike random pruning, spatial coherence regularization leverages geometric priors to reduce outlier Gaussian primitives that hardly conform to the underlying edges, ensuring that pruning is guided by meaningful spatial relationships. Given the current Gaussian primitives, we employ a density-based clustering algorithm (i.e., DBSCAN [38]), and dynamically estimate the neighborhood radius using $k$-nearest neighbors (KNN) distance statistics of the Gaussian positions. Specifically, for each Gaussian $\theta_{i}$, we calculate the average distance $d_{i}$ from its position $\mu_{i}$ to its $k$ nearest neighbors. The neighborhood radius is defined as the scaled median of all distances $\left\{d_{i}\right\}$ using a scaling factor $\gamma$. Afterward, Gaussians not assigned to any cluster are labeled as spatial outliers and thus pruned. This regularization encourages

---

Table 1: Quantitative results on the ABC-NEF [9] dataset under 2-view, 3-view, and 4-view conditions. Best and second-best scores are highlighted.

| Method | Precision $\uparrow$ |  |  | Recall $\uparrow$ |  |  | F-score $\uparrow$ |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | 2-view | 3-view | 4-view | 2-view | 3-view | 4-view | 2-view | 3-view | 4-view |
| EMAP [10] | 0.3562 | 0.5373 | 0.6113 | 0.3912 | 0.5449 | 0.5987 | 0.3729 | 0.5313 | 0.6049 |
| SGCR [13] | 0.3221 | 0.5353 | 0.6391 | 0.2362 | 0.3891 | 0.5128 | 0.2725 | 0.4506 | 0.5690 |
| + SeCuRe (Ours) | 0.5017 | 0.7459 | 0.8098 | 0.3154 | 0.4466 | 0.5133 | 0.3873 | 0.5587 | 0.6283 |
| $\Delta$ | $+55.76 \%$ | $+39.34 \%$ | $+26.71 \%$ | $+33.53 \%$ | $+14.78 \%$ | $+0.10 \%$ | $+42.13 \%$ | $+23.99 \%$ | $+10.42 \%$ |
| CurveGaussian [14] | 0.2413 | 0.2965 | 0.3191 | 0.3160 | 0.3951 | 0.4629 | 0.2623 | 0.3342 | 0.3778 |
| + SeCuRe (Ours) | 0.4217 | 0.5269 | 0.5686 | 0.5163 | 0.5749 | 0.5923 | 0.4642 | 0.5443 | 0.5802 |
| $\Delta$ | $+74.76 \%$ | $+77.71 \%$ | $+78.19 \%$ | $+63.39 \%$ | $+45.51 \%$ | $+27.95 \%$ | $+76.97 \%$ | $+62.87 \%$ | $+53.57 \%$ |
| EdgeGaussians [12] | 0.3416 | 0.4593 | 0.5634 | 0.0338 | 0.1005 | 0.1453 | 0.0585 | 0.1564 | 0.2236 |
| + SeCuRe (Ours) | 0.6933 | 0.7760 | 0.8427 | 0.2686 | 0.4498 | 0.5490 | 0.3761 | 0.5615 | 0.6581 |
| $\Delta$ | $+102.96 \%$ | $+68.95 \%$ | $+49.57 \%$ | $+694.67 \%$ | $+347.56 \%$ | $+277.84 \%$ | $+542.91 \%$ | $+259.02 \%$ | $+194.32 \%$ |

spatially coherent edges while suppressing scattered or isolated points that lack meaningful geometric contribution.

Visibility Consistency Regularization. Under sparse-view con- ${ }^{205}$ ditions, Gaussian primitives that effectively contribute to edge representation are expected to be observable from as many viewpoints as possible, which our visibility consistency regularization aims to enforce. For each current Gaussian $\theta_{i}$, we project its position $\mu_{i}$ onto the $j$-th training view using the camera projection function $\pi_{j}(\cdot)$, and then define its edge visibility score in the view as follows:

$$
v_{i, j}= \begin{cases}1, & \text { if } E_{j}\left(\pi_{j}\left(\mu_{i}\right)\right)=1 \\ 0, & \text { otherwise }\end{cases}
$$

After obtaining the average edge visibility score $\bar{v}_{i}$ for each Gaussian $\theta_{i}$, we prune those with $\bar{v}_{i}<\eta$, where $\eta$ is a predefined pruning threshold. This regularization encourages the re- ${ }^{338}$ tention of Gaussians consistently supported by multi-view edge evidence. It effectively suppresses overfitting to view-specific noise or artifacts.

### 3.5. Sparsity Loss

Conventional 3DGS-based methods for curve reconstruction ${ }^{342}$ typically employ photometric losses and regularization terms. ${ }^{343}$ These components optimize Gaussian primitives while con- ${ }^{344}$ straining their shapes and distributions. They supervise re- ${ }^{345}$ construction quality by minimizing the difference between rendered and ground-truth views. As an auxiliary framework, ${ }^{346}$ our SeCuRe retains these components for Gaussian optimiza- ${ }^{347}$ tion. We denote them collectively as $\mathcal{L}_{\text {3DGS }}$.

Nevertheless, these components are barely specialized for ${ }^{349}$ optimizing Gaussians toward edge representation that serves ${ }^{350}$ curve reconstruction. This phenomenon arises because edge ar-351 eas are sparsely distributed and occupy less image pixels. To ${ }^{352}$ enhance attention to edge areas during pixel-based optimiza-353 tion, we design an additional sparsity loss that encourages spa-354 tial compactness and accelerates convergence. Specifically, our ${ }^{355}$
sparsity loss is based on the Cauchy loss [39] due to its robustness to outliers, and penalizes redundant or scattered Gaussian primitives that correspond to non-edge pixels. It is defined as follows:

$$
\mathcal{L}_{\mathrm{sp}}=\sum_{i} \log \left(1+\frac{\alpha\left(p_{i}\right)}{\delta}\right)
$$

where $i$ indexes non-edge pixels from the input edge maps, $\alpha\left(p_{i}\right)$ denotes the opacity of the Gaussian primitive corresponding to pixel $p_{i}$, and $\delta$ is a scaling factor that controls the loss sensitivity. Hence, the training objective for Gaussian optimization using our framework is defined as follows:

$$
\mathcal{L}_{\text {total }}=\mathcal{L}_{\text {3DGS }}+\lambda_{\text {sp }} \mathcal{L}_{\text {sp }}
$$

where $\lambda_{\text {sp }}$ is a weighting factor to balance the two terms.

## 4. Experiments

We evaluated the proposed auxiliary framework via experiments using 2-view, 3-view, and 4-view images as input, which represent extremely sparse-view conditions. Our experiments were conducted against several state-of-the-art 3D curve reconstruction methods, including EMAP [10], EdgeGaussians [12], SGCR [13], and CurveGaussian [14]. Specifically, EMAP is an implicit neural representation method, whereas the others are explicit 3DGS-based methods and can be seamlessly integrated with our SeCuRe.

### 4.1. Datasets

We conducted the experiments on two representative datasets: ABC-NEF [9] and DTU [40]. ABC-NEF is a curated subset of the ABC dataset [11] and has been widely adopted by recent 3D curve reconstruction methods [10, 12]. It provides parametric ground-truth curves and 50 rendered views per model. DTU consists of multi-view images of real-world tabletop objects. Our experiments followed EMAP's protocol and used the same six scenes. Pseudo ground-truth edges were obtained by projecting dense 3D scans onto 2D edge maps.

---

![img-2.jpeg](assets/2025_JJK_CAG_SparseViewCurve%20(1)_img-2.jpeg)

Fig. 3: Qualitative comparisons on the ABC-NEF [9] dataset under 3-view condition.

This setup ensured a direct and fair comparison with established 3 DGS baselines on both synthetic CAD models and real-world scenes 383 under sparse-view conditions. Following previous studies on 384 sparse-view NVS [16, 41, 42], our experiments were conducted 385 under 2-view, 3-view, and 4-view conditions. This setup al- 386 lowed us to assess reconstruction performance in a challenging 387 manner with extremely limited inputs.

### 4.2. Metrics

In line with EMAP [10] and EdgeGaussians [12], we adopted 391 Precision, Recall, and F-score at a distance threshold $\tau$ as eval-392 uation metrics, while $\tau$ was set to 20 mm on ABC-NEF and 393 5 mm on DTU. Points were uniformly sampled along the pre-dicted parametric curves in proportion to their lengths and compared with ground-truth points sampled at the same resolution. Specifically, Precision measures the percentage of predicted ${ }^{395}$ points within $\tau$ of any ground-truth point. Recall measures the ${ }_{386}$ percentage of ground-truth points having at least one predicted ${ }_{387}$ point within $\tau$. F-score is the harmonic mean of Precision and ${ }_{398}$ Recall. These metrics provided a consistent and reliable basis to ${ }_{399}$ evaluate 3D curve reconstruction performance on both datasets. 400

### 4.3. Implementation Details

For fair comparison, our experiments adopted the same train- 403 test splits, hyperparameters, training configurations, and eval-uation protocols as the state-of-the-art curve reconstruction ${ }_{405}$ methods [10, 12, 13]. Following standard sparse-view $\mathrm{NVS}_{406}$ practices [16, 41, 42], we uniformly sampled training views ${ }_{407}$
from the full camera trajectory. In our integration with each 3DGS-based method, we employed the EGGI strategy to generate the point cloud for Gaussian initialization while incorporating the sparsity loss into the training objective (Eqn. 5) with $\lambda_{s p}=1.00$. The SAGP strategy was applied at the $175^{\text {th }}$ and $370^{\text {th }}$ epochs during the 400 -epoch optimization in EdgeGaussians [12], at the $2500^{\text {th }}$ and $7500^{\text {th }}$ epochs during the 10000 -epoch optimization in CurveGaussians [14], and at the $1500^{\text {th }}$ and $4500^{\text {th }}$ epochs during the 6000 -epoch optimization in SGCR [13]. Meanwhile, we set the neighborhood radius scaling factor $\gamma=2.00$, the visibility consistency pruning threshold $\eta=0.30$, and the sparsity loss scaling factor $\delta=0.50$ (Eqn. 4).

### 4.4. Evaluation on $A B C-N E F$

As demonstrated in Table 1, integrating our SeCuRe with existing 3DGS-based methods yielded considerable improvements on the ABC-NEF [9] dataset across all metrics under sparse-view conditions, enhancing Precision, Recall, and Fscore by up to $55.76 \%, 33.53 \%$, and $42.13 \%$ for SGCR [13], by up to $78.19 \%, 63.39 \%$, and $76.97 \%$ for CurveGaussian [14], and by up to $102.96 \%, 694.67 \%$, and $542.91 \%$ for EdgeGaussians [12]. Except for Recall under the 4-view condition, EdgeGaussian and CurveGaussian achieved state-of-theart performance with our SeCuRe, surpassing the implicit neural representation-based EMAP [10]. These results suggest that the proposed auxiliary framework enhances the sparse-view

---

Table 2: Quantitative results on the DTU [40] dataset under 2-view, 3-view, and 4-view conditions. Best and second-hest scores are highlighted.

| Method | Precision $\uparrow$ |  |  | Recall $\uparrow$ |  |  | F-score $\uparrow$ |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | 2 | 3 | 4 | 2 | 3 | 4 | 2 | 3 | 4 |
| EMAP [10] | 0.4563 | 0.4890 | 0.5217 | 0.1557 | 0.1921 | 0.2185 | 0.2322 | 0.2758 | 0.3080 |
| SGCR [13] | 0.3618 | 0.4130 | 0.4674 | 0.4118 | 0.4495 | 0.4612 | 0.3852 | 0.4305 | 0.4643 |
| + SeCuRe (Ours) | 0.4979 | 0.5265 | 0.5614 | 0.5084 | 0.5425 | 0.5472 | 0.5031 | 0.5344 | 0.5542 |
| $\Delta$ | $+37.62 \%$ | $+27.48 \%$ | $20.11 \%$ | $+23.46 \%$ | $+20.69 \%$ | $+18.65 \%$ | $+30.61 \%$ | $+24.13 \%$ | $+19.36 \%$ |
| CurveGaussian [14] | 0.0136 | 0.0276 | 0.0450 | 0.0301 | 0.0597 | 0.0766 | 0.0187 | 0.0377 | 0.0567 |
| + SeCuRe (Ours) | 0.2656 | 0.2746 | 0.3099 | 0.3701 | 0.4562 | 0.5171 | 0.3093 | 0.3428 | 0.3875 |
| $\Delta$ | $+1852.94 \%$ | $+894.93 \%$ | $+588.67 \%$ | $+1129.57 \%$ | $+664.15 \%$ | $+575.07 \%$ | $+1554.01 \%$ | $+809.28 \%$ | $+583.42 \%$ |
| EdgeGaussians [12] | 0.3958 | 0.4387 | 0.4648 | 0.5477 | 0.5806 | 0.6148 | 0.4595 | 0.4998 | 0.5294 |
| + SeCuRe (Ours) | 0.7677 | 0.7932 | 0.8138 | 0.6634 | 0.7113 | 0.7319 | 0.7117 | 0.7500 | 0.7707 |
| $\Delta$ | $+93.96 \%$ | $+80.81 \%$ | $+75.09 \%$ | $+21.12 \%$ | $+22.51 \%$ | $+19.05 \%$ | $+54.89 \%$ | $+50.06 \%$ | $+45.58 \%$ |

curve reconstruction capability of existing 3DGS-based meth-446 ods on the ABC-NEF dataset, which poses significant chal-447 lenges due to weak appearance cues and highly localized ge-448 ometric constraints.

As illustrated in Fig. 3, under sparse-view conditions, Edge-450 Gaussians suffered from severe initialization failures. Its strat-451 egy of random or surface-based initialization became much less effective without dense multi-view constraints, causing Gaus- ${ }_{452}$ sian primitives to converge to sub-optimal local minima and ${ }_{454}$ fail to reach correct positions. This resulted in visibly frag- ${ }_{455}$ mented or entirely missed edges in the reconstructed curves. CurveGaussian misinterpreted minor sparse-view noise or epi- ${ }_{457}$ polar inconsistencies as true geometric features with its dy- ${ }_{458}$ namic split-merge strategy, leading to obvious structural ar- ${ }_{459}$ tifacts The spherical Gaussian representation of SGCR was ${ }_{460}$ under-constrained when observing an edge from only a few ${ }_{461}$ nearly co-linear viewpoints, causing unstable and inaccurate fit- ${ }_{462}$ ting.

In contrast, applying our SeCuRe effectively mitigated these ${ }_{464}$ issues. Specifically, the EGGI strategy injected robust geo- ${ }_{465}$ metric priors and initialized Gaussian primitives near the true ${ }_{466}$ edges, eliminating fragmentation caused by unsatisfactory con- ${ }_{467}$ vergence. To prevent overfitting, the SAGP strategy and spar- ${ }_{468}$ sity loss guided the Gaussian optimization process by dynami- ${ }_{469}$ cally removing floating artifacts and spatially incoherent prim- ${ }_{470}$ itives caused by view-dependent noise, enforcing geometric ${ }_{471}$ consistency across views. Therefore, the proposed auxiliary ${ }_{472}$ framework managed to boost existing 3DGS-based methods for ${ }_{473}$ curve reconstruction by recovering structurally sound and ge- ${ }_{474}$ ometrically accurate curves, achieving a superior balance be- ${ }_{475}$ tween completeness and fidelity even with extremely sparse inputs.

### 4.5. Evaluation on DTU

Compared to the results on the ABC-NEF [9] dataset, our Se-480 CuRe also enabled the existing 3DGS-based methods to achieve ${ }_{481}$ substantial improvements on the DTU [40] dataset across all482 metrics under sparse-view conditions. In particular, as shown483 in Table 2, Precision, Recall, and F-score increased by up484
to $37.62 \%, 23.46 \%$, and $30.61 \%$ for SGCR [13], by up to $1852.94 \%, 1129.57 \%$, and $1554.01 \%$ for CurveGaussian [14], and by up to $93.96 \%, 22.51 \%$, and $54.89 \%$ for EdgeGaussians [12]. In addition, EdgeGaussians achieved state-of-the-art performance with our SeCuRe in all metrics under each condition.

On the DTU dataset, the challenge for curve reconstruction shifts to topological complexity. These scenes contain intricate and densely interconnected edges, making it particularly difficult to recover fine-grained geometric details, such as sharp corners and complex intersections, from extremely sparse multiview images. As demonstrated in Fig. 4, existing methods fail to reconstruct high-quality curves in different manners. Specifically, CurveGaussian struggled with topological density and its optimization process could hardly disentangle the numerous intersecting and overlapping edges, resulting in a group of incorrectly merged line segments diverging significantly from the ground truth. SGCR avoided topological collapse but produced overly simplified and incomplete reconstructions, as its implicit and spherical representations lacked the expressiveness to capture high-frequency details from sparse observations. Notably, EdgeGaussians exhibited a unique failure due to its initialization strategy on DTU: it started with a dense point cloud from a fully trained 3DGS model, containing both edge and surface points. While this ensured complete coverage, it introduced a strong bias from appearance-rich surfaces. The subsequent optimization overfitted to this noisy initialization and view-specific artifacts, failing to prune non-edge Gaussians and instead producing numerous spurious line segments, severely degrading both precision and recall.

The above method-specific failures were mitigated using the proposed auxiliary framework. For EdgeGaussians, the EGGI strategy replaced the surface-based initialization, eliminating the bias from appearance-rich surfaces and removing spurious line segments. For CurveGaussian, the SAGP strategy enabled the disentanglement of intersections by enforcing spatial coherence, recovering clean and distinct line segments. Similarly, for SGCR, those previously unsecured high-frequency details were successfully captured. Consequently, these methods were able

---

![img-3.jpeg](assets/2025_JJK_CAG_SparseViewCurve%20(1)_img-3.jpeg)

Fig. 4: Qualitative comparisons on the DTU [40] dataset under 3-view condition.
to address topological errors and false positives when integrated ${ }^{500}$ with the proposed auxiliary framework, highlighting its strong ${ }^{501}$ generalization capability.

### 4.6. Ablation Studies

Using EdgeGaussians [12] as the baseline, we conducted ablation studies on the DTU [40] dataset under the 3-view condition to validate the contributions of our SeCuRe's key compo-504 nents. The results are summarized in Table 3. Compared to the505 baseline, the Sparsity Loss $\mathcal{L}_{s p}$ improved the reconstruction of ${ }^{506}$ fine-grained linear structures for edges by penalizing redundant ${ }^{507}$ Gaussian primitives. The EGGI strategy further enhanced geo-508 metric consistency and reduced floating artifacts by incorporat-509 ing robust edge priors. The SAGP strategy effectively removed ${ }^{510}$ spatial outliers and enforced multi-view visibility, leading to ${ }^{511}$ cleaner and more compact reconstructions. Each key compo-512
nent contributed improvements across different metrics individually. As a combination of all these components, the proposed auxiliary framework corresponded to the highest scores, demonstrating their complementary advantages.

Fig. 5 presents example outputs from different configurations in our ablation study. The baseline produced incomplete and noisy curves with many floating artifacts, which could be reduced by adding either $\mathcal{L}_{s p}$ or the SAGP strategy, leading to improved curve continuity. Meanwhile, the EGGI strategy provided high-quality Gaussian initialization, enabling more accurate edge localization. Adopting our SeCuRe achieved the cleanest and most accurate curve reconstructions, with floating artifacts nearly eliminated.

---

![img-4.jpeg](assets/2025_JJK_CAG_SparseViewCurve%20(1)_img-4.jpeg)

Fig. 5: Qualitative ablations on key components of our SeCuRe on the DTU [40] dataset under 3-view condition using EdgeGaussians [12] as the baseline. Integrating the full version of our SeCuRe yields the cleanest and most accurate curve reconstruction result.

Table 3: Ablations on key components of our SeCuRe on the DTU [40] dataset ${ }_{541}$ under 3-view condition using EdgeGaussians [12] as the baseline. Best scores ${ }_{542}$ are displayed in bold.

| Method | Recall | Precision | F-score |
| :-- | :-- | :--: | :--: |
| Baseline | 0.4305 | 0.5524 | 0.4839 |
| $+\mathcal{L}_{s p}$ | 0.4842 | 0.5623 | 0.5203 |
| + SAGP | 0.6363 | 0.6435 | 0.6399 |
| + EGGI | 0.5809 | 0.7034 | 0.6363 |
| + SAGP $+\mathcal{L}_{s p}$ | 0.6494 | 0.6647 | 0.6570 |
| + EGGI $+\mathcal{L}_{s p}$ | 0.6022 | 0.7107 | 0.6520 |
| + SAGP + EGGI | 0.7021 | 0.7634 | 0.7315 |
| + SeCuRe (Ours) | $\mathbf{0 . 7 1 1 3}$ | $\mathbf{0 . 7 9 3 2}$ | $\mathbf{0 . 7 5 0 0}$ |

## 5. Conclusions

In this paper, we proposed SeCuRe, a novel auxiliary frame-558 work for sparse-view 3D curve reconstruction with 3DGS. To ${ }^{559}$ address the critical challenges posed by sparse input views, ${ }_{560}$ our SeCuRe incorporates three key components: an Edge-562 Guided Gaussian Initialization (EGGI) strategy, a Structure-563 Aware Gaussian Pruning (SAGP) strategy, and a sparsity loss. ${ }^{564}$ These components work synergistically to enhance geometric ${ }_{566}$ fidelity, suppress artifacts, and enforce both spatial coherence ${ }_{567}$ and visibility consistency during the optimization of Gaussian ${ }^{568}$ primitives. Extensive evaluations on the ABC-NEF and DTU ${ }^{569}$ datasets demonstrated that our SeCuRe could significantly im- ${ }_{571}$ prove the performance of existing 3DGS-based methods for ${ }_{572}$ curve reconstruction under extremely sparse-view conditions. It ${ }^{573}$ consistently enhanced reconstruction accuracy, continuity, and ${ }_{575}^{574}$ completeness, outperforming state-of-the-art methods, includ- ${ }_{576}$ ing EMAP. Qualitative and quantitative results verified the ro-577 bustness and generalization capability of the proposed auxiliary ${ }^{578}$ framework across different curve reconstruction methods.

There are several directions for future work. First, the ${ }_{581}$ current pipeline of our SeCuRe focuses on existing 3DGS-582 based methods. Extending support to curve reconstruction us- ${ }^{583}$ ing other splatting-based NVS algorithms, such as Deformable ${ }_{585}^{584}$ Beta Splatting (DBS) [43] and 3D Student Splatting and Scoop-586 ing (SSS) [44], remains unsolved. Second, while our Se-587 CuRe leverages external priors for initialization and regular- ${ }^{588}$ ization, it has not yet exploited inline priors [45] that provide ${ }_{590}$ more visual supervision without additional pre-trained models. ${ }^{591}$

Addressing these limitations could improve the adaptability of our SeCuRe.

By enabling high-quality 3D curve reconstruction under spare-view conditions, our SeCuRe reduces the burden of data acquisition and facilitates broader tasks in computer vision and graphics, such as SLAM, CAD modeling, and scene understanding. Our work bridges the gap between sparse-view inputs and high-fidelity curve construction, offering a practical and effective solution for real-world scenarios where dense multiview data is difficult to obtain.

## References

[1] S. Liu, Y. Yu, R. Pautrat, M. Pollefeys, V. Larsson, 3d line mapping revisited, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2023, p. 21445-21455. doi:10.1109/ cvpr52729.2023.02054.
URL http://dx.doi.org/10.1109/CVPR52729.2023.02054
[2] R. Pautrat, I. Suárez, Y. Yu, M. Pollefeys, V. Larsson, Gluestick: Robust image matching by sticking points and lines together, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE, 2023. doi : 10.1109/iccv51070.2023.00890.

URL http://dx.doi.org/10.1109/ICCV51070.2023.00890
[3] X. Wang, L. Wang, H. Wu, G. Xiao, K. Xu, Parametric primitive analysis of cad sketches with vision transformer, IEEE Transactions on Industrial Informatics 20 (10) (2024) 12041-12050. doi:10.1109/tii.2024. 3413358.

URL http://dx.doi.org/10.1109/TII.2024.3413358
[4] L. Liu, C. Bajaj, J. O. Deasy, D. A. Low, T. Ju, Surface reconstruction from non-parallel curve networks, Computer Graphics Forum 27 (2) (2008) 155-163. doi:10.1111/j.1467-8659.2008.01112.x. URL http://dx.doi.org/10.1111/j.1467-8659.2008.01112.x
[5] W. Celes, F. Abraham, Texture-based wireframe rendering, in: 2010 23rd SIBGRAPI Conference on Graphics, Patterns and Images, IEEE, 2010, p. 149-155. doi:10.1109/sibgrapi.2010.28.
URL http://dx.doi.org/10.1109/SIBGRAPI.2010.28
[6] X. Wei, J. Huang, X. Ma, Real-time monocular visual slam by combining points and lines, in: 2019 IEEE International Conference on Multimedia and Expo (ICME), IEEE, 2019, p. 103-108. doi:10.1109/icme. 2019. 00026.

URL http://dx.doi.org/10.1109/ICME.2019.00026
[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, Nerf: representing scenes as neural radiance fields for view synthesis, Communications of the ACM 65 (1) (2021) 99-106. doi: 10.1145/3503250.

URL http://dx.doi.org/10.1145/3503250
[8] B. Kerbl, G. Kopanas, T. Leimkuehler, G. Drettakis, 3d gaussian splatting for real-time radiance field rendering, ACM Transactions on Graphics 42 (4) (2023) 1-14. doi:10.1145/3592433.
URL http://dx.doi.org/10.1145/3592433
[9] Y. Ye, R. Yi, Z. Gao, C. Zhu, Z. Cai, K. Xu, Nef: Neural edge fields for 3d parametric curve reconstruction from multi-view images, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition

---

(CVPR), IEEE, 2023, p. 8486-8495. doi:10.1109/cvpr52729.2023.664 00820 .
URL http://dx.doi.org/10.1109/CVPR52729.2023.00820666
[10] L. Li, S. Peng, Z. Yu, S. Liu, R. Pautraz, X. Yin, M. Pollefeys, 3d neuralset edge reconstruction, in: 2024 IEEE/CVF Conference on Computer Vision 668 and Pattern Recognition (CVPR), IEEE, 2024, p. 21219-21229. doi:10.1109/cvpr52733.2024.02005.
URL http://dx.doi.org/10.1109/CVPR52733.2024.02005
[11] S. Koch, A. Matveev, Z. Jiang, F. Williams, A. Artemov, E. Burnaev,et al. M. Alexa, D. Zorin, D. Panozzo, Abc: A big cad model dataset forest geometric deep learning, in: 2019 IEEE/CVF Conference on Computered Vision and Pattern Recognition (CVPR), IEEE, 2019, p. 9593-9603.e75 doi:10.1109/cvpr. 2019.00983.
URL http://dx.doi.org/10.1109/CVPR. 2019.00983
[12] K. Chelani, A. Benbihi, T. Sattler, F. Kahl, Edgegaussians - 3d edge map-e78 ping via gaussian splatting, in: 2025 IEEE/CVF Winter Conference oned Applications of Computer Vision (WACV), IEEE, 2025, p. 3268-3279.680 doi:10.1109/wacv61041.2025.00323.
URL http://dx.doi.org/10.1109/WACV61041.2025.00323
[13] X. Yang, D. Ji, Y. Li, J. Guo, Y. Guo, J. Xie, Sgcr: Spherical gaussians883 for efficient 3d curve reconstruction, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, p. 88 5793-5803. doi:10.1109/cvpr52734.2025.00544. URL http://dx.doi.org/10.1109/CVPR52734.2025.00544
[14] Z. Gao, R. Yi, Y. Dai, X. Zhu, W. Chen, C. Zhu, K. Xu, Curve-aware88 gaussian splatting for 3d parametric curve reconstruction (2025). doi:10.10.48550/ARXIV.2506.21401.
URL https://arxiv.org/abs/2506.21401
[15] Y. Wan, M. Shao, Y. Cheng, W. Zuo, S2gaussian: Sparse-view super-892 resolution 3d gaussian splatting, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, p. 89 711-721. doi:10.1109/cvpr52734.2025.00075. URL http://dx.doi.org/10.1109/CVPR52734.2025.00075
[16] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, L. Gu, Dngaus-897 sian: Optimizing sparse-view 3d gaussian radiance fields with global-898 local depth normalization, in: 2024 IEEE/CVF Conference on Computer899 Vision and Pattern Recognition (CVPR), IEEE, 2024, p. 20775-20785.700 doi:10.1109/cvpr52733.2024.01963.
URL http://dx.doi.org/10.1109/CVPR52733.2024.01963
[17] H. Xiong, S. Muttukuru, H. Xiao, R. Upadhyay, P. Chari, Y. Zhao, ${ }^{703}$ A. Kadambi, Sparsegs: Sparse view synthesis using 3d gaussian splat-704 ting, in: 2025 International Conference on 3D Vision (3DV), IEEE, 2025, p. 1032-1041. doi:10.1109/3dv66043.2025.00100. URL http://dx.doi.org/10.1109/3DV66043.2025.00100
[18] J. Jung, J. Han, H. An, J. Kang, S. Park, S. Kim, Relaxing accurate ini-708 tialization constraint for 3d gaussian splatting (2024). doi:10.48550/709 ARXIV. 2403.09413.
URL https://arxiv.org/abs/2403.09413
[19] H. Yu, X. Long, P. Tan, Lm-gaussian: Boost sparse-view 3d gaussiant12 splatting with large model priors (2024). doi:10.48550/ARXIV. 2409.713 03456.
URL https://arxiv.org/abs/2409.03456
[20] H. Park, G. Ryu, W. Kim, Dropgaussian: Structural regularization ${ }^{116}$ for sparse-view gaussian splatting, in: 2025 IEEE/CVF Conference717 on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, p. 718 21600-21609. doi:10.1109/cvpr52734.2025.02012.
URL http://dx.doi.org/10.1109/CVPR52734.2025.02012
[21] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, J. Revaud, Dust3r: Geomet-721 ric 3d vision made easy, in: 2024 IEEE/CVF Conference on Computer722 Vision and Pattern Recognition (CVPR), IEEE, 2024, p. 20697-20709.723 doi:10.1109/cvpr52733.2024.01956.
URL http://dx.doi.org/10.1109/CVPR52733.2024.01956
[22] Z. Su, W. Liu, Z. Yu, D. Hu, Q. Liao, Q. Tian, M. Pietikainen, L. Liu, ${ }^{726}$ Pixel difference networks for efficient edge detection, in: 2021 IEEE/CVF727 International Conference on Computer Vision (ICCV), IEEE, 2021, p. 728 5097-5107. doi:10.1109/iccv48922.2021.00507.
URL http://dx.doi.org/10.1109/ICCV48922.2021.00507
[23] X. Zhu, D. Du, W. Chen, Z. Zhao, Y. Nie, X. Han, Nerve: Neural721 volumetric edges for parametric curve extraction from point cloud, in:732 2023 IEEE/CVF Conference on Computer Vision and Pattern Recogni-733 tion (CVPR), IEEE, 2023, p. 13601-13610. doi:10.1109/cvpr52729.734 2023.01307.

URL http://dx.doi.org/10.1109/CVPR52729.2023.01307
[24] Y. Liu, S. D’Aronco, K. Schindler, J. D. Wegner, Pc2wf: 3d wireframe reconstruction from raw point clouds (2021). doi:10.48550/ARXIV. 2103.02766.
URL https://arxiv.org/abs/2103.02766
[25] M. Chandraker, J. Lim, D. Kriegman, Moving in stereo: Efficient structure and motion using lines, in: 2009 IEEE 12th International Conference on Computer Vision, IEEE, 2009, p. 1741-1748. doi:10.1109/iccv. 2009.5459390.
URL http://dx.doi.org/10.1109/ICCV.2009.5459390
[26] G. Schindler, P. Krishnamurthy, F. Dellaert, Line-based structure from motion for urban environments, in: Third International Symposium on 3D Data Processing, Visualization, and Transmission (3DPVT’06), IEEE, 2006, p. 846-853. doi:10.1109/3dpvt.2006.90.
URL http://dx.doi.org/10.1109/3DPVT.2006.90
[27] J. L. Schonberger, J.-M. Frahm, Structure-from-motion revisited, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2016. doi:10.1109/cvpr. 2016.445.
URL http://dx.doi.org/10.1109/CVPR. 2016.445
[28] Y. Yao, Z. Luo, S. Li, T. Fang, L. Quan, MVSNet: Depth Inference for Unstructured Multi-view Stereo, Springer International Publishing, 2018, p. 785-801. doi:10.1007/978-3-030-01237-3_47. URL http://dx.doi.org/10.1007/978-3-030-01237-3_47
[29] D. Tran, L. Bourdev, R. Fergus, L. Torresani, M. Paluri, Learning spatiotemporal features with 3d convolutional networks, in: 2015 IEEE International Conference on Computer Vision (ICCV), IEEE, 2015, p. 4489-4497. doi:10.1109/iccv.2015.510.
URL http://dx.doi.org/10.1109/ICCV.2015.510
[30] Y. Yao, Z. Luo, S. Li, T. Shen, T. Fang, L. Quan, Recurrent mvsnet for high-resolution multi-view stereo depth inference, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2019, p. 5520-5529. doi:10.1109/cvpr. 2019.00567.
URL http://dx.doi.org/10.1109/CVPR.2019.00567
[31] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, I. Polosukhin, Attention is all you need, in: I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, R. Garnett (Eds.), Advances in Neural Information Processing Systems, Vol. 30, Curran Associates, Inc., 2017.
URL https://proceedings.neurips.cc/paper_files/paper/ 2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
[32] V. Leroy, Y. Cabon, J. Revaud, Grounding Image Matching in 3D with MASt3R, Springer Nature Switzerland, 2024, p. 71-91. doi: 10.1007/978-3-031-73220-1_5.
URL http://dx.doi.org/10.1007/978-3-031-73220-1_5
[33] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny, Vggt: Visual geometry grounded transformer, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, p. 5294-5306. doi:10.1109/cvpr52734.2025.00499.
URL http://dx.doi.org/10.1109/CVPR52734.2025.00499
[34] R. Murai, E. Dexheimer, A. J. Davison, Mast3r-slam: Real-time dense slam with 3d reconstruction priors, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, p. 16695-16705. doi:10.1109/cvpr52734.2025.01556.
URL http://dx.doi.org/10.1109/CVPR52734.2025.01556
[35] A. Authors, Segmast3r: Geometry grounded segment matching, in: Advances in Neural Information Processing Systems (NeurIPS), 2025.
[36] S. Dong, S. Wang, S. Liu, L. Cai, Q. Fan, J. Kannala, Y. Yang, Reloc3r: Large-scale training of relative camera pose regression for generalizable, fast, and accurate visual localization, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2025, p. 16739-16752. doi:10.1109/cvpr52734.2025.01560.
URL http://dx.doi.org/10.1109/CVPR52734.2025.01560
[37] J. L. Schönberger, E. Zheng, J.-M. Frahm, M. Pollefeys, Pixelwise View Selection for Unstructured Multi-View Stereo, Springer International Publishing, 2016, p. 501-518. doi:10.1007/978-3-319-46487-9_ 31.

URL http://dx.doi.org/10.1007/978-3-319-46487-9_31
[38] D. Deng, Dbscan clustering algorithm based on density, in: 2020 7th International Forum on Electrical Engineering and Automation (IFEEA), IEEE, 2020, p. 949-953. doi:10.1109/ifeea51475.2020.00199.
URL http://dx.doi.org/10.1109/IFEEA51475.2020.00199
[39] J. T. Barron, A general and adaptive robust loss function, in: 2019

---

[40] R. Jensen, A. Dahl, G. Vogiatzis, E. Tola, H. Aanaes, Large scale multiview stereopsis evaluation, in: 2014 IEEE Conference on Computer Vision and Pattern Recognition, IEEE, 2014. doi:10.1109/cvpr. 2014. 59 .
URL http://dx.doi.org/10.1109/CVPR. 2014.59
[41] J. Yang, M. Pavone, Y. Wang, Freenerf: Improving few-shot neural rendering with free frequency regularization, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2023, p. 8254-8263. doi:10.1109/cvpr52729.2023.00798. URL http://dx.doi.org/10.1109/CVPR52729.2023.00798
[42] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. M. Sajjadi, A. Geiger, N. Radwan, Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2022. doi:10.1109/ cvpr52688.2022.00540.
URL http://dx.doi.org/10.1109/CVPR52688.2022.00540
[43] R. Liu, D. Sun, M. Chen, Y. Wang, A. Feng, Deformable beta splatting, in: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, SIGGRAPH Conference Papers '25, Association for Computing Machinery, New York, NY, USA, 2025. doi:10.1145/3721238.3730716.
[44] J. Zhu, J. Yue, F. He, H. Wang, 3d student splatting and scooping, in: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025, pp. 21045-21054.
[45] Q. Wang, Y. Zhao, J. Ma, J. Li, How to use diffusion priors under sparse views?, in: Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS '24, Curran Associates Inc., Red Hook, NY, USA, 2024. doi:10.5555/3737916.3738873.