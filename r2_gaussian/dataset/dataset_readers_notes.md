### `r2_gaussian/dataset/dataset_readers.py` 笔记

**文件核心功能**:
该文件是 R²-Gaussian 项目的**数据加载与预处理模块**，负责从不同格式的医学影像数据集中读取信息，并转换为项目内部统一的数据结构。

---

**1. 核心数据结构**:

*   **`mode_id`**: 扫描模式（`"parallel"`, `"cone"`）到整数 ID 的映射。
*   **`CameraInfo` (NamedTuple)**:
    *   **作用**: 存储单个相机（X射线投影）的所有相关信息。
    *   **关键属性**: `uid`, `R` (旋转), `T` (平移), `angle` (投影角度), `FovY`/`FovX` (视场角), `image` (2D投影数据), `width`/`height`, `mode`, `scanner_cfg` (扫描仪配置)。
*   **`SceneInfo` (NamedTuple)**:
    *   **作用**: 封装整个场景的信息。
    *   **关键属性**: `train_cameras`, `test_cameras` (CameraInfo 列表), `vol` (3D体数据真实值), `scanner_cfg`, `scene_scale` (场景缩放因子)。

---

**2. 主要数据读取函数**:

*   **`readBlenderInfo(path, eval)`**:
    *   **功能**: 读取 **Blender 格式**的 CT 数据（通常通过 `meta_data.json` 和 `.npy` 文件）。
    *   **流程**: 解析 JSON -> 应用场景缩放 -> 调用 `readCTameras` -> 加载 3D 体数据 -> 返回 `SceneInfo`。
*   **`readNAFInfo(path, eval)`**:
    *   **功能**: 读取 **NAF 格式**的数据（通常是 `.pickle` 文件）。
    *   **流程**: 加载 pickle -> 单位转换与场景缩放 -> 生成 `CameraInfo` 列表 -> 加载 3D 体数据 -> 返回 `SceneInfo`。
*   **`readCTameras(meta_data, source_path, eval, scene_scale)`**:
    *   **功能**: 辅助函数，从元数据中解析每个相机的详细信息。
    *   **流程**: 遍历投影角度 -> 使用 `angle2pose` 计算相机姿态 (`R`, `T`) -> 加载 2D 投影图像 -> 计算视场角 -> 封装为 `CameraInfo`。
*   **`angle2pose(DSO, angle)`**:
    *   **功能**: 根据扫描仪的**源到原点距离 (DSO)** 和**投影角度 (angle)**，计算相机姿态（世界坐标到相机坐标的变换矩阵 `c2w`）。
    *   **原理**: 通过一系列固定旋转和角度相关的旋转模拟 CT 扫描仪运动。

---

**3. 动态加载机制**:

*   **`sceneLoadTypeCallbacks`**:
    *   **作用**: 一个字典，将数据集类型字符串（如 `"Blender"`, `"NAF"`）映射到相应的读取函数。
    *   **优势**: 允许项目根据配置动态选择数据加载器，支持多种数据集格式。

---

**总结**:
`dataset_readers.py` 是项目的数据入口，确保了不同医学影像数据集能够被统一、标准化地加载和预处理，为后续的三维高斯溅射模型提供了高质量、一致性的输入。
