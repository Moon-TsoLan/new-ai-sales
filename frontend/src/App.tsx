import { useEffect, useMemo, useState } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";

import api from "./api";
import "./style.css";

type HistoryItem = {
  id: number;
  product_name: string;
  predicted_sales: number;
  created_at: string;
};

type ModelMetrics = { sales: number; repeat_rate: number; average_rating: number };
type PlotGroup = { summary?: string; bar?: string; waterfall?: string; force_html?: string };
type VisualizationMap = Record<string, Record<string, PlotGroup>>;

type Detail = {
  id: number;
  product_name: string;
  product_desc: string;
  image_url: string;
  shap_plot_url: string;
  features_snapshot: Record<string, any>;
  model_predictions?: Record<string, ModelMetrics>;
  visualizations?: VisualizationMap;
  predicted_sales: number;
  repeat_rate: number;
  average_rating: number;
  created_at: string;
};

const TARGET_LABELS: Record<string, string> = {
  sales: "销量",
  repeat_rate: "复购率",
  average_rating: "平均评分",
};

const FEATURE_LABELS: Record<string, string> = {
  category: "类目",
  sub_category: "子类目",
  brand: "品牌",
  fabric: "材质",
  color: "颜色",
  main_color: "主色",
  style: "风格",
};

function LoginPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const submit = async () => {
    setError("");
    if (!username.trim() || !password.trim()) {
      setError("用户名和密码不能为空。");
      return;
    }

    try {
      const res = await api.post("/auth/login", { username: username.trim(), password });
      localStorage.setItem("access_token", res.data.data.access_token);
      localStorage.setItem("refresh_token", res.data.data.refresh_token);
      navigate("/");
    } catch {
      setError("登录失败，请检查用户名和密码。");
    }
  };

  return (
    <div className="auth-wrap">
      <div className="auth-card">
        <h2>登录 AI-SALE</h2>
        <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="用户名" />
        <input value={password} onChange={(e) => setPassword(e.target.value)} placeholder="密码" type="password" />
        <button className="btn-submit" onClick={submit}>
          登录
        </button>
        <button className="btn-ghost" onClick={() => navigate("/register")}>
          去注册
        </button>
        {error && <p className="err">{error}</p>}
      </div>
    </div>
  );
}

function RegisterPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);

  const submit = async () => {
    setMessage("");
    setIsError(false);
    if (!username.trim()) {
      setIsError(true);
      setMessage("请填写用户名。");
      return;
    }
    if (!email.trim()) {
      setIsError(true);
      setMessage("请填写邮箱。");
      return;
    }
    if (!password.trim()) {
      setIsError(true);
      setMessage("请填写密码。");
      return;
    }
    try {
      await api.post("/auth/register", { username: username.trim(), password, email: email.trim() });
      setMessage("注册成功，请登录。");
      setTimeout(() => navigate("/login"), 600);
    } catch (error: any) {
      setIsError(true);
      const detail = error?.response?.data?.detail;
      if (typeof detail === "string") {
        if (detail.includes("Username")) setMessage("用户名已存在。");
        else if (detail.includes("Email")) setMessage("邮箱已存在或不合法。");
        else setMessage(detail);
      } else {
        setMessage("注册失败，请检查输入项。");
      }
    }
  };

  return (
    <div className="auth-wrap">
      <div className="auth-card">
        <h2>注册 AI-SALE</h2>
        <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="用户名（必填）" />
        <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="邮箱（必填）" />
        <input value={password} onChange={(e) => setPassword(e.target.value)} placeholder="密码（必填）" type="password" />
        <button className="btn-submit" onClick={submit}>
          注册
        </button>
        <button className="btn-ghost" onClick={() => navigate("/login")}>
          去登录
        </button>
        {message && <p className={isError ? "err" : "ok"}>{message}</p>}
      </div>
    </div>
  );
}

function DashboardPage() {
  const navigate = useNavigate();
  const [productName, setProductName] = useState("");
  const [productDesc, setProductDesc] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sessionSubmitted, setSessionSubmitted] = useState(false);
  const [formError, setFormError] = useState("");
  const [selectedModel, setSelectedModel] = useState<"regression" | "tree">("regression");
  const [currentUserId, setCurrentUserId] = useState("-");
  const [zoomImageUrl, setZoomImageUrl] = useState("");

  const authMissing = useMemo(() => !localStorage.getItem("access_token"), []);

  const loadHistory = async () => {
    const res = await api.get("/history?page=1&page_size=20");
    setHistory(res.data.data.items ?? []);
  };

  const chooseModelByData = (data: Detail) => {
    if (data.model_predictions?.regression) setSelectedModel("regression");
    else if (data.model_predictions?.tree) setSelectedModel("tree");
  };

  const loadDetail = async (id: number) => {
    const res = await api.get(`/history/${id}`);
    const data = res.data.data as Detail;
    setDetail(data);
    setSessionSubmitted(true);
    setImage(null);
    setProductName(data.product_name || "");
    setProductDesc(data.product_desc || "");
    setPreviewUrl(`http://127.0.0.1:8000${data.image_url}`);
    setFormError("");
    chooseModelByData(data);
  };

  useEffect(() => {
    if (!authMissing) void loadHistory();
  }, [authMissing]);

  useEffect(() => {
    if (!image) return;
    const url = URL.createObjectURL(image);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) {
      setCurrentUserId("-");
      return;
    }
    try {
      const payload = token.split(".")[1];
      const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
      const decoded = JSON.parse(atob(normalized));
      setCurrentUserId(decoded?.sub ?? "-");
    } catch {
      setCurrentUserId("-");
    }
  }, []);

  if (authMissing) return <Navigate to="/login" replace />;

  const submit = async () => {
    setFormError("");
    if (sessionSubmitted) {
      setFormError("当前问卷已提交，请先点击“开启新问卷”。");
      return;
    }
    if (!productName.trim() || !productDesc.trim() || !image) {
      setFormError("请完整填写产品名称、产品描述并上传图片。");
      return;
    }

    setLoading(true);
    const form = new FormData();
    form.append("product_name", productName.trim());
    form.append("product_desc", productDesc.trim());
    form.append("image_file", image);
    try {
      const res = await api.post("/predict", form, { headers: { "Content-Type": "multipart/form-data" } });
      const data = res.data.data as Detail;
      setDetail(data);
      setSessionSubmitted(true);
      chooseModelByData(data);
      await loadHistory();
    } finally {
      setLoading(false);
    }
  };

  const startNewSession = () => {
    setProductName("");
    setProductDesc("");
    setImage(null);
    setPreviewUrl("");
    setDetail(null);
    setSessionSubmitted(false);
    setFormError("");
    setSelectedModel("regression");
  };

  const logout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    navigate("/login");
  };

  const currentPrediction = detail?.model_predictions?.[selectedModel];
  const currentVisualizations = detail?.visualizations?.[selectedModel] ?? {};
  const extractedFeatures = (detail?.features_snapshot?.extracted_features ?? {}) as Record<string, string>;
  const featureRows = Object.entries(FEATURE_LABELS)
    .map(([key, label]) => ({
      key,
      label,
      value: (extractedFeatures[key] ?? "unknown").toString(),
    }))
    .filter((x) => x.value.trim() !== "");
  const hasRegression = !!detail?.model_predictions?.regression;
  const hasTree = !!detail?.model_predictions?.tree;
  const hasNewVisualizations = Object.keys(currentVisualizations).length > 0;
  const toImageSrc = (p?: string) => (p ? `http://127.0.0.1:8000${p}` : "");
  const openZoom = (src: string) => setZoomImageUrl(src);
  const closeZoom = () => setZoomImageUrl("");

  return (
    <div className="app-container">
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-top">
          <div className="logo-area">
            <div className="logo">
              <div className="logo-icon">AI</div>
              <span className="logo-name">AI-SALE</span>
            </div>
            <button
              className="collapse-btn"
              title={sidebarCollapsed ? "展开" : "收回"}
              onClick={() => setSidebarCollapsed((v) => !v)}
            >
              {sidebarCollapsed ? ">" : "<"}
            </button>
          </div>

          <button className="new-chat-btn" title="新增收集表" onClick={startNewSession}>
            <span className="new-chat-text">+</span>
            <span className="new-chat-label">开启新对话</span>
          </button>
        </div>

        <div className="history-section">
          <div className="history-header">最近记录</div>
          <div className="history-list">
            {history.map((h) => (
              <div key={h.id} className={`history-item ${detail?.id === h.id ? "active" : ""}`} onClick={() => loadDetail(h.id)}>
                <span>{h.product_name}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="user-info">
            <span>{`用户ID: ${currentUserId}`}</span>
            <button className="logout-btn" title="退出登录" onClick={logout}>
              X
            </button>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <div className="content-wrapper">
          <section className="form-card">
            <h2 className="form-title">产品销售预测问卷</h2>
            <div className="form-group">
              <label>产品名称（英文）</label>
              <input
                value={productName}
                onChange={(e) => setProductName(e.target.value)}
                placeholder="e.g. Classic Cotton Summer Dress"
                disabled={sessionSubmitted || loading}
              />
            </div>
            <div className="form-group">
              <label>产品描述（英文）</label>
              <textarea
                rows={3}
                value={productDesc}
                onChange={(e) => setProductDesc(e.target.value)}
                placeholder="Describe product features, target users and key points."
                disabled={sessionSubmitted || loading}
              />
            </div>
            <div className="form-group">
              <label>产品图片</label>
              <label className="upload-area">
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  disabled={sessionSubmitted || loading}
                  onChange={(e) => setImage(e.target.files?.[0] ?? null)}
                />
                点击上传图片（JPG/PNG）
              </label>
              {previewUrl && <img src={previewUrl} className="preview-img" alt="预览图" />}
            </div>
            <button className="btn-submit" onClick={submit} disabled={loading || sessionSubmitted}>
              {loading ? "分析中..." : sessionSubmitted ? "已提交，点击开启新问卷" : "生成预测报告"}
            </button>
            {formError && <p className="err">{formError}</p>}
          </section>

          {detail && (
            <section id="resultArea">
              <div className="result-section">
                <table className="result-table">
                  <thead>
                    <tr>
                      <th>预测指标</th>
                      <th>数值</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>预估销量</td>
                      <td className="metric-value">{(currentPrediction?.sales ?? detail.predicted_sales).toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td>预估复购率</td>
                      <td className="metric-value">{(currentPrediction?.repeat_rate ?? detail.repeat_rate).toFixed(4)}</td>
                    </tr>
                    <tr>
                      <td>预估评分</td>
                      <td className="metric-value">{(currentPrediction?.average_rating ?? detail.average_rating).toFixed(2)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="result-section">
                <table className="result-table">
                  <thead>
                    <tr>
                      <th>提取特征</th>
                      <th>特征值</th>
                    </tr>
                  </thead>
                  <tbody>
                    {featureRows.map((row) => (
                      <tr key={row.key}>
                        <td>{row.label}</td>
                        <td>{row.value || "unknown"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="result-section">
                <div className="toggle-group">
                  <button
                    className={`toggle-btn ${selectedModel === "regression" ? "active" : ""}`}
                    disabled={!hasRegression}
                    onClick={() => setSelectedModel("regression")}
                  >
                    回归结果
                  </button>
                  <button
                    className={`toggle-btn ${selectedModel === "tree" ? "active" : ""}`}
                    disabled={!hasTree}
                    onClick={() => setSelectedModel("tree")}
                  >
                    树模型结果
                  </button>
                </div>

                <div className="shap-container">
                  {!hasNewVisualizations && (
                    <p className="empty-placeholder">
                      这是旧版记录，仅有单张图。请提交新问卷查看全套回归/树 SHAP 图。
                    </p>
                  )}

                  {Object.entries(currentVisualizations).map(([target, group]) => (
                    <div className="plot-group" key={target}>
                      <h3>{TARGET_LABELS[target] ?? target}</h3>
                      <div className="plot-grid">
                        {group.summary && (
                          <div className="plot-card">
                            <p className="plot-title">Summary</p>
                            <img
                              src={toImageSrc(group.summary)}
                              alt={`${target}-summary`}
                              className="plot-img zoomable"
                              onClick={() => openZoom(toImageSrc(group.summary))}
                            />
                          </div>
                        )}
                        {group.bar && (
                          <div className="plot-card">
                            <p className="plot-title">Bar</p>
                            <img
                              src={toImageSrc(group.bar)}
                              alt={`${target}-bar`}
                              className="plot-img zoomable"
                              onClick={() => openZoom(toImageSrc(group.bar))}
                            />
                          </div>
                        )}
                        {group.waterfall && (
                          <div className="plot-card">
                            <p className="plot-title">Waterfall</p>
                            <img
                              src={toImageSrc(group.waterfall)}
                              alt={`${target}-waterfall`}
                              className="plot-img zoomable"
                              onClick={() => openZoom(toImageSrc(group.waterfall))}
                            />
                          </div>
                        )}
                      </div>
                      {group.force_html && (
                        <a href={toImageSrc(group.force_html)} target="_blank" rel="noreferrer">
                          打开 force 图（{TARGET_LABELS[target] ?? target}）
                        </a>
                      )}
                    </div>
                  ))}

                  {!hasNewVisualizations && (
                    <img
                      src={`http://127.0.0.1:8000${detail.shap_plot_url}`}
                      alt="旧版SHAP图"
                      className="plot-img zoomable"
                      onClick={() => openZoom(`http://127.0.0.1:8000${detail.shap_plot_url}`)}
                    />
                  )}
                </div>
              </div>
            </section>
          )}

          <footer>AI-SALE · 基于多模态回归预测 + SHAP 可解释性</footer>
        </div>
      </main>
      {zoomImageUrl && (
        <div className="image-modal" onClick={closeZoom}>
          <button
            className="image-modal-close"
            onClick={(e) => {
              e.stopPropagation();
              closeZoom();
            }}
          >
            ×
          </button>
          <img className="image-modal-content" src={zoomImageUrl} alt="放大预览" onClick={(e) => e.stopPropagation()} />
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<DashboardPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
    </Routes>
  );
}

