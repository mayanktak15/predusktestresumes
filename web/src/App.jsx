import { useMemo, useState } from "react";

const DEFAULT_SETTINGS = {
  model: "yolov8n.pt",
  confidence: 0.3,
  frameSkip: 1,
  maxAge: 30,
  resizeWidth: 0,
  device: "auto",
  outputFps: 0,
  clearSamples: true,
};

const REPORT_ITEMS = [
  {
    title: "Pipeline",
    detail:
      "Video -> Frames -> YOLOv8 -> DeepSORT -> Annotated output + analytics.",
  },
  {
    title: "Identity Consistency",
    detail:
      "Kalman prediction + appearance embeddings reduce ID switches.",
  },
  {
    title: "Challenges",
    detail: "Occlusion, motion blur, camera motion, and similar-looking subjects.",
  },
  {
    title: "Outputs",
    detail: "Annotated video, trajectory trails, heatmap, object count plot.",
  },
];

const CHECKLIST = [
  "Public video link recorded in README + report.",
  "Annotated output video saved to outputs/output.mp4.",
  "Sample frames in outputs/sample_frames/.",
  "Demo video script in demo_script.md.",
];

function formatSize(bytes) {
  if (!bytes) {
    return "0 MB";
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function getObjectUrl(file) {
  if (!file) {
    return "";
  }
  return URL.createObjectURL(file);
}

export default function App() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [inputVideo, setInputVideo] = useState(null);
  const [outputVideo, setOutputVideo] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [countPlot, setCountPlot] = useState(null);
  const [sampleFrames, setSampleFrames] = useState([]);
  const [notice, setNotice] = useState("");

  const inputUrl = useMemo(() => getObjectUrl(inputVideo), [inputVideo]);
  const outputUrl = useMemo(() => getObjectUrl(outputVideo), [outputVideo]);
  const heatmapUrl = useMemo(() => getObjectUrl(heatmap), [heatmap]);
  const countUrl = useMemo(() => getObjectUrl(countPlot), [countPlot]);
  const sampleUrls = useMemo(
    () => sampleFrames.map((file) => getObjectUrl(file)),
    [sampleFrames]
  );

  const previewUrl = outputUrl || inputUrl;
  const previewLabel = outputVideo ? "Output video" : "Input video";

  function updateSetting(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function handleInputChange(event) {
    const file = event.target.files?.[0] || null;
    setInputVideo(file);
  }

  function handleOutputVideo(event) {
    const file = event.target.files?.[0] || null;
    setOutputVideo(file);
  }

  function handleHeatmap(event) {
    const file = event.target.files?.[0] || null;
    setHeatmap(file);
  }

  function handleCountPlot(event) {
    const file = event.target.files?.[0] || null;
    setCountPlot(file);
  }

  function handleSampleFrames(event) {
    const files = Array.from(event.target.files || []);
    setSampleFrames(files.slice(0, 9));
  }

  function clearAll() {
    setInputVideo(null);
    setOutputVideo(null);
    setHeatmap(null);
    setCountPlot(null);
    setSampleFrames([]);
    setNotice("");
  }

  function handleRunClick() {
    setNotice(
      "This is a static app. Run tracking locally, then load outputs here."
    );
  }

  return (
    <div className="page">
      <header className="hero">
        <div className="hero__text">
          <p className="kicker">Tracking Dashboard</p>
          <h1>Multi-Object Detection and Persistent ID Tracking</h1>
          <p className="lede">
            Static React rebuild of the Streamlit UI. Upload local files to
            preview results without running a server.
          </p>
          <div className="hero__actions">
            <button className="primary" onClick={handleRunClick}>
              Run Tracking
            </button>
            <button className="ghost" onClick={clearAll}>
              Clear
            </button>
          </div>
          <p className="hint">
            Tip: Run the pipeline locally and select outputs/output.mp4, heatmap
            images, and sample frames here.
          </p>
          {notice && <div className="notice">{notice}</div>}
        </div>
        <div className="hero__glass">
          <div className="stat">
            <span className="stat__label">Detector</span>
            <span className="stat__value">YOLOv8</span>
          </div>
          <div className="stat">
            <span className="stat__label">Tracker</span>
            <span className="stat__value">DeepSORT</span>
          </div>
          <div className="stat">
            <span className="stat__label">Advanced</span>
            <span className="stat__value">Heatmap + Count</span>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel panel--controls">
          <div className="panel__head">
            <h2>Pipeline Settings</h2>
            <p className="meta">Matches the Streamlit controls.</p>
          </div>

          <div className="control-grid">
            <label>
              YOLOv8 model
              <select
                value={settings.model}
                onChange={(event) => updateSetting("model", event.target.value)}
              >
                <option value="yolov8n.pt">yolov8n.pt</option>
                <option value="yolov8s.pt">yolov8s.pt</option>
                <option value="yolov8m.pt">yolov8m.pt</option>
                <option value="yolov8l.pt">yolov8l.pt</option>
                <option value="yolov8x.pt">yolov8x.pt</option>
              </select>
            </label>
            <label>
              Detection confidence: {settings.confidence.toFixed(2)}
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={settings.confidence}
                onChange={(event) =>
                  updateSetting("confidence", Number(event.target.value))
                }
              />
            </label>
            <label>
              Frame skip: {settings.frameSkip}
              <input
                type="range"
                min="1"
                max="6"
                step="1"
                value={settings.frameSkip}
                onChange={(event) =>
                  updateSetting("frameSkip", Number(event.target.value))
                }
              />
            </label>
            <label>
              Max age (frames): {settings.maxAge}
              <input
                type="range"
                min="5"
                max="90"
                step="5"
                value={settings.maxAge}
                onChange={(event) =>
                  updateSetting("maxAge", Number(event.target.value))
                }
              />
            </label>
            <label>
              Resize width (0 = keep)
              <input
                type="number"
                min="0"
                max="1920"
                step="32"
                value={settings.resizeWidth}
                onChange={(event) =>
                  updateSetting("resizeWidth", Number(event.target.value))
                }
              />
            </label>
            <label>
              Device
              <select
                value={settings.device}
                onChange={(event) => updateSetting("device", event.target.value)}
              >
                <option value="auto">auto</option>
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
            <label>
              Output FPS (0 = auto)
              <input
                type="number"
                min="0"
                max="120"
                step="1"
                value={settings.outputFps}
                onChange={(event) =>
                  updateSetting("outputFps", Number(event.target.value))
                }
              />
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.clearSamples}
                onChange={(event) =>
                  updateSetting("clearSamples", event.target.checked)
                }
              />
              Clear sample frames on refresh
            </label>
          </div>

          <div className="panel__head">
            <h3>Input Video</h3>
            <p className="meta">Preview a local input video.</p>
          </div>
          <label className="file">
            <input type="file" accept="video/*" onChange={handleInputChange} />
            <span>Select input video</span>
          </label>
          <p className="meta">
            {inputVideo
              ? `${inputVideo.name} · ${formatSize(inputVideo.size)}`
              : "No input video selected."}
          </p>

          <div className="panel__head">
            <h3>Output Assets</h3>
            <p className="meta">
              Load files created by the Python pipeline.
            </p>
          </div>
          <label className="file">
            <input type="file" accept="video/*" onChange={handleOutputVideo} />
            <span>Select output video</span>
          </label>
          <label className="file">
            <input type="file" accept="image/*" onChange={handleHeatmap} />
            <span>Select heatmap</span>
          </label>
          <label className="file">
            <input type="file" accept="image/*" onChange={handleCountPlot} />
            <span>Select count plot</span>
          </label>
          <label className="file">
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleSampleFrames}
            />
            <span>Select sample frames</span>
          </label>
        </section>

        <section className="panel panel--results">
          <div className="panel__head">
            <h2>Results</h2>
            <p className="meta">{previewLabel} preview and analytics.</p>
          </div>
          <div className="video-wrap">
            {previewUrl ? (
              <video src={previewUrl} controls playsInline />
            ) : (
              <div className="empty">No video loaded.</div>
            )}
          </div>
          <p className="meta">
            {outputVideo
              ? `${outputVideo.name} · ${formatSize(outputVideo.size)}`
              : "Load outputs/output.mp4 to preview the annotated result."}
          </p>

          <div className="two-col">
            <div className="card">
              <h3>Heatmap</h3>
              {heatmapUrl ? (
                <img src={heatmapUrl} alt="Heatmap" />
              ) : (
                <p className="meta">No heatmap selected.</p>
              )}
            </div>
            <div className="card">
              <h3>Object Count</h3>
              {countUrl ? (
                <img src={countUrl} alt="Object count" />
              ) : (
                <p className="meta">No count plot selected.</p>
              )}
            </div>
          </div>

          <div className="panel__head">
            <h3>Sample Frames</h3>
            <p className="meta">Up to 9 frames from outputs/sample_frames/.</p>
          </div>
          <div className="sample-grid">
            {sampleUrls.length ? (
              sampleUrls.map((url, index) => (
                <img src={url} alt={`Sample frame ${index + 1}`} key={url} />
              ))
            ) : (
              <p className="meta">No sample frames selected.</p>
            )}
          </div>
        </section>

        <section className="panel panel--report">
          <div className="panel__head">
            <h2>Report Summary</h2>
            <p className="meta">Snapshot of the technical report.</p>
          </div>
          <div className="report">
            {REPORT_ITEMS.map((item) => (
              <div className="report__item" key={item.title}>
                <h3>{item.title}</h3>
                <p>{item.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="panel panel--notes">
          <div className="panel__head">
            <h2>Notes</h2>
            <p className="meta">Quick checklist for the assignment.</p>
          </div>
          <ul className="checklist">
            {CHECKLIST.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      </main>

      <footer className="footer">
        <span>Static viewer for Vercel deployment.</span>
      </footer>
    </div>
  );
}
